#!/usr/bin/env python3
"""One-learner/many-rollout-worker runtime for V20."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import imageio
import numpy as np
import torch

from chunk_replay import (
    ACTION_DIM,
    CHUNK_SIZE,
    Z_DIM,
    ChunkReplay,
    ImageChunkReplay,
    ReplaySource,
)
from rlt_models import MolmoAct2RLTCF
from train_rlt import (
    ae_cfgrl_actor_step,
    build_rlt_optimizers,
    cfgrl_actor_step,
    critic_health_metrics,
    flow_critic_td_step,
)
from train_rlt_online import (
    _build_eval_policy,
    _run_isolated_evaluation,
)
from v20_harness import (
    BarrierSnapshot,
    RoundState,
    StaleOperationError,
    allocate_global_uid,
    append_jsonl_locked,
    atomic_copy,
    atomic_write_json,
    barrier_snapshot,
    claim_worker_operation,
    clean_operation_artifacts,
    collection_seed,
    decide_promotion,
    decision_to_dict,
    derive_seed,
    ensure_next_uid,
    initialize_run,
    operation_rows,
    pair_seed,
    publish_operation,
    publish_stop,
    read_jsonl,
    read_round_state,
    register_journal_episode,
    validate_run_dir,
    wait_for_barrier,
    write_worker_done,
    write_worker_heartbeat,
)
from v20_training import (
    ChallengerTrainingConfig,
    _diagnose_candidate,
    atomic_model_save,
    checkpoint_sha256,
    prepare_cfgrl_model,
    target_positive_fraction,
)


log = logging.getLogger("v20_runner")
_STOP_REQUESTED = False
_MODEL_CACHE: dict[tuple[str, int], MolmoAct2RLTCF] = {}


def _request_stop(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _path_in_run(run_dir: Path, path: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(run_dir):
        raise RuntimeError(f"path escapes V20 run directory: {resolved}")
    return resolved


def _atomic_replay_save(replay: ChunkReplay, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    temporary.unlink(missing_ok=True)
    try:
        replay.save_npz(str(temporary))
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_seed_replay(replay: ChunkReplay) -> None:
    if not replay.rows:
        raise RuntimeError("V20 seed replay is empty")
    by_uid: dict[int, tuple[int, bool]] = {}
    for row in replay.rows:
        uid = int(row.trajectory_uid)
        pose_idx = int(row.pose_idx)
        if uid < 0 or pose_idx < 0:
            raise RuntimeError(
                "V20 seed replay requires explicit trajectory_uid and pose_idx"
            )
        value = (pose_idx, float(row.success) > 0.5)
        previous = by_uid.get(uid)
        if previous is not None and previous != value:
            raise RuntimeError(
                f"trajectory {uid} has inconsistent provenance/outcome"
            )
        by_uid[uid] = value


def _mock_seed_replay(path: Path, *, target_pose_idx: int) -> None:
    replay = ChunkReplay(
        max_transitions=10_000,
        pos_frac=0.5,
        benchmark_pose_cycle=24,
        seed=7,
    )
    for episode_id in range(16):
        success = episode_id < 2
        reference = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
        reference[:, 0] = float(episode_id) / 100.0
        replay.add_episode_chunks(
            [np.full((Z_DIM,), episode_id, dtype=np.float32)],
            [np.zeros((ACTION_DIM,), dtype=np.float32)],
            [reference],
            [reference.copy()],
            [np.full((CHUNK_SIZE,), float(success), dtype=np.float32)],
            [np.ones((CHUNK_SIZE,), dtype=np.float32)],
            success=success,
            gamma=0.99,
            episode_id=episode_id,
            trajectory_uid=episode_id,
            pose_idx=int(target_pose_idx),
            source_policy=ReplaySource.OFFLINE_REFERENCE,
            worker_id=0,
            round_id=-1,
            policy_version=0,
        )
    _atomic_replay_save(replay, path)


def _initialize_learner(args: argparse.Namespace) -> RoundState:
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    marker_exists = (run_dir / ".v20_monotone_incumbent").is_file()
    incumbent = run_dir / "checkpoints" / "incumbent_v000.pt"
    pooled = run_dir / "replay" / "pooled.npz"
    if not marker_exists:
        if not args.base_checkpoint or not args.seed_replay:
            raise ValueError(
                "fresh V20 run requires --base_checkpoint and --seed_replay"
            )
        base_checkpoint = Path(args.base_checkpoint).resolve()
        seed_replay = Path(args.seed_replay).resolve()
        if not base_checkpoint.is_file():
            raise FileNotFoundError(base_checkpoint)
        if not seed_replay.is_file():
            raise FileNotFoundError(seed_replay)
        seed = ChunkReplay.load_npz(
            str(seed_replay),
            max_transitions=int(args.replay_capacity),
        )
        _validate_seed_replay(seed)
        pooled.parent.mkdir(parents=True, exist_ok=True)
        _atomic_replay_save(seed, pooled)
        incumbent.parent.mkdir(parents=True, exist_ok=True)
        if args.mock:
            atomic_copy(base_checkpoint, incumbent)
        else:
            model = prepare_cfgrl_model(
                base_checkpoint,
                device=args.device,
                hidden=int(args.hidden),
                n_hidden_actor=int(args.n_hidden_actor),
                n_hidden_critic=int(args.n_hidden_critic),
                z_expand_dim=int(args.z_expand_dim),
                o_dim=int(args.cfgrl_o_dim),
            )
            model.cfgrl_w = float(args.w_deploy)
            atomic_model_save(
                model,
                incumbent,
                meta={
                    "v20": True,
                    "incumbent_version": 0,
                    "deployment_mode": "reference",
                },
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    state = initialize_run(
        run_dir,
        worker_count=int(args.worker_count),
        incumbent_checkpoint=incumbent,
        incumbent_mode="reference",
        rollout_seed_root=int(args.rollout_seed_root),
        config={
            "target_pose_idx": int(args.target_pose_idx),
            "benchmark_episode_idx": int(args.benchmark_episode_idx),
            "collect_waves_per_round": int(args.collect_waves_per_round),
            "rounds_per_offline": int(args.rounds_per_offline),
            "hidden": int(args.hidden),
            "n_hidden_actor": int(args.n_hidden_actor),
            "n_hidden_critic": int(args.n_hidden_critic),
            "z_expand_dim": int(args.z_expand_dim),
            "cfgrl_o_dim": int(args.cfgrl_o_dim),
            "batch_size": int(args.batch_size),
            "actor_lr": float(args.actor_lr),
            "critic_lr": float(args.critic_lr),
            "cfgrl_dropout": float(args.cfgrl_dropout),
            "ref_dropout": float(args.ref_dropout),
            "w_deploy": float(args.w_deploy),
            "updates_per_wave": int(args.updates_per_wave),
            "max_update_sec_per_wave": float(args.max_update_sec_per_wave),
            "promotion_alpha": float(args.promotion_alpha),
            "promotion_min_gain": float(args.promotion_min_gain),
            "clone_mse_max": float(args.clone_mse_max),
            "cond_ref_mse_max": float(args.cond_ref_mse_max),
            "max_normalized_action": float(args.max_normalized_action),
            "ae_accumulate": bool(args.ae_accumulate),
            "ae_steps": int(args.ae_steps),
            "ae_batch_size": int(args.ae_batch_size),
            "ae_lr": float(args.ae_lr),
            "replay_capacity": int(args.replay_capacity),
            "mock": bool(args.mock),
        },
    )
    if not pooled.is_file():
        raise FileNotFoundError(f"pooled replay missing on resume: {pooled}")
    pooled_replay = ChunkReplay.load_npz(
        str(pooled),
        max_transitions=int(args.replay_capacity),
    )
    ensure_next_uid(
        run_dir,
        max(int(row.trajectory_uid) for row in pooled_replay.rows) + 1,
    )
    return state


def _sample_online_batch(
    replay: ChunkReplay,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return replay.sample(
        int(args.batch_size),
        device,
        require_both_outcomes=True,
        target_pose_idx=int(args.target_pose_idx),
        target_positive_fraction=float(args.replay_pos_frac),
        trajectory_first=True,
        temporal_bins=int(args.temporal_bins),
    )


def _run_online_update(
    args: argparse.Namespace,
    model: MolmoAct2RLTCF,
    optimizers: dict[str, torch.optim.Optimizer],
    replay: ChunkReplay,
    device: torch.device,
) -> dict[str, float]:
    """Central-learner updates for one wave, under a real step/wall-clock budget.

    Runs up to ``updates_per_wave`` critic+actor step pairs (each on a fresh
    stratified batch), stopping early when ``max_update_sec_per_wave`` is
    exceeded. Advantage labels activate only while the critic is healthy.
    """
    budget_steps = int(args.updates_per_wave)
    budget_sec = float(args.max_update_sec_per_wave)
    start = time.monotonic()
    critic_rows: list[dict[str, float]] = []
    actor_rows: list[dict[str, float]] = []
    healthy = False
    updates = 0
    for _ in range(budget_steps):
        if time.monotonic() - start >= budget_sec:
            break
        batch = _sample_online_batch(replay, args, device)
        critic_metrics = flow_critic_td_step(
            model,
            optimizers["critic"],
            batch,
            gamma=float(args.gamma),
        )
        healthy = bool(critic_health_metrics(model, batch)["healthy"])
        actor_metrics = cfgrl_actor_step(
            model,
            optimizers["actor"],
            batch,
            cond_dropout=float(args.cfgrl_dropout),
            ref_dropout=float(args.ref_dropout),
            use_advantage_labels=healthy,
        )
        critic_rows.append(critic_metrics)
        actor_rows.append(actor_metrics)
        updates += 1

    def _mean(rows: list[dict[str, float]], key: str) -> float:
        values = [float(row[key]) for row in rows if key in row]
        return float(np.mean(values)) if values else 0.0

    return {
        "critic_td": _mean(critic_rows, "q_td_loss"),
        "q_mean": _mean(critic_rows, "q_mean"),
        "q_target": _mean(critic_rows, "q_target"),
        "q_rank_gap": _mean(critic_rows, "q_rank_gap"),
        "critic_healthy": float(healthy),
        "actor_loss": _mean(actor_rows, "actor_loss"),
        "cfgrl_pos_frac": _mean(actor_rows, "cfgrl_pos_frac"),
        "updates": float(updates),
        "update_sec": float(time.monotonic() - start),
    }


def _run_offline_ae_update(
    args: argparse.Namespace,
    model: MolmoAct2RLTCF,
    device: torch.device,
    run_dir: Path,
    round_id: int,
) -> dict[str, Any]:
    """Offline phase: fine-tune the Molmo action expert via CFGRL flow matching.

    Uses all collected image trajectories. The VLM stays frozen; only the AE
    (LoRA adapters) are updated. Returns a report dict.
    """
    from molmo_ae_backend import MolmoAEBackend

    image_replay_path = run_dir / "replay" / "image_pooled.npz"
    if not image_replay_path.is_file():
        log.warning("No image replay for offline AE update; skipping")
        return {"skipped": True, "reason": "no_image_replay"}
    image_replay = ImageChunkReplay.load_npz(str(image_replay_path))
    if not image_replay.has_both_outcomes():
        log.warning("Image replay lacks both outcomes; skipping AE update")
        return {"skipped": True, "reason": "lacks_both_outcomes"}

    ae_backend = MolmoAEBackend(
        device=device,
        dtype=torch.bfloat16,
        enable_lora=True,
        rlt=model,
        feature_mode="rl_token",
    )
    latest = run_dir / "checkpoints" / "ae_trainable_latest.pt"
    if bool(args.ae_accumulate) and latest.is_file():
        # Iterated improvement: continue from the previous offline phase's
        # LoRA instead of re-training from the base AE.
        ae_backend.load_trainable(latest)
        log.info("Offline AE: accumulated LoRA from %s", latest)
    ae_opt = torch.optim.Adam(
        ae_backend.trainable_parameters(), lr=float(args.ae_lr)
    )
    model.eval()

    steps = int(args.ae_steps)
    batch_size = int(args.ae_batch_size)
    mean_loss = 0.0
    skipped = 0
    for step in range(steps):
        batch = image_replay.sample(
            batch_size, device, require_both_outcomes=True
        )
        metrics = ae_cfgrl_actor_step(
            model,
            ae_backend,
            ae_opt,
            batch,
        )
        mean_loss += float(metrics.get("actor_loss", 0.0))
        skipped += int(metrics.get("ae_cfg_skipped", 0.0) > 0.5)
        if (step + 1) % max(1, steps // 10) == 0:
            log.info(
                "Offline AE step %d/%d loss=%.4f",
                step + 1,
                steps,
                mean_loss / (step + 1),
            )
    mean_loss /= max(steps, 1)

    ae_ckpt = run_dir / "checkpoints" / f"ae_trainable_r{round_id:03d}.pt"
    ae_backend.save_trainable(
        ae_ckpt,
        meta={"round_id": round_id, "ae_steps": steps, "mean_loss": mean_loss},
    )
    latest = run_dir / "checkpoints" / "ae_trainable_latest.pt"
    atomic_copy(ae_ckpt, latest)
    del ae_backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "skipped": False,
        "ae_checkpoint": str(ae_ckpt.resolve()),
        "ae_latest": str(latest.resolve()),
        "ae_steps": steps,
        "ae_steps_skipped_no_positive": skipped,
        "mean_actor_loss": mean_loss,
        "image_rows": len(image_replay),
    }


def _load_imported_uids(run_dir: Path) -> set[int]:
    path = run_dir / "replay" / "imported_uids.json"
    if not path.is_file():
        return set()
    value = json.loads(path.read_text(encoding="utf-8"))
    return {int(uid) for uid in value.get("trajectory_uids", [])}


def _import_collect_snapshot(
    run_dir: Path,
    snapshot: BarrierSnapshot,
    *,
    replay_capacity: int,
) -> dict[str, int]:
    pooled_path = run_dir / "replay" / "pooled.npz"
    pooled = ChunkReplay.load_npz(
        str(pooled_path),
        max_transitions=int(replay_capacity),
    )
    existing = {int(row.trajectory_uid) for row in pooled.rows}
    imported = _load_imported_uids(run_dir)
    added_rows = 0
    added_episodes = 0
    added_successes = 0
    for result in operation_rows(snapshot):
        uid = int(result["trajectory_uid"])
        replay_path = _path_in_run(run_dir, Path(result["replay_path"]))
        if uid in existing:
            imported.add(uid)
            continue
        episode = ChunkReplay.load_npz(
            str(replay_path),
            max_transitions=max(1_000, int(replay_capacity)),
        )
        episode_uids = {int(row.trajectory_uid) for row in episode.rows}
        if episode_uids != {uid}:
            raise RuntimeError(
                f"journal {replay_path} contains UIDs {episode_uids}, expected {uid}"
            )
        pooled.extend(episode.rows)
        existing.add(uid)
        imported.add(uid)
        added_rows += len(episode.rows)
        added_episodes += 1
        added_successes += int(bool(result["success"]))
    pooled.n_episodes = len(existing)
    _atomic_replay_save(pooled, pooled_path)
    atomic_write_json(
        run_dir / "replay" / "imported_uids.json",
        {
            "schema_version": 1,
            "trajectory_uids": sorted(imported),
            "updated_at": time.time(),
        },
    )
    _merge_image_snapshot(run_dir, snapshot, imported_before=set())
    return {
        "rows": added_rows,
        "episodes": added_episodes,
        "successes": added_successes,
    }


def _merge_image_snapshot(
    run_dir: Path,
    snapshot: BarrierSnapshot,
    *,
    imported_before: set[int],
) -> None:
    """Merge per-worker image episodes into the pooled image replay."""
    pooled_path = run_dir / "replay" / "image_pooled.npz"
    if pooled_path.is_file():
        pooled = ImageChunkReplay.load_npz(str(pooled_path))
        existing_uids = {int(row.episode_id) for row in pooled.rows}
    else:
        pooled = ImageChunkReplay(
            max_transitions=100_000, benchmark_pose_cycle=24, seed=0
        )
        existing_uids = set()
    added = False
    for result in operation_rows(snapshot):
        image_path = result.get("image_replay_path") or ""
        if not image_path:
            continue
        uid = int(result["trajectory_uid"])
        if uid in existing_uids:
            continue
        episode = ImageChunkReplay.load_npz(
            str(_path_in_run(run_dir, Path(image_path)))
        )
        pooled.rows.extend(episode.rows)
        pooled.n_episodes += episode.n_episodes
        existing_uids.add(uid)
        added = True
    if added:
        pooled_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = pooled_path.with_name(f".{pooled_path.name}.{os.getpid()}.tmp.npz")
        pooled.save_npz(str(tmp))
        os.replace(tmp, pooled_path)


def _run_online_round(
    args: argparse.Namespace,
    state: RoundState,
    round_id: int,
    model: MolmoAct2RLTCF | None,
    optimizers: dict[str, torch.optim.Optimizer],
    device: torch.device,
) -> dict[str, Any]:
    """One online round: collect waves, apply actor+critic updates each wave."""
    run_dir = Path(args.run_dir).resolve()
    waves = int(args.collect_waves_per_round)
    update_metrics: list[dict[str, float]] = []
    for wave in range(waves):
        if _STOP_REQUESTED:
            raise KeyboardInterrupt
        operation_id = f"r{round_id:03d}_collect_w{wave:03d}"
        snapshot = _dispatch_and_wait(
            run_dir,
            phase="collect",
            round_id=round_id,
            operation_id=operation_id,
            timeout_sec=float(args.barrier_timeout_sec),
            collect_wave=wave,
        )
        imported = _import_collect_snapshot(
            run_dir,
            snapshot,
            replay_capacity=int(args.replay_capacity),
        )
        log.info(
            "Round %d wave %d: imported episodes=%d successes=%d rows=%d",
            round_id,
            wave,
            imported["episodes"],
            imported["successes"],
            imported["rows"],
        )
        replay = ChunkReplay.load_npz(
            str(run_dir / "replay" / "pooled.npz"),
            max_transitions=int(args.replay_capacity),
            pos_frac=float(args.replay_pos_frac),
        )
        if model is None:
            continue
        if not replay.has_both_outcomes():
            log.warning(
                "Round %d wave %d: replay lacks both outcomes; skipping update",
                round_id,
                wave,
            )
            continue
        metrics = _run_online_update(args, model, optimizers, replay, device)
        update_metrics.append(metrics)
        log.info(
            "Round %d wave %d update: critic_td=%.4f healthy=%d actor_loss=%.4f",
            round_id,
            wave,
            metrics["critic_td"],
            int(metrics["critic_healthy"]),
            metrics["actor_loss"],
        )
    # Publish the candidate actor with the deployment guidance weight baked in,
    # then evaluate it against the incumbent in a paired probe.
    candidate_path = (
        run_dir / "checkpoints" / "candidates" / f"candidate_r{round_id:03d}.pt"
    )
    if model is not None:
        model.cfgrl_w = float(args.w_deploy)
        atomic_model_save(
            model,
            candidate_path,
            meta={
                "round_id": round_id,
                "deployment_mode": "actor",
                "w_deploy": float(args.w_deploy),
            },
        )
    else:
        atomic_copy(Path(state.incumbent_checkpoint), candidate_path)
    atomic_copy(candidate_path, run_dir / "checkpoints" / "actor_latest.pt")
    probe = _run_probe(args, state, round_id, candidate_path)
    promotion = _maybe_promote(args, state, round_id, model, candidate_path, probe)
    val_video = _run_val_video(
        args,
        round_id,
        tag=f"round_{round_id:03d}_candidate",
        checkpoint=candidate_path,
        mode="actor",
    )
    mean = {
        key: float(np.mean([m[key] for m in update_metrics]))
        for key in (
            "critic_td",
            "q_mean",
            "q_target",
            "q_rank_gap",
            "critic_healthy",
            "actor_loss",
            "cfgrl_pos_frac",
            "updates",
            "update_sec",
        )
        if update_metrics
    }
    report = {
        "round_id": round_id,
        "waves": waves,
        "updates": len(update_metrics),
        "probe": {key: value for key, value in probe.items() if key != "rows"},
        "promotion": promotion,
        "val_video": val_video,
        **mean,
    }
    atomic_write_json(
        run_dir / "reports" / f"round_{round_id:03d}_final.json", report
    )
    append_jsonl_locked(run_dir / "reports" / "probe_sr.jsonl", probe)
    append_jsonl_locked(
        run_dir / "reports" / "promotions.jsonl",
        {"round_id": round_id, **promotion},
    )
    return report


def _maybe_promote(
    args: argparse.Namespace,
    state: RoundState,
    round_id: int,
    model: MolmoAct2RLTCF | None,
    candidate_path: Path,
    probe: dict[str, Any],
) -> dict[str, Any]:
    """Gate the candidate against the incumbent and advance the incumbent.

    Uses the paired probe rows (McNemar + gain lower bound) plus candidate
    diagnostics (clone MSE / action bounds). On promotion the candidate
    becomes the collection policy for subsequent rounds, which is what makes
    the online loop actually online.
    """
    run_dir = Path(args.run_dir).resolve()
    rows = probe.get("rows") or []
    if not rows:
        return {"promote": False, "reason": "no_paired_rows"}
    diagnostics: dict[str, Any] = {}
    if model is not None and not args.mock:
        try:
            replay = ChunkReplay.load_npz(
                str(run_dir / "replay" / "pooled.npz"),
                max_transitions=int(args.replay_capacity),
                pos_frac=float(args.replay_pos_frac),
            )
            config = ChallengerTrainingConfig(
                batch_size=int(args.batch_size),
                cond_dropout=float(args.cfgrl_dropout),
                target_pose_idx=int(args.target_pose_idx),
                temporal_bins=int(args.temporal_bins),
                clone_mse_max=float(args.clone_mse_max),
                cond_ref_mse_max=float(args.cond_ref_mse_max),
                max_normalized_action=float(args.max_normalized_action),
                diagnostic_batches=int(args.diagnostic_batches),
            )
            target_successes = replay.target_successful_episode_count(
                int(args.target_pose_idx)
            )
            diagnostics = _diagnose_candidate(
                model,
                replay,
                config=config,
                target_fraction=target_positive_fraction(target_successes),
                device=args.device,
                seed=derive_seed(int(args.rollout_seed_root), "diagnose", round_id),
            )
            clone_ok = bool(diagnostics["clone_ok"])
            cond_ok = bool(diagnostics["cond_ok"])
            action_bounds_ok = bool(diagnostics["action_bounds_ok"])
        except Exception as error:  # noqa: BLE001
            log.warning(
                "candidate diagnostics failed; blocking promotion: %r", error
            )
            clone_ok = False
            cond_ok = False
            action_bounds_ok = False
    else:
        # Mock runs exercise the promotion machinery without model diagnostics.
        clone_ok = True
        cond_ok = True
        action_bounds_ok = True
    # Bonferroni correction: one paired promotion test per round, at most
    # max_rounds tests per run, so each test runs at alpha / max_rounds to
    # control the family-wise false-promotion rate.
    alpha_nominal = float(args.promotion_alpha)
    alpha_effective = alpha_nominal / float(max(int(args.max_rounds), 1))
    decision = decide_promotion(
        rows,
        alpha=alpha_effective,
        min_gain=float(args.promotion_min_gain),
        clone_ok=clone_ok,
        cond_ok=cond_ok,
        action_bounds_ok=action_bounds_ok,
    )
    result = decision_to_dict(decision)
    result["diagnostics"] = diagnostics
    result["alpha_nominal"] = alpha_nominal
    result["alpha_effective"] = alpha_effective
    if decision.promote:
        new_version = int(state.incumbent_version) + 1
        incumbent_path = (
            run_dir / "checkpoints" / f"incumbent_v{new_version:03d}.pt"
        )
        atomic_copy(candidate_path, incumbent_path)
        publish_operation(
            run_dir,
            phase="idle",
            round_id=round_id,
            operation_id=f"r{round_id:03d}_promote_v{new_version:03d}",
            incumbent_version=new_version,
            incumbent_checkpoint=incumbent_path,
            incumbent_mode="actor",
            message=(
                f"promoted candidate from round {round_id} "
                f"(paired gain {decision.point_gain:.3f}, "
                f"p={decision.p_value:.4f})"
            ),
        )
        result["incumbent_version"] = new_version
        result["incumbent_checkpoint"] = str(incumbent_path.resolve())
        log.info(
            "Round %d: PROMOTED candidate to incumbent v%d (gain=%.3f p=%.4f)",
            round_id,
            new_version,
            decision.point_gain,
            decision.p_value,
        )
    else:
        log.info(
            "Round %d: candidate not promoted (%s)",
            round_id,
            decision.reason,
        )
    return result


def _run_probe(
    args: argparse.Namespace,
    state: RoundState,
    round_id: int,
    candidate_path: Path,
) -> dict[str, Any]:
    """Paired probe: each worker runs candidate AND incumbent on its pose.

    Both rollouts of a pair share the same seed (``pair_seed``), so outcomes
    are paired on initial conditions and McNemar applies. The candidate is
    evaluated at the deployment guidance weight baked into its checkpoint.
    """
    run_dir = Path(args.run_dir).resolve()
    operation_id = f"r{round_id:03d}_probe"
    snapshot = _dispatch_and_wait(
        run_dir,
        phase="probe",
        round_id=round_id,
        operation_id=operation_id,
        timeout_sec=float(args.barrier_timeout_sec),
        candidate_checkpoint=candidate_path,
        candidate_id=f"r{round_id:03d}",
        candidate_mode="actor",
        eval_stage="paired_probe",
        pair_offset=round_id * 10_000,
        pair_count=int(state.worker_count),
    )
    rows = [row for row in operation_rows(snapshot) if not row.get("skipped")]
    candidate_successes = sum(int(bool(row.get("candidate_success"))) for row in rows)
    incumbent_successes = sum(int(bool(row.get("incumbent_success"))) for row in rows)
    n = len(rows)
    return {
        "round_id": round_id,
        "probe_sr": candidate_successes / max(n, 1),
        "probe_successes": candidate_successes,
        "incumbent_sr": incumbent_successes / max(n, 1),
        "incumbent_successes": incumbent_successes,
        "probe_episodes": n,
        "rows": rows,
        "timestamp": time.time(),
    }


def _dispatch_and_wait(
    run_dir: Path,
    *,
    phase: str,
    round_id: int,
    operation_id: str,
    timeout_sec: float,
    **state_kwargs: Any,
) -> BarrierSnapshot:
    worker_count = read_round_state(run_dir).worker_count
    snapshot = barrier_snapshot(run_dir, operation_id, worker_count)
    if snapshot.completed < worker_count:
        current = read_round_state(run_dir)
        if current.operation_id != operation_id:
            current = publish_operation(
                run_dir,
                phase=phase,
                round_id=round_id,
                operation_id=operation_id,
                **state_kwargs,
            )
        elif current.phase != phase:
            raise RuntimeError(
                f"operation {operation_id} has unexpected phase {current.phase}"
            )
        snapshot = wait_for_barrier(
            run_dir,
            operation_id,
            worker_count,
            timeout_sec=timeout_sec,
        )
    if snapshot.failures:
        errors = [
            str(row.get("error", "unknown worker failure"))
            for row in snapshot.failures
        ]
        raise RuntimeError(
            f"operation {operation_id} has worker failures: {errors}"
        )
    try:
        clean_operation_artifacts(run_dir, operation_id)
    except OSError:
        log.warning(
            "claim cleanup failed for %s; continuing after completed barrier",
            operation_id,
            exc_info=True,
        )
    return snapshot


def _completed_round_ids(run_dir: Path) -> set[int]:
    completed: set[int] = set()
    for path in (run_dir / "reports").glob("round_*_final.json"):
        try:
            completed.add(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return completed


def _offline_due(round_id: int, rounds_per_offline: int) -> bool:
    return (int(round_id) + 1) % int(rounds_per_offline) == 0


def _pending_offline_round_ids(
    run_dir: Path, *, rounds_per_offline: int
) -> list[int]:
    """Completed rounds whose scheduled offline AE never wrote a report."""
    pending: list[int] = []
    for round_id in sorted(_completed_round_ids(run_dir)):
        if not _offline_due(round_id, rounds_per_offline):
            continue
        report = run_dir / "reports" / f"round_{round_id:03d}_offline.json"
        if not report.is_file():
            pending.append(int(round_id))
    return pending


def _complete_offline_ae_phase(
    args: argparse.Namespace,
    model: MolmoAct2RLTCF,
    device: torch.device,
    run_dir: Path,
    round_id: int,
) -> dict[str, Any]:
    log.info("Round %d: starting offline AE update", round_id)
    offline = _run_offline_ae_update(args, model, device, run_dir, round_id)
    atomic_write_json(
        run_dir / "reports" / f"round_{round_id:03d}_offline.json",
        offline,
    )
    if not offline.get("skipped"):
        _restart_ae_servers(args, run_dir)
        state = read_round_state(run_dir)
        _run_val_video(
            args,
            round_id,
            tag=f"round_{round_id:03d}_offline_ae",
            checkpoint=Path(state.incumbent_checkpoint),
            mode="reference",
        )
    return offline


def run_learner(args: argparse.Namespace) -> None:
    _initialize_learner(args)
    run_dir = Path(args.run_dir).resolve()
    device = torch.device(args.device)
    model: MolmoAct2RLTCF | None = None
    optimizers: dict[str, torch.optim.Optimizer] = {}
    if not args.mock:
        model = prepare_cfgrl_model(
            Path(read_round_state(run_dir).incumbent_checkpoint),
            device=args.device,
            hidden=int(args.hidden),
            n_hidden_actor=int(args.n_hidden_actor),
            n_hidden_critic=int(args.n_hidden_critic),
            z_expand_dim=int(args.z_expand_dim),
            o_dim=int(args.cfgrl_o_dim),
        )
        # Every candidate checkpoint saved from this model carries the
        # deployment guidance weight, so probes evaluate the policy that
        # would actually be deployed (never w=0).
        model.cfgrl_w = float(args.w_deploy)
        optimizers = build_rlt_optimizers(
            model,
            lr_actor=float(args.actor_lr),
            lr_critic=float(args.critic_lr),
        )
    completed = _completed_round_ids(run_dir)
    start_round = min(
        (
            round_id
            for round_id in range(int(args.max_rounds))
            if round_id not in completed
        ),
        default=int(args.max_rounds),
    )
    _backfill_missing_val_videos(args, run_dir, completed)
    rounds_per_offline = int(args.rounds_per_offline)
    if model is not None:
        for round_id in _pending_offline_round_ids(
            run_dir, rounds_per_offline=rounds_per_offline
        ):
            if _STOP_REQUESTED:
                break
            _complete_offline_ae_phase(
                args, model, device, run_dir, round_id
            )
    for round_id in range(start_round, int(args.max_rounds)):
        if _STOP_REQUESTED:
            break
        state = read_round_state(run_dir)
        report = _run_online_round(
            args, state, round_id, model, optimizers, device
        )
        log.info(
            "Round %d complete: probe_sr=%.3f updates=%d",
            round_id,
            report["probe"]["probe_sr"],
            report["updates"],
        )
        if _offline_due(round_id, rounds_per_offline) and model is not None:
            _complete_offline_ae_phase(
                args, model, device, run_dir, round_id
            )
    if not _STOP_REQUESTED:
        publish_stop(run_dir, "learner completed configured rounds")


def _restart_ae_servers(
    args: argparse.Namespace, run_dir: Path
) -> None:
    """Signal per-GPU HTTP servers to reload the updated AE checkpoint.

    The launcher watches for ``ae_reload_request.json`` and restarts the
    servers with the latest AE trainable checkpoint.
    """
    atomic_write_json(
        run_dir / "coordination" / "ae_reload_request.json",
        {
            "ae_checkpoint": str(
                (run_dir / "checkpoints" / "ae_trainable_latest.pt").resolve()
            ),
            "requested_at": time.time(),
        },
    )


def _policy_args(
    args: argparse.Namespace,
    model: MolmoAct2RLTCF,
    *,
    rollout_seed: int,
    benchmark_dir: Path | str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        eval_only=True,
        server_host=str(args.server_host),
        server_port=int(args.server_port),
        server_request_timeout_sec=float(args.server_request_timeout_sec),
        use_cf_guide=False,
        actor_mode="rlt",
        tune_token_online=False,
        export_offline_tokens=False,
        retain_tokens=False,
        explore_residual_std=0.0,
        explore_deploy_std=0.0,
        explore_warmup_mult=1.0,
        actor_mixture_prob=0.0,
        guide_on_reference=False,
        residual_clip=None,
        always_collect_actor=False,
        actor_bc_episodes=0,
        always_collect_after_episodes=None,
        seed=int(rollout_seed),
        deterministic_rollout_seeds=True,
        cf_mode="flow",
        cfgrl=bool(model.use_cfgrl),
        cfgrl_w=float(model.cfgrl_w),
        cfgrl_o_dim=int(model.cfgrl_o_dim),
        n_critics=int(model.n_critics),
        use_cf_guide_requested=False,
        flow_steps=int(model.flow_steps),
        guidance_coef=float(model.guidance_coef),
        benchmark_dir=str(benchmark_dir or args.benchmark_dir),
        horizon=int(args.horizon),
    )


def _load_cached_model(checkpoint: Path) -> MolmoAct2RLTCF:
    checkpoint = Path(checkpoint).resolve()
    key = (str(checkpoint), checkpoint.stat().st_mtime_ns)
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = MolmoAct2RLTCF.load(
            str(checkpoint),
            map_location="cpu",
        )
        model.eval()
        for stale_key in [
            existing
            for existing in _MODEL_CACHE
            if existing[0] == str(checkpoint) and existing != key
        ]:
            del _MODEL_CACHE[stale_key]
        if len(_MODEL_CACHE) >= 4:
            del _MODEL_CACHE[next(iter(_MODEL_CACHE))]
        _MODEL_CACHE[key] = model
    return model


def _normalize_trajectory(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Map pop_episode key names to the canonical V20 trajectory schema."""
    aliases = {
        "reference_actions": ("reference_actions", "references"),
        "executed_actions": ("executed_actions", "executed"),
        "action_masks": ("action_masks", "masks"),
    }
    normalized = dict(trajectory)
    for canonical, candidates in aliases.items():
        for key in candidates:
            value = trajectory.get(key)
            if value is not None and len(value) > 0:
                normalized[canonical] = value
                break
        else:
            raise KeyError(
                f"trajectory missing {canonical} (tried {candidates})"
            )
    return normalized


def _val_benchmark_dir(args: argparse.Namespace) -> Path:
    if getattr(args, "val_benchmark_dir", None) is not None:
        return Path(args.val_benchmark_dir).resolve()
    bench = getattr(args, "benchmark_dir", None)
    if bench:
        train_dir = Path(bench).resolve()
        candidate = train_dir.parent / "val"
        if candidate.is_dir():
            return candidate
        return train_dir
    return Path("val")


def _write_validation_mp4(frames: Sequence[Any], path: Path, *, fps: int = 20) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [
        np.asarray(frame, dtype=np.uint8)
        for frame in frames
        if frame is not None and np.asarray(frame).size > 0
    ]
    if not arrays:
        arrays = [np.zeros((64, 64, 3), dtype=np.uint8)]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        try:
            writer = imageio.get_writer(
                str(temporary), format="ffmpeg", fps=int(fps), quality=5
            )
            try:
                for frame in arrays:
                    if frame.ndim == 2:
                        frame = np.repeat(frame[:, :, None], 3, axis=2)
                    writer.append_data(frame)
            finally:
                writer.close()
        except (ImportError, OSError, ValueError, RuntimeError):
            rgb = []
            for frame in arrays:
                if frame.ndim == 2:
                    frame = np.repeat(frame[:, :, None], 3, axis=2)
                rgb.append(frame)
            imageio.mimwrite(str(temporary), rgb, format="mp4", fps=int(fps))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _backfill_missing_val_videos(
    args: argparse.Namespace,
    run_dir: Path,
    completed: set[int],
) -> None:
    """Record held-out videos for rounds that finished before video saving."""
    for round_id in sorted(completed):
        if _STOP_REQUESTED:
            return
        candidate_path = (
            run_dir / "checkpoints" / "candidates" / f"candidate_r{round_id:03d}.pt"
        )
        candidate_video = run_dir / "videos" / f"round_{round_id:03d}_candidate.mp4"
        if candidate_path.is_file() and not candidate_video.is_file():
            log.info("Backfilling validation video for completed round %d", round_id)
            _run_val_video(
                args,
                round_id,
                tag=f"round_{round_id:03d}_candidate",
                checkpoint=candidate_path,
                mode="actor",
            )
        offline_report = run_dir / "reports" / f"round_{round_id:03d}_offline.json"
        offline_video = run_dir / "videos" / f"round_{round_id:03d}_offline_ae.mp4"
        if not offline_report.is_file() or offline_video.is_file():
            continue
        try:
            offline = json.loads(offline_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if offline.get("skipped"):
            continue
        incumbent = Path(read_round_state(run_dir).incumbent_checkpoint)
        if not incumbent.is_file():
            continue
        log.info("Backfilling offline-AE validation video for round %d", round_id)
        _run_val_video(
            args,
            round_id,
            tag=f"round_{round_id:03d}_offline_ae",
            checkpoint=incumbent,
            mode="reference",
        )


def _run_val_video(
    args: argparse.Namespace,
    round_id: int,
    *,
    tag: str,
    checkpoint: Path,
    mode: str,
) -> dict[str, Any]:
    """Ask worker 0 to roll one held-out episode and save an MP4."""
    run_dir = Path(args.run_dir).resolve()
    operation_id = f"r{round_id:03d}_val_{tag}"
    snapshot = _dispatch_and_wait(
        run_dir,
        phase="val_video",
        round_id=round_id,
        operation_id=operation_id,
        timeout_sec=float(args.barrier_timeout_sec),
        candidate_checkpoint=Path(checkpoint),
        candidate_mode=str(mode),
        eval_stage=str(tag),
        message=f"validation video {tag}",
    )
    rows = [row for row in operation_rows(snapshot) if not row.get("skipped")]
    report = rows[0] if rows else {"success": False, "video_path": None}
    log.info(
        "Round %d validation video tag=%s success=%s path=%s",
        round_id,
        tag,
        report.get("success"),
        report.get("video_path"),
    )
    return report


def _perform_val_video(
    args: argparse.Namespace,
    state: RoundState,
    worker_id: int,
) -> dict[str, Any]:
    if worker_id != 0:
        return {
            "kind": "val_video",
            "skipped": True,
            "worker_id": worker_id,
        }
    run_dir = Path(args.run_dir).resolve()
    tag = str(state.eval_stage or "val")
    checkpoint = Path(
        state.candidate_checkpoint or state.incumbent_checkpoint
    )
    mode = str(state.candidate_mode if state.candidate_checkpoint else state.incumbent_mode)
    seed = derive_seed(int(args.rollout_seed_root), "val_video", state.round_id, tag)
    video_path = run_dir / "videos" / f"{tag}.mp4"
    meta_path = run_dir / "videos" / f"{tag}.json"
    if args.mock:
        _write_validation_mp4([], video_path)
        success = True
        rollout_ok = True
        n_frames = 1
    else:
        output_dir = (
            run_dir / "rollouts" / state.operation_id / f"worker_{worker_id:03d}"
        )
        success, rollout_ok, trajectory = _rollout_checkpoint(
            args,
            checkpoint,
            mode=mode,
            rollout_seed=seed,
            episode_id=10_000_000 + int(state.round_id),
            output_dir=output_dir,
            benchmark_dir=_val_benchmark_dir(args),
            episode_idx=int(args.val_episode_idx),
            record_video=True,
        )
        frames = trajectory.get("video_frames") or trajectory.get("external_cams") or []
        _write_validation_mp4(frames, video_path)
        n_frames = len(frames)
    meta = {
        "kind": "val_video",
        "tag": tag,
        "round_id": state.round_id,
        "success": bool(success),
        "rollout_ok": bool(rollout_ok),
        "mode": mode,
        "checkpoint": str(checkpoint.resolve()),
        "video_path": str(video_path.resolve()),
        "n_frames": int(n_frames),
        "benchmark_dir": str(_val_benchmark_dir(args)),
        "episode_idx": int(args.val_episode_idx),
        "rollout_seed": int(seed),
        "timestamp": time.time(),
    }
    atomic_write_json(meta_path, meta)
    if not rollout_ok:
        raise RuntimeError(f"validation video rollout failed for {tag}")
    return meta


def _mock_trajectory(
    *,
    marker: int,
    success: bool,
) -> dict[str, Any]:
    reference = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    reference[:, 0] = (int(marker) % 1000) / 1000.0
    return {
        "zs": [np.full((Z_DIM,), marker % 17, dtype=np.float32)],
        "proprios": [np.zeros((ACTION_DIM,), dtype=np.float32)],
        "reference_actions": [reference],
        "executed_actions": [reference.copy()],
        "rewards": [
            np.full((CHUNK_SIZE,), float(success), dtype=np.float32)
        ],
        "action_masks": [np.ones((CHUNK_SIZE,), dtype=np.float32)],
        "n_steps": CHUNK_SIZE,
    }


def _configure_policy_mode(policy: Any, mode: str) -> None:
    if mode not in {"reference", "actor"}:
        raise ValueError(f"unsupported V20 policy mode {mode}")
    actor = mode == "actor"
    policy.eval_force_reference = not actor
    policy.eval_force_actor = actor
    policy.actor_mixture_prob = 0.0
    policy.always_collect_actor = actor
    policy.deploy_actor = actor
    policy.collect_episode_index = 10**9 if actor else 0


def _rollout_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
    *,
    mode: str,
    rollout_seed: int,
    episode_id: int,
    output_dir: Path,
    benchmark_dir: Path | str | None = None,
    episode_idx: int | None = None,
    record_video: bool = False,
) -> tuple[bool, bool, dict[str, Any]]:
    model = _load_cached_model(checkpoint)
    policy_args = _policy_args(
        args,
        model,
        rollout_seed=rollout_seed,
        benchmark_dir=benchmark_dir,
    )
    policy, _config = _build_eval_policy(
        policy_args,
        model,
        torch.device("cpu"),
        prefer_server_z=True,
        retain_tokens=False,
    )
    _configure_policy_mode(policy, mode)
    isolated = _run_isolated_evaluation(
        policy_args,
        policy,
        episode_idx=int(
            args.benchmark_episode_idx if episode_idx is None else episode_idx
        ),
        episode_dir=output_dir,
        episode_id=int(episode_id),
        rollout_seed=int(rollout_seed),
        rollout_checkpoint=checkpoint,
        record_video=bool(record_video),
    )
    return (
        isolated.success,
        isolated.rollout_ok,
        _normalize_trajectory(isolated.trajectory),
    )


def _atomic_episode_save(replay: ChunkReplay, path: Path) -> str:
    _atomic_replay_save(replay, path)
    return checkpoint_sha256(path)


def _perform_collect(
    args: argparse.Namespace,
    state: RoundState,
    worker_id: int,
) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    uid = allocate_global_uid(run_dir)
    seed = collection_seed(state, worker_id)
    if args.mock:
        success = worker_id < int(args.mock_successes_per_wave)
        rollout_ok = True
        trajectory = _mock_trajectory(marker=seed, success=success)
    else:
        output_dir = (
            run_dir
            / "rollouts"
            / state.operation_id
            / f"worker_{worker_id:03d}"
        )
        success, rollout_ok, trajectory = _rollout_checkpoint(
            args,
            Path(state.incumbent_checkpoint),
            mode=state.incumbent_mode,
            rollout_seed=seed,
            episode_id=uid,
            output_dir=output_dir,
        )
    if not rollout_ok or not trajectory.get("zs"):
        raise RuntimeError("collect rollout was invalid or empty")
    replay = ChunkReplay(
        max_transitions=10_000,
        pos_frac=0.5,
        benchmark_pose_cycle=24,
        seed=seed,
    )
    replay.add_episode_chunks(
        trajectory["zs"],
        trajectory["proprios"],
        trajectory["reference_actions"],
        trajectory["executed_actions"],
        trajectory["rewards"],
        trajectory["action_masks"],
        success=bool(success),
        gamma=float(args.gamma),
        episode_id=uid,
        trajectory_uid=uid,
        pose_idx=int(args.target_pose_idx),
        source_policy=(
            ReplaySource.INCUMBENT
            if state.incumbent_mode == "actor"
            else ReplaySource.ONLINE_REFERENCE
        ),
        worker_id=worker_id,
        round_id=state.round_id,
        policy_version=state.incumbent_version,
    )
    path = (
        run_dir
        / "replay"
        / "journal"
        / f"worker_{worker_id:03d}"
        / f"episode_{uid}.npz"
    )
    digest = _atomic_episode_save(replay, path)
    image_path = ""
    image_sha256 = ""
    if not args.mock and trajectory.get("external_cams"):
        image_replay = ImageChunkReplay(
            max_transitions=10_000,
            benchmark_pose_cycle=24,
            seed=seed,
        )
        image_replay.add_episode(
            zs=trajectory["zs"],
            proprios=trajectory["proprios"],
            references=trajectory["reference_actions"],
            executed=trajectory["executed_actions"],
            rewards=trajectory["rewards"],
            masks=trajectory["action_masks"],
            external_cams=trajectory["external_cams"],
            wrist_cams=trajectory["wrist_cams"],
            instructions=trajectory["instructions"],
            success=bool(success),
            gamma=float(args.gamma),
            episode_id=uid,
            full_references=trajectory.get("full_references"),
            full_executed=trajectory.get("full_executed"),
            sources_native=trajectory.get("sources_native"),
        )
        image_path_obj = (
            run_dir
            / "replay"
            / "image_journal"
            / f"worker_{worker_id:03d}"
            / f"episode_{uid}.npz"
        )
        image_path_obj.parent.mkdir(parents=True, exist_ok=True)
        image_tmp = image_path_obj.with_name(
            f".{image_path_obj.name}.{os.getpid()}.tmp.npz"
        )
        image_replay.save_npz(str(image_tmp))
        os.replace(image_tmp, image_path_obj)
        image_path = str(image_path_obj.resolve())
        image_sha256 = checkpoint_sha256(image_path_obj)
    entry = {
        "schema_version": 1,
        "kind": "collect",
        "operation_id": state.operation_id,
        "round_id": state.round_id,
        "collect_wave": state.collect_wave,
        "trajectory_uid": uid,
        "worker_id": worker_id,
        "pose_idx": int(args.target_pose_idx),
        "policy_version": state.incumbent_version,
        "source_policy": (
            "incumbent"
            if state.incumbent_mode == "actor"
            else "online_reference"
        ),
        "success": bool(success),
        "replay_path": str(path.resolve()),
        "sha256": digest,
        "image_replay_path": image_path,
        "image_sha256": image_sha256,
        "rollout_seed": seed,
    }
    register_journal_episode(run_dir, entry)
    return entry


def _perform_probe(
    args: argparse.Namespace,
    state: RoundState,
    worker_id: int,
) -> dict[str, Any]:
    """Run a paired probe: candidate and incumbent on the same pose and seed."""
    if state.candidate_checkpoint is None:
        raise RuntimeError("probe operation has no candidate checkpoint")
    run_dir = Path(args.run_dir).resolve()
    pair_id = int(state.pair_offset) + worker_id
    seed = pair_seed(state, pair_id)
    if args.mock:
        candidate_success = worker_id < int(args.mock_successes_per_wave)
        incumbent_success = False
        rollout_ok = True
    else:
        output_dir = (
            run_dir
            / "rollouts"
            / state.operation_id
            / f"worker_{worker_id:03d}"
        )
        candidate_success, candidate_ok, _ = _rollout_checkpoint(
            args,
            Path(state.candidate_checkpoint),
            mode="actor",
            rollout_seed=seed,
            episode_id=pair_id * 2,
            output_dir=output_dir / "candidate",
        )
        incumbent_success, incumbent_ok, _ = _rollout_checkpoint(
            args,
            Path(state.incumbent_checkpoint),
            mode=state.incumbent_mode,
            rollout_seed=seed,
            episode_id=pair_id * 2 + 1,
            output_dir=output_dir / "incumbent",
        )
        rollout_ok = bool(candidate_ok and incumbent_ok)
    row = {
        "schema_version": 1,
        "kind": "probe",
        "operation_id": state.operation_id,
        "round_id": state.round_id,
        "worker_id": worker_id,
        "pose_idx": int(args.target_pose_idx),
        "pair_id": pair_id,
        "candidate_success": bool(candidate_success),
        "incumbent_success": bool(incumbent_success),
        # Legacy key: candidate SR remains the headline probe metric.
        "success": bool(candidate_success),
        "valid": True,
        "rollout_ok": bool(rollout_ok),
        "rollout_seed": seed,
        "timestamp": time.time(),
    }
    if not rollout_ok:
        raise RuntimeError("probe rollout was invalid")
    return row


def run_worker(args: argparse.Namespace) -> None:
    run_dir = validate_run_dir(Path(args.run_dir))
    worker_id = int(args.worker_id)
    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    expected_worker_count = int(config["worker_count"])
    if int(args.worker_count) != expected_worker_count:
        raise ValueError(
            f"worker count mismatch: CLI={args.worker_count} "
            f"run={expected_worker_count}"
        )
    if int(args.target_pose_idx) != int(config["target_pose_idx"]):
        raise ValueError(
            f"target pose mismatch: CLI={args.target_pose_idx} "
            f"run={config['target_pose_idx']}"
        )
    if bool(args.mock) != bool(config.get("mock", False)):
        raise ValueError(
            f"mock mode mismatch: CLI={args.mock} run={config.get('mock')}"
        )
    os.environ["RLT_VLA_PREFETCH"] = "0"
    last_operation = ""
    while not _STOP_REQUESTED:
        state = read_round_state(run_dir)
        if worker_id >= state.worker_count:
            raise ValueError(
                f"worker {worker_id} outside configured count {state.worker_count}"
            )
        if state.phase == "stop":
            break
        if state.phase not in {"collect", "probe", "val_video"}:
            if state.operation_id != last_operation:
                write_worker_heartbeat(
                    run_dir,
                    worker_id,
                    operation_id=state.operation_id,
                    status="idle",
                )
                last_operation = state.operation_id
            time.sleep(float(args.worker_poll_sec))
            continue
        with claim_worker_operation(
            run_dir,
            state.operation_id,
            worker_id,
        ) as should_run:
            if not should_run:
                time.sleep(float(args.worker_poll_sec))
                continue
            write_worker_heartbeat(
                run_dir,
                worker_id,
                operation_id=state.operation_id,
                status="running",
            )
            try:
                last_error: Exception | None = None
                result: dict[str, Any] | None = None
                for attempt in range(
                    1,
                    int(args.worker_operation_attempts) + 1,
                ):
                    try:
                        if state.phase == "collect":
                            result = _perform_collect(args, state, worker_id)
                        elif state.phase == "probe":
                            result = _perform_probe(args, state, worker_id)
                        else:
                            result = _perform_val_video(args, state, worker_id)
                        break
                    except Exception as error:  # noqa: BLE001
                        last_error = error
                        if attempt >= int(args.worker_operation_attempts):
                            raise
                        current = read_round_state(run_dir)
                        if current.operation_id != state.operation_id:
                            raise StaleOperationError(
                                f"operation advanced to {current.operation_id}"
                            ) from error
                        log.warning(
                            "Worker %d retrying operation %s after attempt "
                            "%d/%d: %s",
                            worker_id,
                            state.operation_id,
                            attempt,
                            int(args.worker_operation_attempts),
                            error,
                        )
                        time.sleep(float(args.worker_operation_retry_sec))
                if result is None:
                    raise RuntimeError(
                        f"operation produced no result: {last_error!r}"
                    )
                write_worker_done(
                    run_dir,
                    state,
                    worker_id,
                    valid=True,
                    payload={"result": result},
                )
            except StaleOperationError:
                log.warning(
                    "Worker %d completed stale operation %s",
                    worker_id,
                    state.operation_id,
                )
            except Exception as error:  # noqa: BLE001
                log.exception(
                    "Worker %d failed operation %s",
                    worker_id,
                    state.operation_id,
                )
                try:
                    write_worker_done(
                        run_dir,
                        state,
                        worker_id,
                        valid=False,
                        payload={"error": repr(error)},
                    )
                except StaleOperationError:
                    pass
            finally:
                write_worker_heartbeat(
                    run_dir,
                    worker_id,
                    operation_id=state.operation_id,
                    status="complete",
                )
        last_operation = state.operation_id


def run_mock_smoke(args: argparse.Namespace) -> None:
    run_dir = Path(args.run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError(
            f"mock smoke run directory already exists: {run_dir}"
        )
    run_dir.mkdir(parents=True)
    base = run_dir / "mock_base.pt"
    base.write_bytes(b"v20-mock-checkpoint\n")
    seed_replay = run_dir / "mock_seed_replay.npz"
    _mock_seed_replay(
        seed_replay,
        target_pose_idx=int(args.target_pose_idx),
    )
    script = Path(__file__).resolve()
    common = [
        "--run_dir",
        str(run_dir),
        "--worker_count",
        str(args.worker_count),
        "--rollout_seed_root",
        str(args.rollout_seed_root),
        "--target_pose_idx",
        str(args.target_pose_idx),
        "--mock",
        "--collect_waves_per_round",
        str(args.collect_waves_per_round),
        "--rounds_per_offline",
        str(args.rounds_per_offline),
        "--max_rounds",
        str(args.max_rounds),
        "--mock_successes_per_wave",
        str(args.mock_successes_per_wave),
        "--worker_poll_sec",
        "0.02",
        "--barrier_timeout_sec",
        str(args.barrier_timeout_sec),
    ]
    learner_log = (run_dir / "learner.log").open("wb")
    learner = subprocess.Popen(
        [
            sys.executable,
            str(script),
            "learner",
            *common,
            "--base_checkpoint",
            str(base),
            "--seed_replay",
            str(seed_replay),
        ],
        stdout=learner_log,
        stderr=subprocess.STDOUT,
    )
    deadline = time.monotonic() + 30.0
    while (
        not (run_dir / ".v20_monotone_incumbent").is_file()
        and learner.poll() is None
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    workers: list[subprocess.Popen[Any]] = []
    worker_logs: list[Any] = []
    worker_threads: list[threading.Thread] = []
    thread_errors: list[BaseException] = []
    thread_errors_lock = threading.Lock()

    def run_thread_worker(worker_args: argparse.Namespace) -> None:
        try:
            run_worker(worker_args)
        except BaseException as error:  # noqa: BLE001
            with thread_errors_lock:
                thread_errors.append(error)

    try:
        if learner.poll() is not None:
            raise RuntimeError(
                f"mock learner exited during initialization: {learner.returncode}"
            )
        if args.smoke_inprocess_workers:
            for worker_id in range(int(args.worker_count)):
                worker_args = argparse.Namespace(
                    **{
                        **vars(args),
                        "role": "worker",
                        "worker_id": worker_id,
                        "mock": True,
                    }
                )
                thread = threading.Thread(
                    target=run_thread_worker,
                    args=(worker_args,),
                    name=f"v20-smoke-worker-{worker_id:03d}",
                )
                thread.start()
                worker_threads.append(thread)
        else:
            for worker_id in range(int(args.worker_count)):
                handle = (run_dir / f"worker_{worker_id:03d}.log").open("wb")
                worker_logs.append(handle)
                workers.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(script),
                            "worker",
                            *common,
                            "--worker_id",
                            str(worker_id),
                        ],
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                    )
                )
        return_code = learner.wait(timeout=float(args.smoke_timeout_sec))
        if return_code != 0:
            raise RuntimeError(f"mock learner failed with code {return_code}")
        for worker in workers:
            worker.wait(timeout=20)
            if worker.returncode != 0:
                raise RuntimeError(
                    f"mock worker failed with code {worker.returncode}"
                )
        for thread in worker_threads:
            thread.join(timeout=20)
            if thread.is_alive():
                raise RuntimeError(f"mock worker thread did not exit: {thread.name}")
        if thread_errors:
            raise RuntimeError(
                f"mock worker thread failed: {thread_errors[0]!r}"
            )
    finally:
        if learner.poll() is None:
            learner.terminate()
        for worker in workers:
            if worker.poll() is None:
                worker.terminate()
        if (run_dir / ".v20_monotone_incumbent").is_file():
            try:
                publish_stop(run_dir, "mock smoke cleanup")
            except Exception:  # noqa: BLE001
                pass
        for thread in worker_threads:
            thread.join(timeout=5)
        learner_log.close()
        for handle in worker_logs:
            handle.close()
    probe_rows = read_jsonl(run_dir / "reports" / "probe_sr.jsonl")
    if not probe_rows:
        raise RuntimeError("mock smoke produced no probe report")
    rounds = [int(row["round_id"]) for row in probe_rows]
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "probe_rounds": rounds,
                "probe_sr": [float(row["probe_sr"]) for row in probe_rows],
            },
            indent=2,
        )
    )


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--worker_count", type=int, default=32)
    parser.add_argument("--rollout_seed_root", type=int, default=20_260_820)
    parser.add_argument("--target_pose_idx", type=int, default=0)
    parser.add_argument("--benchmark_episode_idx", type=int, default=0)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--collect_waves_per_round", type=int, default=4)
    parser.add_argument("--rounds_per_offline", type=int, default=2)
    parser.add_argument("--max_rounds", type=int, default=100)
    parser.add_argument("--barrier_timeout_sec", type=float, default=3_600.0)
    parser.add_argument("--worker_poll_sec", type=float, default=0.5)
    parser.add_argument("--worker_operation_attempts", type=int, default=3)
    parser.add_argument("--worker_operation_retry_sec", type=float, default=2.0)
    parser.add_argument("--replay_capacity", type=int, default=500_000)
    parser.add_argument("--replay_pos_frac", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--server_host", default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--server_request_timeout_sec", type=float, default=180.0)
    parser.add_argument("--benchmark_dir", type=Path)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--mock_successes_per_wave", type=int, default=8)
    parser.add_argument(
        "--val_benchmark_dir",
        type=Path,
        help="held-out benchmark for post-round/phase validation videos",
    )
    parser.add_argument(
        "--val_episode_idx",
        type=int,
        default=0,
        help="benchmark episode index used for the validation video",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="role", required=True)
    learner = subparsers.add_parser("learner")
    _add_common_arguments(learner)
    learner.add_argument("--base_checkpoint", type=Path)
    learner.add_argument("--seed_replay", type=Path)
    learner.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    learner.add_argument("--hidden", type=int, default=1_024)
    learner.add_argument("--n_hidden_actor", type=int, default=10)
    learner.add_argument("--n_hidden_critic", type=int, default=5)
    learner.add_argument("--z_expand_dim", type=int, default=512)
    learner.add_argument("--cfgrl_o_dim", type=int, default=16)
    learner.add_argument("--batch_size", type=int, default=256)
    learner.add_argument("--actor_lr", type=float, default=1e-4)
    learner.add_argument("--critic_lr", type=float, default=3e-4)
    learner.add_argument("--cfgrl_dropout", type=float, default=0.1)
    learner.add_argument(
        "--ref_dropout",
        type=float,
        default=0.5,
        help="reference-action dropout for the CFGRL actor; matches the "
        "critic TD bootstrap regime",
    )
    learner.add_argument(
        "--w_deploy",
        type=float,
        default=1.0,
        help="CFGRL guidance weight baked into candidate/probe checkpoints",
    )
    learner.add_argument(
        "--updates_per_wave",
        type=int,
        default=32,
        help="critic+actor step pairs per collection wave",
    )
    learner.add_argument(
        "--max_update_sec_per_wave",
        type=float,
        default=120.0,
        help="wall-clock cap for each wave's update budget",
    )
    learner.add_argument("--promotion_alpha", type=float, default=0.05)
    learner.add_argument("--promotion_min_gain", type=float, default=0.03)
    learner.add_argument("--clone_mse_max", type=float, default=0.02)
    learner.add_argument(
        "--cond_ref_mse_max",
        type=float,
        default=0.5,
        help="max deployed-head (w=1) MSE against the reference; gates the "
        "policy that is actually deployed",
    )
    learner.add_argument("--max_normalized_action", type=float, default=12.0)
    learner.add_argument("--diagnostic_batches", type=int, default=4)
    learner.add_argument("--temporal_bins", type=int, default=4)
    learner.add_argument("--ae_steps", type=int, default=512)
    learner.add_argument("--ae_batch_size", type=int, default=16)
    learner.add_argument("--ae_lr", type=float, default=1e-4)
    learner.add_argument(
        "--ae_accumulate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start each offline AE phase from the previous LoRA checkpoint",
    )

    worker = subparsers.add_parser("worker")
    _add_common_arguments(worker)
    worker.add_argument("--worker_id", type=int, required=True)

    smoke = subparsers.add_parser("mock-smoke")
    _add_common_arguments(smoke)
    smoke.add_argument("--smoke_timeout_sec", type=float, default=120.0)
    smoke.add_argument("--smoke_inprocess_workers", action="store_true")
    args = parser.parse_args()
    if args.worker_count <= 0:
        parser.error("--worker_count must be positive")
    if args.worker_operation_attempts <= 0:
        parser.error("--worker_operation_attempts must be positive")
    if args.collect_waves_per_round <= 0:
        parser.error("--collect_waves_per_round must be positive")
    if args.rounds_per_offline <= 0:
        parser.error("--rounds_per_offline must be positive")
    if args.role != "worker" and args.max_rounds <= 0:
        parser.error("--max_rounds must be positive")
    if args.role == "learner":
        if args.updates_per_wave <= 0:
            parser.error("--updates_per_wave must be positive")
        if args.max_update_sec_per_wave <= 0:
            parser.error("--max_update_sec_per_wave must be positive")
        if args.w_deploy < 0:
            parser.error("--w_deploy must be non-negative")
        if not 0.0 < args.promotion_alpha < 1.0:
            parser.error("--promotion_alpha must lie in (0, 1)")
        if args.promotion_min_gain < 0:
            parser.error("--promotion_min_gain must be non-negative")
        if not 0.0 <= args.ref_dropout < 1.0:
            parser.error("--ref_dropout must lie in [0, 1)")
        if args.cond_ref_mse_max <= 0:
            parser.error("--cond_ref_mse_max must be positive")
    if not args.mock and args.role == "worker" and args.benchmark_dir is None:
        parser.error("non-mock worker requires --benchmark_dir")
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    args = parse_args()
    if args.role == "learner":
        run_learner(args)
    elif args.role == "worker":
        run_worker(args)
    elif args.role == "mock-smoke":
        run_mock_smoke(args)
    else:
        raise AssertionError(f"unhandled role {args.role}")


if __name__ == "__main__":
    main()
