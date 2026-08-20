"""Critic-free, target-positive V20 challenger training."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from chunk_replay import ChunkReplay
from rlt_models import CFGRL_O_POS, CFGRL_O_UNCOND, MolmoAct2RLTCF
from train_rlt import build_rlt_optimizers, cfgrl_actor_step


@dataclass(frozen=True)
class ChallengerTrainingConfig:
    actor_steps: int = 2_048
    batch_size: int = 256
    actor_lr: float = 1e-4
    cond_dropout: float = 0.0
    target_pose_idx: int = 0
    temporal_bins: int = 4
    clone_mse_max: float = 0.02
    cond_ref_mse_max: float = 0.5
    max_normalized_action: float = 12.0
    diagnostic_batches: int = 4


@dataclass(frozen=True)
class ChallengerTrainingResult:
    target_success_episodes: int
    target_positive_fraction: float
    actor_steps: int
    mean_actor_loss: float
    final_actor_loss: float
    uncond_ref_mse: float
    cond_ref_mse: float
    max_abs_normalized_action: float
    finite_actions: bool
    clone_ok: bool
    cond_ok: bool
    action_bounds_ok: bool
    critic_updates: int
    token_updates: int


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def target_positive_fraction(target_successes: int) -> float:
    target_successes = int(target_successes)
    if target_successes >= 32:
        return 0.90
    if target_successes >= 16:
        return 0.75
    return 0.50


def optimizer_parameter_ids(
    optimizer: torch.optim.Optimizer,
) -> set[int]:
    return {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def build_v20_optimizers(
    model: MolmoAct2RLTCF,
    *,
    actor_lr: float,
) -> dict[str, torch.optim.Optimizer]:
    optimizers = build_rlt_optimizers(
        model,
        lr_actor=float(actor_lr),
        separate_actor_adapter=True,
    )
    critic_ids = optimizer_parameter_ids(optimizers["critic"])
    actor_ids = optimizer_parameter_ids(optimizers["actor"])
    adapter_ids = {id(parameter) for parameter in model.rlt_adapter_parameters()}
    if critic_ids & actor_ids:
        raise RuntimeError("V20 critic and actor optimizers overlap")
    if adapter_ids & critic_ids:
        raise RuntimeError("V20 critic optimizer owns actor-path adapter")
    if not adapter_ids <= actor_ids:
        raise RuntimeError("V20 actor optimizer does not own the full adapter")
    return optimizers


def prepare_cfgrl_model(
    checkpoint: Path,
    *,
    device: torch.device | str,
    hidden: int = 1_024,
    n_hidden_actor: int = 10,
    n_hidden_critic: int = 5,
    z_expand_dim: int = 512,
    o_dim: int = 16,
) -> MolmoAct2RLTCF:
    model = MolmoAct2RLTCF.load(str(checkpoint), map_location="cpu")
    if not model.use_cfgrl:
        model = model.as_cfgrl(
            hidden=int(hidden),
            n_hidden_actor=int(n_hidden_actor),
            n_hidden_critic=int(n_hidden_critic),
            z_expand_dim=int(z_expand_dim),
            o_dim=int(o_dim),
            layernorm_heads=True,
        )
    else:
        expected = {
            "hidden": int(hidden),
            "n_hidden_actor": int(n_hidden_actor),
            "n_hidden_critic": int(n_hidden_critic),
            "z_expand_dim": int(z_expand_dim),
            "cfgrl_o_dim": int(o_dim),
        }
        mismatches = {
            key: (int(getattr(model, key)), value)
            for key, value in expected.items()
            if int(getattr(model, key)) != value
        }
        if mismatches:
            raise RuntimeError(
                f"incumbent CFGRL architecture mismatch: {mismatches}"
            )
    model.freeze_token_encoder()
    model.to(device)
    return model


def _diagnose_candidate(
    model: MolmoAct2RLTCF,
    replay: ChunkReplay,
    *,
    config: ChallengerTrainingConfig,
    target_fraction: float,
    device: torch.device | str,
    seed: int,
) -> dict[str, float | bool]:
    model.eval()
    uncond_mses: list[float] = []
    cond_mses: list[float] = []
    max_abs = 0.0
    finite = True
    with torch.no_grad():
        for index in range(int(config.diagnostic_batches)):
            batch = replay.sample(
                int(config.batch_size),
                device=device,
                require_both_outcomes=True,
                target_pose_idx=int(config.target_pose_idx),
                target_positive_fraction=float(target_fraction),
                trajectory_first=True,
                temporal_bins=int(config.temporal_bins),
            )
            state = model.encode_state_from_z(
                batch["z"].detach(),
                batch["proprio"],
            )
            reference = model.normalize_action(batch["reference_actions"])
            uncond, _ = model.flow_sample(
                state,
                reference,
                o_cond=CFGRL_O_UNCOND,
                cfg_w=0.0,
                flow_noise_seed=int(seed) + index * 2,
            )
            conditioned, _ = model.flow_sample(
                state,
                reference,
                o_cond=CFGRL_O_POS,
                cfg_w=1.0,
                flow_noise_seed=int(seed) + index * 2 + 1,
            )
            mask = batch["action_mask"].unsqueeze(-1)
            denominator = mask.sum().clamp_min(1.0) * reference.shape[-1]
            uncond_mses.append(
                float((((uncond - reference) * mask) ** 2).sum() / denominator)
            )
            cond_mses.append(
                float(
                    (((conditioned - reference) * mask) ** 2).sum()
                    / denominator
                )
            )
            finite = bool(
                finite
                and torch.isfinite(uncond).all()
                and torch.isfinite(conditioned).all()
            )
            max_abs = max(
                max_abs,
                float(uncond.detach().abs().max()),
                float(conditioned.detach().abs().max()),
            )
    uncond_mse = float(np.mean(uncond_mses))
    cond_mse = float(np.mean(cond_mses))
    return {
        "uncond_ref_mse": uncond_mse,
        "cond_ref_mse": cond_mse,
        "max_abs_normalized_action": max_abs,
        "finite_actions": finite,
        "clone_ok": bool(np.isfinite(uncond_mse))
        and uncond_mse <= float(config.clone_mse_max),
        # The deployed policy is the conditional head at w=1, so its excursion
        # from the reference is gated directly, not just the w=0 anchor's.
        "cond_ok": bool(np.isfinite(cond_mse))
        and cond_mse <= float(config.cond_ref_mse_max),
        "action_bounds_ok": finite
        and max_abs <= float(config.max_normalized_action),
    }


def train_challenger(
    model: MolmoAct2RLTCF,
    replay: ChunkReplay,
    *,
    config: ChallengerTrainingConfig,
    device: torch.device | str,
    seed: int,
) -> ChallengerTrainingResult:
    if not model.use_cfgrl or not model.is_flow:
        raise RuntimeError("V20 challenger requires a CFGRL flow model")
    if not replay.has_both_outcomes():
        raise RuntimeError("V20 challenger requires both replay outcomes")
    target_successes = replay.target_successful_episode_count(
        config.target_pose_idx
    )
    if target_successes <= 0:
        raise RuntimeError("V20 challenger has no target-pose success")
    target_fraction = target_positive_fraction(target_successes)
    replay.rng = np.random.default_rng(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model.freeze_token_encoder()
    for parameter in model.critic.parameters():
        parameter.requires_grad_(False)
    for parameter in model.target_critic.parameters():
        parameter.requires_grad_(False)
    if model.guide is not None:
        for parameter in model.guide.parameters():
            parameter.requires_grad_(False)
    for parameter in model.actor.parameters():
        parameter.requires_grad_(True)
    for parameter in model.rlt_adapter_parameters():
        parameter.requires_grad_(True)
    optimizers = build_v20_optimizers(
        model,
        actor_lr=config.actor_lr,
    )

    losses: list[float] = []
    for _step in range(int(config.actor_steps)):
        batch = replay.sample(
            int(config.batch_size),
            device=device,
            require_both_outcomes=True,
            target_pose_idx=int(config.target_pose_idx),
            target_positive_fraction=float(target_fraction),
            trajectory_first=True,
            temporal_bins=int(config.temporal_bins),
        )
        info = cfgrl_actor_step(
            model,
            optimizers["actor"],
            batch,
            cond_dropout=float(config.cond_dropout),
            use_advantage_labels=False,
            train_adapter=True,
        )
        losses.append(float(info["actor_loss"]))

    diagnostics = _diagnose_candidate(
        model,
        replay,
        config=config,
        target_fraction=target_fraction,
        device=device,
        seed=int(seed) + 1_000_000,
    )
    return ChallengerTrainingResult(
        target_success_episodes=target_successes,
        target_positive_fraction=target_fraction,
        actor_steps=int(config.actor_steps),
        mean_actor_loss=float(np.mean(losses)) if losses else 0.0,
        final_actor_loss=float(losses[-1]) if losses else 0.0,
        uncond_ref_mse=float(diagnostics["uncond_ref_mse"]),
        cond_ref_mse=float(diagnostics["cond_ref_mse"]),
        max_abs_normalized_action=float(
            diagnostics["max_abs_normalized_action"]
        ),
        finite_actions=bool(diagnostics["finite_actions"]),
        clone_ok=bool(diagnostics["clone_ok"]),
        cond_ok=bool(diagnostics["cond_ok"]),
        action_bounds_ok=bool(diagnostics["action_bounds_ok"]),
        critic_updates=0,
        token_updates=0,
    )


def atomic_model_save(
    model: MolmoAct2RLTCF,
    path: Path,
    *,
    meta: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        model.save(str(temporary), meta=meta)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_candidate_variants(
    model: MolmoAct2RLTCF,
    directory: Path,
    *,
    round_id: int,
    weights: Sequence[float],
    training_result: ChallengerTrainingResult,
) -> list[dict[str, Any]]:
    directory = Path(directory)
    original_weight = float(model.cfgrl_w)
    rows: list[dict[str, Any]] = []
    try:
        for weight in weights:
            model.cfgrl_w = float(weight)
            candidate_id = (
                f"round_{int(round_id):03d}_w{int(round(weight * 100)):03d}"
            )
            path = directory / f"{candidate_id}.pt"
            meta = {
                "v20": True,
                "round_id": int(round_id),
                "candidate_id": candidate_id,
                "cfgrl_w": float(weight),
                "training": asdict(training_result),
            }
            atomic_model_save(model, path, meta=meta)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "checkpoint": str(path.resolve()),
                    "cfgrl_w": float(weight),
                    "sha256": checkpoint_sha256(path),
                    "clone_ok": training_result.clone_ok,
                    "action_bounds_ok": training_result.action_bounds_ok,
                }
            )
    finally:
        model.cfgrl_w = original_weight
    manifest_path = directory / f"round_{int(round_id):03d}.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "round_id": int(round_id),
                "training": asdict(training_result),
                "candidates": rows,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest_path)
    return rows


def clone_model(model: MolmoAct2RLTCF) -> MolmoAct2RLTCF:
    return copy.deepcopy(model)
