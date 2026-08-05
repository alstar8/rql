"""MolmoAct2 + CF online train to a target number of *valid* sim env steps.

Cycles FrankaPickDroidMiniBench episodes until ``--target_env_steps`` is reached.
Skips failed/zero-step rollouts (e.g. dead expert server) so they do not pollute
SR or the step budget. Rollout artifacts go under ``/tmp`` and are deleted.

Example (one GPU shard of 12.5M steps):

    python train_100m.py --target_env_steps 12500000 --server_port 8000 \\
      --log_every_steps 1000000 --out_dir runs/molmoact2_cf_100m/shard_0
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from buffer import StratifiedReplay, load_buffer  # noqa: E402
from models import MolmoAct2CF  # noqa: E402
from train_full import (  # noqa: E402
    OnlineReplay,
    _default_bench,
    _make_cf_train_policy,
    _mix_batch,
)
from train_offline import (  # noqa: E402
    build_optimizers,
    critic_is_healthy,
    critic_step,
    refiner_step,
)

log = logging.getLogger("molmoact2_cf.train_100m")


def _server_ok(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"http://{host}:{port}/act", timeout=timeout)
        return r.status_code == 200 and "ok" in r.text
    except Exception:  # noqa: BLE001
        return False


def _wait_server(host: str, port: int, max_wait_sec: float = 600.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < max_wait_sec:
        if _server_ok(host, port):
            return True
        time.sleep(5.0)
    return False


def _bench_size(bench: Path) -> int:
    from molmo_spaces.evaluation.benchmark_schema import load_all_episodes

    return len(load_all_episodes(bench))


def train_100m(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    # Rollouts on local /tmp to avoid filling NFS (workspace ~98% full).
    tmp_rollouts = Path(args.tmp_rollout_dir) / f"cf100m_port{args.server_port}"
    tmp_rollouts.mkdir(parents=True, exist_ok=True)

    if args.cf_ckpt and Path(args.cf_ckpt).is_file():
        model = MolmoAct2CF.load(args.cf_ckpt, map_location=device).to(device)
        log.info("Loaded CF ckpt %s", args.cf_ckpt)
    else:
        model = MolmoAct2CF(use_vla_features=True).to(device)
        log.info("Initialized fresh CF modules (VLA features)")
    if not model.bounded_critic and args.updates_per_episode > 0:
        raise ValueError(
            "Refusing to train a legacy unbounded-Q checkpoint. Collect/evaluate with "
            "--updates_per_episode 0, or start a fresh bounded critic."
        )

    offline = None
    if args.buffer and Path(args.buffer).is_file():
        arrays = load_buffer(Path(args.buffer))
        compatible = "features" in arrays or not model.use_vla_features
        if compatible and "proprio_mean" in arrays:
            model.set_norm_stats(
                arrays["proprio_mean"],
                arrays["proprio_std"],
                arrays["action_mean"],
                arrays["action_std"],
                arrays.get("feature_mean"),
                arrays.get("feature_std"),
            )
        elif compatible and "state_mean" in arrays:
            model.set_norm_stats(
                arrays["state_mean"],
                arrays["state_std"],
                arrays["action_mean"],
                arrays["action_std"],
            )
        if compatible:
            offline = StratifiedReplay(arrays, pos_frac=args.pos_frac, seed=args.seed)
        if offline is None:
            log.warning(
                "Skipping proprio-only offline buffer and its normalization for "
                "VLA training; collect a matched G=0 VLA replay first"
            )

    online = OnlineReplay(
        model, max_transitions=args.online_capacity, pos_frac=args.pos_frac, seed=args.seed
    )
    opt_q, opt_g, opt_alpha = build_optimizers(
        model,
        lr_q=args.lr_q,
        lr_g=args.lr_g,
        lr_alpha=args.lr_alpha,
    )

    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.assets_dir:
        os.environ["MLSPACES_ASSETS_DIR"] = args.assets_dir

    from molmo_spaces.evaluation.configs.evaluation_configs import MolmoAct2PolicyEvalConfig
    from molmo_spaces.evaluation.eval_main import run_evaluation

    CFTrainPolicy = _make_cf_train_policy(model, device)
    exp_cfg = MolmoAct2PolicyEvalConfig()
    exp_cfg.policy_config.remote_config = {
        "host": args.server_host,
        "port": int(args.server_port),
    }
    # Avoid writing huge H5 success filters onto NFS.
    if hasattr(exp_cfg, "filter_for_successful_trajectories"):
        exp_cfg.filter_for_successful_trajectories = False
    if hasattr(exp_cfg, "datagen_profiler"):
        exp_cfg.datagen_profiler = False

    if not _wait_server(args.server_host, args.server_port, max_wait_sec=args.server_wait_sec):
        raise RuntimeError(
            f"Expert server {args.server_host}:{args.server_port} not ready"
        )
    policy = CFTrainPolicy(exp_cfg)
    if args.policy_chunk_size > 0:
        policy.chunk_size = args.policy_chunk_size
    requested_g = not args.disable_g
    policy.enable_g = requested_g and (args.force_g or args.g_start_episodes <= 0)
    policy.explore_residual_std = (
        float(args.explore_residual_std) if args.updates_per_episode > 0 else 0.0
    )

    bench = Path(args.benchmark_dir) if args.benchmark_dir else _default_bench()
    n_bench = _bench_size(bench)
    # Cycle a contiguous shard of the bench (or whole bench if shard_size=0).
    shard_size = int(args.shard_size) if args.shard_size > 0 else n_bench
    shard_start = int(args.start_episode) % n_bench
    log.info(
        "100M-train: target_steps=%d log_every_steps=%d bench_n=%d "
        "shard_start=%d shard_size=%d server=%s:%d",
        args.target_env_steps,
        args.log_every_steps,
        n_bench,
        shard_start,
        shard_size,
        args.server_host,
        args.server_port,
    )

    env_steps = 0
    valid_eps = 0
    skipped_eps = 0
    successes = 0
    recent = deque(maxlen=100)
    t0 = time.time()
    last_q: dict[str, float] = {}
    last_g: dict[str, float] = {}
    last_log_steps = 0
    cycle = 0

    while env_steps < args.target_env_steps and (
        args.max_valid_episodes <= 0 or valid_eps < args.max_valid_episodes
    ):
        if not _server_ok(args.server_host, args.server_port):
            log.warning(
                "Server %s:%d down — waiting (env_steps=%d)",
                args.server_host,
                args.server_port,
                env_steps,
            )
            if not _wait_server(
                args.server_host, args.server_port, max_wait_sec=args.server_wait_sec
            ):
                log.error("Server still down after wait; sleeping 60s and retrying")
                time.sleep(60)
                continue
            # Re-connect policy session after restart.
            try:
                policy.prepare_model()
            except Exception:  # noqa: BLE001
                log.exception("prepare_model after server restart failed")

        recent_adv = float(last_g.get("predicted_advantage", 0.0))
        policy.enable_g = bool(
            requested_g
            and (
                args.force_g
                or (
                    valid_eps >= args.g_start_episodes
                    and online.has_both_outcomes()
                    and critic_is_healthy(last_q, args.g_max_mc_loss)
                    and recent_adv >= args.g_min_advantage
                )
            )
        )
        ep_idx = shard_start + (cycle % shard_size)
        cycle += 1
        ep_t0 = time.time()
        ep_out = tmp_rollouts / f"ep_{cycle:08d}"
        if ep_out.exists():
            shutil.rmtree(ep_out, ignore_errors=True)
        ep_out.mkdir(parents=True, exist_ok=True)

        rollout_ok = True
        try:
            results = run_evaluation(
                eval_config_cls=MolmoAct2PolicyEvalConfig,
                benchmark_dir=bench,
                task_horizon_steps=args.horizon,
                num_workers=1,
                use_wandb=False,
                preloaded_policy=policy,
                episode_idx=ep_idx,
                output_dir=ep_out,
            )
            success = bool(results.success_count > 0)
        except Exception as e:  # noqa: BLE001
            log.warning("episode %d failed: %s", ep_idx, e)
            success = False
            rollout_ok = False

        traj = policy.pop_episode()
        n_steps = len(traj["states"])
        # Drop temp rollout artifacts immediately (videos/H5 are huge).
        shutil.rmtree(ep_out, ignore_errors=True)

        if not rollout_ok or n_steps <= 0:
            skipped_eps += 1
            log.warning(
                "skip invalid/partial ep_idx=%d cycle=%d rollout_ok=%s steps=%d skipped=%d",
                ep_idx,
                cycle,
                rollout_ok,
                n_steps,
                skipped_eps,
            )
            time.sleep(2.0)
            continue

        online.add_episode(
            traj.get("proprio", traj["states"]),
            traj["actions_v"],
            traj["actions"],
            success=success,
            gamma=args.gamma,
            features_raw=traj.get("features"),
        )
        env_steps += n_steps
        valid_eps += 1
        successes += int(success)
        recent.append(float(success))

        if len(online) >= args.batch_size // 4 or (offline is not None and len(offline) > 0):
            for update_idx in range(args.updates_per_episode):
                batch = _mix_batch(
                    offline,
                    online,
                    args.batch_size,
                    online_frac=args.online_frac,
                    device=device,
                )
                last_q = critic_step(
                    model,
                    batch,
                    opt_q,
                    cql_coef=args.cql_coef * 0.5,
                    cql_n_actions=args.cql_n_actions,
                    cql_action_radius=args.cql_action_radius,
                    cql_margin=args.cql_margin,
                    cql_far_scale=args.cql_far_scale,
                )
                can_update_g = (
                    requested_g
                    and valid_eps >= args.g_start_episodes
                    and online.has_both_outcomes()
                    and critic_is_healthy(last_q, args.g_max_mc_loss)
                )
                if can_update_g and update_idx % args.policy_delay == 0:
                    last_g = refiner_step(
                        model,
                        batch,
                        opt_g,
                        opt_alpha,
                        target_divergence=args.target_divergence,
                    )

        log.info(
            "steps=%d/%d (%.2f%%) valid_eps=%d ep_idx=%d success=%s "
            "ep_steps=%d g=%s residual_rms=%.4f feature_age=%.2f dt=%.1fs sr=%.3f "
            "critic_loss=%.4f td=%.4f cql=%.4f q_mean=%.4f "
            "g_loss=%.4f g_adv=%.4f g_rms=%.4f alpha=%.4f",
            env_steps,
            args.target_env_steps,
            100.0 * env_steps / max(args.target_env_steps, 1),
            valid_eps,
            ep_idx,
            success,
            n_steps,
            policy.enable_g,
            float(traj.get("residual_rms", 0.0)),
            float(traj.get("feature_age_mean", 0.0)),
            time.time() - ep_t0,
            successes / max(valid_eps, 1),
            float(last_q.get("critic_loss", float("nan"))),
            float(last_q.get("td_loss", float("nan"))),
            float(last_q.get("cql_loss", float("nan"))),
            float(last_q.get("q_mean", float("nan"))),
            float(last_g.get("refiner_loss", float("nan"))),
            float(last_g.get("predicted_advantage", float("nan"))),
            float(last_g.get("residual_rms", float("nan"))),
            float(last_g.get("alpha", float("nan"))),
        )

        hit_step_log = env_steps - last_log_steps >= args.log_every_steps
        hit_ep_log = valid_eps % args.log_every_episodes == 0
        hit_episode_cap = args.max_valid_episodes > 0 and valid_eps >= args.max_valid_episodes
        if hit_step_log or hit_ep_log or hit_episode_cap or env_steps >= args.target_env_steps:
            last_log_steps = env_steps
            row: dict[str, Any] = {
                "env_steps": env_steps,
                "target_env_steps": args.target_env_steps,
                "valid_episodes": valid_eps,
                "skipped_episodes": skipped_eps,
                "successes": successes,
                "cumulative_success_rate": successes / max(valid_eps, 1),
                "window_success_rate": float(np.mean(recent)) if recent else 0.0,
                "online_transitions": len(online),
                "g_enabled": policy.enable_g,
                "residual_rms_last": float(traj.get("residual_rms", 0.0)),
                "feature_age_mean": float(traj.get("feature_age_mean", 0.0)),
                "elapsed_sec": time.time() - t0,
                "steps_per_sec": env_steps / max(time.time() - t0, 1e-6),
                "server_port": int(args.server_port),
                "ep_idx": ep_idx,
                **{f"q_{k}": v for k, v in last_q.items()},
                **{f"g_{k}": v for k, v in last_g.items()},
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            ckpt = out_dir / f"molmoact2_cf_steps{env_steps:010d}.pt"
            model.save(str(ckpt), meta=row)
            model.save(str(out_dir / "molmoact2_cf_latest.pt"), meta=row)
            if args.replay_out:
                online.save_npz(
                    Path(args.replay_out),
                    fit_norm_stats=args.fit_replay_norm_stats,
                )
            # Keep disk light: retain only latest + last checkpoint path in a pointer file.
            (out_dir / "LATEST_CKPT.txt").write_text(str(ckpt.name))
            log.info(
                "METRICS steps=%d sr=%.3f win_sr=%.3f adv=%.4f rms=%.4f sps=%.2f ckpt=%s",
                env_steps,
                row["cumulative_success_rate"],
                row["window_success_rate"],
                last_g.get("predicted_advantage", float("nan")),
                last_g.get("residual_rms", float("nan")),
                row["steps_per_sec"],
                ckpt.name,
            )
            # Delete older step ckpts except latest (NFS nearly full).
            for old in sorted(out_dir.glob("molmoact2_cf_steps*.pt"))[:-1]:
                try:
                    old.unlink()
                except OSError:
                    pass

    summary = {
        "env_steps": env_steps,
        "target_env_steps": args.target_env_steps,
        "valid_episodes": valid_eps,
        "skipped_episodes": skipped_eps,
        "cumulative_success_rate": successes / max(valid_eps, 1),
        "elapsed_sec": time.time() - t0,
        "metrics_path": str(metrics_path),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if args.replay_out and len(online) > 0:
        replay_info = online.save_npz(
            Path(args.replay_out),
            fit_norm_stats=args.fit_replay_norm_stats,
        )
        log.info("Saved replay %s: %s", args.replay_out, replay_info)
    log.info("Done: %s", json.dumps(summary))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cf_ckpt", type=str, default="")
    p.add_argument("--buffer", type=str, default=str(_HERE / "runs/pick_buffer.npz"))
    p.add_argument("--out_dir", type=str, default=str(_HERE / "runs/molmoact2_cf_100m/shard_0"))
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--benchmark_dir", type=str, default="")
    p.add_argument("--assets_dir", type=str, default=os.path.expanduser("~/.cache/molmospaces/assets"))
    p.add_argument("--tmp_rollout_dir", type=str, default="/tmp/molmoact2_cf_rollouts")
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--shard_size", type=int, default=125, help="Bench episodes to cycle in this shard")
    p.add_argument("--server_host", type=str, default="localhost")
    p.add_argument("--server_port", type=int, default=8000)
    p.add_argument("--server_wait_sec", type=float, default=1800.0)
    p.add_argument("--target_env_steps", type=int, default=12_500_000)
    p.add_argument(
        "--max_valid_episodes",
        type=int,
        default=0,
        help="Optional episode cap; 0 means stop only at target_env_steps",
    )
    p.add_argument("--log_every_steps", type=int, default=1_000_000)
    p.add_argument("--log_every_episodes", type=int, default=100)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--updates_per_episode", type=int, default=5)
    p.add_argument("--online_frac", type=float, default=0.5)
    p.add_argument("--online_capacity", type=int, default=200_000)
    p.add_argument("--disable_g", action="store_true")
    p.add_argument(
        "--force_g",
        action="store_true",
        help="Enable a frozen checkpoint G without critic gating (evaluation only)",
    )
    p.add_argument("--g_start_episodes", type=int, default=20)
    p.add_argument("--g_max_mc_loss", type=float, default=0.20)
    p.add_argument(
        "--g_min_advantage",
        type=float,
        default=0.005,
        help="Deploy G only when recent predicted advantage clears this bar",
    )
    p.add_argument("--policy_delay", type=int, default=2)
    p.add_argument(
        "--policy_chunk_size",
        type=int,
        default=0,
        help="Override MolmoAct2 chunk size; 0 keeps the policy default",
    )
    p.add_argument(
        "--explore_residual_std",
        type=float,
        default=0.02,
        help="Normalized residual exploration noise while learning (0 for eval)",
    )
    p.add_argument("--replay_out", type=str, default="")
    p.add_argument("--fit_replay_norm_stats", action="store_true")
    p.add_argument("--lr_q", type=float, default=3e-4)
    p.add_argument("--lr_g", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--cql_coef", type=float, default=0.1)
    p.add_argument("--cql_n_actions", type=int, default=8)
    p.add_argument("--cql_action_radius", type=float, default=0.05)
    p.add_argument("--cql_far_scale", type=float, default=1.0)
    p.add_argument("--cql_margin", type=float, default=0.0)
    p.add_argument("--target_divergence", type=float, default=2.5e-3)
    p.add_argument("--pos_frac", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.policy_delay < 1:
        p.error("--policy_delay must be >= 1")
    if args.force_g and args.updates_per_episode != 0:
        p.error("--force_g is evaluation-only; use --updates_per_episode 0")
    if args.fit_replay_norm_stats and (not args.disable_g or args.updates_per_episode != 0):
        p.error(
            "--fit_replay_norm_stats is collection-only; use --disable_g "
            "--updates_per_episode 0"
        )
    return args


if __name__ == "__main__":
    train_100m(parse_args())
