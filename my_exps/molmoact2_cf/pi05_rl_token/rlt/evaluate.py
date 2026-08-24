"""Measure a policy: either one episode many times, or the held-out benchmark.

    # baseline of the frozen VLA on a single training episode, 50 rollouts
    python -m rlt.evaluate --episode_idx 137 --repeats 50

    # the same episode with a trained actor
    python -m rlt.evaluate --episode_idx 137 --repeats 50 --actor runs/online/agent.pt

    # the held-out test set (episodes 0-127), 4 workers
    python -m rlt.evaluate --max_episodes 128 --num_workers 4

Rollouts of one fixed episode still differ: pi_vla is stochastic (std 0.044).
The actor is evaluated at its mean -- no exploration noise.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from .cli import describe, parse_into
from .config import CHUNK, EvalConfig
from .mlspaces_env import configure_assets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rlt.evaluate")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a rate; n is small here, so no normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def build_policy(cfg: EvalConfig):
    """Frozen VLA, plus the trained actor when --actor is given."""
    from .eval_config import RLTokenEvalConfig
    from .rl_policy import RLTokenPolicy
    from .vla import TokenEncoder, VlaClient

    vla = VlaClient(cfg.server_host, cfg.server_port, cfg.server_timeout_sec)
    log.info("VLA server: %s", vla.wait_until_ready())
    encoder = TokenEncoder(cfg.token_ae, cfg.device)

    # The actor, if any, comes from the policy config -- the same path a worker
    # process would take, so both are exercised by every run.
    policy = RLTokenPolicy(
        RLTokenEvalConfig(),
        vla=vla,
        encoder=encoder,
        chunk=cfg.chunk,
        stride=cfg.chunk,  # evaluation stores nothing, so one decision per plan is enough
        explore=False,  # deterministic actor mean
    )
    log.info("policy: %s", "VLA + actor" if policy.agent is not None else "frozen VLA only")
    return policy


def repeat_one_episode(cfg: EvalConfig) -> dict:
    """The milestone-1 measurement: one episode, --repeats rollouts."""
    from .rollout import EpisodeRunner

    policy = build_policy(cfg)
    runner = EpisodeRunner(
        policy,
        cfg.benchmark_dir,
        cfg.horizon,
        cfg.tmp_rollout_dir,
        video_dir=cfg.video_dir,
        camera=cfg.camera,
    )

    outcomes: list[bool] = []
    steps: list[int] = []
    executed: list[np.ndarray] = []
    start = time.time()
    for i in range(cfg.repeats):
        rollout = runner.run(cfg.episode_idx)
        if rollout is None:
            log.warning("rollout %d produced nothing; skipping", i)
            continue
        outcomes.append(rollout.success)
        steps.append(rollout.steps)
        if cfg.save_actions:
            # What the arm was actually commanded, step by step: the only way to
            # measure how smooth the motion was without re-deriving it from
            # overlapping replay windows.
            executed.append(np.asarray(rollout.committed[: rollout.steps], dtype=np.float32))
        k, n = sum(outcomes), len(outcomes)
        log.info("rollout %d/%d success=%d steps=%d  sr=%d/%d=%.1f%%",
                 i + 1, cfg.repeats, int(rollout.success), rollout.steps, k, n, 100.0 * k / n)

    if executed:
        out_dir = Path(cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"actions_{cfg.episode_idx}{cfg.tag}.npz"
        np.savez(
            path,
            actions=np.concatenate(executed, axis=0),
            lengths=np.array([len(a) for a in executed]),
            successes=np.array([int(o) for o in outcomes]),
            chunk=cfg.chunk,
        )
        log.info("wrote %s", path)

    k, n = sum(outcomes), len(outcomes)
    lo, hi = wilson(k, n)
    return {
        "mode": "single_episode",
        "chunk": cfg.chunk,
        "episode_idx": cfg.episode_idx,
        "actor": cfg.actor,
        "rollouts": n,
        "successes": k,
        "success_rate": k / max(n, 1),
        "ci95": [lo, hi],
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "successes_mask": [int(o) for o in outcomes],
        # Per rollout, because mean steps mixes two things: failures always run
        # the full horizon, so a policy that merely succeeds more often looks
        # faster. Time-to-success can only be compared among successes.
        "steps_per_rollout": [int(s) for s in steps],
        "seconds": time.time() - start,
    }


def run_benchmark(cfg: EvalConfig) -> dict:
    """The held-out test set. Workers build their own policy from the config."""
    from molmo_spaces.evaluation.eval_main import run_evaluation

    from .eval_config import RLTokenEvalConfig, default_benchmark_dir

    bench = Path(cfg.benchmark_dir) if cfg.benchmark_dir else default_benchmark_dir()
    preloaded = build_policy(cfg) if cfg.num_workers == 1 else None

    start = time.time()
    results = run_evaluation(
        eval_config_cls=RLTokenEvalConfig,
        benchmark_dir=bench,
        task_horizon_steps=cfg.horizon,
        num_workers=cfg.num_workers,
        use_wandb=False,
        preloaded_policy=preloaded,
        max_episodes=cfg.max_episodes,
        output_dir=Path(cfg.out_dir),
    )
    k, n = results.success_count, results.total_count
    lo, hi = wilson(k, n)
    return {
        "mode": "benchmark",
        "actor": cfg.actor,
        "rollouts": n,
        "successes": k,
        "success_rate": k / max(n, 1),
        "ci95": [lo, hi],
        "output_dir": str(results.output_dir),
        "seconds": time.time() - start,
    }


def main() -> None:
    cfg = parse_into(EvalConfig)
    print(describe(cfg))
    configure_assets(cfg.assets_dir, cfg.cache_dir)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    from .eval_config import publish_policy_env

    publish_policy_env(cfg.token_ae, cfg.actor, cfg.server_host, cfg.server_port, cfg.device)
    summary = repeat_one_episode(cfg) if cfg.episode_idx >= 0 else run_benchmark(cfg)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"episode_{cfg.episode_idx}{cfg.tag}.json" if cfg.episode_idx >= 0 else f"benchmark{cfg.tag}.json"
    (out_dir / name).write_text(json.dumps(summary, indent=2))
    log.info("%s", json.dumps({k: v for k, v in summary.items() if k != "successes_mask"}, indent=2))
    log.info("wrote %s", out_dir / name)


if __name__ == "__main__":
    main()
