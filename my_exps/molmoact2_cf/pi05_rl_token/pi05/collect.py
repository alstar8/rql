"""Collect the phase-1 token corpus for one scene, in the regime that will be run.

The encoder is the RL state, so it has to be fit to the token sequences the frozen policy
actually produces. That makes the corpus a property of the execution regime, not of the
checkpoint alone: one collected under a different chunk size or conversion describes a
policy we do not run. Everything here therefore goes through the same `EvalConfig` that
`run_eval.py` uses, and the resolved config is written next to the shards.

Tokens are recorded for the WHOLE episode, gate or no gate. That is deliberate and is not
the same choice as `RLConfig.store_pre_gate`, which governs the actor's replay buffer:

    encoder     every step -- it has to represent states the actor will pass through
                on its way to the gate as well as after it
    actor       only the steps it could have influenced

One server stays up for the whole collection. Restarting it per pass would reset the
shard counter and reload 14 GB of checkpoint for nothing; keeping it up also means every
sequence in the corpus came from one process, with one manifest describing it.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

import requests

from .config import EvalConfig

log = logging.getLogger("pi05.collect")


def recorded(cfg: EvalConfig) -> int:
    """How many sequences the server has written so far."""
    try:
        response = requests.get(f"http://127.0.0.1:{cfg.port}/act", timeout=10)
        return int(response.json().get("tokens_recorded", 0))
    except Exception:  # noqa: BLE001
        return -1


def benchmark_size(cfg: EvalConfig) -> int:
    return len(json.loads((cfg.benchmark_dir() / "benchmark.json").read_text()))


def collect(
    cfg: EvalConfig,
    target: int,
    max_passes: int = 12,
    max_episodes: int | None = None,
) -> dict:
    """Run episodes until `target` token sequences exist, or the passes run out.

    `max_episodes` caps trajectories (rollouts), which is the unit the pretrain
    recipe cares about. Sequence count is a side-effect of horizon and success.
    """
    from molmo_spaces.evaluation.eval_main import run_evaluation

    from .policy import Pi05EvalConfig

    episodes = benchmark_size(cfg)
    scratch = cfg.run_dir() / "scratch"
    summary = {
        "scene": cfg.scene,
        "target_sequences": target,
        "benchmark_dir": str(cfg.benchmark_dir()),
        "chunk_size": cfg.chunk_size,
        "conversion": cfg.conversion,
        "episodes_run": 0,
        "successes": 0,
        "max_episodes": max_episodes,
    }
    started = time.time()

    unreachable = 0
    for pass_index in range(max_passes):
        for episode in range(episodes):
            have = recorded(cfg)
            if have < 0:
                # The server is the only thing that knows the count, and it is what
                # writes the corpus. If it has died, every remaining pass would run
                # episodes and record nothing, for hours. Stop instead.
                unreachable += 1
                if unreachable >= 3:
                    raise RuntimeError(
                        f"the pi0.5 server on port {cfg.port} stopped answering after "
                        f"{summary['episodes_run']} episodes; the corpus is incomplete. "
                        f"See {cfg.run_dir() / 'server.log'}"
                    )
                log.warning("server did not answer (%d/3)", unreachable)
            else:
                unreachable = 0
            if have >= target:
                log.info("target reached: %d sequences", have)
                summary["sequences"] = have
                summary["seconds"] = round(time.time() - started, 1)
                shutil.rmtree(scratch, ignore_errors=True)
                return summary
            if max_episodes is not None and summary["episodes_run"] >= max_episodes:
                log.info("episode cap reached: %d trajectories", summary["episodes_run"])
                summary["sequences"] = have if have >= 0 else recorded(cfg)
                summary["seconds"] = round(time.time() - started, 1)
                shutil.rmtree(scratch, ignore_errors=True)
                return summary

            episode_dir = scratch / f"p{pass_index}_ep{episode:04d}"
            shutil.rmtree(episode_dir, ignore_errors=True)
            try:
                results = run_evaluation(
                    eval_config_cls=Pi05EvalConfig,
                    benchmark_dir=cfg.benchmark_dir(),
                    task_horizon_steps=cfg.horizon,
                    num_workers=1,
                    use_wandb=False,
                    episode_idx=episode,
                    output_dir=episode_dir,
                )
                if results.total_count > 0:
                    summary["episodes_run"] += 1
                    summary["successes"] += int(results.success_count > 0)
            except Exception as error:  # noqa: BLE001
                log.warning("pass %d episode %d failed: %s", pass_index, episode, error)
            finally:
                # The corpus is the product here; the footage would be tens of GB.
                shutil.rmtree(episode_dir, ignore_errors=True)

            if max_episodes is not None and summary["episodes_run"] >= max_episodes:
                have = recorded(cfg)
                log.info("episode cap reached: %d trajectories, %d sequences", summary["episodes_run"], have)
                summary["sequences"] = have
                summary["seconds"] = round(time.time() - started, 1)
                shutil.rmtree(scratch, ignore_errors=True)
                return summary

            if episode % 8 == 0:
                done = summary["episodes_run"]
                have = recorded(cfg)
                rate = have / max(done, 1)
                if max_episodes is not None:
                    left = max_episodes - done
                else:
                    left = (target - have) / max(rate, 0.1)
                log.info(
                    "pass %d ep %d: %d sequences from %d episodes (%.1f/ep), ~%.0f episodes left",
                    pass_index, episode, have, done, rate, left,
                )

    summary["sequences"] = recorded(cfg)
    summary["seconds"] = round(time.time() - started, 1)
    summary["warning"] = f"ran {max_passes} passes without reaching {target}"
    shutil.rmtree(scratch, ignore_errors=True)
    return summary


def run(cfg: EvalConfig, target: int, max_episodes: int | None = None) -> dict:
    """Bring up a recording server, collect, and always shut it down."""
    from .eval import start_server, stop_server

    if not cfg.record_tokens:
        raise ValueError("cfg.record_tokens must name a directory for the corpus")

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    server = start_server(cfg, run_dir / "server.log")
    try:
        # The checkpoint takes minutes to load; nothing can be collected before that.
        from .client import Pi05Client

        Pi05Client(port=cfg.port).wait_until_ready()
        summary = collect(cfg, target, max_episodes=max_episodes)
    finally:
        stop_server(server)
        # The server flushes its partial shard on SIGTERM; give it a moment to land.
        time.sleep(5)

    summary["shards"] = len(list(Path(cfg.record_tokens).glob("*.npz")))
    (run_dir / "collect.json").write_text(json.dumps(summary, indent=2))
    log.info("%s", json.dumps(summary, indent=2))
    return summary
