#!/usr/bin/env python
"""One rollout worker for pi05.train_parallel. See pi05/collectors.py."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.collectors import (
    COUNTER,
    EGL,
    INBOX,
    WEIGHTS,
    EglSlots,
    FileCounter,
    load_live_weights,
    write_inbox,
)
from pi05.config import load_run_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] collector-%(message)s"
    )
    log = logging.getLogger("pi05.collector")

    cfg = load_run_config(args.config)
    os.environ["RLT_COLLECTOR"] = "1"
    os.environ["RLT_COLLECTOR"] = "1"
    from pi05.train import build, prepare_environment

    prepare_environment(cfg)
    out_dir = args.out
    _, agent, _, learner, policy, runner = build(cfg, out_dir, phase="online", with_runner=True)
    learner.updates = False
    policy.use_actor = True

    from rlt.cli import parse_episode_spec
    from rlt.replay import close_transitions

    pool = parse_episode_spec(cfg.episode_pool) or [0]
    counter = FileCounter(out_dir / COUNTER)
    slots = EglSlots(out_dir / EGL, max(1, int(cfg.egl_slots)))
    weights = out_dir / WEIGHTS
    inbox = out_dir / INBOX
    seen_version = -1
    probe_n = int(cfg.probe_episodes)
    total = probe_n + int(cfg.episodes)
    log.info("rank=%d probe=%d online=%d", args.rank, probe_n, cfg.episodes)

    try:
        while True:
            episode = counter.claim()
            if episode >= total:
                break
            probing = episode < probe_n
            seen_version = load_live_weights(agent, weights, seen_version)
            learner.start_episode()
            handle = slots.acquire()
            started = time.time()
            try:
                rollout = runner.run(pool[episode % len(pool)])
            finally:
                slots.release(handle)
            if rollout is None:
                log.warning("rank %d episode %d produced nothing", args.rank, episode)
                continue
            rows = [] if probing else close_transitions(rollout, cfg.chunk_size, cfg.gamma, 0)[0]
            write_inbox(
                inbox,
                episode,
                args.rank,
                {
                    "episode": episode,
                    "rank": args.rank,
                    "probe": probing,
                    "success": float(rollout.success),
                    "steps": int(rollout.steps),
                    "rows": rows,
                    "sec": time.time() - started,
                },
            )
            log.info(
                "rank %d ep %d %s success=%d steps=%d %.0fs",
                args.rank,
                episode,
                "probe" if probing else "actor",
                int(rollout.success),
                rollout.steps,
                time.time() - started,
            )
    finally:
        slots.close()


if __name__ == "__main__":
    main()
