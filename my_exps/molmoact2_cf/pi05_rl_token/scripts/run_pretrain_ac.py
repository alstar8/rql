#!/usr/bin/env python
"""Pretrain the RL Token actor-critic on frozen-VLA rollouts of the train bench.

    scripts/run_pretrain_ac.py --scene desk_mug --encoder desk_mug \
        --episodes 100 --offline-steps 8000 --gpu 0 --port 8512

Fills the replay buffer with VLA-only transitions (actor off), then behaviour-clones
the actor onto a_ref and TD-trains the critic. Online training can load agent.pt with
warmup_episodes=0 so the actor is in control from the first episode.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import RLConfig, apply_overrides, describe  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene")
    parser.add_argument("--encoder")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--offline-steps", type=int, default=8000, dest="offline_steps")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--tag")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = parser.parse_args()

    cfg = RLConfig()
    for name in ("scene", "encoder", "gpu", "port", "tag"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    apply_overrides(cfg, args.set)
    if args.gpu is not None:
        cfg.egl_device = cfg.gpu
    # Saved next to agent.pt so the dump matches what this process actually did.
    cfg.episodes = args.episodes
    cfg.warmup_episodes = 0

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s"
    )
    problem = cfg.validate()
    if problem:
        raise SystemExit(f"this RLConfig cannot be run: {problem}")

    print(describe())
    print("\npretrain:")
    for field in dataclasses.fields(cfg):
        print(f"  {field.name:20s} {getattr(cfg, field.name)}")
    print(f"  {'collect episodes':20s} {args.episodes}")
    print(f"  {'offline steps':20s} {args.offline_steps}")
    print(f"  {'output':20s} {cfg.run_dir()}\n")

    from pi05.train import prepare_environment, pretrain_actor_critic

    prepare_environment(cfg)
    pretrain_actor_critic(cfg, args.episodes, args.offline_steps)


if __name__ == "__main__":
    main()
