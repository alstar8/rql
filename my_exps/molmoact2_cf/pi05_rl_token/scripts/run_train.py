#!/usr/bin/env python
"""Train RL Token on top of the frozen pi0.5, on one scene.

    scripts/run_train.py --scene house21 --encoder house21
    scripts/run_train.py --scene house10 --encoder combined --gpu 3 --port 8603
    scripts/run_train.py --show

The settings live in `pi05/config.py:RLConfig`, not here. What the training loop actually
does -- what the RL state is, when the actor takes over, which actions of the chunk run,
how a transition closes -- is documented at the top of `pi05/train.py`.

Run it with the simulator's interpreter, detached, because SSH drops under load:

    cd <rl_token>
    setsid nohup env PYTHONUNBUFFERED=1 \
        /home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python \
        scripts/run_train.py --scene house21 --encoder house21 --gpu 0 --port 8600 \
        > /home/jovyan/users/staroverov/pi05_molmospaces/pi05_runs/rl/house21.log 2>&1 &

One frozen-VLA server per run, one GPU each: pi0.5 inference blocks the server's event
loop, so two runs sharing a server serialise and then time out. The server is started and
stopped by the trainer; it does not outlive the run.
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


def build_config(argv=None) -> tuple[RLConfig, bool]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", help="house2, house10 or house21")
    parser.add_argument("--encoder", help="phase-1 autoencoder: house2/house10/house21/combined")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--warmup-episodes", type=int, dest="warmup_episodes")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--port", type=int)
    parser.add_argument("--tag", help="name of the output subdirectory")
    parser.add_argument("--init-actor", dest="init_actor", help="pretrained agent.pt")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override any RLConfig field; repeatable",
    )
    parser.add_argument("--show", action="store_true", help="print the config and exit")
    args = parser.parse_args(argv)

    cfg = RLConfig()
    for name in ("scene", "encoder", "episodes", "warmup_episodes", "gpu", "port", "tag", "init_actor"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    apply_overrides(cfg, args.set)
    if args.gpu is not None:
        cfg.egl_device = cfg.gpu
    return cfg, args.show


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s"
    )
    cfg, show_only = build_config()

    problem = cfg.validate()
    if problem:
        raise SystemExit(f"this RLConfig cannot be run: {problem}")

    print(describe())
    print("\nthis run:")
    for field in dataclasses.fields(cfg):
        print(f"  {field.name:20s} {getattr(cfg, field.name)}")
    print(f"  {'benchmark':20s} {cfg.benchmark_dir()}")
    # gate_step=0 is RL from the first env step; -1 resolves to the scene catalog.
    try:
        print(f"  {'gate (resolved)':20s} {cfg.resolved_gate_step()}")
    except ValueError:
        pass
    print(f"  {'encoder file':20s} {cfg.token_ae_path()}")
    print(f"  {'stride':20s} {cfg.resolved_stride()}")
    print(f"  {'output':20s} {cfg.run_dir()}\n")
    if show_only:
        return

    from pi05.train import prepare_environment

    prepare_environment(cfg)

    from pi05.train import train

    train(cfg)


if __name__ == "__main__":
    main()
