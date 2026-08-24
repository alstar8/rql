#!/usr/bin/env python
"""Evaluate pi0.5, or an RL actor on top of it, on one scene.

    scripts/run_eval.py                              # the defaults in pi05/config.py
    scripts/run_eval.py --scene house10
    scripts/run_eval.py --scene house10 --set chunk_size=8
    scripts/run_eval.py --show                       # resolve and print, run nothing

The settings live in `pi05/config.py:EvalConfig`, not here. `--set` is for a one-off
sweep; a value that should hold for the project belongs in the file, where the repository
records it. `--show` prints exactly what would run, including the resolved paths.

Run it with the simulator's interpreter:

    cd <rl_token>
    /home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python \
        scripts/run_eval.py --scene house21

It starts the frozen VLA on its own GPU, runs the benchmark, prints the success rate with
a 95% Wilson interval, and stops the server again -- including when the run fails.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import EvalConfig, apply_overrides, describe  # noqa: E402


def build_config(argv=None) -> tuple[EvalConfig, bool]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", help="scene name, or 'val' for the shipped benchmark")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--gpu", type=int, help="CUDA device for the policy server")
    parser.add_argument("--sim-gpu", type=int, dest="sim_gpu")
    parser.add_argument("--port", type=int)
    parser.add_argument("--actor", help="agent.pt from run_train.py; empty = plain VLA")
    parser.add_argument("--token-ae", dest="token_ae", help="the encoder that actor used")
    parser.add_argument("--tag", help="name of the output subdirectory")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override any EvalConfig field; repeatable",
    )
    parser.add_argument("--show", action="store_true", help="print the config and exit")
    args = parser.parse_args(argv)

    cfg = EvalConfig()
    for name in ("episodes", "gpu", "sim_gpu", "port", "actor", "token_ae", "tag"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.scene is not None:
        cfg.scene = "" if args.scene == "val" else args.scene
    apply_overrides(cfg, args.set)

    # The simulator and the policy server share a machine but not a card unless asked.
    if args.sim_gpu is None and args.gpu is not None:
        cfg.sim_gpu = cfg.gpu
        cfg.egl_device = cfg.gpu
    return cfg, args.show


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, show_only = build_config()

    problem = cfg.validate()
    if problem:
        raise SystemExit(f"this EvalConfig cannot be run: {problem}")

    print(describe())
    print("\nthis run:")
    for field in dataclasses.fields(cfg):
        print(f"  {field.name:18s} {getattr(cfg, field.name)}")
    print(f"  {'benchmark':18s} {cfg.benchmark_dir()}")
    # gate_step=0 is RL from the first env step; -1 resolves to the scene catalog.
    try:
        print(f"  {'gate (resolved)':18s} {cfg.resolved_gate_step()}")
    except ValueError:
        pass
    print(f"  {'output':18s} {cfg.run_dir()}\n")
    if show_only:
        return

    # Before importing anything that pulls in molmo_spaces: MuJoCo picks its EGL device
    # and MolmoSpaces resolves its asset paths at import time.
    from pi05.eval import prepare_environment

    prepare_environment(cfg)

    from pi05.eval import evaluate

    evaluate(cfg)


if __name__ == "__main__":
    main()
