#!/usr/bin/env python
"""Collect the phase-1 token corpus for one scene.

    scripts/run_collect.py --scene house21 --gpu 0 --port 8500 --target 5000

Runs episodes from the scene's training half until `--target` token sequences have been
written, then stops. One sequence is one VLA call: (968, 2048) float16, about 4 MB, so
5000 sequences is roughly 20 GB per scene.

Fewer episodes are needed than it looks. At chunk_size 8 the VLA is called once every
8 env steps, so a 500-step failure contributes ~63 sequences and a quick success ~13.

What is collected covers the whole episode, before and after the gate: the encoder has to
represent the states the actor passes through on its way to the gate as well as the ones
it acts in. That is a different question from which transitions enter the actor's replay
buffer -- see `RLConfig.store_pre_gate`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import ROOT as DATA_ROOT  # noqa: E402
from pi05.config import EvalConfig, apply_overrides  # noqa: E402

DEFAULT_CORPUS = DATA_ROOT / "rlt_tokens_step_time"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scene", required=True)
    ap.add_argument("--target", type=int, default=5000, help="token sequences to collect")
    ap.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        dest="max_episodes",
        help="stop after this many trajectories, even if --target is not reached",
    )
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "pi05_runs/collect")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = EvalConfig(
        scene=args.scene,
        split="train",          # the half RL will learn on; its rate is not a result
        record_tokens=str(args.corpus / args.scene),
        episodes=1,             # episodes are driven one at a time by pi05.collect
        gpu=args.gpu,
        sim_gpu=args.gpu,
        egl_device=args.gpu,
        port=args.port,
        save_video=False,
        out_dir=args.out,
        tag=f"collect_{args.scene}",
    )
    apply_overrides(cfg, args.set)
    problem = cfg.validate()
    if problem:
        raise SystemExit(f"this EvalConfig cannot be run: {problem}")

    print(f"scene      {cfg.scene}")
    print(f"benchmark  {cfg.benchmark_dir()}")
    print(f"regime     chunk_size={cfg.chunk_size} conversion={cfg.conversion}")
    print(f"corpus     {cfg.record_tokens}")
    print(f"target     {args.target} sequences")
    print(f"max eps    {args.max_episodes if args.max_episodes is not None else 'none'}\n")

    from pi05.eval import prepare_environment

    prepare_environment(cfg)

    from pi05.collect import run

    run(cfg, args.target, max_episodes=args.max_episodes)


if __name__ == "__main__":
    main()
