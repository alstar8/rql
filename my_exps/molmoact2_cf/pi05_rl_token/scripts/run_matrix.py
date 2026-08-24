#!/usr/bin/env python
"""Phases 2-3: the RL-Token matrix, every scene crossed with every encoder.

    scripts/run_matrix.py --encoders <dir> --gpus 8

The question the ablation answers: does an encoder trained on this scene beat one trained
elsewhere, or the combined one? That only has an answer per task, so each cell is its own
run -- 3 scenes x 4 encoders = 12.

Each cell needs a frozen VLA of its own. pi0.5 inference blocks the server's event loop,
so two cells sharing a server serialise and then time out during the handshake, and
episodes are skipped silently. One server per cell, one cell per GPU, which means the 12
cells run in waves of `--gpus`.

Every cell inherits `chunk_size`, `conversion` and `rl_action_space` from config.py.
RL drives from env step 0 by default. Pass `--gate SCENE=N` (or `--default-gate N`)
to let the frozen VLA run the first N steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import ROOT as DATA_ROOT  # noqa: E402
from pi05.config import SELECTED_SCENES, VENV_SIM  # noqa: E402

log = logging.getLogger("run_matrix")

#: Per-scene VLA prefix length. Empty = RL from step 0 (the default). Fill via --gate.
GATE_STEPS: dict[str, int] = {}


def gate_for(scene: str, override: int = 0) -> int:
    """0 = RL from the first env step. N = frozen-VLA prefix of N steps."""
    if scene in GATE_STEPS:
        return GATE_STEPS[scene]
    if override:
        return override
    return 0


def cells(scenes: list[str], encoders: list[str]) -> list[tuple[str, str]]:
    return [(scene, encoder) for scene in scenes for encoder in encoders]


def launch(scene: str, encoder: str, gpu: int, port: int, args) -> subprocess.Popen:
    import os

    gate = gate_for(scene, args.default_gate)
    command = [
        str(VENV_SIM), str(ROOT / "scripts/run_train.py"),
        "--scene", scene,
        "--encoder", encoder,
        "--gpu", str(gpu),
        "--port", str(port),
        "--episodes", str(args.episodes),
        "--warmup-episodes", str(args.warmup),
        "--set", f"gate_step={gate}",
        "--set", f"out_dir={args.out}",
        "--tag", f"{scene}_ae-{encoder}",
    ]
    log_path = args.out / f"{scene}_ae-{encoder}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    log.info("cell %s x %s -> GPU %d port %d (gate %d)", scene, encoder, gpu, port, gate)
    return subprocess.Popen(
        command, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
        env=env, cwd=str(ROOT), start_new_session=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scenes", default="", help="default: every scene in config.SCENES")
    ap.add_argument("--encoders", default="", help="default: the scenes plus 'combined'")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument(
        "--first-gpu",
        type=int,
        default=0,
        dest="first_gpu",
        help="lowest CUDA index to use, for leaving a card to something already running",
    )
    ap.add_argument("--first-port", type=int, default=8600, dest="first_port")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument(
        "--default-gate",
        type=int,
        default=0,
        dest="default_gate",
        help="VLA prefix length for every scene; 0 (default) means RL from step 0",
    )
    ap.add_argument(
        "--gate",
        action="append",
        default=[],
        metavar="SCENE=STEP",
        help="per-scene gate step, repeatable",
    )
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "pi05_runs/rlt_matrix")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    for pair in args.gate:
        scene, _, step = pair.partition("=")
        GATE_STEPS[scene.strip()] = int(step)

    # Default to the measured selection, not every scene in the file: SCENES also holds
    # the superseded ones and a rejected wrong-object example.
    scenes = args.scenes.split(",") if args.scenes else list(SELECTED_SCENES)
    encoders = args.encoders.split(",") if args.encoders else [*scenes, "combined"]
    grid = cells(scenes, encoders)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "matrix.json").write_text(json.dumps({
        "scenes": scenes,
        "encoders": encoders,
        "cells": len(grid),
        "episodes": args.episodes,
        "warmup": args.warmup,
        "gate_steps": {s: gate_for(s, args.default_gate) for s in scenes},
    }, indent=2))

    print(f"{len(grid)} cells, {args.gpus} GPUs -> {-(-len(grid) // args.gpus)} waves")
    for scene, encoder in grid:
        print(f"  {scene:14s} x ae_{encoder:14s} gate={gate_for(scene, args.default_gate)}")
    if args.dry_run:
        return

    started = time.time()
    for wave_index in range(0, len(grid), args.gpus):
        wave = grid[wave_index : wave_index + args.gpus]
        log.info("wave %d: %d cells", wave_index // args.gpus + 1, len(wave))
        jobs = [
            launch(scene, encoder, args.first_gpu + offset,
                   args.first_port + args.first_gpu + offset, args)
            for offset, (scene, encoder) in enumerate(wave)
        ]
        for (scene, encoder), job in zip(wave, jobs):
            code = job.wait()
            log.info("cell %s x %s exited %d", scene, encoder, code)
        log.info("wave done after %.1f h", (time.time() - started) / 3600)

    log.info("matrix finished in %.1f h", (time.time() - started) / 3600)


if __name__ == "__main__":
    main()
