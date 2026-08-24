#!/usr/bin/env python
"""Second-stage filter: re-measure shortlisted candidates, with video, on several GPUs.

    scripts/run_probe.py --band zero --repeats 6 --gpus 0,1,2,3

The coarse profile runs four rollouts per candidate, which is enough to rank and not
enough to conclude. Measured: candidate episode 4 came back 0/4 there and 2/12 = 16.7%
here. A scene with a true rate of 17% shows 0/4 about 47% of the time, so a coarse zero
is a shortlist entry, never a finding.

This stage answers the two questions the coarse pass cannot:

    is it really zero?        more rollouts, with a Wilson interval that says how sure
    why does it fail?         video, because the band "~0%" is only useful when the
                              failure is the grasp itself. If the policy goes for the
                              wrong object -- measured on episode 4, where it parks over
                              a mug and ignores the tissue for 500 steps -- then a
                              chunk-level correction cannot fix it and the scene is not
                              an RL target.

Each candidate gets its own jittered benchmark, so what is measured here is the scene RL
would actually be pointed at, not the single unjittered episode.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import ROOT as DATA_ROOT  # noqa: E402
from pi05.config import VENV_SIM  # noqa: E402

log = logging.getLogger("run_probe")

RANKING = DATA_ROOT / "pi05_runs/scene_profile/ranking.json"
DEFAULT_OUT = DATA_ROOT / "pi05_runs/probe"


def candidates(band: str, limit: int) -> list[int]:
    rows = json.loads(RANKING.read_text())
    picked = [r["episode"] for r in rows if r.get("band") == band]
    return picked[:limit] if limit else picked


def build_benchmark(episode: int, repeats: int, out: Path) -> Path:
    path = out / f"bench_ep{episode}"
    if (path / "benchmark.json").exists():
        return path
    subprocess.run(
        [
            str(VENV_SIM), str(ROOT / "scripts/make_repeat_benchmark.py"),
            "--episode-index", str(episode),
            "--repeats", str(repeats),
            "--jitter", "0.02",
            "--seed", "0",
            "--out", str(path),
        ],
        check=True, capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    return path


def shard_script(episodes: list[int], gpu: int, args) -> str:
    """One shell script per GPU: its candidates, one after another."""
    lines = ["#!/bin/bash", "set -u", f"cd {ROOT}"]
    for episode in episodes:
        bench = args.out / f"bench_ep{episode}"
        lines.append(
            f'env PYTHONUNBUFFERED=1 {VENV_SIM} scripts/run_eval.py '
            f'--scene val --episodes {args.repeats} --gpu {gpu} --port {8480 + gpu} '
            f'--set benchmark_override={bench} '
            f'--set out_dir={args.out} --tag ep{episode} '
            f'>> {args.out}/ep{episode}.log 2>&1'
        )
        lines.append(f'echo "ep {episode} done at $(date -u +%H:%M:%S)" >> {args.out}/gpu{gpu}.log')
    lines.append(f'echo "GPU {gpu} PROBES DONE" >> {args.out}/gpu{gpu}.log')
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--band", default="zero", help="zero, low or good")
    ap.add_argument("--episodes", default="", help="explicit list, overrides --band")
    ap.add_argument("--limit", type=int, default=0, help="0 = every candidate in the band")
    ap.add_argument("--repeats", type=int, default=6, help="jittered repeats per candidate")
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.episodes:
        picked = [int(x) for x in args.episodes.split(",")]
    else:
        picked = candidates(args.band, args.limit)
    if not picked:
        raise SystemExit(f"no candidates in band {args.band!r}")

    gpus = [int(g) for g in args.gpus.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    log.info("%d candidates over %d GPUs, %d repeats each", len(picked), len(gpus), args.repeats)
    for episode in picked:
        build_benchmark(episode, args.repeats, args.out)
    log.info("benchmarks built")

    for offset, gpu in enumerate(gpus):
        mine = picked[offset :: len(gpus)]
        if not mine:
            continue
        script = args.out / f"gpu{gpu}.sh"
        script.write_text(shard_script(mine, gpu, args))
        script.chmod(0o755)
        log.info("GPU %d: %s", gpu, mine)
        if args.dry_run:
            continue
        subprocess.Popen(
            ["setsid", "nohup", "bash", str(script)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    if not args.dry_run:
        log.info("launched; watch %s/gpu*.log", args.out)


if __name__ == "__main__":
    main()
