#!/usr/bin/env python
"""Rank candidate scenes by the frozen VLA's success rate, in the current regime.

One shard per GPU:

    scripts/run_profile.py --episodes 0-127 --repeats 4 --shard 0 --num-shards 8 \
        --gpu 0 --port 8400

Then read the merged ranking:

    scripts/run_profile.py --summarize

Why this exists: the three scenes in `config.SCENES` were inherited, and the run that
selected them is recorded as contaminated -- several candidates were written into the same
house directory and their episodes counted together -- with no artefacts left to recheck.
Their success rates were also measured under chunk 1 + plan_time, a regime the project no
longer runs. So the selection is redone by measurement.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import ROOT as DATA_ROOT  # noqa: E402
from pi05.config import EvalConfig, apply_overrides  # noqa: E402

DEFAULT_OUT = DATA_ROOT / "pi05_runs/scene_profile"


def summarize(out_dir: Path) -> None:
    from pi05.profile import collect

    rows = collect(sorted(out_dir.glob("shard*/profile.json")))
    if not rows:
        raise SystemExit(f"no profile.json under {out_dir}")

    print(f"{len(rows)} candidates, coarse rates -- for bucketing only, never to quote\n")
    print(f"{'episode':>8} {'successes':>10} {'rate':>7}  band")
    for row in rows:
        rate = "-" if row["rate"] is None else f"{100 * row['rate']:5.1f}%"
        print(f"{row['episode']:>8} {row['successes']:>4}/{row['trials']:<5} {rate:>7}  {row['band']}")

    print("\nper band:")
    for band in ("good", "low", "zero"):
        members = [r for r in rows if r["band"] == band]
        print(f"  {band:5s} {len(members):3d}  {[r['episode'] for r in members][:16]}")

    merged = out_dir / "ranking.json"
    merged.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {merged}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episodes", default="0-127", help="candidate val episodes")
    ap.add_argument("--repeats", type=int, default=4, help="rollouts per candidate")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1, dest="num_shards")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8400)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument("--summarize", action="store_true", help="merge shards and print")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.summarize:
        summarize(args.out)
        return

    from pi05.profile import parse_episode_spec, shard_of

    episodes = shard_of(parse_episode_spec(args.episodes), args.shard, args.num_shards)
    if not episodes:
        raise SystemExit(f"shard {args.shard} of {args.num_shards} has no episodes")

    # scene="" is the shipped val benchmark, which is where candidates come from.
    cfg = EvalConfig(
        scene="",
        episodes=1,
        gpu=args.gpu,
        sim_gpu=args.gpu,
        egl_device=args.gpu,
        port=args.port,
        save_video=False,
        out_dir=args.out,
        tag=f"shard{args.shard}",
    )
    apply_overrides(cfg, args.set)
    problem = cfg.validate()
    if problem:
        raise SystemExit(f"this EvalConfig cannot be run: {problem}")

    print(f"shard {args.shard}/{args.num_shards}: {len(episodes)} candidates x {args.repeats}")
    print(f"regime: chunk_size={cfg.chunk_size} conversion={cfg.conversion}")
    print(f"output: {cfg.run_dir()}\n")

    from pi05.eval import prepare_environment

    prepare_environment(cfg)

    from pi05.profile import run_shard

    run_shard(cfg, episodes, args.repeats)


if __name__ == "__main__":
    main()
