#!/usr/bin/env python
"""Build a benchmark from a chosen set of val episodes, for multi-task RL.

    scripts/make_subset_benchmark.py --count 20 --seed 20260821 --out <dir>

Every scene so far is ONE episode repeated with jitter, so RL learns one task. This makes
the other kind: many different episodes -- different houses, different objects -- in one
benchmark, so a single actor can be trained across all of them and asked whether sharing
helps.

The draw is seeded and the chosen indices are written next to the benchmark, because "20
random episodes" is not reproducible unless the draw is recorded.

Each episode keeps its own house and object; there is no jitter. Variety comes from the
tasks differing, which is the point.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import ROOT as DATA_ROOT  # noqa: E402
from pi05.config import VAL_BENCHMARK  # noqa: E402


def profiled_rates() -> dict[int, dict]:
    """What the coarse scene profile measured for each candidate, if it ran."""
    path = DATA_ROOT / "pi05_runs/scene_profile/ranking.json"
    if not path.exists():
        return {}
    return {int(r["episode"]): r for r in json.loads(path.read_text())}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260821)
    ap.add_argument("--pool", default="0-127", help="candidates to draw from")
    ap.add_argument("--episodes", default="", help="explicit indices, skips the draw")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    episodes = json.loads((VAL_BENCHMARK / "benchmark.json").read_text())

    if args.episodes:
        chosen = [int(x) for x in args.episodes.split(",")]
    else:
        lo, _, hi = args.pool.partition("-")
        pool = list(range(int(lo), int(hi) + 1))
        chosen = sorted(random.Random(args.seed).sample(pool, args.count))

    args.out.mkdir(parents=True, exist_ok=True)
    subset = [episodes[i] for i in chosen]
    (args.out / "benchmark.json").write_text(json.dumps(subset))

    metadata = VAL_BENCHMARK / "benchmark_metadata.json"
    if metadata.exists():
        shutil.copy2(metadata, args.out / "benchmark_metadata.json")

    rates = profiled_rates()
    manifest = {
        "source": str(VAL_BENCHMARK),
        "seed": args.seed,
        "pool": args.pool,
        "val_episode_indices": chosen,
        "episodes": [
            {
                "position": position,
                "val_episode": index,
                "task": subset[position]["language"]["task_description"],
                "house": subset[position].get("house_index"),
                "object": subset[position]["task"].get("pickup_obj_name", "?"),
                "profiled_rate": rates.get(index, {}).get("rate"),
                "profiled_band": rates.get(index, {}).get("band"),
            }
            for position, index in enumerate(chosen)
        ],
    }
    (args.out / "selection.json").write_text(json.dumps(manifest, indent=2))

    print(f"{len(chosen)} episodes -> {args.out}")
    print(f"{'pos':>3} {'val ep':>7} {'house':>6} {'coarse':>7}  task")
    for row in manifest["episodes"]:
        rate = "-" if row["profiled_rate"] is None else f"{100 * row['profiled_rate']:.0f}%"
        print(f"{row['position']:>3} {row['val_episode']:>7} {row['house']:>6} {rate:>7}  {row['task']}")

    bands: dict[str, int] = {}
    for row in manifest["episodes"]:
        bands[row["profiled_band"] or "unmeasured"] = bands.get(row["profiled_band"] or "unmeasured", 0) + 1
    print("\nby coarse band (4 rollouts each, so only a rough bucket):", bands)
    print(f"distinct houses: {len({r['house'] for r in manifest['episodes']})}")


if __name__ == "__main__":
    main()
