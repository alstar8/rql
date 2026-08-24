"""Success rate per candidate scene, with an interval, from a profiling run.

Picking an RL scene by a single rollout is guesswork, so each candidate was repeated. This
groups the rollouts back by house -- the evaluator reorganises everything by house, and
each candidate came from a distinct one -- and reports the rate with a Wilson interval so
a 3/12 and a 5/12 are not mistaken for a real difference.

    python scripts/summarize_profile.py --eval-root <dir with shard_*/> --manifest <profile_manifest.json>
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import h5py
import numpy as np


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return max(0.0, centre - margin), min(1.0, centre + margin)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()

    manifest = json.loads(args.manifest.read_text())
    by_house = {}
    for row in manifest:
        by_house.setdefault(row["house_index"], row)

    outcomes: dict[int, list[bool]] = collections.defaultdict(list)
    lengths: dict[int, list[int]] = collections.defaultdict(list)

    for h5_path in sorted(args.eval_root.rglob("trajectories_batch_*.h5")):
        house = int(h5_path.parent.name.split("_")[1])
        with h5py.File(h5_path, "r") as f:
            for key in (k for k in f if k.startswith("traj_")):
                traj = f[key]
                if "success" not in traj:
                    continue
                outcomes[house].append(bool(np.asarray(traj["success"])[-1]))
                lengths[house].append(int(np.asarray(traj["success"]).shape[0]))

    if not outcomes:
        raise SystemExit(f"no trajectories under {args.eval_root}")

    rows = []
    for house, results in outcomes.items():
        info = by_house.get(house, {})
        n, s = len(results), sum(results)
        low, high = wilson(s, n)
        rows.append(
            {
                "house": house,
                "episode": info.get("source_episode"),
                "task": info.get("task", "?"),
                "successes": s,
                "runs": n,
                "rate": s / n if n else 0.0,
                "ci": (low, high),
                "median_steps": int(np.median(lengths[house])) if lengths[house] else 0,
            }
        )
    rows.sort(key=lambda r: r["rate"])

    print(f"{'house':>6} {'ep':>4}  {'SR':>10}  {'95% CI':>16}  {'med steps':>9}  task")
    for r in rows:
        print(f"{r['house']:>6} {str(r['episode']):>4}  "
              f"{r['successes']:>2}/{r['runs']:<2} {100*r['rate']:5.0f}%  "
              f"[{100*r['ci'][0]:4.0f},{100*r['ci'][1]:4.0f}]%  "
              f"{r['median_steps']:>9}  {r['task']}")

    print()
    zero = [r for r in rows if r["rate"] == 0]
    mid = [r for r in rows if 0.15 <= r["rate"] <= 0.45]
    good = [r for r in rows if r["rate"] >= 0.6]
    print("candidates for the three RL settings:")
    print(f"  zero SR      {', '.join(f'house {r[chr(104)+chr(111)+chr(117)+chr(115)+chr(101)]}' for r in zero) or 'none'}")
    print(f"  around 30%   {', '.join(f'house {r[chr(104)+chr(111)+chr(117)+chr(115)+chr(101)]} ({100*r[chr(114)+chr(97)+chr(116)+chr(101)]:.0f}%)' for r in mid) or 'none'}")
    print(f"  already good {', '.join(f'house {r[chr(104)+chr(111)+chr(117)+chr(115)+chr(101)]} ({100*r[chr(114)+chr(97)+chr(116)+chr(101)]:.0f}%)' for r in good) or 'none'}")
    print()
    print("With 12 runs the interval spans roughly 25 points, so treat these as a shortlist,")
    print("not a ranking: two scenes a few points apart are indistinguishable here.")


if __name__ == "__main__":
    main()
