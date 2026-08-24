"""Did the same scene get better between visits?

Training on many scenes makes a success-rate curve unreadable: a window's rate
mostly reports which scenes happened to fall into it. With a round-robin pool
every scene is revisited, so the honest comparison is within a scene -- visit 1
against visit 3, paired, on scenes that were visited both times.

    python scripts/compare_visits.py --episodes runs/online_pick_all/episodes.jsonl

McNemar's exact test on the disagreeing scenes: only scenes whose outcome
changed carry information about whether anything improved.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_visits(path: Path) -> dict[int, list[dict]]:
    """benchmark scene -> its visits, in the order they were claimed."""
    visits: dict[int, list[dict]] = defaultdict(list)
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if "benchmark_idx" in record:
            visits[int(record["benchmark_idx"])].append(record)
    for scene in visits.values():
        scene.sort(key=lambda r: r["episode"])
    return visits


def mcnemar(only_first: int, only_second: int) -> float:
    """Exact two-sided p: under no change each disagreement is a coin flip."""
    disagree = only_first + only_second
    if disagree == 0:
        return 1.0
    tail = sum(math.comb(disagree, i) for i in range(0, min(only_first, only_second) + 1))
    return min(1.0, 2 * tail / (2**disagree))


def compare(visits: dict[int, list[dict]], first: int, second: int) -> None:
    pairs = [(v[first], v[second]) for v in visits.values() if len(v) > max(first, second)]
    if not pairs:
        print(f"no scene has been visited {max(first, second) + 1} times yet")
        return

    a = sum(int(x["success"]) for x, _ in pairs)
    b = sum(int(y["success"]) for _, y in pairs)
    only_a = sum(1 for x, y in pairs if x["success"] and not y["success"])
    only_b = sum(1 for x, y in pairs if y["success"] and not x["success"])
    warm = sum(1 for x, _ in pairs if x["phase"] == 0)

    print(f"\n{len(pairs)} scenes visited both at #{first + 1} and #{second + 1}"
          f"  ({warm} of the earlier visits were warmup, i.e. the VLA acting)")
    print(f"  visit #{first + 1}: {a:4d}/{len(pairs)} = {a / len(pairs):.1%}")
    print(f"  visit #{second + 1}: {b:4d}/{len(pairs)} = {b / len(pairs):.1%}")
    print(f"  improved {only_b}, regressed {only_a}, unchanged {len(pairs) - only_a - only_b}")
    p = mcnemar(only_a, only_b)
    print(f"  McNemar exact p = {p:.4f}  ->  " + ("a real change" if p <= 0.05 else "within noise"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", required=True, help="episodes.jsonl from a training run")
    ap.add_argument("--pairs", default="0:1,0:2,1:2", help="visit pairs to compare, zero-based")
    args = ap.parse_args()

    visits = load_visits(Path(args.episodes))
    counts = defaultdict(int)
    for scene in visits.values():
        counts[len(scene)] += 1
    print(f"{len(visits)} scenes seen; visits per scene: "
          + ", ".join(f"{n}x{counts[n]}" for n in sorted(counts)))

    for pair in args.pairs.split(","):
        first, second = (int(v) for v in pair.split(":"))
        compare(visits, first, second)


if __name__ == "__main__":
    main()
