#!/usr/bin/env python
"""Collect a sweep's result.json files into one table, with pairwise significance.

    python scripts/summarize_sweep.py <sweep dir>

Prints the grid, then every comparison that a decision might rest on, tested with
Fisher's exact test. A difference in success rates is not a finding until it survives
one: at 128 episodes, a gap of 8 points is inside the noise.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path


def fisher_two_sided(a: int, b: int, c: int, d: int) -> float:
    """P(table at least as extreme), 2x2, no scipy in every venv here."""
    n = a + b + c + d
    row1, col1 = a + b, a + c
    observed = comb(row1, a) * comb(n - row1, c) if row1 <= n else 0.0
    total = 0.0
    keep = 0.0
    low = max(0, col1 - (n - row1))
    high = min(row1, col1)
    for x in range(low, high + 1):
        weight = comb(row1, x) * comb(n - row1, col1 - x)
        total += weight
        if weight <= observed * (1 + 1e-9):
            keep += weight
    return keep / total if total else 1.0


def load(sweep_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(sweep_dir.glob("*/result.json")):
        data = json.loads(path.read_text())
        if not data.get("episodes_run"):
            print(f"  (skipping {path.parent.name}: no episodes completed)")
            continue
        data["name"] = path.parent.name
        rows.append(data)
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    sweep_dir = Path(sys.argv[1])
    rows = load(sweep_dir)
    if not rows:
        raise SystemExit(f"no finished runs under {sweep_dir}")

    print(f"\n{'run':28s} {'chunk':>5s} {'conversion':>10s} {'successes':>10s} {'rate':>7s}  95% CI")
    print("-" * 84)
    for row in sorted(rows, key=lambda r: (r["chunk_size"], r["conversion"])):
        low, high = row["ci95"]
        print(
            f"{row['name']:28s} {row['chunk_size']:5d} {row['conversion']:>10s} "
            f"{row['successes']:5d}/{row['episodes_run']:<4d} "
            f"{100 * row['success_rate']:6.1f}%  {100 * low:4.1f}-{100 * high:4.1f}%"
        )

    print("\npairwise (Fisher exact, two-sided):")
    for left, right in combinations(sorted(rows, key=lambda r: (r["chunk_size"], r["conversion"])), 2):
        p = fisher_two_sided(
            left["successes"], left["episodes_run"] - left["successes"],
            right["successes"], right["episodes_run"] - right["successes"],
        )
        mark = "  <-- significant" if p < 0.05 else ""
        print(f"  {left['name']:26s} vs {right['name']:26s}  p={p:.4f}{mark}")

    control = [r for r in rows if r["chunk_size"] == 1]
    if len(control) == 2:
        p = fisher_two_sided(
            control[0]["successes"], control[0]["episodes_run"] - control[0]["successes"],
            control[1]["successes"], control[1]["episodes_run"] - control[1]["successes"],
        )
        print(
            f"\ncontrol: the two chunk-1 cells are the same regime by construction, so "
            f"p={p:.4f} here\nbounds the harness noise. A small p would mean the rest of "
            "the table is not trustworthy."
        )


if __name__ == "__main__":
    main()
