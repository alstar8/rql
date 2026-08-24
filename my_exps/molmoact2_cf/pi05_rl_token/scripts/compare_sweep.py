"""Rank sweep runs fairly.

`recon_loss` is the error on a single batch of 32, not a running average, so it
bounces by ~0.28 between adjacent log lines. Comparing runs by one snapshot
each -- especially at different steps -- mixes batch noise and progress with the
real difference. This averages a window at a step every run has reached.

    python scripts/compare_sweep.py --runs_dir runs/ae_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a run still writing can leave a torn last line
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--window", type=int, default=10, help="log lines to average")
    ap.add_argument(
        "--at_step",
        type=int,
        default=0,
        help="compare at this step instead of the common one; runs that have not "
        "reached it are listed separately rather than dragging the bar down",
    )
    args = ap.parse_args()

    runs = {}
    for path in sorted(Path(args.runs_dir).glob("*.metrics.jsonl")):
        rows = load(path)
        if rows:
            runs[path.name.replace(".metrics.jsonl", "")] = rows
    if not runs:
        raise SystemExit(f"no metrics under {args.runs_dir}")

    if args.at_step:
        common = args.at_step
        too_young = {n: max(r["step"] for r in rows) for n, rows in runs.items()
                     if max(r["step"] for r in rows) < common}
        runs = {n: rows for n, rows in runs.items() if n not in too_young}
        if not runs:
            raise SystemExit(f"no run has reached step {common}")
        print(f"comparing at step {common} (requested)")
        if too_young:
            listed = ", ".join(f"{n} @{s}" for n, s in sorted(too_young.items()))
            print(f"not there yet, excluded: {listed}")
    else:
        # The furthest step every run has reached, so nobody is judged early.
        common = min(max(r["step"] for r in rows) for rows in runs.values())
        print(f"comparing at step {common} (the furthest point all runs reached)")

    print(f"averaging the last {args.window} log lines at or before it\n")
    print(f"{'run':<20} {'recon':>9} {'spread':>9} {'|z|':>8} {'at step':>9} {'reached':>9}")

    table = []
    for name, rows in runs.items():
        near = [r for r in rows if r["step"] <= common][-args.window :]
        recon = [r["recon_loss"] for r in near]
        mean = sum(recon) / len(recon)
        table.append((mean, name, max(recon) - min(recon),
                      sum(r["z_norm"] for r in near) / len(near),
                      near[-1]["step"], rows[-1]["step"]))

    for mean, name, spread, znorm, at, reached in sorted(table):
        print(f"{name:<20} {mean:>9.4f} {spread:>9.4f} {znorm:>8.2f} {at:>9} {reached:>9}")

    best = sorted(table)[0]
    runner_up = sorted(table)[1] if len(table) > 1 else None
    print(f"\nbest: {best[1]} ({best[0]:.4f})")
    if runner_up:
        gap = runner_up[0] - best[0]
        print(f"gap to next ({runner_up[1]}): {gap:.4f}", end="  ")
        print("-- larger than the batch noise, so it is real"
              if gap > best[2] else "-- SMALLER than the batch spread, so not yet separable")


if __name__ == "__main__":
    main()
