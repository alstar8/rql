"""Recover metrics JSONL from an openpi training log.

openpi writes its own progress line through tqdm:

    Step 2500: grad_norm=0.1860, loss=0.0060, param_norm=1832.8049

This parses those lines into the `.metrics.jsonl` this project uses, so TensorBoard can
be rebuilt for a run whose JSONL is missing metrics -- which is exactly what happened to
the first full run, where a type check dropped every numpy scalar before it was written.

Safe to re-run on a run still in progress: it rewrites the file from the log each time.
tqdm separates lines with carriage returns, so the log is split on those too.

    python scripts/log_to_jsonl.py --log <train.log> --out <run>.metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STEP_LINE = re.compile(r"Step (\d+):\s*(.+)")
PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=(-?[\d.]+(?:[eE][-+]?\d+)?)")


def parse(log_path: Path) -> list[dict]:
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    rows: dict[int, dict] = {}
    for line in text.splitlines():
        match = STEP_LINE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        row = {"step": step}
        for key, value in PAIR.findall(match.group(2)):
            try:
                row[key] = float(value)
            except ValueError:
                continue
        if len(row) > 1:
            rows[step] = row  # later duplicates win, e.g. after a resume
    return [rows[k] for k in sorted(rows)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        raise SystemExit(f"no 'Step N: ...' lines found in {args.log}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    first, last = rows[0], rows[-1]
    keys = [k for k in last if k != "step"]
    print(f"parsed {len(rows)} logged steps, {first['step']} -> {last['step']}")
    print(f"metrics: {', '.join(keys)}")
    for key in keys:
        if key in first:
            print(f"  {key:12s} {first[key]:12.4f} -> {last[key]:.4f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
