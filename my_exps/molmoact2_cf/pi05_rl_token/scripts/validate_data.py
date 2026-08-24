"""Inspect the token shards before spending GPU hours on them.

Reads headers and a small sample only -- safe to run any time, changes nothing.

    python scripts/validate_data.py --runs <.../runs/rlt_pretrain_demo1k>
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running as `python scripts/x.py` puts scripts/ on sys.path, not the package root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import numpy as np

from rlt.data import TokenSequences, resolve


def check_token_shards(pattern: str, probe: int = 8) -> None:
    print(f"\n=== token shards: {pattern}")
    paths = resolve(pattern)
    total = 0
    for p in paths:
        with np.load(p, allow_pickle=True) as d:
            n = len(d["tokens"])
            first = np.asarray(d["tokens"][0])
        total += n
        print(f"  {p.name:44s} {n:6d} sequences   first {first.shape} {first.dtype}")
    print(f"  TOTAL {total} sequences")

    sample = TokenSequences.load(paths[:1], max_sequences=probe)
    print("  sample stats:", json.dumps(sample.stats()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="dir holding the token_replay npz files")
    ap.add_argument("--probe", type=int, default=8)
    args = ap.parse_args()

    runs = Path(args.runs)
    check_token_shards(str(runs / "token_replay_*_s*.npz"), args.probe)
    print("\ndone (nothing was modified)")


if __name__ == "__main__":
    main()
