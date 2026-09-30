#!/usr/bin/env python
"""Merge ChunkReplay npz dumps into one shared replay.

The shared-policy run pretrains on the union of every task's collected buffer.
Each per-task buffer is a ChunkReplay npz (rlt/replay.py FIELDS + size); this
concatenates them row-wise and writes one npz whose `size` is the sum. All
inputs must agree on field shapes -- a buffer collected with a different
encoder or chunk size is a different MDP and must not be mixed in silently.

    scripts/merge_replays.py --out merged.npz runs/task_a/buffer.npz runs/task_b/buffer.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rlt.replay import ChunkReplay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="ChunkReplay npz files")
    parser.add_argument("--out", type=Path, required=True, help="merged npz")
    args = parser.parse_args()

    merged: dict[str, list[np.ndarray]] = {f: [] for f in ChunkReplay.FIELDS}
    shapes: dict[str, tuple] | None = None
    total = 0
    for path in args.inputs:
        data = np.load(path)
        size = int(data["size"])
        field_shapes = {f: data[f].shape[1:] for f in ChunkReplay.FIELDS}
        if shapes is None:
            shapes = field_shapes
        elif field_shapes != shapes:
            raise ValueError(
                f"{path} field shapes {field_shapes} != {shapes}; buffers from "
                "different encoders or chunk sizes cannot share a replay"
            )
        for field in ChunkReplay.FIELDS:
            merged[field].append(data[field][:size])
        total += size
        print(f"{path}: {size} rows")

    out = {f: np.concatenate(merged[f], axis=0) for f in ChunkReplay.FIELDS}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, size=total, **out)
    print(f"wrote {args.out}: {total} rows")


if __name__ == "__main__":
    main()
