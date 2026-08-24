"""Flatten the token shards once into a memmap the whole sweep can share.

Every training process otherwise decompresses its own copy of the ragged npz --
eight sweep runs means eight times ~20 GB pulled off NFS before a single step.
Writing one padded float16 array instead lets the runs mmap it, so the OS page
cache serves all of them from one read.

    python scripts/prepare_token_cache.py \
        --token_replay "$RUNS/token_replay_*_s*.npz" \
        --out runs/token_cache --max_sequences 8000

Produces <out>/tokens.npy (N, S_max, 2560) float16 and <out>/lengths.npy (N,).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Running as `python scripts/x.py` puts scripts/ on sys.path, not the package root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import numpy as np

from rlt.config import VLA_TOKEN_DIM
from rlt.data import resolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token_replay", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_sequences", type=int, default=8000, help="0 = all")
    ap.add_argument("--max_len", type=int, default=512, help="truncate longer sequences")
    args = ap.parse_args()

    paths = resolve(args.token_replay)
    quota = None if not args.max_sequences else max(1, args.max_sequences // len(paths))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Pass 1: count and measure, so the memmap can be allocated exactly once.
    counts, lengths = [], []
    for p in paths:
        with np.load(p, allow_pickle=True) as d:
            take = len(d["masks"]) if quota is None else min(quota, len(d["masks"]))
            counts.append(take)
            lengths += [min(int(np.asarray(m).shape[0]), args.max_len) for m in d["masks"][:take]]
        print(f"  {p.name}: {counts[-1]} sequences")

    n = sum(counts)
    s_max = max(lengths)
    nbytes = n * s_max * VLA_TOKEN_DIM * 2
    print(f"\n{n} sequences, padded to {s_max} -> {nbytes / 2**30:.1f} GiB float16")

    tokens = np.lib.format.open_memmap(
        out / "tokens.npy", mode="w+", dtype=np.float16, shape=(n, s_max, VLA_TOKEN_DIM)
    )

    # Pass 2: fill. Interleaved by shard quota, so both MolmoBot and DROID are
    # represented -- a prefix of the sorted shards would be DROID only.
    row = 0
    for p, take in zip(paths, counts):
        with np.load(p, allow_pickle=True) as d:
            for t in d["tokens"][:take]:
                t = np.asarray(t, dtype=np.float16)[: args.max_len]
                tokens[row, : t.shape[0]] = t
                row += 1
        print(f"  wrote {p.name} -> {row}/{n}")

    tokens.flush()
    np.save(out / "lengths.npy", np.asarray(lengths, dtype=np.int32))
    (out / "meta.json").write_text(
        json.dumps(
            {"n": n, "s_max": s_max, "token_dim": VLA_TOKEN_DIM, "shards": [p.name for p in paths]},
            indent=2,
        )
    )
    print(f"\nwrote {out}/tokens.npy and lengths.npy")
    print(f"point training at it with:  --token_cache {out}")


if __name__ == "__main__":
    main()
