#!/usr/bin/env bash
# Merge per-GPU TokenReplay shards into one npz for RLT token warmup.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd -P)"
IN_DIR="${1:-${ROOT}/runs/rlt_token_collect_v4/shards}"
OUT="${2:-${ROOT}/runs/rlt_token_collect_v4/token_replay_merged.npz}"
PYTHON="${ROOT}/../../../molmospaces/.venv/bin/python"
"$PYTHON" - <<PY
from pathlib import Path
import numpy as np
from chunk_replay import TokenReplay
root = Path("${IN_DIR}")
out = Path("${OUT}")
paths = sorted(root.glob("*/token_replay.npz"))
if not paths:
    raise SystemExit(f"no token_replay.npz under {root}")
merged = TokenReplay()
for p in paths:
    part = TokenReplay.load_npz(str(p))
    for t, m in zip(part.tokens, part.masks):
        merged.add(t, m)
    print(f"loaded {p} n={len(part)} total={len(merged)}")
out.parent.mkdir(parents=True, exist_ok=True)
merged.save_npz(str(out))
print(f"wrote {out} sequences={len(merged)}")
PY
