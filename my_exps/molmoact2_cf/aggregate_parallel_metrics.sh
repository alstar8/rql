#!/usr/bin/env bash
# Aggregate per-shard metrics.jsonl into one table (window + cumulative SR).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
OUT_ROOT="${1:-$ROOT/runs/molmoact2_cf_full_8gpu}"

python3 - <<PY
import json, glob, os
from pathlib import Path
root = Path("${OUT_ROOT}")
rows = []
for p in sorted(root.glob("shard_*/metrics.jsonl")):
    shard = p.parent.name
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        r["shard"] = shard
        rows.append(r)
rows.sort(key=lambda r: (r.get("shard", ""), r.get("local_episodes", 0)))
print(f"{'shard':10s} {'local':>6s} {'global':>7s} {'win_sr':>7s} {'cum_sr':>7s} {'adv':>8s} {'rms':>7s}")
for r in rows:
    print(
        f"{r.get('shard',''):10s} {r.get('local_episodes',0):6d} {r.get('global_episode',0):7d} "
        f"{r.get('window_success_rate',0):7.3f} {r.get('cumulative_success_rate',0):7.3f} "
        f"{r.get('g_predicted_advantage', float('nan')):8.3f} {r.get('g_residual_rms', float('nan')):7.4f}"
    )
# Overall: last cum_sr per shard, weighted by local eps
last = {}
for r in rows:
    last[r["shard"]] = r
if last:
    tot_n = sum(r.get("local_episodes", 0) for r in last.values())
    wsr = sum(r.get("cumulative_success_rate", 0) * r.get("local_episodes", 0) for r in last.values()) / max(tot_n, 1)
    print(f"\\nweighted_cum_sr_across_shards={wsr:.4f} episodes={tot_n}")
PY
