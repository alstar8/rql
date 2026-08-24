#!/usr/bin/env python3
"""Merge collect.json files for one scene into corpus_stats.json and a markdown note."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: summarize_collect.py <collect_dir> <out_json>")
    collect_dir = Path(sys.argv[1])
    out = Path(sys.argv[2])
    rows = []
    for path in sorted(collect_dir.glob("*/collect.json")):
        rows.append(json.loads(path.read_text()))
    episodes = sum(int(r.get("episodes_run") or 0) for r in rows)
    successes = sum(int(r.get("successes") or 0) for r in rows)
    sequences = sum(int(r.get("sequences") or 0) for r in rows)
    scene = rows[0]["scene"] if rows else collect_dir.name
    token_dir = collect_dir.parent / "tokens" / scene
    shards = len(list(token_dir.glob("*.npz"))) if token_dir.is_dir() else 0
    stats = {
        "scene": scene,
        "collectors": len(rows),
        "episodes": episodes,
        "successes": successes,
        "success_rate": (successes / episodes) if episodes else None,
        "token_sequences_reported": sequences,
        "npz_shards": shards,
        "runs": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    sr = f"{100 * successes / episodes:.1f}%" if episodes else "n/a"
    out.with_suffix(".md").write_text(
        f"# Frozen-pi0.5 token corpus: `{scene}`\n\n"
        f"Collected for phase-1 AE pretrain (whole episode, gate or no gate). "
        f"Success is **not** an AE training label — reconstruction only. "
        f"SR below is the frozen VLA rate on the **train** repeats used for collection.\n\n"
        f"| Item | Value |\n| --- | ---: |\n"
        f"| Collectors | {len(rows)} |\n"
        f"| Trajectories (episodes) | **{episodes}** |\n"
        f"| Successes | **{successes}** |\n"
        f"| Collect SR | **{sr}** |\n"
        f"| Token sequences (server count) | {sequences} |\n"
        f"| npz shards | {shards} |\n\n"
        f"Collectors stop at a trajectory cap (100 total across workers) so the "
        f"AE/pretrain corpus is not padded with extra rollouts.\n"
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
