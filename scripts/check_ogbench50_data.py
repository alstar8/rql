#!/usr/bin/env python3
"""Verify OGBench datasets required for the 50-task RQL benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ogbench50_config import expand_tasks, unique_dataset_names

OGBENCH_DATA_DIR = Path(
    os.environ.get(
        "OGBENCH_DATA_DIR",
        "/workspace-SR008.nfs2/users/staroverov/ogbench/data",
    )
)
OGBENCH_100M_ROOT = Path(
    os.environ.get(
        "OGBENCH_100M_ROOT",
        "/workspace-SR008.nfs2/users/staroverov/ogbench/100m",
    )
)
MIN_100M_SHARDS = int(os.environ.get("OGBENCH_100M_MIN_SHARDS", "50"))


def main() -> int:
    missing: list[str] = []

    for name in unique_dataset_names():
        for suffix in ("", "-val"):
            path = OGBENCH_DATA_DIR / f"{name}{suffix}.npz"
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))

    env_to_dir = {
        "OGBENCH_PUZZLE_4X4_100M_DIR": os.environ.get(
            "OGBENCH_PUZZLE_4X4_100M_DIR",
            str(OGBENCH_100M_ROOT / "puzzle-4x4-play-100m-v0"),
        ),
        "OGBENCH_CUBE_QUADRUPLE_100M_DIR": os.environ.get(
            "OGBENCH_CUBE_QUADRUPLE_100M_DIR",
            str(OGBENCH_100M_ROOT / "cube-quadruple-play-100m-v0"),
        ),
    }

    tasks_needing_100m = {t.dataset_100m_dir_env for t in expand_tasks() if t.dataset_100m_dir_env}
    for env_var in tasks_needing_100m:
        dir_path = Path(env_to_dir[env_var])
        shard_count = len(
            [p for p in dir_path.glob("*.npz") if "-val" not in p.name]
        )
        if shard_count < MIN_100M_SHARDS:
            missing.append(
                f"{env_var}={dir_path} (have {shard_count} shards, need >= {MIN_100M_SHARDS})"
            )

    if missing:
        print("Missing OGBench 50-task datasets:", file=sys.stderr)
        for item in missing:
            print(f"  {item}", file=sys.stderr)
        return 1

    print(f"All required datasets present under {OGBENCH_DATA_DIR}")
    for env_var, dir_path in env_to_dir.items():
        if env_var in tasks_needing_100m:
            shards = len(list(Path(dir_path).glob("*.npz")))
            print(f"  {env_var}: {shards} .npz files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
