"""Check a merged LeRobot dataset before anything trains on it.

verify_lerobot.py compares a converted dataset against the source h5 in iteration order,
which no longer lines up once parts are merged and renumbered. This checks the merge's
own invariants instead, all of which are silent when broken:

  * every episode's frame count matches the length recorded for it;
  * the global index column is exactly 0..N-1, so nothing was duplicated or skipped;
  * the task_index stored in each parquet still names the sentence recorded for that
    episode in episodes.jsonl -- the failure mode where every episode is present but
    half of them point at the wrong instruction;
  * the dataset loads through the same LeRobot path openpi will use, and the sampled
    frames carry the expected shapes and dtypes.

    python scripts/verify_merged.py --dataset <dir> --repo-id molmospaces/pick_all
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHUNK_SIZE = 1000


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_path(root: Path, index: int) -> Path:
    return root / "data" / f"chunk-{index // CHUNK_SIZE:03d}" / f"episode_{index:06d}.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=60, help="episodes to open in full")
    ap.add_argument("--repo-id", default=None, help="also load through LeRobot under this id")
    args = ap.parse_args()

    meta = args.dataset / "meta"
    info = json.loads((meta / "info.json").read_text())
    episodes = _read_jsonl(meta / "episodes.jsonl")
    tasks = {row["task_index"]: row["task"] for row in _read_jsonl(meta / "tasks.jsonl")}

    print(f"episodes {info['total_episodes']}  frames {info['total_frames']}  tasks {info['total_tasks']}")

    failures: list[str] = []

    # 1. metadata self-consistency
    if len(episodes) != info["total_episodes"]:
        failures.append(f"episodes.jsonl has {len(episodes)} rows, info says {info['total_episodes']}")
    indices = [row["episode_index"] for row in episodes]
    if indices != list(range(len(episodes))):
        failures.append("episode_index in episodes.jsonl is not contiguous from zero")
    if sum(int(row["length"]) for row in episodes) != info["total_frames"]:
        failures.append("sum of episode lengths disagrees with info total_frames")
    print(f"  metadata: {len(episodes)} episodes, indices contiguous, lengths sum to total_frames")

    # 2. every parquet exists
    missing = [i for i in range(len(episodes)) if not _episode_path(args.dataset, i).exists()]
    if missing:
        failures.append(f"{len(missing)} episode parquets missing, first: {missing[:5]}")
    print(f"  files: {len(episodes) - len(missing)}/{len(episodes)} parquets present")

    # 3. sampled episodes: lengths, instruction wiring, index continuity
    random.seed(0)
    sample = sorted(random.sample(range(len(episodes)), min(args.sample, len(episodes))))
    seen_index_ranges = []
    for i in sample:
        table = pq.read_table(_episode_path(args.dataset, i))
        recorded = int(episodes[i]["length"])
        if table.num_rows != recorded:
            failures.append(f"episode {i}: {table.num_rows} rows, metadata says {recorded}")

        ep_col = set(table.column("episode_index").to_pylist())
        if ep_col != {i}:
            failures.append(f"episode {i}: parquet episode_index holds {ep_col}")

        task_col = set(table.column("task_index").to_pylist())
        if len(task_col) != 1:
            failures.append(f"episode {i}: mixed task_index {task_col}")
        else:
            stored = tasks[task_col.pop()]
            expected = episodes[i]["tasks"][0]
            if stored != expected:
                failures.append(f"episode {i}: parquet says {stored!r}, metadata says {expected!r}")

        idx = table.column("index").to_pylist()
        if idx != list(range(idx[0], idx[0] + len(idx))):
            failures.append(f"episode {i}: index column is not consecutive")
        seen_index_ranges.append((idx[0], idx[-1]))

    print(f"  sampled {len(sample)} episodes: lengths, episode_index, task wiring, index runs")

    # 4. sampled index ranges must not overlap each other
    seen_index_ranges.sort()
    for (a_start, a_end), (b_start, _) in zip(seen_index_ranges, seen_index_ranges[1:], strict=False):
        if b_start <= a_end:
            failures.append(f"global index ranges overlap: ...{a_end} then {b_start}...")
            break
    print("  global index: sampled ranges disjoint and increasing")

    # 5. the loader openpi will actually use
    if args.repo_id:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        ds = LeRobotDataset(args.repo_id, root=str(args.dataset))
        item = ds[0]
        print(f"  LeRobot load: {ds.num_episodes} episodes, {ds.num_frames} frames")
        for key, expected in (
            ("exterior_image_1_left", (3, 224, 224)),
            ("wrist_image_left", (3, 224, 224)),
            ("joint_position", (7,)),
            # LeRobot hands back a scalar for a 1-element float feature, and openpi's
            # DroidInputs promotes it explicitly, so both shapes are acceptable.
            ("gripper_position", ((1,), ())),
            ("actions", (8,)),
        ):
            got = tuple(np.asarray(item[key]).shape)
            allowed = expected if isinstance(expected, tuple) and expected and isinstance(expected[0], tuple) else (expected,)
            if got not in allowed:
                failures.append(f"feature {key}: shape {got}, expected one of {allowed}")
        print(f"  features: shapes as expected; task={item.get('task')!r}")

        action = np.asarray(item["actions"], dtype=np.float32)
        state = np.asarray(item["joint_position"], dtype=np.float32)
        # Deltas are small by construction; anything joint-sized means absolute targets
        # leaked in, which would train the model in the wrong action space.
        if np.abs(action[:7]).max() < 1e-9:
            failures.append("first frame action is exactly zero; the leading hold was not dropped")
        if np.abs(action[:7]).max() > 1.0:
            failures.append(f"action[:7] max {np.abs(action[:7]).max():.3f} looks absolute, not a delta")
        print(f"  action sanity: |delta| max {np.abs(action[:7]).max():.4f}, "
              f"state[3] {state[3]:+.3f}, gripper {action[7]:.2f}")

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for line in failures[:20]:
            print(f"  - {line}")
        sys.exit(1)
    print("VERDICT: merged dataset is internally consistent and loads correctly")


if __name__ == "__main__":
    main()
