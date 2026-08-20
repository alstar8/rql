#!/usr/bin/env python3
"""Build a provenance-correct V20 replay from per-worker VLA collects."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from chunk_replay import ChunkReplay, ReplaySource


DEFAULT_UID_STRIDE = 1_000_000
DEFAULT_ONLINE_UID_BASE = 10_000_000


def _numeric_shard_id(path: Path) -> int:
    token = path.name.rsplit("_", 1)[-1]
    try:
        return int(token)
    except ValueError as error:
        raise ValueError(f"cannot parse shard id from {path}") from error


def _trajectory_summary(replay: ChunkReplay, target_pose_idx: int) -> dict[str, int]:
    by_uid: dict[int, tuple[int, bool]] = {}
    target_positive_rows = 0
    for row in replay.rows:
        uid = int(row.trajectory_uid)
        pose_idx = int(row.pose_idx)
        success = float(row.success) > 0.5
        previous = by_uid.get(uid)
        current = (pose_idx, success)
        if previous is not None and previous != current:
            raise ValueError(
                f"trajectory {uid} has inconsistent provenance/outcome: "
                f"{previous} vs {current}"
            )
        by_uid[uid] = current
        if pose_idx == int(target_pose_idx) and success:
            target_positive_rows += 1
    target = [
        success
        for pose_idx, success in by_uid.values()
        if pose_idx == int(target_pose_idx)
    ]
    return {
        "rows": len(replay.rows),
        "episodes": len(by_uid),
        "success_episodes": sum(success for _, success in by_uid.values()),
        "target_pose_idx": int(target_pose_idx),
        "target_episodes": len(target),
        "target_success_episodes": sum(target),
        "target_positive_rows": target_positive_rows,
    }


def migrate_offline_shards(
    input_root: Path,
    *,
    pose_cycle: int,
    target_pose_idx: int,
    uid_stride: int = DEFAULT_UID_STRIDE,
    max_transitions: int = 500_000,
) -> tuple[ChunkReplay, dict[str, Any]]:
    """Merge original shard replays without destroying local pose identity."""

    shard_paths = sorted(
        Path(input_root).glob("shard_*/chunk_replay.npz"),
        key=lambda path: _numeric_shard_id(path.parent),
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"no shard_*/chunk_replay.npz files below {Path(input_root)}"
        )
    merged: ChunkReplay | None = None
    shard_rows: list[dict[str, int]] = []
    seen_uids: set[int] = set()
    for path in shard_paths:
        shard_id = _numeric_shard_id(path.parent)
        replay = ChunkReplay.load_npz(
            str(path),
            max_transitions=max_transitions,
            benchmark_pose_cycle=pose_cycle,
        )
        if merged is None:
            merged = ChunkReplay(
                max_transitions=max_transitions,
                chunk_size=replay.chunk_size,
                action_dim=replay.action_dim,
                z_dim=replay.z_dim,
                pos_frac=replay.pos_frac,
                benchmark_pose_cycle=pose_cycle,
            )
        local_ids: set[int] = set()
        for row in replay.rows:
            local_episode_id = int(row.episode_id)
            uid = shard_id * int(uid_stride) + local_episode_id
            local_ids.add(local_episode_id)
            if uid < 0:
                raise ValueError(f"negative migrated uid {uid}")
            row.episode_id = uid
            row.trajectory_uid = uid
            stored_pose = int(row.pose_idx)
            if stored_pose >= 0:
                row.pose_idx = stored_pose
            else:
                row.pose_idx = local_episode_id % int(pose_cycle)
            row.source_policy = int(ReplaySource.OFFLINE_REFERENCE)
            row.worker_id = shard_id
            row.round_id = -1
            row.policy_version = 0
            merged.rows.append(row)
            seen_uids.add(uid)
        shard_rows.append(
            {
                "worker_id": shard_id,
                "rows": len(replay.rows),
                "episodes": len(local_ids),
            }
        )
    assert merged is not None
    merged.n_episodes = len(seen_uids)
    stats = _trajectory_summary(merged, target_pose_idx)
    return merged, {"offline": stats, "shards": shard_rows}


def apply_reencoded_states(
    replay: ChunkReplay,
    reencoded_path: Path,
) -> dict[str, Any]:
    """Replace only z/next_z after verifying identical merged row ordering."""

    with np.load(reencoded_path, allow_pickle=False) as data:
        required = {
            "z",
            "next_z",
            "success",
            "start_step",
            "reference_actions",
            "action_mask",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(
                f"reencoded replay {reencoded_path} is missing {missing}"
            )
        count = len(replay.rows)
        arrays = {key: np.asarray(data[key]) for key in required}
    if any(len(values) != count for values in arrays.values()):
        lengths = {key: len(value) for key, value in arrays.items()}
        raise ValueError(
            f"reencoded replay row count mismatch: expected {count}, got {lengths}"
        )
    original_success = np.asarray(
        [row.success for row in replay.rows],
        dtype=np.float32,
    )
    original_start = np.asarray(
        [row.start_step for row in replay.rows],
        dtype=np.int64,
    )
    original_reference = np.stack(
        [row.reference_actions for row in replay.rows]
    )
    original_mask = np.stack([row.action_mask for row in replay.rows])
    if not (
        np.array_equal(original_success, arrays["success"])
        and np.array_equal(original_start, arrays["start_step"])
        and np.array_equal(original_reference, arrays["reference_actions"])
        and np.array_equal(original_mask, arrays["action_mask"])
    ):
        raise ValueError(
            "reencoded replay order/content does not match original shards"
        )
    if (
        arrays["z"].ndim != 2
        or arrays["next_z"].shape != arrays["z"].shape
        or arrays["z"].shape[1] != replay.z_dim
    ):
        raise ValueError(
            f"reencoded z shapes {arrays['z'].shape}/{arrays['next_z'].shape} "
            f"do not match replay z_dim={replay.z_dim}"
        )
    for index, row in enumerate(replay.rows):
        row.z = arrays["z"][index].astype(np.float32, copy=True)
        row.next_z = arrays["next_z"][index].astype(np.float32, copy=True)
    return {
        "path": str(Path(reencoded_path).resolve()),
        "rows": count,
        "z_dim": int(replay.z_dim),
    }


def _v19_collect_sources(shard_dir: Path) -> dict[int, ReplaySource]:
    metrics_path = shard_dir / "metrics.jsonl"
    sources: dict[int, ReplaySource] = {}
    if not metrics_path.is_file():
        return sources
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        episode_id = int(row.get("valid_episodes", 0)) - 1
        if episode_id < 0:
            continue
        policy = str(row.get("episode_collect_policy", "reference"))
        sources[episode_id] = (
            ReplaySource.ONLINE_REFERENCE
            if policy == "reference"
            else ReplaySource.CHALLENGER
        )
    return sources


def import_v19_online_tails(
    replay: ChunkReplay,
    v19_run_dir: Path,
    *,
    seed_rows: int,
    pose_idx: int,
    online_uid_base: int = DEFAULT_ONLINE_UID_BASE,
) -> dict[str, int]:
    """Import append-only V19 online rows after a consistent stopped snapshot."""

    shard_paths = sorted(
        (Path(v19_run_dir) / "flow_cfgrl").glob("shard_*/chunk_replay.npz"),
        key=lambda path: _numeric_shard_id(path.parent),
    )
    imported_uids: set[int] = set()
    imported_rows = 0
    imported_successes: set[int] = set()
    for path in shard_paths:
        shard_id = _numeric_shard_id(path.parent)
        shard_replay = ChunkReplay.load_npz(
            str(path),
            max_transitions=max(replay.max_transitions, seed_rows + 100_000),
            benchmark_pose_cycle=1,
        )
        if len(shard_replay.rows) < int(seed_rows):
            raise ValueError(
                f"{path} has {len(shard_replay.rows)} rows, below seed_rows={seed_rows}"
            )
        sources = _v19_collect_sources(path.parent)
        for row in shard_replay.rows[int(seed_rows) :]:
            local_episode_id = int(row.episode_id)
            uid = (
                int(online_uid_base)
                + shard_id * 100_000
                + local_episode_id
            )
            row.episode_id = uid
            row.trajectory_uid = uid
            row.pose_idx = int(pose_idx)
            row.source_policy = int(
                sources.get(local_episode_id, ReplaySource.UNKNOWN)
            )
            row.worker_id = shard_id
            row.round_id = local_episode_id + 1
            row.policy_version = -1
            replay.rows.append(row)
            imported_rows += 1
            imported_uids.add(uid)
            if float(row.success) > 0.5:
                imported_successes.add(uid)
    replay.n_episodes = len({int(row.trajectory_uid) for row in replay.rows})
    return {
        "rows": imported_rows,
        "episodes": len(imported_uids),
        "success_episodes": len(imported_successes),
    }


def _validate_expected(
    stats: dict[str, Any],
    *,
    target_episodes: int,
    target_successes: int,
    target_positive_rows: int,
) -> None:
    offline = dict(stats["offline"])
    expected = {
        "target_episodes": int(target_episodes),
        "target_success_episodes": int(target_successes),
        "target_positive_rows": int(target_positive_rows),
    }
    mismatches = {
        key: (offline.get(key), value)
        for key, value in expected.items()
        if value >= 0 and int(offline.get(key, -1)) != value
    }
    if mismatches:
        raise ValueError(f"offline target-pose validation failed: {mismatches}")


def _atomic_save(replay: ChunkReplay, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    replay.save_npz(str(temporary))
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "runs" / "rlt_pretrain_house0_kettle",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-cycle", type=int, default=24)
    parser.add_argument("--target-pose-idx", type=int, default=0)
    parser.add_argument("--reencoded-replay", type=Path)
    parser.add_argument("--uid-stride", type=int, default=DEFAULT_UID_STRIDE)
    parser.add_argument("--max-transitions", type=int, default=500_000)
    parser.add_argument("--expect-target-episodes", type=int, default=56)
    parser.add_argument("--expect-target-successes", type=int, default=2)
    parser.add_argument("--expect-target-positive-rows", type=int, default=76)
    parser.add_argument("--v19-run-dir", type=Path)
    parser.add_argument("--v19-seed-rows", type=int, default=-1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pose_cycle <= 0:
        raise ValueError("--pose-cycle must be positive")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} exists; pass --force to replace it")
    replay, stats = migrate_offline_shards(
        args.input_root,
        pose_cycle=args.pose_cycle,
        target_pose_idx=args.target_pose_idx,
        uid_stride=args.uid_stride,
        max_transitions=args.max_transitions,
    )
    _validate_expected(
        stats,
        target_episodes=args.expect_target_episodes,
        target_successes=args.expect_target_successes,
        target_positive_rows=args.expect_target_positive_rows,
    )
    if args.reencoded_replay is not None:
        stats["reencoded"] = apply_reencoded_states(
            replay,
            args.reencoded_replay,
        )
    if args.v19_run_dir is not None:
        seed_rows = (
            len(replay.rows)
            if args.v19_seed_rows < 0
            else int(args.v19_seed_rows)
        )
        stats["v19_online"] = import_v19_online_tails(
            replay,
            args.v19_run_dir,
            seed_rows=seed_rows,
            pose_idx=args.target_pose_idx,
        )
    stats["combined"] = _trajectory_summary(replay, args.target_pose_idx)
    _atomic_save(replay, args.output)
    manifest = args.output.with_suffix(".provenance.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input_root": str(args.input_root.resolve()),
                "output": str(args.output.resolve()),
                **stats,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
