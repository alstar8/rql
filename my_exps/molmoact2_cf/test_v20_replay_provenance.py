"""CPU regression tests for V20 replay provenance and sampling."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_replay import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    Z_DIM,
    ChunkReplay,
    ReplaySource,
)
from merge_chunk_replay_provenance import (  # noqa: E402
    apply_reencoded_states,
    migrate_offline_shards,
)


def _add_episode(
    replay: ChunkReplay,
    *,
    episode_id: int,
    trajectory_uid: int,
    pose_idx: int,
    success: bool,
    rows: int = 4,
) -> None:
    zs = [
        np.full((Z_DIM,), float(trajectory_uid + index), dtype=np.float32)
        for index in range(rows)
    ]
    proprios = [
        np.full((ACTION_DIM,), float(index), dtype=np.float32)
        for index in range(rows)
    ]
    references = [
        np.full(
            (CHUNK_SIZE, ACTION_DIM),
            float(trajectory_uid + index),
            dtype=np.float32,
        )
        for index in range(rows)
    ]
    replay.add_episode_chunks(
        zs,
        proprios,
        references,
        [value + 0.1 for value in references],
        [
            np.full((CHUNK_SIZE,), float(success), dtype=np.float32)
            for _ in range(rows)
        ],
        [
            np.ones((CHUNK_SIZE,), dtype=np.float32)
            for _ in range(rows)
        ],
        success=success,
        gamma=0.99,
        episode_id=episode_id,
        trajectory_uid=trajectory_uid,
        pose_idx=pose_idx,
        source_policy=ReplaySource.OFFLINE_REFERENCE,
        worker_id=trajectory_uid // 1_000_000,
        round_id=-1,
        policy_version=0,
    )


def test_provenance_round_trip_and_uid_grouping(tmp_path: Path) -> None:
    replay = ChunkReplay(
        max_transitions=100,
        pos_frac=0.5,
        benchmark_pose_cycle=24,
        seed=1,
    )
    _add_episode(
        replay,
        episode_id=0,
        trajectory_uid=1_000_000,
        pose_idx=7,
        success=True,
    )
    _add_episode(
        replay,
        episode_id=0,
        trajectory_uid=2_000_000,
        pose_idx=0,
        success=False,
    )
    assert replay.successful_episode_count() == 1
    assert replay.target_successful_episode_count(7) == 1

    path = tmp_path / "provenance.npz"
    replay.save_npz(str(path))
    restored = ChunkReplay.load_npz(str(path))
    assert {row.trajectory_uid for row in restored.rows} == {
        1_000_000,
        2_000_000,
    }
    assert {row.pose_idx for row in restored.rows} == {0, 7}
    assert {row.source_policy for row in restored.rows} == {
        int(ReplaySource.OFFLINE_REFERENCE)
    }
    natural = restored.sample_natural(20)
    uids = natural["trajectory_uid"].cpu().numpy()
    assert np.count_nonzero(uids == 1_000_000) == 10
    assert np.count_nonzero(uids == 2_000_000) == 10


def test_target_pose_sampling_is_trajectory_first_and_stratified() -> None:
    replay = ChunkReplay(
        max_transitions=1_000,
        pos_frac=0.5,
        benchmark_pose_cycle=24,
        seed=2,
    )
    for uid in range(4):
        _add_episode(
            replay,
            episode_id=uid,
            trajectory_uid=uid,
            pose_idx=0,
            success=True,
            rows=1 + uid,
        )
    for uid in range(10, 14):
        _add_episode(
            replay,
            episode_id=uid,
            trajectory_uid=uid,
            pose_idx=3,
            success=True,
            rows=1 + uid - 10,
        )
    for uid in range(20, 28):
        _add_episode(
            replay,
            episode_id=uid,
            trajectory_uid=uid,
            pose_idx=uid % 2,
            success=False,
            rows=1 + uid - 20,
        )

    batch = replay.sample(
        100,
        require_both_outcomes=True,
        target_pose_idx=0,
        target_positive_fraction=0.75,
        trajectory_first=True,
        temporal_bins=4,
    )
    success = batch["success"] > 0.5
    target_positive = success & (batch["pose_idx"] == 0)
    assert int(success.sum()) == 50
    assert int(target_positive.sum()) == 38
    assert len(torch.unique(batch["trajectory_uid"][target_positive])) > 1


def test_offline_migration_preserves_local_pose_before_global_uid(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    for shard_id in range(2):
        replay = ChunkReplay(
            max_transitions=100,
            pos_frac=0.5,
            benchmark_pose_cycle=3,
        )
        for local_id in range(4):
            _add_episode(
                replay,
                episode_id=local_id,
                trajectory_uid=local_id,
                pose_idx=local_id % 3,
                success=local_id == shard_id,
                rows=1,
            )
        path = source_root / f"shard_{shard_id}" / "chunk_replay.npz"
        path.parent.mkdir(parents=True)
        replay.save_npz(str(path))

    merged, stats = migrate_offline_shards(
        source_root,
        pose_cycle=3,
        target_pose_idx=0,
        max_transitions=100,
    )
    assert merged.n_episodes == 8
    assert len({row.trajectory_uid for row in merged.rows}) == 8
    assert {
        row.pose_idx
        for row in merged.rows
        if row.trajectory_uid == 1_000_003
    } == {0}
    assert stats["offline"]["target_episodes"] == 4
    assert stats["offline"]["target_success_episodes"] == 1

    reencoded_path = tmp_path / "reencoded.npz"
    merged.save_npz(str(reencoded_path))
    with np.load(reencoded_path, allow_pickle=False) as data:
        payload = {
            key: np.asarray(data[key]).copy()
            for key in data.files
        }
    payload["z"] += 7.0
    payload["next_z"] += 11.0
    np.savez_compressed(reencoded_path, **payload)
    before_z = merged.rows[0].z.copy()
    reencoded_stats = apply_reencoded_states(merged, reencoded_path)
    assert reencoded_stats["rows"] == len(merged.rows)
    assert np.array_equal(merged.rows[0].z, before_z + 7.0)
