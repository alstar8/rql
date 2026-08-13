"""Focused CPU tests for V14 replay retention and persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_replay import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    FULL_ACTION_HORIZON,
    Z_DIM,
    ChunkReplay,
    ChunkTransition,
    ImageChunkReplay,
)


def _add_image_episode(
    replay: ImageChunkReplay,
    *,
    episode_id: int,
    success: bool,
    row_count: int = 1,
) -> None:
    markers = [float(episode_id * 100 + index) for index in range(row_count)]
    references = [
        (
            np.arange(CHUNK_SIZE * ACTION_DIM, dtype=np.float32).reshape(
                CHUNK_SIZE,
                ACTION_DIM,
            )
            + marker
        )
        for marker in markers
    ]
    sources = []
    for marker in markers:
        source = np.zeros((FULL_ACTION_HORIZON, 32), dtype=np.float32)
        source[:, :ACTION_DIM] = marker + 0.5
        sources.append(source)
    replay.add_episode(
        zs=[
            np.full((Z_DIM,), marker, dtype=np.float32)
            for marker in markers
        ],
        proprios=[
            np.full((ACTION_DIM,), marker, dtype=np.float32)
            for marker in markers
        ],
        references=references,
        executed=[value + 0.25 for value in references],
        rewards=[
            np.full((CHUNK_SIZE,), float(success), dtype=np.float32)
            for _ in markers
        ],
        masks=[
            np.ones((CHUNK_SIZE,), dtype=np.float32)
            for _ in markers
        ],
        external_cams=[
            np.full((2, 3, 3), int(marker) % 255, dtype=np.uint8)
            for marker in markers
        ],
        wrist_cams=[
            np.full((1, 2, 3), int(marker) % 255, dtype=np.uint8)
            for marker in markers
        ],
        instructions=[
            f"episode-{episode_id}-row-{index}"
            for index in range(row_count)
        ],
        success=success,
        gamma=0.99,
        episode_id=episode_id,
        sources_native=sources,
    )


def _chunk_transition(episode_id: int, success: bool) -> ChunkTransition:
    marker = float(episode_id)
    reference = np.full(
        (CHUNK_SIZE, ACTION_DIM),
        marker,
        dtype=np.float32,
    )
    return ChunkTransition(
        z=np.full((Z_DIM,), marker, dtype=np.float32),
        proprio=np.full((ACTION_DIM,), marker, dtype=np.float32),
        reference_actions=reference,
        executed_actions=reference + 0.25,
        rewards=np.full(
            (CHUNK_SIZE,),
            float(success),
            dtype=np.float32,
        ),
        action_mask=np.ones((CHUNK_SIZE,), dtype=np.float32),
        next_z=np.full((Z_DIM,), marker + 1.0, dtype=np.float32),
        next_proprio=np.full(
            (ACTION_DIM,),
            marker + 1.0,
            dtype=np.float32,
        ),
        next_reference_actions=reference + 1.0,
        terminal=True,
        mc_return=float(success),
        success=float(success),
        episode_id=episode_id,
        start_step=0,
    )


def _copy_npz_without(
    source: Path,
    destination: Path,
    excluded: set[str],
) -> None:
    with np.load(source, allow_pickle=False) as data:
        payload = {
            key: np.asarray(data[key]).copy()
            for key in data.files
            if key not in excluded
        }
    np.savez_compressed(destination, **payload)


def test_outcome_retention_keeps_recent_rows_and_survives_oversized_load(
    tmp_path: Path,
) -> None:
    direct = ImageChunkReplay(
        max_transitions=6,
        pos_frac=0.5,
        seed=1,
    )
    source = ImageChunkReplay(
        max_transitions=100,
        pos_frac=0.5,
        seed=1,
    )
    for episode_id in range(13):
        success = episode_id < 4
        _add_image_episode(
            direct,
            episode_id=episode_id,
            success=success,
        )
        _add_image_episode(
            source,
            episode_id=episode_id,
            success=success,
        )

    expected_ids = [1, 2, 3, 10, 11, 12]
    assert [row.episode_id for row in direct.rows] == expected_ids
    assert direct.outcome_counts() == (3, 3)
    assert direct.successful_episode_count() == 3

    path = tmp_path / "oversized_image_replay.npz"
    source.save_npz(str(path))
    restored = ImageChunkReplay.load_npz(
        str(path),
        max_transitions=6,
        pos_frac=0.5,
        seed=999,
    )
    assert [row.episode_id for row in restored.rows] == expected_ids
    assert restored.outcome_counts() == direct.outcome_counts()


@pytest.mark.parametrize(
    ("pos_frac", "expected_counts"),
    ((0.0, (1, 3)), (1.0, (3, 1))),
)
def test_retention_reserves_each_present_outcome_at_endpoints(
    pos_frac: float,
    expected_counts: tuple[int, int],
) -> None:
    replay = ImageChunkReplay(
        max_transitions=4,
        pos_frac=pos_frac,
        seed=2,
    )
    for episode_id in range(8):
        _add_image_episode(
            replay,
            episode_id=episode_id,
            success=episode_id < 4,
        )
    assert replay.outcome_counts() == expected_counts
    assert replay.has_both_outcomes()


@pytest.mark.parametrize(
    ("pos_frac", "expected_successes"),
    ((0.0, 1), (1.0, 7)),
)
def test_required_outcome_sampling_handles_fraction_endpoints(
    pos_frac: float,
    expected_successes: int,
) -> None:
    image = ImageChunkReplay(
        max_transitions=16,
        pos_frac=pos_frac,
        seed=3,
    )
    _add_image_episode(image, episode_id=0, success=False)
    _add_image_episode(image, episode_id=1, success=True)

    chunk = ChunkReplay(
        max_transitions=16,
        pos_frac=pos_frac,
        seed=3,
    )
    chunk.add(_chunk_transition(0, False))
    chunk.add(_chunk_transition(1, True))

    for replay in (image, chunk):
        batch = replay.sample(8, require_both_outcomes=True)
        assert int(batch["success"].sum().item()) == expected_successes
        with pytest.raises(ValueError, match="at least 2"):
            replay.sample(1, require_both_outcomes=True)

    one_outcome = ImageChunkReplay(max_transitions=4, seed=4)
    _add_image_episode(one_outcome, episode_id=2, success=False)
    with pytest.raises(RuntimeError, match="both outcomes"):
        one_outcome.sample(2, require_both_outcomes=True)


def test_natural_sampling_is_episode_balanced_not_outcome_stratified() -> None:
    replay = ImageChunkReplay(
        max_transitions=32,
        pos_frac=1.0,
        seed=5,
    )
    _add_image_episode(
        replay,
        episode_id=10,
        success=True,
        row_count=9,
    )
    _add_image_episode(
        replay,
        episode_id=20,
        success=False,
        row_count=1,
    )

    batch = replay.sample_natural(20)
    episode_ids = batch["episode_id"].cpu().numpy()
    assert np.count_nonzero(episode_ids == 10) == 10
    assert np.count_nonzero(episode_ids == 20) == 10
    assert int(batch["success"].sum().item()) == 10


def test_retention_is_balanced_across_outcome_and_pose_cycle() -> None:
    replay = ImageChunkReplay(
        max_transitions=8,
        pos_frac=0.5,
        seed=6,
        benchmark_pose_cycle=4,
    )
    for episode_id in range(24):
        _add_image_episode(
            replay,
            episode_id=episode_id,
            success=episode_id < 8,
        )

    positive_poses = {
        row.episode_id % 4
        for row in replay.rows
        if row.success > 0.5
    }
    negative_poses = {
        row.episode_id % 4
        for row in replay.rows
        if row.success <= 0.5
    }
    assert positive_poses == {0, 1, 2, 3}
    assert negative_poses == {0, 1, 2, 3}
    assert replay.outcome_counts() == (4, 4)


def test_native_source_and_next_source_round_trip(tmp_path: Path) -> None:
    replay = ImageChunkReplay(max_transitions=8, seed=7)
    _add_image_episode(
        replay,
        episode_id=3,
        success=True,
        row_count=2,
    )
    path = tmp_path / "native_sources.npz"
    replay.save_npz(str(path))
    restored = ImageChunkReplay.load_npz(str(path), max_transitions=8)

    assert np.array_equal(
        restored.rows[0].next_source_native,
        restored.rows[1].source_native,
    )
    assert np.array_equal(
        restored.rows[0].source_native,
        replay.rows[0].source_native,
    )
    assert restored.rows[0].source_native[:, :ACTION_DIM].any()
    assert not restored.rows[1].next_source_native.any()


def test_chunk_and_image_rng_state_round_trip_exactly(
    tmp_path: Path,
) -> None:
    chunk = ChunkReplay(max_transitions=16, pos_frac=0.35, seed=11)
    image = ImageChunkReplay(max_transitions=16, pos_frac=0.35, seed=11)
    for episode_id in range(6):
        success = episode_id % 2 == 0
        chunk.add(_chunk_transition(episode_id, success))
        _add_image_episode(
            image,
            episode_id=episode_id,
            success=success,
        )

    chunk.sample(5, require_both_outcomes=True)
    image.sample_natural(5)
    chunk_path = tmp_path / "chunk.npz"
    image_path = tmp_path / "image.npz"
    chunk.save_npz(str(chunk_path))
    image.save_npz(str(image_path))

    restored_chunk = ChunkReplay.load_npz(
        str(chunk_path),
        max_transitions=16,
        seed=999,
    )
    restored_image = ImageChunkReplay.load_npz(
        str(image_path),
        max_transitions=16,
        pos_frac=0.35,
        seed=999,
    )
    for _ in range(3):
        expected_chunk = chunk.sample(9, require_both_outcomes=True)
        actual_chunk = restored_chunk.sample(
            9,
            require_both_outcomes=True,
        )
        assert torch.equal(
            expected_chunk["episode_id"],
            actual_chunk["episode_id"],
        )
        expected_image = image.sample_natural(9)
        actual_image = restored_image.sample_natural(9)
        assert torch.equal(
            expected_image["episode_id"],
            actual_image["episode_id"],
        )

    with np.load(image_path, allow_pickle=False) as data:
        state = json.loads(str(data["rng_state_json"].item()))
    assert state["bit_generator"] == image.rng.bit_generator.__class__.__name__


def test_legacy_files_without_rng_or_full_actions_remain_loadable(
    tmp_path: Path,
) -> None:
    chunk = ChunkReplay(max_transitions=8, seed=17)
    image = ImageChunkReplay(max_transitions=8, seed=17)
    for episode_id in range(4):
        success = episode_id % 2 == 0
        chunk.add(_chunk_transition(episode_id, success))
        _add_image_episode(
            image,
            episode_id=episode_id,
            success=success,
        )

    modern_chunk = tmp_path / "modern_chunk.npz"
    modern_image = tmp_path / "modern_image.npz"
    legacy_chunk = tmp_path / "legacy_chunk.npz"
    legacy_image = tmp_path / "legacy_image.npz"
    chunk.save_npz(str(modern_chunk))
    image.save_npz(str(modern_image))
    _copy_npz_without(
        modern_chunk,
        legacy_chunk,
        {"rng_state_json", "pos_frac"},
    )
    _copy_npz_without(
        modern_image,
        legacy_image,
        {
            "rng_state_json",
            "pos_frac",
            "full_action_horizon",
            "full_reference_actions",
            "full_executed_actions",
            "next_full_reference_actions",
        },
    )

    chunk_a = ChunkReplay.load_npz(str(legacy_chunk), seed=23)
    chunk_b = ChunkReplay.load_npz(str(legacy_chunk), seed=23)
    assert torch.equal(
        chunk_a.sample(7)["episode_id"],
        chunk_b.sample(7)["episode_id"],
    )
    image_a = ImageChunkReplay.load_npz(str(legacy_image), seed=23)
    image_b = ImageChunkReplay.load_npz(str(legacy_image), seed=23)
    assert torch.equal(
        image_a.sample_natural(7)["episode_id"],
        image_b.sample_natural(7)["episode_id"],
    )
    row = image_a.rows[0]
    assert row.full_reference_actions.shape == (
        FULL_ACTION_HORIZON,
        ACTION_DIM,
    )
    assert np.array_equal(
        row.full_reference_actions[:CHUNK_SIZE],
        row.reference_actions,
    )
    assert np.all(
        row.full_reference_actions[CHUNK_SIZE:]
        == row.reference_actions[-1]
    )


def test_full_horizon_actions_next_fields_and_round_trip(
    tmp_path: Path,
) -> None:
    replay = ImageChunkReplay(max_transitions=8, seed=7)
    compact_references = [
        np.full((CHUNK_SIZE, ACTION_DIM), value, dtype=np.float32)
        for value in (1.0, 2.0)
    ]
    compact_executed = [value + 0.5 for value in compact_references]
    full_references = [
        (
            np.arange(
                FULL_ACTION_HORIZON * ACTION_DIM,
                dtype=np.float32,
            ).reshape(FULL_ACTION_HORIZON, ACTION_DIM)
            + offset
        )
        for offset in (100.0, 200.0)
    ]
    full_executed = [value + 0.5 for value in full_references]
    replay.add_episode(
        zs=[
            np.full((Z_DIM,), value, dtype=np.float32)
            for value in (1.0, 2.0)
        ],
        proprios=[
            np.full((ACTION_DIM,), value, dtype=np.float32)
            for value in (1.0, 2.0)
        ],
        references=compact_references,
        executed=compact_executed,
        rewards=[
            np.zeros((CHUNK_SIZE,), dtype=np.float32)
            for _ in range(2)
        ],
        masks=[
            np.ones((CHUNK_SIZE,), dtype=np.float32)
            for _ in range(2)
        ],
        external_cams=[
            np.full((2, 3, 3), value, dtype=np.uint8)
            for value in (1, 2)
        ],
        wrist_cams=[
            np.full((1, 2, 3), value, dtype=np.uint8)
            for value in (1, 2)
        ],
        instructions=["first", "second"],
        success=True,
        gamma=0.99,
        episode_id=30,
        full_references=full_references,
        full_executed=full_executed,
    )

    first, terminal = replay.rows
    assert first.reference_actions.shape == (
        CHUNK_SIZE,
        ACTION_DIM,
    )
    assert np.array_equal(
        first.next_full_reference_actions,
        full_references[1],
    )
    assert first.next_full_reference_actions is terminal.full_reference_actions
    assert terminal.next_full_reference_actions.shape == (
        FULL_ACTION_HORIZON,
        ACTION_DIM,
    )
    assert np.count_nonzero(terminal.next_full_reference_actions) == 0
    batch = replay.sample_natural(2)
    assert batch["full_reference_actions"].shape == (
        2,
        FULL_ACTION_HORIZON,
        ACTION_DIM,
    )
    assert batch["next_full_reference_actions"].shape == (
        2,
        FULL_ACTION_HORIZON,
        ACTION_DIM,
    )

    path = tmp_path / "full_horizon.npz"
    replay.save_npz(str(path))
    with np.load(path, allow_pickle=False) as data:
        assert int(data["full_action_horizon"]) == FULL_ACTION_HORIZON
    restored = ImageChunkReplay.load_npz(str(path), max_transitions=8)
    assert np.array_equal(
        restored.rows[0].full_reference_actions,
        full_references[0],
    )
    assert np.array_equal(
        restored.rows[0].next_full_reference_actions,
        full_references[1],
    )

    legacy_default = ImageChunkReplay(max_transitions=4)
    _add_image_episode(
        legacy_default,
        episode_id=40,
        success=False,
    )
    compact = legacy_default.rows[0].reference_actions
    expanded = legacy_default.rows[0].full_reference_actions
    assert np.array_equal(expanded[:CHUNK_SIZE], compact)
    assert np.all(expanded[CHUNK_SIZE:] == compact[-1])


def test_legacy_image_archive_defaults_full_horizon_and_seed(
    tmp_path: Path,
) -> None:
    replay = ImageChunkReplay(max_transitions=8, pos_frac=0.25, seed=13)
    _add_image_episode(
        replay,
        episode_id=45,
        success=True,
        row_count=2,
    )
    modern_path = tmp_path / "modern.npz"
    legacy_path = tmp_path / "legacy.npz"
    replay.save_npz(str(modern_path))
    omitted = {
        "full_reference_actions",
        "full_executed_actions",
        "next_full_reference_actions",
        "full_action_horizon",
        "pos_frac",
        "rng_state_json",
    }
    with np.load(modern_path, allow_pickle=False) as data:
        legacy_payload = {
            key: data[key]
            for key in data.files
            if key not in omitted
        }
    np.savez_compressed(legacy_path, **legacy_payload)

    first = ImageChunkReplay.load_npz(
        str(legacy_path),
        max_transitions=8,
        seed=17,
    )
    second = ImageChunkReplay.load_npz(
        str(legacy_path),
        max_transitions=8,
        seed=17,
    )
    assert first.full_action_horizon == FULL_ACTION_HORIZON
    assert first.pos_frac == pytest.approx(0.4)
    assert np.array_equal(
        first.rows[0].full_reference_actions[:CHUNK_SIZE],
        first.rows[0].reference_actions,
    )
    assert np.array_equal(
        first.rows[0].next_full_reference_actions[:CHUNK_SIZE],
        first.rows[0].next_reference_actions,
    )
    assert torch.equal(
        first.sample_natural(5)["episode_id"],
        second.sample_natural(5)["episode_id"],
    )


def test_storage_nbytes_deduplicates_shared_row_arrays() -> None:
    replay = ImageChunkReplay(max_transitions=8)
    _add_image_episode(
        replay,
        episode_id=50,
        success=True,
        row_count=2,
    )
    first, second = replay.rows
    assert first.next_z is second.z
    assert first.next_external_cam is second.external_cam
    assert (
        first.next_full_reference_actions
        is second.full_reference_actions
    )

    arrays = [
        value
        for row in replay.rows
        for value in vars(row).values()
        if isinstance(value, np.ndarray)
    ]
    expected = sum(
        value.nbytes
        for value in {id(value): value for value in arrays}.values()
    )
    naive = sum(value.nbytes for value in arrays)
    assert replay.storage_nbytes() == expected
    assert expected < naive
