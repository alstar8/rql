"""Focused tests for AR-QDFL FastSAC experiment plumbing (non-method)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.ar_qdfl_fast_sac_plumbing import (  # noqa: E402
    PHASE_OFFLINE,
    PHASE_ONLINE,
    PHASE_WARMUP,
    OnlineTransitionJournal,
    merge_batches,
    online_replay_fraction,
    plot_step,
    sample_stratified_batch,
    should_write_eval,
    stratified_batch_counts,
)
from utils.datasets import ReplayBuffer  # noqa: E402


def _traj_batch(batch_size: int, value: float, h: int = 1):
    horizon = h + 1
    return {
        "observations": np.full((horizon, batch_size, 4), value, dtype=np.float32),
        "actions": np.full((horizon, batch_size, 2), value, dtype=np.float32),
        "rewards": np.full((horizon, batch_size), value, dtype=np.float32),
        "terminals": np.zeros((horizon, batch_size), dtype=np.float32),
        "masks": np.ones((horizon, batch_size), dtype=np.float32),
    }


class _FakeReplay:
    def __init__(self, size: int, fill: float, *, has_terminals: bool = True):
        self.size = int(size)
        self.fill = float(fill)
        self.terminal_locs = (
            np.asarray([max(size - 1, 0)], dtype=np.int64)
            if has_terminals and size > 0
            else np.asarray([], dtype=np.int64)
        )

    def sample(self, batch_size: int):
        return _traj_batch(batch_size, self.fill)


def test_online_replay_fraction_ramp_and_clip():
    assert online_replay_fraction(0, ramp_steps=100, fraction_max=0.5) == pytest.approx(
        0.0
    )
    assert online_replay_fraction(
        50, ramp_steps=100, fraction_max=0.5
    ) == pytest.approx(0.25)
    assert online_replay_fraction(
        100, ramp_steps=100, fraction_max=0.5
    ) == pytest.approx(0.5)
    assert online_replay_fraction(
        10_000, ramp_steps=100, fraction_max=0.5
    ) == pytest.approx(0.5)
    assert online_replay_fraction(
        50, ramp_steps=0, fraction_max=0.5
    ) == pytest.approx(0.5)


def test_stratified_batch_counts_respect_empty_online():
    assert stratified_batch_counts(256, 0.5, online_size=0) == (256, 0)
    assert stratified_batch_counts(256, 0.0, online_size=1000) == (256, 0)
    n_off, n_on = stratified_batch_counts(256, 0.5, online_size=10_000)
    assert n_off + n_on == 256
    assert n_on == 128
    n_off, n_on = stratified_batch_counts(10, 0.5, online_size=3)
    assert n_off + n_on == 10
    assert n_on == 3


def test_merge_and_sample_stratified_batch_shapes():
    offline = _FakeReplay(1000, 0.0)
    online = _FakeReplay(1000, 1.0)
    batch, info = sample_stratified_batch(
        offline,
        online,
        batch_size=8,
        online_env_step=100,
        ramp_steps=100,
        fraction_max=0.5,
    )
    assert info["n_offline"] == 4
    assert info["n_online"] == 4
    assert batch["observations"].shape == (2, 8, 4)
    # First half offline fill, second half online fill along batch axis.
    np.testing.assert_allclose(batch["observations"][:, :4], 0.0)
    np.testing.assert_allclose(batch["observations"][:, 4:], 1.0)

    empty_online = _FakeReplay(0, 1.0)
    batch0, info0 = sample_stratified_batch(
        offline,
        empty_online,
        batch_size=8,
        online_env_step=100,
        ramp_steps=100,
        fraction_max=0.5,
    )
    assert info0["n_online"] == 0
    assert batch0["observations"].shape == (2, 8, 4)
    np.testing.assert_allclose(batch0["observations"], 0.0)

    # Online rows exist but no finished episode yet → stay offline-only.
    early_online = _FakeReplay(50, 1.0, has_terminals=False)
    batch_early, info_early = sample_stratified_batch(
        offline,
        early_online,
        batch_size=8,
        online_env_step=100,
        ramp_steps=100,
        fraction_max=0.5,
    )
    assert info_early["n_online"] == 0
    np.testing.assert_allclose(batch_early["observations"], 0.0)

    merged = merge_batches(_traj_batch(2, 0.0), _traj_batch(3, 1.0))
    assert merged["actions"].shape == (2, 5, 2)


def test_plot_step_mapping_hides_warmup():
    assert plot_step(PHASE_OFFLINE, offline_update_count=0) == 0
    assert plot_step(PHASE_OFFLINE, offline_update_count=1_000_000) == 1_000_000
    assert (
        plot_step(PHASE_WARMUP, offline_update_count=1_000_000, online_env_step=0)
        is None
    )
    assert not should_write_eval(PHASE_WARMUP)
    assert should_write_eval(PHASE_OFFLINE)
    assert should_write_eval(PHASE_ONLINE)
    assert (
        plot_step(
            PHASE_ONLINE,
            offline_update_count=1_000_000,
            online_env_step=0,
            offline_steps=1_000_000,
        )
        == 1_000_000
    )
    assert (
        plot_step(
            PHASE_ONLINE,
            offline_update_count=1_000_000,
            online_env_step=250_000,
            offline_steps=1_000_000,
        )
        == 1_250_000
    )
    assert (
        plot_step(
            PHASE_ONLINE,
            offline_update_count=1_000_000,
            online_env_step=1_000_000,
            offline_steps=1_000_000,
        )
        == 2_000_000
    )


def test_online_transition_journal_roundtrip(tmp_path: Path):
    journal = OnlineTransitionJournal(tmp_path / "journal", shard_size=3)
    example = {
        "observations": np.zeros((4,), dtype=np.float32),
        "actions": np.zeros((2,), dtype=np.float32),
        "rewards": np.asarray(0.0, dtype=np.float32),
        "terminals": np.asarray(0.0, dtype=np.float32),
        "masks": np.asarray(1.0, dtype=np.float32),
        "next_observations": np.ones((4,), dtype=np.float32),
    }
    for i in range(7):
        transition = {
            "observations": np.full((4,), float(i), dtype=np.float32),
            "actions": np.full((2,), float(i), dtype=np.float32),
            "rewards": np.asarray(-1.0, dtype=np.float32),
            "terminals": np.asarray(1.0 if i == 6 else 0.0, dtype=np.float32),
            "masks": np.asarray(0.0 if i == 6 else 1.0, dtype=np.float32),
            "next_observations": np.full((4,), float(i + 1), dtype=np.float32),
        }
        journal.append(transition)
    journal.flush()
    assert journal.count == 7

    restored = OnlineTransitionJournal(tmp_path / "journal", shard_size=3)
    assert restored.count == 7
    replay = restored.rebuild_replay(example, max_size=16)
    assert isinstance(replay, ReplayBuffer)
    assert replay.size == 7
    np.testing.assert_allclose(replay["observations"][0], 0.0)
    np.testing.assert_allclose(replay["observations"][6], 6.0)
    np.testing.assert_allclose(replay["terminals"][6], 1.0)

    snapshot = tmp_path / "snapshot.npz"
    restored.save_snapshot(snapshot)
    with np.load(snapshot) as data:
        assert data["observations"].shape[0] == 7
        np.testing.assert_allclose(data["actions"][3], 3.0)
