"""Tests for Monte Carlo return-to-go and trajectory-success dataset helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.datasets import (  # noqa: E402
    Dataset,
    compute_discounted_mc_returns,
    compute_episode_success_flags,
)


def test_compute_mc_returns_single_episode_discount():
    rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    terminals = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    gamma = 0.9
    g = compute_discounted_mc_returns(rewards, terminals, gamma)
    assert g.dtype == np.float32
    assert g.shape == (3,)
    expected2 = 3.0
    expected1 = 2.0 + gamma * expected2
    expected0 = 1.0 + gamma * expected1
    np.testing.assert_allclose(g, [expected0, expected1, expected2], atol=1e-6)


def test_compute_mc_returns_respects_terminal_boundaries():
    # Two episodes: [r=1, r=10(term)], [r=2, r=20(term)]
    rewards = np.array([1.0, 10.0, 2.0, 20.0], dtype=np.float32)
    terminals = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    gamma = 0.5
    g = compute_discounted_mc_returns(rewards, terminals, gamma)
    # Episode 0
    assert float(g[1]) == pytest.approx(10.0)
    assert float(g[0]) == pytest.approx(1.0 + 0.5 * 10.0)
    # Episode 1 must not leak episode-0 return
    assert float(g[3]) == pytest.approx(20.0)
    assert float(g[2]) == pytest.approx(2.0 + 0.5 * 20.0)


def test_compute_mc_returns_final_transition_and_gamma_zero():
    rewards = np.array([5.0], dtype=np.float32)
    terminals = np.array([1.0], dtype=np.float32)
    g = compute_discounted_mc_returns(rewards, terminals, discount=0.99)
    np.testing.assert_allclose(g, [5.0])

    rewards2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    terminals2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    g0 = compute_discounted_mc_returns(rewards2, terminals2, discount=0.0)
    np.testing.assert_allclose(g0, rewards2)


def test_compute_mc_returns_empty_and_shape_errors():
    empty = compute_discounted_mc_returns(
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
        0.99,
    )
    assert empty.shape == (0,)
    assert empty.dtype == np.float32

    with pytest.raises(ValueError, match="1-D"):
        compute_discounted_mc_returns(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            0.9,
        )
    with pytest.raises(ValueError, match="shape"):
        compute_discounted_mc_returns(
            np.zeros((3,), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
            0.9,
        )


def test_sample_traj_includes_mc_returns_only_when_present():
    n = 12
    terminals = np.zeros(n, dtype=np.float32)
    terminals[5] = 1.0
    terminals[11] = 1.0
    rewards = np.arange(n, dtype=np.float32)
    mc = compute_discounted_mc_returns(rewards, terminals, discount=0.9)

    base = dict(
        observations=np.zeros((n, 3), dtype=np.float32),
        actions=np.zeros((n, 2), dtype=np.float32),
        rewards=rewards,
        terminals=terminals,
        masks=1.0 - terminals,
        next_observations=np.zeros((n, 3), dtype=np.float32),
    )
    ds_plain = Dataset.create(**base)
    ds_plain.config = {"h": 1}
    batch_plain = ds_plain.sample_traj(4)
    assert "mc_returns" not in batch_plain
    assert batch_plain["rewards"].shape[0] == 2  # h+1

    ds_mc = Dataset.create(**{**base, "mc_returns": mc})
    ds_mc.config = {"h": 1}
    batch_mc = ds_mc.sample_traj(4)
    assert "mc_returns" in batch_mc
    assert batch_mc["mc_returns"].shape == batch_mc["rewards"].shape
    assert batch_mc["mc_returns"].shape == (2, 4)


def test_compute_episode_success_flags_across_episodes():
    # Ep0: success mid-episode → all 1s including before success.
    # Ep1: no success → all 0s.
    # Ep2: success at terminal → all 1s.
    successes = np.array([0, 1, 0, 0, 0, 0, 0, 1], dtype=np.float32)
    terminals = np.array([0, 0, 1, 0, 0, 1, 0, 1], dtype=np.float32)
    flags = compute_episode_success_flags(successes, terminals)
    assert flags.dtype == np.float32
    np.testing.assert_array_equal(
        flags, np.array([1, 1, 1, 0, 0, 0, 1, 1], dtype=np.float32)
    )


def test_compute_episode_success_before_and_at_terminal():
    # Success before terminal marks early steps.
    succ = np.array([0, 1, 0], dtype=bool)
    term = np.array([0, 0, 1], dtype=np.float32)
    flags = compute_episode_success_flags(succ, term)
    np.testing.assert_array_equal(flags, [1.0, 1.0, 1.0])

    # Success exactly at terminal.
    succ2 = np.array([0, 0, 1], dtype=bool)
    flags2 = compute_episode_success_flags(succ2, term)
    np.testing.assert_array_equal(flags2, [1.0, 1.0, 1.0])


def test_compute_episode_success_no_success_and_final_unterminated():
    succ = np.array([0, 0, 0], dtype=np.float32)
    term = np.array([0, 0, 1], dtype=np.float32)
    flags = compute_episode_success_flags(succ, term)
    np.testing.assert_array_equal(flags, [0.0, 0.0, 0.0])

    # Final episode has no terminal but contains a success.
    succ2 = np.array([0, 0, 1, 0, 1], dtype=bool)
    term2 = np.array([0, 1, 0, 0, 0], dtype=np.float32)
    flags2 = compute_episode_success_flags(succ2, term2)
    np.testing.assert_array_equal(flags2, [0.0, 0.0, 1.0, 1.0, 1.0])


def test_compute_episode_success_empty_and_shape_errors():
    empty = compute_episode_success_flags(
        np.zeros((0,), dtype=bool),
        np.zeros((0,), dtype=np.float32),
    )
    assert empty.shape == (0,)
    assert empty.dtype == np.float32

    with pytest.raises(ValueError, match="1-D"):
        compute_episode_success_flags(
            np.zeros((2, 2), dtype=bool),
            np.zeros((2, 2), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="shape"):
        compute_episode_success_flags(
            np.zeros((3,), dtype=bool),
            np.zeros((2,), dtype=np.float32),
        )


def test_sample_traj_includes_trajectory_success_only_when_present():
    n = 12
    terminals = np.zeros(n, dtype=np.float32)
    terminals[5] = 1.0
    terminals[11] = 1.0
    # masks<=0 on one step in ep0 only → whole ep0 flagged.
    masks = np.ones(n, dtype=np.float32)
    masks[2] = 0.0
    step_succ = masks <= 0
    traj = compute_episode_success_flags(step_succ, terminals)

    base = dict(
        observations=np.zeros((n, 3), dtype=np.float32),
        actions=np.zeros((n, 2), dtype=np.float32),
        rewards=np.arange(n, dtype=np.float32),
        terminals=terminals,
        masks=masks,
        next_observations=np.zeros((n, 3), dtype=np.float32),
    )
    ds_plain = Dataset.create(**base)
    ds_plain.config = {"h": 1}
    batch_plain = ds_plain.sample_traj(4)
    assert "trajectory_success" not in batch_plain

    ds_ts = Dataset.create(**{**base, "trajectory_success": traj})
    ds_ts.config = {"h": 1}
    batch_ts = ds_ts.sample_traj(4)
    assert "trajectory_success" in batch_ts
    assert batch_ts["trajectory_success"].shape == batch_ts["rewards"].shape
    assert batch_ts["trajectory_success"].shape == (2, 4)
    assert batch_ts["trajectory_success"].dtype == np.float32
