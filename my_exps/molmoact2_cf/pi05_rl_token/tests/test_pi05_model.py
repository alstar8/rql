"""The action path: how a chunk is unpacked and how a delta becomes an env command.

These pin the two things that are invisible from a training curve and only show up as a
wrong success rate hours later: the delta-to-absolute conversion, and which arm state
each action of a chunk is resolved against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05.model import (  # noqa: E402
    ACTION_DIM,
    ARM_DOF,
    GRIPPER_CMD_MAX,
    ChunkExecutor,
    to_absolute,
    to_env_action,
)

HORIZON = 16


def make_chunk(horizon: int = HORIZON, value: float = 0.01) -> np.ndarray:
    """A chunk whose row k holds arm deltas of (k+1)*value and a closed gripper."""
    chunk = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    for k in range(horizon):
        chunk[k, :ARM_DOF] = (k + 1) * value
        chunk[k, ARM_DOF] = 1.0
    return chunk


# --- one action ------------------------------------------------------------------------


def test_to_absolute_adds_the_arm_state_and_leaves_the_gripper():
    action = np.array([0.1] * ARM_DOF + [0.8], dtype=np.float32)
    state = np.arange(ARM_DOF, dtype=np.float32)
    out = to_absolute(action, state)
    assert np.allclose(out[:ARM_DOF], state + 0.1)
    assert out[ARM_DOF] == pytest.approx(0.8)


def test_to_absolute_rejects_a_wrong_arm_width():
    with pytest.raises(ValueError):
        to_absolute(np.zeros(ACTION_DIM), np.zeros(3))


def test_env_action_binarises_the_gripper():
    state = np.zeros(ARM_DOF, dtype=np.float32)
    closed = to_env_action(np.array([0.0] * ARM_DOF + [0.9]), state)
    opened = to_env_action(np.array([0.0] * ARM_DOF + [0.1]), state)
    assert closed["gripper"][0] == GRIPPER_CMD_MAX
    assert opened["gripper"][0] == 0.0


def test_env_action_can_pass_the_gripper_through_continuously():
    state = np.zeros(ARM_DOF, dtype=np.float32)
    action = np.array([0.0] * ARM_DOF + [0.4], dtype=np.float32)
    out = to_env_action(action, state, grasping="continuous")
    assert out["gripper"][0] == pytest.approx(0.4 * GRIPPER_CMD_MAX)


def test_env_action_arm_is_absolute_not_delta():
    state = np.full(ARM_DOF, 1.5, dtype=np.float32)
    action = np.array([0.02] * ARM_DOF + [0.0], dtype=np.float32)
    out = to_env_action(action, state)
    # The bug this guards: returning 0.02 instead of 1.52 folds the arm every step.
    assert np.allclose(out["arm"], 1.52)


# --- spending a chunk -------------------------------------------------------------------


def test_executor_asks_for_a_chunk_then_spends_exactly_chunk_size():
    executor = ChunkExecutor(chunk_size=4)
    assert executor.needs_new_chunk
    executor.load(make_chunk(), np.zeros(ARM_DOF))
    for _ in range(4):
        assert not executor.needs_new_chunk
        executor.next_action(np.zeros(ARM_DOF))
    assert executor.needs_new_chunk


def test_executor_hands_out_the_chunk_in_order():
    executor = ChunkExecutor(chunk_size=3)
    executor.load(make_chunk(), np.zeros(ARM_DOF))
    first = executor.next_action(np.zeros(ARM_DOF))
    second = executor.next_action(np.zeros(ARM_DOF))
    assert first[0] == pytest.approx(0.01)
    assert second[0] == pytest.approx(0.02)


def test_executor_refuses_a_chunk_size_longer_than_the_prediction():
    executor = ChunkExecutor(chunk_size=20)
    with pytest.raises(ValueError, match="exceeds"):
        executor.load(make_chunk(horizon=16), np.zeros(ARM_DOF))


def test_executor_refuses_to_overrun_a_spent_chunk():
    executor = ChunkExecutor(chunk_size=1)
    executor.load(make_chunk(), np.zeros(ARM_DOF))
    executor.next_action(np.zeros(ARM_DOF))
    with pytest.raises(RuntimeError, match="spent"):
        executor.next_action(np.zeros(ARM_DOF))


def test_executor_needs_a_chunk_before_acting():
    with pytest.raises(RuntimeError, match="no chunk loaded"):
        ChunkExecutor().next_action(np.zeros(ARM_DOF))


def test_executor_accepts_a_batched_chunk():
    executor = ChunkExecutor(chunk_size=1)
    executor.load(make_chunk()[None, ...], np.zeros(ARM_DOF))
    assert executor.next_action(np.zeros(ARM_DOF)).shape == (ACTION_DIM,)


def test_unknown_conversion_is_rejected_at_construction():
    with pytest.raises(ValueError, match="plan_time"):
        ChunkExecutor(1, conversion="whenever")


# --- the two conversion regimes ---------------------------------------------------------


def test_plan_time_resolves_every_action_against_the_planning_state():
    executor = ChunkExecutor(chunk_size=3, conversion="plan_time")
    plan_state = np.full(ARM_DOF, 1.0, dtype=np.float32)
    executor.load(make_chunk(), plan_state)

    # The arm moves while the chunk is being spent; plan_time must ignore that.
    moved = [np.full(ARM_DOF, 1.0 + 0.5 * k, dtype=np.float32) for k in range(3)]
    out = [executor.next_action(moved[k]) for k in range(3)]

    for k in range(3):
        assert np.allclose(out[k][:ARM_DOF], 1.0 + (k + 1) * 0.01)


def test_step_time_resolves_each_action_against_the_state_of_its_own_step():
    executor = ChunkExecutor(chunk_size=3, conversion="step_time")
    executor.load(make_chunk(), np.full(ARM_DOF, 1.0, dtype=np.float32))

    moved = [np.full(ARM_DOF, 1.0 + 0.5 * k, dtype=np.float32) for k in range(3)]
    out = [executor.next_action(moved[k]) for k in range(3)]

    for k in range(3):
        assert np.allclose(out[k][:ARM_DOF], (1.0 + 0.5 * k) + (k + 1) * 0.01)


def test_the_two_regimes_agree_at_chunk_size_one():
    """The whole measured baseline lives at chunk_size 1, where the choice cannot matter:
    planning and execution happen on the same step, against the same state."""
    chunk = make_chunk()
    state = np.full(ARM_DOF, 0.3, dtype=np.float32)

    plan = ChunkExecutor(1, "plan_time")
    step = ChunkExecutor(1, "step_time")
    plan.load(chunk, state)
    step.load(chunk, state)
    assert np.allclose(plan.next_action(state), step.next_action(state))


def test_plan_time_reproduces_the_old_server_side_conversion():
    """Before the refactor the server added the arm state to the whole chunk and the
    client executed the result untouched. plan_time must be numerically identical, or
    every previously measured success rate stops describing this code."""
    chunk = make_chunk()
    plan_state = np.full(ARM_DOF, 0.7, dtype=np.float32)
    chunk_size = 8

    # The old path: convert once, server-side, then execute.
    old = chunk.copy()
    old[:, :ARM_DOF] += plan_state
    old_actions = [old[k] for k in range(chunk_size)]

    executor = ChunkExecutor(chunk_size, "plan_time")
    executor.load(chunk, plan_state)
    # The arm really does move between steps; the old path ignored that too.
    new_actions = [
        executor.next_action(plan_state + 0.1 * k) for k in range(chunk_size)
    ]

    for old_action, new_action in zip(old_actions, new_actions):
        assert np.allclose(old_action, new_action)
