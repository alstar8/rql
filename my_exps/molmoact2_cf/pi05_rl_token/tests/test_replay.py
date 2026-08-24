"""Transition assembly: discounting, episode boundaries, streaming.

These are the parts that fail silently -- a wrong terminal flag or a dangling
successor still trains, it just trains on a lie.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlt.replay import ChunkReplay, Decision, Rollout, build_transitions, close_transitions

CHUNK = 8
STRIDE = 2
STATE_DIM = 4
ACTION_DIM = 3
CHUNK_DIM = CHUNK * ACTION_DIM


def make_rollout(steps: int, terminal_step: int | None = None, stride: int = STRIDE) -> Rollout:
    """An episode where every value encodes its own step, so mixups are visible."""
    boundaries = -(-steps // CHUNK)  # committed actions are planned a whole chunk at a time
    committed = [np.full(ACTION_DIM, float(t), np.float32) for t in range(boundaries * CHUNK)]
    decisions = [
        Decision(
            step=t,
            state=np.full(STATE_DIM, float(t), np.float32),
            reference=np.full(CHUNK_DIM, -float(t), np.float32),
        )
        for t in range(0, steps, stride)
    ]
    rewards = np.zeros(steps, np.float32)
    if terminal_step is not None:
        rewards[terminal_step] = 1.0
    return Rollout(
        steps=steps,
        decisions=decisions,
        committed=committed,
        rewards=rewards,
        terminal_step=terminal_step,
        success=terminal_step is not None,
    )


def test_rows_carry_their_own_window():
    rows = build_transitions(make_rollout(64), CHUNK, gamma=0.99)
    for row in rows:
        t = int(row["state"][0])
        assert row["action"].reshape(CHUNK, ACTION_DIM)[0, 0] == pytest.approx(t)
        assert row["action"].reshape(CHUNK, ACTION_DIM)[-1, 0] == pytest.approx(t + CHUNK - 1)
        assert row["next_state"][0] == pytest.approx(t + CHUNK)


def test_truncated_tail_is_dropped_not_zero_bootstrapped():
    steps = 64  # decisions at 0..62, so the last row with a successor opens at 54
    last = (steps - STRIDE) - CHUNK
    rows = build_transitions(make_rollout(steps), CHUNK, gamma=0.99)
    assert [int(r["state"][0]) for r in rows] == list(range(0, last + 1, STRIDE))
    assert all(r["done"] == 0.0 for r in rows)
    # No surviving row may bootstrap off a zero state -- that is the failure the
    # ordering exists to prevent.
    assert all(np.any(r["next_state"] != 0) for r in rows[1:])


def test_terminal_row_stops_the_reward_sum_and_the_bootstrap():
    steps, terminal = 20, 19  # success on the last step; boundaries commit through 23
    rows = build_transitions(make_rollout(steps, terminal_step=terminal), CHUNK, gamma=0.9)
    rewarded = [r for r in rows if r["reward"] > 0]
    assert rewarded, "the success reward reached no row"
    for row in rewarded:
        t = int(row["state"][0])
        assert row["done"] == 1.0
        assert row["reward"] == pytest.approx(0.9 ** (terminal - t))
        assert np.all(row["next_state"] == 0)
        assert np.all(row["next_reference"] == 0)


def test_reward_outside_the_window_is_not_counted():
    rows = build_transitions(make_rollout(40, terminal_step=39), CHUNK, gamma=0.9)
    for row in rows:
        t = int(row["state"][0])
        assert (row["reward"] > 0) == (t + CHUNK > 39)


def test_streaming_matches_one_shot():
    """close_transitions during the episode must produce exactly the same rows."""
    finished = make_rollout(64, terminal_step=63)
    one_shot = build_transitions(finished, CHUNK, gamma=0.97)

    streamed: list[dict] = []
    first = 0
    for n in range(0, 64, STRIDE):  # as if a decision had just been recorded at step n
        partial = Rollout(
            steps=n,
            decisions=[d for d in finished.decisions if d.step < n],
            committed=finished.committed[: -(-max(n, 1) // CHUNK) * CHUNK],
            rewards=np.zeros(n, np.float32),
        )
        rows, first = close_transitions(partial, CHUNK, 0.97, first)
        streamed += rows
    rows, first = close_transitions(finished, CHUNK, 0.97, first)
    streamed += rows

    assert len(streamed) == len(one_shot)
    for a, b in zip(streamed, one_shot):
        for key in a:
            assert np.allclose(a[key], b[key]), key


def test_stride_equal_to_the_chunk_stores_whole_decisions():
    """With no subsampling every row is one decision: the action window starts
    where the plan started, so nothing is spliced from two of them."""
    rollout = make_rollout(64, stride=CHUNK)
    rows = build_transitions(rollout, CHUNK, gamma=0.99)

    assert [int(r["state"][0]) for r in rows] == list(range(0, 64 - CHUNK, CHUNK))
    for row in rows:
        t = int(row["state"][0])
        window = row["action"].reshape(CHUNK, ACTION_DIM)[:, 0]
        assert list(window) == [float(t + i) for i in range(CHUNK)]
        assert row["next_state"][0] == pytest.approx(t + CHUNK)


def test_no_decision_no_row():
    rollout = make_rollout(64)
    rollout.decisions = []
    assert build_transitions(rollout, CHUNK, 0.99) == []


def test_buffer_roundtrip_and_coverage():
    buffer = ChunkReplay(capacity=16, state_dim=STATE_DIM, chunk_dim=CHUNK_DIM, seed=0)
    rows = build_transitions(make_rollout(32, terminal_step=31), CHUNK, 0.99)
    buffer.extend(rows)
    assert len(buffer) == len(rows)
    assert buffer.reward_rows() == sum(r["reward"] > 0 for r in rows)

    batch = buffer.sample(8)
    assert batch["state"].shape == (8, STATE_DIM)
    assert batch["action"].shape == (8, CHUNK_DIM)

    # Executed actions differ from the reference here, so coverage is non-zero.
    assert buffer.action_coverage() > 0
    buffer.action[: len(buffer)] = buffer.reference[: len(buffer)]
    assert buffer.action_coverage() == 0.0
