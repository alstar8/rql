"""What the training policy records, and what the learner makes of it.

The rollout recorder is the one place where a bug is completely silent: the episode still
runs, the loss still falls, and only the success rate -- hours later -- says anything is
wrong. So the invariants are pinned here without a simulator or a GPU, by driving the
same recording logic `Pi05RLPolicy.on_plan` uses and handing the result to the real
`rlt.replay` code.

Two alignments matter and are easy to get wrong:

* `committed` is indexed by env step, and every plan extends it by exactly chunk_size --
  including plans before the gate, which record no decision at all. If a pre-gate plan
  skipped the extend, every later transition would read actions from the wrong steps.
* a row closes only once the *next* decision exists, so its reward window covers the
  whole chunk rather than the single step that had executed when the decision fired.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05.config import ACTION_DIM, ARM_DOF, PROPRIO_DIM  # noqa: E402
from pi05.rl_token import RLTokenCorrector, StepGate  # noqa: E402
from rlt.replay import Decision, Rollout, close_transitions  # noqa: E402

Z_DIM = 4
CHUNK = 2
GATE = 4  # opens at env step 4, i.e. at the third plan when CHUNK == 2

TOKENS = np.ones((968, 2048), dtype=np.float32)
MASK = np.ones(968, dtype=np.float32)
PROPRIO = np.arange(PROPRIO_DIM, dtype=np.float32)
ARM = np.full(ARM_DOF, 0.5, dtype=np.float32)


class StubEncoder:
    z_dim = Z_DIM

    def encode(self, tokens, mask):
        return np.full(Z_DIM, float(np.mean(tokens)), dtype=np.float32)


class ShiftAgent:
    """A visible correction: the actor's chunk is the reference plus one."""

    def act(self, state, reference, explore):
        return np.asarray(reference, dtype=np.float32) + 1.0


def vla_chunk(step: int) -> np.ndarray:
    """A chunk whose values identify the step it was planned at."""
    chunk = np.zeros((16, ACTION_DIM), dtype=np.float32)
    chunk[:, :ARM_DOF] = 0.01 * (step + 1)
    chunk[:, ARM_DOF] = 1.0
    return chunk


def record_episode(steps: int, *, use_actor: bool, store_pre_gate: bool):
    """Drive the recording logic exactly as Pi05RLPolicy.on_plan does."""
    corrector = RLTokenCorrector(
        ShiftAgent(), StubEncoder(), CHUNK, StepGate(GATE), action_space="delta"
    )
    decisions: list[Decision] = []
    committed: list[np.ndarray] = []

    for step in range(0, steps, CHUNK):
        # The policy only pays for tokens when something will read them.
        want = store_pre_gate or corrector.gate.is_open(step)
        chunk = corrector.correct(
            chunk=vla_chunk(step),
            tokens=TOKENS if want else None,
            mask=MASK if want else None,
            step=step,
            proprio=PROPRIO,
            arm_state=ARM,
            use_actor=use_actor,
        )
        if corrector.last_state is None:
            # No RL state for this plan: the actions still execute, nothing is stored.
            committed.extend(np.asarray(chunk, dtype=np.float32)[:CHUNK])
            continue
        rows = corrector.reference_from_chunk(chunk, ARM)
        decisions.append(
            Decision(step=step, state=corrector.last_state, reference=corrector.last_reference)
        )
        committed.extend(rows.reshape(CHUNK, -1))
    return corrector, decisions, committed


def make_rollout(steps, decisions, committed, terminal_step=None):
    rewards = np.zeros(steps, dtype=np.float32)
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


# --- committed stays aligned with env steps ------------------------------------------


def test_every_plan_extends_committed_by_exactly_chunk_size():
    _, _, committed = record_episode(12, use_actor=True, store_pre_gate=False)
    assert len(committed) == 12


def test_pre_gate_plans_execute_but_record_no_decision():
    """With store_pre_gate off, the first GATE steps run the VLA and are not stored."""
    _, decisions, committed = record_episode(12, use_actor=True, store_pre_gate=False)
    assert len(committed) == 12
    assert [d.step for d in decisions] == [4, 6, 8, 10]


def test_storing_pre_gate_records_every_plan():
    _, decisions, _ = record_episode(12, use_actor=True, store_pre_gate=True)
    assert [d.step for d in decisions] == [0, 2, 4, 6, 8, 10]


def test_committed_rows_still_line_up_with_their_env_steps():
    """The gap left by unstored pre-gate plans must not shift later actions."""
    _, decisions, committed = record_episode(12, use_actor=True, store_pre_gate=False)
    first = decisions[0]
    # The decision at step 4 was planned from vla_chunk(4), whose arm value is 0.05,
    # and the actor adds 1.0 on top.
    assert np.allclose(committed[first.step][:ARM_DOF], 0.05 + 1.0)


# --- transitions ----------------------------------------------------------------------


def test_a_row_closes_only_when_the_next_decision_exists():
    _, decisions, committed = record_episode(12, use_actor=True, store_pre_gate=False)
    # Truncate to the state right after the first stored decision fired.
    partial = make_rollout(5, decisions[:1], committed[:6])
    rows, first = close_transitions(partial, CHUNK, 0.99, 0)
    assert rows == [] and first == 0


def test_the_reward_window_covers_the_whole_chunk_not_one_step():
    _, decisions, committed = record_episode(12, use_actor=True, store_pre_gate=False)
    # Success on step 5, the second step of the chunk opened by the decision at step 4.
    rollout = make_rollout(12, decisions, committed, terminal_step=5)
    rows, _ = close_transitions(rollout, CHUNK, 1.0, 0)
    assert rows, "the terminal row should close"
    assert rows[0]["reward"] == pytest.approx(1.0)
    assert rows[0]["done"] == pytest.approx(1.0)


def test_a_stored_action_matches_what_was_committed_for_those_steps():
    _, decisions, committed = record_episode(12, use_actor=True, store_pre_gate=False)
    rollout = make_rollout(12, decisions, committed)
    rows, _ = close_transitions(rollout, CHUNK, 0.99, 0)
    assert rows
    row, decision = rows[0], decisions[0]
    expected = np.asarray(committed[decision.step : decision.step + CHUNK]).reshape(-1)
    assert np.allclose(row["action"], expected)


def test_during_warmup_the_stored_action_equals_the_reference():
    """Warmup is the frozen VLA measured in place, so the buffer must record it as such:
    a row whose action differs from the reference would credit RL for the VLA's work."""
    _, decisions, committed = record_episode(12, use_actor=False, store_pre_gate=True)
    rollout = make_rollout(12, decisions, committed)
    rows, _ = close_transitions(rollout, CHUNK, 0.99, 0)
    assert rows
    for row in rows:
        assert np.allclose(row["action"], row["reference"])


def test_after_the_gate_the_stored_action_differs_from_the_reference():
    _, decisions, committed = record_episode(12, use_actor=True, store_pre_gate=False)
    rollout = make_rollout(12, decisions, committed)
    rows, _ = close_transitions(rollout, CHUNK, 0.99, 0)
    assert rows
    assert not np.allclose(rows[0]["action"], rows[0]["reference"])


def test_a_truncated_tail_is_dropped_rather_than_bootstrapped_off_nothing():
    """The last chunk of a horizon-truncated episode has no successor state. Storing it
    would bootstrap the critic off a zero state that never occurred."""
    _, decisions, committed = record_episode(12, use_actor=True, store_pre_gate=False)
    rollout = make_rollout(12, decisions, committed)  # no terminal step: truncated
    rows, _ = close_transitions(rollout, CHUNK, 0.99, 0)
    assert len(rows) == len(decisions) - 1


def test_state_and_reference_widths_match_what_the_agent_is_built_for():
    """pi05/train.py sizes the networks as encoder.z_dim + PROPRIO_DIM and
    chunk_size * ACTION_DIM. A mismatch here is a shape error deep inside the learner."""
    _, decisions, _ = record_episode(12, use_actor=True, store_pre_gate=True)
    assert decisions[0].state.shape == (Z_DIM + PROPRIO_DIM,)
    assert decisions[0].reference.shape == (CHUNK * ACTION_DIM,)
