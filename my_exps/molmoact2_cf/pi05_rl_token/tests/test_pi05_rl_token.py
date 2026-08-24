"""The gate and the correction: when RL drives, and in which space it refines.

No torch and no simulator here -- the agent and encoder are stubs, so what is under test
is the wiring, which is the part that has been wrong before.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05.config import ACTION_DIM, ARM_DOF, PROPRIO_DIM  # noqa: E402
from pi05.rl_token import RLTokenCorrector, StepGate  # noqa: E402

Z_DIM = 5
CHUNK_SIZE = 2


class StubEncoder:
    z_dim = Z_DIM

    def encode(self, tokens, mask):
        return np.full(Z_DIM, float(np.mean(tokens)), dtype=np.float32)


class StubAgent:
    """Returns a fixed chunk, and remembers what it was asked."""

    def __init__(self, reply=None):
        self.reply = reply
        self.calls = []

    def act(self, state, reference, explore):
        self.calls.append((np.asarray(state), np.asarray(reference), explore))
        return self.reply if self.reply is not None else np.asarray(reference) + 1.0


def make_corrector(action_space="absolute", gate_step=40, agent=None):
    return RLTokenCorrector(
        agent or StubAgent(),
        StubEncoder(),
        CHUNK_SIZE,
        StepGate(gate_step),
        action_space=action_space,
    )


def make_chunk(horizon=16, value=0.01):
    chunk = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    for k in range(horizon):
        chunk[k, :ARM_DOF] = (k + 1) * value
        chunk[k, ARM_DOF] = 1.0
    return chunk


TOKENS = np.ones((968, 2048), dtype=np.float32)
MASK = np.ones(968, dtype=np.float32)
PROPRIO = np.arange(PROPRIO_DIM, dtype=np.float32)
ARM = np.full(ARM_DOF, 0.5, dtype=np.float32)


# --- the gate ---------------------------------------------------------------------------


def test_latched_gate_opens_at_its_step_and_stays_open():
    gate = StepGate(40, latch=True)
    assert not gate.is_open(39)
    assert gate.is_open(40)
    assert gate.is_open(499)


def test_unlatched_gate_is_a_single_step():
    gate = StepGate(40, latch=False)
    assert not gate.is_open(39)
    assert gate.is_open(40)
    assert not gate.is_open(41)


def test_gate_zero_means_rl_drives_the_whole_episode():
    assert StepGate(0).is_open(0)


def test_gate_mask_matches_is_open():
    gate = StepGate(3)
    mask = gate.mask(6)
    assert list(mask) == [gate.is_open(t) for t in range(6)]


def test_negative_gate_step_is_rejected():
    with pytest.raises(ValueError):
        StepGate(-1)


# --- when the correction applies ---------------------------------------------------------


def test_a_closed_gate_leaves_the_vla_chunk_untouched():
    agent = StubAgent()
    corrector = make_corrector(agent=agent)
    chunk = make_chunk()
    out = corrector.correct(
        chunk=chunk, tokens=TOKENS, mask=MASK, step=10, proprio=PROPRIO, arm_state=ARM
    )
    assert np.allclose(out, chunk)
    assert agent.calls == []


def test_an_open_gate_replaces_the_chunk():
    corrector = make_corrector()
    chunk = make_chunk()
    out = corrector.correct(
        chunk=chunk, tokens=TOKENS, mask=MASK, step=40, proprio=PROPRIO, arm_state=ARM
    )
    assert out.shape == (CHUNK_SIZE, ACTION_DIM)
    assert not np.allclose(out, chunk[:CHUNK_SIZE])
    assert corrector.corrections == 1


def test_warmup_records_the_decision_but_does_not_act():
    agent = StubAgent()
    corrector = make_corrector(agent=agent)
    chunk = make_chunk()
    out = corrector.correct(
        chunk=chunk, tokens=TOKENS, mask=MASK, step=40, proprio=PROPRIO,
        arm_state=ARM, use_actor=False,
    )
    assert np.allclose(out, chunk)
    assert agent.calls == []
    # The point of warmup: the buffer still fills.
    assert corrector.last_state is not None
    assert corrector.last_reference is not None


def test_the_decision_is_recorded_even_before_the_gate_opens():
    corrector = make_corrector()
    corrector.correct(
        chunk=make_chunk(), tokens=TOKENS, mask=MASK, step=5, proprio=PROPRIO, arm_state=ARM
    )
    assert corrector.last_state.shape == (Z_DIM + PROPRIO_DIM,)
    assert corrector.last_reference.shape == (CHUNK_SIZE * ACTION_DIM,)


def test_missing_tokens_before_the_gate_are_fine():
    corrector = make_corrector()
    chunk = make_chunk()
    out = corrector.correct(
        chunk=chunk, tokens=None, mask=None, step=5, proprio=PROPRIO, arm_state=ARM
    )
    assert np.allclose(out, chunk)
    assert corrector.last_state is None


def test_missing_tokens_after_the_gate_is_a_loud_failure():
    corrector = make_corrector()
    with pytest.raises(RuntimeError, match="no tokens"):
        corrector.correct(
            chunk=make_chunk(), tokens=None, mask=None, step=40,
            proprio=PROPRIO, arm_state=ARM,
        )


def test_the_rl_state_is_the_encoder_output_followed_by_proprioception():
    corrector = make_corrector()
    state = corrector.rl_state(TOKENS, MASK, PROPRIO)
    assert state.shape == (Z_DIM + PROPRIO_DIM,)
    assert np.allclose(state[Z_DIM:], PROPRIO)


def test_wrong_proprio_width_is_rejected():
    corrector = make_corrector()
    with pytest.raises(ValueError, match="proprio"):
        corrector.rl_state(TOKENS, MASK, np.zeros(3))


# --- which space the actor refines -------------------------------------------------------


def test_absolute_space_shows_the_actor_absolute_joint_targets():
    corrector = make_corrector("absolute")
    reference = corrector.reference_from_chunk(make_chunk(), ARM).reshape(CHUNK_SIZE, ACTION_DIM)
    assert np.allclose(reference[0, :ARM_DOF], ARM + 0.01)
    assert np.allclose(reference[1, :ARM_DOF], ARM + 0.02)


def test_delta_space_shows_the_actor_the_raw_deltas():
    corrector = make_corrector("delta")
    reference = corrector.reference_from_chunk(make_chunk(), ARM).reshape(CHUNK_SIZE, ACTION_DIM)
    assert np.allclose(reference[0, :ARM_DOF], 0.01)
    assert np.allclose(reference[1, :ARM_DOF], 0.02)


@pytest.mark.parametrize("space", ["absolute", "delta"])
def test_the_space_round_trip_is_exact(space):
    """What leaves the corrector is always deltas, because the executor converts. An
    inexact round trip would shift the actor's intent by the arm state."""
    corrector = make_corrector(space)
    chunk = make_chunk()
    reference = corrector.reference_from_chunk(chunk, ARM)
    back = corrector.chunk_from_action(reference, ARM)
    assert np.allclose(back, chunk[:CHUNK_SIZE])


def test_an_unchanged_actor_reproduces_the_vla_chunk_exactly():
    """If the actor returns the reference it was given, the executed chunk must equal the
    VLA's own -- otherwise the actor starts from a handicap that is pure plumbing."""
    reference_holder = {}

    class Echo(StubAgent):
        def act(self, state, reference, explore):
            reference_holder["value"] = reference
            return reference

    corrector = make_corrector("absolute", agent=Echo())
    chunk = make_chunk()
    out = corrector.correct(
        chunk=chunk, tokens=TOKENS, mask=MASK, step=40, proprio=PROPRIO, arm_state=ARM
    )
    assert np.allclose(out, chunk[:CHUNK_SIZE])


def test_an_unknown_action_space_is_rejected():
    with pytest.raises(ValueError, match="action_space"):
        RLTokenCorrector(StubAgent(), StubEncoder(), 1, StepGate(0), action_space="joint")
