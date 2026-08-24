"""The delta-to-absolute step at serving time.

If this is wrong the arm is commanded to a nonsense pose every step while the model
itself is fine. That is the failure already visible in the stock pi0.5 baseline, and no
training metric would reveal it.
"""

import numpy as np
import pytest

from pi05.serve import DeltaToAbsolutePolicy


class StubPolicy:
    """Returns a fixed action chunk, ignoring the observation."""

    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.float32)
        self.seen = None
        self.was_reset = False

    def infer(self, obs):
        self.seen = obs
        return {"actions": self.actions.copy(), "extra": "kept"}

    def reset(self):
        self.was_reset = True


ARM = np.array([0.0, -0.5, 0.1, -2.0, 0.0, 2.0, 0.3], dtype=np.float32)


def _obs(arm=ARM):
    return {"observation/joint_position": arm, "observation/gripper_position": np.array([0.2])}


def test_delta_is_added_to_the_current_arm_state():
    chunk = np.zeros((4, 8), dtype=np.float32)
    chunk[:, :7] = 0.01
    out = DeltaToAbsolutePolicy(StubPolicy(chunk)).infer(_obs())
    for k in range(4):
        np.testing.assert_allclose(out["actions"][k, :7], ARM + 0.01, atol=1e-6)


def test_zero_delta_holds_position_instead_of_folding_the_arm():
    # The stock server returns near-zero deltas; read as absolute they collapse the arm.
    # After conversion a zero prediction has to mean stay put.
    out = DeltaToAbsolutePolicy(StubPolicy(np.zeros((3, 8), dtype=np.float32))).infer(_obs())
    for k in range(3):
        np.testing.assert_allclose(out["actions"][k, :7], ARM, atol=1e-7)


def test_gripper_channel_is_left_alone():
    chunk = np.zeros((2, 8), dtype=np.float32)
    chunk[:, 7] = 0.75
    out = DeltaToAbsolutePolicy(StubPolicy(chunk)).infer(_obs())
    np.testing.assert_allclose(out["actions"][:, 7], 0.75, atol=1e-6)


def test_each_chunk_element_uses_the_same_base_state():
    # Only the state at chunk start is known, which is why eval must use chunk_size=1.
    chunk = np.zeros((3, 8), dtype=np.float32)
    chunk[:, 3] = [0.1, 0.2, 0.3]
    out = DeltaToAbsolutePolicy(StubPolicy(chunk)).infer(_obs())
    np.testing.assert_allclose(out["actions"][:, 3], ARM[3] + np.array([0.1, 0.2, 0.3]), atol=1e-6)


def test_missing_state_is_an_error_not_a_silent_passthrough():
    with pytest.raises(KeyError, match="joint_position"):
        DeltaToAbsolutePolicy(StubPolicy(np.zeros((2, 8)))).infer(
            {"observation/gripper_position": 0.1}
        )


def test_other_result_fields_survive():
    out = DeltaToAbsolutePolicy(StubPolicy(np.zeros((2, 8), dtype=np.float32))).infer(_obs())
    assert out["extra"] == "kept"


def test_the_wrapped_policy_sees_the_untouched_observation():
    stub = StubPolicy(np.zeros((2, 8), dtype=np.float32))
    DeltaToAbsolutePolicy(stub).infer(_obs())
    np.testing.assert_array_equal(stub.seen["observation/joint_position"], ARM)


def test_reset_is_forwarded():
    stub = StubPolicy(np.zeros((2, 8), dtype=np.float32))
    DeltaToAbsolutePolicy(stub).reset()
    assert stub.was_reset
