"""The action contract, tested on the parts that fail quietly.

A wrong delta sign or gripper scale still trains to a falling loss; only these checks
and a full eval would ever notice.
"""

import numpy as np
import pytest

from pi05.action_space import (
    GRIPPER_CMD_MAX,
    GRIPPER_QPOS_OPEN,
    absolute_arm_from_delta,
    action_from_command,
    model_output_to_env_action,
    normalize_instruction,
    state_from_qpos,
)


def _qpos(arm, grip=0.0):
    return {"arm": list(arm), "base": [], "gripper": [grip, grip]}


def test_state_places_gripper_last_and_normalises_it():
    arm = [0.1, -0.2, 0.3, -2.0, 0.5, 2.0, 0.7]
    state = state_from_qpos(_qpos(arm, GRIPPER_QPOS_OPEN))
    assert state.shape == (8,)
    np.testing.assert_allclose(state[:7], arm, rtol=1e-6)
    assert state[7] == pytest.approx(1.0)


def test_gripper_normalisation_clips_above_open():
    state = state_from_qpos(_qpos([0] * 7, GRIPPER_QPOS_OPEN * 2))
    assert state[7] == pytest.approx(1.0)


def test_action_is_the_delta_from_the_state_at_the_same_step():
    arm = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.0])
    target = arm + np.array([0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 0.0])
    action = action_from_command({"arm": list(target), "gripper": [GRIPPER_CMD_MAX]}, _qpos(arm))
    np.testing.assert_allclose(action[:7], target - arm, atol=1e-6)
    assert action[7] == pytest.approx(1.0)


def test_delta_round_trips_back_to_the_absolute_target():
    arm = np.array([0.2, -0.7, 0.1, -2.3, 0.4, 1.9, -0.3])
    target = np.array([0.21, -0.68, 0.13, -2.25, 0.38, 1.95, -0.28])
    action = action_from_command({"arm": list(target), "gripper": [0.0]}, _qpos(arm))
    np.testing.assert_allclose(absolute_arm_from_delta(action, arm), target, atol=1e-6)


def test_env_action_rebuilds_absolute_targets_and_rescales_the_gripper():
    arm = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.0])
    output = np.concatenate([np.full(7, 0.01), [0.75]])
    action = model_output_to_env_action(output, arm)
    np.testing.assert_allclose(action["arm"], arm + 0.01, atol=1e-6)
    assert action["gripper"][0] == pytest.approx(0.75 * GRIPPER_CMD_MAX)


def test_binary_grasping_snaps_to_the_environment_limits():
    arm = np.zeros(7)
    closed = model_output_to_env_action(
        np.concatenate([np.zeros(7), [0.9]]), arm, grasping_type="binary"
    )
    opened = model_output_to_env_action(
        np.concatenate([np.zeros(7), [0.1]]), arm, grasping_type="binary"
    )
    assert closed["gripper"][0] == GRIPPER_CMD_MAX
    assert opened["gripper"][0] == 0.0


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Pick up the green cone", "pick up the green cone."),
        ("pick up the kettle.", "pick up the kettle."),
        ("  Pick up the MUG  ", "pick up the mug."),
    ],
)
def test_instruction_matches_the_benchmark_wording(raw, expected):
    assert normalize_instruction(raw) == expected


def test_zero_delta_leaves_the_arm_where_it_is():
    # A model that predicts nothing must hold position, not fly to the origin. This is
    # the failure the stock eval bridge shows when it reads deltas as absolute targets.
    arm = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.0])
    action = model_output_to_env_action(np.zeros(8), arm)
    np.testing.assert_allclose(action["arm"], arm, atol=1e-7)
