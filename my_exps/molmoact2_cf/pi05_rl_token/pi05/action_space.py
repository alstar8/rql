"""The single definition of what a pi0.5 action means for MolmoSpaces Pick.

Training targets and the serving conversion both come from this module, because the
two drifting apart is invisible from the training curve: the loss falls either way and
only the success rate, measured hours later, reveals the mismatch.

The contract, established by measurement (scripts/inspect_demos.py, probe_pi05_action_scale.py):

    state   [7 arm joint positions, gripper opening scaled to 0..1]
    action  [7 joint deltas (commanded target minus current state), gripper 0..1]

The environment executes *absolute* joint targets, so serving adds the current arm
state back before handing the action to MolmoSpaces.

Why deltas rather than the absolute targets the environment consumes: the pretrained
pi05_droid checkpoint emits deltas. Probed with a real Pick state it returned +0.27 for
joint 4 while that joint sits at -2.09 rad, and its action norm stats are zero-centred
with std ~0.1. Those stats match the demonstrations' deltas (mean 0.027/0.017/0.012 on
joints 2/4/6 against DROID's 0.024/0.018/0.011) and match nothing about their absolute
targets. Training on deltas therefore keeps the pretrained action prior usable.
"""

from __future__ import annotations

import numpy as np

# Gripper joint opening that the eval client treats as fully open. Taken from
# molmo_spaces/policy/learned_policy/pi_policy.py so state matches at eval time.
GRIPPER_QPOS_OPEN = 0.824033

# The environment's gripper command range; demonstrations already record 0 or 255.
GRIPPER_CMD_MAX = 255.0

ARM_DOF = 7
ACTION_DIM = 8
STATE_DIM = 8


def state_from_qpos(qpos: dict) -> np.ndarray:
    """Observation the model conditions on: 7 arm joints plus normalised gripper."""
    arm = np.asarray(qpos["arm"][:ARM_DOF], dtype=np.float32)
    grip = np.clip(float(qpos["gripper"][0]) / GRIPPER_QPOS_OPEN, 0.0, 1.0)
    return np.concatenate([arm, np.asarray([grip], dtype=np.float32)])


def action_from_command(command: dict, qpos: dict) -> np.ndarray:
    """Training target for one step: joint delta plus normalised gripper command.

    ``command`` is a demonstration's ``actions/joint_pos`` row, which holds the absolute
    joint target the environment executed. Subtracting the state at the same step gives
    the delta the pretrained model speaks in.
    """
    target = np.asarray(command["arm"][:ARM_DOF], dtype=np.float32)
    current = np.asarray(qpos["arm"][:ARM_DOF], dtype=np.float32)
    grip = float(command["gripper"][0]) / GRIPPER_CMD_MAX
    return np.concatenate([target - current, np.asarray([grip], dtype=np.float32)]).astype(np.float32)


def absolute_arm_from_delta(delta: np.ndarray, current_arm: np.ndarray) -> np.ndarray:
    """Invert :func:`action_from_command` for the arm: the environment wants absolutes."""
    return np.asarray(delta[:ARM_DOF], dtype=np.float32) + np.asarray(current_arm[:ARM_DOF], dtype=np.float32)


def model_output_to_env_action(
    model_output: np.ndarray,
    current_arm: np.ndarray,
    *,
    grasping_type: str = "continuous",
    grasping_threshold: float = 0.5,
) -> dict:
    """Turn one predicted 8-vector into the ``{"arm", "gripper"}`` MolmoSpaces expects.

    Mirrors the gripper handling in molmo_spaces' pi_policy so a fine-tuned model can be
    served to the stock eval client without touching it.
    """
    arm = absolute_arm_from_delta(np.asarray(model_output), current_arm)
    grip_raw = float(model_output[ARM_DOF])
    if grasping_type == "continuous":
        gripper = np.asarray([grip_raw * GRIPPER_CMD_MAX], dtype=np.float32)
    else:
        gripper = np.asarray(
            [GRIPPER_CMD_MAX if grip_raw > grasping_threshold else 0.0], dtype=np.float32
        )
    return {"arm": arm.astype(np.float32), "gripper": gripper}


def normalize_instruction(text: str) -> str:
    """Match the eval benchmark's wording exactly: lowercase, one trailing period.

    All 1000 benchmark episodes read ``"pick up the <noun>."`` while the demonstrations
    record ``"Pick up the <referral>"``. The eval client lowercases before sending, so
    case and the final period are the parts we can align; the referral wording itself is
    left alone because there is no reliable mapping to the benchmark's plain nouns.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return cleaned
    return cleaned if cleaned.endswith(".") else cleaned + "."
