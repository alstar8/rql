"""Distance from the gripper to the pickup object, for gating RL on proximity.

The two poses MolmoSpaces records live in different frames, and mixing them up is silent:
``obs/extra/tcp_pose`` is in the robot's base frame while ``obs/extra/obj_start`` is in
world coordinates. Subtracting them directly gives 5-12 metres on episodes where the arm
demonstrably grasped the object, and the number still varies plausibly over time, so
nothing looks broken.

Transforming the TCP into world coordinates with ``obs/extra/robot_base_pose`` gives
distances that match the physics: about 0.4 m at episode start, falling to 0.5-5 cm at
the moment of a successful grasp.

The stored quaternion is [w, x, y, z] (MuJoCo order). That was determined by measurement,
not assumed -- reading it as [x, y, z, w] yields 1.6-1.8 m at the grasp instead, which is
wrong but not obviously so.
"""

from __future__ import annotations

import numpy as np


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices for quaternions stored as [w, x, y, z], shape (n, 4) -> (n, 3, 3)."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def tcp_to_world(tcp_pose: np.ndarray, robot_base_pose: np.ndarray) -> np.ndarray:
    """Gripper position in world coordinates, shape (n, 3).

    Both inputs are (n, 7) as recorded: position followed by a [w, x, y, z] quaternion.
    """
    tcp_pose = np.atleast_2d(np.asarray(tcp_pose, dtype=np.float64))
    robot_base_pose = np.atleast_2d(np.asarray(robot_base_pose, dtype=np.float64))
    rotation = _quat_wxyz_to_matrix(robot_base_pose[:, 3:7])
    local = tcp_pose[:, :3]
    rotated = np.einsum("nij,nj->ni", rotation, local)
    return robot_base_pose[:, :3] + rotated


def gripper_object_distance(
    tcp_pose: np.ndarray, robot_base_pose: np.ndarray, object_pose: np.ndarray
) -> np.ndarray:
    """Euclidean distance from the gripper to the object centre, per step, in metres."""
    world = tcp_to_world(tcp_pose, robot_base_pose)
    obj = np.atleast_2d(np.asarray(object_pose, dtype=np.float64))[:, :3]
    return np.linalg.norm(world - obj, axis=1)


def first_entry(distance: np.ndarray, threshold: float) -> int | None:
    """Step at which the gripper first comes within ``threshold`` metres, if ever.

    This is the moment RL would take over under a proximity gate, so it is reported
    explicitly rather than left implicit in a boolean mask.
    """
    below = np.flatnonzero(np.asarray(distance) <= threshold)
    return int(below[0]) if below.size else None


def within(distance: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of steps considered close enough to hand over to RL."""
    return np.asarray(distance) <= threshold


# The gate the RL phase uses. 0.10 m was chosen from measured episodes: the gripper
# starts 0.31-0.45 m away and reaches 0.005-0.065 m at a successful grasp, while the
# house-2 failure never came closer than 0.230 m.
GATE_THRESHOLD_M = 0.10


def gate_mask(distance, threshold: float = GATE_THRESHOLD_M, *, latch: bool = True):
    """Steps on which RL controls the arm.

    With ``latch`` the gate opens the first time the gripper comes within ``threshold``
    and stays open for the rest of the episode, which is the behaviour in the paper:
    control is handed over once, not traded back and forth. Without it the gate follows
    the distance step by step, so a policy that drifts away briefly would bounce between
    controllers in the middle of a reach.
    """
    distance = np.asarray(distance)
    close = distance <= threshold
    if not latch:
        return close
    start = first_entry(distance, threshold)
    mask = np.zeros_like(close, dtype=bool)
    if start is not None:
        mask[start:] = True
    return mask


# Step at which RL takes over, chosen by watching rollouts of the toilet scene rather
# than computed. A distance gate is not usable here: the evaluator records the object's
# spec pose, not its live position, so once the policy nudges the object the number stops
# describing reality -- and on an elongated object the centre stays far while the gripper
# is already at a graspable end.
GATE_STEP = 40


def step_gate_mask(n_steps: int, start_step: int = GATE_STEP, *, latch: bool = True):
    """Steps on which RL controls the arm, gated by step index.

    Latched by default, matching the paper: control is handed over once and kept. The
    unlatched form exists only for symmetry with the distance gate and marks the single
    step, which is unlikely to be what anyone wants.
    """
    mask = np.zeros(int(n_steps), dtype=bool)
    if start_step < n_steps:
        if latch:
            mask[int(start_step):] = True
        else:
            mask[int(start_step)] = True
    return mask
