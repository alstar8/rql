"""The proximity measure that will gate RL, tested on the part that fails silently.

Mixing the robot-base frame with world coordinates, or reading the quaternion in the
wrong order, both produce distances that vary plausibly over an episode while being
completely wrong. Only physics catches it, so the conventions are pinned here.
"""

import numpy as np
import pytest

from pi05.proximity import (
    first_entry,
    gripper_object_distance,
    tcp_to_world,
    within,
)


def pose(x, y, z, quat=(1.0, 0.0, 0.0, 0.0)):
    return np.array([[x, y, z, *quat]], dtype=np.float64)


def test_identity_base_leaves_tcp_untouched():
    world = tcp_to_world(pose(0.4, 0.1, 0.3), pose(0, 0, 0))
    np.testing.assert_allclose(world[0], [0.4, 0.1, 0.3], atol=1e-9)


def test_base_translation_is_added():
    world = tcp_to_world(pose(0.4, 0.0, 0.0), pose(5.0, 3.0, 1.0))
    np.testing.assert_allclose(world[0], [5.4, 3.0, 1.0], atol=1e-9)


def test_quaternion_is_read_as_wxyz():
    # 90 degrees about z as [w, x, y, z]: a TCP one metre ahead maps onto +y.
    q = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    world = tcp_to_world(pose(1.0, 0.0, 0.0), pose(0, 0, 0, q))
    np.testing.assert_allclose(world[0], [0.0, 1.0, 0.0], atol=1e-9)


def test_reading_the_quaternion_as_xyzw_would_give_a_different_answer():
    # Guards the specific mistake: the same numbers read in the other order rotate
    # differently, which on real data moved the grasp distance from 5 cm to 1.7 m.
    q_wxyz = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    correct = tcp_to_world(pose(1.0, 0.0, 0.0), pose(0, 0, 0, q_wxyz))
    swapped = tcp_to_world(pose(1.0, 0.0, 0.0), pose(0, 0, 0, tuple(np.roll(q_wxyz, -1))))
    assert not np.allclose(correct, swapped, atol=1e-6)


def test_distance_is_zero_when_gripper_sits_on_the_object():
    d = gripper_object_distance(pose(0.4, 0, 0), pose(5, 3, 1), pose(5.4, 3, 1))
    assert d[0] == pytest.approx(0.0, abs=1e-9)


def test_distance_uses_world_frame_not_raw_subtraction():
    # Raw subtraction of a base-frame TCP from a world-frame object is the bug this
    # module exists to prevent; it would report ~5 m where the true distance is 0.
    tcp, base, obj = pose(0.4, 0, 0), pose(5, 3, 1), pose(5.4, 3, 1)
    raw = np.linalg.norm(tcp[:, :3] - obj[:, :3])
    assert raw > 4.0
    assert gripper_object_distance(tcp, base, obj)[0] < 1e-6


def test_distance_is_computed_per_step():
    tcp = np.repeat(pose(0.4, 0, 0), 3, axis=0)
    base = np.repeat(pose(0, 0, 0), 3, axis=0)
    obj = np.array(
        [[0.4, 0, 0, 1, 0, 0, 0], [0.5, 0, 0, 1, 0, 0, 0], [0.9, 0, 0, 1, 0, 0, 0]],
        dtype=np.float64,
    )
    np.testing.assert_allclose(gripper_object_distance(tcp, base, obj), [0.0, 0.1, 0.5], atol=1e-9)


def test_first_entry_reports_the_handover_step():
    d = np.array([0.40, 0.30, 0.12, 0.08, 0.05, 0.09])
    assert first_entry(d, 0.10) == 3


def test_first_entry_is_none_when_never_close():
    assert first_entry(np.array([0.4, 0.3, 0.25]), 0.10) is None


def test_within_marks_every_close_step_not_only_the_first():
    d = np.array([0.40, 0.08, 0.20, 0.05])
    np.testing.assert_array_equal(within(d, 0.10), [False, True, False, True])


def test_threshold_is_inclusive():
    assert within(np.array([0.10]), 0.10)[0]


def test_latched_gate_stays_open_after_the_first_approach():
    from pi05.proximity import gate_mask

    # Dips under 0.10 at step 2, drifts back out, then returns. Latched, control is
    # handed over once and kept.
    d = np.array([0.40, 0.30, 0.08, 0.25, 0.30, 0.06])
    np.testing.assert_array_equal(
        gate_mask(d, 0.10, latch=True), [False, False, True, True, True, True]
    )


def test_unlatched_gate_follows_the_distance():
    from pi05.proximity import gate_mask

    d = np.array([0.40, 0.30, 0.08, 0.25, 0.30, 0.06])
    np.testing.assert_array_equal(
        gate_mask(d, 0.10, latch=False), [False, False, True, False, False, True]
    )


def test_gate_never_opens_when_the_arm_stays_far():
    from pi05.proximity import gate_mask

    # The house-2 failure: closest approach 0.230 m, so RL is never handed control.
    d = np.array([0.31, 0.28, 0.23, 0.26, 0.30])
    assert not gate_mask(d, 0.10, latch=True).any()


def test_default_threshold_is_the_measured_one():
    from pi05.proximity import GATE_THRESHOLD_M

    assert GATE_THRESHOLD_M == 0.10


def test_step_gate_opens_at_the_chosen_step_and_stays_open():
    from pi05.proximity import step_gate_mask

    mask = step_gate_mask(6, start_step=3)
    np.testing.assert_array_equal(mask, [False, False, False, True, True, True])


def test_step_gate_never_opens_for_a_shorter_episode():
    from pi05.proximity import step_gate_mask

    # An episode that ends before the handover step hands over nothing.
    assert not step_gate_mask(3, start_step=40).any()


def test_step_gate_default_is_the_watched_value():
    from pi05.proximity import GATE_STEP

    assert GATE_STEP == 40


def test_step_gate_unlatched_marks_only_that_step():
    from pi05.proximity import step_gate_mask

    np.testing.assert_array_equal(
        step_gate_mask(5, start_step=2, latch=False), [False, False, True, False, False]
    )
