"""Unit tests for V17 collect / phase helpers (no GPU)."""

from __future__ import annotations

from v17_helpers import actor_phase_for_episode


def test_q_ramp_curriculum() -> None:
    warm = actor_phase_for_episode(10, bc_episodes=150, q_ramp_episodes=100)
    assert warm.phase == "bc_warmup"
    assert warm.q_coef == 0.0
    mid = actor_phase_for_episode(200, bc_episodes=150, q_ramp_episodes=100)
    assert mid.phase == "q_ramp"
    assert abs(mid.q_coef - 0.5) < 1e-9
    full = actor_phase_for_episode(260, bc_episodes=150, q_ramp_episodes=100)
    assert full.phase == "clipped_q"
    assert full.q_coef == 1.0


def test_zero_ramp_matches_cliff() -> None:
    phase = actor_phase_for_episode(50, bc_episodes=50, q_ramp_episodes=0)
    assert phase.phase == "clipped_q"
    assert phase.q_coef == 1.0
