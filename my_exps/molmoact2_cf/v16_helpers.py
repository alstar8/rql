"""V16 helpers: reuse V15 empirical/phase utilities with paper-aligned defaults."""

from __future__ import annotations

from v15_helpers import (
    ActorPhaseConfig,
    EmpiricalGateTracker,
    actor_phase_for_episode as _v15_actor_phase_for_episode,
    empirical_delta_lcb,
    wilson_lower,
    wilson_upper,
)

__all__ = [
    "ActorPhaseConfig",
    "EmpiricalGateTracker",
    "actor_phase_for_episode",
    "empirical_delta_lcb",
    "wilson_lower",
    "wilson_upper",
]


def actor_phase_for_episode(
    valid_episodes: int,
    *,
    bc_episodes: int = 50,
    residual_clip: float = 0.02,
    advantage_clip: float = 0.05,
    endpoint_ref_mse_max: float = 0.01,
    deploy_ref_dropout: float = 0.5,
) -> ActorPhaseConfig:
    """V16 curriculum: BC warmup, then Q + soft-β with paper reference dropout."""
    return _v15_actor_phase_for_episode(
        valid_episodes,
        bc_episodes=bc_episodes,
        residual_clip=residual_clip,
        advantage_clip=advantage_clip,
        endpoint_ref_mse_max=endpoint_ref_mse_max,
        deploy_ref_dropout=deploy_ref_dropout,
    )
