"""V17 helpers: soft Q ramp after BC + episode-level collect utilities."""

from __future__ import annotations

from v15_helpers import (
    ActorPhaseConfig,
    EmpiricalGateTracker,
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
    bc_episodes: int = 150,
    q_ramp_episodes: int = 100,
    residual_clip: float = 0.02,
    advantage_clip: float = 0.05,
    endpoint_ref_mse_max: float = 0.01,
    deploy_ref_dropout: float = 0.5,
) -> ActorPhaseConfig:
    """V17 curriculum: BC warmup → linear q_coef ramp → full clipped Q.

    Softens the V16 cliff at ``bc_episodes`` where ``q_coef`` jumped 0→1 while
    ``always_collect_actor`` flipped on and collapsed train SR.
    """
    bc = int(max(0, bc_episodes))
    ramp = int(max(0, q_ramp_episodes))
    ep = int(valid_episodes)
    if ep < bc:
        return ActorPhaseConfig(
            q_coef=0.0,
            ref_dropout=0.0,
            residual_clip=float(residual_clip),
            advantage_clip=float(advantage_clip),
            endpoint_ref_mse_max=float(endpoint_ref_mse_max),
            phase="bc_warmup",
        )
    if ramp <= 0:
        return ActorPhaseConfig(
            q_coef=1.0,
            ref_dropout=float(deploy_ref_dropout),
            residual_clip=float(residual_clip),
            advantage_clip=float(advantage_clip),
            endpoint_ref_mse_max=float(endpoint_ref_mse_max),
            phase="clipped_q",
        )
    progress = min(1.0, float(ep - bc) / float(ramp))
    if progress < 1.0:
        return ActorPhaseConfig(
            q_coef=float(progress),
            ref_dropout=float(deploy_ref_dropout) * float(progress),
            residual_clip=float(residual_clip),
            advantage_clip=float(advantage_clip),
            endpoint_ref_mse_max=float(endpoint_ref_mse_max),
            phase="q_ramp",
        )
    return ActorPhaseConfig(
        q_coef=1.0,
        ref_dropout=float(deploy_ref_dropout),
        residual_clip=float(residual_clip),
        advantage_clip=float(advantage_clip),
        endpoint_ref_mse_max=float(endpoint_ref_mse_max),
        phase="clipped_q",
    )
