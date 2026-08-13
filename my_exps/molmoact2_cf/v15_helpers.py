"""V15 training helpers: mixture bookkeeping, empirical gate, actor phases."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float:
    """Wilson score lower bound for a binomial proportion."""
    if n <= 0:
        return 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return max(0.0, (center - margin) / denom)


def wilson_upper(successes: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 1.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p + z2 / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
    return min(1.0, (center + margin) / denom)


def empirical_delta_lcb(
    actor_successes: int,
    actor_n: int,
    ref_successes: int,
    ref_n: int,
    *,
    z: float = 1.96,
) -> float:
    """Conservative LCB on (actor_sr - ref_sr) via Wilson bounds."""
    if actor_n <= 0 or ref_n <= 0:
        return float("-inf")
    return wilson_lower(actor_successes, actor_n, z=z) - wilson_upper(
        ref_successes, ref_n, z=z
    )


@dataclass
class EmpiricalGateTracker:
    """Tracks mixture/on-policy outcomes for return-calibrated gating."""

    actor_successes: int = 0
    actor_episodes: int = 0
    ref_successes: int = 0
    ref_episodes: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self, *, used_actor: bool, success: bool) -> None:
        if used_actor:
            self.actor_episodes += 1
            self.actor_successes += int(bool(success))
        else:
            self.ref_episodes += 1
            self.ref_successes += int(bool(success))
        self.history.append(
            {
                "used_actor": bool(used_actor),
                "success": bool(success),
            }
        )

    @property
    def empirical_lcb(self) -> float:
        return empirical_delta_lcb(
            self.actor_successes,
            self.actor_episodes,
            self.ref_successes,
            self.ref_episodes,
        )

    def metrics(self) -> dict[str, float]:
        actor_sr = (
            self.actor_successes / self.actor_episodes
            if self.actor_episodes
            else 0.0
        )
        ref_sr = (
            self.ref_successes / self.ref_episodes if self.ref_episodes else 0.0
        )
        lcb = self.empirical_lcb
        return {
            "empirical_actor_episodes": float(self.actor_episodes),
            "empirical_ref_episodes": float(self.ref_episodes),
            "empirical_actor_sr": float(actor_sr),
            "empirical_ref_sr": float(ref_sr),
            "empirical_lcb": float(lcb) if math.isfinite(lcb) else -1.0,
            "empirical_ready": float(
                self.actor_episodes > 0 and self.ref_episodes > 0
            ),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "actor_successes": self.actor_successes,
            "actor_episodes": self.actor_episodes,
            "ref_successes": self.ref_successes,
            "ref_episodes": self.ref_episodes,
        }

    def load_state_dict(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        self.actor_successes = int(payload.get("actor_successes", 0))
        self.actor_episodes = int(payload.get("actor_episodes", 0))
        self.ref_successes = int(payload.get("ref_successes", 0))
        self.ref_episodes = int(payload.get("ref_episodes", 0))


@dataclass(frozen=True)
class ActorPhaseConfig:
    q_coef: float
    ref_dropout: float
    residual_clip: float | None
    advantage_clip: float | None
    endpoint_ref_mse_max: float | None
    phase: str


def actor_phase_for_episode(
    valid_episodes: int,
    *,
    bc_episodes: int = 50,
    residual_clip: float = 0.02,
    advantage_clip: float = 0.05,
    endpoint_ref_mse_max: float = 0.01,
    deploy_ref_dropout: float = 0.0,
) -> ActorPhaseConfig:
    """V15 curriculum: BC-only warmup, then clipped local Q refinement."""
    if valid_episodes < int(bc_episodes):
        return ActorPhaseConfig(
            q_coef=0.0,
            ref_dropout=0.0,
            residual_clip=float(residual_clip),
            advantage_clip=float(advantage_clip),
            endpoint_ref_mse_max=float(endpoint_ref_mse_max),
            phase="bc_warmup",
        )
    return ActorPhaseConfig(
        q_coef=1.0,
        ref_dropout=float(deploy_ref_dropout),
        residual_clip=float(residual_clip),
        advantage_clip=float(advantage_clip),
        endpoint_ref_mse_max=float(endpoint_ref_mse_max),
        phase="clipped_q",
    )
