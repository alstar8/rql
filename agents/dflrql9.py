import jax
import jax.numpy as jnp

from agents.dflrql8 import DFLRQL8Agent, get_config as get_v8_config


class DFLRQL9Agent(DFLRQL8Agent):
    """DFL-RQL v9: trust-weighted conflict and near-hard scale-free consensus.

    V8 improved the 50-task aggregate (+0.07–0.08 vs baseline) with much lower
    seed variance, but diagnostics showed two transferable weaknesses:

    1. Soft consensus with floor c=0.1 systematically lowered ||W||
       (~0.50–0.58 vs v6 ~0.60–0.69), attenuating useful late guidance.
    2. Hard behavior-conflict projection always removed the anti-BC
       component.  That protects immature critics, but also blocks mature
       ensemble consensus from performing real policy improvement — the
       failure mode behind several v8 regressions vs v7
       (puzzle-4x4-task3, humanoidmaze-large-task3).

    V9 keeps v8's scale-free soft consensus and behavior-aware dynamics, with
    two task-agnostic fixes:

    A) Near-hard relative floor ``consensus_floor=0.01``.  Locally flat
       gradients still vanish; typical-batch targets stay near unit norm, so
       ||W|| recovers the v6 agreement signal without an absolute tau.

    B) Trust-weighted conflict projection.  Let u = Proj_{||.||<=1}(W) and
       v_hat be the unit behavior velocity.  With trust = ||u|| in [0, 1],

           kill = (1 - trust^p) * min(<u, v_hat>, 0)
           u_conf = u - kill * v_hat

       Immature critics (small ||u||) remain BC-safe as in v8; fully agreed
       ensembles (||u|| -> 1) may override BC exactly as in v6.  Default
       ``conflict_power=2`` makes the gate conservative until agreement is
       strong.

    C) Light residual damping when BC and consensus already agree:

           damp = 1 - rho * max(cos(u, v), 0) * trust
           u_safe = damp * Proj(u_conf)

       with small ``residual_coef`` (default 0.25).  This avoids double-pushing
       along an already-correct behavior direction without introducing env
       schedules or privileged eval signals.

    Shared RQL hyperparameters stay identical to the baseline comparison.
    """

    def _behavior_safe_direction(self, w, behavior_velocity):
        """Trust-weighted conflict projection with residual damping."""
        w = self._project_unit_ball(w)
        w_norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
        trust = jax.lax.stop_gradient(w_norm)

        behavior_velocity = jax.lax.stop_gradient(behavior_velocity)
        velocity_norm = jnp.linalg.norm(
            behavior_velocity, axis=-1, keepdims=True
        )
        velocity_unit = behavior_velocity / jnp.maximum(velocity_norm, 1e-6)
        parallel = (w * velocity_unit).sum(axis=-1, keepdims=True)

        power = self.config["conflict_power"]
        kill_frac = 1.0 - jnp.power(jnp.clip(trust, 0.0, 1.0), power)
        conflicting_parallel = jnp.minimum(parallel, 0.0)
        conflict_free = w - kill_frac * conflicting_parallel * velocity_unit
        conflict_free = self._project_unit_ball(conflict_free)

        alignment_cos = parallel / (w_norm + 1e-6)
        residual_coef = self.config["residual_coef"]
        damp = 1.0 - residual_coef * jnp.maximum(alignment_cos, 0.0) * trust
        damp = jnp.clip(damp, 0.0, 1.0)
        safe_w = self._project_unit_ball(conflict_free * damp)

        safe_norm = jnp.linalg.norm(safe_w, axis=-1, keepdims=True)
        diagnostics = {
            "behavior_alignment_cos": alignment_cos,
            "behavior_conflict": (parallel < 0.0).astype(w.dtype),
            "conflict_kill_frac": kill_frac,
            "guidance_retained": safe_norm / (w_norm + 1e-6),
            "residual_damp": damp,
            "safe_w_norm": safe_norm,
            "trust": trust,
        }
        return safe_w, diagnostics


def get_config():
    config = get_v8_config()
    config.agent_name = "dflrql9"
    # Near-hard relative floor: restores v6-like magnitude while still
    # suppressing batch-relative flat gradients.
    config.consensus_floor = 0.01
    # Quadratic trust: only strongly agreed ensembles may override BC.
    config.conflict_power = 2.0
    # Mild damping when BC and consensus already point the same way.
    config.residual_coef = 0.25
    return config
