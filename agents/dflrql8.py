from functools import partial

import jax
import jax.numpy as jnp
from einops import rearrange, repeat

from agents.dflrql6 import DFLRQL6Agent, get_config as get_v6_config


class DFLRQL8Agent(DFLRQL6Agent):
    """DFL-RQL v8: scale-free soft consensus with behavior-safe guidance.

    V8 keeps v6's ensemble-consensus guidance, removes v7's value-uncertainty
    critic calls from flow integration, and makes two task-agnostic changes:

    1. Scale-invariant soft consensus.  For a sampled ensemble member gradient
       g_k, the distillation target is

           g_k / (||g_k|| + c * mean_batch(||g_k||)).

       The dimensionless floor c suppresses gradients that are locally flat
       relative to the current batch while remaining invariant to reward/Q
       rescaling.  Averaging these bounded targets still makes ||W|| a joint
       measure of gradient strength and ensemble directional agreement.

    2. Behavior-conflict projection.  Let u be the projected guidance head and
       v the actor flow velocity at the same (s, x, f).  Before applying u, V8
       projects it onto the half-space that cannot oppose v:

           u_safe = u - min(<u, v_hat>, 0) * v_hat.

       This is the minimum-L2 change that removes only the component fighting
       the behavior flow.  Orthogonal and behavior-aligned critic improvement
       components are preserved, avoiding a global attenuation schedule.

    The same safe guided dynamics are used for critic reversal, actor q_pe
    lookahead, and inference.
    """

    def _project_unit_ball(self, w):
        w_norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
        return w * jnp.minimum(1.0, 1.0 / (w_norm + 1e-6))

    def _behavior_safe_direction(self, w, behavior_velocity):
        """Project guidance so it has no component opposing behavior flow."""
        w = self._project_unit_ball(w)
        behavior_velocity = jax.lax.stop_gradient(behavior_velocity)
        velocity_norm = jnp.linalg.norm(
            behavior_velocity, axis=-1, keepdims=True
        )
        velocity_unit = behavior_velocity / jnp.maximum(velocity_norm, 1e-6)
        parallel = (w * velocity_unit).sum(axis=-1, keepdims=True)
        conflicting_parallel = jnp.minimum(parallel, 0.0)
        safe_w = w - conflicting_parallel * velocity_unit
        safe_w = self._project_unit_ball(safe_w)

        w_norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
        safe_norm = jnp.linalg.norm(safe_w, axis=-1, keepdims=True)
        diagnostics = {
            "behavior_alignment_cos": parallel / (w_norm + 1e-6),
            "behavior_conflict": (parallel < 0.0).astype(w.dtype),
            "guidance_retained": safe_norm / (w_norm + 1e-6),
            "safe_w_norm": safe_norm,
        }
        return safe_w, diagnostics

    def guidance_direction(
        self,
        observations,
        actions,
        times,
        behavior_velocity,
    ):
        """Return time-gated consensus guidance after conflict projection."""
        w = self.network.select("guidance")(
            jnp.concatenate([observations, actions, times], axis=-1)
        )
        safe_w, _ = self._behavior_safe_direction(w, behavior_velocity)
        return self.config["guidance_coef"] * times * safe_w

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        batch_size = self.config["batch_size"]
        action_dim = self.config["action_dim"]

        rng, n_rng, u_rng, r_rng = jax.random.split(rng, 4)

        next_state = jnp.concatenate(
            [
                batch["observations"][-1],
                jax.random.normal(n_rng, (batch_size, action_dim)),
                jnp.zeros((batch_size, 1)),
            ],
            axis=-1,
        )
        next_qs = self.network.select("target_value")(next_state)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        d = jnp.concatenate(
            [
                jax.random.uniform(u_rng, (batch_size // 2,)),
                jax.random.randint(
                    r_rng,
                    (batch_size // 2,),
                    0,
                    self.config["flow_steps"] + 1,
                )
                / self.config["flow_steps"],
            ],
            axis=0,
        )
        d_b = d / self.config["flow_steps"]

        actions = rearrange(
            batch["actions"][: self.config["h"]], "h b d -> b (h d)"
        )

        # Reverse exactly the behavior-safe dynamics used at inference.
        x_f = jnp.copy(actions)
        f = jnp.ones((batch_size, 1))
        for _ in range(self.config["flow_steps"]):
            fm_actor = jnp.concatenate(
                [batch["observations"][0], x_f, f], axis=-1
            )
            behavior_velocity = self.network.select("actor")(fm_actor).mode()
            guidance = self.guidance_direction(
                batch["observations"][0],
                x_f,
                f,
                behavior_velocity,
            )
            x_f = x_f - (behavior_velocity + guidance) * d_b[..., None]
            f = f - d_b[..., None]

        state = jnp.concatenate(
            [batch["observations"][0], jax.lax.stop_gradient(x_f), f],
            axis=-1,
        )
        q = self.network.select("value")(state, params=grad_params)

        rs_terminals = jnp.concatenate(
            [
                jnp.zeros_like(batch["terminals"][:1]),
                batch["terminals"][:-1],
            ],
            axis=0,
        )
        n_rews = (
            batch["rewards"]
            * self.config["discount_mul"][..., None]
            * (1 - rs_terminals)
        ).sum(0)
        target_q = (
            n_rews
            + self.config["discount"] ** self.config["h"]
            * next_q
            * batch["masks"][-2]
        )

        terminal_count = rs_terminals.sum(0)
        valids = (terminal_count <= 1).astype(terminal_count.dtype)
        critic_loss = (
            self.expectile_loss(
                target_q - q,
                target_q - q,
                self.config["expectile"],
            )
            * valids
        ).mean()

        # Behavior flow loss and one-step policy-improvement lookahead.
        rng, x_rng, t_rng, k_rng = jax.random.split(rng, 4)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = rearrange(
            batch["actions"][: self.config["h"]], "h b d -> b (h d)"
        )
        t = jax.random.uniform(t_rng, (batch_size, 1))

        x_t = (1 - t) * x_0 + t * x_1
        target_velocity = x_1 - x_0
        fm_actor = jnp.concatenate(
            [batch["observations"][0], x_t, t], axis=-1
        )
        behavior_velocity = self.network.select("actor")(
            fm_actor, params=grad_params
        ).mode()
        guided_velocity = behavior_velocity + jax.lax.stop_gradient(
            self.guidance_direction(
                batch["observations"][0],
                x_t,
                t,
                behavior_velocity,
            )
        )
        q_pe = self.network.select("value")(
            jnp.concatenate(
                [
                    batch["observations"][0],
                    x_t
                    + guided_velocity
                    * jnp.minimum(
                        1 / self.config["flow_steps"],
                        1 - t,
                    ),
                    jnp.clip(
                        t + 1 / self.config["flow_steps"],
                        max=1,
                    ),
                ],
                axis=-1,
            )
        )
        q_pe = q_pe.mean(axis=0)

        ac_mask = repeat(
            1 - rs_terminals[:-1],
            "h b -> b (h r)",
            r=self.config["action_dim"] // self.config["h"],
        )
        bc_loss = (jnp.square(behavior_velocity - target_velocity) * ac_mask).mean()
        actor_loss = -(q_pe * valids).mean()

        # Scale-invariant soft ensemble-consensus distillation.
        member = jax.random.randint(
            k_rng,
            (batch_size,),
            0,
            self.config["ensemble_ct"],
        )
        member_onehot = jax.nn.one_hot(member, self.config["ensemble_ct"])

        def q_member_sum(x):
            q_in = jnp.concatenate(
                [batch["observations"][0], x, t], axis=-1
            )
            qs = self.network.select("target_value")(q_in)
            return (qs * member_onehot.T).sum()

        q_grad = jax.lax.stop_gradient(jax.grad(q_member_sum)(x_t))
        q_grad_norm = jnp.linalg.norm(q_grad, axis=-1, keepdims=True)
        valid_count = valids.sum() + 1e-6
        grad_scale = jax.lax.stop_gradient(
            (q_grad_norm[..., 0] * valids).sum() / valid_count
        )
        relative_floor = self.config["consensus_floor"] * grad_scale
        consensus_target = q_grad / (
            q_grad_norm + relative_floor + 1e-6
        )

        w = self.network.select("guidance")(
            jnp.concatenate(
                [batch["observations"][0], x_t, t],
                axis=-1,
            ),
            params=grad_params,
        )
        distill_loss = (
            jnp.square(w - consensus_target).sum(axis=-1) * valids
        ).mean()

        safe_w, safety = self._behavior_safe_direction(w, behavior_velocity)
        unit_q_grad = q_grad / (q_grad_norm + 1e-6)
        w_norm_per = jnp.linalg.norm(w, axis=-1)
        w_grad_cos_per = (w * unit_q_grad).sum(axis=-1) / (
            w_norm_per + 1e-6
        )

        total_loss = (
            actor_loss
            + bc_loss * self.config["alpha"]
            + critic_loss
            + distill_loss * self.config["distill_coef"]
        )

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "q": q.mean(),
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "q_pe_mean": q_pe.mean(),
            "q_pe_max": q_pe.max(),
            "q_pe_min": q_pe.min(),
            "distill_loss": distill_loss,
            "q_grad_norm": q_grad_norm.mean(),
            "q_grad_scale": grad_scale,
            "consensus_target_norm": jnp.linalg.norm(
                consensus_target, axis=-1
            ).mean(),
            "w_norm": w_norm_per.mean(),
            "w_norm_max": w_norm_per.max(),
            "w_grad_cos": (w_grad_cos_per * valids).sum() / valid_count,
            "behavior_alignment_cos": safety["behavior_alignment_cos"].mean(),
            "behavior_conflict_fraction": safety["behavior_conflict"].mean(),
            "guidance_retained": safety["guidance_retained"].mean(),
            "safe_w_norm": jnp.linalg.norm(safe_w, axis=-1).mean(),
            # Optional keys populated by v9+ trust / residual diagnostics.
            **{
                key: safety[key].mean()
                for key in (
                    "conflict_kill_frac",
                    "residual_damp",
                    "trust",
                )
                if key in safety
            },
        }

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(
        self,
        observations,
        noise,
        seed=None,
        temperature=0.0,
    ):
        del seed
        actions = noise
        actor_name = "actor" if temperature > 0 else "target_actor"
        for i in range(self.config["flow_steps"]):
            t = jnp.full(
                (*observations.shape[:-1], 1),
                i / self.config["flow_steps"],
            )
            fm_actor = jnp.concatenate(
                [observations, actions, t],
                axis=-1,
            )
            behavior_velocity = self.network.select(actor_name)(
                fm_actor
            ).mode()
            guidance = self.guidance_direction(
                observations,
                actions,
                t,
                behavior_velocity,
            )
            actions = actions + (
                behavior_velocity + guidance
            ) / self.config["flow_steps"]
        return jnp.clip(actions, -1, 1)


def get_config():
    config = get_v6_config()
    config.agent_name = "dflrql8"
    # Dimensionless relative gradient floor. A typical-gradient target has
    # norm 1 / (1 + consensus_floor), while locally flat gradients vanish.
    config.consensus_floor = 0.1
    return config
