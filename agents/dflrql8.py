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
        return self._scaled_guidance(w, times, behavior_velocity)[0]

    def _scaled_guidance(
        self,
        w,
        times,
        behavior_velocity,
        *,
        bypass_safety=False,
    ):
        """Apply scaling used for residual RL / inference guidance."""
        if bypass_safety:
            # Keep the unit-ball range of consensus targets, but do not kill
            # anti-BC components during residual RL. Inference still uses the
            # full behavior-safe path via guidance_direction().
            unit_w = self._project_unit_ball(w)
            scaled = self.config["guidance_coef"] * times * unit_w
            return scaled, unit_w
        safe_w, _ = self._behavior_safe_direction(w, behavior_velocity)
        scaled = self.config["guidance_coef"] * times * safe_w
        return scaled, safe_w

    def _target_q_from_action(self, observations, flat_action, times):
        """Ensemble-mean(-ρ·std) target critic score at a flow point."""
        qs = self.network.select("target_value")(
            jnp.concatenate([observations, flat_action, times], axis=-1)
        )
        return qs.mean(axis=0) - self.config["rho"] * qs.std(axis=0)

    def _actor_q_action(self, flat_action):
        """Hook for the actor q_pe one-step lookahead action only.

        Default is identity. Subclasses (e.g. QuantizedDFLRQL9) may project
        this flat action before it is scored by ``value``. Critic reversal,
        BC, distillation, guidance, and behavior-action paths must not call
        this hook.
        """
        return flat_action

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
        # Default: score the guided one-step lookahead so v_θ can absorb the
        # RL residual that G would otherwise provide at sample time. Set
        # actor_lookahead_use_guidance=False to force policy improvement to
        # rely on G at inference (G-role ablation).
        guidance = jax.lax.stop_gradient(
            self.guidance_direction(
                batch["observations"][0],
                x_t,
                t,
                behavior_velocity,
            )
        )
        if bool(self.config.get("actor_lookahead_use_guidance", True)):
            guided_velocity = behavior_velocity + guidance
        else:
            guided_velocity = behavior_velocity
        lookahead_action = x_t + guided_velocity * jnp.minimum(
            1 / self.config["flow_steps"],
            1 - t,
        )
        lookahead_action = self._actor_q_action(lookahead_action)
        q_pe = self.network.select("value")(
            jnp.concatenate(
                [
                    batch["observations"][0],
                    lookahead_action,
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
        # Q-lookahead policy improvement for the flow actor. Set actor_q_coef=0
        # for the BC-only-v / RL-only-G ablation (experiment B): v_θ trains
        # only on flow matching, while W_φ still gets consensus distillation.
        actor_loss = -(q_pe * valids).mean()
        actor_q_coef = self.config.get("actor_q_coef", 1.0)

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
        rl_bypass_safety = bool(
            self.config.get("guidance_rl_bypass_safety", False)
        )
        use_advantage = bool(self.config.get("guidance_use_advantage", False))
        scaled_guidance, _ = self._scaled_guidance(
            w,
            t,
            behavior_velocity,
            bypass_safety=rl_bypass_safety,
        )

        # Direct one-step residual policy improvement. Unlike actor_loss,
        # gradients flow only through G: v and the target critic are constants.
        # This is useful when a pretrained flow expert must remain frozen.
        guidance_q_coef = self.config.get("guidance_q_coef", 0.0)
        guidance_q_loss = jnp.asarray(0.0, dtype=q.dtype)
        guidance_q_mean = jnp.asarray(0.0, dtype=q.dtype)
        guidance_adv_mean = jnp.asarray(0.0, dtype=q.dtype)
        if guidance_q_coef != 0.0:
            guidance_dt = jnp.minimum(
                1 / self.config["flow_steps"],
                1 - t,
            )
            frozen_velocity = jax.lax.stop_gradient(behavior_velocity)
            guidance_q_action = x_t + (
                frozen_velocity + scaled_guidance
            ) * guidance_dt
            guidance_q_action = self._actor_q_action(guidance_q_action)
            guidance_t_next = jnp.clip(
                t + 1 / self.config["flow_steps"],
                max=1,
            )
            guidance_q = self._target_q_from_action(
                batch["observations"][0],
                guidance_q_action,
                guidance_t_next,
            )
            if use_advantage:
                baseline_action = self._actor_q_action(
                    x_t + frozen_velocity * guidance_dt
                )
                baseline_q = jax.lax.stop_gradient(
                    self._target_q_from_action(
                        batch["observations"][0],
                        baseline_action,
                        guidance_t_next,
                    )
                )
                guidance_adv = guidance_q - baseline_q
                guidance_q_loss = -(guidance_adv * valids).mean()
                guidance_adv_mean = (guidance_adv * valids).sum() / valid_count
            else:
                guidance_q_loss = -(guidance_q * valids).mean()
            guidance_q_mean = (guidance_q * valids).sum() / valid_count

        # Reparameterized residual policy improvement through the full frozen
        # flow. Actor outputs are stop-gradded, so this never backpropagates
        # through v (important for large VLA experts); G receives gradients
        # through every Euler step and the final target-Q score.
        guidance_rollout_q_coef = self.config.get(
            "guidance_rollout_q_coef", 0.0
        )
        guidance_energy_coef = self.config.get("guidance_energy_coef", 0.0)
        guidance_rollout_q_loss = jnp.asarray(0.0, dtype=q.dtype)
        guidance_rollout_q_mean = jnp.asarray(0.0, dtype=q.dtype)
        guidance_rollout_adv_mean = jnp.asarray(0.0, dtype=q.dtype)
        guidance_energy = jnp.asarray(0.0, dtype=q.dtype)
        if guidance_rollout_q_coef != 0.0 or guidance_energy_coef != 0.0:
            rollout_action = x_0
            baseline_action = x_0
            rollout_energy = jnp.zeros((batch_size,), dtype=q.dtype)
            for flow_idx in range(self.config["flow_steps"]):
                rollout_t = jnp.full(
                    (batch_size, 1),
                    flow_idx / self.config["flow_steps"],
                )
                # Evaluate the frozen expert along the *guided* trajectory so
                # the residual is a local correction, not a separate open-loop
                # baseline path that diverges immediately.
                rollout_actor_in = jnp.concatenate(
                    [
                        batch["observations"][0],
                        rollout_action,
                        rollout_t,
                    ],
                    axis=-1,
                )
                rollout_velocity = jax.lax.stop_gradient(
                    self.network.select("target_actor")(
                        rollout_actor_in
                    ).mode()
                )
                baseline_actor_in = jnp.concatenate(
                    [
                        batch["observations"][0],
                        baseline_action,
                        rollout_t,
                    ],
                    axis=-1,
                )
                baseline_velocity = jax.lax.stop_gradient(
                    self.network.select("target_actor")(
                        baseline_actor_in
                    ).mode()
                )
                rollout_w = self.network.select("guidance")(
                    rollout_actor_in,
                    params=grad_params,
                )
                rollout_guidance, _ = self._scaled_guidance(
                    rollout_w,
                    rollout_t,
                    rollout_velocity,
                    bypass_safety=rl_bypass_safety,
                )
                rollout_energy = rollout_energy + jnp.square(
                    rollout_guidance
                ).sum(axis=-1)
                rollout_action = rollout_action + (
                    rollout_velocity + rollout_guidance
                ) / self.config["flow_steps"]
                baseline_action = baseline_action + baseline_velocity / self.config[
                    "flow_steps"
                ]

            rollout_action = self._actor_q_action(rollout_action)
            baseline_action = self._actor_q_action(baseline_action)
            ones = jnp.ones((batch_size, 1))
            rollout_q = self._target_q_from_action(
                batch["observations"][0],
                rollout_action,
                ones,
            )
            if use_advantage:
                baseline_q = jax.lax.stop_gradient(
                    self._target_q_from_action(
                        batch["observations"][0],
                        baseline_action,
                        ones,
                    )
                )
                rollout_adv = rollout_q - baseline_q
                guidance_rollout_q_loss = -(rollout_adv * valids).mean()
                guidance_rollout_adv_mean = (
                    rollout_adv * valids
                ).sum() / valid_count
            else:
                guidance_rollout_q_loss = -(rollout_q * valids).mean()
            guidance_rollout_q_mean = (
                rollout_q * valids
            ).sum() / valid_count
            guidance_energy = (
                rollout_energy / self.config["flow_steps"] * valids
            ).sum() / valid_count

        unit_q_grad = q_grad / (q_grad_norm + 1e-6)
        w_norm_per = jnp.linalg.norm(w, axis=-1)
        w_grad_cos_per = (w * unit_q_grad).sum(axis=-1) / (
            w_norm_per + 1e-6
        )

        total_loss = (
            actor_loss * actor_q_coef
            + bc_loss * self.config["alpha"]
            + critic_loss
            + distill_loss * self.config["distill_coef"]
            + guidance_q_loss * guidance_q_coef
            + guidance_rollout_q_loss * guidance_rollout_q_coef
            + guidance_energy * guidance_energy_coef
        )

        return total_loss, {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "actor_q_coef": jnp.asarray(actor_q_coef, dtype=jnp.float32),
            "actor_lookahead_use_guidance": jnp.asarray(
                bool(self.config.get("actor_lookahead_use_guidance", True)),
                dtype=jnp.float32,
            ),
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
            "guidance_q_loss": guidance_q_loss,
            "guidance_q_mean": guidance_q_mean,
            "guidance_adv_mean": guidance_adv_mean,
            "guidance_q_coef": jnp.asarray(
                guidance_q_coef,
                dtype=jnp.float32,
            ),
            "guidance_rollout_q_loss": guidance_rollout_q_loss,
            "guidance_rollout_q_mean": guidance_rollout_q_mean,
            "guidance_rollout_adv_mean": guidance_rollout_adv_mean,
            "guidance_rollout_q_coef": jnp.asarray(
                guidance_rollout_q_coef,
                dtype=jnp.float32,
            ),
            "guidance_energy": guidance_energy,
            "guidance_energy_coef": jnp.asarray(
                guidance_energy_coef,
                dtype=jnp.float32,
            ),
            "guidance_use_advantage": jnp.asarray(
                use_advantage,
                dtype=jnp.float32,
            ),
            "guidance_rl_bypass_safety": jnp.asarray(
                rl_bypass_safety,
                dtype=jnp.float32,
            ),
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


    @jax.jit
    def update(self, batch):
        """Update agent; optionally freeze the flow actor (online G-only phase)."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        freeze_actor = bool(self.config.get("freeze_actor", False))
        if freeze_actor:
            # Hard freeze: revert any actor / target-actor drift (e.g. from
            # BC/Q terms or numerical leak) and skip target-actor EMA below.
            new_network.params["modules_actor"] = self.network.params[
                "modules_actor"
            ]
            new_network.params["modules_target_actor"] = self.network.params[
                "modules_target_actor"
            ]
        info = {
            **info,
            "freeze_actor": jnp.asarray(freeze_actor, dtype=jnp.float32),
        }
        self.target_update(new_network, "value", d=self.config["tau"])
        if not freeze_actor:
            self.target_update(new_network, "actor", d=1 - self.config["ema"])

        return self.replace(network=new_network, rng=new_rng), info


def get_config():
    config = get_v6_config()
    config.agent_name = "dflrql8"
    # Dimensionless relative gradient floor. A typical-gradient target has
    # norm 1 / (1 + consensus_floor), while locally flat gradients vanish.
    config.consensus_floor = 0.1
    # Weight on -Q lookahead for v_θ. Default 1 keeps legacy CF; 0 = BC-only
    # actor (guidance / distill still trained).
    config.actor_q_coef = 1.0
    # If True, actor Q-lookahead scores v+sg(G). If False, scores v alone so
    # G must carry sample-time improvement (G-role ablation).
    config.actor_lookahead_use_guidance = True
    # If True, zero actor grads and skip target-actor EMA (online G-only).
    config.freeze_actor = False
    # Direct one-step target-Q policy improvement for G. The actor and target
    # critic remain stop-gradded; zero preserves the published v8/v9 behavior.
    config.guidance_q_coef = 0.0
    # Full-flow target-Q policy improvement for G under a frozen target actor.
    config.guidance_rollout_q_coef = 0.0
    # Mean squared applied guidance along that rollout (residual trust cost).
    config.guidance_energy_coef = 0.0
    # If True, residual RL maximizes Q(v+G)-Q(v) instead of absolute Q(v+G).
    config.guidance_use_advantage = False
    # If True, residual RL scales W without behavior-conflict projection.
    # Inference still uses the full safe guidance_direction path.
    config.guidance_rl_bypass_safety = False
    return config
