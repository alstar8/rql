"""RQL → Q-Flow two-phase agent.

Offline (``training_phase='rql_offline'``): exact RQL actor/critic objectives plus
an isolated intermediate-value regression that cannot influence actor/critic
gradients.

Online (``training_phase='qflow_online'``): faithful Q-Flow outer Bellman critic,
inner-value regression to a stop-grad policy-rollout terminal target Q, and
actor velocity matching to
``sg[(a - noise) + (1 / lambda) * grad_x V]``.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange, repeat

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value


class RQLQFlowAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def aggregate_q(self, qs):
        """Aggregate ensemble Q/V predictions (mean by default).

        ``Value`` with ``num_ensembles == 1`` returns ``(batch,)``; ensembles
        return ``(ensemble, batch)``.
        """
        if qs.ndim == 1:
            return qs
        q_agg = self.config["q_agg"]
        if q_agg == "min":
            return qs.min(axis=0)
        if q_agg == "mean":
            return qs.mean(axis=0)
        raise ValueError(f"Unsupported q_agg={q_agg!r}")

    def encode_sa_t(self, observations, actions_or_x, times):
        """Concatenate (s, x, t) in the RQL network convention."""
        if times.ndim == 1:
            times = times[..., None]
        return jnp.concatenate([observations, actions_or_x, times], axis=-1)

    def roll_flow_to_terminal(
        self,
        observations,
        x_t,
        t,
        *,
        module_name="target_actor",
        params=None,
    ):
        """Euler-integrate the flow from time ``t`` to ``1``.

        Uses a fixed number of Euler steps equal to ``flow_steps``, with per-sample
        step size ``(1 - t) / flow_steps``. This is batch-friendly and equivalent to
        integrating the remaining horizon with a uniform partition (no BPTT when
        ``params`` is omitted / targets are stop-grad'd).
        """
        flow_steps = int(self.config["flow_steps"])
        actions = x_t
        t_curr = t if t.ndim == 2 else t[..., None]
        for i in range(flow_steps):
            # Time at the beginning of micro-step i on [t, 1].
            frac = float(i) / float(flow_steps)
            t_i = t_curr + (1.0 - t_curr) * frac
            dt = (1.0 - t_curr) / float(flow_steps)
            fm = self.encode_sa_t(observations, actions, t_i)
            if params is None:
                out = self.network.select(module_name)(fm).mode()
            else:
                out = self.network.select(module_name)(fm, params=params).mode()
            actions = actions + out * dt
        return actions

    def inner_value_loss(
        self,
        observations,
        actions,
        grad_params,
        rng,
        *,
        rollout_module="target_actor",
        use_policy_params=False,
    ):
        """Regress V(s, x_t, t) to stop-grad terminal target Q at time=1."""
        batch_size, action_dim = observations.shape[0], actions.shape[-1]
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = t * actions + (1.0 - t) * noise

        # Rollout without flowing grads into the actor (no BPTT through solver).
        if use_policy_params:
            x_1_hat = self.roll_flow_to_terminal(
                observations,
                x_t,
                t,
                module_name="actor",
                params=None,
            )
        else:
            x_1_hat = self.roll_flow_to_terminal(
                observations,
                x_t,
                t,
                module_name=rollout_module,
                params=None,
            )
        x_1_hat = jax.lax.stop_gradient(x_1_hat)

        ones = jnp.ones((batch_size, 1), dtype=actions.dtype)
        terminal_inp = self.encode_sa_t(observations, x_1_hat, ones)
        target_qs = self.network.select("target_value")(terminal_inp)
        target_q = jax.lax.stop_gradient(self.aggregate_q(target_qs))

        v_inp = self.encode_sa_t(observations, x_t, t)
        vs = self.network.select("inner_value")(v_inp, params=grad_params)
        v_pred = self.aggregate_q(vs)
        loss = jnp.square(v_pred - target_q).mean()
        return loss, {
            "inner_value_loss": loss,
            "inner_v_mean": v_pred.mean(),
            "inner_target_q_mean": target_q.mean(),
        }

    def rql_actor_critic_loss(self, batch, grad_params, rng):
        """Exact RQL actor/critic objectives (no inner-value term)."""
        info = {}
        batch_size, action_dim = self.config["batch_size"], self.config["action_dim"]

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
                jax.random.randint(r_rng, (batch_size // 2,), 0, self.config["flow_steps"] + 1)
                / self.config["flow_steps"],
            ],
            0,
        )
        d_b = d / self.config["flow_steps"]

        actions = rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")

        x_f = jnp.copy(actions)
        f = jnp.ones((batch_size, 1))
        for _ in range(self.config["flow_steps"]):
            fm_actor = jnp.concatenate([batch["observations"][0], x_f, f], -1)
            out = self.network.select("actor")(fm_actor).mode()
            x_f = x_f - out * d_b[..., None]
            f = f - d_b[..., None]

        state = jnp.concatenate(
            [batch["observations"][0], jax.lax.stop_gradient(x_f), f], axis=-1
        )

        q = self.network.select("value")(state, params=grad_params)

        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]], axis=0
        )
        n_rews = (
            batch["rewards"]
            * self.config["discount_mul"][..., None]
            * (1 - rs_terminals)
        ).sum(0)
        tqt_q = (
            n_rews
            + (self.config["discount"] ** (self.config["h"]))
            * next_q
            * batch["masks"][-2]
        )

        s = rs_terminals.sum(0)
        valids = (s <= 1).astype(s.dtype)
        critic_loss = (
            self.expectile_loss(tqt_q - q, tqt_q - q, self.config["expectile"]) * valids
        ).mean()

        rng, x_rng, t_rng = jax.random.split(rng, 3)
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")
        t = jax.random.uniform(t_rng, (batch_size, 1))

        x_t = (1 - t) * x_0 + t * x_1
        tgt = x_1 - x_0
        fm_actor = jnp.concatenate([batch["observations"][0], x_t, t], axis=-1)
        pred = self.network.select("actor")(fm_actor, params=grad_params).mode()
        q_pe = self.network.select("value")(
            jnp.concatenate(
                [
                    batch["observations"][0],
                    x_t
                    + pred
                    * jnp.minimum(1 / self.config["flow_steps"], 1 - t),
                    jnp.clip(t + 1 / self.config["flow_steps"], max=1),
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
        bc_loss = (jnp.square(pred - tgt) * ac_mask).mean()
        actor_loss = -(q_pe * valids).mean()

        rql_loss = actor_loss + bc_loss * self.config["alpha"] + critic_loss
        info.update(
            {
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
                "rql_loss": rql_loss,
            }
        )
        return rql_loss, info, rng

    def qflow_online_loss(self, batch, grad_params, rng):
        """Faithful Q-Flow online objectives under the RQL (s, x, t) convention."""
        batch_size = self.config["batch_size"]
        action_dim = self.config["action_dim"]
        h = self.config["h"]
        observations = batch["observations"][0]
        actions = rearrange(batch["actions"][:h], "h b d -> b (h d)")
        next_observations = batch["observations"][-1]

        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]], axis=0
        )
        n_rews = (
            batch["rewards"]
            * self.config["discount_mul"][..., None]
            * (1 - rs_terminals)
        ).sum(0)
        s = rs_terminals.sum(0)
        valids = (s <= 1).astype(s.dtype)

        # --- 1. Outer Bellman critic at terminal actions (time=1) ---
        rng, n_rng = jax.random.split(rng)
        next_noise = jax.random.normal(n_rng, (batch_size, action_dim))
        next_actions = self.compute_flow_actions(
            next_observations, next_noise, temperature=0.0
        )
        ones = jnp.ones((batch_size, 1), dtype=actions.dtype)
        next_inp = self.encode_sa_t(next_observations, next_actions, ones)
        next_qs = self.network.select("target_value")(next_inp)
        next_q = self.aggregate_q(next_qs)

        tqt_q = (
            n_rews
            + (self.config["discount"] ** h) * next_q * batch["masks"][-2]
        )
        tqt_q = jax.lax.stop_gradient(tqt_q)

        q_inp = self.encode_sa_t(observations, actions, ones)
        qs = self.network.select("value")(q_inp, params=grad_params)
        q = self.aggregate_q(qs)
        critic_loss = (jnp.square(qs - tqt_q) * valids).mean()

        # --- 2. Inner-value regression (policy rollout, stop-grad target) ---
        rng, iv_rng = jax.random.split(rng)
        iv_loss, iv_info = self.inner_value_loss(
            observations,
            actions,
            grad_params,
            iv_rng,
            use_policy_params=True,
        )

        # --- 3. Actor velocity matching ---
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = t * actions + (1.0 - t) * noise

        def v_scalar(x_single, obs_single, t_single):
            inp = self.encode_sa_t(
                obs_single[None], x_single[None], t_single[None]
            )
            vs = self.network.select("inner_value")(inp)
            return self.aggregate_q(vs).squeeze()

        grad_x = jax.vmap(jax.grad(v_scalar))(x_t, observations, t.squeeze(-1))
        lam = self.config["qflow_lambda"]
        v_target = (actions - noise) + (1.0 / lam) * grad_x
        v_target = jax.lax.stop_gradient(v_target)

        fm_actor = self.encode_sa_t(observations, x_t, t)
        pred = self.network.select("actor")(fm_actor, params=grad_params).mode()
        ac_mask = repeat(
            1 - rs_terminals[:-1],
            "h b -> b (h r)",
            r=action_dim // h,
        )
        actor_loss = (jnp.square(pred - v_target) * ac_mask).mean()
        actor_coef = jnp.asarray(self.config["qflow_actor_coef"], dtype=actor_loss.dtype)

        cfm_vec = actions - noise
        inner_contrib = (1.0 / lam) * grad_x
        cfm_target_norm = jnp.linalg.norm(cfm_vec, axis=-1).mean()
        inner_grad_norm = jnp.linalg.norm(inner_contrib, axis=-1).mean()
        inner_grad_to_cfm_ratio = inner_grad_norm / jnp.maximum(cfm_target_norm, 1e-8)
        pred_velocity_norm = jnp.linalg.norm(pred, axis=-1).mean()

        total_loss = critic_loss + iv_loss + actor_coef * actor_loss
        info = {
            "total_loss": total_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "q": q.mean(),
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "bc_loss": jnp.square(pred - (actions - noise)).mean(),
            "cfm_target_norm": cfm_target_norm,
            "inner_grad_norm": inner_grad_norm,
            "inner_grad_to_cfm_ratio": inner_grad_to_cfm_ratio,
            "pred_velocity_norm": pred_velocity_norm,
            "actor_coef": actor_coef,
            **iv_info,
        }
        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        phase = self.config["training_phase"]

        if phase == "rql_offline":
            rql_loss, rql_info, rng = self.rql_actor_critic_loss(
                batch, grad_params, rng
            )
            observations = batch["observations"][0]
            actions = rearrange(
                batch["actions"][: self.config["h"]], "h b d -> b (h d)"
            )
            rng, iv_rng = jax.random.split(rng)
            iv_loss, iv_info = self.inner_value_loss(
                observations,
                actions,
                grad_params,
                iv_rng,
                rollout_module="target_actor",
                use_policy_params=False,
            )
            total_loss = rql_loss + iv_loss
            info.update(rql_info)
            info.update(iv_info)
            info["total_loss"] = total_loss
            return total_loss, info

        if phase == "qflow_online":
            return self.qflow_online_loss(batch, grad_params, rng)

        # Exhaustive static phase check (nonpytree config).
        raise ValueError(f"Unknown training_phase={phase!r}")

    def target_update(self, network, module_name, d):
        """Update the target network (Polyak / EMA)."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        self.target_update(new_network, "value", d=self.config["tau"])
        # Freeze target_actor EMA when online actor loss is disabled (coef=0).
        skip_actor_ema = (
            self.config["training_phase"] == "qflow_online"
            and float(self.config["qflow_actor_coef"]) == 0.0
        )
        if not skip_actor_ema:
            self.target_update(new_network, "actor", d=1 - self.config["ema"])

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(
        self,
        observations,
        noise,
        seed=None,
        temperature=0.0,
    ):
        actions = noise
        for i in range(self.config["flow_steps"]):
            t = jnp.full((*observations.shape[:-1], 1), i / self.config["flow_steps"])
            fm_actor = jnp.concatenate([observations, actions, t], axis=-1)
            out = self.network.select(
                "actor" if temperature > 0 else "target_actor"
            )(fm_actor).mode()
            actions = actions + (out / self.config["flow_steps"])
        actions = jnp.clip(actions, -1, 1)
        return actions

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(
        self,
        obs,
        seed=None,
        temperature=0.0,
    ):
        action_rng, n_rng = jax.random.split(seed)

        obs = jnp.atleast_2d(obs)[-1:]
        noise = jax.random.normal(
            n_rng,
            (
                1,
                self.config["action_dim"],
            ),
        )
        actions = self.compute_flow_actions(
            obs, seed=action_rng, noise=noise, temperature=temperature
        )[0]
        actions = rearrange(actions, "(h d) -> h d", h=self.config["h"])
        return actions

    def replace_training_phase(self, training_phase):
        """Return a copy with a new static ``training_phase`` (nonpytree config)."""
        if training_phase not in ("rql_offline", "qflow_online"):
            raise ValueError(f"Unknown training_phase={training_phase!r}")
        new_config = dict(self.config)
        new_config["training_phase"] = training_phase
        return self.replace(config=flax.core.FrozenDict(**new_config))

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        config = dict(config)
        phase = config.get("training_phase", "rql_offline")
        if phase not in ("rql_offline", "qflow_online"):
            raise ValueError(f"Unknown training_phase={phase!r}")
        config["training_phase"] = phase
        # Config-only; default preserves older checkpoints without this key.
        config.setdefault("qflow_actor_coef", 1.0)

        ex_actions = jnp.concatenate([ex_actions] * config["h"], -1)
        ex_times = ex_actions[..., :1]
        ex_in = jnp.concatenate([ex_observations, ex_actions, ex_times], -1)
        action_dim = ex_actions.shape[-1]

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        inner_value_def = Value(
            hidden_dims=config["inner_value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["inner_ensemble_ct"],
        )

        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=1,
        )

        network_info = dict(
            value=(value_def, (ex_in,)),
            target_value=(copy.deepcopy(value_def), (ex_in,)),
            actor=(actor_def, (ex_in,)),
            target_actor=(copy.deepcopy(actor_def), (ex_in,)),
            inner_value=(inner_value_def, (ex_in,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]
        params["modules_target_actor"] = params["modules_actor"]
        config["action_dim"] = action_dim

        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf])
        )

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="rql_qflow",
            training_phase="rql_offline",
            h=3,
            alpha=1.0,
            expectile=0.5,
            ensemble_ct=10,
            inner_ensemble_ct=1,
            rho=0.0,
            lr=3e-4,
            discount=0.99,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            inner_value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            tau=0.005,
            ema=0.999,
            flow_steps=10,
            q_agg="mean",
            qflow_lambda=1.0,
            qflow_actor_coef=1.0,
        )
    )
    return config
