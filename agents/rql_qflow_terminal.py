"""RQL → Q-Flow with a separately trained terminal-action Bellman critic.

Fixes the action-blind RQL terminal critic used by ``rql_qflow``: RQL's
``value`` head is trained on intermediate flow states, so querying it at
``t=1`` with terminal actions does not provide meaningful action gradients.

This agent keeps the exact RQL actor/critic objective on ``value``/``actor``,
and adds ``terminal_q`` / ``target_terminal_q`` ensembles trained with a
standard terminal-action Bellman backup. Inner-value regression and online
Q-Flow use the terminal head only.
"""

from __future__ import annotations

import copy

import flax
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange, repeat

from agents.rql_qflow import RQLQFlowAgent
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import Actor, Value


class RQLQFlowTerminalAgent(RQLQFlowAgent):
    """RQLQFlowAgent with an isolated terminal-action Bellman Q ensemble."""

    def terminal_q_bellman_loss(self, batch, grad_params, rng):
        """Standard outer Bellman loss on dataset actions at ``t=1``."""
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

        rng, n_rng = jax.random.split(rng)
        next_noise = jax.random.normal(n_rng, (batch_size, action_dim))
        next_actions = self.compute_flow_actions(
            next_observations, next_noise, temperature=0.0
        )
        ones = jnp.ones((batch_size, 1), dtype=actions.dtype)
        next_inp = self.encode_sa_t(next_observations, next_actions, ones)
        next_qs = self.network.select("target_terminal_q")(next_inp)
        next_q = self.aggregate_q(next_qs)

        tqt_q = (
            n_rews
            + (self.config["discount"] ** h) * next_q * batch["masks"][-2]
        )
        tqt_q = jax.lax.stop_gradient(tqt_q)

        q_inp = self.encode_sa_t(observations, actions, ones)
        qs = self.network.select("terminal_q")(q_inp, params=grad_params)
        q = self.aggregate_q(qs)
        terminal_q_loss = (jnp.square(qs - tqt_q) * valids).mean()

        # Action sensitivity of the live terminal head (diagnostic only).
        diag_params = jax.lax.stop_gradient(grad_params)

        def q_scalar(a_single, obs_single):
            inp = self.encode_sa_t(
                obs_single[None], a_single[None], ones[:1]
            )
            q_ens = self.network.select("terminal_q")(inp, params=diag_params)
            return self.aggregate_q(q_ens).squeeze()

        action_grads = jax.vmap(jax.grad(q_scalar))(actions, observations)
        terminal_q_action_grad_norm = jnp.linalg.norm(action_grads, axis=-1).mean()

        info = {
            "terminal_q_loss": terminal_q_loss,
            "terminal_q_mean": q.mean(),
            "terminal_q_max": q.max(),
            "terminal_q_min": q.min(),
            "terminal_q_action_grad_norm": terminal_q_action_grad_norm,
        }
        return terminal_q_loss, info, rng

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
        """Regress V(s, x_t, t) to stop-grad ``target_terminal_q`` at time=1."""
        batch_size, action_dim = observations.shape[0], actions.shape[-1]
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = t * actions + (1.0 - t) * noise

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
        target_qs = self.network.select("target_terminal_q")(terminal_inp)
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

    def qflow_online_loss(self, batch, grad_params, rng):
        """Online Q-Flow with terminal_q as the outer/inner Bellman target."""
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

        # --- 1. Outer Bellman critic on terminal_q at t=1 ---
        rng, n_rng = jax.random.split(rng)
        next_noise = jax.random.normal(n_rng, (batch_size, action_dim))
        next_actions = self.compute_flow_actions(
            next_observations, next_noise, temperature=0.0
        )
        ones = jnp.ones((batch_size, 1), dtype=actions.dtype)
        next_inp = self.encode_sa_t(next_observations, next_actions, ones)
        next_qs = self.network.select("target_terminal_q")(next_inp)
        next_q = self.aggregate_q(next_qs)

        tqt_q = (
            n_rews
            + (self.config["discount"] ** h) * next_q * batch["masks"][-2]
        )
        tqt_q = jax.lax.stop_gradient(tqt_q)

        q_inp = self.encode_sa_t(observations, actions, ones)
        qs = self.network.select("terminal_q")(q_inp, params=grad_params)
        q = self.aggregate_q(qs)
        critic_loss = (jnp.square(qs - tqt_q) * valids).mean()

        def q_scalar(a_single, obs_single):
            inp = self.encode_sa_t(
                obs_single[None], a_single[None], ones[:1]
            )
            q_ens = self.network.select("terminal_q")(
                inp, params=jax.lax.stop_gradient(grad_params)
            )
            return self.aggregate_q(q_ens).squeeze()

        action_grads = jax.vmap(jax.grad(q_scalar))(actions, observations)
        terminal_q_action_grad_norm = jnp.linalg.norm(action_grads, axis=-1).mean()

        # --- 2. Inner-value regression (policy rollout → target_terminal_q) ---
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
            "terminal_q_loss": critic_loss,
            "terminal_q_mean": q.mean(),
            "terminal_q_max": q.max(),
            "terminal_q_min": q.min(),
            "terminal_q_action_grad_norm": terminal_q_action_grad_norm,
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
            rng, tq_rng = jax.random.split(rng)
            terminal_q_loss, tq_info, rng = self.terminal_q_bellman_loss(
                batch, grad_params, tq_rng
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
            total_loss = rql_loss + terminal_q_loss + iv_loss
            info.update(rql_info)
            info.update(tq_info)
            info.update(iv_info)
            info["total_loss"] = total_loss
            return total_loss, info

        if phase == "qflow_online":
            return self.qflow_online_loss(batch, grad_params, rng)

        raise ValueError(f"Unknown training_phase={phase!r}")

    @jax.jit
    def update(self, batch):
        """Update agent; Polyak-update value, actor, and terminal_q targets."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        self.target_update(new_network, "value", d=self.config["tau"])
        self.target_update(
            new_network, "terminal_q", d=self.config["terminal_q_tau"]
        )
        skip_actor_ema = (
            self.config["training_phase"] == "qflow_online"
            and float(self.config["qflow_actor_coef"]) == 0.0
        )
        if not skip_actor_ema:
            self.target_update(new_network, "actor", d=1 - self.config["ema"])

        return self.replace(network=new_network, rng=new_rng), info

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
        config.setdefault("qflow_actor_coef", 1.0)
        config.setdefault("terminal_q_tau", 0.005)

        ex_actions = jnp.concatenate([ex_actions] * config["h"], -1)
        ex_times = ex_actions[..., :1]
        ex_in = jnp.concatenate([ex_observations, ex_actions, ex_times], -1)
        action_dim = ex_actions.shape[-1]

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        terminal_q_def = Value(
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
            terminal_q=(terminal_q_def, (ex_in,)),
            target_terminal_q=(copy.deepcopy(terminal_q_def), (ex_in,)),
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
        params["modules_target_terminal_q"] = params["modules_terminal_q"]
        params["modules_target_actor"] = params["modules_actor"]
        config["action_dim"] = action_dim

        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf])
        )

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="rql_qflow_terminal",
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
            terminal_q_tau=0.005,
            ema=0.999,
            flow_steps=10,
            q_agg="mean",
            qflow_lambda=1.0,
            qflow_actor_coef=1.0,
        )
    )
    return config
