"""Faithful pure Q-Flow (paper Algorithm 1).

Same network and loss offline and online:
- Deterministic vector-field MLP (4x512 GELU) with Fourier time.
- Time-free outer ``Q(s, a)`` ensemble-2 + Polyak target (tau=0.005).
- Inner ``V(s, x, t)`` ensemble-2 with Fourier-16 time embedding.
- Terminal Bellman Q; inner V via current-policy Euler-10 rollout to a
  stop-grad target Q; actor matches stop-grad
  ``(a - noise) + (1/lambda) grad_x V``.

No RQL reverse critic, expectile, q_pe, alpha, target actor, Gaussian
Actor, or warm-start bridge. Fourier / vector-field modules are shared
with ``agents/qflow_rql_warmstart.py``.
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

from agents.qflow_rql_warmstart import (
    DeterministicVectorField,
    FourierFeatures,
    FourierValue,
    fourier_time_features,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value


# Re-export shared Fourier/VF components for tests and callers.
__all__ = [
    "QFlowAgent",
    "DeterministicVectorField",
    "FourierFeatures",
    "FourierValue",
    "fourier_time_features",
    "get_config",
]


class QFlowAgent(flax.struct.PyTreeNode):
    """Pure Algorithm-1 Q-Flow (identical offline / online objective)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def aggregate_q(self, qs):
        if qs.ndim == 1:
            return qs
        q_agg = self.config["q_agg"]
        if q_agg == "min":
            return qs.min(axis=0)
        if q_agg == "mean":
            return qs.mean(axis=0)
        raise ValueError(f"Unsupported q_agg={q_agg!r}")

    def actor_velocity(self, observations, actions, times, *, params=None):
        """Predict deterministic vector field (no Gaussian /.mode())."""
        if times.ndim == 1:
            times = times[..., None]
        return self.network.select("actor")(
            observations, actions, times, params=params
        )

    def roll_flow_to_terminal(self, observations, x_t, t, *, params=None):
        """Euler-integrate from arbitrary ``t`` to ``1`` with current actor.

        Fixed ``flow_steps`` micro-steps, ``dt = (1 - t) / flow_steps``.
        Pass stored params (``params=None``) so rollouts do not BPTT into the
        actor when used as an inner-value target.
        """
        flow_steps = int(self.config["flow_steps"])
        actions = x_t
        t_curr = t if t.ndim == 2 else t[..., None]
        for i in range(flow_steps):
            frac = float(i) / float(flow_steps)
            t_i = t_curr + (1.0 - t_curr) * frac
            dt = (1.0 - t_curr) / float(flow_steps)
            out = self.actor_velocity(
                observations, actions, t_i, params=params
            )
            actions = actions + out * dt
        return actions

    def inner_value_loss(self, observations, actions, grad_params, rng):
        """Regress Fourier V to stop-grad target terminal Q via current-policy rollout."""
        batch_size, action_dim = observations.shape[0], actions.shape[-1]
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = t * actions + (1.0 - t) * noise

        # Stored actor params ⇒ no BPTT into the vector field.
        x_1_hat = self.roll_flow_to_terminal(observations, x_t, t, params=None)
        x_1_hat = jax.lax.stop_gradient(x_1_hat)

        target_qs = self.network.select("target_terminal_q")(observations, x_1_hat)
        target_q = jax.lax.stop_gradient(self.aggregate_q(target_qs))

        vs = self.network.select("inner_value")(
            observations, x_t, t, params=grad_params
        )
        v_pred = self.aggregate_q(vs)
        loss = jnp.square(v_pred - target_q).mean()
        return loss, {
            "inner_value_loss": loss,
            "inner_v_mean": v_pred.mean(),
            "inner_target_q_mean": target_q.mean(),
        }

    def total_loss(self, batch, grad_params, rng=None):
        """Algorithm-1: terminal Bellman Q + inner V + velocity matching."""
        rng = rng if rng is not None else self.rng
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

        # --- 1. Outer Bellman critic (time-free terminal Q) ---
        rng, n_rng = jax.random.split(rng)
        next_noise = jax.random.normal(n_rng, (batch_size, action_dim))
        next_actions = self.compute_flow_actions(next_observations, next_noise)
        next_qs = self.network.select("target_terminal_q")(
            next_observations, next_actions
        )
        next_q = self.aggregate_q(next_qs)
        tqt_q = jax.lax.stop_gradient(
            n_rews
            + (self.config["discount"] ** h) * next_q * batch["masks"][-2]
        )
        qs = self.network.select("terminal_q")(
            observations, actions, params=grad_params
        )
        q = self.aggregate_q(qs)
        critic_loss = (jnp.square(qs - tqt_q) * valids).mean()

        diag_params = jax.lax.stop_gradient(grad_params)

        def q_scalar(a_single, obs_single):
            q_ens = self.network.select("terminal_q")(
                obs_single[None], a_single[None], params=diag_params
            )
            return self.aggregate_q(q_ens).squeeze()

        action_grads = jax.vmap(jax.grad(q_scalar))(actions, observations)
        terminal_q_action_grad_norm = jnp.linalg.norm(action_grads, axis=-1).mean()

        # --- 2. Inner-value regression (current-policy Euler, no BPTT) ---
        rng, iv_rng = jax.random.split(rng)
        iv_loss, iv_info = self.inner_value_loss(
            observations, actions, grad_params, iv_rng
        )

        # --- 3. Actor: match stopped (a - noise) + (1/lambda) grad_x V ---
        rng, x_rng, t_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(x_rng, (batch_size, action_dim))
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = t * actions + (1.0 - t) * noise
        cfm_vec = actions - noise

        def v_scalar(x_single, obs_single, t_single):
            vs = self.network.select("inner_value")(
                obs_single[None], x_single[None], t_single[None]
            )
            return self.aggregate_q(vs).squeeze()

        grad_x = jax.vmap(jax.grad(v_scalar))(x_t, observations, t.squeeze(-1))
        lam = self.config["qflow_lambda"]
        v_target = jax.lax.stop_gradient(cfm_vec + (1.0 / lam) * grad_x)

        pred = self.actor_velocity(
            observations, x_t, t, params=grad_params
        )
        ac_mask = repeat(
            1 - rs_terminals[:-1],
            "h b -> b (h r)",
            r=action_dim // h,
        )
        actor_loss = (jnp.square(pred - v_target) * ac_mask).mean()

        inner_contrib = (1.0 / lam) * grad_x
        cfm_target_norm = jnp.linalg.norm(cfm_vec, axis=-1).mean()
        inner_grad_norm = jnp.linalg.norm(inner_contrib, axis=-1).mean()
        inner_grad_to_cfm_ratio = inner_grad_norm / jnp.maximum(cfm_target_norm, 1e-8)
        pred_velocity_norm = jnp.linalg.norm(pred, axis=-1).mean()

        total_loss = critic_loss + iv_loss + actor_loss
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
            "bc_loss": jnp.square(pred - cfm_vec).mean(),
            "cfm_target_norm": cfm_target_norm,
            "inner_grad_norm": inner_grad_norm,
            "inner_grad_to_cfm_ratio": inner_grad_to_cfm_ratio,
            "pred_velocity_norm": pred_velocity_norm,
            **iv_info,
        }
        return total_loss, info

    def target_update(self, network, module_name, d):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1.0 - d),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        # Only outer Q target Polyak (no target actor / reverse critic).
        self.target_update(
            new_network, "terminal_q", d=self.config["terminal_q_tau"]
        )
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(self, observations, noise, seed=None, temperature=0.0):
        """Euler-10 from noise with the *current* vector field."""
        del seed, temperature
        actions = noise
        flow_steps = int(self.config["flow_steps"])
        for i in range(flow_steps):
            t = jnp.full(
                (*observations.shape[:-1], 1), float(i) / float(flow_steps)
            )
            out = self.actor_velocity(observations, actions, t, params=None)
            actions = actions + out / float(flow_steps)
        return jnp.clip(actions, -1.0, 1.0)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        action_rng, n_rng = jax.random.split(seed)
        del action_rng
        obs = jnp.atleast_2d(obs)[-1:]
        noise = jax.random.normal(
            n_rng, (1, self.config["action_dim"])
        )
        actions = self.compute_flow_actions(
            obs, noise, temperature=temperature
        )[0]
        return rearrange(actions, "(h d) -> h d", h=self.config["h"])

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        config = dict(config)
        # training_phase is intentionally unused (same loss offline/online).
        config.pop("training_phase", None)
        config.setdefault("terminal_q_tau", 0.005)
        config.setdefault("actor_fourier_dim", 32)
        config.setdefault("inner_fourier_dim", 16)
        config.setdefault("ensemble_ct", 2)
        config.setdefault("inner_ensemble_ct", 2)

        ex_actions_h = jnp.concatenate([ex_actions] * config["h"], -1)
        ex_times = jnp.zeros((*ex_observations.shape[:-1], 1), dtype=ex_actions.dtype)
        action_dim = ex_actions_h.shape[-1]

        terminal_q_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        inner_value_def = FourierValue(
            hidden_dims=config["inner_value_hidden_dims"],
            fourier_dim=config["inner_fourier_dim"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["inner_ensemble_ct"],
        )
        actor_def = DeterministicVectorField(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            fourier_dim=config["actor_fourier_dim"],
            layer_norm=config["actor_layer_norm"],
        )

        actor_args = dict(
            observations=ex_observations,
            actions=ex_actions_h,
            times=ex_times,
        )
        terminal_q_args = dict(
            observations=ex_observations, actions=ex_actions_h
        )
        inner_args = dict(
            observations=ex_observations,
            actions=ex_actions_h,
            times=ex_times,
        )

        network_info = dict(
            terminal_q=(terminal_q_def, terminal_q_args),
            target_terminal_q=(copy.deepcopy(terminal_q_def), terminal_q_args),
            actor=(actor_def, actor_args),
            inner_value=(inner_value_def, inner_args),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_terminal_q"] = params["modules_terminal_q"]
        config["action_dim"] = action_dim
        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(config["h"])) + [jnp.inf])
        )

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """Default config: paper-faithful pure Q-Flow (h=1 HL)."""
    config = mlc.ConfigDict(
        dict(
            agent_name="qflow",
            h=1,
            ensemble_ct=2,
            inner_ensemble_ct=2,
            lr=3e-4,
            discount=0.995,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            inner_value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            actor_fourier_dim=32,
            inner_fourier_dim=16,
            terminal_q_tau=0.005,
            flow_steps=10,
            q_agg="mean",
            qflow_lambda=1.0,
        )
    )
    return config
