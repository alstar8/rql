"""Paper-aligned Q-Flow v2 with RQL warm start.

Phases
------
``rql_warmstart``
    Tuned RQL actor / reverse-flow critic objective. Time-free outer Q and
    Fourier inner V are isolated auxiliaries (no grads into RQL heads).
``qflow_bridge``
    Offline replay. Outer Q + inner V + actor velocity matching with
    configurable ``qflow_actor_coef`` and ``qflow_bridge_blend`` (0 = CFM /
    RQL BC target, 1 = exact Q-Flow gradient matching).
``qflow_online``
    Exact Algorithm-1 losses only: terminal Bellman Q, inner V, actor
    gradient matching. No expectile / reverse critic / q_pe / target actor /
    alpha in the Q-Flow loss.

Architecture
------------
- Deterministic vector-field MLP (4x512 GELU) with Fourier time embedding.
- Time-free outer ``Q(s, a)`` ensemble-2 + Polyak target.
- Inner ``V(s, x, t)`` ensemble-2 with Fourier-16 time embedding.
- Separate RQL reverse-flow critic only for the warm-start phase.
- No target-actor EMA in bridge/online; current-policy Euler rollouts.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange, repeat

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Value, ensemblize


TRAINING_PHASES = ("rql_warmstart", "qflow_bridge", "qflow_online")


def fourier_time_features(times, frequencies):
    """Paper-style Fourier features of scalar time.

    Args:
        times: ``(..., 1)`` or ``(...)`` in ``[0, 1]``.
        frequencies: ``(num_freqs,)`` positive frequencies.

    Returns:
        ``(..., 2 * num_freqs)`` = ``[sin(2 pi f t), cos(2 pi f t)]``.
    """
    if times.ndim == 1:
        times = times[..., None]
    # (..., 1) * (F,) -> (..., F)
    angles = 2.0 * jnp.pi * times * frequencies
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class FourierFeatures(nn.Module):
    """Fixed Gaussian Fourier features for scalar flow time."""

    features: int = 16
    scale: float = 1.0

    def setup(self):
        if self.features % 2 != 0:
            raise ValueError(f"Fourier features dim must be even, got {self.features}")
        num_freqs = self.features // 2
        # Fixed log-spaced frequencies (not trainable).
        self.frequencies = self.scale * jnp.exp(
            -jnp.log(10000.0) * jnp.arange(num_freqs) / max(num_freqs - 1, 1)
        )

    def __call__(self, times):
        return fourier_time_features(times, self.frequencies)


class DeterministicVectorField(nn.Module):
    """Deterministic flow vector field ``v_theta(s, x, t)`` with Fourier time."""

    hidden_dims: Sequence[int]
    action_dim: int
    fourier_dim: int = 32
    layer_norm: bool = False

    def setup(self):
        self.time_features = FourierFeatures(features=self.fourier_dim)
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, actions, times):
        t_emb = self.time_features(times)
        inputs = jnp.concatenate([observations, actions, t_emb], axis=-1)
        return self.mlp(inputs)


class FourierValue(nn.Module):
    """Inner value ``V(s, x, t)`` with Fourier time embedding."""

    hidden_dims: Sequence[int]
    fourier_dim: int = 16
    layer_norm: bool = True
    num_ensembles: int = 2

    def setup(self):
        self.time_features = FourierFeatures(features=self.fourier_dim)
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        self.value_net = mlp_class(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, actions, times):
        t_emb = self.time_features(times)
        inputs = jnp.concatenate([observations, actions, t_emb], axis=-1)
        return self.value_net(inputs).squeeze(-1)


class QFlowRQLWarmstartAgent(flax.struct.PyTreeNode):
    """RQL warm start → bridge → paper Q-Flow online."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1.0 - expectile))
        return weight * (diff**2)

    def aggregate_q(self, qs):
        if qs.ndim == 1:
            return qs
        q_agg = self.config["q_agg"]
        if q_agg == "min":
            return qs.min(axis=0)
        if q_agg == "mean":
            return qs.mean(axis=0)
        raise ValueError(f"Unsupported q_agg={q_agg!r}")

    def actor_velocity(self, observations, actions, times, *, params=None, module="actor"):
        """Predict deterministic vector field (no Gaussian /.mode())."""
        if times.ndim == 1:
            times = times[..., None]
        return self.network.select(module)(
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
                observations, actions, t_i, params=params, module="actor"
            )
            actions = actions + out * dt
        return actions

    def terminal_q_bellman_loss(self, batch, grad_params, rng):
        """Time-free outer Bellman Q on dataset actions."""
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
        next_actions = self.compute_flow_actions(next_observations, next_noise)
        next_qs = self.network.select("target_terminal_q")(
            next_observations, next_actions
        )
        next_q = self.aggregate_q(next_qs)

        tqt_q = (
            n_rews
            + (self.config["discount"] ** h) * next_q * batch["masks"][-2]
        )
        tqt_q = jax.lax.stop_gradient(tqt_q)

        qs = self.network.select("terminal_q")(
            observations, actions, params=grad_params
        )
        q = self.aggregate_q(qs)
        terminal_q_loss = (jnp.square(qs - tqt_q) * valids).mean()

        diag_params = jax.lax.stop_gradient(grad_params)

        def q_scalar(a_single, obs_single):
            q_ens = self.network.select("terminal_q")(
                obs_single[None], a_single[None], params=diag_params
            )
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

    def rql_actor_critic_loss(self, batch, grad_params, rng):
        """Tuned RQL reverse-flow critic + actor (BC CFM + one-step q_pe)."""
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
                    r_rng, (batch_size // 2,), 0, self.config["flow_steps"] + 1
                )
                / self.config["flow_steps"],
            ],
            0,
        )
        d_b = d / self.config["flow_steps"]

        actions = rearrange(batch["actions"][: self.config["h"]], "h b d -> b (h d)")
        observations = batch["observations"][0]

        x_f = jnp.copy(actions)
        f = jnp.ones((batch_size, 1))
        for _ in range(self.config["flow_steps"]):
            out = self.actor_velocity(observations, x_f, f, params=None)
            x_f = x_f - out * d_b[..., None]
            f = f - d_b[..., None]

        state = jnp.concatenate(
            [observations, jax.lax.stop_gradient(x_f), f], axis=-1
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
            + (self.config["discount"] ** self.config["h"])
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
        x_1 = actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1.0 - t) * x_0 + t * x_1
        tgt = x_1 - x_0

        pred = self.actor_velocity(
            observations, x_t, t, params=grad_params
        )
        step = 1.0 / self.config["flow_steps"]
        q_pe = self.network.select("value")(
            jnp.concatenate(
                [
                    observations,
                    x_t + pred * jnp.minimum(step, 1.0 - t),
                    jnp.clip(t + step, max=1.0),
                ],
                axis=-1,
            )
        )
        q_pe = q_pe.mean(axis=0)

        ac_mask = repeat(
            1 - rs_terminals[:-1],
            "h b -> b (h r)",
            r=action_dim // self.config["h"],
        )
        bc_loss = (jnp.square(pred - tgt) * ac_mask).mean()
        actor_loss = -(q_pe * valids).mean()
        rql_loss = actor_loss + bc_loss * self.config["alpha"] + critic_loss

        info = {
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
        return rql_loss, info, rng

    def qflow_policy_loss(
        self,
        batch,
        grad_params,
        rng,
        *,
        blend,
        actor_coef,
    ):
        """Outer Q + inner V + blended CFM→Q-Flow velocity matching."""
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

        # --- 3. Actor: blend RQL CFM target → exact Q-Flow matching ---
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
        blend = jnp.asarray(blend, dtype=cfm_vec.dtype)
        # blend=0 → CFM (RQL BC target); blend=1 → exact Q-Flow target.
        v_target = cfm_vec + blend * (1.0 / lam) * grad_x
        v_target = jax.lax.stop_gradient(v_target)

        pred = self.actor_velocity(
            observations, x_t, t, params=grad_params
        )
        ac_mask = repeat(
            1 - rs_terminals[:-1],
            "h b -> b (h r)",
            r=action_dim // h,
        )
        actor_loss = (jnp.square(pred - v_target) * ac_mask).mean()
        actor_coef = jnp.asarray(actor_coef, dtype=actor_loss.dtype)

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
            "bc_loss": jnp.square(pred - cfm_vec).mean(),
            "cfm_target_norm": cfm_target_norm,
            "inner_grad_norm": inner_grad_norm,
            "inner_grad_to_cfm_ratio": inner_grad_to_cfm_ratio,
            "pred_velocity_norm": pred_velocity_norm,
            "actor_coef": actor_coef,
            "qflow_bridge_blend": blend,
            **iv_info,
        }
        return total_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        phase = self.config["training_phase"]

        if phase == "rql_warmstart":
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
                observations, actions, grad_params, iv_rng
            )
            total_loss = rql_loss + terminal_q_loss + iv_loss
            info = {**rql_info, **tq_info, **iv_info, "total_loss": total_loss}
            return total_loss, info

        if phase == "qflow_bridge":
            return self.qflow_policy_loss(
                batch,
                grad_params,
                rng,
                blend=self.config["qflow_bridge_blend"],
                actor_coef=self.config["qflow_actor_coef"],
            )

        if phase == "qflow_online":
            # Exact Algorithm-1 target (blend=1); no RQL terms.
            return self.qflow_policy_loss(
                batch,
                grad_params,
                rng,
                blend=1.0,
                actor_coef=self.config["qflow_actor_coef"],
            )

        raise ValueError(f"Unknown training_phase={phase!r}")

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
        phase = self.config["training_phase"]

        # Outer Q target Polyak in all phases.
        self.target_update(
            new_network, "terminal_q", d=self.config["terminal_q_tau"]
        )

        if phase == "rql_warmstart":
            self.target_update(new_network, "value", d=self.config["tau"])
            self.target_update(new_network, "actor", d=1.0 - self.config["ema"])
        else:
            # Bridge / online: no reverse-critic or target-actor EMA.
            # Freeze actor when coef=0 (probe / readiness gate).
            skip_actor = float(self.config["qflow_actor_coef"]) == 0.0
            if skip_actor:
                # Restore pre-update actor params (optimizer still steps other modules).
                new_network.params["modules_actor"] = self.network.params[
                    "modules_actor"
                ]

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

    def replace_training_phase(self, training_phase):
        if training_phase not in TRAINING_PHASES:
            raise ValueError(f"Unknown training_phase={training_phase!r}")
        new_config = dict(self.config)
        new_config["training_phase"] = training_phase
        return self.replace(config=flax.core.FrozenDict(**new_config))

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        config = dict(config)
        phase = config.get("training_phase", "rql_warmstart")
        if phase not in TRAINING_PHASES:
            raise ValueError(f"Unknown training_phase={phase!r}")
        config["training_phase"] = phase
        config.setdefault("qflow_actor_coef", 1.0)
        config.setdefault("qflow_bridge_blend", 1.0)
        config.setdefault("terminal_q_tau", 0.005)
        config.setdefault("actor_fourier_dim", 32)
        config.setdefault("inner_fourier_dim", 16)
        config.setdefault("rql_ensemble_ct", config.get("rql_ensemble_ct", 10))
        config.setdefault("ensemble_ct", 2)
        config.setdefault("inner_ensemble_ct", 2)

        ex_actions_h = jnp.concatenate([ex_actions] * config["h"], -1)
        ex_times = jnp.zeros((*ex_observations.shape[:-1], 1), dtype=ex_actions.dtype)
        action_dim = ex_actions_h.shape[-1]
        # RQL reverse critic still consumes concatenated (s, x, t).
        ex_rql_in = jnp.concatenate(
            [ex_observations, ex_actions_h, ex_times], axis=-1
        )

        rql_value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["rql_ensemble_ct"],
        )
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
            value=(rql_value_def, (ex_rql_in,)),
            target_value=(copy.deepcopy(rql_value_def), (ex_rql_in,)),
            terminal_q=(terminal_q_def, terminal_q_args),
            target_terminal_q=(copy.deepcopy(terminal_q_def), terminal_q_args),
            actor=(actor_def, actor_args),
            target_actor=(copy.deepcopy(actor_def), actor_args),
            inner_value=(inner_value_def, inner_args),
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
    """Default config: tuned RQL warm start → paper-like Q-Flow online."""
    config = mlc.ConfigDict(
        dict(
            agent_name="qflow_rql_warmstart",
            training_phase="rql_warmstart",
            h=1,
            alpha=0.3,
            expectile=0.5,
            # Outer Q / inner V (paper standard).
            ensemble_ct=2,
            inner_ensemble_ct=2,
            # RQL reverse-flow critic ensemble (warm start only).
            rql_ensemble_ct=10,
            rho=0.0,
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
            tau=0.005,
            terminal_q_tau=0.005,
            ema=0.999,
            flow_steps=10,
            q_agg="mean",
            qflow_lambda=1.0,
            qflow_actor_coef=1.0,
            qflow_bridge_blend=1.0,
        )
    )
    return config
