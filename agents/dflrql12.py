"""Decoupled ConsensusFlow with an RL policy in flow-latent space."""

from __future__ import annotations

import copy
from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange

from agents.dflrql11 import DFLRQL11Agent
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import Actor, MLP, Value


class LatentGaussianPolicy(nn.Module):
    """State-conditioned Gaussian initialized exactly at ``N(0, I)``."""

    hidden_dims: Any
    latent_dim: int
    layer_norm: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations):
        features = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )(observations)
        means = nn.Dense(
            self.latent_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(features)
        log_stds = nn.Dense(
            self.latent_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(features)
        return means, jnp.clip(log_stds, self.log_std_min, self.log_std_max)


class DFLRQL12Agent(DFLRQL11Agent):
    """BC-only flow decoder ``v`` plus KL-regularized latent RL policy ``G_z``."""

    def _latent_critic_scores(
        self,
        name,
        observations,
        latents,
        params=None,
    ):
        return self.network.select(name)(
            observations,
            latents,
            params=params,
        )

    def _latent_distribution(self, name, observations, params=None):
        return self.network.select(name)(observations, params=params)

    def _sample_latent_candidates(
        self,
        observations,
        rng,
        *,
        policy_name,
        candidate_count,
        params=None,
    ):
        means, log_stds = self._latent_distribution(
            policy_name,
            observations,
            params=params,
        )
        noises = jax.random.normal(
            rng,
            (candidate_count, *means.shape),
        )
        latents = means[None, ...] + jnp.exp(log_stds)[None, ...] * noises
        return latents, means, log_stds

    def reverse_flow_latents(self, observations, endpoint_actions):
        """Integrate the frozen BC ODE backward from actions to initial noise."""

        latents = endpoint_actions
        for reverse_idx in reversed(range(self.config["flow_steps"])):
            times = jnp.full(
                (*observations.shape[:-1], 1),
                (reverse_idx + 1) / self.config["flow_steps"],
                dtype=latents.dtype,
            )
            actor_input = jnp.concatenate(
                [observations, latents, times],
                axis=-1,
            )
            velocity = jax.lax.stop_gradient(
                self.network.select("target_actor")(actor_input).mode()
            )
            latents = latents - velocity / self.config["flow_steps"]
        return jax.lax.stop_gradient(latents)

    def _best_target_latent_q(self, observations, rng):
        candidate_latents, _, _ = self._sample_latent_candidates(
            observations,
            rng,
            policy_name="target_latent_policy",
            candidate_count=self.config["target_candidates"],
        )
        candidate_observations = jnp.broadcast_to(
            observations[None, ...],
            (
                self.config["target_candidates"],
                *observations.shape,
            ),
        )
        candidate_qs = self._latent_critic_scores(
            "target_latent_critic",
            candidate_observations,
            candidate_latents,
        )
        candidate_q = self._aggregate_target_q(candidate_qs)
        return candidate_q.max(axis=0), candidate_latents, candidate_q

    def _latent_critic_loss(self, batch, grad_params, rng):
        _, discounted_rewards, valids, bootstrap_mask = self._valids_and_returns(
            batch
        )
        next_q, _, _ = self._best_target_latent_q(
            batch["observations"][-1],
            rng,
        )
        target_q = jax.lax.stop_gradient(
            discounted_rewards
            + self.config["discount"] ** self.config["h"]
            * bootstrap_mask
            * next_q
        )

        replay_actions = self._chunk_actions(batch)
        replay_latents = self.reverse_flow_latents(
            batch["observations"][0],
            replay_actions,
        )
        qs = self._latent_critic_scores(
            "latent_critic",
            batch["observations"][0],
            replay_latents,
            params=grad_params,
        )
        critic_per = jnp.square(qs - target_q[None, :]).mean(axis=0)
        valid_count = valids.sum() + 1e-6
        critic_loss = (critic_per * valids).sum() / valid_count
        prior_nll = 0.5 * (
            jnp.square(replay_latents) + jnp.log(2.0 * jnp.pi)
        ).sum(axis=-1)
        return critic_loss, {
            "latent_critic_loss": critic_loss,
            "latent_q_mean": qs.mean(),
            "latent_q_max": qs.max(),
            "latent_q_min": qs.min(),
            "latent_target_q_mean": (target_q * valids).sum() / valid_count,
            "replay_latent_norm": jnp.linalg.norm(
                replay_latents, axis=-1
            ).mean(),
            "replay_latent_prior_nll": prior_nll.mean(),
            "valid_fraction": valids.mean(),
        }

    def _latent_policy_loss(self, batch, grad_params, rng):
        observations = batch["observations"][0]
        latents, means, log_stds = self._sample_latent_candidates(
            observations,
            rng,
            policy_name="latent_policy",
            candidate_count=1,
            params=grad_params,
        )
        latents = latents[0]
        qs = self._latent_critic_scores(
            "latent_critic",
            observations,
            latents,
        )
        q = self._aggregate_target_q(qs)
        q_scale = jax.lax.stop_gradient(
            jnp.maximum(jnp.abs(q).mean(), self.config["q_normalization_floor"])
        )
        q_loss = -q.mean()
        if bool(self.config["normalize_q_loss"]):
            q_loss = q_loss / q_scale

        variances = jnp.exp(2.0 * log_stds)
        kl_per = 0.5 * (
            jnp.square(means) + variances - 1.0 - 2.0 * log_stds
        ).sum(axis=-1)
        latent_kl = kl_per.mean()
        policy_loss = q_loss + self.config["latent_kl_beta"] * latent_kl
        return policy_loss, {
            "latent_policy_loss": policy_loss,
            "latent_policy_q_loss": q_loss,
            "latent_policy_q_mean": q.mean(),
            "latent_kl": latent_kl,
            "latent_kl_loss": self.config["latent_kl_beta"] * latent_kl,
            "latent_mean_norm": jnp.linalg.norm(means, axis=-1).mean(),
            "latent_std_mean": jnp.exp(log_stds).mean(),
            "sampled_latent_norm": jnp.linalg.norm(
                latents, axis=-1
            ).mean(),
            "q_scale": q_scale,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, critic_rng, bc_rng, policy_rng = jax.random.split(rng, 4)
        critic_loss, critic_info = self._latent_critic_loss(
            batch,
            grad_params,
            critic_rng,
        )
        bc_loss, bc_info = self._bc_loss(batch, grad_params, bc_rng)
        policy_loss, policy_info = self._latent_policy_loss(
            batch,
            grad_params,
            policy_rng,
        )

        actor_delay = jnp.asarray(self.config["actor_delay"], dtype=jnp.int32)
        policy_update = (self.network.step % actor_delay == 0).astype(jnp.float32)
        bc_weight = jnp.asarray(
            0.0 if bool(self.config["freeze_v"]) else self.config["bc_coef"],
            dtype=jnp.float32,
        )
        total_loss = (
            critic_loss + bc_weight * bc_loss + policy_update * policy_loss
        )
        return total_loss, {
            "total_loss": total_loss,
            **critic_info,
            **bc_info,
            **policy_info,
            "bc_weight": bc_weight,
            "freeze_v": jnp.asarray(
                bool(self.config["freeze_v"]), dtype=jnp.float32
            ),
            "latent_policy_update": policy_update,
        }

    @jax.jit
    def update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        policy_update = self.network.step % self.config["actor_delay"] == 0
        policy_key = "modules_latent_policy"
        new_network.params[policy_key] = self._select_tree(
            policy_update,
            new_network.params[policy_key],
            self.network.params[policy_key],
        )

        if bool(self.config["freeze_v"]):
            for module_name in ("actor", "target_actor"):
                key = f"modules_{module_name}"
                new_network.params[key] = self.network.params[key]

        self._target_update(
            new_network,
            "latent_critic",
            self.config["tau"],
        )
        self._target_update(
            new_network,
            "latent_policy",
            self.config["tau"],
            predicate=policy_update,
        )
        if not bool(self.config["freeze_v"]):
            self._target_update(
                new_network,
                "actor",
                1.0 - self.config["ema"],
            )

        return self.replace(network=new_network, rng=new_rng), info

    def _select_deployment_latent(
        self,
        observations,
        rng,
        policy_name,
        candidate_count,
    ):
        candidates, _, _ = self._sample_latent_candidates(
            observations,
            rng,
            policy_name=policy_name,
            candidate_count=candidate_count,
        )
        candidate_observations = jnp.broadcast_to(
            observations[None, ...],
            (candidate_count, *observations.shape),
        )
        candidate_qs = self._latent_critic_scores(
            "target_latent_critic",
            candidate_observations,
            candidates,
        )
        candidate_q = self._aggregate_target_q(candidate_qs)
        best_indices = candidate_q.argmax(axis=0)
        batch_indices = jnp.arange(observations.shape[0])
        return candidates[best_indices, batch_indices]

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        observations = jnp.atleast_2d(obs)[-1:]
        if bool(self.config["disable_rl_policy"]):
            latent = jax.random.normal(
                seed,
                (1, self.config["action_dim"]),
            )
            actor_name = "actor" if temperature > 0 else "target_actor"
            actions = self._flow_actions(observations, latent, actor_name)
            return rearrange(
                actions[0],
                "(h d) -> h d",
                h=self.config["h"],
            )
        policy_name = (
            "latent_policy" if temperature > 0 else "target_latent_policy"
        )
        candidate_count = (
            1
            if temperature > 0
            else int(self.config["deployment_candidates"])
        )
        latent = self._select_deployment_latent(
            observations,
            seed,
            policy_name,
            candidate_count,
        )
        actor_name = "actor" if temperature > 0 else "target_actor"
        actions = self._flow_actions(observations, latent, actor_name)
        return rearrange(actions[0], "(h d) -> h d", h=self.config["h"])

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ex_actions = jnp.concatenate([ex_actions] * config["h"], axis=-1)
        ex_times = ex_actions[..., :1]
        ex_actor_input = jnp.concatenate(
            [ex_observations, ex_actions, ex_times],
            axis=-1,
        )
        action_dim = ex_actions.shape[-1]

        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=False,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=1.0,
        )
        latent_policy_def = LatentGaussianPolicy(
            hidden_dims=config["latent_policy_hidden_dims"],
            latent_dim=action_dim,
            layer_norm=config["layer_norm"],
            log_std_min=config["latent_log_std_min"],
            log_std_max=config["latent_log_std_max"],
        )
        latent_critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )

        network_info = {
            "actor": (actor_def, (ex_actor_input,)),
            "target_actor": (copy.deepcopy(actor_def), (ex_actor_input,)),
            "latent_policy": (latent_policy_def, (ex_observations,)),
            "target_latent_policy": (
                copy.deepcopy(latent_policy_def),
                (ex_observations,),
            ),
            "latent_critic": (
                latent_critic_def,
                (ex_observations, ex_actions),
            ),
            "target_latent_critic": (
                copy.deepcopy(latent_critic_def),
                (ex_observations, ex_actions),
            ),
        }
        networks = {name: definition for name, (definition, _) in network_info.items()}
        network_args = {name: args for name, (_, args) in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        params = network_def.init(init_rng, **network_args)["params"]
        params["modules_target_actor"] = params["modules_actor"]
        params["modules_target_latent_policy"] = params["modules_latent_policy"]
        params["modules_target_latent_critic"] = params["modules_latent_critic"]
        network = TrainState.create(network_def, params, tx=network_tx)

        config["action_dim"] = action_dim
        config["discount_mul"] = jnp.asarray(
            config["discount"]
            ** jnp.asarray(list(range(config["h"])) + [jnp.inf])
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    return mlc.ConfigDict(
        {
            "agent_name": "dflrql12",
            "h": 3,
            "lr": 3e-4,
            "batch_size": 256,
            "discount": 0.99,
            "tau": 0.005,
            "ema": 0.999,
            "flow_steps": 10,
            "actor_hidden_dims": (512, 512, 512, 512),
            "actor_layer_norm": False,
            "value_hidden_dims": (512, 512, 512, 512),
            "latent_policy_hidden_dims": (512, 512, 512, 512),
            "layer_norm": True,
            "ensemble_ct": 10,
            "q_agg": "min",
            "rho": 0.0,
            "bc_coef": 10.0,
            "freeze_v": True,
            "normalize_q_loss": True,
            "q_normalization_floor": 1.0,
            "latent_kl_beta": 0.01,
            "latent_log_std_min": -5.0,
            "latent_log_std_max": 1.0,
            "actor_delay": 2,
            "target_candidates": 4,
            "deployment_candidates": 4,
            "disable_rl_policy": False,
        }
    )
