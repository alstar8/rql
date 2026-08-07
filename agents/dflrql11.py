"""Decoupled ConsensusFlow with a BC flow and an RL endpoint refiner.

The behavior vector field ``v`` is optimized only by flow matching.  Policy
improvement belongs exclusively to an instance-conditioned endpoint refiner
``G(s, a_v)`` trained with deterministic policy gradients under an adaptive
residual trust-region constraint.
"""

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
from einops import rearrange, repeat

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, MLP, Value


class EndpointRefiner(nn.Module):
    """Instance-conditioned residual ``(s, a_v) -> delta_a``."""

    hidden_dims: Any
    action_dim: int
    layer_norm: bool = True
    zero_init: bool = True

    @nn.compact
    def __call__(self, inputs):
        features = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )(inputs)
        kernel_init = (
            nn.initializers.zeros
            if self.zero_init
            else nn.initializers.variance_scaling(1.0, "fan_avg", "uniform")
        )
        return nn.Dense(
            self.action_dim,
            kernel_init=kernel_init,
            bias_init=nn.initializers.zeros,
        )(features)


class LogLagrange(nn.Module):
    """Trainable logarithm of the residual trust-region multiplier."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        return self.param(
            "value",
            lambda rng: jnp.asarray(jnp.log(self.init_value), dtype=jnp.float32),
        )


class DFLRQL11Agent(flax.struct.PyTreeNode):
    """BC-only flow ``v`` plus RL-only endpoint refinement ``G``."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _chunk_actions(self, batch):
        return rearrange(
            batch["actions"][: self.config["h"]],
            "h b d -> b (h d)",
        )

    def _critic_input(self, observations, flat_actions):
        # The endpoint marker preserves shape compatibility with dflrql9's
        # t-conditioned value ensemble while making this a standard Q(s, a).
        endpoint = jnp.ones((*flat_actions.shape[:-1], 1), dtype=flat_actions.dtype)
        return jnp.concatenate([observations, flat_actions, endpoint], axis=-1)

    def _critic_scores(self, name, observations, flat_actions, params=None):
        return self.network.select(name)(
            self._critic_input(observations, flat_actions),
            params=params,
        )

    def _aggregate_target_q(self, qs):
        q_agg = self.config["q_agg"]
        if q_agg == "min":
            return qs.min(axis=0)
        if q_agg == "mean":
            return qs.mean(axis=0)
        if q_agg == "pessimistic":
            return qs.mean(axis=0) - self.config["rho"] * qs.std(axis=0)
        raise ValueError(f"unsupported q_agg={q_agg!r}")

    def _flow_actions(self, observations, noises, actor_name):
        actions = noises
        for flow_idx in range(self.config["flow_steps"]):
            times = jnp.full(
                (*observations.shape[:-1], 1),
                flow_idx / self.config["flow_steps"],
                dtype=actions.dtype,
            )
            actor_input = jnp.concatenate(
                [observations, actions, times],
                axis=-1,
            )
            velocity = self.network.select(actor_name)(actor_input).mode()
            actions = actions + velocity / self.config["flow_steps"]
        return jnp.clip(actions, -1.0, 1.0)

    def _refiner_input(self, observations, base_actions):
        endpoint = jnp.ones((*base_actions.shape[:-1], 1), dtype=base_actions.dtype)
        return jnp.concatenate(
            [observations, base_actions, endpoint],
            axis=-1,
        )

    def _consensus_trust(self, observations, base_actions):
        """Directional agreement of target-critic action gradients."""

        ensemble_ct = int(self.config["ensemble_ct"])

        def member_gradient(member_idx):
            def member_sum(actions):
                qs = self._critic_scores(
                    "target_critic",
                    observations,
                    actions,
                )
                return qs[member_idx].sum()

            return jax.grad(member_sum)(base_actions)

        gradients = jax.vmap(member_gradient)(jnp.arange(ensemble_ct))
        norms = jnp.linalg.norm(gradients, axis=-1, keepdims=True)
        directions = gradients / (norms + 1e-6)
        consensus = jnp.linalg.norm(directions.mean(axis=0), axis=-1, keepdims=True)
        return jax.lax.stop_gradient(jnp.clip(consensus, 0.0, 1.0))

    def _refined_actions(
        self,
        observations,
        base_actions,
        *,
        refiner_name,
        params=None,
        use_consensus=None,
    ):
        raw_delta = self.network.select(refiner_name)(
            self._refiner_input(observations, base_actions),
            params=params,
        )
        if use_consensus is None:
            use_consensus = bool(self.config["consensus_trust"])
        if use_consensus:
            trust = self._consensus_trust(observations, base_actions)
        else:
            trust = jnp.ones((*raw_delta.shape[:-1], 1), dtype=raw_delta.dtype)
        delta = raw_delta * trust
        actions = jnp.clip(base_actions + delta, -1.0, 1.0)
        return actions, raw_delta, delta, trust

    def _valids_and_returns(self, batch):
        shifted_terminals = jnp.concatenate(
            [
                jnp.zeros_like(batch["terminals"][:1]),
                batch["terminals"][:-1],
            ],
            axis=0,
        )
        discounted_rewards = (
            batch["rewards"]
            * self.config["discount_mul"][..., None]
            * (1.0 - shifted_terminals)
        ).sum(axis=0)
        terminal_count = shifted_terminals.sum(axis=0)
        valids = (terminal_count <= 1).astype(discounted_rewards.dtype)
        bootstrap_mask = batch["masks"][-2]
        return shifted_terminals, discounted_rewards, valids, bootstrap_mask

    def _critic_loss(self, batch, grad_params, rng):
        batch_size = self.config["batch_size"]
        action_dim = self.config["action_dim"]
        rng, noise_rng, smooth_rng = jax.random.split(rng, 3)

        _, discounted_rewards, valids, bootstrap_mask = self._valids_and_returns(
            batch
        )
        next_observations = batch["observations"][-1]
        next_noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        next_base = jax.lax.stop_gradient(
            self._flow_actions(next_observations, next_noises, "target_actor")
        )
        next_actions, _, _, _ = self._refined_actions(
            next_observations,
            next_base,
            refiner_name="target_refiner",
        )
        target_noise = jax.random.normal(smooth_rng, next_actions.shape)
        target_noise = jnp.clip(
            target_noise * self.config["target_policy_noise"],
            -self.config["target_noise_clip"],
            self.config["target_noise_clip"],
        )
        next_actions = jnp.clip(next_actions + target_noise, -1.0, 1.0)
        next_qs = self._critic_scores(
            "target_critic",
            next_observations,
            next_actions,
        )
        next_q = self._aggregate_target_q(next_qs)
        target_q = jax.lax.stop_gradient(
            discounted_rewards
            + self.config["discount"] ** self.config["h"]
            * bootstrap_mask
            * next_q
        )

        flat_actions = self._chunk_actions(batch)
        qs = self._critic_scores(
            "critic",
            batch["observations"][0],
            flat_actions,
            params=grad_params,
        )
        critic_per = jnp.square(qs - target_q[None, :]).mean(axis=0)
        valid_count = valids.sum() + 1e-6
        critic_loss = (critic_per * valids).sum() / valid_count
        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": (target_q * valids).sum() / valid_count,
            "target_q_max": target_q.max(),
            "target_q_min": target_q.min(),
            "valid_fraction": valids.mean(),
        }

    def _bc_loss(self, batch, grad_params, rng):
        batch_size = self.config["batch_size"]
        action_dim = self.config["action_dim"]
        rng, noise_rng, time_rng = jax.random.split(rng, 3)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        targets = self._chunk_actions(batch)
        times = jax.random.uniform(time_rng, (batch_size, 1))
        flow_points = (1.0 - times) * noises + times * targets
        target_velocity = targets - noises
        actor_input = jnp.concatenate(
            [batch["observations"][0], flow_points, times],
            axis=-1,
        )
        velocity = self.network.select("actor")(
            actor_input,
            params=grad_params,
        ).mode()

        shifted_terminals, _, _, _ = self._valids_and_returns(batch)
        action_mask = repeat(
            1.0 - shifted_terminals[:-1],
            "h b -> b (h r)",
            r=action_dim // self.config["h"],
        )
        bc_loss = (jnp.square(velocity - target_velocity) * action_mask).mean()
        return bc_loss, {
            "bc_loss": bc_loss,
            "velocity_norm": jnp.linalg.norm(velocity, axis=-1).mean(),
            "target_velocity_norm": jnp.linalg.norm(
                target_velocity, axis=-1
            ).mean(),
        }

    def _refiner_loss(self, batch, grad_params, rng):
        batch_size = self.config["batch_size"]
        action_dim = self.config["action_dim"]
        noise = jax.random.normal(rng, (batch_size, action_dim))
        observations = batch["observations"][0]
        base_actions = jax.lax.stop_gradient(
            self._flow_actions(observations, noise, "target_actor")
        )
        refined_actions, raw_delta, delta, trust = self._refined_actions(
            observations,
            base_actions,
            refiner_name="refiner",
            params=grad_params,
        )

        qs = self._critic_scores("critic", observations, refined_actions)
        q = qs.mean(axis=0)
        q_scale = jax.lax.stop_gradient(
            jnp.maximum(jnp.abs(q).mean(), self.config["q_normalization_floor"])
        )
        q_loss = -q.mean()
        if bool(self.config["normalize_q_loss"]):
            q_loss = q_loss / q_scale

        raw_residual_mse = jnp.square(raw_delta).mean()
        residual_mse = jnp.square(delta).mean()
        log_alpha = self.network.select("log_alpha")(params=grad_params)
        alpha = jnp.exp(
            jnp.clip(
                log_alpha,
                self.config["log_alpha_min"],
                self.config["log_alpha_max"],
            )
        )
        trust_region_loss = jax.lax.stop_gradient(alpha) * residual_mse
        constraint_error = jax.lax.stop_gradient(
            residual_mse - self.config["target_divergence"]
        )
        alpha_loss = -log_alpha * constraint_error
        refiner_loss = q_loss + trust_region_loss + alpha_loss

        base_q = self._critic_scores("critic", observations, base_actions).mean(
            axis=0
        )
        predicted_advantage = q - jax.lax.stop_gradient(base_q)
        return refiner_loss, {
            "refiner_loss": refiner_loss,
            "refiner_q_loss": q_loss,
            "refiner_q_mean": q.mean(),
            "base_q_mean": base_q.mean(),
            "predicted_advantage": predicted_advantage.mean(),
            "raw_residual_mse": raw_residual_mse,
            "residual_mse": residual_mse,
            "residual_rms": jnp.sqrt(residual_mse + 1e-12),
            "residual_norm": jnp.linalg.norm(delta, axis=-1).mean(),
            "raw_residual_norm": jnp.linalg.norm(raw_delta, axis=-1).mean(),
            "trust_region_loss": trust_region_loss,
            "constraint_error": constraint_error,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "consensus_trust": trust.mean(),
            "q_scale": q_scale,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        rng, critic_rng, bc_rng, refiner_rng = jax.random.split(rng, 4)
        critic_loss, critic_info = self._critic_loss(
            batch, grad_params, critic_rng
        )
        bc_loss, bc_info = self._bc_loss(batch, grad_params, bc_rng)
        refiner_loss, refiner_info = self._refiner_loss(
            batch, grad_params, refiner_rng
        )

        actor_delay = jnp.asarray(self.config["actor_delay"], dtype=jnp.int32)
        refiner_update = (self.network.step % actor_delay == 0).astype(jnp.float32)
        bc_weight = jnp.asarray(
            0.0 if bool(self.config["freeze_v"]) else self.config["bc_coef"],
            dtype=jnp.float32,
        )
        total_loss = (
            critic_loss
            + bc_weight * bc_loss
            + refiner_update * refiner_loss
        )
        return total_loss, {
            "total_loss": total_loss,
            **critic_info,
            **bc_info,
            **refiner_info,
            "bc_weight": bc_weight,
            "freeze_v": jnp.asarray(
                bool(self.config["freeze_v"]), dtype=jnp.float32
            ),
            "refiner_update": refiner_update,
        }

    @staticmethod
    def _select_tree(predicate, new_tree, old_tree):
        return jax.tree_util.tree_map(
            lambda new, old: jnp.where(predicate, new, old),
            new_tree,
            old_tree,
        )

    @staticmethod
    def _target_update(network, module_name, tau, predicate=True):
        source = network.params[f"modules_{module_name}"]
        target_key = f"modules_target_{module_name}"
        old_target = network.params[target_key]
        candidate = jax.tree_util.tree_map(
            lambda source_param, target_param: (
                tau * source_param + (1.0 - tau) * target_param
            ),
            source,
            old_target,
        )
        network.params[target_key] = DFLRQL11Agent._select_tree(
            predicate,
            candidate,
            old_target,
        )

    @jax.jit
    def update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        refiner_update = self.network.step % self.config["actor_delay"] == 0

        # Delayed deterministic-policy updates must not move G via Adam momentum.
        for module_name in ("refiner", "log_alpha"):
            key = f"modules_{module_name}"
            new_network.params[key] = self._select_tree(
                refiner_update,
                new_network.params[key],
                self.network.params[key],
            )

        if bool(self.config["freeze_v"]):
            for module_name in ("actor", "target_actor"):
                key = f"modules_{module_name}"
                new_network.params[key] = self.network.params[key]

        self._target_update(
            new_network,
            "critic",
            self.config["tau"],
        )
        self._target_update(
            new_network,
            "refiner",
            self.config["tau"],
            predicate=refiner_update,
        )
        if not bool(self.config["freeze_v"]):
            self._target_update(
                new_network,
                "actor",
                1.0 - self.config["ema"],
            )

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(
        self,
        observations,
        noise,
        seed=None,
        temperature=0.0,
    ):
        del seed
        actor_name = "actor" if temperature > 0 else "target_actor"
        return self._flow_actions(observations, noise, actor_name)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        _, noise_rng = jax.random.split(seed)
        observations = jnp.atleast_2d(obs)[-1:]
        noise = jax.random.normal(
            noise_rng,
            (1, self.config["action_dim"]),
        )
        actor_name = "actor" if temperature > 0 else "target_actor"
        refiner_name = "refiner" if temperature > 0 else "target_refiner"
        base_actions = self._flow_actions(observations, noise, actor_name)
        if bool(self.config["disable_rl_policy"]):
            return rearrange(
                base_actions[0],
                "(h d) -> h d",
                h=self.config["h"],
            )
        actions, _, _, _ = self._refined_actions(
            observations,
            base_actions,
            refiner_name=refiner_name,
        )
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
        ex_critic_input = jnp.concatenate(
            [ex_observations, ex_actions, jnp.ones_like(ex_times)],
            axis=-1,
        )
        ex_refiner_input = ex_critic_input
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
        critic_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        refiner_def = EndpointRefiner(
            hidden_dims=config["refiner_hidden_dims"],
            action_dim=action_dim,
            layer_norm=config["layer_norm"],
            zero_init=config["zero_init_refiner"],
        )
        log_alpha_def = LogLagrange(config["initial_alpha"])

        network_info = {
            "actor": (actor_def, (ex_actor_input,)),
            "target_actor": (copy.deepcopy(actor_def), (ex_actor_input,)),
            "critic": (critic_def, (ex_critic_input,)),
            "target_critic": (copy.deepcopy(critic_def), (ex_critic_input,)),
            "refiner": (refiner_def, (ex_refiner_input,)),
            "target_refiner": (
                copy.deepcopy(refiner_def),
                (ex_refiner_input,),
            ),
            "log_alpha": (log_alpha_def, ()),
        }
        networks = {name: definition for name, (definition, _) in network_info.items()}
        network_args = {name: args for name, (_, args) in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        params = network_def.init(init_rng, **network_args)["params"]
        params["modules_target_actor"] = params["modules_actor"]
        params["modules_target_critic"] = params["modules_critic"]
        params["modules_target_refiner"] = params["modules_refiner"]
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
            "agent_name": "dflrql11",
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
            "refiner_hidden_dims": (512, 512, 512, 512),
            "layer_norm": True,
            "ensemble_ct": 10,
            "q_agg": "min",
            "rho": 0.0,
            "bc_coef": 10.0,
            "freeze_v": True,
            "zero_init_refiner": True,
            "normalize_q_loss": True,
            "q_normalization_floor": 1.0,
            "target_divergence": 1e-3,
            "initial_alpha": 1.0,
            "log_alpha_min": -10.0,
            "log_alpha_max": 10.0,
            "actor_delay": 2,
            "target_policy_noise": 0.1,
            "target_noise_clip": 0.2,
            "consensus_trust": False,
            "disable_rl_policy": False,
        }
    )
