"""ConsensusFlowRL: CF no-CRF teacher -> AR OATTok student -> FastSAC.

The offline update is the paired QuantizedDFLRQL9/AR distillation objective.
Online optimization is a JAX-native adaptation of the Holosoma FastSAC recipe:
categorical distributional critics, delayed actor updates, mean-Q aggregation,
Polyak targets, and entropy-temperature tuning.  Holosoma's PyTorch Gaussian
actor is deliberately not reused because its density and checkpoint contracts
do not match the autoregressive categorical policy.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange

from agents.discrete_ar_iql import BOS_ID
from agents.discrete_ar_qdfl_distill import (
    DiscreteARQdflDistillAgent,
    get_config as get_ar_qdfl_config,
)
from agents.oattok_jax import FSQ_LEVELS, build_codebook, indices_to_codes
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field


class CategoricalQNetwork(nn.Module):
    """Independent C51 critics over decoded OATTok actions."""

    ensemble_ct: int
    hidden_dims: Sequence[int]
    num_atoms: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        outputs = []
        for ensemble_idx in range(self.ensemble_ct):
            h = x
            for layer_idx, width in enumerate(self.hidden_dims):
                h = nn.Dense(
                    int(width),
                    name=f"q{ensemble_idx}_dense{layer_idx}",
                )(h)
                if self.layer_norm:
                    h = nn.LayerNorm(
                        name=f"q{ensemble_idx}_ln{layer_idx}"
                    )(h)
                h = nn.silu(h)
            outputs.append(
                nn.Dense(
                    self.num_atoms,
                    name=f"q{ensemble_idx}_atoms",
                )(h)
            )
        return jnp.stack(outputs, axis=0)


def categorical_projection(
    probabilities: jnp.ndarray,
    rewards: jnp.ndarray,
    discounts: jnp.ndarray,
    support: jnp.ndarray,
) -> jnp.ndarray:
    """Project an ensemble C51 target onto a fixed equally spaced support.

    Args:
        probabilities: ``(E, B, A)`` categorical probabilities.
        rewards: ``(B,)`` entropy-adjusted immediate rewards.
        discounts: ``(B,)`` bootstrap discounts, already terminal-masked.
        support: ``(A,)`` monotonically increasing support.
    """
    num_atoms = support.shape[0]
    v_min = support[0]
    v_max = support[-1]
    delta = (v_max - v_min) / jnp.asarray(num_atoms - 1, jnp.float32)
    target = rewards[:, None] + discounts[:, None] * support[None, :]
    target = jnp.clip(target, v_min, v_max)
    locations = (target - v_min) / delta
    lower = jnp.floor(locations).astype(jnp.int32)
    upper = jnp.ceil(locations).astype(jnp.int32)
    equal = lower == upper
    lower_weight = jnp.where(equal, 1.0, upper.astype(jnp.float32) - locations)
    upper_weight = jnp.where(equal, 0.0, locations - lower.astype(jnp.float32))
    lower_oh = jax.nn.one_hot(lower, num_atoms)
    upper_oh = jax.nn.one_hot(upper, num_atoms)
    transport = (
        lower_weight[..., None] * lower_oh
        + upper_weight[..., None] * upper_oh
    )
    projected = jnp.einsum("eba,bac->ebc", probabilities, transport)
    return projected / jnp.maximum(projected.sum(axis=-1, keepdims=True), 1e-8)


class ARQDFLFastSACAgent(DiscreteARQdflDistillAgent):
    """Phase-aware AR-QDFL agent with discrete FastSAC online updates."""

    critic: Any
    log_alpha: Any
    alpha_opt_state: Any
    reference_actor_params: Any
    offline_update_count: Any
    critic_update_count: Any
    actor_update_count: Any
    online_env_step: Any
    alpha_tx: Any = nonpytree_field()

    @property
    def support(self):
        return jnp.linspace(
            float(self.config["v_min"]),
            float(self.config["v_max"]),
            int(self.config["num_atoms"]),
            dtype=jnp.float32,
        )

    @staticmethod
    def normalized_log_prob(token_log_probs):
        """Joint AR log-probability normalized per register."""
        return token_log_probs.mean(axis=-1)

    @staticmethod
    def categorical_kl(logits, reference_logits):
        """KL(current || reference), averaged over batch and registers."""
        log_p = jax.nn.log_softmax(logits, axis=-1)
        log_q = jax.nn.log_softmax(reference_logits, axis=-1)
        p = jnp.exp(log_p)
        return jnp.sum(p * (log_p - log_q), axis=-1).mean()

    @staticmethod
    def polyak_update(network, module_name, tau):
        online = network.params[f"modules_{module_name}"]
        target = network.params[f"modules_target_{module_name}"]
        updated = jax.tree_util.tree_map(
            lambda source, old: tau * source + (1.0 - tau) * old,
            online,
            target,
        )
        network.params[f"modules_target_{module_name}"] = updated

    def with_offline_reference(self):
        """Snapshot the deployed 1M actor for the online trust-region anchor."""
        reference = jax.tree_util.tree_map(
            lambda x: jnp.array(x, copy=True),
            self.network.params["modules_target_actor"],
        )
        return self.replace(reference_actor_params=reference)

    def record_online_env_step(self, count: int = 1):
        return self.replace(
            online_env_step=self.online_env_step
            + jnp.asarray(count, dtype=jnp.int32)
        )

    def offline_update(self, batch):
        """Joint no-CRF teacher and AR-student offline update."""
        agent, info = super().update(batch)
        count = self.offline_update_count + jnp.asarray(1, dtype=jnp.int32)
        info = dict(info)
        info["phase_offline"] = jnp.asarray(1.0)
        info["offline_update_count"] = count.astype(jnp.float32)
        return agent.replace(offline_update_count=count), info

    def update(self, batch):
        """Compatibility entrypoint used by the generic offline trainer."""
        return self.offline_update(batch)

    def _reference_logits(self, observations, token_inputs):
        variables = {
            "params": {
                "modules_actor": self.reference_actor_params,
            }
        }
        return self.network.apply_fn(
            variables,
            observations,
            token_inputs,
            name="actor",
        )

    def _sample_ar_relaxed(
        self,
        observations,
        rng,
        *,
        actor_name: str,
        actor_params=None,
        straight_through: bool,
    ):
        """Sequential AR sample with hard prefixes and optional ST gradients."""
        batch_size = observations.shape[0]
        num_registers = int(self.config["num_registers"])
        temperature = jnp.asarray(
            self.config["st_temperature"], dtype=jnp.float32
        )
        tokens = jnp.zeros((batch_size, num_registers), dtype=jnp.int32)
        codes = []
        selected_log_probs = []
        entropies = []
        kls = []
        codebook = self.codebook

        for register_idx in range(num_registers):
            token_inputs = jnp.full(
                (batch_size, num_registers), BOS_ID, dtype=jnp.int32
            )
            if register_idx > 0:
                token_inputs = token_inputs.at[
                    :, 1 : register_idx + 1
                ].set(tokens[:, :register_idx])
            logits = self._actor_logits(
                observations,
                token_inputs,
                actor_name,
                params=actor_params,
            )[:, register_idx, :]
            rng, sample_rng = jax.random.split(rng)
            if straight_through:
                gumbel = jax.random.gumbel(sample_rng, logits.shape)
                soft = jax.nn.softmax((logits + gumbel) / temperature, axis=-1)
                token = jnp.argmax(soft, axis=-1).astype(jnp.int32)
                hard = jax.nn.one_hot(token, soft.shape[-1])
                weights = soft + jax.lax.stop_gradient(hard - soft)
                code = weights @ codebook
            else:
                token = jax.random.categorical(sample_rng, logits).astype(
                    jnp.int32
                )
                code = indices_to_codes(token, FSQ_LEVELS)
            tokens = tokens.at[:, register_idx].set(token)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            probs = jnp.exp(log_probs)
            selected = jnp.take_along_axis(
                log_probs, token[:, None], axis=-1
            )[:, 0]
            entropy = -jnp.sum(probs * log_probs, axis=-1)
            reference_logits = self._reference_logits(
                observations, token_inputs
            )[:, register_idx, :]
            reference_log_probs = jax.nn.log_softmax(
                reference_logits, axis=-1
            )
            kl = jnp.sum(
                probs * (log_probs - reference_log_probs), axis=-1
            )
            codes.append(code)
            selected_log_probs.append(selected)
            entropies.append(entropy)
            kls.append(kl)

        return {
            "tokens": tokens,
            "codes": jnp.stack(codes, axis=1),
            "token_log_probs": jnp.stack(selected_log_probs, axis=1),
            "entropy": jnp.stack(entropies, axis=1),
            "kl": jnp.stack(kls, axis=1),
            "rng": rng,
        }

    def _decode_relaxed_codes(self, codes):
        decoded = self._decode_codes(codes)
        return jnp.clip(decoded, -1.0, 1.0)

    def _batch_transition(self, batch):
        observations = batch["observations"][0]
        next_observations = batch["observations"][1]
        actions_btd = rearrange(
            batch["actions"][: int(self.config["h"])],
            "h b d -> b h d",
        )
        _, quantized_codes = self._encode_actions(actions_btd)
        actions = self._decode_codes(quantized_codes)
        rewards = batch["rewards"][0].astype(jnp.float32)
        masks = batch["masks"][0].astype(jnp.float32)
        return observations, actions, rewards, masks, next_observations

    def critic_loss(self, batch, grad_params, rng):
        observations, actions, rewards, masks, next_observations = (
            self._batch_transition(batch)
        )
        next_sample = self._sample_ar_relaxed(
            next_observations,
            rng,
            actor_name="target_actor",
            actor_params=None,
            straight_through=False,
        )
        next_actions = self._decode_relaxed_codes(next_sample["codes"])
        next_log_prob = self.normalized_log_prob(
            next_sample["token_log_probs"]
        )
        alpha = jnp.exp(self.log_alpha)
        discounts = (
            jnp.asarray(self.config["discount"], dtype=jnp.float32) * masks
        )
        entropy_adjusted_rewards = rewards - discounts * alpha * next_log_prob
        target_logits = self.critic.select("target_q")(
            next_observations, next_actions
        )
        target_probabilities = jax.nn.softmax(target_logits, axis=-1)
        target_distribution = jax.lax.stop_gradient(
            categorical_projection(
                target_probabilities,
                entropy_adjusted_rewards,
                discounts,
                self.support,
            )
        )
        logits = self.critic.select("q")(
            observations, actions, params=grad_params
        )
        log_probabilities = jax.nn.log_softmax(logits, axis=-1)
        critic_loss = -jnp.sum(
            target_distribution * log_probabilities, axis=-1
        ).mean()
        target_values = jnp.sum(
            target_distribution * self.support[None, None, :], axis=-1
        )
        lower_saturation = target_distribution[..., 0].mean()
        upper_saturation = target_distribution[..., -1].mean()
        return critic_loss, {
            "critic_loss": critic_loss,
            "target_q_mean": target_values.mean(),
            "target_q_std": target_values.std(),
            "support_lower_mass": lower_saturation,
            "support_upper_mass": upper_saturation,
            "alpha": alpha,
            "next_log_prob_per_register": next_log_prob.mean(),
        }

    @jax.jit
    def critic_update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.critic_loss(batch, grad_params, loss_rng)

        new_critic, info = self.critic.apply_loss_fn(loss_fn)
        self.polyak_update(
            new_critic, "q", jnp.asarray(self.config["critic_tau"])
        )
        count = self.critic_update_count + jnp.asarray(1, dtype=jnp.int32)
        info["critic_update_count"] = count.astype(jnp.float32)
        return self.replace(
            critic=new_critic,
            critic_update_count=count,
            rng=new_rng,
        ), info

    def actor_loss(self, batch, actor_params, rng):
        observations = batch["observations"][0]
        sample = self._sample_ar_relaxed(
            observations,
            rng,
            actor_name="actor",
            actor_params=actor_params,
            straight_through=True,
        )
        actions = self._decode_relaxed_codes(sample["codes"])
        q_logits = self.critic.select("q")(observations, actions)
        q_probabilities = jax.nn.softmax(q_logits, axis=-1)
        q_values = jnp.sum(
            q_probabilities * self.support[None, None, :], axis=-1
        ).mean(axis=0)
        entropy = sample["entropy"].mean(axis=-1)
        kl = sample["kl"].mean(axis=-1)
        alpha = jax.lax.stop_gradient(jnp.exp(self.log_alpha))
        kl_coef = jnp.asarray(self.config["offline_kl_coef"], jnp.float32)
        loss = (-q_values - alpha * entropy + kl_coef * kl).mean()
        return loss, {
            "actor_sac_loss": loss,
            "actor_q_mean": q_values.mean(),
            "actor_entropy_per_register": entropy.mean(),
            "actor_reference_kl": kl.mean(),
            "actor_action_abs_mean": jnp.abs(actions).mean(),
        }

    @jax.jit
    def actor_update(self, batch):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.actor_loss(batch, grad_params, loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn)
        self.target_update(
            new_network,
            "actor",
            d=1.0 - float(self.config["actor_ema"]),
        )
        entropy = jax.lax.stop_gradient(
            info["actor_entropy_per_register"]
        )
        target_entropy = jnp.asarray(
            self.config["target_entropy_per_register"], dtype=jnp.float32
        )

        def alpha_loss_fn(log_alpha):
            return jnp.exp(log_alpha) * (entropy - target_entropy)

        alpha_loss, alpha_grad = jax.value_and_grad(alpha_loss_fn)(
            self.log_alpha
        )
        alpha_updates, alpha_opt_state = self.alpha_tx.update(
            alpha_grad, self.alpha_opt_state, self.log_alpha
        )
        log_alpha = optax.apply_updates(self.log_alpha, alpha_updates)
        log_alpha = jnp.clip(
            log_alpha,
            jnp.log(float(self.config["alpha_min"])),
            jnp.log(float(self.config["alpha_max"])),
        )
        count = self.actor_update_count + jnp.asarray(1, dtype=jnp.int32)
        info["alpha_loss"] = alpha_loss
        info["alpha"] = jnp.exp(log_alpha)
        info["actor_update_count"] = count.astype(jnp.float32)
        return self.replace(
            network=new_network,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
            actor_update_count=count,
            rng=new_rng,
        ), info

    def online_update(self, batch, *, update_actor: bool):
        """One critic update and an optional delayed actor/temperature update."""
        agent, critic_info = self.critic_update(batch)
        actor_info = {}
        if update_actor:
            agent, actor_info = agent.actor_update(batch)
        info = dict(critic_info)
        info.update(actor_info)
        info["phase_online"] = jnp.asarray(1.0)
        info["online_env_step"] = agent.online_env_step.astype(jnp.float32)
        return agent, info

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = dict(config)
        base = DiscreteARQdflDistillAgent.create(
            seed, ex_observations, ex_actions, config
        )
        ex_observations = jnp.asarray(ex_observations)
        ex_actions = jnp.asarray(ex_actions)
        if ex_observations.ndim == 1:
            ex_observations = ex_observations[None, :]
        if ex_observations.ndim == 3:
            ex_observations = ex_observations[0]
        primitive_action_dim = int(ex_actions.shape[-1])
        action_dim = int(config["h"]) * primitive_action_dim
        ex_flat_actions = jnp.zeros(
            (ex_observations.shape[0], action_dim), dtype=jnp.float32
        )
        critic_def = CategoricalQNetwork(
            ensemble_ct=int(config["num_q_networks"]),
            hidden_dims=tuple(config["critic_hidden_dims"]),
            num_atoms=int(config["num_atoms"]),
            layer_norm=bool(config["critic_layer_norm"]),
        )
        critic_modules = ModuleDict(
            {
                "q": critic_def,
                "target_q": copy.deepcopy(critic_def),
            }
        )
        critic_args = {
            "q": (ex_observations, ex_flat_actions),
            "target_q": (ex_observations, ex_flat_actions),
        }
        rng, critic_rng = jax.random.split(base.rng)
        critic_params = critic_modules.init(critic_rng, **critic_args)["params"]
        critic_tx = optax.adamw(
            learning_rate=float(config["critic_lr"]),
            weight_decay=float(config["critic_weight_decay"]),
            b1=0.9,
            b2=0.95,
        )
        critic = TrainState.create(
            critic_modules, critic_params, tx=critic_tx
        )
        critic.params["modules_target_q"] = jax.tree_util.tree_map(
            jnp.array, critic.params["modules_q"]
        )
        reference_actor_params = jax.tree_util.tree_map(
            jnp.array, base.network.params["modules_target_actor"]
        )
        alpha_tx = optax.adam(float(config["alpha_lr"]))
        log_alpha = jnp.asarray(
            jnp.log(float(config["alpha_init"])), dtype=jnp.float32
        )
        return cls(
            rng=rng,
            teacher=base.teacher,
            network=base.network,
            tokenizer_def=base.tokenizer_def,
            tokenizer_params=base.tokenizer_params,
            codebook=build_codebook(FSQ_LEVELS),
            config=base.config,
            critic=critic,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_tx.init(log_alpha),
            reference_actor_params=reference_actor_params,
            offline_update_count=jnp.asarray(0, dtype=jnp.int32),
            critic_update_count=jnp.asarray(0, dtype=jnp.int32),
            actor_update_count=jnp.asarray(0, dtype=jnp.int32),
            online_env_step=jnp.asarray(0, dtype=jnp.int32),
            alpha_tx=alpha_tx,
        )


def get_config():
    """Humanoidmaze-large defaults for ConsensusFlowRL."""
    config = dict(get_ar_qdfl_config())
    config.update(
        {
            "agent_name": "ar_qdfl_fast_sac",
            # CF no-CRF teacher.
            "consensus_floor": 0.0,
            "conflict_power": 0.0,
            "residual_coef": 0.0,
            "rho": 0.0,
            # Distributional FastSAC.
            "num_q_networks": 2,
            "critic_hidden_dims": (768, 384, 192),
            "critic_layer_norm": True,
            "critic_lr": 3e-4,
            "critic_weight_decay": 1e-3,
            "critic_tau": 0.01,
            "num_atoms": 101,
            # humanoidmaze reward is -1 until goal, 0 on success; gamma=.995.
            "v_min": -200.0,
            "v_max": 0.0,
            "policy_frequency": 4,
            "utd": 4,
            # Per-register entropy regularization.
            "alpha_init": 0.001,
            "alpha_lr": 3e-4,
            "alpha_min": 1e-5,
            "alpha_max": 1.0,
            "target_entropy_per_register": 1.0,
            # Stable discrete pathwise gradient / offline trust region.
            "st_temperature": 1.0,
            "offline_kl_coef": 0.1,
            "actor_ema": 0.999,
            # Replay schedule consumed by the dedicated runner.
            "critic_warmup_updates": 100_000,
            "online_replay_fraction_max": 0.5,
            "online_replay_ramp_steps": 100_000,
        }
    )
    return mlc.ConfigDict(config)
