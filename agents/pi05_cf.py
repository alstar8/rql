"""π₀.₅-style CF fine-tune scaffold: frozen base expert + LoRA BC + endpoint G.

Observations are treated as frozen VLM features ``h``.  The pretrained flow
expert (``actor``) stays frozen; plasticity is a zero-init LoRA velocity
adapter trained with BC only.  Policy improvement is an endpoint residual
``G`` under a Lagrange trust region, with a CQL-regularized critic for sparse
success labels.

See ``my_exps/pi05_cf_finetune_pipeline.md``.
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

from agents.dflrql11 import EndpointRefiner, LogLagrange
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, MLP, Value


class LoRAVelocityAdapter(nn.Module):
    """Low-rank residual on flow velocity; zero-init so identity at start."""

    action_dim: int
    rank: int
    hidden_dims: Any
    layer_norm: bool = True

    @nn.compact
    def __call__(self, inputs):
        features = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )(inputs)
        down = nn.Dense(
            self.rank,
            kernel_init=nn.initializers.variance_scaling(
                1.0, "fan_avg", "uniform"
            ),
            bias_init=nn.initializers.zeros,
            name="lora_down",
        )(features)
        return nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
            name="lora_up",
        )(down)


class Pi05CFAgent(flax.struct.PyTreeNode):
    """Frozen base flow + LoRA BC + RL endpoint refiner + CQL critic."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _chunk_actions(self, batch):
        return rearrange(
            batch["actions"][: self.config["h"]],
            "h b d -> b (h d)",
        )

    def _success_weights(self, batch):
        """Per-example BC weights; boosts successful transitions when present."""

        batch_size = self.config["batch_size"]
        boost = jnp.asarray(self.config["success_bc_boost"], dtype=jnp.float32)
        if "successes" not in batch:
            return jnp.ones((batch_size,), dtype=jnp.float32)
        successes = jnp.asarray(batch["successes"], dtype=jnp.float32)
        if successes.ndim > 1:
            successes = successes.reshape(successes.shape[0], -1)[0]
        successes = successes.reshape(batch_size)
        return 1.0 + (boost - 1.0) * successes

    def _critic_input(self, observations, flat_actions):
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

    def _velocity(self, observations, actions, times, actor_name, lora_name, params=None):
        actor_input = jnp.concatenate([observations, actions, times], axis=-1)
        base = self.network.select(actor_name)(actor_input, params=params).mode()
        if bool(self.config["use_lora"]):
            delta = self.network.select(lora_name)(actor_input, params=params)
            return base + delta
        return base

    def _flow_actions(self, observations, noises, actor_name, lora_name):
        actions = noises
        for flow_idx in range(self.config["flow_steps"]):
            times = jnp.full(
                (*observations.shape[:-1], 1),
                flow_idx / self.config["flow_steps"],
                dtype=actions.dtype,
            )
            velocity = self._velocity(
                observations,
                actions,
                times,
                actor_name,
                lora_name,
            )
            actions = actions + velocity / self.config["flow_steps"]
        return jnp.clip(actions, -1.0, 1.0)

    def _refiner_input(self, observations, base_actions):
        endpoint = jnp.ones((*base_actions.shape[:-1], 1), dtype=base_actions.dtype)
        return jnp.concatenate(
            [observations, base_actions, endpoint],
            axis=-1,
        )

    def _refined_actions(
        self,
        observations,
        base_actions,
        *,
        refiner_name,
        params=None,
    ):
        raw_delta = self.network.select(refiner_name)(
            self._refiner_input(observations, base_actions),
            params=params,
        )
        actions = jnp.clip(base_actions + raw_delta, -1.0, 1.0)
        return actions, raw_delta

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

    def _cql_penalty(self, observations, data_actions, grad_params, rng):
        """Random-action CQL regularizer for sparse-success offline Q."""

        cql_coef = jnp.asarray(self.config["cql_coef"], dtype=jnp.float32)
        n_actions = int(self.config["cql_n_actions"])
        action_dim = self.config["action_dim"]
        batch_size = observations.shape[0]
        rng, sample_rng = jax.random.split(rng)
        random_actions = jax.random.uniform(
            sample_rng,
            (n_actions, batch_size, action_dim),
            minval=-1.0,
            maxval=1.0,
        )
        obs_rep = jnp.repeat(observations[None, ...], n_actions, axis=0)
        flat_obs = rearrange(obs_rep, "n b d -> (n b) d")
        flat_rand = rearrange(random_actions, "n b d -> (n b) d")
        rand_qs = self._critic_scores(
            "critic",
            flat_obs,
            flat_rand,
            params=grad_params,
        )
        # Ensemble mean then logsumexp over random actions.
        rand_q = rearrange(rand_qs.mean(axis=0), "(n b) -> n b", n=n_actions)
        logsumexp_q = jax.nn.logsumexp(rand_q, axis=0)
        data_q = self._critic_scores(
            "critic",
            observations,
            data_actions,
            params=grad_params,
        ).mean(axis=0)
        cql_loss = (logsumexp_q - data_q).mean()
        return cql_coef * cql_loss, {
            "cql_loss": cql_loss,
            "cql_coef": cql_coef,
            "cql_logsumexp_q": logsumexp_q.mean(),
            "cql_data_q": data_q.mean(),
        }

    def _critic_loss(self, batch, grad_params, rng):
        batch_size = self.config["batch_size"]
        action_dim = self.config["action_dim"]
        rng, noise_rng, smooth_rng, cql_rng = jax.random.split(rng, 4)

        _, discounted_rewards, valids, bootstrap_mask = self._valids_and_returns(
            batch
        )
        next_observations = batch["observations"][-1]
        next_noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        next_base = jax.lax.stop_gradient(
            self._flow_actions(
                next_observations,
                next_noises,
                "target_actor",
                "target_actor_lora",
            )
        )
        next_actions, _ = self._refined_actions(
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
        td_loss = (critic_per * valids).sum() / valid_count
        cql_loss, cql_info = self._cql_penalty(
            batch["observations"][0],
            flat_actions,
            grad_params,
            cql_rng,
        )
        critic_loss = td_loss + cql_loss
        return critic_loss, {
            "critic_loss": critic_loss,
            "td_loss": td_loss,
            "q_mean": qs.mean(),
            "q_max": qs.max(),
            "q_min": qs.min(),
            "target_q_mean": (target_q * valids).sum() / valid_count,
            "valid_fraction": valids.mean(),
            **cql_info,
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
        # Frozen base expert: never take BC grads into ``actor``.
        base_velocity = jax.lax.stop_gradient(
            self.network.select("actor")(actor_input).mode()
        )
        if bool(self.config["use_lora"]):
            lora_delta = self.network.select("actor_lora")(
                actor_input,
                params=grad_params,
            )
            velocity = base_velocity + lora_delta
        elif bool(self.config["freeze_base_actor"]):
            lora_delta = jnp.zeros_like(base_velocity)
            velocity = base_velocity
        else:
            lora_delta = jnp.zeros_like(base_velocity)
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
        success_w = self._success_weights(batch)[:, None]
        per = jnp.square(velocity - target_velocity) * action_mask * success_w
        bc_loss = per.mean()
        return bc_loss, {
            "bc_loss": bc_loss,
            "velocity_norm": jnp.linalg.norm(velocity, axis=-1).mean(),
            "lora_delta_norm": jnp.linalg.norm(lora_delta, axis=-1).mean(),
            "success_weight_mean": success_w.mean(),
        }

    def _refiner_loss(self, batch, grad_params, rng):
        batch_size = self.config["batch_size"]
        action_dim = self.config["action_dim"]
        noise = jax.random.normal(rng, (batch_size, action_dim))
        observations = batch["observations"][0]
        base_actions = jax.lax.stop_gradient(
            self._flow_actions(
                observations,
                noise,
                "target_actor",
                "target_actor_lora",
            )
        )
        refined_actions, raw_delta = self._refined_actions(
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

        residual_mse = jnp.square(raw_delta).mean()
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
            "residual_mse": residual_mse,
            "residual_rms": jnp.sqrt(residual_mse + 1e-12),
            "trust_region_loss": trust_region_loss,
            "constraint_error": constraint_error,
            "alpha_loss": alpha_loss,
            "alpha": alpha,
            "q_scale": q_scale,
        }

    def _phase_refiner_weight(self):
        # Phase 1: critic + LoRA BC only. Phases 2/3: enable residual RL.
        phase = int(self.config["train_phase"])
        return jnp.asarray(0.0 if phase <= 1 else 1.0, dtype=jnp.float32)

    def _phase_bc_weight(self):
        if not bool(self.config["use_lora"]) and bool(self.config["freeze_base_actor"]):
            return jnp.asarray(0.0, dtype=jnp.float32)
        return jnp.asarray(self.config["bc_coef"], dtype=jnp.float32)

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
        delayed = (self.network.step % actor_delay == 0).astype(jnp.float32)
        refiner_weight = self._phase_refiner_weight() * delayed
        bc_weight = self._phase_bc_weight()
        total_loss = (
            critic_loss
            + bc_weight * bc_loss
            + refiner_weight * refiner_loss
        )
        return total_loss, {
            "total_loss": total_loss,
            **critic_info,
            **bc_info,
            **refiner_info,
            "bc_weight": bc_weight,
            "refiner_weight": refiner_weight,
            "train_phase": jnp.asarray(
                self.config["train_phase"], dtype=jnp.float32
            ),
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
        network.params[target_key] = Pi05CFAgent._select_tree(
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
        delayed = self.network.step % self.config["actor_delay"] == 0
        phase_on = jnp.asarray(
            int(self.config["train_phase"]) > 1,
            dtype=jnp.bool_,
        )
        refiner_update = jnp.logical_and(delayed, phase_on)

        for module_name in ("refiner", "log_alpha"):
            key = f"modules_{module_name}"
            new_network.params[key] = self._select_tree(
                refiner_update,
                new_network.params[key],
                self.network.params[key],
            )

        if bool(self.config["freeze_base_actor"]):
            for module_name in ("actor", "target_actor"):
                key = f"modules_{module_name}"
                new_network.params[key] = self.network.params[key]

        self._target_update(new_network, "critic", self.config["tau"])
        self._target_update(
            new_network,
            "refiner",
            self.config["tau"],
            predicate=refiner_update,
        )
        if bool(self.config["use_lora"]):
            self._target_update(
                new_network,
                "actor_lora",
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
        lora_name = "actor_lora" if temperature > 0 else "target_actor_lora"
        return self._flow_actions(observations, noise, actor_name, lora_name)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        _, noise_rng = jax.random.split(seed)
        observations = jnp.atleast_2d(obs)[-1:]
        noise = jax.random.normal(
            noise_rng,
            (1, self.config["action_dim"]),
        )
        actor_name = "actor" if temperature > 0 else "target_actor"
        lora_name = "actor_lora" if temperature > 0 else "target_actor_lora"
        refiner_name = "refiner" if temperature > 0 else "target_refiner"
        base_actions = self._flow_actions(
            observations, noise, actor_name, lora_name
        )
        if bool(self.config["disable_rl_policy"]):
            return rearrange(
                base_actions[0],
                "(h d) -> h d",
                h=self.config["h"],
            )
        actions, _ = self._refined_actions(
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
        lora_def = LoRAVelocityAdapter(
            action_dim=action_dim,
            rank=config["lora_rank"],
            hidden_dims=config["lora_hidden_dims"],
            layer_norm=config["layer_norm"],
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
            "actor_lora": (lora_def, (ex_actor_input,)),
            "target_actor_lora": (
                copy.deepcopy(lora_def),
                (ex_actor_input,),
            ),
            "critic": (critic_def, (ex_critic_input,)),
            "target_critic": (copy.deepcopy(critic_def), (ex_critic_input,)),
            "refiner": (refiner_def, (ex_critic_input,)),
            "target_refiner": (
                copy.deepcopy(refiner_def),
                (ex_critic_input,),
            ),
            "log_alpha": (log_alpha_def, ()),
        }
        networks = {name: definition for name, (definition, _) in network_info.items()}
        network_args = {name: args for name, (_, args) in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        params = network_def.init(init_rng, **network_args)["params"]
        params["modules_target_actor"] = params["modules_actor"]
        params["modules_target_actor_lora"] = params["modules_actor_lora"]
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
            "agent_name": "pi05_cf",
            "h": 3,
            "lr": 3e-4,
            "batch_size": 256,
            "discount": 0.99,
            "tau": 0.005,
            "ema": 0.999,
            "flow_steps": 10,
            "actor_hidden_dims": (512, 512, 512, 512),
            "actor_layer_norm": False,
            "lora_hidden_dims": (256, 256),
            "lora_rank": 16,
            "use_lora": True,
            "freeze_base_actor": True,
            "value_hidden_dims": (512, 512, 512, 512),
            "refiner_hidden_dims": (512, 512, 512, 512),
            "layer_norm": True,
            "ensemble_ct": 10,
            "q_agg": "min",
            "rho": 0.0,
            "bc_coef": 10.0,
            "success_bc_boost": 2.0,
            "cql_coef": 1.0,
            "cql_n_actions": 4,
            "train_phase": 2,
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
            "disable_rl_policy": False,
        }
    )
