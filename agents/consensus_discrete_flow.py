"""ConsensusDiscreteFlow: CRAFT tokenization + ConsensusFlow guidance (OGBench).

Frozen OATTok codes + categorical flow actor + expectile critics on decoded
actions + latent FSQ-space consensus guidance with trust-weighted safety.
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
from einops import rearrange

from agents.oattok_jax import (
    CODEBOOK_SIZE,
    FSQ_DIM,
    FSQ_LEVELS,
    OATTok,
    build_codebook,
    indices_to_codes,
    load_tokenizer,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, Value, default_init


class DiscreteFlowActor(nn.Module):
    """MLP categorical flow actor: (s, flat FSQ codes, t) -> (K, V) logits."""

    hidden_dims: Any
    num_registers: int
    vocab_size: int = CODEBOOK_SIZE
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs_codes_times):
        h = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )(obs_codes_times)
        logits = nn.Dense(
            self.num_registers * self.vocab_size,
            kernel_init=default_init(1.0),
        )(h)
        return logits.reshape((-1, self.num_registers, self.vocab_size))


class LatentGuidance(nn.Module):
    """Guidance head over flattened FSQ latents: (s, codes, t) -> (K*q)."""

    hidden_dims: Any
    latent_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, obs_codes_times):
        return MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(obs_codes_times)


class ConsensusDiscreteFlowAgent(flax.struct.PyTreeNode):
    """ConsensusDiscreteFlow agent for OGBench offline RL."""

    rng: Any
    network: Any
    tokenizer_def: Any = nonpytree_field()
    tokenizer_params: Any = nonpytree_field()
    codebook: Any = nonpytree_field()
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def _project_unit_ball(self, w):
        w_norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
        return w * jnp.minimum(1.0, 1.0 / (w_norm + 1e-6))

    def _behavior_safe_direction(self, w, behavior_velocity):
        """Trust-weighted conflict + residual damping (ConsensusFlow / dflrql9)."""
        w = self._project_unit_ball(w)
        w_norm = jnp.linalg.norm(w, axis=-1, keepdims=True)
        trust = jax.lax.stop_gradient(w_norm)

        behavior_velocity = jax.lax.stop_gradient(behavior_velocity)
        velocity_norm = jnp.linalg.norm(behavior_velocity, axis=-1, keepdims=True)
        velocity_unit = behavior_velocity / jnp.maximum(velocity_norm, 1e-6)
        parallel = (w * velocity_unit).sum(axis=-1, keepdims=True)

        power = self.config["conflict_power"]
        kill_frac = 1.0 - jnp.power(jnp.clip(trust, 0.0, 1.0), power)
        conflicting_parallel = jnp.minimum(parallel, 0.0)
        conflict_free = w - kill_frac * conflicting_parallel * velocity_unit
        conflict_free = self._project_unit_ball(conflict_free)

        alignment_cos = parallel / (w_norm + 1e-6)
        residual_coef = self.config["residual_coef"]
        damp = 1.0 - residual_coef * jnp.maximum(alignment_cos, 0.0) * trust
        damp = jnp.clip(damp, 0.0, 1.0)
        safe_w = self._project_unit_ball(conflict_free * damp)
        return safe_w

    def _encode_actions(self, actions_btd):
        """(B, T, D) -> tokens (B, K), codes (B, K, q)."""
        quant, tokens = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            actions_btd,
            method=OATTok.encode,
            deterministic=True,
        )
        return tokens, quant

    def _decode_codes(self, codes):
        """(B, K, q) -> flattened continuous action (B, h*Da)."""
        recons = self.tokenizer_def.apply(
            {"params": self.tokenizer_params},
            codes,
            method=OATTok.decode,
            deterministic=True,
        )
        return rearrange(recons, "b t d -> b (t d)")

    def _actor_input(self, observations, codes, times):
        flat_codes = rearrange(codes, "b k q -> b (k q)")
        return jnp.concatenate([observations, flat_codes, times], axis=-1)

    def _soft_targets(self, tokens):
        """CRAFT geometry-aware soft CE targets over FSQ codes."""
        # tokens: (B, K)
        codebook = self.codebook  # (V, q)
        true_codes = indices_to_codes(tokens, FSQ_LEVELS)  # (B, K, q)
        # distances: (B, K, V)
        diff = true_codes[:, :, None, :] - codebook[None, None, :, :]
        dist2 = jnp.sum(diff * diff, axis=-1)
        neighbor = jax.nn.softmax(-dist2 / self.config["soft_target_temperature"], axis=-1)
        one_hot = jax.nn.one_hot(tokens, CODEBOOK_SIZE)
        eps = self.config["soft_target_eps"]
        return (1.0 - eps) * one_hot + eps * neighbor

    def _expected_latent_displacement(self, probs, codes):
        """b_i = E_p[r(v)] - r(x_i); probs (B,K,V), codes (B,K,q)."""
        expected = jnp.einsum("bkv,vq->bkq", probs, self.codebook)
        return rearrange(expected - codes, "b k q -> b (k q)")

    def _guided_logits(self, observations, tokens, codes, times, logits, stop_guidance=False):
        """Apply latent consensus guidance as FSQ-geometry logit bias."""
        actor_probs = jax.nn.softmax(logits, axis=-1)
        behavior = self._expected_latent_displacement(actor_probs, codes)
        w = self.network.select("guidance")(
            self._actor_input(observations, codes, times)
        )
        safe_w = self._behavior_safe_direction(w, behavior)
        if stop_guidance:
            safe_w = jax.lax.stop_gradient(safe_w)
        # Delta L_i(v) = lambda * t * <u_i, r(v) - r(x_i)>
        u = rearrange(safe_w, "b (k q) -> b k q", k=self.config["num_registers"], q=FSQ_DIM)
        code_delta = self.codebook[None, None, :, :] - codes[:, :, None, :]  # (B,K,V,q)
        energy = jnp.sum(u[:, :, None, :] * code_delta, axis=-1)  # (B,K,V)
        bias = self.config["guidance_coef"] * times[..., None] * energy
        return logits + bias, w, safe_w, behavior

    @staticmethod
    def _corrupt_tokens(rng, clean_tokens, t):
        """Mixture corruption q_t = t * clean + (1-t) * uniform."""
        # t: (B, 1)
        b, k = clean_tokens.shape
        rng, u_rng, n_rng = jax.random.split(rng, 3)
        uniform = jax.random.randint(u_rng, (b, k), 0, CODEBOOK_SIZE)
        keep = jax.random.uniform(n_rng, (b, k)) < t
        return jnp.where(keep, clean_tokens, uniform)

    @staticmethod
    def mixture_replace_probability(t, dt):
        """Bernoulli replace prob on the mixture path: clip(dt / (1-t), 0, 1)."""
        return jnp.clip(dt / jnp.maximum(1.0 - t, 0.0), 0.0, 1.0)

    @staticmethod
    def posterior_mixture_update(tokens, logits, rng, t, dt, force_replace=False):
        """One mixture-path step: sample clean posterior, replace w/ p=dt/(1-t).

        tokens: (..., K) int ids; logits: (..., K, V); t broadcasts with tokens.
        force_replace: if True, replace every token (exact final-step commit).
            Prefer a Python bool from a static loop so JIT can constant-fold;
            array bools also work via jnp.where.
        Returns (new_tokens, new_rng).
        """
        rng, cat_rng, bern_rng = jax.random.split(rng, 3)
        candidates = jax.random.categorical(cat_rng, logits).astype(tokens.dtype)
        # Exact commit path: avoid float32 p=clip(dt/(1-t)) < 1 when t=(N-1)/N.
        if force_replace is True:
            return candidates, rng
        p_replace = ConsensusDiscreteFlowAgent.mixture_replace_probability(t, dt)
        replace = jax.random.uniform(bern_rng, tokens.shape) < p_replace
        # Non-literal True (e.g. traced bool) still forces full replacement.
        replace = jnp.logical_or(replace, jnp.asarray(force_replace, dtype=bool))
        new_tokens = jnp.where(replace, candidates, tokens)
        return new_tokens, rng

    @staticmethod
    def chunk_token_ce_weights(rs_terminals, h):
        """Per-example CE weight from right-shifted terminals.

        OAT tokens jointly encode the full action chunk, so we use a
        conservative factor: weight 0 if any of the h actions is post-terminal
        padding. For h=1, rs_terminals[0] is always 0 → weight 1.
        rs_terminals: (H+, B); returns (B,).
        """
        action_valid = 1.0 - rs_terminals[:h]
        return action_valid.min(axis=0)

    @staticmethod
    def safe_masked_mean(values, mask):
        """Mean of values over mask; 0 when mask has no True entries."""
        mask_f = mask.astype(values.dtype)
        denom = jnp.maximum(mask_f.sum(), 1.0)
        return (values * mask_f).sum() / denom

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        rng = rng if rng is not None else self.rng
        batch_size = self.config["batch_size"]
        h = self.config["h"]

        # ---- Critic bootstrap at next state with uniform token decode ----
        rng, n_rng, c_rng, t_rng, k_rng, u_rng = jax.random.split(rng, 6)
        noise_tokens = jax.random.randint(
            n_rng, (batch_size, self.config["num_registers"]), 0, CODEBOOK_SIZE
        )
        noise_codes = indices_to_codes(noise_tokens, FSQ_LEVELS)
        noise_actions = jax.lax.stop_gradient(self._decode_codes(noise_codes))
        next_state = jnp.concatenate(
            [
                batch["observations"][-1],
                noise_actions,
                jnp.zeros((batch_size, 1)),
            ],
            axis=-1,
        )
        next_qs = self.network.select("target_value")(next_state)
        next_q = next_qs.mean(axis=0) - self.config["rho"] * next_qs.std(axis=0)

        # ---- Encode clean action chunks ----
        actions_hbd = batch["actions"][:h]  # (h, B, Da)
        actions_btd = rearrange(actions_hbd, "h b d -> b h d")
        clean_tokens, clean_codes = self._encode_actions(actions_btd)
        clean_tokens = jax.lax.stop_gradient(clean_tokens)
        clean_codes = jax.lax.stop_gradient(clean_codes)

        # Corrupted behavior tokens for critic states (mixture path).
        t_crit = jax.random.uniform(c_rng, (batch_size, 1))
        t_crit = jnp.clip(t_crit, 1e-3, 1.0 - 1e-3)
        x_tokens = self._corrupt_tokens(u_rng, clean_tokens, t_crit)
        x_codes = indices_to_codes(x_tokens, FSQ_LEVELS)
        y_t = jax.lax.stop_gradient(self._decode_codes(x_codes))
        state = jnp.concatenate([batch["observations"][0], y_t, t_crit], axis=-1)
        q = self.network.select("value")(state, params=grad_params)

        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(batch["terminals"][:1]), batch["terminals"][:-1]],
            axis=0,
        )
        n_rews = (
            batch["rewards"]
            * self.config["discount_mul"][..., None]
            * (1 - rs_terminals)
        ).sum(0)
        target_q = (
            n_rews
            + (self.config["discount"] ** h) * next_q * batch["masks"][-2]
        )
        terminal_count = rs_terminals.sum(0)
        valids = (terminal_count <= 1).astype(terminal_count.dtype)
        critic_loss = (
            self.expectile_loss(target_q - q, target_q - q, self.config["expectile"])
            * valids
        ).mean()

        # ---- Categorical BC (geometry-aware CE) ----
        t = jax.random.uniform(t_rng, (batch_size, 1))
        t = jnp.clip(t, 1e-3, 1.0 - 1e-3)
        rng, corr_rng = jax.random.split(rng)
        x_t_tokens = self._corrupt_tokens(corr_rng, clean_tokens, t)
        x_t_codes = indices_to_codes(x_t_tokens, FSQ_LEVELS)
        logits = self.network.select("actor")(
            self._actor_input(batch["observations"][0], x_t_codes, t),
            params=grad_params,
        )
        soft_tgt = self._soft_targets(clean_tokens)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce_tok = -(soft_tgt * log_probs).sum(axis=-1)  # (B, K)
        ce_weight = self.chunk_token_ce_weights(rs_terminals, h)  # (B,)
        ce_loss = (ce_tok.mean(axis=-1) * ce_weight).sum() / jnp.maximum(
            ce_weight.sum(), 1e-6
        )

        # Guidance diagnostics (not in the optimized objective).
        guided_logits, _, safe_w, _ = self._guided_logits(
            batch["observations"][0],
            x_t_tokens,
            x_t_codes,
            t,
            logits,
            stop_guidance=True,
        )

        # ---- Latent consensus distillation ----
        member = jax.random.randint(k_rng, (batch_size,), 0, self.config["ensemble_ct"])
        member_onehot = jax.nn.one_hot(member, self.config["ensemble_ct"])

        def q_member_sum(flat_codes):
            codes = rearrange(
                flat_codes,
                "b (k q) -> b k q",
                k=self.config["num_registers"],
                q=FSQ_DIM,
            )
            y = self._decode_codes(codes)
            q_in = jnp.concatenate([batch["observations"][0], y, t], axis=-1)
            qs = self.network.select("target_value")(q_in)
            return (qs * member_onehot.T).sum()

        flat_codes = rearrange(x_t_codes, "b k q -> b (k q)")
        q_grad = jax.lax.stop_gradient(jax.grad(q_member_sum)(flat_codes))
        q_grad_norm = jnp.linalg.norm(q_grad, axis=-1, keepdims=True)
        valid_count = valids.sum() + 1e-6
        grad_scale = jax.lax.stop_gradient(
            (q_grad_norm[..., 0] * valids).sum() / valid_count
        )
        relative_floor = self.config["consensus_floor"] * grad_scale
        consensus_target = q_grad / (q_grad_norm + relative_floor + 1e-6)

        w_train = self.network.select("guidance")(
            self._actor_input(batch["observations"][0], x_t_codes, t),
            params=grad_params,
        )
        distill_loss = (
            jnp.square(w_train - consensus_target).sum(axis=-1) * valids
        ).mean()

        # Policy improvement is guidance-only; categorical actor is CE-only.
        total_loss = (
            self.config["alpha"] * ce_loss
            + critic_loss
            + self.config["distill_coef"] * distill_loss
        )

        pred_tokens = jnp.argmax(logits, axis=-1)
        guided_pred = jnp.argmax(guided_logits, axis=-1)
        correct = pred_tokens == clean_tokens
        corrupted = x_t_tokens != clean_tokens
        retained = jnp.logical_not(corrupted)
        actions_flat = rearrange(actions_btd, "b h d -> b (h d)")
        clean_recons = self._decode_codes(clean_codes)
        decode_rmse = jnp.sqrt(jnp.mean(jnp.square(clean_recons - actions_flat)))

        return total_loss, {
            "total_loss": total_loss,
            "bc_loss": ce_loss,
            "ce_loss": ce_loss,
            "token_acc": correct.mean(),
            "token_acc_corrupted": self.safe_masked_mean(correct, corrupted),
            "ce_corrupted": self.safe_masked_mean(ce_tok, corrupted),
            "token_acc_retained": self.safe_masked_mean(correct, retained),
            "corruption_frac": corrupted.mean(),
            "guidance_flip_rate": (pred_tokens != guided_pred).mean(),
            "decode_rmse": decode_rmse,
            "ce_weight_mean": ce_weight.mean(),
            "critic_loss": critic_loss,
            "distill_loss": distill_loss,
            "q": q.mean(),
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
            "w_norm": jnp.linalg.norm(w_train, axis=-1).mean(),
            "safe_w_norm": jnp.linalg.norm(safe_w, axis=-1).mean(),
            "consensus_target_norm": jnp.linalg.norm(
                consensus_target, axis=-1
            ).mean(),
        }

    def target_update(self, network, module_name, d):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
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
        self.target_update(new_network, "value", d=self.config["tau"])
        self.target_update(new_network, "actor", d=1 - self.config["ema"])
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_flow_actions(self, observations, noise_tokens, seed=None, temperature=0.0):
        """Refine uniform tokens with mixture-path guided posteriors; decode."""
        tokens = noise_tokens
        actor_name = "actor" if temperature > 0 else "target_actor"
        n = self.config["flow_steps"]
        dt = 1.0 / n
        rng = seed if seed is not None else self.rng
        # Python for-loop is unrolled under jit; force_replace on the last
        # step is a compile-time constant (exact p=1, immune to float32 t).
        for i in range(n):
            rng, step_rng = jax.random.split(rng)
            t = jnp.full((*observations.shape[:-1], 1), i / n)
            codes = indices_to_codes(tokens, FSQ_LEVELS)
            logits = self.network.select(actor_name)(
                self._actor_input(observations, codes, t)
            )
            guided_logits, _, _, _ = self._guided_logits(
                observations, tokens, codes, t, logits, stop_guidance=False
            )
            # temperature>0 sharpens/flattens posterior sampling; temperature=0
            # still uses seeded categorical draws (not hard all-token argmax).
            if temperature > 0:
                sample_logits = guided_logits / temperature
            else:
                sample_logits = guided_logits
            tokens, _ = self.posterior_mixture_update(
                tokens,
                sample_logits,
                step_rng,
                t,
                dt,
                force_replace=(i == n - 1),
            )
        codes = indices_to_codes(tokens, FSQ_LEVELS)
        actions = self._decode_codes(codes)
        return jnp.clip(actions, -1, 1)

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        action_rng, n_rng = jax.random.split(seed)
        obs = jnp.atleast_2d(obs)[-1:]
        noise_tokens = jax.random.randint(
            n_rng, (1, self.config["num_registers"]), 0, CODEBOOK_SIZE
        )
        actions = self.compute_flow_actions(
            obs, noise_tokens=noise_tokens, seed=action_rng, temperature=temperature
        )[0]
        actions = rearrange(actions, "(h d) -> h d", h=self.config["h"])
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_actions = jnp.asarray(ex_actions)
        ex_observations = jnp.asarray(ex_observations)
        # sample_traj returns (H, B, D); take a single-step action vector.
        if ex_actions.ndim == 3:
            prim_dim = int(ex_actions.shape[-1])
            ex_obs = ex_observations[0]
            ex_act_step = ex_actions[0]
        elif ex_actions.ndim == 2:
            prim_dim = int(ex_actions.shape[-1])
            ex_obs = ex_observations if ex_observations.ndim == 2 else ex_observations[None, :]
            ex_act_step = ex_actions
        else:
            prim_dim = int(ex_actions.shape[0])
            ex_obs = ex_observations[None, :] if ex_observations.ndim == 1 else ex_observations
            ex_act_step = ex_actions[None, :]

        h = int(config["h"])
        action_dim = prim_dim * h
        num_registers = int(config["num_registers"])
        latent_dim = num_registers * FSQ_DIM

        tokenizer_path = config.get("tokenizer_path", "")
        if not tokenizer_path:
            raise ValueError(
                "ConsensusDiscreteFlow requires agent.tokenizer_path to a "
                "frozen OATTok checkpoint (.pkl)."
            )
        tok_params, tok_meta = load_tokenizer(tokenizer_path)
        if int(tok_meta["sample_dim"]) != prim_dim:
            raise ValueError(
                f"Tokenizer sample_dim={tok_meta['sample_dim']} != action dim {prim_dim}"
            )
        if int(tok_meta["sample_horizon"]) != h:
            raise ValueError(
                f"Tokenizer sample_horizon={tok_meta['sample_horizon']} != h={h}"
            )
        # Prefer tokenizer checkpoint's register count (mazes may use K=16).
        num_registers = int(tok_meta.get("num_registers", num_registers))
        config = dict(config)
        config["num_registers"] = num_registers
        latent_dim = num_registers * FSQ_DIM
        tokenizer_def = OATTok(
            sample_dim=prim_dim,
            sample_horizon=h,
            num_registers=num_registers,
            emb_dim=int(tok_meta.get("emb_dim", 256)),
            encoder_depth=int(tok_meta.get("encoder_depth", 2)),
            decoder_depth=int(tok_meta.get("decoder_depth", 4)),
        )

        ex_flat_actions = jnp.concatenate([ex_act_step] * h, axis=-1)
        ex_times = jnp.zeros((ex_obs.shape[0], 1), dtype=jnp.float32)
        ex_codes = jnp.zeros(
            (ex_obs.shape[0], num_registers, FSQ_DIM), dtype=jnp.float32
        )
        ex_actor_in = jnp.concatenate(
            [ex_obs, rearrange(ex_codes, "b k q -> b (k q)"), ex_times],
            axis=-1,
        )
        ex_value_in = jnp.concatenate([ex_obs, ex_flat_actions, ex_times], axis=-1)

        value_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        actor_def = DiscreteFlowActor(
            hidden_dims=config["actor_hidden_dims"],
            num_registers=num_registers,
            vocab_size=CODEBOOK_SIZE,
            layer_norm=config["layer_norm"],
        )
        guidance_def = LatentGuidance(
            hidden_dims=config["guidance_hidden_dims"],
            latent_dim=latent_dim,
            layer_norm=config["layer_norm"],
        )

        network_info = dict(
            value=(value_def, (ex_value_in,)),
            target_value=(copy.deepcopy(value_def), (ex_value_in,)),
            actor=(actor_def, (ex_actor_in,)),
            target_actor=(copy.deepcopy(actor_def), (ex_actor_in,)),
            guidance=(guidance_def, (ex_actor_in,)),
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

        config = dict(config)
        config["action_dim"] = action_dim
        config["prim_action_dim"] = prim_dim
        config["discount_mul"] = jnp.array(
            config["discount"] ** jnp.array(list(range(h)) + [jnp.inf])
        )
        codebook = build_codebook(FSQ_LEVELS)

        return cls(
            rng=rng,
            network=network,
            tokenizer_def=tokenizer_def,
            tokenizer_params=tok_params,
            codebook=codebook,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = mlc.ConfigDict(
        dict(
            agent_name="consensus_discrete_flow",
            h=3,
            alpha=1.0,
            expectile=0.5,
            ensemble_ct=10,
            rho=0.0,
            lr=3e-4,
            discount=0.99,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            guidance_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            tau=0.005,
            ema=0.999,
            flow_steps=10,
            guidance_coef=0.5,
            distill_coef=1.0,
            consensus_floor=0.01,
            conflict_power=2.0,
            residual_coef=0.25,
            num_registers=12,
            soft_target_eps=0.1,
            soft_target_temperature=0.25,
            # Categorical actor is CE-only; RL improvement is guidance tilt.
            tokenizer_path="",
        )
    )
    return config
