"""DiscreteARIQL (v6+ / v8): Causal AR over OATTok register tokens + IQL/AWR.

Genuinely discrete successor to DiscreteCoordMaskIQL (v5). Frozen OATTok
encode yields ordered token IDs z_0..z_{K-1} (vocab=1000). A causal
Transformer models the exact joint p(z|s)=prod_i p(z_i|s,z_<i), eliminating
factorized FSQ-coordinate incoherence. Critics support three advantage sources:
standard IQL Q/V (default); Monte Carlo return-to-go
(``advantage_source=mc_return``) with expectile V(s)→G and shift-stable AWR;
or trajectory-success weighted BC (``advantage_source=trajectory_success``)
keeping ordinary IQL V→Q while upweighting all transitions from successful
episodes. Optional two-pass scheduled sampling (loss-only; network params
unchanged) and empirical register CE weights.

v8 adds an optional direct action-level critic gradient: straight-through
hard-forward / soft-backward FSQ codes from actor logits, frozen OATTok
decode (no stop-grad on that branch), and frozen ``target_q`` scoring
without ``params=grad_params``. Default ``q_actor_coef=0`` preserves the
legacy CE/IQL objective. Deploy remains autoregressive sampling only —
no critics.
"""

from __future__ import annotations

import copy
from functools import partial
from typing import Any, Sequence, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections as mlc
import optax
from einops import rearrange

from agents.oattok_jax import (
    CODEBOOK_SIZE,
    FSQ_LEVELS,
    OATTok,
    TransformerBlock,
    build_codebook,
    indices_to_codes,
    load_tokenizer,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Value, default_init

BOS_ID = CODEBOOK_SIZE  # 1000; token embeddings size = vocab + 1
DEFAULT_PREFIX_LENGTHS = (1, 2, 4, 8)

# Empirical decoder-swap sensitivity on humanoidmaze-large (K=16), mean=1 after
# normalize. Source: params_200000 exposure-bias diagnostics (AWR5 mean seeds).
DEFAULT_EMPIRICAL_REGISTER_CE_WEIGHTS_K16 = (
    0.030,
    0.052,
    0.076,
    0.051,
    0.118,
    0.079,
    0.083,
    0.061,
    0.067,
    0.032,
    0.060,
    0.061,
    0.094,
    0.066,
    0.036,
    0.034,
)


class CausalARActor(nn.Module):
    """Causal Transformer: obs context + teacher-forced tokens -> (B, K, V).

    Sequence layout (length K+1):
      [obs_ctx, emb(BOS), emb(z_0), ..., emb(z_{K-2})]
    Causal tril mask: position i+1 predicts z_i and may attend obs + inputs
    at positions <= i+1 (never future target tokens).

    ``dropout`` must remain 0.0 (RNG not plumbed through training).
    """

    emb_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    vocab_size: int = CODEBOOK_SIZE
    num_registers: int = 16
    dropout: float = 0.0

    @staticmethod
    def causal_attn_mask(num_registers: int) -> jnp.ndarray:
        """Boolean mask (1, 1, K+1, K+1); True = allow attend."""
        length = num_registers + 1
        return jnp.tril(jnp.ones((length, length), dtype=bool))[None, None, :, :]

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        token_inputs: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        # observations: (B, obs_dim); token_inputs: (B, K) int in [0, vocab]
        k = self.num_registers
        obs_tok = nn.Dense(
            self.emb_dim, kernel_init=default_init(), name="obs_proj"
        )(observations)[:, None, :]
        tok_emb = nn.Embed(
            self.vocab_size + 1,
            self.emb_dim,
            embedding_init=nn.initializers.normal(0.02),
            name="token_embed",
        )(token_inputs)
        pos = self.param(
            "pos_embed",
            nn.initializers.normal(0.02),
            (1, k, self.emb_dim),
        )
        tok_emb = tok_emb + pos
        x = jnp.concatenate([obs_tok, tok_emb], axis=1)  # (B, K+1, D)
        attn_mask = self.causal_attn_mask(k)
        for i in range(self.depth):
            x = TransformerBlock(
                self.emb_dim,
                self.num_heads,
                self.dropout,
                name=f"block_{i}",
            )(x, mask=attn_mask, deterministic=deterministic)
        x = nn.LayerNorm(name="final_ln")(x)
        token_out = x[:, 1:, :]  # (B, K, D)
        logits = nn.Dense(
            self.vocab_size,
            kernel_init=default_init(1.0),
            name="logits",
        )(token_out)
        return logits


class DiscreteARIQLAgent(flax.struct.PyTreeNode):
    """Discrete autoregressive IQL agent for OGBench offline RL."""

    rng: Any
    network: Any
    tokenizer_def: Any = nonpytree_field()
    tokenizer_params: Any = nonpytree_field()
    codebook: Any = nonpytree_field()
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(diff, expectile):
        """IQL expectile regression on residual ``target - prediction``."""
        weight = jnp.where(diff >= 0, expectile, (1.0 - expectile))
        return weight * (diff**2)

    @staticmethod
    def chunk_ce_weights(rs_terminals, h):
        """Per-example CE weight: 0 if any of the h actions is post-terminal."""
        action_valid = 1.0 - rs_terminals[:h]
        return action_valid.min(axis=0)

    # Alias matching v5 name for shared tests / callers.
    chunk_coord_ce_weights = chunk_ce_weights

    @staticmethod
    def safe_masked_mean(values, mask):
        """Mean of values over mask; 0 when mask has no True entries."""
        mask_f = mask.astype(values.dtype)
        denom = jnp.maximum(mask_f.sum(), 1.0)
        return (values * mask_f).sum() / denom

    @staticmethod
    def make_teacher_inputs(tokens, bos_id: int = BOS_ID):
        """Shifted teacher inputs: BOS at pos 0, then z_0..z_{K-2}.

        tokens: (B, K) clean targets. Returns (B, K) int inputs with no
        target leakage (position i never sees z_i or later).
        """
        bos = jnp.full((tokens.shape[0], 1), bos_id, dtype=jnp.int32)
        return jnp.concatenate([bos, tokens[:, :-1].astype(jnp.int32)], axis=-1)

    @staticmethod
    def ss_schedule(step, start_steps, ramp_steps, max_coef, p_min, p_max):
        """Scheduled-sampling (λ, p) from network step.

        Before ``start_steps``: λ=0 and p=0.
        Over the next ``ramp_steps``: λ ramps 0→max_coef and p ramps p_min→p_max.
        """
        step_f = jnp.asarray(step, dtype=jnp.float32)
        start = jnp.asarray(start_steps, dtype=jnp.float32)
        ramp = jnp.maximum(jnp.asarray(ramp_steps, dtype=jnp.float32), 1.0)
        frac = jnp.clip(jnp.maximum(step_f - start, 0.0) / ramp, 0.0, 1.0)
        coef = frac * jnp.asarray(max_coef, dtype=jnp.float32)
        p_lo = jnp.asarray(p_min, dtype=jnp.float32)
        p_hi = jnp.asarray(p_max, dtype=jnp.float32)
        prefix_p = jnp.where(
            frac > 0.0,
            p_lo + (p_hi - p_lo) * frac,
            jnp.zeros((), dtype=jnp.float32),
        )
        return coef, prefix_p

    @staticmethod
    def resolve_register_ce_weights(
        num_registers, use_register_weights, register_ce_weights
    ):
        """Per-register CE weights with mean 1, or uniform fallback.

        Wrong-length packs (e.g. K≠16 empirical defaults) fall back to uniform
        ones — never silently broadcast a mismatched weight vector.
        """
        k = int(num_registers)
        if not bool(use_register_weights):
            return jnp.ones((k,), dtype=jnp.float32)
        w = jnp.asarray(register_ce_weights, dtype=jnp.float32).reshape(-1)
        if int(w.shape[0]) != k:
            return jnp.ones((k,), dtype=jnp.float32)
        mean = jnp.maximum(w.mean(), 1e-8)
        return w / mean

    @staticmethod
    def make_scheduled_sampling_inputs(teacher_inputs, pred_tokens, replace_mask):
        """Bernoulli-corrupt shifted GT previous tokens with stop-grad predictions.

        ``teacher_inputs``: (B, K) with BOS at position 0 and z_{i-1} at i>0.
        ``pred_tokens``: (B, K) stop-grad model predictions of z_0..z_{K-1}.
        ``replace_mask``: (B, K) bool; True replaces input. Position 0 (BOS) is
        forced False — never replaced. Position i>0 replaces z_{i-1} with
        ``pred_tokens[:, i-1]`` (never future tokens).
        """
        replace_mask = replace_mask.astype(bool).at[:, 0].set(False)
        pred_as_inputs = jnp.concatenate(
            [teacher_inputs[:, :1], pred_tokens[:, :-1].astype(teacher_inputs.dtype)],
            axis=-1,
        )
        return jnp.where(replace_mask, pred_as_inputs, teacher_inputs)

    @staticmethod
    def make_self_conditioned_inputs(pred_tokens, bos_id: int = BOS_ID):
        """Full predicted-prefix inputs: BOS + shifted stop-grad argmax tokens.

        Same shift layout as teacher forcing, but every replaceable previous
        token is the model's own prediction (not Bernoulli SS). Used by the
        v8 Q-actor branch in ``q_actor_prefix_mode='self_conditioned'``.
        ``pred_tokens`` should already be stop-grad'd by the caller.
        """
        return DiscreteARIQLAgent.make_teacher_inputs(pred_tokens, bos_id=bos_id)

    @staticmethod
    def straight_through_fsq_codes(logits, codebook, temperature=1.0):
        """Hard-forward / soft-backward FSQ codes from actor logits.

        ``probs = softmax(logits / temperature)``, ``soft = probs @ codebook``,
        ``hard = indices_to_codes(argmax(logits))``,
        ``codes = soft + stop_gradient(hard - soft)``.

        Forward values equal exact hard codes; backward uses soft expectation
        so gradients reach logits. Returns ``(codes, probs)`` with shapes
        ``(B, K, q)`` and ``(B, K, V)``.
        """
        temp = jnp.maximum(jnp.asarray(temperature, dtype=logits.dtype), 1e-6)
        probs = jax.nn.softmax(logits / temp, axis=-1)
        soft = jnp.einsum("bkv,vq->bkq", probs, codebook)
        hard_ids = jnp.argmax(logits, axis=-1)
        hard = indices_to_codes(hard_ids, FSQ_LEVELS)
        codes = soft + jax.lax.stop_gradient(hard - soft)
        return codes, probs

    @staticmethod
    def q_actor_weight_from_step(step, warmup_steps, ramp_steps, coef):
        """Effective Q-actor coefficient: 0 during warmup, then coef * ramp."""
        ramp_frac = DiscreteARIQLAgent.awr_ramp_fraction(
            step, warmup_steps, ramp_steps
        )
        return jnp.asarray(coef, dtype=jnp.float32) * ramp_frac

    @staticmethod
    def differentiable_action_clip(actions, lo=-1.0, hi=1.0):
        """Clip actions to ``[lo, hi]`` (subgradient 0 outside; matches deploy)."""
        return jnp.clip(actions, lo, hi)

    @staticmethod
    def action_clip_saturation_frac(actions, lo=-1.0, hi=1.0):
        """Fraction of action dims at or beyond clip boundaries."""
        return ((actions <= lo) | (actions >= hi)).astype(jnp.float32).mean()

    @staticmethod
    def weighted_token_ce_mean(ce_tok, register_weights):
        """Per-example mean CE with register weights (mean-1 → weighted mean)."""
        # ce_tok: (B, K); register_weights: (K,)
        return (ce_tok * register_weights[None, :]).mean(axis=-1)

    @staticmethod
    def token_ce(logits, targets):
        """Per-token CE; logits (B, K, V), targets (B, K) -> (B, K)."""
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        oh = jax.nn.one_hot(targets, logits.shape[-1], dtype=log_probs.dtype)
        return -(oh * log_probs).sum(axis=-1)

    @staticmethod
    def prefix_exact_rates(
        pred, targets, lengths: Sequence[int] = DEFAULT_PREFIX_LENGTHS
    ):
        """Exact-match rates for prefixes of given lengths and full sequence.

        Returns dict keyed by prefix length (ints) plus ``seq_exact``.
        """
        match = pred == targets
        k = pred.shape[-1]
        out = {"seq_exact": match.all(axis=-1).astype(jnp.float32).mean()}
        for length in lengths:
            if length <= k:
                out[length] = (
                    match[:, :length].all(axis=-1).astype(jnp.float32).mean()
                )
        if k not in out:
            out[k] = out["seq_exact"]
        return out

    @staticmethod
    def awr_ramp_fraction(step, bc_warmup_steps, awr_ramp_steps):
        """0 during BC warmup, then linear ramp to 1 over awr_ramp_steps."""
        step_f = jnp.asarray(step, dtype=jnp.float32)
        warmup = jnp.asarray(bc_warmup_steps, dtype=jnp.float32)
        ramp = jnp.maximum(jnp.asarray(awr_ramp_steps, dtype=jnp.float32), 1.0)
        return jnp.clip(jnp.maximum(step_f - warmup, 0.0) / ramp, 0.0, 1.0)

    @staticmethod
    def awr_example_weights(advantage, temperature, max_weight, ramp_frac):
        """Blend identity BC weights into clipped exp(A/T) via ramp_frac."""
        awr = jnp.exp(advantage / jnp.maximum(temperature, 1e-6))
        awr = jnp.minimum(awr, max_weight)
        ones = jnp.ones_like(awr)
        return (1.0 - ramp_frac) * ones + ramp_frac * awr

    @staticmethod
    def stable_awr_example_weights(advantage, temperature, max_weight, ramp_frac):
        """Shift-invariant AWR weights (mean-1, clipped) blended with BC.

        Numerically stable when ``A`` has a large negative offset (e.g. MC
        returns ``G≈-180`` while ``V≈0`` at init): work in log-space, subtract
        the batch max before ``exp``, normalize to mean 1, clip at
        ``max_weight``, then renormalize to mean 1. Underflow of naive
        ``exp(A/T)`` is avoided so the actor can learn during V warmup.
        """
        log_w = advantage / jnp.maximum(temperature, 1e-6)
        log_w = log_w - jnp.max(log_w)
        awr = jnp.exp(log_w)
        awr = awr / jnp.maximum(awr.mean(), 1e-8)
        awr = jnp.minimum(awr, max_weight)
        awr = awr / jnp.maximum(awr.mean(), 1e-8)
        ones = jnp.ones_like(awr)
        return (1.0 - ramp_frac) * ones + ramp_frac * awr

    @staticmethod
    def trajectory_success_example_weights(success_flags, success_weight, ramp_frac):
        """BC weights that upweight successful-episode transitions (no exp/T).

        ``w = (1-ramp)*1 + ramp*(1 + (success_weight-1)*flag)`` with
        ``flag∈{0,1}``. At ``success_weight=20`` and ~7.7% success rate this
        yields ~62.5% of total training mass on success trajectories when
        ``ramp=1``. Chunk/valid masks are applied by the caller afterward.
        """
        flag = success_flags.astype(jnp.float32)
        w_succ = 1.0 + (jnp.asarray(success_weight, dtype=jnp.float32) - 1.0) * flag
        ones = jnp.ones_like(w_succ)
        return (1.0 - ramp_frac) * ones + ramp_frac * w_succ

    @staticmethod
    def awr_ess(weights):
        """Effective sample size of positive weights; 0 if empty."""
        w = jnp.maximum(weights, 0.0)
        denom = jnp.maximum(jnp.square(w).sum(), 1e-12)
        return jnp.square(w.sum()) / denom

    @staticmethod
    def pearson_corr(x, y):
        """Pearson correlation of flattened arrays; 0 if either has ~0 variance."""
        x = jnp.reshape(x, (-1,)).astype(jnp.float32)
        y = jnp.reshape(y, (-1,)).astype(jnp.float32)
        x = x - x.mean()
        y = y - y.mean()
        denom = jnp.sqrt(jnp.sum(jnp.square(x)) * jnp.sum(jnp.square(y)))
        return jnp.where(denom > 1e-12, jnp.sum(x * y) / denom, jnp.zeros(()))

    @staticmethod
    def validate_create_config(config):
        """Raise ValueError for invalid / unsupported actor hyperparameters."""
        emb_dim = int(config["actor_emb_dim"])
        num_heads = int(config["actor_num_heads"])
        if num_heads <= 0:
            raise ValueError(f"actor_num_heads must be positive, got {num_heads}")
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"actor_emb_dim ({emb_dim}) must be divisible by "
                f"actor_num_heads ({num_heads})"
            )
        dropout = float(config.get("actor_dropout", 0.0))
        if dropout != 0.0:
            raise ValueError(
                f"actor_dropout={dropout} is not supported: dropout RNG is not "
                "plumbed through training (DiscreteARIQL uses dropout=0 only). "
                "Set actor_dropout=0.0."
            )
        ss_mode = str(config.get("ss_pred_mode", "argmax"))
        if ss_mode not in ("argmax", "categorical"):
            raise ValueError(
                f"ss_pred_mode must be 'argmax' or 'categorical', got {ss_mode!r}"
            )
        max_coef = float(config.get("ss_loss_coef", 0.5))
        if max_coef < 0.0 or max_coef > 1.0:
            raise ValueError(f"ss_loss_coef must be in [0, 1], got {max_coef}")
        p_min = float(config.get("ss_prefix_prob_min", 0.1))
        p_max = float(config.get("ss_prefix_prob_max", 0.35))
        if not (0.0 <= p_min <= p_max <= 1.0):
            raise ValueError(
                f"Need 0 <= ss_prefix_prob_min ({p_min}) <= "
                f"ss_prefix_prob_max ({p_max}) <= 1"
            )
        weights = tuple(config.get("register_ce_weights", ()))
        if any(float(x) < 0.0 for x in weights):
            raise ValueError("register_ce_weights must be non-negative")

        advantage_source = str(config.get("advantage_source", "iql"))
        # Exhaustive: only these three modes are supported.
        if advantage_source == "iql":
            pass
        elif advantage_source == "mc_return":
            if not bool(config.get("use_mc_returns", False)):
                raise ValueError(
                    "advantage_source='mc_return' requires use_mc_returns=True "
                    "(attach dataset mc_returns in process_train_dataset)."
                )
            mc_expectile = float(config.get("mc_expectile", 0.7))
            if not (0.0 < mc_expectile < 1.0):
                raise ValueError(
                    f"mc_expectile must be in (0, 1), got {mc_expectile}"
                )
            mc_warmup = int(config.get("mc_actor_warmup_steps", 0))
            if mc_warmup < 0:
                raise ValueError(
                    f"mc_actor_warmup_steps must be >= 0, got {mc_warmup}"
                )
        elif advantage_source == "trajectory_success":
            if not bool(config.get("use_trajectory_success", False)):
                raise ValueError(
                    "advantage_source='trajectory_success' requires "
                    "use_trajectory_success=True (attach dataset "
                    "trajectory_success in process_train_dataset)."
                )
            success_weight = float(config.get("success_weight", 20.0))
            if success_weight <= 0.0:
                raise ValueError(
                    f"success_weight must be > 0, got {success_weight}"
                )
            succ_warmup = int(config.get("success_actor_warmup_steps", 0))
            if succ_warmup < 0:
                raise ValueError(
                    f"success_actor_warmup_steps must be >= 0, got {succ_warmup}"
                )
        else:
            raise ValueError(
                f"advantage_source must be 'iql', 'mc_return', or "
                f"'trajectory_success', got {advantage_source!r}"
            )

        q_actor_coef = float(config.get("q_actor_coef", 0.0))
        if q_actor_coef < 0.0:
            raise ValueError(f"q_actor_coef must be >= 0, got {q_actor_coef}")
        q_warmup = int(config.get("q_actor_warmup_steps", 50_000))
        if q_warmup < 0:
            raise ValueError(f"q_actor_warmup_steps must be >= 0, got {q_warmup}")
        q_ramp = int(config.get("q_actor_ramp_steps", 50_000))
        if q_ramp < 0:
            raise ValueError(f"q_actor_ramp_steps must be >= 0, got {q_ramp}")
        prefix_mode = str(config.get("q_actor_prefix_mode", "teacher_forced"))
        if prefix_mode not in ("teacher_forced", "self_conditioned"):
            raise ValueError(
                "q_actor_prefix_mode must be 'teacher_forced' or "
                f"'self_conditioned', got {prefix_mode!r}"
            )
        st_temp = float(config.get("st_temperature", 1.0))
        if st_temp <= 0.0:
            raise ValueError(f"st_temperature must be > 0, got {st_temp}")

    @staticmethod
    def finalize_register_weight_config(config, num_registers: int):
        """Enable empirical weights only when length matches runtime K.

        Wrong-length packs (other domains / K≠16) explicitly disable weighting
        so resolve falls back to uniform — never silently mis-apply K=16 weights.
        """
        if not bool(config.get("use_register_weights", True)):
            return config
        weights = tuple(config.get("register_ce_weights", ()))
        if len(weights) == 0:
            config["use_register_weights"] = False
            return config
        if len(weights) != int(num_registers):
            config["use_register_weights"] = False
        return config

    @staticmethod
    def h_step_td_target(rewards, terminals, masks, discount, discount_mul, h, next_v):
        """RQL h-step TD target with right-shifted terminals and V bootstrap.

        rewards/terminals/masks: (H+, B); next_v: (B,); returns (target, valids, rs).
        """
        rs_terminals = jnp.concatenate(
            [jnp.zeros_like(terminals[:1]), terminals[:-1]],
            axis=0,
        )
        n_rews = (rewards * discount_mul[..., None] * (1.0 - rs_terminals)).sum(0)
        target = n_rews + (discount**h) * next_v * masks[-2]
        terminal_count = rs_terminals.sum(0)
        valids = (terminal_count <= 1).astype(target.dtype)
        return target, valids, rs_terminals

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

    def _actor_logits(self, observations, token_inputs, actor_name, params=None):
        return self.network.select(actor_name)(
            observations, token_inputs, params=params
        )

    def _resolve_actor_and_sampling_temp(self, temperature: float) -> Tuple[str, float]:
        """Map external eval temperature to (actor_name, sampling_temperature).

        temperature==0 → target_actor + config eval_sampling_temperature
        (default 1.0 categorical; 0.0 for argmax).
        temperature>0 → online actor + that provided sampling temperature.
        """
        if temperature > 0.0:
            return "actor", float(temperature)
        return "target_actor", float(self.config["eval_sampling_temperature"])

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        batch_size = self.config["batch_size"]
        h = self.config["h"]
        k = self.config["num_registers"]
        vocab = CODEBOOK_SIZE
        if rng is None:
            rng = self.rng

        # ---- Encode clean action chunks -> discrete tokens ----
        actions_hbd = batch["actions"][:h]
        actions_btd = rearrange(actions_hbd, "h b d -> b h d")
        clean_tokens, clean_codes = self._encode_actions(actions_btd)
        clean_tokens = jax.lax.stop_gradient(clean_tokens.astype(jnp.int32))
        clean_codes = jax.lax.stop_gradient(clean_codes)
        clean_actions = jax.lax.stop_gradient(self._decode_codes(clean_codes))

        # ---- IQL Q on clean decoded actions (no t) ----
        q_in = jnp.concatenate([batch["observations"][0], clean_actions], axis=-1)
        q_values = self.network.select("q")(q_in, params=grad_params)

        next_v = self.network.select("target_v")(batch["observations"][-1])
        if next_v.ndim > 1:
            next_v = next_v.mean(axis=0)
        target_q, valids, rs_terminals = self.h_step_td_target(
            batch["rewards"],
            batch["terminals"],
            batch["masks"],
            self.config["discount"],
            self.config["discount_mul"],
            h,
            next_v,
        )
        td_target = jax.lax.stop_gradient(target_q)
        q_err = q_values - td_target
        if q_err.ndim == 1:
            q_loss = (jnp.square(q_err) * valids).mean()
            q_mean = q_values.mean()
        else:
            q_loss = (jnp.square(q_err) * valids[None, :]).mean()
            q_mean = q_values.mean()

        # ---- V expectile + actor advantage / example weights ----
        advantage_source = str(self.config["advantage_source"])
        use_mc = advantage_source == "mc_return"
        use_traj = advantage_source == "trajectory_success"

        target_q_values = self.network.select("target_q")(q_in)
        if target_q_values.ndim > 1:
            q_behavior = target_q_values.mean(axis=0)
            if self.config["rho"] != 0.0:
                q_behavior = q_behavior - self.config["rho"] * target_q_values.std(
                    axis=0
                )
        else:
            q_behavior = target_q_values
        q_behavior = jax.lax.stop_gradient(q_behavior)

        v_values = self.network.select("v")(
            batch["observations"][0], params=grad_params
        )
        if use_mc:
            # Monte Carlo return-to-go at the chunk start state.
            if "mc_returns" not in batch:
                raise KeyError(
                    "advantage_source='mc_return' requires batch['mc_returns']; "
                    "set agent.use_mc_returns=True so process_train_dataset "
                    "attaches Monte Carlo return-to-go."
                )
            mc_g = batch["mc_returns"][0]
            v_target = jax.lax.stop_gradient(mc_g)
            v_expectile = self.config["mc_expectile"]
        else:
            # IQL and trajectory_success: ordinary expectile V → Q_behavior.
            v_target = q_behavior
            v_expectile = self.config["expectile"]

        if v_values.ndim > 1:
            v_pred = v_values.mean(axis=0)
            v_diff = v_target - v_values
            v_loss = (
                self.expectile_loss(v_diff, v_expectile).mean(axis=0) * valids
            ).mean()
        else:
            v_pred = v_values
            v_diff = v_target - v_values
            v_loss = (self.expectile_loss(v_diff, v_expectile) * valids).mean()

        # Actor example weights (labels/weights stop-grad; CE grads only).
        if use_traj:
            if "trajectory_success" not in batch:
                raise KeyError(
                    "advantage_source='trajectory_success' requires "
                    "batch['trajectory_success']; set "
                    "agent.use_trajectory_success=True so "
                    "process_train_dataset attaches episode-success flags."
                )
            # Diagnostic Q-V advantage (V still trains to Q); weights use flags.
            advantage = jax.lax.stop_gradient(q_behavior - v_pred)
            succ_flag = jax.lax.stop_gradient(batch["trajectory_success"][0])
            actor_warmup = self.config["success_actor_warmup_steps"]
            ramp_frac = self.awr_ramp_fraction(
                self.network.step,
                actor_warmup,
                self.config["awr_ramp_steps"],
            )
            awr_w = jax.lax.stop_gradient(
                self.trajectory_success_example_weights(
                    succ_flag,
                    self.config["success_weight"],
                    ramp_frac,
                )
            )
        elif use_mc:
            advantage = jax.lax.stop_gradient(v_target - v_pred)
            actor_warmup = self.config["mc_actor_warmup_steps"]
            ramp_frac = self.awr_ramp_fraction(
                self.network.step,
                actor_warmup,
                self.config["awr_ramp_steps"],
            )
            # Stable weights: avoid exp underflow when G≪0 and V≈0 at init.
            awr_w = self.stable_awr_example_weights(
                advantage,
                self.config["awr_temperature"],
                self.config["max_weight"],
                ramp_frac,
            )
        else:
            advantage = jax.lax.stop_gradient(q_behavior - v_pred)
            actor_warmup = self.config["bc_warmup_steps"]
            ramp_frac = self.awr_ramp_fraction(
                self.network.step,
                actor_warmup,
                self.config["awr_ramp_steps"],
            )
            awr_w = self.awr_example_weights(
                advantage,
                self.config["awr_temperature"],
                self.config["max_weight"],
                ramp_frac,
            )
        chunk_w = self.chunk_ce_weights(rs_terminals, h)
        example_w = awr_w * chunk_w * valids
        weight_sum = jnp.maximum(example_w.sum(), 1e-6)

        # ---- Register CE weights (mean-1; uniform if disabled / wrong length) ----
        reg_w = self.resolve_register_ce_weights(
            k,
            self.config["use_register_weights"],
            self.config["register_ce_weights"],
        )

        # ---- Pass A: teacher-forced AR CE ----
        token_inputs = self.make_teacher_inputs(clean_tokens, BOS_ID)
        logits = self._actor_logits(
            batch["observations"][0], token_inputs, "actor", params=grad_params
        )
        ce_tok = self.token_ce(logits, clean_tokens)  # (B, K)
        ce_ex_tf = self.weighted_token_ce_mean(ce_tok, reg_w)
        tf_actor_loss = (ce_ex_tf * example_w).sum() / weight_sum

        # ---- SS schedule (checkpoint-compatible; no new params) ----
        ss_coef, ss_prefix_prob = self.ss_schedule(
            self.network.step,
            self.config["ss_start_steps"],
            self.config["ss_ramp_steps"],
            self.config["ss_loss_coef"],
            self.config["ss_prefix_prob_min"],
            self.config["ss_prefix_prob_max"],
        )
        ss_pred_mode = str(self.config["ss_pred_mode"])
        rng, ss_rng = jax.random.split(rng)

        def _ss_branch(ss_rng_):
            # Stop-grad predictions from pass-A logits (no feedback through zhat).
            logits_sg = jax.lax.stop_gradient(logits)
            if ss_pred_mode == "categorical":
                ss_rng_, cat_rng = jax.random.split(ss_rng_)
                pred_tokens = jax.random.categorical(cat_rng, logits_sg).astype(
                    jnp.int32
                )
            else:
                pred_tokens = jnp.argmax(logits_sg, axis=-1).astype(jnp.int32)
            ss_rng_, corr_rng = jax.random.split(ss_rng_)
            replace = jax.random.bernoulli(
                corr_rng, p=ss_prefix_prob, shape=token_inputs.shape
            )
            replace = replace.at[:, 0].set(False)
            mixed_inputs = self.make_scheduled_sampling_inputs(
                token_inputs, pred_tokens, replace
            )
            logits_ss = self._actor_logits(
                batch["observations"][0],
                mixed_inputs,
                "actor",
                params=grad_params,
            )
            ce_tok_ss = self.token_ce(logits_ss, clean_tokens)
            ce_ex_ss = self.weighted_token_ce_mean(ce_tok_ss, reg_w)
            ss_actor_loss = (ce_ex_ss * example_w).sum() / weight_sum
            pred_ss = jnp.argmax(logits_ss, axis=-1)
            ss_token_acc = (pred_ss == clean_tokens).astype(jnp.float32).mean()
            # Fraction over replaceable positions (exclude BOS).
            replaced_frac = replace[:, 1:].astype(jnp.float32).mean()
            return (
                ce_tok_ss,
                ss_actor_loss,
                ss_token_acc,
                replaced_frac,
            )

        def _skip_ss(_):
            zero = jnp.zeros((), dtype=ce_tok.dtype)
            return (
                jnp.zeros_like(ce_tok),
                zero,
                zero,
                zero,
            )

        # λ=0 skips pass B (avoids 0*NaN contamination under JIT).
        ce_tok_ss, ss_actor_loss, ss_token_acc, replaced_frac = jax.lax.cond(
            ss_coef > 0.0,
            _ss_branch,
            _skip_ss,
            operand=ss_rng,
        )

        # Prefer combining already-weighted example losses (λ=0 ⇒ TF only).
        actor_loss = (1.0 - ss_coef) * tf_actor_loss + ss_coef * ss_actor_loss

        # ---- v8 Q-actor: ST codes → frozen decode → frozen target_q ----
        # Weights: chunk/valid only (NOT AWR example weights — avoid mixing
        # objectives). Gate with lax.cond when effective coef==0.
        q_actor_weight = self.q_actor_weight_from_step(
            self.network.step,
            self.config["q_actor_warmup_steps"],
            self.config["q_actor_ramp_steps"],
            self.config["q_actor_coef"],
        )
        q_chunk_w = chunk_w * valids
        q_weight_sum = jnp.maximum(q_chunk_w.sum(), 1e-6)
        actions_flat = rearrange(actions_btd, "b h d -> b (h d)")
        q_prefix_mode = str(self.config["q_actor_prefix_mode"])
        st_temp = self.config["st_temperature"]

        def _q_actor_branch(_):
            if q_prefix_mode == "self_conditioned":
                # Stop-grad argmax from pass-A; full predicted prefix (not SS).
                pred_sg = jax.lax.stop_gradient(
                    jnp.argmax(logits, axis=-1).astype(jnp.int32)
                )
                q_inputs = self.make_self_conditioned_inputs(pred_sg, BOS_ID)
                q_logits = self._actor_logits(
                    batch["observations"][0],
                    q_inputs,
                    "actor",
                    params=grad_params,
                )
            else:
                # Teacher-forced: reuse pass-A logits.
                q_logits = logits

            codes, st_probs = self.straight_through_fsq_codes(
                q_logits, self.codebook, temperature=st_temp
            )
            # Differentiable decode — no stop_gradient on this actor branch.
            decoded = self._decode_codes(codes)
            # Match deploy clip in compute_ar_actions.
            policy_actions = self.differentiable_action_clip(decoded, -1.0, 1.0)
            sat_frac = self.action_clip_saturation_frac(decoded, -1.0, 1.0)

            q_in_pol = jnp.concatenate(
                [batch["observations"][0], policy_actions], axis=-1
            )
            # Frozen target_q params (no params=); grads flow through inputs.
            q_pol = self.network.select("target_q")(q_in_pol)
            if q_pol.ndim > 1:
                q_pe = q_pol.mean(axis=0)
                if self.config["rho"] != 0.0:
                    q_pe = q_pe - self.config["rho"] * q_pol.std(axis=0)
            else:
                q_pe = q_pol

            actor_q_loss = -(q_pe * q_chunk_w).sum() / q_weight_sum

            pred_q = jnp.argmax(q_logits, axis=-1)
            correct_q = (pred_q == clean_tokens).astype(jnp.float32)
            q_token_acc = correct_q.mean()
            q_prefix = self.prefix_exact_rates(pred_q, clean_tokens)
            policy_rmse = jnp.sqrt(
                jnp.mean(jnp.square(policy_actions - actions_flat))
            )
            st_log_p = jnp.log(jnp.maximum(st_probs, 1e-8))
            st_entropy = -(st_probs * st_log_p).sum(axis=-1).mean()
            return (
                actor_q_loss,
                q_pe,
                q_token_acc,
                q_prefix["seq_exact"],
                q_prefix.get(1, q_prefix["seq_exact"]),
                q_prefix.get(2, q_prefix["seq_exact"]),
                q_prefix.get(4, q_prefix["seq_exact"]),
                q_prefix.get(8, q_prefix["seq_exact"]),
                policy_rmse,
                sat_frac,
                st_entropy,
            )

        def _skip_q_actor(_):
            zero = jnp.zeros((), dtype=actor_loss.dtype)
            q_pe_zero = jnp.zeros((batch_size,), dtype=actor_loss.dtype)
            return (
                zero,
                q_pe_zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
            )

        (
            actor_q_loss,
            q_pe,
            q_token_acc,
            q_seq_exact,
            q_prefix_exact_1,
            q_prefix_exact_2,
            q_prefix_exact_4,
            q_prefix_exact_8,
            q_policy_rmse,
            q_sat_frac,
            st_entropy,
        ) = jax.lax.cond(
            q_actor_weight > 0.0,
            _q_actor_branch,
            _skip_q_actor,
            operand=None,
        )

        total_loss = (
            self.config["q_coef"] * q_loss
            + self.config["v_coef"] * v_loss
            + self.config["alpha"] * actor_loss
            + q_actor_weight * actor_q_loss
        )

        # ---- Diagnostics ----
        probs = jax.nn.softmax(logits, axis=-1)
        pred = jnp.argmax(logits, axis=-1)
        correct = (pred == clean_tokens).astype(jnp.float32)
        token_acc = correct.mean()  # unweighted TF accuracy
        prefix = self.prefix_exact_rates(pred, clean_tokens)

        decode_rmse = jnp.sqrt(jnp.mean(jnp.square(clean_actions - actions_flat)))

        clip_frac = (
            (awr_w >= self.config["max_weight"] - 1e-6).astype(jnp.float32).mean()
        )
        ess = self.awr_ess(example_w)

        # Per-register entropy (nats).
        log_probs = jnp.log(jnp.maximum(probs, 1e-8))
        entropy = -(probs * log_probs).sum(axis=-1)  # (B, K)

        # Weighted CE diagnostics (register weights applied).
        weighted_ce_tf = (ce_tok * reg_w[None, :]).mean()
        weighted_ce_ss = (ce_tok_ss * reg_w[None, :]).mean()

        info = {
            "total_loss": total_loss,
            "q_loss": q_loss,
            "v_loss": v_loss,
            "actor_loss": actor_loss,
            "tf_actor_loss": tf_actor_loss,
            "ss_actor_loss": ss_actor_loss,
            "q_actor_loss": actor_q_loss,
            "q_actor_weight": q_actor_weight,
            "q_policy_mean": q_pe.mean(),
            "q_policy_min": q_pe.min(),
            "q_policy_max": q_pe.max(),
            "q_policy_rmse": q_policy_rmse,
            "q_action_sat_frac": q_sat_frac,
            "st_entropy_mean": st_entropy,
            "q_token_acc": q_token_acc,
            "q_seq_exact": q_seq_exact,
            "q_prefix_exact_1": q_prefix_exact_1,
            "q_prefix_exact_2": q_prefix_exact_2,
            "q_prefix_exact_4": q_prefix_exact_4,
            "q_prefix_exact_8": q_prefix_exact_8,
            "token_acc": token_acc,
            "ss_token_acc": ss_token_acc,
            "seq_exact": prefix["seq_exact"],
            "ce_mean": ce_tok.mean(),
            "weighted_ce_tf": weighted_ce_tf,
            "weighted_ce_ss": weighted_ce_ss,
            "ss_coef": ss_coef,
            "ss_prefix_prob": ss_prefix_prob,
            "ss_replaced_frac": replaced_frac,
            "register_weight_min": reg_w.min(),
            "register_weight_max": reg_w.max(),
            "q": q_mean,
            "q_mean": q_mean,
            "q_max": jnp.max(q_values),
            "q_min": jnp.min(q_values),
            "v_mean": v_pred.mean(),
            "v_max": v_pred.max(),
            "v_min": v_pred.min(),
            "adv_mean": advantage.mean(),
            "adv_std": advantage.std(),
            "adv_max": advantage.max(),
            "adv_min": advantage.min(),
            "awr_weight_mean": awr_w.mean(),
            "awr_weight_max": awr_w.max(),
            "awr_clip_frac": clip_frac,
            "awr_ess": ess,
            "actor_weight_ramp": ramp_frac,
            "example_weight_mean": example_w.mean(),
            "chunk_weight_mean": chunk_w.mean(),
            "decode_rmse": decode_rmse,
            "valids_mean": valids.mean(),
            "entropy_mean": entropy.mean(),
            # 0 = iql, 1 = mc_return, 2 = trajectory_success.
            "advantage_source": jnp.asarray(
                2.0 if use_traj else (1.0 if use_mc else 0.0),
                dtype=jnp.float32,
            ),
        }
        if use_mc:
            info["mc_return_mean"] = v_target.mean()
            info["mc_return_std"] = v_target.std()
            info["mc_return_min"] = v_target.min()
            info["mc_return_max"] = v_target.max()
            info["mc_v_error"] = (v_target - v_pred).mean()
            # Raw (unstabilized) advantage for diagnosing G-V scale.
            info["mc_adv_raw_mean"] = advantage.mean()
            info["mc_adv_raw_min"] = advantage.min()
            info["mc_adv_raw_max"] = advantage.max()
        if use_traj:
            succ_f = succ_flag.astype(jnp.float32)
            info["traj_success_frac"] = succ_f.mean()
            info["traj_success_weight_mean"] = self.safe_masked_mean(awr_w, succ_f > 0.5)
            info["traj_fail_weight_mean"] = self.safe_masked_mean(awr_w, succ_f < 0.5)
            w_sum = jnp.maximum(awr_w.sum(), 1e-6)
            info["traj_weighted_success_share"] = (awr_w * succ_f).sum() / w_sum
            info["success_weight"] = jnp.asarray(
                float(self.config["success_weight"]), dtype=jnp.float32
            )
        for i in range(k):
            info[f"reg{i}_acc"] = correct[:, i].mean()
            info[f"reg{i}_ce"] = ce_tok[:, i].mean()
            info[f"reg{i}_ce_w"] = (ce_tok[:, i] * reg_w[i]).mean()
            info[f"reg{i}_entropy"] = entropy[:, i].mean()
        for length in DEFAULT_PREFIX_LENGTHS:
            if length in prefix:
                info[f"prefix_exact_{length}"] = prefix[length]
        info[f"prefix_exact_{k}"] = prefix.get(k, prefix["seq_exact"])
        # Keep vocab size visible in diagnostics.
        info["vocab_size"] = jnp.asarray(float(vocab))
        info["batch_size"] = jnp.asarray(float(batch_size))
        return total_loss, info

    def target_update(self, network, module_name, d):
        """Polyak/EMA from *post-update* online params."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * d + tp * (1 - d),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        def _leaf_norm(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            if not leaves:
                return jnp.zeros(())
            return jnp.sqrt(sum(jnp.square(x).sum() for x in leaves))

        for name in ("q", "v", "actor"):
            delta = jax.tree_util.tree_map(
                lambda a, b: a - b,
                new_network.params[f"modules_{name}"],
                self.network.params[f"modules_{name}"],
            )
            info[f"grad_delta_{name}"] = _leaf_norm(delta)

        self.target_update(new_network, "q", d=self.config["tau"])
        self.target_update(new_network, "v", d=self.config["tau"])
        self.target_update(new_network, "actor", d=1.0 - self.config["ema"])
        return self.replace(network=new_network, rng=new_rng), info

    def _sample_token(self, logits, rng, temperature):
        """Sample or argmax one token from (B, V) logits. Returns (ids, new_rng)."""
        if temperature == 0.0:
            return jnp.argmax(logits, axis=-1).astype(jnp.int32), rng
        rng, s_rng = jax.random.split(rng)
        sample_logits = logits / jnp.maximum(temperature, 1e-6)
        ids = jax.random.categorical(s_rng, sample_logits).astype(jnp.int32)
        return ids, rng

    def _ar_fill_tokens(self, observations, actor_name, sampling_temp, rng):
        """Autoregressive K-step token fill. Returns (tokens (B, K), rng)."""
        b = observations.shape[0]
        k = int(self.config["num_registers"])
        tokens = jnp.zeros((b, k), dtype=jnp.int32)
        # Static Python loop; K is config-static under JIT.
        for i in range(k):
            rng, step_rng = jax.random.split(rng)
            inp = jnp.full((b, k), BOS_ID, dtype=jnp.int32)
            if i > 0:
                inp = inp.at[:, 1 : i + 1].set(tokens[:, :i])
            logits = self._actor_logits(observations, inp, actor_name)
            next_ids, _ = self._sample_token(
                logits[:, i, :], step_rng, sampling_temp
            )
            tokens = tokens.at[:, i].set(next_ids)
        return tokens, rng

    @partial(jax.jit, static_argnames=("temperature",))
    def compute_ar_actions(self, observations, seed=None, temperature=0.0):
        """Autoregressive K-step decode via frozen OATTok. No critic at deploy.

        Seed contract (main/eval always pass a seed via ``supply_rng``):
        - sampling_temperature==0: deterministic argmax; seed unused for draws.
        - sampling_temperature>0 with supplied seed: fully reproducible.
        - ``seed is None``: falls back to ``self.rng`` but does **not** write
          the advanced key back — pure repeated calls with ``seed=None``
          therefore repeat the same sample.

        Actor selection: ``temperature==0`` → target_actor with config
        ``eval_sampling_temperature`` (default 1.0); ``temperature>0`` →
        online actor with that provided sampling temperature.
        """
        actor_name, sampling_temp = self._resolve_actor_and_sampling_temp(temperature)
        rng = seed if seed is not None else self.rng
        tokens, _ = self._ar_fill_tokens(
            observations, actor_name, sampling_temp, rng
        )
        codes = indices_to_codes(tokens, FSQ_LEVELS)
        actions = self._decode_codes(codes)
        return jnp.clip(actions, -1.0, 1.0)

    @partial(jax.jit, static_argnames=("temperature", "force_argmax"))
    def diagnose_teacher_vs_freerun(
        self,
        observations,
        actions_hbd,
        seed=None,
        temperature=0.0,
        force_argmax=True,
    ):
        """Compare teacher-forced vs free-running AR on one batch.

        Not used in training updates (avoids K transformer calls per step).
        Call from ``scripts/diagnose_discrete_ar_iql.py`` or occasional eval.

        Args:
            observations: (B, obs_dim) — typically ``batch['observations'][0]``.
            actions_hbd: (h+, B, Da) action chunk (uses first ``h`` steps).
            seed: optional PRNG key for freerun sampling.
            temperature: actor selection (same contract as ``compute_ar_actions``).
            force_argmax: if True (default), freerun uses argmax for deterministic
                token-accuracy reporting regardless of sampling temperature.
        """
        h = int(self.config["h"])
        k = int(self.config["num_registers"])
        actions_btd = rearrange(actions_hbd[:h], "h b d -> b h d")
        clean_tokens, clean_codes = self._encode_actions(actions_btd)
        clean_tokens = jax.lax.stop_gradient(clean_tokens.astype(jnp.int32))
        clean_codes = jax.lax.stop_gradient(clean_codes)
        clean_actions = jax.lax.stop_gradient(self._decode_codes(clean_codes))
        actions_flat = rearrange(actions_btd, "b h d -> b (h d)")

        actor_name, sampling_temp = self._resolve_actor_and_sampling_temp(temperature)
        if force_argmax:
            sampling_temp = 0.0

        # Teacher-forced (one transformer call).
        token_inputs = self.make_teacher_inputs(clean_tokens, BOS_ID)
        logits_tf = self._actor_logits(observations, token_inputs, actor_name)
        pred_tf = jnp.argmax(logits_tf, axis=-1)
        correct_tf = (pred_tf == clean_tokens).astype(jnp.float32)
        prefix_tf = self.prefix_exact_rates(pred_tf, clean_tokens)
        codes_tf = indices_to_codes(pred_tf, FSQ_LEVELS)
        actions_tf = jnp.clip(self._decode_codes(codes_tf), -1.0, 1.0)

        # Free-running AR (K transformer calls).
        rng = seed if seed is not None else self.rng
        tokens_fr, _ = self._ar_fill_tokens(
            observations, actor_name, sampling_temp, rng
        )
        correct_fr = (tokens_fr == clean_tokens).astype(jnp.float32)
        prefix_fr = self.prefix_exact_rates(tokens_fr, clean_tokens)
        codes_fr = indices_to_codes(tokens_fr, FSQ_LEVELS)
        actions_fr = jnp.clip(self._decode_codes(codes_fr), -1.0, 1.0)

        info = {
            "tf_token_acc": correct_tf.mean(),
            "tf_seq_exact": prefix_tf["seq_exact"],
            "fr_token_acc": correct_fr.mean(),
            "fr_seq_exact": prefix_fr["seq_exact"],
            "tf_action_rmse_gt": jnp.sqrt(
                jnp.mean(jnp.square(actions_tf - actions_flat))
            ),
            "tf_action_corr_gt": self.pearson_corr(actions_tf, actions_flat),
            "fr_action_rmse_gt": jnp.sqrt(
                jnp.mean(jnp.square(actions_fr - actions_flat))
            ),
            "fr_action_corr_gt": self.pearson_corr(actions_fr, actions_flat),
            "fr_action_rmse_clean": jnp.sqrt(
                jnp.mean(jnp.square(actions_fr - clean_actions))
            ),
            "fr_action_corr_clean": self.pearson_corr(actions_fr, clean_actions),
            "decode_rmse": jnp.sqrt(
                jnp.mean(jnp.square(clean_actions - actions_flat))
            ),
        }
        for i in range(k):
            info[f"tf_reg{i}_acc"] = correct_tf[:, i].mean()
            info[f"fr_reg{i}_acc"] = correct_fr[:, i].mean()
        for length in DEFAULT_PREFIX_LENGTHS:
            if length in prefix_tf:
                info[f"tf_prefix_exact_{length}"] = prefix_tf[length]
            if length in prefix_fr:
                info[f"fr_prefix_exact_{length}"] = prefix_fr[length]
        info[f"tf_prefix_exact_{k}"] = prefix_tf.get(k, prefix_tf["seq_exact"])
        info[f"fr_prefix_exact_{k}"] = prefix_fr.get(k, prefix_fr["seq_exact"])
        return info

    @partial(jax.jit, static_argnames=("temperature",))
    def sample_actions(self, obs, seed=None, temperature=0.0):
        """Sample an action chunk; see ``compute_ar_actions`` seed contract."""
        obs = jnp.atleast_2d(obs)[-1:]
        actions = self.compute_ar_actions(
            obs, seed=seed, temperature=temperature
        )[0]
        actions = rearrange(actions, "(h d) -> h d", h=self.config["h"])
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        config = dict(config)
        cls.validate_create_config(config)

        ex_actions = jnp.asarray(ex_actions)
        ex_observations = jnp.asarray(ex_observations)
        if ex_actions.ndim == 3:
            prim_dim = int(ex_actions.shape[-1])
            ex_obs = ex_observations[0]
            ex_act_step = ex_actions[0]
        elif ex_actions.ndim == 2:
            prim_dim = int(ex_actions.shape[-1])
            ex_obs = (
                ex_observations
                if ex_observations.ndim == 2
                else ex_observations[None, :]
            )
            ex_act_step = ex_actions
        else:
            prim_dim = int(ex_actions.shape[0])
            ex_obs = (
                ex_observations[None, :]
                if ex_observations.ndim == 1
                else ex_observations
            )
            ex_act_step = ex_actions[None, :]

        h = int(config["h"])
        action_dim = prim_dim * h
        num_registers = int(config["num_registers"])

        tokenizer_path = config.get("tokenizer_path", "")
        if not tokenizer_path:
            raise ValueError(
                "DiscreteARIQL requires agent.tokenizer_path to a "
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
        num_registers = int(tok_meta.get("num_registers", num_registers))
        config["num_registers"] = num_registers
        config = cls.finalize_register_weight_config(config, num_registers)

        tokenizer_def = OATTok(
            sample_dim=prim_dim,
            sample_horizon=h,
            num_registers=num_registers,
            emb_dim=int(tok_meta.get("emb_dim", 256)),
            encoder_depth=int(tok_meta.get("encoder_depth", 2)),
            decoder_depth=int(tok_meta.get("decoder_depth", 4)),
        )

        ex_flat_actions = jnp.concatenate([ex_act_step] * h, axis=-1)
        ex_token_inputs = jnp.full(
            (ex_obs.shape[0], num_registers), BOS_ID, dtype=jnp.int32
        )
        ex_q_in = jnp.concatenate([ex_obs, ex_flat_actions], axis=-1)

        q_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["ensemble_ct"],
        )
        v_def = Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            num_ensembles=config["v_ensemble_ct"],
        )
        # Dropout is rejected by validate_create_config; always pass 0.0.
        actor_def = CausalARActor(
            emb_dim=int(config["actor_emb_dim"]),
            depth=int(config["actor_depth"]),
            num_heads=int(config["actor_num_heads"]),
            vocab_size=CODEBOOK_SIZE,
            num_registers=num_registers,
            dropout=0.0,
        )

        network_info = dict(
            q=(q_def, (ex_q_in,)),
            target_q=(copy.deepcopy(q_def), (ex_q_in,)),
            v=(v_def, (ex_obs,)),
            target_v=(copy.deepcopy(v_def), (ex_obs,)),
            actor=(actor_def, (ex_obs, ex_token_inputs)),
            target_actor=(copy.deepcopy(actor_def), (ex_obs, ex_token_inputs)),
        )
        networks = {name: spec[0] for name, spec in network_info.items()}
        network_args = {name: spec[1] for name, spec in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_q"] = params["modules_q"]
        params["modules_target_v"] = params["modules_v"]
        params["modules_target_actor"] = params["modules_actor"]

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
            agent_name="discrete_ar_iql",
            h=1,
            alpha=1.0,
            q_coef=1.0,
            v_coef=1.0,
            expectile=0.9,
            ensemble_ct=2,
            v_ensemble_ct=1,
            rho=0.0,
            lr=3e-4,
            discount=0.995,
            batch_size=256,
            actor_emb_dim=256,
            actor_depth=4,
            actor_num_heads=4,
            # Must stay 0: dropout RNG is not plumbed; nonzero raises at create().
            actor_dropout=0.0,
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            tau=0.005,
            ema=0.999,
            num_registers=16,
            awr_temperature=5.0,
            max_weight=100.0,
            bc_warmup_steps=50_000,
            awr_ramp_steps=50_000,
            # Eval: temperature=0 selects target_actor; actual draws use this
            # (1.0 = categorical; 0.0 = argmax). External temperature>0 uses
            # the online actor with that sampling temperature.
            eval_sampling_temperature=1.0,
            tokenizer_path="",
            # Advantage source: 'iql' (Q-V AWR, default), 'mc_return'
            # (expectile V→G, A=G-V with shift-stable AWR), or
            # 'trajectory_success' (IQL V→Q preserved; actor BC weights from
            # episode-success flags). MC requires use_mc_returns=True;
            # trajectory_success requires use_trajectory_success=True.
            advantage_source="iql",
            use_mc_returns=False,
            mc_expectile=0.7,
            # MC actor BC warmup (default 0): stable weights are already
            # informative at init (no exp underflow), so focus immediately.
            # Set >0 to ablate delayed AWR; IQL mode still uses bc_warmup_steps.
            mc_actor_warmup_steps=0,
            # Trajectory-success weighted BC (current experiment). Off by default.
            # OGBench: masks<=0 → per-step success; all steps in any successful
            # episode get flag=1. Actor w=(1-r)+r*(1+(W-1)*flag); no exp/T.
            use_trajectory_success=False,
            success_weight=20.0,
            success_actor_warmup_steps=0,
            # Scheduled sampling (two-pass; no new network params).
            # At resume step>=ss_start+ss_ramp (e.g. 200k), λ and p are at max.
            # For initial MC/success experiments, launcher should set ss_loss_coef=0
            # (SS failed on maze); config keeps SS available for ablations.
            ss_start_steps=50_000,
            ss_ramp_steps=50_000,
            ss_loss_coef=0.5,
            ss_prefix_prob_min=0.1,
            ss_prefix_prob_max=0.35,
            # Stop-grad predictions from pass-A logits; argmax for stability.
            ss_pred_mode="argmax",
            # Empirical decoder-swap sens (K=16 maze); mean-normalized at use.
            # For initial MC/success experiments, launcher should disable unless set.
            use_register_weights=True,
            register_ce_weights=DEFAULT_EMPIRICAL_REGISTER_CE_WEIGHTS_K16,
            # v8 direct action-level critic gradients (off by default → legacy).
            # ST hard-forward/soft-backward codes → frozen decode → target_q.
            # Warmup/ramp gate via lax.cond; coef=0 skips decode/Q branch.
            q_actor_coef=0.0,
            q_actor_warmup_steps=50_000,
            q_actor_ramp_steps=50_000,
            # 'teacher_forced' reuses pass-A logits; 'self_conditioned' runs a
            # second actor pass on BOS + shifted stop-grad argmax predictions.
            q_actor_prefix_mode="teacher_forced",
            st_temperature=1.0,
        )
    )
    return config
