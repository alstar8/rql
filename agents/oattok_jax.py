"""Flax Ordered Action Tokenizer (CRAFT-style) for OGBench / RQL.

Faithful mirror of oat OATTok pieces used by CRAFT:
  RegisterEncoder -> FSQ([8,5,5,5]) -> ResidualBoostDecoder

Trained once per OGBench domain, then frozen inside ConsensusDiscreteFlow.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

FSQ_LEVELS = (8, 5, 5, 5)
FSQ_DIM = len(FSQ_LEVELS)
CODEBOOK_SIZE = int(np.prod(FSQ_LEVELS))  # 1000


def default_init(scale: float = 1.0):
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def fsq_basis(levels: Sequence[int] = FSQ_LEVELS) -> jnp.ndarray:
    levels_t = jnp.asarray(levels, dtype=jnp.int32)
    return jnp.cumprod(
        jnp.concatenate([jnp.ones((1,), dtype=jnp.int32), levels_t[:-1]])
    )


def fsq_bound(
    z: jnp.ndarray, levels: Sequence[int] = FSQ_LEVELS, eps: float = 1e-3
) -> jnp.ndarray:
    levels_t = jnp.asarray(levels, dtype=z.dtype)
    half_l = (levels_t - 1.0) * (1.0 + eps) / 2.0
    offset = jnp.where(levels_t % 2 == 0, 0.5, 0.0)
    shift = jnp.arctanh(offset / half_l)
    return jnp.tanh(z + shift) * half_l - offset


def fsq_quantize(z: jnp.ndarray, levels: Sequence[int] = FSQ_LEVELS) -> jnp.ndarray:
    levels_t = jnp.asarray(levels, dtype=z.dtype)
    bounded = fsq_bound(z, levels)
    quantized = bounded + jax.lax.stop_gradient(jnp.round(bounded) - bounded)
    half_width = levels_t // 2
    return quantized / half_width


def codes_to_indices(
    zhat: jnp.ndarray, levels: Sequence[int] = FSQ_LEVELS
) -> jnp.ndarray:
    levels_t = jnp.asarray(levels, dtype=jnp.int32)
    basis = fsq_basis(levels)
    half_width = levels_t // 2
    zhat_shifted = (zhat * half_width) + half_width
    return (zhat_shifted * basis).sum(axis=-1).astype(jnp.int32)


def indices_to_codes(
    indices: jnp.ndarray, levels: Sequence[int] = FSQ_LEVELS
) -> jnp.ndarray:
    levels_t = jnp.asarray(levels, dtype=jnp.int32)
    basis = fsq_basis(levels)
    half_width = levels_t // 2
    indices = indices[..., None]
    codes_non_centered = (indices // basis) % levels_t
    return (codes_non_centered - half_width) / half_width.astype(jnp.float32)


def build_codebook(levels: Sequence[int] = FSQ_LEVELS) -> jnp.ndarray:
    """Implicit FSQ codebook of shape (V, q)."""
    return indices_to_codes(
        jnp.arange(int(np.prod(levels)), dtype=jnp.int32), levels
    )


def token_ids_to_coord_indices(
    indices: jnp.ndarray, levels: Sequence[int] = FSQ_LEVELS
) -> jnp.ndarray:
    """Unpack flat FSQ token IDs to mixed-radix coordinate indices.

    For ``levels=(8,5,5,5)`` the basis is ``[1,8,40,200]``. Returns integer
    coords in ``[0, L_i)`` with shape ``(..., q)``.
    """
    levels_t = jnp.asarray(levels, dtype=jnp.int32)
    basis = fsq_basis(levels)
    return ((indices[..., None] // basis) % levels_t).astype(jnp.int32)


def coord_indices_to_token_ids(
    coords: jnp.ndarray, levels: Sequence[int] = FSQ_LEVELS
) -> jnp.ndarray:
    """Pack mixed-radix coordinate indices to flat FSQ token IDs."""
    basis = fsq_basis(levels)
    return (coords.astype(jnp.int32) * basis).sum(axis=-1).astype(jnp.int32)


def fsq_class_valid_mask(
    levels: Sequence[int] = FSQ_LEVELS, max_classes: Optional[int] = None
) -> jnp.ndarray:
    """Boolean mask ``(q, max_classes)`` of valid logit slots per FSQ axis."""
    levels_t = np.asarray(levels, dtype=np.int32)
    width = int(max_classes) if max_classes is not None else int(levels_t.max())
    mask = np.zeros((len(levels_t), width), dtype=bool)
    for i, level in enumerate(levels_t):
        mask[i, : int(level)] = True
    return jnp.asarray(mask)


class TransformerBlock(nn.Module):
    emb_dim: int
    num_heads: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.emb_dim,
            dropout_rate=self.dropout,
            deterministic=deterministic,
        )(h, h, mask=mask)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(4 * self.emb_dim, kernel_init=default_init())(h)
        h = nn.gelu(h)
        h = nn.Dense(self.emb_dim, kernel_init=default_init())(h)
        return x + h


class RegisterEncoder(nn.Module):
    """Action chunk -> K continuous FSQ latents (CRAFT register encoder)."""

    sample_dim: int
    sample_horizon: int
    emb_dim: int = 256
    head_dim: int = 64
    depth: int = 2
    latent_dim: int = FSQ_DIM
    num_registers: int = 12
    dropout: float = 0.0

    @nn.compact
    def __call__(self, sample: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        b, t, _ = sample.shape
        num_heads = max(1, self.emb_dim // self.head_dim)
        x = nn.Dense(self.emb_dim, kernel_init=default_init())(sample)
        pos = self.param(
            "action_pos",
            nn.initializers.normal(0.02),
            (1, self.sample_horizon, self.emb_dim),
        )
        x = x + pos[:, :t, :]
        registers = self.param(
            "registers",
            nn.initializers.normal(0.02),
            (1, self.num_registers, self.emb_dim),
        )
        x = jnp.concatenate(
            [
                x,
                jnp.broadcast_to(registers, (b, self.num_registers, self.emb_dim)),
            ],
            axis=1,
        )

        total = t + self.num_registers
        allow = jnp.ones((total, total), dtype=bool)
        allow = allow.at[:t, t:].set(False)
        reg_causal = jnp.tril(
            jnp.ones((self.num_registers, self.num_registers), dtype=bool)
        )
        allow = allow.at[t:, t:].set(reg_causal)
        attn_mask = allow[None, None, :, :]

        for i in range(self.depth):
            x = TransformerBlock(
                self.emb_dim, num_heads, self.dropout, name=f"enc_block_{i}"
            )(x, mask=attn_mask, deterministic=deterministic)
        latents = nn.Dense(self.latent_dim, kernel_init=default_init())(x[:, t:, :])
        return latents


class ResidualBoostDecoder(nn.Module):
    """Exact prefix residual decoder (CRAFT ResidualBoostDecoder)."""

    sample_dim: int
    sample_horizon: int
    emb_dim: int = 256
    head_dim: int = 64
    depth: int = 4
    latent_dim: int = FSQ_DIM
    latent_horizon: int = 12
    boost_hidden_dim: int = 64
    gain_decay: float = 0.9
    dropout: float = 0.0

    @nn.compact
    def __call__(
        self,
        latents: jnp.ndarray,
        keep_k: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        b, k, _ = latents.shape
        num_heads = max(1, self.emb_dim // self.head_dim)
        ctx = nn.Dense(
            self.emb_dim, kernel_init=default_init(), name="latent_proj"
        )(latents)
        pos = self.param(
            "latent_pos",
            nn.initializers.normal(0.02),
            (1, self.latent_horizon, self.emb_dim),
        )
        ctx = ctx + pos[:, :k, :]
        causal = jnp.tril(jnp.ones((k, k), dtype=bool))[None, None, :, :]
        for i in range(self.depth):
            ctx = TransformerBlock(
                self.emb_dim, num_heads, self.dropout, name=f"dec_block_{i}"
            )(ctx, mask=causal, deterministic=deterministic)

        time_pos = self.param(
            "time_pos",
            nn.initializers.normal(0.02),
            (self.sample_horizon, self.emb_dim),
        )
        q = nn.silu(
            nn.Dense(
                self.boost_hidden_dim, kernel_init=default_init(), name="time_proj"
            )(time_pos)
        )
        c = nn.silu(
            nn.Dense(
                self.boost_hidden_dim, kernel_init=default_init(), name="ctx_proj"
            )(ctx)
        )
        h = q[None, None, :, :] * c[:, :, None, :]
        delta = nn.Dense(
            self.sample_dim,
            kernel_init=nn.initializers.zeros,
            name="delta_head",
        )(h)

        gains0 = self.gain_decay ** jnp.arange(self.latent_horizon, dtype=jnp.float32)
        gains = self.param("stage_gain", lambda *_: gains0)
        contrib = gains[:k][None, :, None, None] * delta

        if keep_k is None:
            keep_k = jnp.full((b,), k, dtype=jnp.int32)
        stage_ids = jnp.arange(k)[None, :]
        stage_mask = (stage_ids < keep_k[:, None]).astype(contrib.dtype)
        samples = jnp.einsum("bktd,bk->btd", contrib, stage_mask)
        return samples


class OATTok(nn.Module):
    """Full CRAFT-style ordered action tokenizer."""

    sample_dim: int
    sample_horizon: int
    num_registers: int = 12
    emb_dim: int = 256
    encoder_depth: int = 2
    decoder_depth: int = 4
    levels: Sequence[int] = FSQ_LEVELS

    def setup(self):
        self.encoder = RegisterEncoder(
            sample_dim=self.sample_dim,
            sample_horizon=self.sample_horizon,
            emb_dim=self.emb_dim,
            depth=self.encoder_depth,
            latent_dim=len(self.levels),
            num_registers=self.num_registers,
        )
        self.decoder = ResidualBoostDecoder(
            sample_dim=self.sample_dim,
            sample_horizon=self.sample_horizon,
            emb_dim=self.emb_dim,
            depth=self.decoder_depth,
            latent_dim=len(self.levels),
            latent_horizon=self.num_registers,
        )

    def encode(self, sample: jnp.ndarray, deterministic: bool = True):
        latents = self.encoder(sample, deterministic=deterministic)
        quant = fsq_quantize(latents, self.levels)
        tokens = codes_to_indices(quant, self.levels)
        return quant, tokens

    def decode(self, codes: jnp.ndarray, keep_k=None, deterministic: bool = True):
        return self.decoder(codes, keep_k=keep_k, deterministic=deterministic)

    def detokenize(self, tokens: jnp.ndarray, deterministic: bool = True):
        codes = indices_to_codes(tokens, self.levels)
        return self.decode(codes, deterministic=deterministic)

    def __call__(self, sample: jnp.ndarray, keep_k=None, deterministic: bool = True):
        quant, tokens = self.encode(sample, deterministic=deterministic)
        recons = self.decode(quant, keep_k=keep_k, deterministic=deterministic)
        return recons, tokens, quant


def sample_pow2_keep_k(rng, batch_size: int, num_registers: int) -> jnp.ndarray:
    budgets = [1]
    k = 2
    while k < num_registers:
        budgets.append(k)
        k *= 2
    if num_registers not in budgets:
        budgets.append(num_registers)
    budgets = jnp.asarray(budgets, dtype=jnp.int32)
    idx = jax.random.randint(rng, (batch_size,), 0, budgets.shape[0])
    return budgets[idx]


def create_tokenizer_state(
    rng,
    sample_dim: int,
    sample_horizon: int,
    lr: float = 5e-5,
    num_registers: int = 12,
    emb_dim: int = 256,
) -> Tuple[OATTok, train_state.TrainState]:
    model = OATTok(
        sample_dim=sample_dim,
        sample_horizon=sample_horizon,
        num_registers=num_registers,
        emb_dim=emb_dim,
    )
    dummy = jnp.zeros((1, sample_horizon, sample_dim), dtype=jnp.float32)
    params = model.init(rng, dummy, deterministic=True)["params"]
    tx = optax.adamw(learning_rate=lr, weight_decay=0.0, b1=0.9, b2=0.95)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    return model, state


@partial(jax.jit, static_argnames=("num_registers",))
def tokenizer_train_step(state, batch, rng, num_registers: int = 12):
    rng, k_rng = jax.random.split(rng)

    def loss_fn(params):
        keep_k = sample_pow2_keep_k(k_rng, batch.shape[0], num_registers)
        recons, _, _ = state.apply_fn(
            {"params": params},
            batch,
            keep_k=keep_k,
            deterministic=True,
        )
        full, tokens, _ = state.apply_fn(
            {"params": params},
            batch,
            keep_k=None,
            deterministic=True,
        )
        loss = jnp.mean(jnp.square(recons - batch))
        full_rmse = jnp.sqrt(jnp.mean(jnp.square(full - batch)))
        return loss, (full_rmse, tokens)

    (loss, (full_rmse, tokens)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    state = state.apply_gradients(grads=grads)
    return state, {
        "tok_loss": loss,
        "tok_rmse": full_rmse,
        "token_mean": tokens.astype(jnp.float32).mean(),
    }


def save_tokenizer(path: str, params: Any, meta: dict):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(
            {
                "params": jax.tree_util.tree_map(lambda x: np.asarray(x), params),
                "meta": meta,
            },
            f,
        )


def load_tokenizer(path: str):
    import pickle
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["params"], payload["meta"]
