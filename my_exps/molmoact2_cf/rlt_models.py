"""RLT-style RL token + chunk actor/critic + CF consensus guide for MolmoAct2."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import (
    FEATURE_DIM,
    PROPRIO_DIM,
    Z_DIM,
    EnsembleCQL,
    LogAlpha,
    mlp,
)

CHUNK_SIZE = 8
ACTION_DIM = 8
STATE_DIM = Z_DIM + PROPRIO_DIM  # 264
SCHEMA_VERSION = 4


def _sinusoidal_positions(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(dim, device=device, dtype=dtype).unsqueeze(0)
    angles = pos / (10000 ** (2 * (i // 2) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe


class RLTokenEncoder(nn.Module):
    """Append a learned <rl> token and read it out via a small transformer encoder."""

    def __init__(
        self,
        token_dim: int = FEATURE_DIM,
        z_dim: int = Z_DIM,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_len: int = 768,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.z_dim = int(z_dim)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.in_proj = nn.Linear(token_dim, d_model)
        self.rl_embed = nn.Parameter(torch.randn(d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, z_dim))

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode tokens (B,S,D) → z_rl (B,z_dim). Mask: 1=valid."""
        b, s, _ = tokens.shape
        if s + 1 > self.max_len:
            raise ValueError(f"sequence length {s}+1 exceeds max_len={self.max_len}")
        x = self.in_proj(tokens)
        rl = self.rl_embed.view(1, 1, -1).expand(b, 1, -1)
        x = torch.cat([x, rl], dim=1)
        pe = _sinusoidal_positions(s + 1, self.d_model, x.device, x.dtype)
        x = x + pe.unsqueeze(0)
        if attention_mask is None:
            attn_mask = None
            key_padding = None
        else:
            pad = attention_mask.new_ones(b, 1)
            full = torch.cat([attention_mask, pad], dim=1).bool()
            key_padding = ~full  # True = ignore
            attn_mask = None
        h = self.encoder(x, mask=attn_mask, src_key_padding_mask=key_padding)
        z = h[:, -1]
        return self.out_proj(z)


class RLTokenDecoder(nn.Module):
    """Teacher-forced causal decoder that reconstructs VLA tokens from z_rl."""

    def __init__(
        self,
        token_dim: int = FEATURE_DIM,
        z_dim: int = Z_DIM,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_len: int = 768,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.z_proj = nn.Linear(z_dim, d_model)
        self.tok_proj = nn.Linear(token_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, token_dim)

    def forward(self, z_rl: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Predict tokens (B,S,D) from z_rl (B,z) and teacher targets (B,S,D)."""
        b, s, _ = target_tokens.shape
        memory = self.z_proj(z_rl).unsqueeze(1)  # (B,1,d)
        # Teacher forcing: position i sees z_rl-memory + tokens 0..i-1.
        # Use shifted targets with a learned start from z_rl itself.
        start = memory
        if s > 1:
            prev = self.tok_proj(target_tokens[:, :-1].detach())
            tgt = torch.cat([start, prev], dim=1)
        else:
            tgt = start
        pe = _sinusoidal_positions(s, self.d_model, tgt.device, tgt.dtype)
        tgt = tgt + pe.unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(s, device=tgt.device)
        out = self.decoder(tgt, memory, tgt_mask=causal)
        return self.out_proj(out)


class RLTokenAutoencoder(nn.Module):
    """Encoder-decoder bottleneck for the RLT readout."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.encoder = RLTokenEncoder(**kwargs)
        self.decoder = RLTokenDecoder(**kwargs)

    def encode(self, tokens: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(tokens, attention_mask)

    def reconstruction_loss(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(tokens, attention_mask)
        pred = self.decoder(z, tokens.detach())
        err = (pred - tokens.detach()) ** 2
        if attention_mask is None:
            loss = err.mean()
        else:
            w = attention_mask.to(dtype=err.dtype).unsqueeze(-1)
            loss = (err * w).sum() / w.sum().clamp_min(1.0) / err.shape[-1]
        return loss, {"z_rl": z, "recon_mse": loss.detach()}


class ChunkGaussianActor(nn.Module):
    """Reference-conditioned Gaussian chunk actor (RLT-style)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden: int = 256,
        log_std: float = -2.0,
        residual: bool = True,
        max_delta: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.flat_action = self.action_dim * self.chunk_size
        self.residual = bool(residual)
        self.max_delta = float(max_delta)
        self.net = mlp(state_dim + self.flat_action + 1, self.flat_action, hidden, n_hidden=2, zero_out=True)
        self.register_buffer("log_std", torch.full((self.flat_action,), float(log_std)))

    def forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        reference_present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean chunk (B,C,A) and flat log_std (flat_action,)."""
        b = state.shape[0]
        ref = reference.reshape(b, -1)
        if reference_present is None:
            present = torch.ones(b, 1, device=state.device, dtype=state.dtype)
            ref_in = ref
        else:
            present = reference_present.reshape(b, 1).to(dtype=state.dtype)
            ref_in = ref * present
        raw = self.net(torch.cat([state, ref_in.detach(), present], dim=-1))
        if self.residual:
            delta = self.max_delta * torch.tanh(raw)
            mean = ref.detach() + delta
        else:
            mean = raw
        mean = mean.reshape(b, self.chunk_size, self.action_dim)
        return mean, self.log_std

    def sample(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        reference_present: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state, reference, reference_present)
        if deterministic:
            return mean, mean
        std = log_std.exp().view(1, self.chunk_size, self.action_dim)
        eps = torch.randn_like(mean)
        return mean + std * eps, mean


class CFGradientGuide(nn.Module):
    """Distill common-scale-normalized ensemble action gradients into a bounded guide."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden: int = 256,
        max_delta: float = 0.05,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.flat_action = self.action_dim * self.chunk_size
        self.max_delta = float(max_delta)
        self.net = mlp(state_dim + self.flat_action, self.flat_action, hidden, n_hidden=2, zero_out=True)

    def forward(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        flat = reference.reshape(b, -1).detach()
        raw = self.net(torch.cat([state, flat], dim=-1))
        delta = self.max_delta * torch.tanh(raw)
        return delta.reshape(b, self.chunk_size, self.action_dim)

    def guide(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        actor_delta: torch.Tensor | None = None,
        trust: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.forward(state, reference)
        if actor_delta is not None:
            # Conflict damping: shrink guide where it opposes the actor correction.
            oppose = ((delta * actor_delta.detach()) < 0).to(dtype=delta.dtype)
            delta = delta * (1.0 - 0.5 * oppose)
        delta = delta * float(trust)
        return reference.detach() + delta, delta


class MolmoAct2RLTCF(nn.Module):
    """Full RLT+CF stack: RL token, chunk critic/actor, optional CF guide."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        z_dim: int = Z_DIM,
        proprio_dim: int = PROPRIO_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden: int = 256,
        n_critics: int = 10,  # ConsensusFlow default ensemble size K=10
        bounded_critic: bool = True,
        residual_actor: bool = True,
        max_delta: float = 0.05,
        use_cf_guide: bool = True,
        tune_token_online: bool = True,
        token_d_model: int = 256,
        token_layers: int = 2,
        token_heads: int = 4,
    ) -> None:
        super().__init__()
        self.schema_version = SCHEMA_VERSION
        self.feature_dim = int(feature_dim)
        self.z_dim = int(z_dim)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.state_dim = self.z_dim + self.proprio_dim
        self.bounded_critic = bool(bounded_critic)
        self.use_cf_guide = bool(use_cf_guide)
        self.tune_token_online = bool(tune_token_online)
        self.flat_action = self.action_dim * self.chunk_size
        self.hidden = int(hidden)
        self.n_critics = int(n_critics)
        self.residual_actor = bool(residual_actor)
        self.max_delta = float(max_delta)
        self.token_d_model = int(token_d_model)
        self.token_layers = int(token_layers)
        self.token_heads = int(token_heads)

        self.token_ae = RLTokenAutoencoder(
            token_dim=self.feature_dim,
            z_dim=self.z_dim,
            d_model=token_d_model,
            n_heads=token_heads,
            n_layers=token_layers,
        )
        # EMA target encoder for stable TD bootstrap states.
        self.target_token_encoder = RLTokenEncoder(
            token_dim=self.feature_dim,
            z_dim=self.z_dim,
            d_model=token_d_model,
            n_heads=token_heads,
            n_layers=token_layers,
        )
        self.target_token_encoder.load_state_dict(self.token_ae.encoder.state_dict())
        for p in self.target_token_encoder.parameters():
            p.requires_grad_(False)

        self.actor = ChunkGaussianActor(
            self.state_dim,
            self.action_dim,
            self.chunk_size,
            hidden=hidden,
            residual=residual_actor,
            max_delta=max_delta,
        )
        self.critic = EnsembleCQL(
            self.state_dim,
            self.flat_action,
            n_critics=n_critics,
            hidden=hidden,
            bounded=self.bounded_critic,
        )
        self.target_critic = EnsembleCQL(
            self.state_dim,
            self.flat_action,
            n_critics=n_critics,
            hidden=hidden,
            bounded=self.bounded_critic,
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

        self.guide = (
            CFGradientGuide(
                self.state_dim,
                self.action_dim,
                self.chunk_size,
                hidden=hidden,
                max_delta=max_delta,
            )
            if self.use_cf_guide
            else None
        )
        self.log_alpha = LogAlpha(1.0)

        self.register_buffer("proprio_mean", torch.zeros(self.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(self.proprio_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))

    def set_norm_stats(
        self,
        proprio_mean: torch.Tensor | np.ndarray,
        proprio_std: torch.Tensor | np.ndarray,
        action_mean: torch.Tensor | np.ndarray,
        action_std: torch.Tensor | np.ndarray,
        feature_mean: torch.Tensor | np.ndarray | None = None,
        feature_std: torch.Tensor | np.ndarray | None = None,
    ) -> None:
        self.proprio_mean.copy_(torch.as_tensor(proprio_mean, dtype=torch.float32))
        self.proprio_std.copy_(torch.as_tensor(proprio_std, dtype=torch.float32))
        self.action_mean.copy_(torch.as_tensor(action_mean, dtype=torch.float32))
        self.action_std.copy_(torch.as_tensor(action_std, dtype=torch.float32))
        if feature_mean is not None:
            self.feature_mean.copy_(torch.as_tensor(feature_mean, dtype=torch.float32))
        if feature_std is not None:
            self.feature_std.copy_(torch.as_tensor(feature_std, dtype=torch.float32))

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return (proprio - self.proprio_mean) / self.proprio_std.clamp_min(1e-6)

    def normalize_features(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.feature_mean) / self.feature_std.clamp_min(1e-3)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std.clamp_min(1e-6)

    def denormalize_action(self, action_n: torch.Tensor) -> torch.Tensor:
        return action_n * self.action_std + self.action_mean

    def encode_z(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_target: bool = False,
        detach: bool = False,
    ) -> torch.Tensor:
        h = self.normalize_features(tokens)
        enc = self.target_token_encoder if use_target else self.token_ae.encoder
        z = enc(h, attention_mask)
        return z.detach() if detach else z

    def encode_state_from_z(self, z: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return torch.cat([z, self.normalize_proprio(proprio)], dim=-1)

    def encode_state(
        self,
        tokens: torch.Tensor,
        proprio: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        use_target: bool = False,
        detach_token: bool = False,
    ) -> torch.Tensor:
        z = self.encode_z(tokens, attention_mask, use_target=use_target, detach=detach_token)
        return self.encode_state_from_z(z, proprio)

    def freeze_token_encoder(self) -> None:
        for p in self.token_ae.parameters():
            p.requires_grad_(False)
        self.tune_token_online = False

    def unfreeze_token_encoder(self) -> None:
        for p in self.token_ae.parameters():
            p.requires_grad_(True)
        self.tune_token_online = True

    @torch.no_grad()
    def soft_update_targets(self, tau: float = 0.005) -> None:
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for p, tp in zip(self.token_ae.encoder.parameters(), self.target_token_encoder.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def actor_chunk(
        self,
        state: torch.Tensor,
        reference_n: torch.Tensor,
        *,
        deterministic: bool = True,
        apply_guide: bool = False,
        reference_present: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Actor/guide see stop-grad state so token updates come only from recon/critic.
        s = state.detach()
        sample, mean = self.actor.sample(
            s, reference_n, reference_present=reference_present, deterministic=deterministic
        )
        info: dict[str, torch.Tensor] = {"actor_mean": mean, "actor_delta": mean - reference_n.detach()}
        out = sample
        if apply_guide and self.guide is not None:
            guided, g_delta = self.guide.guide(s, reference_n, actor_delta=info["actor_delta"])
            out = guided
            info["guide_delta"] = g_delta
        return out, info

    def q_chunk(self, state: torch.Tensor, actions_n: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        critic = self.target_critic if target else self.critic
        flat = actions_n.reshape(actions_n.shape[0], -1)
        return critic(state, flat)

    def q_min_chunk(self, state: torch.Tensor, actions_n: torch.Tensor, *, target: bool = False) -> torch.Tensor:
        critic = self.target_critic if target else self.critic
        flat = actions_n.reshape(actions_n.shape[0], -1)
        return critic.q_min(state, flat)

    def save(self, path: str, meta: dict[str, Any] | None = None) -> None:
        payload = {
            "schema_version": self.schema_version,
            "state_dict": self.state_dict(),
            "feature_dim": self.feature_dim,
            "z_dim": self.z_dim,
            "proprio_dim": self.proprio_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "bounded_critic": self.bounded_critic,
            "use_cf_guide": self.use_cf_guide,
            "tune_token_online": self.tune_token_online,
            "hidden": self.hidden,
            "n_critics": self.n_critics,
            "residual_actor": self.residual_actor,
            "max_delta": self.max_delta,
            "token_d_model": self.token_d_model,
            "token_layers": self.token_layers,
            "token_heads": self.token_heads,
            "meta": meta or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "MolmoAct2RLTCF":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            feature_dim=int(payload.get("feature_dim", FEATURE_DIM)),
            z_dim=int(payload.get("z_dim", Z_DIM)),
            proprio_dim=int(payload.get("proprio_dim", PROPRIO_DIM)),
            action_dim=int(payload.get("action_dim", ACTION_DIM)),
            chunk_size=int(payload.get("chunk_size", CHUNK_SIZE)),
            bounded_critic=bool(payload.get("bounded_critic", True)),
            use_cf_guide=bool(payload.get("use_cf_guide", True)),
            tune_token_online=bool(payload.get("tune_token_online", True)),
            hidden=int(payload.get("hidden", 256)),
            n_critics=int(payload.get("n_critics", 10)),
            residual_actor=bool(payload.get("residual_actor", True)),
            max_delta=float(payload.get("max_delta", 0.05)),
            token_d_model=int(payload.get("token_d_model", 256)),
            token_layers=int(payload.get("token_layers", 2)),
            token_heads=int(payload.get("token_heads", 4)),
        )
        model.load_state_dict(payload["state_dict"], strict=False)
        return model


def normalized_grad_target(
    grads: list[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Common-scale normalize per-critic action gradients and average them."""
    stacked = torch.stack(grads, dim=0)  # (E,B,C,A)
    scales = stacked.flatten(2).norm(dim=-1, keepdim=True).clamp_min(eps)  # (E,B,1)
    scales = scales.unsqueeze(-1)
    unit = stacked / scales
    return unit.mean(dim=0)


def chunk_return(
    rewards: torch.Tensor,
    gamma: float,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discounted return over a chunk: rewards (B,C), optional mask (B,C)."""
    b, c = rewards.shape
    disc = torch.pow(rewards.new_tensor(gamma), torch.arange(c, device=rewards.device))
    r = rewards * disc.view(1, -1)
    if action_mask is not None:
        r = r * action_mask.to(dtype=r.dtype)
    return r.sum(dim=-1)


def bootstrap_scale(
    gamma: float,
    chunk_size: int,
    action_mask: torch.Tensor | None,
    terminal: torch.Tensor,
) -> torch.Tensor:
    """γ^k * (1-terminal) where k is valid actions in the chunk."""
    if action_mask is None:
        k = torch.full((terminal.shape[0],), float(chunk_size), device=terminal.device)
    else:
        k = action_mask.to(dtype=torch.float32).sum(dim=-1)
    return (gamma ** k) * (1.0 - terminal.to(dtype=torch.float32))
