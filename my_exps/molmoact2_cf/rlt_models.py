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
SCHEMA_VERSION = 5
CF_MODE_RESIDUAL = "residual"
CF_MODE_FLOW = "flow"
DEFAULT_FLOW_STEPS = 10
DEFAULT_GUIDANCE_COEF = 0.5
DEFAULT_CONSENSUS_FLOOR = 0.01
DEFAULT_CONFLICT_POWER = 2.0
DEFAULT_RESIDUAL_DAMP = 0.25


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


def sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) or (B,1) in [0,1] → (B, dim) Fourier features."""
    if t.ndim == 2:
        t = t.squeeze(-1)
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 2.0 * math.pi
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class TimeCriticHead(nn.Module):
    """Success-return head Q(s, x, t) with bounded sigmoid output."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        time_dim: int = 64,
        bounded: bool = True,
    ) -> None:
        super().__init__()
        self.bounded = bool(bounded)
        self.time_dim = int(time_dim)
        self.net = mlp(state_dim + action_dim + time_dim, 1, hidden, n_hidden=2)

    def forward(self, state: torch.Tensor, actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = actions.reshape(actions.shape[0], -1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        temb = sinusoidal_time_embed(t.squeeze(-1), self.time_dim)
        logits = self.net(torch.cat([state, flat, temb], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits) if self.bounded else logits


class EnsembleTimeCQL(nn.Module):
    """Time-conditioned ensemble critic for flow-time ConsensusFlow."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_critics: int = 10,
        hidden: int = 256,
        time_dim: int = 64,
        bounded: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.bounded = bool(bounded)
        self.critics = nn.ModuleList(
            [
                TimeCriticHead(state_dim, action_dim, hidden, time_dim, bounded=bounded)
                for _ in range(n_critics)
            ]
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.stack([c(state, actions, t) for c in self.critics], dim=0)

    def q_mean(self, state: torch.Tensor, actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(state, actions, t).mean(dim=0)

    def q_min(self, state: torch.Tensor, actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward(state, actions, t).min(dim=0).values

    def cql_penalty(
        self,
        state: torch.Tensor,
        data_actions: torch.Tensor,
        t: torch.Tensor,
        n_actions: int = 4,
        coef: float = 1.0,
        action_radius: float = 0.05,
        margin: float = 0.0,
        far_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if n_actions < 1:
            zero = data_actions.new_zeros(())
            return zero, {
                "cql_loss": zero.detach(),
                "cql_logmeanexp_q": zero.detach(),
                "cql_data_q": self.q_mean(state, data_actions, t).mean().detach(),
                "cql_gap": zero.detach(),
            }
        b = state.shape[0]
        noise = torch.randn(
            n_actions, b, self.action_dim, device=state.device, dtype=state.dtype
        ) * float(far_scale)
        norms = noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        min_norm = float(action_radius) + 1e-3
        noise = noise * (min_norm + F.relu(norms - min_norm)) / norms
        far_actions = data_actions.unsqueeze(0) + noise
        state_rep = state.unsqueeze(0).expand(n_actions, -1, -1).reshape(n_actions * b, -1)
        far_flat = far_actions.reshape(n_actions * b, -1)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        t_rep = t.unsqueeze(0).expand(n_actions, -1, -1).reshape(n_actions * b, -1)
        far_q = self.q_mean(state_rep, far_flat, t_rep).reshape(n_actions, b)
        logmeanexp_q = torch.logsumexp(far_q, dim=0) - math.log(n_actions)
        data_q = self.q_mean(state, data_actions, t)
        gap = logmeanexp_q - data_q + margin
        cql = F.relu(gap).mean()
        return coef * cql, {
            "cql_loss": cql.detach(),
            "cql_logmeanexp_q": logmeanexp_q.mean().detach(),
            "cql_data_q": data_q.mean().detach(),
            "cql_gap": gap.mean().detach(),
        }


class FlowVelocityActor(nn.Module):
    """VLA-conditioned flow velocity: v_θ(s, x_t, t, a_VLA)."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden: int = 256,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.flat_action = self.action_dim * self.chunk_size
        self.time_dim = int(time_dim)
        # state + x_t + a_ref + t
        self.net = mlp(
            state_dim + 2 * self.flat_action + time_dim,
            self.flat_action,
            hidden,
            n_hidden=3,
            zero_out=False,
        )

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        b = state.shape[0]
        x_flat = x_t.reshape(b, -1)
        ref_flat = reference.reshape(b, -1).detach()
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        temb = sinusoidal_time_embed(t.squeeze(-1), self.time_dim)
        v = self.net(torch.cat([state, x_flat, ref_flat, temb], dim=-1))
        return v.reshape(b, self.chunk_size, self.action_dim)


class FlowCFGuide(nn.Module):
    """Paper ConsensusFlow guide: W_φ(s,x,t) → bounded G_φ with trust/conflict/damping."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden: int = 256,
        time_dim: int = 64,
        guidance_coef: float = DEFAULT_GUIDANCE_COEF,
        conflict_power: float = DEFAULT_CONFLICT_POWER,
        residual_coef: float = DEFAULT_RESIDUAL_DAMP,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.flat_action = self.action_dim * self.chunk_size
        self.time_dim = int(time_dim)
        self.guidance_coef = float(guidance_coef)
        self.conflict_power = float(conflict_power)
        self.residual_coef = float(residual_coef)
        self.net = mlp(
            state_dim + self.flat_action + time_dim,
            self.flat_action,
            hidden,
            n_hidden=3,
            zero_out=True,
        )

    def raw_w(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        b = state.shape[0]
        x_flat = x_t.reshape(b, -1).detach()
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        temb = sinusoidal_time_embed(t.squeeze(-1), self.time_dim)
        return self.net(torch.cat([state, x_flat, temb], dim=-1))

    @staticmethod
    def _project_unit_ball(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        norms = w.norm(dim=-1, keepdim=True).clamp_min(eps)
        return w * torch.minimum(torch.ones_like(norms), 1.0 / norms)

    def behavior_safe_direction(
        self,
        w: torch.Tensor,
        behavior_velocity: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat_v = behavior_velocity.reshape(behavior_velocity.shape[0], -1).detach()
        w = self._project_unit_ball(w)
        w_norm = w.norm(dim=-1, keepdim=True)
        trust = w_norm.detach()
        v_norm = flat_v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v_unit = flat_v / v_norm
        parallel = (w * v_unit).sum(dim=-1, keepdim=True)
        kill_frac = 1.0 - torch.pow(trust.clamp(0.0, 1.0), self.conflict_power)
        conflict_free = self._project_unit_ball(w - kill_frac * torch.minimum(parallel, torch.zeros_like(parallel)) * v_unit)
        alignment_cos = parallel / (w_norm + 1e-6)
        damp = (1.0 - self.residual_coef * torch.relu(alignment_cos) * trust).clamp(0.0, 1.0)
        safe_w = self._project_unit_ball(conflict_free * damp)
        return safe_w, {
            "trust": trust.mean().detach(),
            "behavior_conflict": (parallel < 0).float().mean().detach(),
            "residual_damp": damp.mean().detach(),
        }

    def guidance(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        behavior_velocity: torch.Tensor,
        *,
        bypass_safety: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return (G chunk-shaped, W flat, diagnostics)."""
        b = state.shape[0]
        w = self.raw_w(state, x_t, t)
        if bypass_safety:
            safe = self._project_unit_ball(w)
            diag: dict[str, torch.Tensor] = {}
        else:
            safe, diag = self.behavior_safe_direction(w, behavior_velocity)
        if t.ndim == 1:
            t_col = t.unsqueeze(-1)
        else:
            t_col = t
        g_flat = self.guidance_coef * t_col * safe
        g = g_flat.reshape(b, self.chunk_size, self.action_dim)
        return g, w, diag

    # Residual-API compatibility shim used by online residual paths.
    def guide(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        actor_delta: torch.Tensor | None = None,
        trust: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del actor_delta, trust
        t = torch.ones(state.shape[0], 1, device=state.device, dtype=state.dtype)
        x = reference.detach()
        v = torch.zeros_like(x)
        g, _, _ = self.guidance(state, x, t, v, bypass_safety=True)
        return reference.detach() + g, g


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
    """RLT+CF stack: residual one-shot CF or full flow-time ConsensusFlow."""

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
        cf_mode: str = CF_MODE_RESIDUAL,
        flow_steps: int = DEFAULT_FLOW_STEPS,
        guidance_coef: float = DEFAULT_GUIDANCE_COEF,
        time_dim: int = 64,
    ) -> None:
        super().__init__()
        mode = str(cf_mode).lower().strip()
        if mode not in {CF_MODE_RESIDUAL, CF_MODE_FLOW}:
            raise ValueError(f"cf_mode must be residual|flow, got {cf_mode!r}")
        self.schema_version = SCHEMA_VERSION
        self.cf_mode = mode
        self.is_flow = mode == CF_MODE_FLOW
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
        self.flow_steps = int(flow_steps)
        self.guidance_coef = float(guidance_coef)
        self.time_dim = int(time_dim)
        # "rlt" = FlowVelocityActor; "molmo_ae" = MolmoAct2 Action Expert (V11_1).
        self.v_source = "rlt"

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

        if self.is_flow:
            self.actor = FlowVelocityActor(
                self.state_dim,
                self.action_dim,
                self.chunk_size,
                hidden=hidden,
                time_dim=time_dim,
            )
            self.critic = EnsembleTimeCQL(
                self.state_dim,
                self.flat_action,
                n_critics=n_critics,
                hidden=hidden,
                time_dim=time_dim,
                bounded=self.bounded_critic,
            )
            self.target_critic = EnsembleTimeCQL(
                self.state_dim,
                self.flat_action,
                n_critics=n_critics,
                hidden=hidden,
                time_dim=time_dim,
                bounded=self.bounded_critic,
            )
            self.guide = (
                FlowCFGuide(
                    self.state_dim,
                    self.action_dim,
                    self.chunk_size,
                    hidden=hidden,
                    time_dim=time_dim,
                    guidance_coef=guidance_coef,
                )
                if self.use_cf_guide
                else None
            )
        else:
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
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)

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

    def flow_velocity(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        reference_n: torch.Tensor,
    ) -> torch.Tensor:
        if not self.is_flow:
            raise RuntimeError("flow_velocity requires cf_mode=flow")
        return self.actor(state, x_t, t, reference_n)

    def flow_sample(
        self,
        state: torch.Tensor,
        reference_n: torch.Tensor,
        *,
        apply_guide: bool = False,
        n_steps: int | None = None,
        x0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Euler integrate dx/dt = v_θ(+G) from noise to action chunk."""
        if not self.is_flow:
            raise RuntimeError("flow_sample requires cf_mode=flow")
        s = state.detach()
        ref = reference_n.detach()
        steps = int(n_steps or self.flow_steps)
        b = s.shape[0]
        x = torch.randn_like(ref) if x0 is None else x0
        dt = 1.0 / float(steps)
        guide_norm = s.new_zeros(())
        for i in range(steps):
            t = torch.full((b, 1), i / float(steps), device=s.device, dtype=s.dtype)
            v = self.flow_velocity(s, x, t, ref)
            g = torch.zeros_like(v)
            if apply_guide and self.guide is not None:
                g, _, _ = self.guide.guidance(s, x, t, v)
                guide_norm = guide_norm + g.detach().flatten(1).norm(dim=-1).mean()
            x = x + (v + g) * dt
        info = {
            "actor_mean": x,
            "actor_delta": x - ref,
            "guide_norm": guide_norm / float(steps),
        }
        return x, info

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
        if self.is_flow:
            # reference_present dropout: blend toward zeros so actor must not rely only on ref.
            ref = reference_n
            if reference_present is not None:
                present = reference_present.reshape(-1, 1, 1).to(dtype=ref.dtype)
                ref = ref * present
            return self.flow_sample(s, ref, apply_guide=apply_guide)

        sample, mean = self.actor.sample(
            s, reference_n, reference_present=reference_present, deterministic=deterministic
        )
        info: dict[str, torch.Tensor] = {"actor_mean": mean, "actor_delta": mean - reference_n.detach()}
        # Paper CF_VLA deploy: a = ã + Δ_π (+ Δ_g). Guide is additive on the
        # actor residual, not a replacement of the actor output.
        out = sample
        if apply_guide and self.guide is not None:
            _guided_ref, g_delta = self.guide.guide(
                s, reference_n, actor_delta=info["actor_delta"]
            )
            base = mean if deterministic else sample
            out = base + g_delta
            info["guide_delta"] = g_delta
        return out, info

    def q_chunk(
        self,
        state: torch.Tensor,
        actions_n: torch.Tensor,
        *,
        target: bool = False,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        critic = self.target_critic if target else self.critic
        flat = actions_n.reshape(actions_n.shape[0], -1)
        if self.is_flow:
            if t is None:
                t = torch.ones(state.shape[0], 1, device=state.device, dtype=state.dtype)
            return critic(state, flat, t)
        return critic(state, flat)

    def q_min_chunk(
        self,
        state: torch.Tensor,
        actions_n: torch.Tensor,
        *,
        target: bool = False,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        critic = self.target_critic if target else self.critic
        flat = actions_n.reshape(actions_n.shape[0], -1)
        if self.is_flow:
            if t is None:
                t = torch.ones(state.shape[0], 1, device=state.device, dtype=state.dtype)
            return critic.q_min(state, flat, t)
        return critic.q_min(state, flat)

    def save(self, path: str, meta: dict[str, Any] | None = None) -> None:
        payload = {
            "schema_version": self.schema_version,
            "cf_mode": self.cf_mode,
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
            "flow_steps": self.flow_steps,
            "guidance_coef": self.guidance_coef,
            "time_dim": self.time_dim,
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
            cf_mode=str(payload.get("cf_mode", CF_MODE_RESIDUAL)),
            flow_steps=int(payload.get("flow_steps", DEFAULT_FLOW_STEPS)),
            guidance_coef=float(payload.get("guidance_coef", DEFAULT_GUIDANCE_COEF)),
            time_dim=int(payload.get("time_dim", 64)),
        )
        model.load_state_dict(payload["state_dict"], strict=False)
        return model

    @classmethod
    def from_token_ckpt_as_flow(
        cls,
        path: str,
        map_location: str | torch.device = "cpu",
        *,
        use_cf_guide: bool = True,
        n_critics: int = 10,
        flow_steps: int = DEFAULT_FLOW_STEPS,
        guidance_coef: float = DEFAULT_GUIDANCE_COEF,
    ) -> "MolmoAct2RLTCF":
        """Build a flow-CF model, copying token AE + norm stats from a residual/token ckpt."""
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(
            feature_dim=int(payload.get("feature_dim", FEATURE_DIM)),
            z_dim=int(payload.get("z_dim", Z_DIM)),
            proprio_dim=int(payload.get("proprio_dim", PROPRIO_DIM)),
            action_dim=int(payload.get("action_dim", ACTION_DIM)),
            chunk_size=int(payload.get("chunk_size", CHUNK_SIZE)),
            bounded_critic=bool(payload.get("bounded_critic", True)),
            use_cf_guide=use_cf_guide,
            tune_token_online=False,
            hidden=int(payload.get("hidden", 256)),
            n_critics=int(n_critics),
            token_d_model=int(payload.get("token_d_model", 256)),
            token_layers=int(payload.get("token_layers", 2)),
            token_heads=int(payload.get("token_heads", 4)),
            cf_mode=CF_MODE_FLOW,
            flow_steps=flow_steps,
            guidance_coef=guidance_coef,
            time_dim=64,
        )
        src = payload["state_dict"]
        dst = model.state_dict()
        copied = {
            k: v
            for k, v in src.items()
            if k.startswith(("token_ae.", "target_token_encoder.", "proprio_", "action_", "feature_"))
            and k in dst
            and dst[k].shape == v.shape
        }
        dst.update(copied)
        model.load_state_dict(dst, strict=True)
        model.freeze_token_encoder()
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


def common_scale_normalize(
    grad: torch.Tensor,
    batch_valid: torch.Tensor | None = None,
    floor_coef: float = DEFAULT_CONSENSUS_FLOOR,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Paper z_k = g / (||g|| + c * m_B + eps) for flat grads (B,D)."""
    norms = grad.norm(dim=-1, keepdim=True)
    if batch_valid is None:
        m_b = norms.mean()
    else:
        w = batch_valid.reshape(-1).to(dtype=grad.dtype)
        m_b = (norms.squeeze(-1) * w).sum() / w.sum().clamp_min(1.0)
    return grad / (norms + float(floor_coef) * m_b + eps)


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
