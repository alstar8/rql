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
SCHEMA_VERSION = 7
CF_MODE_RESIDUAL = "residual"
CF_MODE_FLOW = "flow"
DEFAULT_FLOW_STEPS = 10
DEFAULT_GUIDANCE_COEF = 0.5
DEFAULT_CONSENSUS_FLOOR = 0.01
DEFAULT_CONFLICT_POWER = 2.0
DEFAULT_RESIDUAL_DAMP = 0.25
DEFAULT_Q_TAIL_FRACTION = 0.25
DEFAULT_Q_TAIL_MIN_HEADS = 2
# CFGRL optimality tokens: uncond / A<0 / A>=0 (or success-conditioned).
CFGRL_O_UNCOND = 0
CFGRL_O_NEG = 1
CFGRL_O_POS = 2
CFGRL_O_CLASSES = 3
DEFAULT_CFGRL_O_DIM = 16
DEFAULT_CFGRL_W = 1.0
DEFAULT_CFGRL_DROPOUT = 0.1
DEFAULT_N_HIDDEN_ACTOR = 3
DEFAULT_N_HIDDEN_CRITIC = 2
DEFAULT_Z_EXPAND_DIM = 0


def lower_tail_mean(
    values: torch.Tensor,
    *,
    fraction: float = DEFAULT_Q_TAIL_FRACTION,
    min_heads: int = DEFAULT_Q_TAIL_MIN_HEADS,
    dim: int = 0,
) -> torch.Tensor:
    """Mean of the pessimistic lower ensemble tail.

    For the default ten-head critic this averages the lowest three heads.  With
    a two-head test critic both heads participate, avoiding a disguised hard
    minimum.
    """
    if values.ndim == 0:
        raise ValueError("lower_tail_mean requires an ensemble dimension")
    heads = int(values.shape[dim])
    if heads < 1:
        raise ValueError("lower_tail_mean received an empty ensemble")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if int(min_heads) < 1:
        raise ValueError(f"min_heads must be positive, got {min_heads}")
    selected = min(
        heads,
        max(int(min_heads), int(math.ceil(float(fraction) * heads))),
    )
    tail = torch.topk(values, k=selected, dim=dim, largest=False).values
    return tail.mean(dim=dim)


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
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 4,
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
        d_model: int = 512,
        n_heads: int = 4,
        n_layers: int = 4,
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
        n_hidden: int = DEFAULT_N_HIDDEN_CRITIC,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.bounded = bool(bounded)
        self.time_dim = int(time_dim)
        self.net = mlp(
            state_dim + action_dim + time_dim,
            1,
            hidden,
            n_hidden=int(n_hidden),
            layernorm=bool(layernorm),
        )

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
        n_hidden: int = DEFAULT_N_HIDDEN_CRITIC,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.bounded = bool(bounded)
        self.critics = nn.ModuleList(
            [
                TimeCriticHead(
                    state_dim,
                    action_dim,
                    hidden,
                    time_dim,
                    bounded=bounded,
                    n_hidden=n_hidden,
                    layernorm=layernorm,
                )
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


class RLTZExpander(nn.Module):
    """Keep pretrained z and append a learned extra embedding.

    Online replay stores frozen-token z (256-d). The expander concatenates z
    with a zero-init tail so actor/critic still see the pretrained code at
    step 0, then grow capacity as the extra MLP trains.
    """

    def __init__(self, z_dim: int, out_dim: int) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.out_dim = int(out_dim)
        if self.out_dim < self.z_dim:
            raise ValueError(
                f"z_expand_dim ({self.out_dim}) must be >= z_dim ({self.z_dim})"
            )
        extra = self.out_dim - self.z_dim
        if extra == 0:
            self.extra: nn.Module | None = None
            return
        self.extra = nn.Sequential(
            nn.Linear(self.z_dim, extra),
            nn.LayerNorm(extra),
            nn.GELU(),
            nn.Linear(extra, extra),
            nn.LayerNorm(extra),
        )
        last_linear = self.extra[3]
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.extra is None:
            return z
        return torch.cat([z, self.extra(z)], dim=-1)


class FlowVelocityActor(nn.Module):
    """VLA-conditioned flow velocity: v_θ(s, x_t, t, a_VLA[, o])."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        chunk_size: int = CHUNK_SIZE,
        hidden: int = 256,
        time_dim: int = 64,
        o_dim: int = 0,
        n_hidden: int = DEFAULT_N_HIDDEN_ACTOR,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.flat_action = self.action_dim * self.chunk_size
        self.time_dim = int(time_dim)
        self.o_dim = int(o_dim)
        self.o_embed = (
            nn.Embedding(CFGRL_O_CLASSES, self.o_dim) if self.o_dim > 0 else None
        )
        # state + x_t + a_ref + t (+ optional optimality embedding)
        self.net = mlp(
            state_dim + 2 * self.flat_action + time_dim + self.o_dim,
            self.flat_action,
            hidden,
            n_hidden=int(n_hidden),
            zero_out=False,
            layernorm=bool(layernorm),
        )

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        reference: torch.Tensor,
        o: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = state.shape[0]
        x_flat = x_t.reshape(b, -1)
        ref_flat = reference.reshape(b, -1).detach()
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        temb = sinusoidal_time_embed(t.squeeze(-1), self.time_dim)
        pieces = [state, x_flat, ref_flat, temb]
        if self.o_embed is not None:
            if o is None:
                o = torch.full(
                    (b,),
                    CFGRL_O_UNCOND,
                    device=state.device,
                    dtype=torch.long,
                )
            else:
                o = o.reshape(b).to(dtype=torch.long)
            pieces.append(self.o_embed(o))
        v = self.net(torch.cat(pieces, dim=-1))
        return v.reshape(b, self.chunk_size, self.action_dim)

    def copy_pretrained_into_o_actor(self, source: "FlowVelocityActor") -> bool:
        """Keep a pretrained flow net; zero-init extra o columns so v(·,o) starts as v_pre.

        Returns True when the first-layer copy actually ran; False when the
        source architecture is incompatible (caller must decide whether that
        is acceptable).
        """
        if self.o_embed is not None:
            nn.init.zeros_(self.o_embed.weight)
        src_first = source.net[0]
        dst_first = self.net[0]
        if not isinstance(src_first, nn.Linear) or not isinstance(dst_first, nn.Linear):
            return False
        old_in = int(src_first.weight.shape[1])
        new_in = int(dst_first.weight.shape[1])
        if (
            src_first.weight.shape[0] != dst_first.weight.shape[0]
            or new_in < old_in
        ):
            return False
        with torch.no_grad():
            dst_first.weight[:, :old_in].copy_(src_first.weight)
            if new_in > old_in:
                dst_first.weight[:, old_in:].zero_()
            dst_first.bias.copy_(src_first.bias)
            src_sd = source.net.state_dict()
            dst_sd = self.net.state_dict()
            for key, value in src_sd.items():
                if key.startswith("0."):
                    continue
                if key in dst_sd and dst_sd[key].shape == value.shape:
                    dst_sd[key].copy_(value)
            self.net.load_state_dict(dst_sd)
        return True


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
    """Distill common-scale-normalized critic gradients into a deploy-safe guide."""

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

    def raw_w(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Unbounded distillation output W; deployment bounds are applied separately."""
        b = state.shape[0]
        flat = reference.reshape(b, -1).detach()
        return self.net(torch.cat([state, flat], dim=-1))

    def forward(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        b = state.shape[0]
        raw = self.raw_w(state, reference)
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
        token_d_model: int = 512,
        token_layers: int = 4,
        token_heads: int = 4,
        cf_mode: str = CF_MODE_RESIDUAL,
        flow_steps: int = DEFAULT_FLOW_STEPS,
        guidance_coef: float = DEFAULT_GUIDANCE_COEF,
        time_dim: int = 64,
        q_tail_fraction: float = DEFAULT_Q_TAIL_FRACTION,
        q_tail_min_heads: int = DEFAULT_Q_TAIL_MIN_HEADS,
        use_cfgrl: bool = False,
        cfgrl_o_dim: int = DEFAULT_CFGRL_O_DIM,
        cfgrl_w: float = DEFAULT_CFGRL_W,
        n_hidden_actor: int = DEFAULT_N_HIDDEN_ACTOR,
        n_hidden_critic: int = DEFAULT_N_HIDDEN_CRITIC,
        z_expand_dim: int = DEFAULT_Z_EXPAND_DIM,
        layernorm_heads: bool = False,
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
        self.q_tail_fraction = float(q_tail_fraction)
        self.q_tail_min_heads = int(q_tail_min_heads)
        self.use_cfgrl = bool(use_cfgrl)
        self.cfgrl_o_dim = int(cfgrl_o_dim) if self.use_cfgrl else 0
        self.cfgrl_w = float(cfgrl_w)
        self.n_hidden_actor = int(n_hidden_actor)
        self.n_hidden_critic = int(n_hidden_critic)
        self.z_expand_dim = int(z_expand_dim) if int(z_expand_dim) > 0 else 0
        self.layernorm_heads = bool(layernorm_heads)
        if self.n_hidden_actor < 1 or self.n_hidden_critic < 1:
            raise ValueError("n_hidden_actor and n_hidden_critic must be >= 1")
        if not 0.0 < self.q_tail_fraction <= 1.0:
            raise ValueError("q_tail_fraction must be in (0, 1]")
        if self.q_tail_min_heads < 1:
            raise ValueError("q_tail_min_heads must be positive")
        self.loaded_meta: dict[str, Any] = {}
        # "rlt" = FlowVelocityActor; "molmo_ae" = MolmoAct2 Action Expert (V11_1).
        self.v_source = "rlt"
        if self.z_expand_dim > 0:
            self.z_expand: nn.Module | None = RLTZExpander(self.z_dim, self.z_expand_dim)
            self.state_dim = self.z_expand_dim + self.proprio_dim
        else:
            self.z_expand = None

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
                o_dim=self.cfgrl_o_dim,
                n_hidden=self.n_hidden_actor,
                layernorm=self.layernorm_heads,
            )
            self.critic = EnsembleTimeCQL(
                self.state_dim,
                self.flat_action,
                n_critics=n_critics,
                hidden=hidden,
                time_dim=time_dim,
                bounded=self.bounded_critic,
                n_hidden=self.n_hidden_critic,
                layernorm=self.layernorm_heads,
            )
            self.target_critic = EnsembleTimeCQL(
                self.state_dim,
                self.flat_action,
                n_critics=n_critics,
                hidden=hidden,
                time_dim=time_dim,
                bounded=self.bounded_critic,
                n_hidden=self.n_hidden_critic,
                layernorm=self.layernorm_heads,
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
        if self.z_expand is not None:
            z = self.z_expand(z)
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

    def rlt_adapter_parameters(self) -> list[nn.Parameter]:
        """Trainable z expander; kept unfrozen when the token AE is frozen."""

        if self.z_expand is None:
            return []
        return list(self.z_expand.parameters())

    def freeze_token_encoder(self) -> None:
        for p in self.token_ae.parameters():
            p.requires_grad_(False)
        self.tune_token_online = False
        for p in self.rlt_adapter_parameters():
            p.requires_grad_(True)

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
        o: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_flow:
            raise RuntimeError("flow_velocity requires cf_mode=flow")
        return self.actor(state, x_t, t, reference_n, o=o)

    def flow_sample(
        self,
        state: torch.Tensor,
        reference_n: torch.Tensor,
        *,
        apply_guide: bool = False,
        n_steps: int | None = None,
        x0: torch.Tensor | None = None,
        flow_noise_seed: int | None = None,
        cfg_w: float | None = None,
        o_cond: int = CFGRL_O_POS,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Euler integrate dx/dt = v_θ(+G) from noise to action chunk.

        When ``use_cfgrl`` is set, compose uncond and o=POS velocities:
        v = (1-w) v(∅) + w v(o=POS). Guide is not used on the CFGRL arm.
        """
        if not self.is_flow:
            raise RuntimeError("flow_sample requires cf_mode=flow")
        s = state.detach()
        ref = reference_n.detach()
        steps = int(n_steps or self.flow_steps)
        b = s.shape[0]
        if x0 is not None and flow_noise_seed is not None:
            raise ValueError("pass either x0 or flow_noise_seed, not both")
        if x0 is None and flow_noise_seed is not None:
            generator = torch.Generator(device=ref.device)
            generator.manual_seed(int(flow_noise_seed) & ((1 << 63) - 1))
            x = torch.randn(
                ref.shape,
                dtype=ref.dtype,
                device=ref.device,
                generator=generator,
            )
        else:
            x = torch.randn_like(ref) if x0 is None else x0
        dt = 1.0 / float(steps)
        guide_norm = s.new_zeros(())
        weight = float(self.cfgrl_w if cfg_w is None else cfg_w)
        o_uncond = torch.full((b,), CFGRL_O_UNCOND, device=s.device, dtype=torch.long)
        o_pos = torch.full((b,), int(o_cond), device=s.device, dtype=torch.long)
        for i in range(steps):
            t = torch.full((b, 1), i / float(steps), device=s.device, dtype=s.dtype)
            if self.use_cfgrl:
                v_u = self.flow_velocity(s, x, t, ref, o=o_uncond)
                v_c = self.flow_velocity(s, x, t, ref, o=o_pos)
                v = (1.0 - weight) * v_u + weight * v_c
            else:
                v = self.flow_velocity(s, x, t, ref)
            g = torch.zeros_like(v)
            if apply_guide and self.guide is not None and not self.use_cfgrl:
                g, _, _ = self.guide.guidance(s, x, t, v)
                guide_norm = guide_norm + g.detach().flatten(1).norm(dim=-1).mean()
            x = x + (v + g) * dt
        info = {
            "actor_mean": x,
            "actor_delta": x - ref,
            "guide_norm": guide_norm / float(steps),
            "cfgrl_w": s.new_tensor(weight),
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
        flow_noise_seed: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Actor/guide see stop-grad state so token updates come only from recon/critic.
        s = state.detach()
        if self.is_flow:
            # reference_present dropout: blend toward zeros so actor must not rely only on ref.
            ref = reference_n
            if reference_present is not None:
                present = reference_present.reshape(-1, 1, 1).to(dtype=ref.dtype)
                ref = ref * present
            return self.flow_sample(
                s,
                ref,
                apply_guide=apply_guide,
                flow_noise_seed=flow_noise_seed,
            )

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

    def q_lower_tail_chunk(
        self,
        state: torch.Tensor,
        actions_n: torch.Tensor,
        *,
        target: bool = False,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deployment-relevant robust pessimistic value aggregate."""
        values = self.q_chunk(
            state,
            actions_n,
            target=target,
            t=t,
        )
        return lower_tail_mean(
            values,
            fraction=self.q_tail_fraction,
            min_heads=self.q_tail_min_heads,
            dim=0,
        )

    def reinitialize_critic_heads(
        self,
        head_indices: list[int],
        *,
        seed: int,
    ) -> list[nn.Parameter]:
        """Explicitly reset selected online/target heads for controlled recovery."""
        online_heads = getattr(self.critic, "critics", None)
        target_heads = getattr(self.target_critic, "critics", None)
        if online_heads is None or target_heads is None:
            raise RuntimeError("Critic ensemble does not expose head modules")
        unique = sorted(set(int(index) for index in head_indices))
        if any(index < 0 or index >= len(online_heads) for index in unique):
            raise IndexError(
                f"Invalid critic head indices {unique} for {len(online_heads)} heads"
            )
        devices: list[int] = []
        parameter = next(self.critic.parameters())
        if parameter.is_cuda:
            devices = [parameter.device.index or 0]
        reset_parameters: list[nn.Parameter] = []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            if parameter.is_cuda:
                torch.cuda.manual_seed_all(int(seed))
            for index in unique:
                for module in online_heads[index].modules():
                    reset = getattr(module, "reset_parameters", None)
                    if callable(reset):
                        reset()
                target_heads[index].load_state_dict(
                    online_heads[index].state_dict(),
                    strict=True,
                )
                reset_parameters.extend(online_heads[index].parameters())
        return reset_parameters

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
            "q_tail_fraction": self.q_tail_fraction,
            "q_tail_min_heads": self.q_tail_min_heads,
            "use_cfgrl": self.use_cfgrl,
            "cfgrl_o_dim": self.cfgrl_o_dim,
            "cfgrl_w": self.cfgrl_w,
            "n_hidden_actor": self.n_hidden_actor,
            "n_hidden_critic": self.n_hidden_critic,
            "z_expand_dim": self.z_expand_dim,
            "layernorm_heads": self.layernorm_heads,
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
            q_tail_fraction=float(
                payload.get("q_tail_fraction", DEFAULT_Q_TAIL_FRACTION)
            ),
            q_tail_min_heads=int(
                payload.get("q_tail_min_heads", DEFAULT_Q_TAIL_MIN_HEADS)
            ),
            use_cfgrl=bool(payload.get("use_cfgrl", False)),
            cfgrl_o_dim=int(payload.get("cfgrl_o_dim", DEFAULT_CFGRL_O_DIM)),
            cfgrl_w=float(payload.get("cfgrl_w", DEFAULT_CFGRL_W)),
            n_hidden_actor=int(payload.get("n_hidden_actor", DEFAULT_N_HIDDEN_ACTOR)),
            n_hidden_critic=int(payload.get("n_hidden_critic", DEFAULT_N_HIDDEN_CRITIC)),
            z_expand_dim=int(payload.get("z_expand_dim", DEFAULT_Z_EXPAND_DIM)),
            layernorm_heads=bool(payload.get("layernorm_heads", False)),
        )
        src = payload["state_dict"]
        dst = model.state_dict()
        compatible = {
            key: value
            for key, value in src.items()
            if key in dst and dst[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        model.loaded_meta = dict(payload.get("meta") or {})
        return model

    def as_cfgrl(
        self,
        *,
        cfgrl_w: float = DEFAULT_CFGRL_W,
        o_dim: int = DEFAULT_CFGRL_O_DIM,
        hidden: int | None = None,
        n_hidden_actor: int | None = None,
        n_hidden_critic: int | None = None,
        z_expand_dim: int | None = None,
        layernorm_heads: bool | None = None,
        require_transfer: bool = True,
    ) -> "MolmoAct2RLTCF":
        """Upgrade a flow model to CFGRL (fresh o-conditioned actor, keep token AE).

        With require_transfer=True (default), any pretrained weight that fails
        to carry over because of an architecture mismatch raises instead of
        being silently re-initialized. The only legitimate shape changes are
        the actor's first layer (grows o_dim input columns, copied explicitly)
        and the o_embed table (new, zero-initialized).
        """
        if not self.is_flow:
            raise RuntimeError("CFGRL requires cf_mode=flow")
        want_hidden = int(self.hidden if hidden is None else hidden)
        want_n_act = int(self.n_hidden_actor if n_hidden_actor is None else n_hidden_actor)
        want_n_q = int(self.n_hidden_critic if n_hidden_critic is None else n_hidden_critic)
        want_z = int(self.z_expand_dim if z_expand_dim is None else z_expand_dim)
        want_ln = bool(self.layernorm_heads if layernorm_heads is None else layernorm_heads)
        already = (
            self.use_cfgrl
            and int(self.cfgrl_o_dim) == int(o_dim)
            and want_hidden == self.hidden
            and want_n_act == self.n_hidden_actor
            and want_n_q == self.n_hidden_critic
            and want_z == self.z_expand_dim
            and want_ln == self.layernorm_heads
        )
        if already:
            self.cfgrl_w = float(cfgrl_w)
            self.guide = None
            self.use_cf_guide = False
            return self
        upgraded = self.__class__(
            feature_dim=self.feature_dim,
            z_dim=self.z_dim,
            proprio_dim=self.proprio_dim,
            action_dim=self.action_dim,
            chunk_size=self.chunk_size,
            bounded_critic=self.bounded_critic,
            use_cf_guide=False,
            tune_token_online=False,
            hidden=want_hidden,
            n_critics=self.n_critics,
            residual_actor=self.residual_actor,
            max_delta=self.max_delta,
            token_d_model=self.token_d_model,
            token_layers=self.token_layers,
            token_heads=self.token_heads,
            cf_mode=CF_MODE_FLOW,
            flow_steps=self.flow_steps,
            guidance_coef=self.guidance_coef,
            time_dim=self.time_dim,
            q_tail_fraction=self.q_tail_fraction,
            q_tail_min_heads=self.q_tail_min_heads,
            use_cfgrl=True,
            cfgrl_o_dim=int(o_dim),
            cfgrl_w=float(cfgrl_w),
            n_hidden_actor=want_n_act,
            n_hidden_critic=want_n_q,
            z_expand_dim=want_z,
            layernorm_heads=want_ln,
        )
        src = self.state_dict()
        dst = upgraded.state_dict()
        compatible = {}
        mismatched: list[str] = []
        for key, value in src.items():
            if key not in dst:
                continue
            if dst[key].shape == value.shape:
                compatible[key] = value
            else:
                mismatched.append(key)
        upgraded.load_state_dict(compatible, strict=False)
        # actor.net.0.weight legitimately grows by o_dim columns (handled by
        # copy_pretrained_into_o_actor below); actor.o_embed.weight is new.
        # Anything else failing to transfer means the requested architecture
        # does not match the checkpoint and the pretrained weights would be
        # silently re-initialized.
        _ALLOWED_RESIZE = {"actor.net.0.weight", "actor.o_embed.weight"}
        unexpected = [key for key in mismatched if key not in _ALLOWED_RESIZE]
        if require_transfer and unexpected:
            raise RuntimeError(
                "as_cfgrl: pretrained weights do not fit the requested "
                f"architecture; {len(unexpected)} tensors would be silently "
                f"re-initialized (e.g. {unexpected[:5]}). Build the pretrain "
                "at the target architecture instead, or pass "
                "require_transfer=False if a fresh start is intended."
            )
        arch_matches = (
            want_hidden == self.hidden
            and want_n_act == self.n_hidden_actor
            and want_z == self.z_expand_dim
        )
        if arch_matches:
            transferred = upgraded.actor.copy_pretrained_into_o_actor(self.actor)
            if require_transfer and not transferred:
                raise RuntimeError(
                    "as_cfgrl: pretrained actor weights did not transfer into "
                    "the o-conditioned actor despite matching architecture."
                )
        elif require_transfer:
            raise RuntimeError(
                "as_cfgrl: requested architecture "
                f"(hidden={want_hidden}, n_hidden_actor={want_n_act}, "
                f"z_expand_dim={want_z}) differs from the checkpoint "
                f"(hidden={self.hidden}, n_hidden_actor={self.n_hidden_actor}, "
                f"z_expand_dim={self.z_expand_dim}); refusing to silently "
                "re-initialize actor/critic. Pass require_transfer=False if a "
                "fresh start is intended."
            )
        upgraded.loaded_meta = dict(self.loaded_meta)
        upgraded.freeze_token_encoder()
        upgraded.guide = None
        return upgraded

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
        q_tail_fraction: float = DEFAULT_Q_TAIL_FRACTION,
        q_tail_min_heads: int = DEFAULT_Q_TAIL_MIN_HEADS,
        hidden: int | None = None,
        n_hidden_actor: int | None = None,
        n_hidden_critic: int | None = None,
        z_expand_dim: int | None = None,
        layernorm_heads: bool | None = None,
        use_cfgrl: bool | None = None,
        cfgrl_o_dim: int | None = None,
        cfgrl_w: float = DEFAULT_CFGRL_W,
    ) -> "MolmoAct2RLTCF":
        """Build a flow-CF model, copying token AE + norm stats from a residual/token ckpt.

        Head architecture and CFGRL settings default to the values stored in
        the checkpoint payload (falling back to module defaults), and can be
        overridden explicitly. This lets the flow pretrain build a born-CFGRL
        model at the exact architecture the online run will use, so no
        weight-transfer guessing is needed downstream.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)

        def _pick(override: Any, key: str, default: Any) -> Any:
            if override is not None:
                return override
            return payload.get(key, default)

        model = cls(
            feature_dim=int(payload.get("feature_dim", FEATURE_DIM)),
            z_dim=int(payload.get("z_dim", Z_DIM)),
            proprio_dim=int(payload.get("proprio_dim", PROPRIO_DIM)),
            action_dim=int(payload.get("action_dim", ACTION_DIM)),
            chunk_size=int(payload.get("chunk_size", CHUNK_SIZE)),
            bounded_critic=bool(payload.get("bounded_critic", True)),
            use_cf_guide=use_cf_guide,
            tune_token_online=False,
            hidden=int(_pick(hidden, "hidden", 256)),
            n_critics=int(n_critics),
            token_d_model=int(payload.get("token_d_model", 256)),
            token_layers=int(payload.get("token_layers", 2)),
            token_heads=int(payload.get("token_heads", 4)),
            cf_mode=CF_MODE_FLOW,
            flow_steps=flow_steps,
            guidance_coef=guidance_coef,
            time_dim=64,
            q_tail_fraction=q_tail_fraction,
            q_tail_min_heads=q_tail_min_heads,
            n_hidden_actor=int(_pick(n_hidden_actor, "n_hidden_actor", DEFAULT_N_HIDDEN_ACTOR)),
            n_hidden_critic=int(_pick(n_hidden_critic, "n_hidden_critic", DEFAULT_N_HIDDEN_CRITIC)),
            z_expand_dim=int(_pick(z_expand_dim, "z_expand_dim", DEFAULT_Z_EXPAND_DIM)),
            layernorm_heads=bool(_pick(layernorm_heads, "layernorm_heads", False)),
            use_cfgrl=bool(_pick(use_cfgrl, "use_cfgrl", False)),
            cfgrl_o_dim=int(_pick(cfgrl_o_dim, "cfgrl_o_dim", DEFAULT_CFGRL_O_DIM)),
            cfgrl_w=float(payload.get("cfgrl_w", cfgrl_w)),
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
        if model.use_cfgrl and model.actor.o_embed is not None:
            # v(·,o) starts identical across o; the o-path is neutral at birth.
            nn.init.zeros_(model.actor.o_embed.weight)
        model.freeze_token_encoder()
        return model


def normalized_grad_target(
    grads: list[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Deterministic ensemble mean of common-scale targets (diagnostics only).

    Online guide training samples one critic per example. This helper retains
    the full common-scale magnitude for probes that need an explicit ensemble
    mean; it must never re-normalize the resultant.
    """
    stacked = torch.stack(grads, dim=0)  # (E,B,C,A)
    ensemble, batch = stacked.shape[:2]
    flat = stacked.reshape(ensemble * batch, -1)
    normalized = common_scale_normalize(flat, eps=eps)
    return normalized.reshape_as(stacked).mean(dim=0)


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
