"""PyTorch CF modules: VLA-feature projector, endpoint residual G, ensemble CQL."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 2560  # MolmoAct2-DROID mean-pooled last_hidden
Z_DIM = 256
PROPRIO_DIM = 8
STATE_DIM = Z_DIM + PROPRIO_DIM  # 264 = x = (z, s^p)


def mlp(
    in_dim: int,
    out_dim: int,
    hidden: int = 256,
    n_hidden: int = 2,
    zero_out: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_hidden):
        layers.extend([nn.Linear(d, hidden), nn.ReLU()])
        d = hidden
    out = nn.Linear(d, out_dim)
    if zero_out:
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class FeatureProjector(nn.Module):
    """Map frozen VLA pooled features h (2560) → compact z (256)."""

    def __init__(self, in_dim: int = FEATURE_DIM, out_dim: int = Z_DIM) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class EndpointG(nn.Module):
    """Zero-init residual refiner: delta = scale * tanh(G(x, sg[a_v]))."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        n_hidden: int = 2,
        max_delta: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_delta = float(max_delta)
        self.net = mlp(state_dim + action_dim, action_dim, hidden, n_hidden, zero_out=True)

    def forward(self, state: torch.Tensor, base_actions: torch.Tensor) -> torch.Tensor:
        flat = base_actions.reshape(base_actions.shape[0], -1)
        x = torch.cat([state, flat.detach()], dim=-1)
        raw = self.net(x).reshape_as(base_actions)
        return self.max_delta * torch.tanh(raw)

    def refine(
        self,
        state: torch.Tensor,
        base_actions: torch.Tensor,
        delta_clip: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta = self.forward(state, base_actions)
        if delta_clip is not None:
            delta = torch.clamp(delta, -delta_clip, delta_clip)
        return base_actions + delta, delta


class CriticHead(nn.Module):
    """Success-return head with an intrinsically bounded output."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        bounded: bool = True,
    ) -> None:
        super().__init__()
        self.bounded = bool(bounded)
        self.net = mlp(state_dim + action_dim, 1, hidden, n_hidden=2)

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        flat = actions.reshape(actions.shape[0], -1)
        logits = self.net(torch.cat([state, flat], dim=-1)).squeeze(-1)
        # Sparse discounted success targets are in [0, 1]. Bounding Q prevents
        # the unbounded CQL/actor feedback loop seen in the first 100M run.
        return torch.sigmoid(logits) if self.bounded else logits


class EnsembleCQL(nn.Module):
    """Ensemble success-return critic with local conservative regularization."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_critics: int = 2,
        hidden: int = 256,
        bounded: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.bounded = bool(bounded)
        self.critics = nn.ModuleList(
            [
                CriticHead(state_dim, action_dim, hidden, bounded=self.bounded)
                for _ in range(n_critics)
            ]
        )

    def forward(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([c(state, actions) for c in self.critics], dim=0)

    def q_mean(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.forward(state, actions).mean(dim=0)

    def q_min(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Pessimistic ensemble estimate used by the residual policy."""
        return self.forward(state, actions).min(dim=0).values

    def cql_penalty(
        self,
        state: torch.Tensor,
        data_actions: torch.Tensor,
        n_actions: int = 4,
        coef: float = 1.0,
        action_radius: float = 0.05,
        margin: float = 0.0,
        far_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Penalize optimistic Q on *far* OOD actions, not inside the residual ball.

        Local CQL around ``±action_radius`` flattened the only region G can move
        in (smoke_v2: adv≈2e-4, G inert). Far hinged samples keep Q conservative
        without erasing residual-scale advantages. The hinge is non-negative so
        the old ``Uniform(-1,1) logsumexp - Q`` death spiral cannot return.
        """
        if n_actions < 1:
            zero = data_actions.new_zeros(())
            return zero, {
                "cql_loss": zero.detach(),
                "cql_logmeanexp_q": zero.detach(),
                "cql_data_q": self.q_mean(state, data_actions).mean().detach(),
                "cql_gap": zero.detach(),
            }
        b = state.shape[0]
        # Isotropic noise, then push each sample outside the residual ball so G
        # is free to create local Q structure within ``action_radius``.
        noise = torch.randn(
            n_actions, b, self.action_dim, device=state.device, dtype=state.dtype
        ) * float(far_scale)
        norms = noise.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        min_norm = float(action_radius) + 1e-3
        noise = noise * (min_norm + F.relu(norms - min_norm)) / norms
        far_actions = data_actions.unsqueeze(0) + noise
        state_rep = state.unsqueeze(0).expand(n_actions, -1, -1).reshape(n_actions * b, -1)
        far_flat = far_actions.reshape(n_actions * b, -1)
        far_q = self.q_mean(state_rep, far_flat).reshape(n_actions, b)
        logmeanexp_q = torch.logsumexp(far_q, dim=0) - math.log(n_actions)
        data_q = self.q_mean(state, data_actions)
        gap = logmeanexp_q - data_q + margin
        cql = F.relu(gap).mean()
        return coef * cql, {
            "cql_loss": cql.detach(),
            "cql_logmeanexp_q": logmeanexp_q.mean().detach(),
            "cql_data_q": data_q.mean().detach(),
            "cql_gap": gap.mean().detach(),
        }


class LogAlpha(nn.Module):
    def __init__(self, initial_alpha: float = 1.0) -> None:
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(math.log(initial_alpha), dtype=torch.float32))

    def forward(self, lo: float = -10.0, hi: float = 10.0) -> torch.Tensor:
        return self.log_alpha.clamp(lo, hi).exp()


class MolmoAct2CF(nn.Module):
    """G + critic on x=(z, proprio) where z = Projector(VLA_h)."""

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        z_dim: int = Z_DIM,
        proprio_dim: int = PROPRIO_DIM,
        action_dim: int = 8,
        hidden: int = 256,
        n_critics: int = 2,
        initial_alpha: float = 1.0,
        use_vla_features: bool = True,
        bounded_critic: bool = True,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.z_dim = int(z_dim)
        self.proprio_dim = int(proprio_dim)
        self.use_vla_features = bool(use_vla_features)
        self.bounded_critic = bool(bounded_critic)
        self.state_dim = (self.z_dim + self.proprio_dim) if self.use_vla_features else self.proprio_dim
        self.action_dim = action_dim

        self.projector = (
            FeatureProjector(self.feature_dim, self.z_dim) if self.use_vla_features else None
        )
        self.refiner = EndpointG(
            self.state_dim, action_dim, hidden=hidden, max_delta=0.05
        )
        self.critic = EnsembleCQL(
            self.state_dim,
            action_dim,
            n_critics=n_critics,
            hidden=hidden,
            bounded=self.bounded_critic,
        )
        self.target_critic = EnsembleCQL(
            self.state_dim,
            action_dim,
            n_critics=n_critics,
            hidden=hidden,
            bounded=self.bounded_critic,
        )
        self.target_critic.load_state_dict(self.critic.state_dict())
        for p in self.target_critic.parameters():
            p.requires_grad_(False)
        self.log_alpha = LogAlpha(initial_alpha)

        # Proprio / action norm (raw space). z is LayerNorm'd inside projector.
        self.register_buffer("proprio_mean", torch.zeros(self.proprio_dim))
        self.register_buffer("proprio_std", torch.ones(self.proprio_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        # Optional raw feature whitening (set from buffer).
        self.register_buffer("feature_mean", torch.zeros(self.feature_dim))
        self.register_buffer("feature_std", torch.ones(self.feature_dim))
        # Back-compat aliases used by older call sites.
        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))

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
        # Keep state_* as proprio padded for any legacy code paths.
        if self.use_vla_features:
            sm = torch.zeros(self.state_dim, device=self.proprio_mean.device)
            ss = torch.ones(self.state_dim, device=self.proprio_std.device)
            sm[self.z_dim :] = self.proprio_mean
            ss[self.z_dim :] = self.proprio_std
            self.state_mean.copy_(sm)
            self.state_std.copy_(ss)
        else:
            self.state_mean.copy_(self.proprio_mean)
            self.state_std.copy_(self.proprio_std)

    def normalize_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
        return (proprio - self.proprio_mean) / self.proprio_std

    def normalize_features(self, h: torch.Tensor) -> torch.Tensor:
        return (h - self.feature_mean) / self.feature_std.clamp_min(1e-3)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return (action - self.action_mean) / self.action_std

    def denormalize_action(self, action_n: torch.Tensor) -> torch.Tensor:
        return action_n * self.action_std + self.action_mean

    def encode_state(
        self,
        features: torch.Tensor | None,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """Build critic/G state x = concat(z, s_n). Gradients flow through projector."""
        s_n = self.normalize_proprio(proprio)
        if not self.use_vla_features or self.projector is None:
            return s_n
        if features is None:
            raise ValueError("features required when use_vla_features=True")
        h = self.normalize_features(features)
        z = self.projector(h)
        return torch.cat([z, s_n], dim=-1)

    def refine_raw(
        self,
        proprio_raw: torch.Tensor,
        action_raw: torch.Tensor,
        features: torch.Tensor | None = None,
        delta_clip: float | None = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply G in encoded state space; return raw-space refined actions + delta."""
        x = self.encode_state(features, proprio_raw)
        a_n = self.normalize_action(action_raw)
        refined_n, _delta_n = self.refiner.refine(x, a_n, delta_clip=delta_clip)
        refined_raw = self.denormalize_action(refined_n)
        delta_raw = refined_raw - action_raw
        return refined_raw, delta_raw

    @torch.no_grad()
    def soft_update_target(self, tau: float = 0.005) -> None:
        for p, tp in zip(self.critic.parameters(), self.target_critic.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def save(self, path: str, meta: dict[str, Any] | None = None) -> None:
        payload = {
            "state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "feature_dim": self.feature_dim,
            "z_dim": self.z_dim,
            "proprio_dim": self.proprio_dim,
            "use_vla_features": self.use_vla_features,
            "bounded_critic": self.bounded_critic,
            "meta": meta or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu") -> "MolmoAct2CF":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        use_vla = bool(payload.get("use_vla_features", payload.get("feature_dim", 0) > 0))
        # Old proprio-only checkpoints: state_dim==8, no projector.
        if int(payload.get("state_dim", 8)) == PROPRIO_DIM and "feature_dim" not in payload:
            use_vla = False
        model = cls(
            feature_dim=int(payload.get("feature_dim", FEATURE_DIM)),
            z_dim=int(payload.get("z_dim", Z_DIM)),
            proprio_dim=int(payload.get("proprio_dim", PROPRIO_DIM)),
            action_dim=int(payload.get("action_dim", 8)),
            use_vla_features=use_vla,
            # Checkpoints created before the stability rewrite retain their
            # original raw-Q semantics for reproducibility.
            bounded_critic=bool(payload.get("bounded_critic", False)),
        )
        # Allow loading older ckpts that lack projector keys when use_vla=False.
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        if unexpected:
            # Drop unexpected projector keys when loading mismatched layouts.
            pass
        if missing and use_vla:
            # Fresh projector if missing from an incomplete checkpoint.
            pass
        return model
