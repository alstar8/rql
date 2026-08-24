"""Actor and critic for the online stage (paper Sec. III-B, appendix).

Both are plain MLPs over the RL state x = (z_rl, s^p). The actor additionally
sees the VLA's reference chunk and refines it; the critic never does -- it
scores the action that was actually executed.

    pi(a | x, a_ref) = N(mu(x, a_ref), sigma^2 I)      Eq. 4
    Q(x, a) -> R, two heads, min for targets           TD3
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int, layer_norm: bool = False) -> nn.Sequential:
    """n_layers hidden layers of `hidden_dim`, ReLU, linear head.

    `layer_norm` is off by default: the paper says two-layer MLP and nothing
    more. It exists because normalising the critic trunk is the standard way to
    keep a high update-to-data ratio from diverging.
    """
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(dim, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.ReLU())
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Refines a reference chunk. Gaussian with a fixed, non-learned std."""

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        sigma: float = 0.02,
    ) -> None:
        super().__init__()
        self.chunk_dim = chunk_dim
        self.sigma = sigma
        self.net = mlp(state_dim + chunk_dim, hidden_dim, chunk_dim, n_layers)

    def forward(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Deterministic mean mu(x, a_ref). Callers zero `reference` to drop it."""
        return self.net(torch.cat([state, reference], dim=-1))

    def sample(self, state: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """One draw from the Gaussian. Reparameterized, so actor gradients flow."""
        mean = self(state, reference)
        return mean + self.sigma * torch.randn_like(mean)


class FlowActor(nn.Module):
    """ConsensusFlow behavior velocity: v(s, x_t, t) -> R^{chunk_dim}.

    The paper's actor MLP has no LayerNorm. t is concatenated as one extra feature.
    With ref_dim > 0 the VLA reference chunk is concatenated too, turning the flow
    into a corrector: at BC convergence it copies the reference, and the RL term
    bends the endpoint away from it (the RL-Token actor's conditioning, Eq. 5).
    """

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
        ref_dim: int = 0,
    ) -> None:
        super().__init__()
        self.ref_dim = ref_dim
        self.net = mlp(state_dim + chunk_dim + ref_dim + 1, hidden_dim, chunk_dim, n_layers, layer_norm=False)

    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        parts = [state, x_t]
        if self.ref_dim:
            parts.append(reference)
        parts.append(t)
        return self.net(torch.cat(parts, dim=-1))


class EnsembleCritic(nn.Module):
    """K scalar Q heads over (s, x, t). Distillation samples one member per row."""

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
        ensemble: int = 10,
        layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.ensemble = ensemble
        self.heads = nn.ModuleList(
            [mlp(state_dim + chunk_dim + 1, hidden_dim, 1, n_layers, layer_norm) for _ in range(ensemble)]
        )

    def _pack(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return torch.cat([state, action, t], dim=-1)

    def forward(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """(K, B) values."""
        x = self._pack(state, action, t)
        return torch.stack([head(x).squeeze(-1) for head in self.heads], dim=0)

    def mean_std(self, state: torch.Tensor, action: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        qs = self(state, action, t)
        return qs.mean(dim=0), qs.std(dim=0, unbiased=False)


class Guidance(nn.Module):
    """W_phi(s, x, t) in action space. LayerNorm on, matching the paper student."""

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.net = mlp(state_dim + chunk_dim + 1, hidden_dim, chunk_dim, n_layers, layer_norm=True)

    def forward(self, state: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.net(torch.cat([state, x_t, t], dim=-1))


class DoubleCritic(nn.Module):
    """Two independent Q heads over (x, a). Targets use the minimum (TD3)."""

    def __init__(
        self,
        state_dim: int,
        chunk_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 2,
        layer_norm: bool = False,
    ) -> None:
        super().__init__()
        self.q1 = mlp(state_dim + chunk_dim, hidden_dim, 1, n_layers, layer_norm)
        self.q2 = mlp(state_dim + chunk_dim, hidden_dim, 1, n_layers, layer_norm)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """The head the actor maximizes, following TD3."""
        return self.q1(torch.cat([state, action], dim=-1)).squeeze(-1)

    def min_q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self(state, action)
        return torch.min(q1, q2)
