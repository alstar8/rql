"""Advantage-Weighted Regression on the RL-Token Gaussian actor (Peng et al. 2019).

V21 maximises Q by backpropagating through a reparameterized sample (Eq. 5).
AWR never does that. The critic (twin Q + state-value V) is stop-grad for the
actor; the actor is weighted regression onto the *replay* action:

    A(s, a) = min_i Q_i(s, a) - V(s)
    w       = exp(A / τ) clipped and mean-normalised
    L_π     = E[ -w * log π(a | s, ã) ]

π is the same one-pass Gaussian as V21, conditioned on the frozen-VLA reference.
The temperature τ is the KL-to-replay dual (AWR's β); it is *not* V21's Euclidean
pull. That pull is off: replay already contains the specialist chunks, and
weighting them by advantage is the constraint.

Q uses V21's clipped TD (bootstrap through the online actor, target in [0, 1]).
V uses the same backup but from V(s'), so it is a state baseline, not a second Q.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .config import OnlineConfig
from .networks import Actor, DoubleCritic, Value
from .shared import atomic_torch_save


def advantage_weights(adv: torch.Tensor, temp: float, clip: float) -> torch.Tensor:
    """Stop-grad AWR weights: exp(A/τ), clipped, then mean-normalised to 1."""
    weights = torch.exp(adv / temp).clamp(max=clip)
    return weights / weights.mean().clamp_min(1e-6)


class AWRAgent:
    """Gaussian RL-Token actor trained by AWR; twin Q + V trained by clipped TD."""

    def __init__(self, cfg: OnlineConfig, state_dim: int, chunk_dim: int, chunk: int) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.chunk_dim = chunk_dim
        self.chunk_discount = cfg.gamma**chunk
        self.awr_temp = float(getattr(cfg, "awr_temp", 1.0))
        self.awr_clip = float(getattr(cfg, "awr_clip", 20.0))
        if self.awr_temp <= 0:
            raise ValueError(f"awr_temp must be > 0, got {self.awr_temp}")
        if self.awr_clip <= 0:
            raise ValueError(f"awr_clip must be > 0, got {self.awr_clip}")

        self.actor = Actor(state_dim, chunk_dim, cfg.hidden_dim, cfg.n_layers, cfg.sigma).to(self.device)
        self.critic = DoubleCritic(
            state_dim, chunk_dim, cfg.hidden_dim, cfg.n_layers, cfg.critic_layer_norm
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.value = Value(state_dim, cfg.hidden_dim, cfg.n_layers, cfg.critic_layer_norm).to(self.device)
        self.value_target = copy.deepcopy(self.value).requires_grad_(False)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False) if cfg.target_actor else None
        self.q_bounds = (0.0, 1.0)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic.parameters()) + list(self.value.parameters()), lr=cfg.lr
        )
        self.critic_steps = 0
        self.actor_steps = 0

    # --- acting -------------------------------------------------------------

    @torch.no_grad()
    def act(self, state: np.ndarray, reference: np.ndarray, explore: bool = False) -> np.ndarray:
        """One action chunk, flat (C * action_dim,). Same Gaussian as V21."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        r = torch.as_tensor(reference, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor.sample(s, r) if explore else self.actor(s, r)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def q_values(self, state: np.ndarray, action: np.ndarray) -> float:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic.min_q(s, a).item())

    # --- learning -----------------------------------------------------------

    def _to_torch(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in batch.items():
            if key in ("tokens", "mask", "next_tokens", "next_mask"):
                out[key] = value
                continue
            out[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return out

    def critic_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        b = self._to_torch(batch)

        with torch.no_grad():
            actor = self.actor_target if self.actor_target is not None else self.actor
            next_action = actor.sample(b["next_state"], b["next_reference"])
            if self.cfg.target_smoothing > 0:
                next_action = next_action + self.cfg.target_smoothing * torch.randn_like(next_action)
            next_q = self.critic_target.min_q(b["next_state"], next_action)
            q_target = b["reward"] + (1.0 - b["done"]) * self.chunk_discount * next_q
            v_target = b["reward"] + (1.0 - b["done"]) * self.chunk_discount * self.value_target(b["next_state"])
            if self.cfg.clip_target:
                q_target = q_target.clamp(*self.q_bounds)
                v_target = v_target.clamp(*self.q_bounds)

        q1, q2 = self.critic(b["state"], b["action"])
        v = self.value(b["state"])
        loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target) + F.mse_loss(v, v_target)

        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self._soft_update()
        self.critic_steps += 1

        return {
            "critic_loss": loss.item(),
            "q_mean": q1.mean().item(),
            "v_mean": v.mean().item(),
            "target_mean": q_target.mean().item(),
        }

    def actor_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Weighted NLL onto the replay action. Q and V are stop-grad."""
        b = self._to_torch(batch)

        with torch.no_grad():
            q = self.critic.min_q(b["state"], b["action"])
            v = self.value(b["state"])
            adv = q - v
            weights = advantage_weights(adv, self.awr_temp, self.awr_clip)

        keep = (torch.rand(b["reference"].shape[0], 1, device=self.device) >= self.cfg.reference_dropout).float()
        logp = self.actor.log_prob(b["state"], b["reference"] * keep, b["action"])
        loss = -(weights * logp).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        self.actor_steps += 1
        if self.actor_target is not None:
            self._polyak(self.actor, self.actor_target)

        with torch.no_grad():
            mean = self.actor(b["state"], b["reference"])
            mse = (mean - b["action"]).pow(2).mean()
            ref_mse = (mean - b["reference"]).pow(2).mean()

        return {
            "actor_loss": loss.item(),
            "actor_bc_loss": loss.item(),
            "actor_bc_rmse": float(mse.sqrt().item()),
            "actor_ref_rmse": float(ref_mse.sqrt().item()),
            "awr_w_mean": float(weights.mean().item()),
            "awr_w_max": float(weights.max().item()),
            "awr_adv_mean": float(adv.mean().item()),
            "actor_q": float(q.mean().item()),
        }

    def actor_bc_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """AC pretrain uses this name. It is the AWR update, not unweighted MSE."""
        return self.actor_step(batch)

    def _soft_update(self) -> None:
        self._polyak(self.critic, self.critic_target)
        self._polyak(self.value, self.value_target)

    def _polyak(self, net, target) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(tau * p)

    # --- diagnostics --------------------------------------------------------

    @torch.no_grad()
    def probe(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        b = self._to_torch(batch)
        mean = self.actor(b["state"], b["reference"])
        q_actor = self.critic.min_q(b["state"], mean)
        q_ref = self.critic.min_q(b["state"], b["reference"])
        v = self.value(b["state"])
        deviation = (mean - b["reference"]).pow(2).mean(dim=-1).sqrt()
        replay_dev = (mean - b["action"]).pow(2).mean(dim=-1).sqrt()
        return {
            "probe/q_actor": q_actor.mean().item(),
            "probe/q_reference": q_ref.mean().item(),
            "probe/q_gap": (q_actor - q_ref).mean().item(),
            "probe/v": v.mean().item(),
            "probe/actor_deviation": deviation.mean().item(),
            "probe/replay_deviation": replay_dev.mean().item(),
        }

    # --- checkpoints --------------------------------------------------------

    def save(self, path: str) -> None:
        atomic_torch_save(
            {
                "algorithm": "awr",
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "value": self.value.state_dict(),
                "value_target": self.value_target.state_dict(),
                "actor_target": self.actor_target.state_dict() if self.actor_target else None,
                "critic_steps": self.critic_steps,
                "actor_steps": self.actor_steps,
                "config": vars(self.cfg),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.value.load_state_dict(ckpt["value"])
        self.value_target.load_state_dict(ckpt.get("value_target", ckpt["value"]))
        if self.actor_target is not None:
            self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic_steps = int(ckpt.get("critic_steps", 0))
        self.actor_steps = int(ckpt.get("actor_steps", 0))
