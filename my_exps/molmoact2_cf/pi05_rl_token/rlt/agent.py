"""The actor-critic updates of Algorithm 1.

    critic   L_Q  = E[(Qhat - Q(x, a))^2],
             Qhat = sum_t' gamma^(t'-1) r_t' + gamma^C * min_i Q'_i(x', a'),
                    a' ~ pi(.|x', a_ref')                              Eq. 3
    actor    L_pi = E[-Q(x, a) + beta ||a - a_ref||^2],
                    a ~ pi(.|x, a_ref or 0)                            Eq. 5

Two details the paper fixes and we follow literally: the target uses the
*online* actor (Eq. 3 writes a' ~ pi_theta, not pi_theta'), and the minimum of
the two Q heads is used for targets only. The actor follows TD3 and maximizes
the first head.

Reference dropout applies to the actor's *input* while it learns; the target's
actor call is an inference use and always gets the reference. The penalty in
L_pi is always measured against the true reference, dropped or not.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .shared import atomic_torch_save

from .config import OnlineConfig
from .networks import Actor, DoubleCritic


class RLTokenAgent:
    def __init__(self, cfg: OnlineConfig, state_dim: int, chunk_dim: int, chunk: int) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.chunk_dim = chunk_dim
        self.chunk_discount = cfg.gamma**chunk

        self.actor = Actor(state_dim, chunk_dim, cfg.hidden_dim, cfg.n_layers, cfg.sigma).to(self.device)
        self.critic = DoubleCritic(
            state_dim, chunk_dim, cfg.hidden_dim, cfg.n_layers, cfg.critic_layer_norm
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        # Only built when asked for: Eq. 3 bootstraps through the online actor.
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False) if cfg.target_actor else None
        # A single +1 on a terminal step means the true return never leaves [0, 1].
        self.q_bounds = (0.0, 1.0)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.critic_steps = 0
        self.actor_steps = 0

    # --- acting -------------------------------------------------------------

    @torch.no_grad()
    def act(self, state: np.ndarray, reference: np.ndarray, explore: bool) -> np.ndarray:
        """One action chunk, flat (C * action_dim,). The reference is always given."""
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
        # Token prefixes are stored as a list of ragged float16 arrays for AE
        # training. This actor-critic never reads them; converting the list with
        # torch.as_tensor copies ~2 GB per field on the CPU and starves the GPU.
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
            target = b["reward"] + (1.0 - b["done"]) * self.chunk_discount * next_q
            if self.cfg.clip_target:
                target = target.clamp(*self.q_bounds)

        q1, q2 = self.critic(b["state"], b["action"])
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_opt.step()
        self._soft_update()
        self.critic_steps += 1

        return {"critic_loss": loss.item(), "q_mean": q1.mean().item(), "target_mean": target.mean().item()}

    def actor_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        b = self._to_torch(batch)

        # Reference dropout: a random half of the batch sees zeros instead of a_ref,
        # so the actor keeps a pathway that does not depend on copying the VLA.
        keep = (torch.rand(b["reference"].shape[0], 1, device=self.device) >= self.cfg.reference_dropout).float()
        action = self.actor.sample(b["state"], b["reference"] * keep)

        q = self.critic.q1_value(b["state"], action)
        deviation = (action - b["reference"]).pow(2).sum(dim=-1)
        loss = (-q + self.cfg.beta * deviation).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        self.actor_steps += 1
        if self.actor_target is not None:
            self._polyak(self.actor, self.actor_target)

        return {
            "actor_loss": loss.item(),
            "actor_q": q.mean().item(),
            "actor_ref_rmse": float(deviation.mean().div(self.chunk_dim).sqrt().item()),
        }

    def actor_bc_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Copy the VLA reference: mu(x, a_ref) -> a_ref.

        Frozen-VLA collect has action == reference, so the RL actor loss is uninformative
        (Q cannot tell good actions from good states). Behaviour cloning is the pretrain
        that lets online start at episode 0 without a VLA-only warmup.
        """
        b = self._to_torch(batch)
        mean = self.actor(b["state"], b["reference"])
        loss = F.mse_loss(mean, b["reference"])

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_opt.step()
        self.actor_steps += 1
        if self.actor_target is not None:
            self._polyak(self.actor, self.actor_target)

        deviation = (mean - b["reference"]).pow(2).mean()
        return {
            "actor_bc_loss": loss.item(),
            "actor_bc_rmse": float(deviation.sqrt().item()),
        }

    def _soft_update(self) -> None:
        self._polyak(self.critic, self.critic_target)

    def _polyak(self, net, target) -> None:
        tau = self.cfg.tau
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(tau * p)

    # --- diagnostics --------------------------------------------------------

    @torch.no_grad()
    def probe(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Does the critic prefer the actor's action over the VLA's own?

        A flat Q -- the failure mode of the offline attempt -- shows up here as
        q_gap ~ 0 while the actor still deviates from the reference.
        """
        b = self._to_torch(batch)
        mean = self.actor(b["state"], b["reference"])  # deterministic: no exploration noise
        q_actor = self.critic.min_q(b["state"], mean)
        q_ref = self.critic.min_q(b["state"], b["reference"])
        deviation = (mean - b["reference"]).pow(2).mean(dim=-1).sqrt()
        return {
            "probe/q_actor": q_actor.mean().item(),
            "probe/q_reference": q_ref.mean().item(),
            "probe/q_gap": (q_actor - q_ref).mean().item(),
            "probe/actor_deviation": deviation.mean().item(),
        }

    # --- checkpoints --------------------------------------------------------

    def save(self, path: str) -> None:
        atomic_torch_save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
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
        if self.actor_target is not None:
            self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic_steps = int(ckpt.get("critic_steps", 0))
        self.actor_steps = int(ckpt.get("actor_steps", 0))
