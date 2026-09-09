"""On-policy PPO on the RL-Token Gaussian actor (Schulman et al. 2017).

V21 maximises Q by backpropagating through a reparameterized sample (Eq. 5).
AWR does weighted regression onto *replay* actions. PPO does neither.

Each collector payload is one contiguous chunk-MDP trajectory. GAE(λ) is computed
on that trajectory, then K epochs of the clipped surrogate run once enough on-policy
rows have accumulated:

    r_t = π_θ(a_t | s_t, ã_t) / π_old(a_t | s_t, ã_t)
    L_π = −E[ min( r A, clip(r, 1−ε, 1+ε) A ) ]
    L_V = ½ E[ max( (V−R)², (V_clip−R)² ) ]

π is the same one-pass Gaussian as V21, conditioned on the frozen-VLA reference.
σ is fixed (matched to V21), so the Gaussian entropy is a constant and is not a
loss term. π_old is the log-probability stored at act() time — recomputing it later
would use the learner's weights, not the collector's. A payload that omits that
scalar is refused.

The critic is a state-value V, not a twin Q. Advantages are stop-grad. There is no
Eq. 5 Euclidean pull and no reference dropout on the PPO update (dropout would
break the importance ratio). GAE runs on a finished episode, not on a mid-episode
close (that would collapse GAE(λ) to TD(0)).

AC pretrain cannot be PPO: frozen-VLA chunks are not draws from this Gaussian.
pretrain uses V21's BC (μ → ã) plus clipped TD on V. PPO starts at the probe.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import OnlineConfig
from .networks import Actor, Value
from .shared import atomic_torch_save


def generalized_advantage_estimate(
    reward: torch.Tensor,
    value: torch.Tensor,
    next_value: torch.Tensor,
    done: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(λ) on a contiguous trajectory.

    δ_t = r_t + γ (1 − d_t) V(s'_t) − V(s_t)
    A_t = δ_t + γ λ (1 − d_t) A_{t+1}

    Terminals (`done=1`) zero both the bootstrap and the GAE carry, so two
    concatenated episodes do not leak. Timeouts keep `done=0` and bootstrap from
    V(s'). Returns (advantage, return) with return = advantage + value.
    """
    if reward.ndim != 1 or value.shape != reward.shape:
        raise ValueError(f"GAE expects 1-d (T,) tensors, got reward {tuple(reward.shape)}")
    t_len = reward.shape[0]
    adv = torch.zeros_like(reward)
    gae = reward.new_zeros(())
    for t in range(t_len - 1, -1, -1):
        not_done = 1.0 - done[t]
        delta = reward[t] + gamma * not_done * next_value[t] - value[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    return adv, adv + value


def ppo_surrogate(ratio: torch.Tensor, adv: torch.Tensor, clip: float) -> torch.Tensor:
    """−mean min(r A, clip(r, 1−ε, 1+ε) A). `adv` must be stop-grad."""
    surr_1 = ratio * adv
    surr_2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv
    return -torch.min(surr_1, surr_2).mean()


def ppo_value_loss(value: torch.Tensor, old_value: torch.Tensor, ret: torch.Tensor, clip: float) -> torch.Tensor:
    """PPO2 clipped value loss. `old_value` and `ret` are stop-grad."""
    clipped = old_value + (value - old_value).clamp(-clip, clip)
    return 0.5 * torch.max((value - ret).pow(2), (clipped - ret).pow(2)).mean()


class PPOAgent:
    """Gaussian RL-Token actor trained by PPO; V trained by GAE returns (online) and TD (pretrain)."""

    def __init__(self, cfg: OnlineConfig, state_dim: int, chunk_dim: int, chunk: int) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.chunk_dim = chunk_dim
        self.chunk_discount = cfg.gamma**chunk
        self.ppo_clip = float(getattr(cfg, "ppo_clip", 0.2))
        self.ppo_gae_lambda = float(getattr(cfg, "ppo_gae_lambda", 0.95))
        self.ppo_epochs = int(getattr(cfg, "ppo_epochs", 4))
        self.ppo_horizon = int(getattr(cfg, "ppo_horizon", 256))
        self.ppo_minibatch = int(getattr(cfg, "ppo_minibatch", 64))
        self.ppo_vf_coef = float(getattr(cfg, "ppo_vf_coef", 0.5))
        self.ppo_max_grad_norm = float(getattr(cfg, "ppo_max_grad_norm", 0.5))
        self.ppo_norm_adv = bool(getattr(cfg, "ppo_norm_adv", True))
        if self.ppo_clip <= 0:
            raise ValueError(f"ppo_clip must be > 0, got {self.ppo_clip}")
        if not 0.0 <= self.ppo_gae_lambda <= 1.0:
            raise ValueError(f"ppo_gae_lambda must be in [0, 1], got {self.ppo_gae_lambda}")
        if self.ppo_epochs < 1:
            raise ValueError(f"ppo_epochs must be >= 1, got {self.ppo_epochs}")
        if self.ppo_horizon < 1:
            raise ValueError(f"ppo_horizon must be >= 1, got {self.ppo_horizon}")
        if self.ppo_minibatch < 1:
            raise ValueError(f"ppo_minibatch must be >= 1, got {self.ppo_minibatch}")
        if self.ppo_vf_coef < 0:
            raise ValueError(f"ppo_vf_coef must be >= 0, got {self.ppo_vf_coef}")
        if self.ppo_max_grad_norm <= 0:
            raise ValueError(f"ppo_max_grad_norm must be > 0, got {self.ppo_max_grad_norm}")

        self.actor = Actor(state_dim, chunk_dim, cfg.hidden_dim, cfg.n_layers, cfg.sigma).to(self.device)
        self.value = Value(state_dim, cfg.hidden_dim, cfg.n_layers, cfg.critic_layer_norm).to(self.device)
        self.value_target = copy.deepcopy(self.value).requires_grad_(False)
        self.q_bounds = (0.0, 1.0)
        self.last_log_prob: float | None = None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=cfg.lr)
        self.critic_steps = 0
        self.actor_steps = 0
        self._pending: list[dict] = []

    # --- acting -------------------------------------------------------------

    @torch.no_grad()
    def act(self, state: np.ndarray, reference: np.ndarray, explore: bool = False) -> np.ndarray:
        """One action chunk, flat (C * action_dim,). Stores log π of the executed chunk."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        r = torch.as_tensor(reference, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean = self.actor(s, r)
        action = mean + self.actor.sigma * torch.randn_like(mean) if explore else mean
        self.last_log_prob = float(self.actor.log_prob(s, r, action).item())
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def q_values(self, state: np.ndarray, action: np.ndarray) -> float:
        """PPO has no Q. Reported as V(s) so generic probe printers still get a scalar."""
        del action
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.value(s).item())

    # --- learning -----------------------------------------------------------

    def _to_torch(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        out = {}
        for key, value in batch.items():
            if key in ("tokens", "mask", "next_tokens", "next_mask"):
                out[key] = value
                continue
            out[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return out

    def _stack_rows(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        stacked = {
            "state": np.stack([r["state"] for r in rows]),
            "action": np.stack([r["action"] for r in rows]),
            "reference": np.stack([r["reference"] for r in rows]),
            "next_state": np.stack([r["next_state"] for r in rows]),
            "reward": np.asarray([r["reward"] for r in rows], dtype=np.float32),
            "done": np.asarray([r["done"] for r in rows], dtype=np.float32),
        }
        out = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in stacked.items()}
        out["log_prob"] = torch.as_tensor(
            [float(r["log_prob"]) for r in rows], dtype=torch.float32, device=self.device
        )
        return out

    def critic_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Clipped TD on V. Used in AC pretrain (shuffled frozen-VLA replay), not online."""
        b = self._to_torch(batch)
        with torch.no_grad():
            target = b["reward"] + (1.0 - b["done"]) * self.chunk_discount * self.value_target(b["next_state"])
            if self.cfg.clip_target:
                target = target.clamp(*self.q_bounds)
        v = self.value(b["state"])
        loss = F.mse_loss(v, target)
        self.value_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.ppo_max_grad_norm)
        self.value_opt.step()
        self._polyak(self.value, self.value_target)
        self.critic_steps += 1
        return {
            "critic_loss": loss.item(),
            "q_mean": v.mean().item(),
            "v_mean": v.mean().item(),
            "target_mean": target.mean().item(),
        }

    def actor_bc_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Copy the VLA reference: μ(x, ã) → ã. Pretrain only; this is not PPO."""
        b = self._to_torch(batch)
        keep = (torch.rand(b["reference"].shape[0], 1, device=self.device) >= self.cfg.reference_dropout).float()
        mean = self.actor(b["state"], b["reference"] * keep)
        loss = F.mse_loss(mean, b["reference"])
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.ppo_max_grad_norm)
        self.actor_opt.step()
        self.actor_steps += 1
        deviation = (mean - b["reference"]).pow(2).mean()
        return {
            "actor_bc_loss": loss.item(),
            "actor_bc_rmse": float(deviation.sqrt().item()),
            "actor_ref_rmse": float(deviation.sqrt().item()),
        }

    def actor_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        del batch
        raise RuntimeError(
            "PPO is on-policy. The learner must call consume_on_policy on sequential "
            "trajectory rows, not actor_step on shuffled replay."
        )

    def consume_on_policy(self, rows: list[dict]) -> dict[str, float]:
        """GAE on one contiguous trajectory, then PPO epochs once `ppo_horizon` rows are ready.

        `π_old` is the log-probability stored at act(). Recomputing it here would use the
        learner's weights, not the collector's. A payload with no log π (warmup: the frozen
        VLA ran) is not on-policy for this Gaussian and is skipped. A mixed payload is a bug.
        """
        if not rows:
            return {"ppo_pending": float(len(self._pending))}
        has_log = [r.get("log_prob") is not None for r in rows]
        if not any(has_log):
            return {"ppo_pending": float(len(self._pending))}
        if not all(has_log):
            missing = sum(1 for flag in has_log if not flag)
            raise RuntimeError(
                "PPO needs log π stored at act() on every row of an on-policy payload; "
                f"{missing}/{len(rows)} rows are missing it. Recomputing under the learner "
                "would be a fake π_old."
            )
        self._pending.extend(self._gae_segment(rows))
        if len(self._pending) < self.ppo_horizon:
            return {"ppo_pending": float(len(self._pending))}
        return self._ppo_epochs()

    def _gae_segment(self, rows: list[dict]) -> list[dict]:
        b = self._stack_rows(rows)
        with torch.no_grad():
            value = self.value(b["state"])
            next_value = self.value(b["next_state"])
            adv, ret = generalized_advantage_estimate(
                b["reward"], value, next_value, b["done"], self.chunk_discount, self.ppo_gae_lambda
            )
            logp_old = b["log_prob"]
        out = []
        for i, row in enumerate(rows):
            out.append(
                {
                    "state": np.asarray(row["state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "reference": np.asarray(row["reference"], dtype=np.float32),
                    "log_prob": float(logp_old[i].item()),
                    "adv": float(adv[i].item()),
                    "ret": float(ret[i].item()),
                    "value": float(value[i].item()),
                }
            )
        return out

    def _ppo_epochs(self) -> dict[str, float]:
        n = len(self._pending)
        state = torch.as_tensor(np.stack([r["state"] for r in self._pending]), dtype=torch.float32, device=self.device)
        action = torch.as_tensor(np.stack([r["action"] for r in self._pending]), dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(
            np.stack([r["reference"] for r in self._pending]), dtype=torch.float32, device=self.device
        )
        logp_old = torch.as_tensor([r["log_prob"] for r in self._pending], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor([r["adv"] for r in self._pending], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor([r["ret"] for r in self._pending], dtype=torch.float32, device=self.device)
        old_value = torch.as_tensor([r["value"] for r in self._pending], dtype=torch.float32, device=self.device)
        if self.ppo_norm_adv:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        mb = min(self.ppo_minibatch, n)
        last: dict[str, float] = {}
        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n - mb + 1, mb):
                idx = perm[start : start + mb]
                last = self._minibatch_step(
                    state[idx],
                    action[idx],
                    reference[idx],
                    logp_old[idx],
                    adv[idx],
                    ret[idx],
                    old_value[idx],
                )
        self._pending = []
        last["ppo_batch"] = float(n)
        return last

    def _minibatch_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
        logp_old: torch.Tensor,
        adv: torch.Tensor,
        ret: torch.Tensor,
        old_value: torch.Tensor,
    ) -> dict[str, float]:
        logp = self.actor.log_prob(state, reference, action)
        # Clamp the log-ratio so exp() cannot overflow on a 64-d N(μ, 0.02² I).
        # The PPO clip ε still binds: exp(±20) is far outside [1−ε, 1+ε].
        ratio = (logp - logp_old).clamp(-20.0, 20.0).exp()
        loss_pi = ppo_surrogate(ratio, adv.detach(), self.ppo_clip)
        v = self.value(state)
        loss_v = ppo_value_loss(v, old_value.detach(), ret.detach(), self.ppo_clip)
        loss = loss_pi + self.ppo_vf_coef * loss_v

        self.actor_opt.zero_grad(set_to_none=True)
        self.value_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.ppo_max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.ppo_max_grad_norm)
        self.actor_opt.step()
        self.value_opt.step()
        self._polyak(self.value, self.value_target)
        self.actor_steps += 1
        self.critic_steps += 1

        with torch.no_grad():
            mean = self.actor(state, reference)
            ref_mse = (mean - reference).pow(2).mean()
            clip_frac = ((ratio < 1.0 - self.ppo_clip) | (ratio > 1.0 + self.ppo_clip)).float().mean()
            approx_kl = 0.5 * (logp_old - logp).pow(2).mean()
            entropy = 0.5 * self.chunk_dim * (1.0 + math.log(2.0 * math.pi * self.actor.sigma**2))
        return {
            "actor_loss": float(loss_pi.item()),
            "critic_loss": float(loss_v.item()),
            "v_mean": float(v.mean().item()),
            "q_mean": float(v.mean().item()),
            "actor_ref_rmse": float(ref_mse.sqrt().item()),
            "ppo_ratio_mean": float(ratio.mean().item()),
            "ppo_clip_frac": float(clip_frac.item()),
            "ppo_approx_kl": float(approx_kl.item()),
            "ppo_adv_mean": float(adv.mean().item()),
            "ppo_entropy": float(entropy),
        }

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
        v = self.value(b["state"])
        deviation = (mean - b["reference"]).pow(2).mean(dim=-1).sqrt()
        logp = self.actor.log_prob(b["state"], b["reference"], b["action"])
        return {
            "probe/v": v.mean().item(),
            "probe/actor_deviation": deviation.mean().item(),
            "probe/log_prob": logp.mean().item(),
        }

    # --- checkpoints --------------------------------------------------------

    def save(self, path: str) -> None:
        atomic_torch_save(
            {
                "algorithm": "ppo",
                "actor": self.actor.state_dict(),
                "value": self.value.state_dict(),
                "value_target": self.value_target.state_dict(),
                "critic_steps": self.critic_steps,
                "actor_steps": self.actor_steps,
                "config": vars(self.cfg),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.value.load_state_dict(ckpt["value"])
        self.value_target.load_state_dict(ckpt.get("value_target", ckpt["value"]))
        self.critic_steps = int(ckpt.get("critic_steps", 0))
        self.actor_steps = int(ckpt.get("actor_steps", 0))
        self._pending = []
        self.last_log_prob = None
