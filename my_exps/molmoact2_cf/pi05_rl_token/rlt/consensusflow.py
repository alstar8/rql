"""ConsensusFlow losses on the RL-Token state (V22).

Geometry from the method card `ConsensusFlow/ConsensusFlow.tex`: flow-matching
BC, reverse-state expectile ensemble, one-step actor lookahead, and
common-scale-normalized distillation into a bounded guidance field. Inference
integrates v_theta + G_phi and does not evaluate critics.

The observation is x = (z_rl, proprio). The VLA chunk is the BC target (behavior
prior). The frozen pi0.5 backbone is never updated.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .config import OnlineConfig
from .networks import EnsembleCritic, FlowActor, Guidance
from .shared import atomic_torch_save
from .token_ae import RLTokenAE


def pad_token_batch(rows: list[np.ndarray | None], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack ragged float16 sequences to (B, S, D) plus a 1=valid mask."""
    present = [np.asarray(row) for row in rows if row is not None]
    if not present:
        raise ValueError("token batch is empty")
    dim = int(present[0].shape[-1])
    lengths = [0 if row is None else int(np.asarray(row).shape[0]) for row in rows]
    s_max = max(lengths)
    tok = np.zeros((len(rows), s_max, dim), dtype=np.float32)
    msk = np.zeros((len(rows), s_max), dtype=np.float32)
    for i, row in enumerate(rows):
        if row is None:
            continue
        arr = np.asarray(row, dtype=np.float32)
        s = arr.shape[0]
        tok[i, :s] = arr
        msk[i, :s] = 1.0
    return torch.as_tensor(tok, device=device), torch.as_tensor(msk, device=device)


class ConsensusFlowAgent:
    """PyTorch ConsensusFlow actor / ensemble critic / guidance student."""

    def __init__(
        self,
        cfg: OnlineConfig,
        state_dim: int,
        chunk_dim: int,
        chunk: int,
        token_ae: RLTokenAE | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.chunk_dim = chunk_dim
        self.chunk = chunk
        self.z_dim = int(getattr(cfg, "z_dim", 256))
        self.chunk_discount = cfg.gamma**chunk
        hidden = int(getattr(cfg, "cf_hidden_dim", 512))
        n_layers = int(getattr(cfg, "cf_n_layers", 4))
        ensemble = int(getattr(cfg, "cf_ensemble", 10))
        self.flow_steps = int(getattr(cfg, "cf_flow_steps", 10))
        self.alpha = float(getattr(cfg, "cf_alpha", 1.0))
        self.distill_coef = float(getattr(cfg, "cf_distill_coef", 1.0))
        self.guidance_coef = float(getattr(cfg, "cf_guidance_coef", 0.5))
        self.expectile = float(getattr(cfg, "cf_expectile", 0.5))
        self.rho = float(getattr(cfg, "cf_rho", 0.0))
        self.consensus_floor = float(getattr(cfg, "cf_consensus_floor", 0.01))
        self.conflict_power = float(getattr(cfg, "cf_conflict_power", 2.0))
        self.residual_coef = float(getattr(cfg, "cf_residual_coef", 0.25))
        self.ema = float(getattr(cfg, "cf_ema", 0.999))
        self.ae_coef = float(getattr(cfg, "ae_finetune_coef", 1.0))
        self.q_bounds = (0.0, 1.0)

        self.actor = FlowActor(state_dim, chunk_dim, hidden, n_layers).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic = EnsembleCritic(
            state_dim, chunk_dim, hidden, n_layers, ensemble, layer_norm=True
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.guidance = Guidance(state_dim, chunk_dim, hidden, n_layers).to(self.device)

        self.token_ae = token_ae.to(self.device) if token_ae is not None else None
        self.train_token = bool(getattr(cfg, "train_token", False))
        self.ae_finetune = bool(getattr(cfg, "ae_finetune", False))
        self.token_corpus = None
        self.token_corpus_rng = np.random.default_rng(cfg.seed)
        if self.token_ae is not None:
            self.z_dim = self.token_ae.z_dim

        params = list(self.actor.parameters()) + list(self.critic.parameters()) + list(self.guidance.parameters())
        if self.token_ae is not None:
            params += list(self.token_ae.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.critic_steps = 0
        self.actor_steps = 0

    def freeze_token(self) -> None:
        """Online GPU-5 scheme: encoder gets no RL (or recon) gradients."""
        self.train_token = False
        if self.token_ae is not None and not self.ae_finetune:
            self.token_ae.requires_grad_(False)

    def set_ae_finetune(self, enabled: bool) -> None:
        self.ae_finetune = enabled
        if self.token_ae is not None and enabled:
            self.token_ae.requires_grad_(True)

    def _to_torch(self, batch: dict) -> dict:
        out = {}
        for key, value in batch.items():
            if key in ("tokens", "mask", "next_tokens", "next_mask"):
                out[key] = value
                continue
            out[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return out

    def _states(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """RL states. Re-encode a token subset when the encoder is learning."""
        extra: dict[str, float] = {}
        state = batch["state"]
        next_state = batch["next_state"]
        if self.token_ae is None or not self.train_token or "tokens" not in batch:
            return state, next_state, extra
        token_bs = int(getattr(self.cfg, "token_batch_size", 8))
        n = min(token_bs, state.shape[0])
        tok, msk = pad_token_batch(batch["tokens"][:n], self.device)
        nxt, nmsk = pad_token_batch(batch["next_tokens"][:n], self.device)
        z = self.token_ae.encode(tok, msk)
        z_next = self.token_ae.encode(nxt, nmsk)
        proprio = state[:n, self.z_dim :]
        next_proprio = next_state[:n, self.z_dim :]
        encoded = torch.cat([z, proprio], dim=-1)
        encoded_next = torch.cat([z_next, next_proprio], dim=-1)
        if n < state.shape[0]:
            state = torch.cat([encoded, state[n:].detach()], dim=0)
            next_state = torch.cat([encoded_next, next_state[n:].detach()], dim=0)
        else:
            state = encoded
            next_state = encoded_next
        extra["token_encode_n"] = float(n)
        extra["z_norm"] = float(z.norm(dim=-1).mean().item())
        return state, next_state, extra

    def _project_unit_ball(self, w: torch.Tensor) -> torch.Tensor:
        w_norm = w.norm(dim=-1, keepdim=True)
        return w * torch.minimum(torch.ones_like(w_norm), 1.0 / (w_norm + 1e-6))

    def _behavior_safe_direction(self, w: torch.Tensor, behavior_velocity: torch.Tensor) -> torch.Tensor:
        w = self._project_unit_ball(w)
        w_norm = w.norm(dim=-1, keepdim=True)
        trust = w_norm.detach()
        velocity = behavior_velocity.detach()
        velocity_unit = velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        parallel = (w * velocity_unit).sum(dim=-1, keepdim=True)
        kill_frac = 1.0 - trust.clamp(0.0, 1.0).pow(self.conflict_power)
        conflict_free = self._project_unit_ball(w - kill_frac * parallel.clamp(max=0.0) * velocity_unit)
        alignment_cos = parallel / (w_norm + 1e-6)
        damp = (1.0 - self.residual_coef * alignment_cos.clamp(min=0.0) * trust).clamp(0.0, 1.0)
        return self._project_unit_ball(conflict_free * damp)

    def guidance_field(self, state: torch.Tensor, x: torch.Tensor, t: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        w = self.guidance(state, x, t)
        safe = self._behavior_safe_direction(w, velocity)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.guidance_coef * t * safe

    def _euler(self, state: torch.Tensor, noise: torch.Tensor, use_target: bool) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        action = noise
        for i in range(self.flow_steps):
            t = torch.full((state.shape[0], 1), i / self.flow_steps, device=self.device, dtype=state.dtype)
            velocity = actor(state, action, t)
            guide = self.guidance_field(state, action, t, velocity)
            action = action + (velocity + guide) / self.flow_steps
        return action

    @torch.no_grad()
    def act(self, state: np.ndarray, reference: np.ndarray, explore: bool) -> np.ndarray:
        del reference
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        noise = torch.randn(1, self.chunk_dim, device=self.device)
        action = self._euler(s, noise, use_target=not explore)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def q_values(self, state: np.ndarray, action: np.ndarray) -> float:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        t = torch.ones(1, 1, device=self.device)
        mean, _ = self.critic.mean_std(s, a, t)
        return float(mean.item())

    def _recon_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        if self.token_ae is None or not self.ae_finetune:
            zero = torch.zeros((), device=self.device)
            return zero, {}
        token_bs = int(getattr(self.cfg, "token_batch_size", 8))
        if "tokens" in batch:
            tok, msk = pad_token_batch(batch["tokens"][:token_bs], self.device)
        elif self.token_corpus is not None:
            tok, msk = self.token_corpus.sample(token_bs, self.device, self.token_corpus_rng)
        else:
            zero = torch.zeros((), device=self.device)
            return zero, {}
        loss, stats = self.token_ae.reconstruction_loss(tok, msk)
        return loss, {f"ae_{k}": v for k, v in stats.items()}

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One joint ConsensusFlow step: actor + alpha BC + critic + eta distill [+ AE]."""
        b = self._to_torch(batch)
        state, next_state, enc_stats = self._states(b)
        n = state.shape[0]
        dt = 1.0 / self.flow_steps

        with torch.no_grad():
            next_noise = torch.randn(n, self.chunk_dim, device=self.device)
            t0 = torch.zeros(n, 1, device=self.device)
            next_q, next_std = self.critic_target.mean_std(next_state, next_noise, t0)
            bootstrap = next_q - self.rho * next_std
            target = b["reward"] + (1.0 - b["done"]) * self.chunk_discount * bootstrap
            if self.cfg.clip_target:
                target = target.clamp(*self.q_bounds)

        x = b["action"].clone()
        t = torch.ones(n, 1, device=self.device)
        for _ in range(self.flow_steps):
            velocity = self.actor(state, x, t)
            guide = self.guidance_field(state, x, t, velocity)
            x = x - (velocity + guide) * dt
            t = t - dt
        x_crit = x.detach()
        t_crit = t.detach()
        q = self.critic(state, x_crit, t_crit)
        err = target.unsqueeze(0) - q
        weight = torch.where(err >= 0, self.expectile, 1.0 - self.expectile)
        critic_loss = (weight * err.pow(2)).mean()

        x0 = torch.randn(n, self.chunk_dim, device=self.device)
        t_bc = torch.rand(n, 1, device=self.device)
        x1 = b["reference"]
        x_t = (1.0 - t_bc) * x0 + t_bc * x1
        target_velocity = x1 - x0
        velocity = self.actor(state, x_t, t_bc)
        bc_loss = F.mse_loss(velocity, target_velocity)

        guide = self.guidance_field(state, x_t, t_bc, velocity).detach()
        lookahead = x_t + (velocity + guide) * torch.minimum(
            torch.full_like(t_bc, dt), 1.0 - t_bc
        )
        t_plus = (t_bc + dt).clamp(max=1.0)
        q_pe, _ = self.critic.mean_std(state, lookahead, t_plus)
        actor_loss = -q_pe.mean()

        member = torch.randint(0, self.critic.ensemble, (n,), device=self.device)
        x_t_grad = x_t.detach().requires_grad_(True)

        def q_picked() -> torch.Tensor:
            qs = self.critic_target(state, x_t_grad, t_bc)
            picked = qs[member, torch.arange(n, device=self.device)]
            return picked.sum()

        (q_grad,) = torch.autograd.grad(q_picked(), x_t_grad, create_graph=False)
        q_grad = q_grad.detach()
        q_grad_norm = q_grad.norm(dim=-1, keepdim=True)
        grad_scale = q_grad_norm.mean()
        z_k = q_grad / (q_grad_norm + self.consensus_floor * grad_scale + 1e-6)
        w = self.guidance(state, x_t.detach(), t_bc)
        distill_loss = (w - z_k).pow(2).sum(dim=-1).mean()

        recon_loss, recon_stats = self._recon_loss(batch)
        loss = actor_loss + self.alpha * bc_loss + critic_loss + self.distill_coef * distill_loss
        if self.ae_finetune:
            loss = loss + self.ae_coef * recon_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self._polyak(self.critic, self.critic_target, self.cfg.tau)
        self._polyak(self.actor, self.actor_target, 1.0 - self.ema)
        self.critic_steps += 1
        self.actor_steps += 1

        stats = {
            "total_loss": float(loss.item()),
            "actor_loss": float(actor_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "distill_loss": float(distill_loss.item()),
            "q_mean": float(q.mean().item()),
            "target_mean": float(target.mean().item()),
            "actor_q": float(q_pe.mean().item()),
            "w_norm": float(w.norm(dim=-1).mean().item()),
            **enc_stats,
            **recon_stats,
        }
        if self.ae_finetune:
            stats["ae_recon_loss"] = float(recon_loss.item())
        return stats

    def critic_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        return self.update(batch)

    def actor_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        return {}

    def actor_bc_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        stats = self.update(batch)
        stats["actor_bc_loss"] = stats.get("bc_loss", 0.0)
        stats["actor_bc_rmse"] = float(np.sqrt(max(stats.get("bc_loss", 0.0), 0.0)))
        return stats

    def _polyak(self, net, target, tau: float) -> None:
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.mul_(1.0 - tau).add_(tau * p)

    @torch.no_grad()
    def probe(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        b = self._to_torch(batch)
        state = b["state"]
        sampled = self._euler(state, torch.randn_like(b["action"]), use_target=True)
        t1 = torch.ones(state.shape[0], 1, device=self.device)
        q_actor, _ = self.critic.mean_std(state, sampled, t1)
        q_ref, _ = self.critic.mean_std(state, b["reference"], t1)
        deviation = (sampled - b["reference"]).pow(2).mean(dim=-1).sqrt()
        return {
            "probe/q_actor": float(q_actor.mean().item()),
            "probe/q_reference": float(q_ref.mean().item()),
            "probe/q_gap": float((q_actor - q_ref).mean().item()),
            "probe/actor_deviation": float(deviation.mean().item()),
        }

    def save(self, path: str) -> None:
        payload = {
            "algorithm": "consensusflow",
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "guidance": self.guidance.state_dict(),
            "token_ae": None if self.token_ae is None else self.token_ae.state_dict(),
            "token_ae_config": None if self.token_ae is None else self.token_ae.config,
            "critic_steps": self.critic_steps,
            "actor_steps": self.actor_steps,
            "train_token": self.train_token,
            "ae_finetune": self.ae_finetune,
            "config": vars(self.cfg),
        }
        atomic_torch_save(payload, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.guidance.load_state_dict(ckpt["guidance"])
        if self.token_ae is not None and ckpt.get("token_ae") is not None:
            self.token_ae.load_state_dict(ckpt["token_ae"])
        self.critic_steps = int(ckpt.get("critic_steps", 0))
        self.actor_steps = int(ckpt.get("actor_steps", 0))


def make_agent(cfg: OnlineConfig, state_dim: int, chunk_dim: int, chunk: int, token_ae=None):
    """Gaussian RL-Token, ConsensusFlow, or flow_rlt, chosen by OnlineConfig.algorithm."""
    from .agent import RLTokenAgent
    from .flow_rlt import FlowRLTAgent

    algorithm = getattr(cfg, "algorithm", "rl_token")
    if algorithm == "consensusflow":
        return ConsensusFlowAgent(cfg, state_dim, chunk_dim, chunk, token_ae=token_ae)
    if algorithm == "flow_rlt":
        return FlowRLTAgent(cfg, state_dim, chunk_dim, chunk, token_ae=token_ae)
    return RLTokenAgent(cfg, state_dim, chunk_dim, chunk)
