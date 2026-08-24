"""Corrected V22: flow-matching actor on the RL-Token state, Q from a frozen RLT critic.

Why the first V22 (ConsensusFlow) did not learn: its actor maximized an unbounded
ensemble-Q lookahead, so actor_q left [0, 1] within a few hundred offline steps
(~3e3 offline, ~1e5 online) while the flow-matching BC term stayed O(1). The flow
never fit the VLA chunk distribution, and Euler integration from noise emitted
joint deltas up to 1e9 -- orders of magnitude past the ~0.2 rad the scene allows.

Here the Q signal is a frozen RL-Token critic (DoubleCritic from a finished V21
run), trained on real mug rollouts with the TD target clipped to [0, 1]. Frozen
alone is not enough: a ReLU MLP critic extrapolates without bound off its data
manifold, and a bare -Q ascent finds those rays within a few dozen steps (measured
2026-08-24: actor_q ~ 1e22 by offline step 50). So the actor keeps the paper's
Eq. 5 anchor, -Q(s, a) + beta * ||a - a_ref||^2 with beta = 100, the value this
project calibrated for the delta action space. The anchor gradient grows linearly
with the deviation while the critic's is piecewise constant, so the ascent is
confined to the trust region where the frozen critic is meaningful. The flow actor
is the only RL-trained module: pi0.5, the token encoder that builds z_rl, and the
critic are all frozen. The base AE finetunes separately via reconstruction
(ae_finetune); the encoder that produces z_rl for the buffer stays the frozen
on-disk one, so the critic's input distribution never shifts.

The flow is conditioned on the VLA reference chunk (ref_dim = chunk_dim), the same
conditioning the RL-Token actor uses. An unconditional flow must regenerate the
whole chunk distribution from z_rl, and the VLA's proposal given the state is
dispersed enough (~0.1 rad RMS) that its samples start at 0/10 on mug; a
reference-conditioned flow is a corrector -- it copies the reference at BC
convergence (dev ~0.03, like the V21 pretrain) and the frozen critic bends the
endpoint online.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .config import OnlineConfig
from .networks import DoubleCritic, FlowActor
from .shared import atomic_torch_save
from .token_ae import RLTokenAE


def load_frozen_critic(path: str, state_dim: int, chunk_dim: int, device: torch.device) -> DoubleCritic:
    """The critic half of an RLTokenAgent checkpoint, frozen and in eval mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    stored = ckpt.get("config", {})
    critic = DoubleCritic(
        state_dim,
        chunk_dim,
        int(stored.get("hidden_dim", 256)),
        int(stored.get("n_layers", 2)),
        bool(stored.get("critic_layer_norm", False)),
    )
    critic.load_state_dict(ckpt["critic"])
    return critic.to(device).eval().requires_grad_(False)


class FlowRLTAgent:
    """Flow actor trained by BC to the VLA chunk plus the frozen critic's Q."""

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
        hidden = int(getattr(cfg, "cf_hidden_dim", 512))
        n_layers = int(getattr(cfg, "cf_n_layers", 4))
        self.flow_steps = int(getattr(cfg, "cf_flow_steps", 10))
        self.bc_coef = float(getattr(cfg, "flow_bc_coef", 1.0))
        self.actor_coef = float(getattr(cfg, "flow_actor_coef", 1.0))
        # Eq. 5 anchor weight. RLConfig passes its calibrated delta-space value
        # (100); OnlineConfig's own default 1.0 is calibrated for absolute actions
        # and leaves the flow actor free to walk off the critic's data manifold.
        self.beta = float(getattr(cfg, "beta", 100.0))
        self.ema = float(getattr(cfg, "cf_ema", 0.999))
        self.ae_coef = float(getattr(cfg, "ae_finetune_coef", 1.0))

        self.chunk_discount = cfg.gamma**chunk
        self.q_bounds = (0.0, 1.0)
        self.tau = float(getattr(cfg, "tau", 0.005))
        self.clip_target = bool(getattr(cfg, "clip_target", True))
        # Two critic modes. With rlt_critic set, the critic is a finished RL-Token
        # critic, frozen forever (the standalone corrected-V22 test). With it empty,
        # the critic is trained by TD alongside the actor: offline always, and online
        # unless build() freezes it (flow_freeze_critic_online) -- arms 2 and 3 of the
        # one-pass-vs-flow comparison share this code path.
        self.external_critic = bool(cfg.rlt_critic)
        if self.external_critic:
            self.critic = load_frozen_critic(cfg.rlt_critic, state_dim, chunk_dim, self.device)
            self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
            self.critic_frozen = True
        else:
            self.critic = DoubleCritic(
                state_dim, chunk_dim, cfg.hidden_dim, cfg.n_layers, cfg.critic_layer_norm
            ).to(self.device)
            self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
            self.critic_frozen = False
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)

        # compose=False: the actor is the full corrector velocity v(x, x_t, t, ref).
        # compose=True (CF composition): the actor is an unconditioned guidance field
        # G(x, x_t, t); the base OT velocity toward the reference (ref - noise) is
        # added analytically, so V = v_pi05_base + G and the policy is an exact pi05
        # copy at init (G ~ 0). This is the "V = pi05 actor + G(RLT actor)" arm.
        self.compose = bool(getattr(cfg, "flow_compose", False))
        ref_dim = 0 if self.compose else chunk_dim
        self.actor = FlowActor(state_dim, chunk_dim, hidden, n_layers, ref_dim=ref_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)

        self.token_ae = token_ae.to(self.device) if token_ae is not None else None
        self.train_token = False  # z_rl always comes from the frozen on-disk encoder
        self.ae_finetune = bool(getattr(cfg, "ae_finetune", False))
        self.token_corpus = None
        self.token_corpus_rng = np.random.default_rng(cfg.seed)
        if self.token_ae is not None:
            self.z_dim = self.token_ae.z_dim

        params = list(self.actor.parameters())
        if self.token_ae is not None:
            params += list(self.token_ae.parameters())
        self.opt = torch.optim.Adam(params, lr=cfg.lr)
        self.critic_steps = 0
        self.actor_steps = 0

    def freeze_token(self) -> None:
        """The encoder gets no RL gradients; recon only when ae_finetune is on."""
        self.train_token = False
        if self.token_ae is not None and not self.ae_finetune:
            self.token_ae.requires_grad_(False)

    def set_critic_frozen(self, frozen: bool) -> None:
        """Online arm 2 freezes the offline-pretrained critic; arm 3 keeps it learning."""
        if self.external_critic:
            return  # an external critic is frozen regardless
        self.critic_frozen = frozen
        self.critic.requires_grad_(not frozen)

    def set_ae_finetune(self, enabled: bool) -> None:
        self.ae_finetune = enabled
        if self.token_ae is not None and enabled:
            self.token_ae.requires_grad_(True)

    # --- flow inference -----------------------------------------------------

    def _integrate(
        self, net: FlowActor, state: torch.Tensor, noise: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        """Euler-integrate the velocity from t=0 (noise) to t=1 (corrected chunk).

        In compose mode the network outputs only the guidance G; the base OT
        velocity (reference - noise) is added so V = v_base + G reproduces the
        reference exactly when G = 0.
        """
        x = noise
        dt = 1.0 / self.flow_steps
        base = (reference - noise) if self.compose else 0.0
        for i in range(self.flow_steps):
            t = torch.full((state.shape[0],), i * dt, device=state.device)
            x = x + dt * (net(state, x, t, reference) + base)
        return x

    def act(self, state: np.ndarray, reference: np.ndarray, explore: bool = False) -> np.ndarray:
        """One corrected chunk per state row, conditioned on the VLA reference."""
        net = self.actor if explore else self.actor_target
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            ref = torch.as_tensor(reference, dtype=torch.float32, device=self.device)
            if s.ndim == 1:
                s = s.unsqueeze(0)
                ref = ref.unsqueeze(0)
            noise = torch.randn(s.shape[0], self.chunk_dim, device=self.device)
            return self._integrate(net, s, noise, ref).cpu().numpy()

    # --- losses -------------------------------------------------------------

    def _recon_loss(self, batch: dict) -> tuple[torch.Tensor | None, dict[str, float]]:
        if not self.ae_finetune or self.token_ae is None:
            return None, {}
        token_bs = int(getattr(self.cfg, "token_batch_size", 8))
        if "tokens" in batch:
            from .consensusflow import pad_token_batch

            tok, msk = pad_token_batch(list(batch["tokens"])[:token_bs], self.device)
        elif self.token_corpus is not None:
            tok, msk = self.token_corpus.sample(token_bs, self.device, self.token_corpus_rng)
        else:
            return None, {}
        loss, stats = self.token_ae.reconstruction_loss(tok, msk)
        return loss, {f"ae_{k}": v for k, v in stats.items()}

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One joint step: flow-matching BC to the VLA chunk + frozen-critic ascent."""
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(batch["reference"], dtype=torch.float32, device=self.device)

        x0 = torch.randn_like(reference)
        t = torch.rand(state.shape[0], device=self.device)
        x_t = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * reference
        target_v = reference - x0
        guidance = self.actor(state, x_t, t, reference)
        # compose: the base OT velocity (target_v) is added to G, so BC drives G -> 0
        # (the composed flow then reproduces the reference); the Q term grows G online.
        pred_v = guidance + target_v if self.compose else guidance
        bc_loss = F.mse_loss(pred_v, target_v)

        action = self._integrate(self.actor, state, torch.randn_like(reference), reference)
        actor_q = self.critic.min_q(state, action)
        deviation = (action - reference).pow(2).sum(dim=-1)
        q_loss = -actor_q.mean()
        # The anchor stays on in every phase; actor_coef only gates the Q ascent.
        # Pretrain runs with flow_actor_coef=0 so the flow starts as a VLA copy
        # (V21's BC-pretrain shape); online turns the frozen-critic ascent on.
        anchor_loss = (self.beta * deviation).mean()
        actor_loss = q_loss + anchor_loss

        recon_loss, recon_stats = self._recon_loss(batch)
        loss = self.bc_coef * bc_loss + self.actor_coef * q_loss + anchor_loss
        if recon_loss is not None:
            loss = loss + self.ae_coef * recon_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.actor_steps += 1
        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.mul_(self.ema).add_((1.0 - self.ema) * p)

        return {
            "bc_loss": bc_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_q": actor_q.mean().item(),
            "actor_ref_rmse": float(deviation.mean().div(self.chunk_dim).sqrt().item()),
            "total_loss": loss.item(),
            **recon_stats,
        }

    # --- learner interface --------------------------------------------------

    def critic_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """TD on the twin critic (Eq. 3), bootstrapped through the flow actor.

        A no-op once the critic is frozen (external critic, or online arm 2). The
        target is clipped to [0, 1]: a single terminal +1 means the true return
        never leaves that range, and clipping removes the overestimation spiral.
        """
        if self.critic_frozen:
            return {}
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        next_reference = torch.as_tensor(batch["next_reference"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_action = self._integrate(
                self.actor_target, next_state, torch.randn_like(next_reference), next_reference
            )
            next_q = self.critic_target.min_q(next_state, next_action)
            target = reward + (1.0 - done) * self.chunk_discount * next_q
            if self.clip_target:
                target = target.clamp(*self.q_bounds)

        q1, q2 = self.critic(state, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.critic_steps += 1
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return {
            "critic_loss": critic_loss.item(),
            "q_mean": q1.mean().item(),
            "target_mean": target.mean().item(),
        }

    def actor_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        return self.update(batch)

    def actor_bc_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        stats = self.update(batch)
        stats["actor_bc_loss"] = stats["bc_loss"]  # the pretrain log line reads this key
        stats["actor_bc_rmse"] = float(np.sqrt(max(stats["bc_loss"], 0.0)))
        return stats

    # --- diagnostics --------------------------------------------------------

    @torch.no_grad()
    def probe(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Does the frozen critic prefer the actor's sample over the VLA's own chunk?"""
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(batch["reference"], dtype=torch.float32, device=self.device)
        sampled = self._integrate(self.actor_target, state, torch.randn_like(reference), reference)
        q_actor = self.critic.min_q(state, sampled)
        q_ref = self.critic.min_q(state, reference)
        deviation = (sampled - reference).pow(2).mean(dim=-1).sqrt()
        return {
            "probe/q_actor": q_actor.mean().item(),
            "probe/q_reference": q_ref.mean().item(),
            "probe/q_gap": (q_actor - q_ref).mean().item(),
            "probe/actor_deviation": deviation.mean().item(),
        }

    def q_values(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            a = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            return self.critic.min_q(s, a).cpu().numpy()

    # --- checkpoints --------------------------------------------------------

    def save(self, path: str) -> None:
        atomic_torch_save(
            {
                "algorithm": "flow_rlt",
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "token_ae": self.token_ae.state_dict() if self.token_ae is not None else None,
                "token_ae_config": self.token_ae.config if self.token_ae is not None else None,
                "critic_steps": self.critic_steps,
                "actor_steps": self.actor_steps,
                "config": vars(self.cfg),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_target.load_state_dict(ckpt.get("actor_target", ckpt["actor"]))
        # An external frozen critic keeps its own weights; a learned one resumes.
        if not self.external_critic and "critic" in ckpt:
            self.critic.load_state_dict(ckpt["critic"])
            self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
        if self.token_ae is not None and ckpt.get("token_ae") is not None:
            self.token_ae.load_state_dict(ckpt["token_ae"])
        self.critic_steps = int(ckpt.get("critic_steps", 0))
        self.actor_steps = int(ckpt.get("actor_steps", 0))
