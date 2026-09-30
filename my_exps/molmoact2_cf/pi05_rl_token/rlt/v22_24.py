"""V22_24: ConsensusFlow losses on the tested V22/V23 (flow_rlt cf_ae) kettle recipe.

The V22/V23 paper method (my_exps/molmoact2_cf/V22_V23_METHODS.md) with the actor's
improvement channel split in two, after the ConsensusFlow method card
(ConsensusFlow.tex / ConsensusFlow_supplement.tex):

    V   reference-conditioned flow corrector v(s, x_t, t, a_ref). Trained by
        flow-matching BC to the frozen pi0.5 chunk, the beta endpoint anchor on
        its own unroll (the stabilizer that fixed the first V22), and -- stage 1
        only -- the one-step guided lookahead
        -mean_k Q_k(s, x_t + (v + sg(G)) * step, t + 1/N). This is the channel
        that absorbs the guidance signal into the base field.
    G   guidance student W(s, x_t, t) (the RLT actor MLP), trained ONLY by
        common-scale-normalized distillation of target-critic gradients,
        z_k = g_k / (|g_k| + c * m_B), one of the K=10 heads sampled per row.
        Deployed through the paper's radial clip, trust-weighted conflict
        contraction against sg(v), residual damping, and the lambda * t gate.
    Q   10-head ensemble over (s, x, t) -- the RLT critic of the paper method,
        now time-conditioned. Trained by reverse-state TD: the critic is
        evaluated at (x^t, t) reached by integrating v + G backwards from the
        executed chunk (delta ~ U[0,1] or k/N with equal probability), target
        clipped to [0, 1], bootstrapped at fresh noise (s', x0', 0).

Stage 0 (cf_actor_coef=0 online) is the supplement's "BC-v / distilled-G"
routing: V gets BC + anchor only, so all value improvement must arrive through
the distilled G, and the paired guidance_coef=0 held-out eval measures G's
sample-time contribution. Stage 1 (cf_actor_coef=1) is the full joint method.

Identity at init: V is BC + anchor pretrained into a tight specialist copy
(cf_actor_coef=0 offline, exactly the tested AC-pretrain recipe), and W's head
is zero-initialised so G == 0 until the critic ensemble teaches it. The
deployed policy integrates v + G from noise; inference never evaluates a
critic.
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


class V2224Agent:
    """Flow corrector V + distilled guidance G + 10-head timed ensemble critic."""

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
        # Weight on the one-step guided lookahead -Q. 0 (stage 0) routes all
        # value improvement through the distilled guide; 1 (stage 1) is the
        # full joint method. Offline pretrain always runs with 0 (the tested
        # flow_actor_coef=0 recipe), so both stages share one pretrain.
        self.actor_coef = float(getattr(cfg, "cf_actor_coef", 1.0))
        self.beta = float(getattr(cfg, "beta", 1.0))
        self.chunk_discount = cfg.gamma**chunk
        self.q_bounds = (0.0, 1.0)
        self.tau = float(getattr(cfg, "tau", 0.005))
        self.clip_target = bool(getattr(cfg, "clip_target", True))
        # Shared-task arm: W and Q also read the VLA reference chunk. The
        # reference is the specialist's plan, so it identifies the task far
        # more directly than the 256-dim z bottleneck (trained for
        # reconstruction, not discrimination) can.
        self.ref_conditioned = bool(getattr(cfg, "cf_ref_conditioned", False))
        ref_dim = chunk_dim if self.ref_conditioned else 0

        self.actor = FlowActor(state_dim, chunk_dim, hidden, n_layers, ref_dim=chunk_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic = EnsembleCritic(
            state_dim, chunk_dim, hidden, n_layers, ensemble, layer_norm=True, ref_dim=ref_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.guidance = Guidance(state_dim, chunk_dim, hidden, n_layers, ref_dim=ref_dim).to(self.device)
        # Exact G == 0 at initialisation: the deployed policy starts as V's
        # specialist copy and the guide grows only under distillation.
        head = self.guidance.net[-1]
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)

        self.token_ae = token_ae.to(self.device) if token_ae is not None else None
        self.train_token = False  # z_rl always comes from the frozen on-disk encoder
        self.ae_finetune = bool(getattr(cfg, "ae_finetune", False))
        self.token_corpus = None
        self.token_corpus_rng = np.random.default_rng(cfg.seed)
        if self.token_ae is not None:
            self.z_dim = self.token_ae.z_dim

        actor_params = list(self.actor.parameters()) + list(self.guidance.parameters())
        if self.token_ae is not None:
            actor_params += list(self.token_ae.parameters())
        self.opt = torch.optim.Adam(actor_params, lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.critic_frozen = False
        self.critic_steps = 0
        self.actor_steps = 0

    # --- flags the trainer sets ---------------------------------------------

    def freeze_token(self) -> None:
        """The encoder gets no RL gradients; recon only when ae_finetune is on."""
        self.train_token = False
        if self.token_ae is not None and not self.ae_finetune:
            self.token_ae.requires_grad_(False)

    def set_ae_finetune(self, enabled: bool) -> None:
        self.ae_finetune = enabled
        if self.token_ae is not None and enabled:
            self.token_ae.requires_grad_(True)

    def set_critic_frozen(self, frozen: bool) -> None:
        """Optional arm flag (flow_freeze_critic_online); the default keeps TD on."""
        self.critic_frozen = frozen
        self.critic.requires_grad_(not frozen)

    def set_guidance_coef(self, coef: float) -> None:
        """Eval-time lambda override: 0 switches the guide off (paired intervention)."""
        self.guidance_coef = float(coef)

    # --- the guided field ----------------------------------------------------

    def _project_unit_ball(self, w: torch.Tensor) -> torch.Tensor:
        w_norm = w.norm(dim=-1, keepdim=True)
        return w * torch.minimum(torch.ones_like(w_norm), 1.0 / (w_norm + 1e-6))

    def _behavior_safe_direction(self, w: torch.Tensor, behavior_velocity: torch.Tensor) -> torch.Tensor:
        """Radial clip, trust-weighted conflict contraction, residual damping."""
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

    def guidance_field(
        self,
        state: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        velocity: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """G_phi(s, x, t) = lambda * t * u_damp, the bounded deployed guidance."""
        w = self.guidance(state, x, t, reference)
        safe = self._behavior_safe_direction(w, velocity)
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        return self.guidance_coef * t * safe

    def _unroll(
        self, state: torch.Tensor, noise: torch.Tensor, reference: torch.Tensor, *, guided: bool, target: bool
    ) -> torch.Tensor:
        """Euler-integrate from t=0 (noise) to t=1 (chunk). t convention: 0 noise, 1 data."""
        net = self.actor_target if target else self.actor
        x = noise
        dt = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full((state.shape[0],), i * dt, device=state.device, dtype=state.dtype)
            v = net(state, x, t, reference)
            if guided:
                v = v + self.guidance_field(state, x, t, v, reference)
            x = x + dt * v
        return x

    # --- acting ---------------------------------------------------------------

    @torch.no_grad()
    def act(self, state: np.ndarray, reference: np.ndarray, explore: bool = False) -> np.ndarray:
        """One corrected chunk per state row: integrate v + G from fresh noise."""
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        ref = torch.as_tensor(reference, dtype=torch.float32, device=self.device)
        if s.ndim == 1:
            s = s.unsqueeze(0)
            ref = ref.unsqueeze(0)
        noise = torch.randn(s.shape[0], self.chunk_dim, device=self.device)
        return self._unroll(s, noise, ref, guided=True, target=not explore).cpu().numpy()

    @torch.no_grad()
    def q_values(self, state: np.ndarray, action: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        ref = None if reference is None else torch.as_tensor(reference, dtype=torch.float32, device=self.device)
        t = torch.ones(s.shape[0], 1, device=self.device)
        mean, _ = self.critic.mean_std(s, a, t, ref)
        return mean.cpu().numpy()

    # --- critic: reverse-state TD ----------------------------------------------

    def critic_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """TD on the 10-head timed critic at reverse-integrated flow states.

        The target bootstraps at fresh noise (s', x0', 0) -- no actor unroll in
        the backup -- and is clipped to [0, 1]: one terminal +1 means the true
        return never leaves that range.
        """
        if self.critic_frozen:
            return {}
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        action = torch.as_tensor(batch["action"], dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(batch["reference"], dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=self.device)
        next_reference = torch.as_tensor(batch["next_reference"], dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device)
        n = state.shape[0]

        with torch.no_grad():
            next_noise = torch.randn(n, self.chunk_dim, device=self.device)
            t0 = torch.zeros(n, 1, device=self.device)
            next_q, next_std = self.critic_target.mean_std(next_state, next_noise, t0, next_reference)
            bootstrap = next_q - self.rho * next_std
            target = reward + (1.0 - done) * self.chunk_discount * bootstrap
            if self.clip_target:
                target = target.clamp(*self.q_bounds)

            # Reverse states: from (a, 1), integrate v + G backwards by a random
            # fraction delta -- U[0,1] or k/N with equal probability, per row.
            uniform = torch.rand(n, 1, device=self.device)
            discrete = torch.randint(0, self.flow_steps + 1, (n, 1), device=self.device).float() / self.flow_steps
            delta = torch.where(torch.rand(n, 1, device=self.device) < 0.5, uniform, discrete)
            dt = delta / self.flow_steps
            x = action.clone()
            t = torch.ones(n, 1, device=self.device)
            for _ in range(self.flow_steps):
                v = self.actor(state, x, t, reference)
                x = x - (v + self.guidance_field(state, x, t, v, reference)) * dt
                t = (t - dt).clamp(min=0.0)

        q = self.critic(state, x, t, reference)  # (K, B)
        err = target.unsqueeze(0) - q
        weight = torch.where(err >= 0, self.expectile, 1.0 - self.expectile)
        critic_loss = (weight * err.pow(2)).mean()

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.critic_steps += 1
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1.0 - self.tau).add_(self.tau * p)

        return {
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(q.mean().item()),
            "target_mean": float(target.mean().item()),
            "reverse_t_mean": float(t.mean().item()),
        }

    # --- actor + guidance -------------------------------------------------------

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

    def actor_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """One joint step for V and G: BC + anchor + lookahead + distillation."""
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(batch["reference"], dtype=torch.float32, device=self.device)
        n = state.shape[0]
        dt = 1.0 / self.flow_steps

        # BC points along the OT path to the reference chunk.
        x0 = torch.randn(n, self.chunk_dim, device=self.device)
        t_bc = torch.rand(n, 1, device=self.device)
        x_t = (1.0 - t_bc) * x0 + t_bc * reference
        target_velocity = reference - x0
        velocity = self.actor(state, x_t, t_bc, reference)
        bc_loss = F.mse_loss(velocity, target_velocity)

        # One-step guided lookahead into the online ensemble mean (stage 1).
        guide = self.guidance_field(state, x_t, t_bc, velocity, reference).detach()
        step = torch.minimum(torch.full_like(t_bc, dt), 1.0 - t_bc)
        lookahead = x_t + (velocity + guide) * step
        t_plus = (t_bc + dt).clamp(max=1.0)
        q_plus = self.critic(state, lookahead, t_plus, reference).mean(dim=0)
        lookahead_loss = -q_plus.mean()

        # The anchor stays on V's own unguided unroll: V alone remains a tight
        # specialist copy, so the guide is the only sanctioned deviation channel
        # and V never learns to cancel it.
        unrolled = self._unroll(state, torch.randn(n, self.chunk_dim, device=self.device), reference,
                                guided=False, target=False)
        anchor_loss = (self.beta * (unrolled - reference).pow(2).sum(dim=-1)).mean()

        # Common-scale-normalized distillation of the target ensemble into W.
        member = torch.randint(0, self.critic.ensemble, (n,), device=self.device)
        x_grad = x_t.detach().requires_grad_(True)
        qs = self.critic_target(state, x_grad, t_bc, reference)
        picked = qs[member, torch.arange(n, device=self.device)]
        (q_grad,) = torch.autograd.grad(picked.sum(), x_grad, create_graph=False)
        q_grad = q_grad.detach()
        grad_norm = q_grad.norm(dim=-1, keepdim=True)
        grad_scale = grad_norm.mean()
        z_k = q_grad / (grad_norm + self.consensus_floor * grad_scale + 1e-6)
        w = self.guidance(state, x_t.detach(), t_bc, reference)
        distill_loss = (w - z_k).pow(2).sum(dim=-1).mean()

        recon_loss, recon_stats = self._recon_loss(batch)
        loss = (
            self.alpha * bc_loss
            + self.actor_coef * lookahead_loss
            + anchor_loss
            + self.distill_coef * distill_loss
        )
        if recon_loss is not None:
            loss = loss + self.ae_coef * recon_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        self.actor_steps += 1
        with torch.no_grad():
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.mul_(self.ema).add_((1.0 - self.ema) * p)

        stats = {
            "bc_loss": float(bc_loss.item()),
            "actor_loss": float(lookahead_loss.item()),
            "actor_q": float(q_plus.mean().item()),
            "anchor_loss": float(anchor_loss.item()),
            "actor_ref_rmse": float((unrolled - reference).pow(2).mean().sqrt().item()),
            "distill_loss": float(distill_loss.item()),
            "w_norm": float(w.norm(dim=-1).mean().item()),
            "q_grad_scale": float(grad_scale.item()),
            "total_loss": float(loss.item()),
            **recon_stats,
        }
        stats.update(self._guidance_stats(state, x_t, t_bc, velocity, reference))
        return stats

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Critic TD + actor step, for callers that run one joint update."""
        stats = self.critic_step(batch)
        stats.update(self.actor_step(batch))
        return stats

    def actor_bc_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        stats = self.actor_step(batch)
        stats["actor_bc_loss"] = stats["bc_loss"]  # the pretrain log line reads this key
        stats["actor_bc_rmse"] = float(np.sqrt(max(stats["bc_loss"], 0.0)))
        return stats

    @torch.no_grad()
    def _guidance_stats(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t_bc: torch.Tensor,
        velocity: torch.Tensor,
        reference: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Geometry of the deployed guide at the BC points: size, alignment, trust."""
        w = self._project_unit_ball(self.guidance(state, x_t, t_bc, reference))
        w_norm = w.norm(dim=-1)
        v_norm = velocity.norm(dim=-1).clamp_min(1e-6)
        cos = (w * velocity).sum(dim=-1) / (w_norm * v_norm + 1e-6)
        g = self.guidance_field(state, x_t, t_bc, velocity, reference)
        return {
            "g_over_v": float((g.norm(dim=-1) / v_norm).mean().item()),
            "g_cos": float(cos.mean().item()),
            "g_anti_frac": float((cos < 0).float().mean().item()),
            "g_trust": float(w_norm.mean().item()),
        }

    # --- diagnostics ---------------------------------------------------------

    @torch.no_grad()
    def probe(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        """Does the critic prefer the guided EMA sample over the VLA's own chunk?"""
        state = torch.as_tensor(batch["state"], dtype=torch.float32, device=self.device)
        reference = torch.as_tensor(batch["reference"], dtype=torch.float32, device=self.device)
        sampled = self._unroll(state, torch.randn_like(reference), reference, guided=True, target=True)
        t1 = torch.ones(state.shape[0], 1, device=self.device)
        q_actor, _ = self.critic.mean_std(state, sampled, t1, reference)
        q_ref, _ = self.critic.mean_std(state, reference, t1, reference)
        deviation = (sampled - reference).pow(2).mean(dim=-1).sqrt()
        return {
            "probe/q_actor": float(q_actor.mean().item()),
            "probe/q_reference": float(q_ref.mean().item()),
            "probe/q_gap": float((q_actor - q_ref).mean().item()),
            "probe/actor_deviation": float(deviation.mean().item()),
        }

    # --- checkpoints -----------------------------------------------------------

    def save(self, path: str) -> None:
        atomic_torch_save(
            {
                "algorithm": "v22_24",
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "guidance": self.guidance.state_dict(),
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
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt.get("critic_target", ckpt["critic"]))
        self.guidance.load_state_dict(ckpt["guidance"])
        if self.token_ae is not None and ckpt.get("token_ae") is not None:
            self.token_ae.load_state_dict(ckpt["token_ae"])
        self.critic_steps = int(ckpt.get("critic_steps", 0))
        self.actor_steps = int(ckpt.get("actor_steps", 0))
