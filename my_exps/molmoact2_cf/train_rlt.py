"""Training steps for RLT-Consensus CF-VLA (token recon, chunk TD, actor, CF guide)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from rlt_models import (
    MolmoAct2RLTCF,
    bootstrap_scale,
    chunk_return,
    normalized_grad_target,
)


def build_rlt_optimizers(
    model: MolmoAct2RLTCF,
    lr_token: float = 1e-4,
    lr_critic: float = 3e-4,
    lr_actor: float = 1e-4,
    lr_guide: float = 1e-4,
    lr_alpha: float = 1e-4,
) -> dict[str, torch.optim.Optimizer]:
    opts: dict[str, torch.optim.Optimizer] = {
        "token": torch.optim.Adam(model.token_ae.parameters(), lr=lr_token),
        "critic": torch.optim.Adam(model.critic.parameters(), lr=lr_critic),
        "actor": torch.optim.Adam(model.actor.parameters(), lr=lr_actor),
        "alpha": torch.optim.Adam(model.log_alpha.parameters(), lr=lr_alpha),
    }
    if model.guide is not None:
        opts["guide"] = torch.optim.Adam(model.guide.parameters(), lr=lr_guide)
    return opts


def token_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    model.train()
    loss, info = model.token_ae.reconstruction_loss(batch["tokens"], batch.get("attention_mask"))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.token_ae.parameters(), 1.0)
    opt.step()
    return {"token_recon_loss": float(loss.detach()), "z_norm": float(info["z_rl"].detach().norm(dim=-1).mean())}


def _batch_state(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
    *,
    next_state: bool = False,
    detach_token: bool = True,
    use_target: bool = False,
) -> torch.Tensor:
    # ChunkReplay stores precomputed current/next z values.  A target encoder is
    # only applicable when the batch contains raw tokens; z batches must stay on
    # their explicit next_z path during TD bootstrapping.
    if "z" in batch:
        z = batch["next_z" if next_state else "z"]
        if detach_token:
            z = z.detach()
        proprio = batch["next_proprio" if next_state else "proprio"]
        return model.encode_state_from_z(z, proprio)
    tokens = batch["next_tokens" if next_state else "tokens"]
    mask = batch.get("next_attention_mask" if next_state else "attention_mask")
    proprio = batch["next_proprio" if next_state else "proprio"]
    return model.encode_state(
        tokens,
        proprio,
        mask,
        use_target=use_target,
        detach_token=detach_token,
    )


def critic_td_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    gamma: float = 0.99,
    mc_coef: float = 0.1,
    cql_coef: float = 0.1,
    cql_n_actions: int = 4,
    cql_action_radius: float = 0.2,
    ref_dropout: float = 0.5,
    rank_coef: float = 1.0,
    rank_margin: float = 0.05,
    rank_noise: float = 0.08,
    far_rank_coef: float = 0.5,
    far_rank_noise: float = 0.35,
    shuffle_rank_coef: float = 0.5,
    target_noise: float = 0.02,
) -> dict[str, float]:
    model.train()
    # Critic may receive gradients into the live token encoder when tokens are present
    # and tune_token_online is set; otherwise use precomputed z with stop-grad.
    detach_token = not model.tune_token_online
    state = _batch_state(model, batch, detach_token=detach_token, use_target=False)
    with torch.no_grad():
        next_state = _batch_state(model, batch, next_state=True, detach_token=True, use_target=True)
        next_ref = model.normalize_action(batch["next_reference_actions"])
        b = next_ref.shape[0]
        present = (torch.rand(b, device=next_ref.device) > ref_dropout).float()
        next_act, _ = model.actor_chunk(
            next_state,
            next_ref,
            deterministic=True,
            apply_guide=False,
            reference_present=present,
        )
        # TD3-style target policy smoothing so bootstrap sees nearby actions.
        if target_noise > 0.0:
            next_act = next_act + target_noise * torch.randn_like(next_act)
            next_act = next_act.clamp(-2.0, 2.0)
        q_next = model.q_min_chunk(next_state, next_act, target=True)
        boot = bootstrap_scale(gamma, model.chunk_size, batch["action_mask"], batch["terminal"])
        r_sum = chunk_return(batch["rewards"], gamma, batch["action_mask"])
        y_td = (r_sum + boot * q_next).clamp(0.0, 1.0)
        y_mc = batch["mc_return"].clamp(0.0, 1.0)

    a = model.normalize_action(batch["executed_actions"])
    qs = model.q_chunk(state, a)  # (E,B)
    td = ((qs - y_td.unsqueeze(0)) ** 2).mean()
    mc = ((qs - y_mc.unsqueeze(0)) ** 2).mean()
    flat_a = a.reshape(a.shape[0], -1)
    cql, cql_info = model.critic.cql_penalty(
        state.detach() if detach_token else state,
        flat_a.detach(),
        n_actions=cql_n_actions,
        coef=cql_coef,
        action_radius=cql_action_radius,
        far_scale=1.0,
    )
    # Action-ranking suite: v6 local noise alone left |Q(a)-Q(a+ε)| ~ 1e-3.
    # Add far noise + cross-batch shuffled actions so Q must depend on a.
    rank = a.new_zeros(())
    q_gap = a.new_zeros(())
    w = y_mc.detach().clamp(0.0, 1.0)
    q_pos = model.q_min_chunk(state, a)
    if rank_coef > 0.0 and rank_noise > 0.0:
        a_neg = a.detach() + float(rank_noise) * torch.randn_like(a)
        q_neg = model.q_min_chunk(state, a_neg)
        rank = rank + float(rank_coef) * (F.relu(q_neg - q_pos + float(rank_margin)) * w).mean()
        q_gap = q_gap + (q_pos - q_neg).detach().mean()
    if far_rank_coef > 0.0 and far_rank_noise > 0.0:
        a_far = a.detach() + float(far_rank_noise) * torch.randn_like(a)
        q_far = model.q_min_chunk(state, a_far)
        rank = rank + float(far_rank_coef) * (
            F.relu(q_far - q_pos + float(rank_margin)) * w
        ).mean()
        q_gap = q_gap + (q_pos - q_far).detach().mean()
    if shuffle_rank_coef > 0.0 and a.shape[0] > 1:
        perm = torch.randperm(a.shape[0], device=a.device)
        a_shuf = a.detach()[perm]
        q_shuf = model.q_min_chunk(state, a_shuf)
        rank = rank + float(shuffle_rank_coef) * (
            F.relu(q_shuf - q_pos + float(rank_margin)) * w
        ).mean()
        q_gap = q_gap + (q_pos - q_shuf).detach().mean()
    loss = td + float(mc_coef) * mc + cql + rank
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = list(model.critic.parameters())
    if model.tune_token_online:
        params += list(model.token_ae.encoder.parameters())
    nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    model.soft_update_targets(0.005)
    return {
        "q_td_loss": float(td.detach()),
        "q_mc_loss": float(mc.detach()),
        "cql_loss": float(cql_info["cql_loss"]),
        "q_rank_loss": float(rank.detach()),
        "q_rank_gap": float(q_gap.detach()),
        "q_mean": float(qs.mean().detach()),
        "q_std": float(qs.std(unbiased=False).detach()),
        "q_target": float(y_td.mean().detach()),
    }


def actor_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    beta: float = 1.0,
    target_divergence: float = 0.0025,
    ref_dropout: float = 0.5,
) -> dict[str, float]:
    model.train()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        ref = model.normalize_action(batch["reference_actions"])
    b = ref.shape[0]
    present = (torch.rand(b, device=ref.device) > ref_dropout).float()
    act, info = model.actor_chunk(
        state,
        ref,
        deterministic=True,
        apply_guide=False,
        reference_present=present,
    )
    q = model.q_min_chunk(state, act)
    with torch.no_grad():
        base_q = model.q_min_chunk(state, ref)
    adv = q - base_q
    # Regularize toward the *original* reference even under dropout.
    mask = batch["action_mask"].unsqueeze(-1)
    diff = (act - ref.detach()) * mask
    residual_mse = (diff**2).sum() / mask.sum().clamp_min(1.0) / act.shape[-1]
    alpha = model.log_alpha()
    trust = alpha.detach() * residual_mse
    actor_loss = -adv.mean() + float(beta) * residual_mse + trust
    opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
    opt.step()

    constraint = (residual_mse - target_divergence).detach()
    alpha_loss = -model.log_alpha.log_alpha * constraint
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()
    return {
        "actor_loss": float(actor_loss.detach()),
        "actor_adv": float(adv.mean().detach()),
        "residual_mse": float(residual_mse.detach()),
        "alpha": float(alpha.detach()),
    }


def guide_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    beta: float = 1.0,
) -> dict[str, float]:
    if model.guide is None:
        return {"guide_loss": 0.0, "guide_adv": 0.0}
    model.train()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        ref = model.normalize_action(batch["reference_actions"])
        actor_mean, ainfo = model.actor_chunk(state, ref, deterministic=True, apply_guide=False)

    # Build normalized ensemble gradient target around the actor mean.
    grads = []
    for head in model.critic.critics:
        a = actor_mean.detach().requires_grad_(True)
        flat = a.reshape(a.shape[0], -1)
        q = head(state.detach(), flat)
        g = torch.autograd.grad(q.sum(), a, retain_graph=False)[0]
        grads.append(g.detach())
    target = normalized_grad_target(grads)

    guided, g_delta = model.guide.guide(state.detach(), ref, actor_delta=ainfo["actor_delta"])
    # Match guide direction to consensus gradient via cosine/MSE on deltas.
    # Scale target to guide magnitude budget.
    t_flat = target.reshape(target.shape[0], -1)
    t_unit = t_flat / t_flat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    pred = g_delta.reshape(g_delta.shape[0], -1)
    # Encourage guide delta to align with unit consensus grad, with small magnitude.
    align = 1.0 - F.cosine_similarity(pred, t_unit, dim=-1).mean()
    q = model.q_min_chunk(state.detach(), guided)
    with torch.no_grad():
        base_q = model.q_min_chunk(state.detach(), ref)
    adv = q - base_q
    mag = (g_delta**2).mean()
    loss = -adv.mean() + align + float(beta) * mag
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.guide.parameters(), 1.0)
    opt.step()
    return {
        "guide_loss": float(loss.detach()),
        "guide_adv": float(adv.mean().detach()),
        "guide_align": float(align.detach()),
        "guide_mse": float(mag.detach()),
    }


def predicted_lcb_advantage(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
) -> float:
    """Ensemble lower-confidence-bound advantage of guided/actor chunk vs reference."""
    model.eval()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        ref = model.normalize_action(batch["reference_actions"])
        act, _ = model.actor_chunk(
            state,
            ref,
            deterministic=True,
            apply_guide=model.guide is not None,
        )
        qs_act = model.q_chunk(state, act)  # (E,B)
        qs_ref = model.q_chunk(state, ref)
        # LCB ≈ mean - std across ensemble
        adv = qs_act - qs_ref
        mu = adv.mean(dim=0)
        std = adv.std(dim=0, unbiased=False)
        lcb = mu - std
        return float(lcb.mean().detach())


def action_sensitivity(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
    *,
    noise: float = 0.03,
) -> float:
    """Mean |Q(a)-Q(a+ε)| — used to refuse deploy when the critic ignores actions."""
    model.eval()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        a = model.normalize_action(batch["executed_actions"])
        q0 = model.q_min_chunk(state, a)
        q1 = model.q_min_chunk(state, a + float(noise) * torch.randn_like(a))
        return float((q0 - q1).abs().mean().detach())


def critic_is_healthy(model: MolmoAct2RLTCF, batch: dict[str, torch.Tensor]) -> bool:
    model.eval()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True)
        a = model.normalize_action(batch["executed_actions"])
        q = model.q_chunk(state, a)
        if not torch.isfinite(q).all():
            return False
        if model.bounded_critic and ((q < -1e-3) | (q > 1.0 + 1e-3)).any():
            return False
        # Reject near-constant critics (state-only collapse of the Q surface).
        if float(q.std(unbiased=False).detach()) < 1e-3:
            return False
        return True
