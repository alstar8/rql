"""Training steps for RLT-Consensus CF-VLA (token recon, chunk TD, actor, CF guide)."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from rlt_models import (
    CFGRL_O_POS,
    CFGRL_O_UNCOND,
    DEFAULT_CFGRL_DROPOUT,
    MolmoAct2RLTCF,
    bootstrap_scale,
    chunk_return,
    common_scale_normalize,
    lower_tail_mean,
)

VelocityProvider = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]

AE_NATIVE_HORIZON = 15
AE_NATIVE_ACTION_DIM = 8
AE_PADDED_ACTION_DIM = 32


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


def stochastic_target_critic_gradient(
    model: MolmoAct2RLTCF,
    state: torch.Tensor,
    actions: torch.Tensor,
    *,
    t: torch.Tensor | None = None,
    member: torch.Tensor | None = None,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Sample target members and expose the unnormalized critic-gradient health."""
    action_leaf = actions.detach().requires_grad_(True)
    qs = model.q_chunk(
        state.detach(),
        action_leaf,
        target=True,
        t=t,
    )
    batch_size = action_leaf.shape[0]
    if member is None:
        member = torch.randint(
            0,
            qs.shape[0],
            (batch_size,),
            device=action_leaf.device,
        )
    else:
        member = member.to(device=action_leaf.device, dtype=torch.long)
        if member.shape != (batch_size,):
            raise ValueError(
                f"member must have shape ({batch_size},), got {tuple(member.shape)}"
            )
    head_gradients = []
    for head_index in range(qs.shape[0]):
        gradient = torch.autograd.grad(
            qs[head_index].sum(),
            action_leaf,
            retain_graph=head_index + 1 < qs.shape[0],
        )[0]
        head_gradients.append(gradient.detach().reshape(batch_size, -1))
    gradients = torch.stack(head_gradients, dim=0)
    gather_index = member.view(1, batch_size, 1).expand(
        1,
        batch_size,
        gradients.shape[-1],
    )
    selected_gradient = gradients.gather(0, gather_index).squeeze(0)
    target = common_scale_normalize(selected_gradient)
    if not return_diagnostics:
        return target

    raw_norms = gradients.norm(dim=-1)
    selected_raw_norms = selected_gradient.norm(dim=-1)
    unit = gradients / raw_norms.clamp_min(1e-12).unsqueeze(-1)
    mean_direction = unit.mean(dim=0)
    direction_agreement = mean_direction.norm(dim=-1)
    diagnostics = {
        "critic_gradient_raw_norm_mean": float(raw_norms.mean()),
        "critic_gradient_raw_norm_min": float(raw_norms.min()),
        # Guide skip must use the *selected* members (the actual teacher), not the
        # min over all heads×batch — one dead ensemble head previously zeroed CF.
        "critic_gradient_selected_norm_mean": float(selected_raw_norms.mean()),
        "critic_gradient_selected_norm_min": float(selected_raw_norms.min()),
        "critic_gradient_nonzero_fraction": float(
            (selected_raw_norms > 1e-8).float().mean()
        ),
        "critic_gradient_direction_agreement": float(direction_agreement.mean()),
    }
    return target, diagnostics


def _guide_teacher_is_dead(gradient_diagnostics: dict[str, float]) -> bool:
    """True only when every selected teacher gradient is numerically dead."""
    selected_mean = float(
        gradient_diagnostics.get(
            "critic_gradient_selected_norm_mean",
            gradient_diagnostics.get("critic_gradient_raw_norm_mean", 0.0),
        )
    )
    nonzero = float(
        gradient_diagnostics.get("critic_gradient_nonzero_fraction", 0.0)
    )
    return selected_mean <= 1e-8 or nonzero <= 0.0


def _per_head_rank_loss(
    q_positive: torch.Tensor,
    q_negative: torch.Tensor,
    weights: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Apply ranking to every critic head instead of only an aggregate."""
    if q_positive.shape != q_negative.shape or q_positive.ndim != 2:
        raise ValueError("per-head rank inputs must have matching (E,B) shapes")
    if weights.shape != (q_positive.shape[1],):
        raise ValueError("per-head rank weights must have shape (B,)")
    return (
        F.relu(q_negative - q_positive + float(margin))
        * weights.unsqueeze(0)
    ).mean()


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
    critic_target_use_guide: bool = False,
    actor_cql_coef: float = 0.0,
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
            apply_guide=bool(critic_target_use_guide and model.guide is not None),
            reference_present=present,
        )
        # TD3-style target policy smoothing so bootstrap sees nearby actions.
        if target_noise > 0.0:
            next_act = next_act + target_noise * torch.randn_like(next_act)
            next_act = next_act.clamp(-2.0, 2.0)
        q_next = model.q_lower_tail_chunk(next_state, next_act, target=True)
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
    q_pos_heads = model.q_chunk(state, a)
    q_pos = model.q_lower_tail_chunk(state, a)
    if rank_coef > 0.0 and rank_noise > 0.0:
        a_neg = a.detach() + float(rank_noise) * torch.randn_like(a)
        q_neg_heads = model.q_chunk(state, a_neg)
        q_neg = model.q_lower_tail_chunk(state, a_neg)
        rank = rank + float(rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_neg_heads,
            w,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_neg).detach().mean()
    if far_rank_coef > 0.0 and far_rank_noise > 0.0:
        a_far = a.detach() + float(far_rank_noise) * torch.randn_like(a)
        q_far_heads = model.q_chunk(state, a_far)
        q_far = model.q_lower_tail_chunk(state, a_far)
        rank = rank + float(far_rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_far_heads,
            w,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_far).detach().mean()
    if shuffle_rank_coef > 0.0 and a.shape[0] > 1:
        perm = torch.randperm(a.shape[0], device=a.device)
        a_shuf = a.detach()[perm]
        q_shuf_heads = model.q_chunk(state, a_shuf)
        q_shuf = model.q_lower_tail_chunk(state, a_shuf)
        rank = rank + float(shuffle_rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_shuf_heads,
            w,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_shuf).detach().mean()
    actor_cql = a.new_zeros(())
    if float(actor_cql_coef) > 0.0 and hasattr(model, "actor"):
        with torch.no_grad():
            ref = model.normalize_action(batch["reference_actions"])
            actor_act, _ = model.actor_chunk(
                state.detach(),
                ref,
                deterministic=True,
                apply_guide=False,
            )
        q_actor_heads = model.q_chunk(state, actor_act)
        # Penalize over-optimistic Q on actor-proposed actions vs data.
        actor_cql = (
            torch.logsumexp(q_actor_heads, dim=0) - torch.logsumexp(qs, dim=0)
        ).mean()
    loss = td + float(mc_coef) * mc + cql + rank + float(actor_cql_coef) * actor_cql
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
        "actor_cql_loss": float(actor_cql.detach()),
        "q_rank_loss": float(rank.detach()),
        "q_rank_gap": float(q_gap.detach()),
        "q_mean": float(qs.mean().detach()),
        "q_std": float(qs.std(unbiased=False).detach()),
        "q_target": float(y_td.mean().detach()),
    }


def endpoint_critic_mc_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    rank_coef: float = 1.0,
    rank_margin: float = 0.05,
    rank_noise: float = 0.08,
    far_rank_coef: float = 0.5,
    far_rank_noise: float = 0.35,
    shuffle_rank_coef: float = 0.5,
    actions_already_normalized: bool = False,
) -> dict[str, float]:
    """Fit endpoint values/ranks from compact replay without image encoding."""
    model.train()
    detach_token = not model.tune_token_online
    state = _batch_state(
        model,
        batch,
        detach_token=detach_token,
        use_target=False,
    )
    actions = (
        batch["executed_actions"].float()
        if actions_already_normalized
        else model.normalize_action(batch["executed_actions"])
    )
    t_end = None
    if model.is_flow:
        t_end = torch.ones(
            actions.shape[0],
            1,
            device=actions.device,
            dtype=actions.dtype,
        )
    q_positive_heads = model.q_chunk(state, actions, t=t_end)
    y_mc = batch["mc_return"].clamp(0.0, 1.0)
    mc = ((q_positive_heads - y_mc.unsqueeze(0)) ** 2).mean()
    weights = y_mc.detach()
    rank = actions.new_zeros(())
    q_gap = actions.new_zeros(())
    q_positive = model.q_lower_tail_chunk(state, actions, t=t_end)

    if rank_coef > 0.0 and rank_noise > 0.0:
        local = actions.detach() + float(rank_noise) * torch.randn_like(actions)
        local_heads = model.q_chunk(state, local, t=t_end)
        local_q = model.q_lower_tail_chunk(state, local, t=t_end)
        rank = rank + float(rank_coef) * _per_head_rank_loss(
            q_positive_heads,
            local_heads,
            weights,
            rank_margin,
        )
        q_gap = q_gap + (q_positive - local_q).detach().mean()
    if far_rank_coef > 0.0 and far_rank_noise > 0.0:
        far = actions.detach() + float(far_rank_noise) * torch.randn_like(actions)
        far_heads = model.q_chunk(state, far, t=t_end)
        far_q = model.q_lower_tail_chunk(state, far, t=t_end)
        rank = rank + float(far_rank_coef) * _per_head_rank_loss(
            q_positive_heads,
            far_heads,
            weights,
            rank_margin,
        )
        q_gap = q_gap + (q_positive - far_q).detach().mean()
    if shuffle_rank_coef > 0.0 and actions.shape[0] > 1:
        shuffled = actions.detach()[
            torch.randperm(actions.shape[0], device=actions.device)
        ]
        shuffled_heads = model.q_chunk(state, shuffled, t=t_end)
        shuffled_q = model.q_lower_tail_chunk(state, shuffled, t=t_end)
        rank = rank + float(shuffle_rank_coef) * _per_head_rank_loss(
            q_positive_heads,
            shuffled_heads,
            weights,
            rank_margin,
        )
        q_gap = q_gap + (q_positive - shuffled_q).detach().mean()

    loss = mc + rank
    opt.zero_grad(set_to_none=True)
    loss.backward()
    parameters = list(model.critic.parameters())
    if model.tune_token_online:
        parameters += list(model.token_ae.encoder.parameters())
    nn.utils.clip_grad_norm_(parameters, 1.0)
    opt.step()
    model.soft_update_targets(0.005)
    return {
        "q_td_loss": 0.0,
        "q_mc_loss": float(mc.detach()),
        "q_rank_loss": float(rank.detach()),
        "q_rank_gap": float(q_gap.detach()),
        "q_mean": float(q_positive_heads.mean().detach()),
        "q_std": float(q_positive_heads.std(unbiased=False).detach()),
        "q_target": float(y_mc.mean().detach()),
        "compact_endpoint_update": 1.0,
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
    q_coef: float = 1.0,
    residual_clip: float | None = None,
    advantage_clip: float | None = 0.05,
) -> dict[str, float]:
    """Train residual actor with optional BC-only phase and hard residual ball.

    Soft BC ``beta * ||a - ã||^2`` is the primary trust-region stabilizer
    (RLT paper; ``beta=0`` is their worst ablation). ``q_coef`` scales the
    critic advantage term and should stay secondary to ``beta``.

    ``q_coef=0`` disables critic maximization (phase-0 BC / identity).
    ``residual_clip`` bounds per-element ``|Δ|`` after the actor forward
    as a hard safety ball, not a substitute for ``beta``.
    ``advantage_clip`` truncates the advantage used in the loss.
    """

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
    if residual_clip is not None and float(residual_clip) > 0.0:
        delta = (act - ref.detach()).clamp(
            -float(residual_clip),
            float(residual_clip),
        )
        act = ref.detach() + delta
        info = dict(info)
        info["actor_delta"] = delta
        info["actor_mean"] = act
    q = model.q_lower_tail_chunk(state, act)
    with torch.no_grad():
        base_q = model.q_lower_tail_chunk(state, ref)
    adv = q - base_q
    if advantage_clip is not None and float(advantage_clip) > 0.0:
        adv = adv.clamp(-float(advantage_clip), float(advantage_clip))
    # Regularize toward the *original* reference even under dropout.
    mask = batch["action_mask"].unsqueeze(-1)
    diff = (act - ref.detach()) * mask
    residual_mse = (diff**2).sum() / mask.sum().clamp_min(1.0) / act.shape[-1]
    residual_rms = torch.sqrt(
        (diff**2).sum() / mask.sum().clamp_min(1.0) / act.shape[-1]
    )
    alpha = model.log_alpha()
    trust = alpha.detach() * residual_mse
    actor_loss = (
        -float(q_coef) * adv.mean() + float(beta) * residual_mse + trust
    )
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
        "residual_rms": float(residual_rms.detach()),
        "actor_ref_mse": float(residual_mse.detach()),
        "actor_q_coef": float(q_coef),
        "alpha": float(alpha.detach()),
    }


def guide_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    beta: float = 0.05,
    target_delta_frac: float | None = None,
    guide_on_reference: bool = False,
) -> dict[str, float]:
    # Deprecated compatibility argument. ConsensusFlow learns raw W against z;
    # deployment bounding, not a fixed training magnitude, determines delta.
    del beta, target_delta_frac
    if model.guide is None:
        return {"guide_loss": 0.0, "guide_adv": 0.0}
    model.train()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        ref = model.normalize_action(batch["reference_actions"])
        if guide_on_reference:
            actor_mean = ref
            actor_delta = torch.zeros_like(ref)
        else:
            actor_mean, ainfo = model.actor_chunk(
                state, ref, deterministic=True, apply_guide=False
            )
            actor_delta = ainfo["actor_delta"]

    # One stochastic target-critic member per example avoids opposed-member
    # cancellation while preserving the common-scale target magnitude.
    target, gradient_diagnostics = stochastic_target_critic_gradient(
        model,
        state,
        actor_mean,
        return_diagnostics=True,
    )
    if _guide_teacher_is_dead(gradient_diagnostics):
        return {
            "guide_loss": 0.0,
            "guide_adv": 0.0,
            "guide_update_skipped": 1.0,
            "guide_skip_tiny_critic_gradient": 1.0,
            "guide_on_reference": float(guide_on_reference),
            **gradient_diagnostics,
        }

    w_flat = model.guide.raw_w(state.detach(), ref)
    _guided_ref, g_delta = model.guide.guide(
        state.detach(), ref, actor_delta=actor_delta
    )
    # Additive composition: guide rides on top of the actor residual, or
    # directly on the frozen reference when guide_on_reference=True.
    composed = actor_mean.detach() + g_delta
    distill = F.mse_loss(w_flat, target)
    align = 1.0 - F.cosine_similarity(
        w_flat, target, dim=-1
    ).mean()
    with torch.no_grad():
        q = model.q_lower_tail_chunk(state.detach(), composed)
        base_q = model.q_lower_tail_chunk(state.detach(), actor_mean.detach())
    adv = q - base_q
    mag = (g_delta**2).mean()
    # The conditional-moment identity is the population optimum of the pure
    # squared teacher loss E||W-z_J||². Direct Q maximization or a magnitude
    # penalty on G changes that optimum, so advantage and magnitude are
    # diagnostics only.
    loss = distill
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.guide.parameters(), 1.0)
    opt.step()
    return {
        "guide_loss": float(loss.detach()),
        "guide_adv": float(adv.mean().detach()),
        "guide_update_skipped": 0.0,
        "guide_skip_tiny_critic_gradient": 0.0,
        "guide_align": float(align.detach()),
        "guide_distill": float(distill.detach()),
        "guide_mse": float(mag.detach()),
        "guide_delta_rms": float(mag.detach().sqrt()),
        "w_norm": float(w_flat.norm(dim=-1).mean().detach()),
        "target_norm": float(target.norm(dim=-1).mean().detach()),
        "guide_on_reference": float(guide_on_reference),
        **gradient_diagnostics,
    }


def predicted_lcb_advantage(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
) -> float:
    """Pessimistic ensemble LCB advantage (logging / diagnostics)."""
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
        adv = qs_act - qs_ref
        mu = adv.mean(dim=0)
        std = adv.std(dim=0, unbiased=False)
        lcb = mu - std
        return float(lcb.mean().detach())


def predicted_deploy_advantage(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
) -> float:
    """Lower-tail advantage of deploy action over the frozen reference.

    v8 gated on ensemble LCB (= mean-std), which stayed ≪ τ even when the
    robust actor advantage already cleared the threshold.
    """
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
        t = None
        if model.is_flow:
            t = torch.ones(ref.shape[0], 1, device=ref.device, dtype=ref.dtype)
        q_act = model.q_lower_tail_chunk(state, act, t=t)
        q_ref = model.q_lower_tail_chunk(state, ref, t=t)
        return float((q_act - q_ref).mean().detach())


def predicted_guide_advantage(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
) -> float:
    """Lower-tail advantage of guided actor over unguided actor."""
    if model.guide is None:
        return 0.0
    model.eval()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        ref = model.normalize_action(batch["reference_actions"])
        act_g, _ = model.actor_chunk(
            state, ref, deterministic=True, apply_guide=True
        )
        act_u, _ = model.actor_chunk(
            state, ref, deterministic=True, apply_guide=False
        )
        t = None
        if model.is_flow:
            t = torch.ones(ref.shape[0], 1, device=ref.device, dtype=ref.dtype)
        q_g = model.q_lower_tail_chunk(state, act_g, t=t)
        q_u = model.q_lower_tail_chunk(state, act_u, t=t)
        return float((q_g - q_u).mean().detach())


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
        t = None
        if model.is_flow:
            t = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
        q0 = model.q_lower_tail_chunk(state, a, t=t)
        q1 = model.q_lower_tail_chunk(
            state, a + float(noise) * torch.randn_like(a), t=t
        )
        return float((q0 - q1).abs().mean().detach())


def critic_health_metrics(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
    *,
    sensitivity_noise: float = 0.03,
) -> dict[str, float]:
    """Check every critic head for a usable, outcome-sensitive Q surface."""
    model.eval()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True)
        a = model.normalize_action(batch["executed_actions"])
        t = None
        if model.is_flow:
            t = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
        q = model.q_chunk(state, a, t=t)
        pattern = torch.arange(
            a.numel(),
            device=a.device,
            dtype=a.dtype,
        ).reshape_as(a)
        perturbation = float(sensitivity_noise) * torch.where(
            torch.sin(pattern + 1.0) >= 0.0,
            torch.ones_like(a),
            -torch.ones_like(a),
        )
        q_perturbed = model.q_chunk(state, a + perturbation, t=t)
        sensitivity = (q - q_perturbed).abs().mean(dim=1)
        lower_tail_sensitivity = (
            lower_tail_mean(
                q,
                fraction=model.q_tail_fraction,
                min_heads=model.q_tail_min_heads,
                dim=0,
            )
            - lower_tail_mean(
                q_perturbed,
                fraction=model.q_tail_fraction,
                min_heads=model.q_tail_min_heads,
                dim=0,
            )
        ).abs().mean()
        legacy_min_sensitivity = (
            q.min(dim=0).values - q_perturbed.min(dim=0).values
        ).abs().mean()
        head_std = q.std(dim=1, unbiased=False)
        head_finite = torch.isfinite(q).all(dim=1)
        head_finite = head_finite & torch.isfinite(q_perturbed).all(dim=1)
        if model.bounded_critic:
            head_in_range = (
                (q >= -1e-3) & (q <= 1.0 + 1e-3)
            ).all(dim=1)
            head_in_range = head_in_range & (
                (q_perturbed >= -1e-3) & (q_perturbed <= 1.0 + 1e-3)
            ).all(dim=1)
        else:
            head_in_range = torch.ones(
                q.shape[0],
                device=q.device,
                dtype=torch.bool,
            )
        if model.bounded_critic:
            saturation = ((q <= 1e-3) | (q >= 1.0 - 1e-3)).float().mean(dim=1)
        else:
            saturation = torch.zeros_like(sensitivity)
        min_owner = torch.bincount(
            q.argmin(dim=0),
            minlength=q.shape[0],
        ).float()
        min_owner_concentration = min_owner.max() / max(q.shape[1], 1)

        outcomes = batch.get("success", batch.get("mc_return"))
        separation = torch.full_like(sensitivity, float("nan"))
        has_both_outcomes = False
        if outcomes is not None:
            positive = outcomes.reshape(-1) > 0.5
            negative = ~positive
            has_both_outcomes = bool(positive.any() and negative.any())
            if has_both_outcomes:
                separation = q[:, positive].mean(dim=1) - q[:, negative].mean(dim=1)

        head_finite = (
            head_finite
            & torch.isfinite(sensitivity)
            & torch.isfinite(head_std)
            & torch.isfinite(saturation)
        )
        sensitivity_ok = sensitivity >= 1e-4
        variance_ok = head_std >= 1e-3
        saturation_ok = saturation <= 0.95
        owner_fraction = min_owner / max(q.shape[1], 1)
        owner_ok = owner_fraction <= 0.95
        separation_ok = (
            torch.isfinite(separation) & (separation > 1e-3)
            if has_both_outcomes
            else torch.zeros_like(head_finite)
        )
        per_head_ok = (
            head_finite
            & head_in_range
            & sensitivity_ok
            & variance_ok
            & saturation_ok
            & owner_ok
            & separation_ok
        )
        lower_tail_ok = bool(
            torch.isfinite(lower_tail_sensitivity)
            and lower_tail_sensitivity >= 1e-4
        )
        healthy = bool(per_head_ok.all() and lower_tail_ok)
        metrics = {
            "healthy": float(healthy),
            "head_finite_all": float(bool(head_finite.all())),
            "head_in_range_all": float(bool(head_in_range.all())),
            "head_sensitivity_min": float(sensitivity.min().detach()),
            "head_sensitivity_mean": float(sensitivity.mean().detach()),
            "lower_tail_action_sensitivity": float(
                lower_tail_sensitivity.detach()
            ),
            "legacy_min_action_sensitivity": float(
                legacy_min_sensitivity.detach()
            ),
            "head_std_min": float(head_std.min().detach()),
            "head_saturation_max": float(saturation.max().detach()),
            "outcome_separation_min": (
                float(separation.min().detach()) if has_both_outcomes else float("nan")
            ),
            "has_both_outcomes": float(has_both_outcomes),
            "min_head_owner_concentration": float(
                min_owner_concentration.detach()
            ),
        }
        for index in range(q.shape[0]):
            prefix = f"head_{index}"
            metrics[f"{prefix}_finite"] = float(bool(head_finite[index]))
            metrics[f"{prefix}_in_range"] = float(bool(head_in_range[index]))
            metrics[f"{prefix}_action_sensitivity"] = float(
                sensitivity[index].detach()
            )
            metrics[f"{prefix}_std"] = float(head_std[index].detach())
            metrics[f"{prefix}_saturation"] = float(saturation[index].detach())
            metrics[f"{prefix}_outcome_separation"] = (
                float(separation[index].detach())
                if has_both_outcomes
                else float("nan")
            )
            metrics[f"{prefix}_min_owner_fraction"] = float(
                owner_fraction[index].detach()
            )
            metrics[f"{prefix}_healthy"] = float(bool(per_head_ok[index]))
        return metrics


def critic_is_healthy(model: MolmoAct2RLTCF, batch: dict[str, torch.Tensor]) -> bool:
    return bool(critic_health_metrics(model, batch)["healthy"])


def _flow_reverse_state(
    model: MolmoAct2RLTCF,
    state: torch.Tensor,
    actions_n: torch.Tensor,
    reference_n: torch.Tensor,
    *,
    apply_guide: bool = True,
    velocity_provider: VelocityProvider | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-integrate from (a,1) → (x^t, t) for critic training states."""
    b = state.shape[0]
    device = state.device
    dtype = state.dtype
    # Mix continuous and discrete reverse fractions (paper CF).
    half = b // 2
    d_cont = torch.rand(half, device=device, dtype=dtype)
    d_disc = torch.randint(0, model.flow_steps + 1, (b - half,), device=device).to(dtype) / float(
        model.flow_steps
    )
    delta = torch.cat([d_cont, d_disc], dim=0)
    # Shuffle so continuous/discrete mix isn't ordered.
    delta = delta[torch.randperm(b, device=device)]
    step = (delta / float(model.flow_steps)).view(b, 1, 1)
    x = actions_n.detach().clone()
    t = torch.ones(b, 1, device=device, dtype=dtype)
    s = state.detach()
    ref = reference_n.detach()
    provider = model.flow_velocity if velocity_provider is None else velocity_provider
    for _ in range(model.flow_steps):
        v = provider(s, x, t, ref)
        g = torch.zeros_like(v)
        if apply_guide and model.guide is not None:
            g, _, _ = model.guide.guidance(s, x, t, v.detach())
            g = g.detach()
        x = x - (v.detach() + g) * step
        t = t - step.view(b, 1)
    t = t.clamp(0.0, 1.0)
    return x.detach(), t.detach()


def _ae_batch_contexts(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    batch: dict[str, Any],
    *,
    next_state: bool = False,
) -> list[Any | None]:
    contract = _ae_native_contract(model, ae_backend)
    prefix = "next_" if next_state else ""
    required = (
        f"{prefix}external_cam",
        f"{prefix}wrist_cam",
        f"{prefix}instruction",
        f"{prefix}proprio",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(
            "Molmo AE context batch is missing required fields: "
            + ", ".join(missing)
        )
    external = batch[required[0]]
    wrist = batch[required[1]]
    instructions = batch[required[2]]
    proprio = batch[required[3]]
    terminal = batch.get("terminal")
    contexts: list[Any | None] = []
    for i in range(len(external)):
        if next_state and terminal is not None and bool(terminal[i].detach().cpu().item() > 0.5):
            contexts.append(None)
            continue
        ctx, _ = ae_backend.encode_context(
            external[i],
            wrist[i],
            str(instructions[i]),
            proprio[i].detach().float().cpu().numpy(),
            action_horizon=contract["action_horizon"],
        )
        for name, expected in (
            ("action_horizon", contract["action_horizon"]),
            ("action_dim", contract["action_dim"]),
            ("max_action_dim", contract["max_action_dim"]),
        ):
            actual = int(getattr(ctx, name, -1))
            if actual != expected:
                raise RuntimeError(
                    f"Molmo AE context {name}={actual}, expected {expected}"
                )
        contexts.append(ctx)
    return contexts


def _ae_native_contract(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
) -> dict[str, int]:
    action_contract = getattr(ae_backend, "action_contract", None)
    if not callable(action_contract):
        raise RuntimeError(
            "Molmo AE backend must expose action_contract() for native training"
        )
    raw_contract = action_contract()
    contract = {
        name: int(raw_contract[name])
        for name in ("action_horizon", "action_dim", "max_action_dim")
    }
    expected = {
        "action_horizon": AE_NATIVE_HORIZON,
        "action_dim": AE_NATIVE_ACTION_DIM,
        "max_action_dim": AE_PADDED_ACTION_DIM,
    }
    if contract != expected:
        raise RuntimeError(
            f"Molmo AE native contract {contract} does not match V14 {expected}"
        )
    if (
        int(model.chunk_size) != AE_NATIVE_ACTION_DIM
        or int(model.action_dim) != AE_NATIVE_ACTION_DIM
    ):
        raise RuntimeError(
            "RLT critic/guide projection must be the first native 8x8 chunk"
        )
    return contract


def _require_ae_full_fields(batch: dict[str, Any], *fields: str) -> None:
    missing = [
        field
        for field in fields
        if field not in batch or batch[field] is None
    ]
    if missing:
        raise KeyError(
            "Molmo AE V14 requires recorded full-horizon replay fields; missing: "
            + ", ".join(missing)
        )


def _ae_pad_mask(
    batch_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.ones(
        batch_size,
        1,
        AE_PADDED_ACTION_DIM,
        device=device,
        dtype=torch.bool,
    )
    mask[:, :, :AE_NATIVE_ACTION_DIM] = False
    return mask


def _mask_ae_native(actions: torch.Tensor) -> torch.Tensor:
    if actions.ndim != 3 or actions.shape[1:] != (
        AE_NATIVE_HORIZON,
        AE_PADDED_ACTION_DIM,
    ):
        raise ValueError(
            "Native Molmo AE tensor must have shape "
            f"(B,{AE_NATIVE_HORIZON},{AE_PADDED_ACTION_DIM}), got "
            f"{tuple(actions.shape)}"
        )
    masked = actions.masked_fill(
        _ae_pad_mask(actions.shape[0], device=actions.device),
        0.0,
    )
    if not torch.isfinite(masked).all():
        raise RuntimeError("Non-finite native Molmo AE tensor")
    return masked


def _ae_normalized_full_actions(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    batch: dict[str, Any],
    field: str,
) -> torch.Tensor:
    _ae_native_contract(model, ae_backend)
    _require_ae_full_fields(batch, field)
    raw = torch.as_tensor(batch[field])
    expected = (AE_NATIVE_HORIZON, AE_NATIVE_ACTION_DIM)
    if raw.ndim != 3 or tuple(raw.shape[1:]) != expected:
        raise ValueError(
            f"Replay field {field} must have shape (B,{expected[0]},{expected[1]}), "
            f"got {tuple(raw.shape)}"
        )
    if not torch.isfinite(raw).all():
        raise RuntimeError(f"Replay field {field} contains non-finite raw actions")
    normalized = torch.as_tensor(
        ae_backend.normalize_actions(
            raw.detach().float().cpu().numpy()
        ),
        device=raw.device,
    )
    if normalized.ndim != 3 or tuple(normalized.shape[1:]) != expected:
        raise ValueError(
            f"Normalized replay field {field} must have shape "
            f"(B,{expected[0]},{expected[1]}), got {tuple(normalized.shape)}"
        )
    if not torch.isfinite(normalized).all():
        raise RuntimeError(
            f"Normalized replay field {field} contains non-finite actions"
        )
    padding = normalized.new_zeros(
        normalized.shape[0],
        AE_NATIVE_HORIZON,
        AE_PADDED_ACTION_DIM - AE_NATIVE_ACTION_DIM,
    )
    return _mask_ae_native(torch.cat([normalized, padding], dim=-1))


def _ae_source_native(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    contexts: list[Any | None],
    batch: dict[str, Any],
    field: str,
    *,
    template: torch.Tensor,
) -> torch.Tensor:
    _ae_native_contract(model, ae_backend)
    batch_size = len(contexts)
    if template.shape != (
        batch_size,
        AE_NATIVE_HORIZON,
        AE_PADDED_ACTION_DIM,
    ):
        raise ValueError("Native source template does not match AE batch")
    _require_ae_full_fields(batch, field)
    source = torch.as_tensor(
        batch[field],
        device=template.device,
        dtype=template.dtype,
    )
    if source.shape != template.shape:
        raise ValueError(
            f"Replay field {field} must have shape {tuple(template.shape)}, "
            f"got {tuple(source.shape)}"
        )
    return _mask_ae_native(source)


def _ae_native_velocity(
    ae_backend: Any,
    contexts: list[Any | None],
    trajectory: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    if len(contexts) != trajectory.shape[0]:
        raise ValueError("Molmo AE context and trajectory batch sizes differ")
    rows: list[torch.Tensor] = []
    for index, context in enumerate(contexts):
        if context is None:
            rows.append(torch.zeros_like(trajectory[index : index + 1]))
            continue
        row = ae_backend.velocity(
            context,
            trajectory[index : index + 1],
            t[index : index + 1],
        )
        if row.shape != trajectory[index : index + 1].shape:
            raise ValueError(
                f"Molmo AE velocity row has shape {tuple(row.shape)}, expected "
                f"{tuple(trajectory[index:index + 1].shape)}"
            )
        rows.append(row.to(device=trajectory.device, dtype=trajectory.dtype))
    return _mask_ae_native(torch.cat(rows, dim=0))


def _project_native_chunk(
    model: MolmoAct2RLTCF,
    actions_native: torch.Tensor,
) -> torch.Tensor:
    actions_native = _mask_ae_native(actions_native)
    return actions_native[
        :, : int(model.chunk_size), : int(model.action_dim)
    ].float()


def _native_guide_correction(
    model: MolmoAct2RLTCF,
    state: torch.Tensor,
    trajectory: torch.Tensor,
    t: torch.Tensor,
    velocity: torch.Tensor,
    *,
    detach_guide: bool,
) -> torch.Tensor:
    correction = torch.zeros_like(trajectory)
    if model.guide is None:
        return correction
    compact_x = _project_native_chunk(model, trajectory)
    compact_v = _project_native_chunk(model, velocity)
    compact_g, _, _ = model.guide.guidance(
        state.detach().float(),
        compact_x,
        t.float(),
        compact_v.detach(),
    )
    if detach_guide:
        compact_g = compact_g.detach()
    correction[
        :, : int(model.chunk_size), : int(model.action_dim)
    ] = compact_g.to(dtype=correction.dtype)
    return _mask_ae_native(correction)


def _ae_integrate_native(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    contexts: list[Any | None],
    state: torch.Tensor,
    source_native: torch.Tensor,
    *,
    steps: int,
    apply_guide: bool,
    detach_guide: bool = True,
) -> torch.Tensor:
    """Integrate the complete 15x32 native endpoint before RLT projection."""
    if steps < 1:
        raise ValueError(f"AE integration steps must be positive, got {steps}")
    trajectory = _mask_ae_native(source_native)
    batch_size = trajectory.shape[0]
    dt = 1.0 / float(steps)
    for index in range(steps):
        t = torch.full(
            (batch_size, 1),
            index / float(steps),
            device=trajectory.device,
            dtype=torch.float32,
        )
        velocity = _ae_native_velocity(ae_backend, contexts, trajectory, t)
        correction = torch.zeros_like(velocity)
        if apply_guide and model.guide is not None:
            correction = _native_guide_correction(
                model,
                state,
                trajectory,
                t,
                velocity,
                detach_guide=detach_guide,
            )
        trajectory = _mask_ae_native(
            trajectory + (velocity + correction) * dt
        )
    return trajectory


def _ae_reverse_native_state(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    contexts: list[Any | None],
    state: torch.Tensor,
    endpoint_native: torch.Tensor,
    *,
    apply_guide: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse-flow on full native trajectories, projecting only afterward."""
    batch_size = endpoint_native.shape[0]
    half = batch_size // 2
    continuous = torch.rand(
        half,
        device=endpoint_native.device,
        dtype=torch.float32,
    )
    discrete = torch.randint(
        0,
        model.flow_steps + 1,
        (batch_size - half,),
        device=endpoint_native.device,
    ).float() / float(model.flow_steps)
    delta = torch.cat([continuous, discrete], dim=0)
    delta = delta[torch.randperm(batch_size, device=endpoint_native.device)]
    step = (delta / float(model.flow_steps)).view(batch_size, 1, 1)
    trajectory = _mask_ae_native(endpoint_native.detach().clone())
    t = torch.ones(
        batch_size,
        1,
        device=endpoint_native.device,
        dtype=torch.float32,
    )
    for _ in range(model.flow_steps):
        velocity = _ae_native_velocity(ae_backend, contexts, trajectory, t)
        correction = torch.zeros_like(velocity)
        if apply_guide and model.guide is not None:
            correction = _native_guide_correction(
                model,
                state,
                trajectory,
                t,
                velocity,
                detach_guide=True,
            )
        trajectory = _mask_ae_native(
            trajectory - (velocity.detach() + correction.detach()) * step
        )
        t = t - step.view(batch_size, 1)
    return trajectory.detach(), t.clamp(0.0, 1.0).detach()


def ae_flow_gate_metrics(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    batch: dict[str, Any],
    *,
    sensitivity_noise: float = 0.03,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Paired frozen-base/actor/guide diagnostics under identical native noise."""
    if not model.is_flow:
        raise RuntimeError("ae_flow_gate_metrics requires cf_mode=flow")
    _require_ae_full_fields(
        batch,
        "full_reference_actions",
        "source_native",
    )
    model.eval()
    ae_backend.eval()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        contexts = _ae_batch_contexts(model, ae_backend, batch)
        reference_native = _ae_normalized_full_actions(
            model,
            ae_backend,
            batch,
            "full_reference_actions",
        )
        source_native = _ae_source_native(
            model,
            ae_backend,
            contexts,
            batch,
            "source_native",
            template=reference_native,
        )
        adapter_disabled = getattr(ae_backend, "adapter_disabled", None)
        if not callable(adapter_disabled):
            raise RuntimeError(
                "AE gate requires adapter_disabled() for its frozen base reference"
            )
        with adapter_disabled():
            base_native = _ae_integrate_native(
                model,
                ae_backend,
                contexts,
                state,
                source_native,
                steps=model.flow_steps,
                apply_guide=False,
            )
        actor_native = _ae_integrate_native(
            model,
            ae_backend,
            contexts,
            state,
            source_native,
            steps=model.flow_steps,
            apply_guide=False,
        )
        guided_native = actor_native
        if model.guide is not None:
            guided_native = _ae_integrate_native(
                model,
                ae_backend,
                contexts,
                state,
                source_native,
                steps=model.flow_steps,
                apply_guide=True,
            )
        base = _project_native_chunk(model, base_native)
        actor = _project_native_chunk(model, actor_native)
        guided = _project_native_chunk(model, guided_native)
        t1 = torch.ones(
            actor.shape[0],
            1,
            device=actor.device,
            dtype=actor.dtype,
        )
        actor_pairs = (
            model.q_chunk(state, actor, t=t1)
            - model.q_chunk(state, base, t=t1)
        )
        actor_lcb = actor_pairs.mean(dim=0) - actor_pairs.std(
            dim=0,
            unbiased=False,
        )
        actor_advantage = (
            model.q_lower_tail_chunk(state, actor, t=t1)
            - model.q_lower_tail_chunk(state, base, t=t1)
        )
        guide_lcb = actor.new_zeros(actor.shape[0])
        guide_advantage = actor.new_zeros(actor.shape[0])
        if model.guide is not None:
            guide_pairs = (
                model.q_chunk(state, guided, t=t1)
                - model.q_chunk(state, actor, t=t1)
            )
            guide_lcb = guide_pairs.mean(dim=0) - guide_pairs.std(
                dim=0,
                unbiased=False,
            )
            guide_advantage = (
                model.q_lower_tail_chunk(state, guided, t=t1)
                - model.q_lower_tail_chunk(state, actor, t=t1)
            )
        actor_noise = torch.randn(
            actor.shape,
            device=actor.device,
            dtype=actor.dtype,
            generator=generator,
        )
        actor_perturbed = actor + float(sensitivity_noise) * actor_noise
        actor_sensitivity = (
            model.q_lower_tail_chunk(state, actor, t=t1)
            - model.q_lower_tail_chunk(state, actor_perturbed, t=t1)
        ).abs()
        guide_sensitivity = actor_sensitivity
        if model.guide is not None:
            guide_noise = torch.randn(
                guided.shape,
                device=guided.device,
                dtype=guided.dtype,
                generator=generator,
            )
            guide_perturbed = guided + float(sensitivity_noise) * guide_noise
            guide_sensitivity = (
                model.q_lower_tail_chunk(state, guided, t=t1)
                - model.q_lower_tail_chunk(state, guide_perturbed, t=t1)
            ).abs()
        actor_paired_lcb = float(actor_lcb.mean().detach())
        actor_advantage_value = float(actor_advantage.mean().detach())
        guide_paired_lcb = float(guide_lcb.mean().detach())
        guide_advantage_value = float(guide_advantage.mean().detach())
        actor_sensitivity_value = float(actor_sensitivity.mean().detach())
        guide_sensitivity_value = float(guide_sensitivity.mean().detach())
    return {
        "actor_paired_lcb": actor_paired_lcb,
        "actor_advantage": actor_advantage_value,
        "guide_paired_lcb": guide_paired_lcb,
        "guide_advantage": guide_advantage_value,
        "actor_sensitivity": actor_sensitivity_value,
        "guide_sensitivity": guide_sensitivity_value,
        "sensitivity": actor_sensitivity_value,
        # Backward-compatible gate aliases consumed by train_rlt_online.py.
        "paired_lcb": actor_paired_lcb,
        "q_min_advantage": actor_advantage_value,
    }


def flow_gate_metrics(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
    *,
    sensitivity_noise: float = 0.03,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Gate diagnostics for lightweight flow using one paired source-noise draw."""
    if not model.is_flow:
        raise RuntimeError("flow_gate_metrics requires cf_mode=flow")
    model.eval()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        reference = model.normalize_action(batch["reference_actions"])
        x0 = torch.randn_like(reference)
        actor, _ = model.flow_sample(
            state,
            reference,
            apply_guide=False,
            x0=x0,
        )
        guided = actor
        if model.guide is not None:
            guided, _ = model.flow_sample(
                state,
                reference,
                apply_guide=True,
                x0=x0,
            )
        t1 = torch.ones(
            actor.shape[0],
            1,
            device=actor.device,
            dtype=actor.dtype,
        )
        qs_actor = model.q_chunk(state, actor, t=t1)
        qs_reference = model.q_chunk(state, reference, t=t1)
        actor_pair = qs_actor - qs_reference
        actor_lcb = actor_pair.mean(dim=0) - actor_pair.std(
            dim=0,
            unbiased=False,
        )
        actor_advantage = (
            model.q_lower_tail_chunk(state, actor, t=t1)
            - model.q_lower_tail_chunk(state, reference, t=t1)
        )
        guide_lcb = actor.new_zeros(actor.shape[0])
        guide_advantage = actor.new_zeros(actor.shape[0])
        if model.guide is not None:
            guide_pair = model.q_chunk(state, guided, t=t1) - qs_actor
            guide_lcb = guide_pair.mean(dim=0) - guide_pair.std(
                dim=0,
                unbiased=False,
            )
            guide_advantage = (
                model.q_lower_tail_chunk(state, guided, t=t1)
                - model.q_lower_tail_chunk(state, actor, t=t1)
            )
        actor_noise = torch.randn(
            actor.shape,
            device=actor.device,
            dtype=actor.dtype,
            generator=generator,
        )
        actor_perturbed = actor + float(sensitivity_noise) * actor_noise
        actor_sensitivity = (
            model.q_lower_tail_chunk(state, actor, t=t1)
            - model.q_lower_tail_chunk(state, actor_perturbed, t=t1)
        ).abs()
        guide_sensitivity = actor_sensitivity
        if model.guide is not None:
            guide_noise = torch.randn(
                guided.shape,
                device=guided.device,
                dtype=guided.dtype,
                generator=generator,
            )
            guide_perturbed = guided + float(sensitivity_noise) * guide_noise
            guide_sensitivity = (
                model.q_lower_tail_chunk(state, guided, t=t1)
                - model.q_lower_tail_chunk(state, guide_perturbed, t=t1)
            ).abs()
    return {
        "actor_paired_lcb": float(actor_lcb.mean().detach()),
        "actor_advantage": float(actor_advantage.mean().detach()),
        "guide_paired_lcb": float(guide_lcb.mean().detach()),
        "guide_advantage": float(guide_advantage.mean().detach()),
        "actor_sensitivity": float(actor_sensitivity.mean().detach()),
        "guide_sensitivity": float(guide_sensitivity.mean().detach()),
        "paired_lcb": float(actor_lcb.mean().detach()),
        "q_min_advantage": float(actor_advantage.mean().detach()),
        "sensitivity": float(actor_sensitivity.mean().detach()),
    }


def flow_critic_td_step(
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
    critic_target_use_guide: bool = False,
    actor_cql_coef: float = 0.0,
) -> dict[str, float]:
    """Time-dependent critic TD on reverse-integrated flow states."""
    if not model.is_flow:
        return critic_td_step(
            model,
            opt,
            batch,
            gamma=gamma,
            mc_coef=mc_coef,
            cql_coef=cql_coef,
            cql_n_actions=cql_n_actions,
            cql_action_radius=cql_action_radius,
            ref_dropout=ref_dropout,
            rank_coef=rank_coef,
            rank_margin=rank_margin,
            rank_noise=rank_noise,
            far_rank_coef=far_rank_coef,
            far_rank_noise=far_rank_noise,
            shuffle_rank_coef=shuffle_rank_coef,
            target_noise=target_noise,
            critic_target_use_guide=critic_target_use_guide,
            actor_cql_coef=actor_cql_coef,
        )
    model.train()
    detach_token = not model.tune_token_online
    state = _batch_state(model, batch, detach_token=detach_token, use_target=False)
    a = model.normalize_action(batch["executed_actions"])
    ref = model.normalize_action(batch["reference_actions"])
    with torch.no_grad():
        next_state = _batch_state(model, batch, next_state=True, detach_token=True, use_target=True)
        next_ref = model.normalize_action(batch["next_reference_actions"])
        b = next_ref.shape[0]
        present = (torch.rand(b, device=next_ref.device) > ref_dropout).float()
        next_act, _ = model.actor_chunk(
            next_state,
            next_ref,
            deterministic=True,
            apply_guide=bool(critic_target_use_guide and model.guide is not None),
            reference_present=present,
        )
        if target_noise > 0.0:
            next_act = (next_act + target_noise * torch.randn_like(next_act)).clamp(-2.0, 2.0)
        # Endpoint action-values live at t=1. Bootstrapping at t=0 (noise time)
        # evaluates Q(s', a', 0) which collapses to ~0 and starves TD targets.
        t1 = torch.ones(b, 1, device=next_state.device, dtype=next_state.dtype)
        q_next = model.q_lower_tail_chunk(
            next_state, next_act, target=True, t=t1
        )
        boot = bootstrap_scale(gamma, model.chunk_size, batch["action_mask"], batch["terminal"])
        r_sum = chunk_return(batch["rewards"], gamma, batch["action_mask"])
        y_td = (r_sum + boot * q_next).clamp(0.0, 1.0)
        y_mc = batch["mc_return"].clamp(0.0, 1.0)
        x_t, t = _flow_reverse_state(
            model,
            state,
            a,
            ref,
            apply_guide=bool(critic_target_use_guide and model.guide is not None),
        )

    qs = model.q_chunk(state, x_t, t=t)  # (E,B)
    td = ((qs - y_td.unsqueeze(0)) ** 2).mean()
    # Also fit endpoint action-values at t=1 (deploy / gate / ∇Q near actions).
    t_end = torch.ones(a.shape[0], 1, device=a.device, dtype=a.dtype)
    qs_end = model.q_chunk(state, a, t=t_end)
    td = td + ((qs_end - y_td.unsqueeze(0)) ** 2).mean()
    mc = ((qs - y_mc.unsqueeze(0)) ** 2).mean() + ((qs_end - y_mc.unsqueeze(0)) ** 2).mean()
    flat_x = x_t.reshape(x_t.shape[0], -1)
    cql, cql_info = model.critic.cql_penalty(
        state.detach() if detach_token else state,
        flat_x.detach(),
        t,
        n_actions=cql_n_actions,
        coef=cql_coef,
        action_radius=cql_action_radius,
        far_scale=1.0,
    )
    rank = a.new_zeros(())
    q_gap = a.new_zeros(())
    w = y_mc.detach().clamp(0.0, 1.0)
    # Rank primarily on endpoint actions at t=1 (where deploy decisions live).
    q_pos_heads = model.q_chunk(state, a, t=t_end)
    q_pos = model.q_lower_tail_chunk(state, a, t=t_end)
    rank_base = a
    if rank_coef > 0.0 and rank_noise > 0.0:
        a_neg = rank_base.detach() + float(rank_noise) * torch.randn_like(rank_base)
        q_neg_heads = model.q_chunk(state, a_neg, t=t_end)
        q_neg = model.q_lower_tail_chunk(state, a_neg, t=t_end)
        rank = rank + float(rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_neg_heads,
            w,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_neg).detach().mean()
    if far_rank_coef > 0.0 and far_rank_noise > 0.0:
        a_far = rank_base.detach() + float(far_rank_noise) * torch.randn_like(rank_base)
        q_far_heads = model.q_chunk(state, a_far, t=t_end)
        q_far = model.q_lower_tail_chunk(state, a_far, t=t_end)
        rank = rank + float(far_rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_far_heads,
            w,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_far).detach().mean()
    if shuffle_rank_coef > 0.0 and a.shape[0] > 1:
        perm = torch.randperm(a.shape[0], device=a.device)
        a_shuf = rank_base.detach()[perm]
        q_shuf_heads = model.q_chunk(state, a_shuf, t=t_end)
        q_shuf = model.q_lower_tail_chunk(state, a_shuf, t=t_end)
        rank = rank + float(shuffle_rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_shuf_heads,
            w,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_shuf).detach().mean()
    actor_cql = a.new_zeros(())
    if float(actor_cql_coef) > 0.0:
        with torch.no_grad():
            present = (torch.rand(a.shape[0], device=a.device) > ref_dropout).float()
            actor_end, _ = model.flow_sample(
                state.detach(),
                ref,
                apply_guide=False,
                x0=torch.randn_like(a),
            )
            del present
        q_actor_heads = model.q_chunk(state, actor_end, t=t_end)
        actor_cql = (
            torch.logsumexp(q_actor_heads, dim=0) - torch.logsumexp(qs_end, dim=0)
        ).mean()
    loss = td + float(mc_coef) * mc + cql + rank + float(actor_cql_coef) * actor_cql
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
        "actor_cql_loss": float(actor_cql.detach()),
        "q_rank_loss": float(rank.detach()),
        "q_rank_gap": float(q_gap.detach()),
        "q_mean": float(qs_end.mean().detach()),
        "q_std": float(qs_end.std(unbiased=False).detach()),
        "q_target": float(y_td.mean().detach()),
        "flow_t_mean": float(t.mean().detach()),
    }


def ae_flow_critic_td_step(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    opt: torch.optim.Optimizer,
    batch: dict[str, Any],
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
    critic_target_use_guide: bool = False,
    actor_cql_coef: float = 0.0,
) -> dict[str, float]:
    """AE critic TD from full native reverse states and bootstrap endpoints."""
    del actor_cql_coef
    del ref_dropout
    if not model.is_flow:
        raise RuntimeError("ae_flow_critic_td_step requires cf_mode=flow")
    _require_ae_full_fields(
        batch,
        "full_reference_actions",
        "full_executed_actions",
        "next_full_reference_actions",
        "source_native",
        "next_source_native",
    )
    model.train()
    ae_backend.eval()
    detach_token = not model.tune_token_online
    state = _batch_state(model, batch, detach_token=detach_token, use_target=False)
    contexts = _ae_batch_contexts(model, ae_backend, batch)
    actions_native = _ae_normalized_full_actions(
        model,
        ae_backend,
        batch,
        "full_executed_actions",
    )
    reference_native = _ae_normalized_full_actions(
        model,
        ae_backend,
        batch,
        "full_reference_actions",
    )
    actions = _project_native_chunk(model, actions_native)

    with torch.no_grad():
        next_state = _batch_state(
            model,
            batch,
            next_state=True,
            detach_token=True,
            use_target=True,
        )
        next_contexts = _ae_batch_contexts(
            model,
            ae_backend,
            batch,
            next_state=True,
        )
        next_reference_native = _ae_normalized_full_actions(
            model,
            ae_backend,
            batch,
            "next_full_reference_actions",
        )
        next_source_native = _ae_source_native(
            model,
            ae_backend,
            next_contexts,
            batch,
            "next_source_native",
            template=next_reference_native,
        )
        next_endpoint_native = _ae_integrate_native(
            model,
            ae_backend,
            next_contexts,
            next_state,
            next_source_native,
            steps=model.flow_steps,
            apply_guide=bool(critic_target_use_guide and model.guide is not None),
        )
        next_action = _project_native_chunk(model, next_endpoint_native)
        terminal_mask = batch["terminal"].reshape(-1, 1, 1) > 0.5
        next_action = torch.where(terminal_mask, torch.zeros_like(next_action), next_action)
        if target_noise > 0.0:
            next_action = (
                next_action + float(target_noise) * torch.randn_like(next_action)
            ).clamp(-2.0, 2.0)
        t1 = torch.ones(
            next_action.shape[0],
            1,
            device=next_action.device,
            dtype=next_action.dtype,
        )
        q_next = model.q_lower_tail_chunk(
            next_state,
            next_action,
            target=True,
            t=t1,
        )
        boot = bootstrap_scale(
            gamma,
            model.chunk_size,
            batch["action_mask"],
            batch["terminal"],
        )
        r_sum = chunk_return(batch["rewards"], gamma, batch["action_mask"])
        y_td = (r_sum + boot * q_next).clamp(0.0, 1.0)
        y_mc = batch["mc_return"].clamp(0.0, 1.0)
        x_t_native, t = _ae_reverse_native_state(
            model,
            ae_backend,
            contexts,
            state,
            actions_native,
            apply_guide=bool(critic_target_use_guide and model.guide is not None),
        )
        x_t = _project_native_chunk(model, x_t_native)

    qs = model.q_chunk(state, x_t, t=t)
    td = ((qs - y_td.unsqueeze(0)) ** 2).mean()
    t_end = torch.ones(
        actions.shape[0],
        1,
        device=actions.device,
        dtype=actions.dtype,
    )
    qs_end = model.q_chunk(state, actions, t=t_end)
    td = td + ((qs_end - y_td.unsqueeze(0)) ** 2).mean()
    mc = ((qs - y_mc.unsqueeze(0)) ** 2).mean()
    mc = mc + ((qs_end - y_mc.unsqueeze(0)) ** 2).mean()
    cql, cql_info = model.critic.cql_penalty(
        state.detach() if detach_token else state,
        x_t.reshape(x_t.shape[0], -1).detach(),
        t,
        n_actions=cql_n_actions,
        coef=cql_coef,
        action_radius=cql_action_radius,
        far_scale=1.0,
    )

    rank = actions.new_zeros(())
    q_gap = actions.new_zeros(())
    weight = y_mc.detach().clamp(0.0, 1.0)
    q_pos_heads = model.q_chunk(state, actions, t=t_end)
    q_pos = model.q_lower_tail_chunk(state, actions, t=t_end)
    if rank_coef > 0.0 and rank_noise > 0.0:
        action_neg = actions.detach() + float(rank_noise) * torch.randn_like(actions)
        q_neg_heads = model.q_chunk(state, action_neg, t=t_end)
        q_neg = model.q_lower_tail_chunk(state, action_neg, t=t_end)
        rank = rank + float(rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_neg_heads,
            weight,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_neg).detach().mean()
    if far_rank_coef > 0.0 and far_rank_noise > 0.0:
        action_far = actions.detach() + float(far_rank_noise) * torch.randn_like(actions)
        q_far_heads = model.q_chunk(state, action_far, t=t_end)
        q_far = model.q_lower_tail_chunk(state, action_far, t=t_end)
        rank = rank + float(far_rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_far_heads,
            weight,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_far).detach().mean()
    if shuffle_rank_coef > 0.0 and actions.shape[0] > 1:
        perm = torch.randperm(actions.shape[0], device=actions.device)
        action_shuffled = actions.detach()[perm]
        q_shuffled = model.q_lower_tail_chunk(
            state, action_shuffled, t=t_end
        )
        q_shuffled_heads = model.q_chunk(state, action_shuffled, t=t_end)
        rank = rank + float(shuffle_rank_coef) * _per_head_rank_loss(
            q_pos_heads,
            q_shuffled_heads,
            weight,
            rank_margin,
        )
        q_gap = q_gap + (q_pos - q_shuffled).detach().mean()

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
        "q_mean": float(qs_end.mean().detach()),
        "q_std": float(qs_end.std(unbiased=False).detach()),
        "q_target": float(y_td.mean().detach()),
        "flow_t_mean": float(t.mean().detach()),
        "v_source": 1.0,
        "native_horizon": float(reference_native.shape[1]),
    }


def flow_actor_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    beta: float = 1.0,
    target_divergence: float = 0.0025,
    ref_dropout: float = 0.5,
    bc_coef: float = 1.0,
    bc_ref_coef: float = 1.0,
    endpoint_aux_coef: float = 0.5,
    endpoint_aux_steps: int = 4,
    q_coef: float = 1.0,
    residual_clip: float | None = None,
    advantage_clip: float | None = 0.05,
    endpoint_ref_mse_max: float | None = None,
) -> dict[str, float]:
    """Train the flow actor against its complete deployment endpoint.

    Phase control:
    - ``q_coef=0`` keeps pure BC toward the frozen reference.
    - ``endpoint_ref_mse_max`` blocks endpoint Q until the clone is close enough.
    """
    if not model.is_flow:
        return actor_step(
            model,
            opt,
            alpha_opt,
            batch,
            beta=beta,
            target_divergence=target_divergence,
            ref_dropout=ref_dropout,
            q_coef=q_coef,
            residual_clip=residual_clip,
            advantage_clip=advantage_clip,
        )
    del endpoint_aux_steps
    model.train()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
    a_data = model.normalize_action(batch["executed_actions"])
    a_ref = model.normalize_action(batch["reference_actions"])
    # Mix BC target: default fully toward frozen VLA reference.
    ref_w = float(max(0.0, min(1.0, bc_ref_coef)))
    a_bc = ref_w * a_ref + (1.0 - ref_w) * a_data
    b = a_data.shape[0]
    present = (torch.rand(b, device=a_data.device) > ref_dropout).float()
    ref_in = a_ref * present.view(b, 1, 1)
    x0 = torch.randn_like(a_bc)
    t = torch.rand(b, 1, device=a_bc.device, dtype=a_bc.dtype)
    x_t = (1.0 - t.view(b, 1, 1)) * x0 + t.view(b, 1, 1) * a_bc
    target_v = a_bc - x0
    v = model.flow_velocity(state, x_t, t, ref_in)
    mask = batch["action_mask"].unsqueeze(-1)
    bc = (((v - target_v.detach()) * mask) ** 2).sum() / mask.sum().clamp_min(1.0) / v.shape[-1]

    # Keep a one-step value diagnostic, but do not optimize this surrogate.
    dt = torch.minimum(
        torch.full_like(t, 1.0 / float(model.flow_steps)),
        1.0 - t,
    )
    x_plus = x_t + v * dt.view(b, 1, 1)
    t_plus = (t + 1.0 / float(model.flow_steps)).clamp(0.0, 1.0)
    q_look = model.q_lower_tail_chunk(state, x_plus, t=t_plus)
    with torch.no_grad():
        q_base = model.q_lower_tail_chunk(state, x_t.detach(), t=t)
    local_advantage = q_look - q_base

    # This source is shared by BC, endpoint optimization, and its reference
    # comparison. Backpropagation traverses every deployment integration step.
    endpoint = x0
    for step_index in range(model.flow_steps):
        step_t = torch.full(
            (b, 1),
            step_index / float(model.flow_steps),
            device=endpoint.device,
            dtype=endpoint.dtype,
        )
        endpoint_velocity = model.flow_velocity(
            state,
            endpoint,
            step_t,
            ref_in,
        )
        endpoint = endpoint + endpoint_velocity / float(model.flow_steps)
    if residual_clip is not None and float(residual_clip) > 0.0:
        delta = (endpoint - a_ref.detach()).clamp(
            -float(residual_clip),
            float(residual_clip),
        )
        endpoint = a_ref.detach() + delta
    t_end = torch.ones(b, 1, device=endpoint.device, dtype=endpoint.dtype)
    q_end = model.q_lower_tail_chunk(state, endpoint, t=t_end)
    with torch.no_grad():
        q_reference = model.q_lower_tail_chunk(state, a_ref, t=t_end)
    endpoint_advantage = q_end - q_reference
    if advantage_clip is not None and float(advantage_clip) > 0.0:
        endpoint_advantage = endpoint_advantage.clamp(
            -float(advantage_clip),
            float(advantage_clip),
        )

    residual_mse = (
        ((endpoint - a_ref.detach()) * mask) ** 2
    ).sum() / mask.sum().clamp_min(1.0) / endpoint.shape[-1]
    endpoint_ref_mse = float(residual_mse.detach())
    effective_q_coef = float(q_coef) * float(endpoint_aux_coef)
    if (
        endpoint_ref_mse_max is not None
        and float(endpoint_ref_mse_max) > 0.0
        and endpoint_ref_mse > float(endpoint_ref_mse_max)
    ):
        effective_q_coef = 0.0
    alpha = model.log_alpha()
    actor_loss = (
        -effective_q_coef * endpoint_advantage.mean()
        + float(bc_coef) * bc
        + float(beta) * residual_mse
        + alpha.detach() * residual_mse
    )
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
        "actor_adv": float(endpoint_advantage.mean().detach()),
        "actor_local_adv": float(local_advantage.mean().detach()),
        "actor_endpoint_adv": float(endpoint_advantage.mean().detach()),
        "actor_q_look": float(q_look.mean().detach()),
        "actor_q_end": float(q_end.mean().detach()),
        "residual_mse": float(residual_mse.detach()),
        "residual_rms": float(torch.sqrt(residual_mse).detach()),
        "actor_ref_mse": endpoint_ref_mse,
        "endpoint_ref_mse": endpoint_ref_mse,
        "bc_loss": float(bc.detach()),
        "bc_ref_coef": float(ref_w),
        "actor_q_coef": float(effective_q_coef),
        "alpha": float(alpha.detach()),
        "endpoint_steps": float(model.flow_steps),
        "endpoint_t": float(t_end.mean().detach()),
    }


def cfgrl_condition_and_target(
    a_data: torch.Tensor,
    a_ref: torch.Tensor,
    *,
    success: torch.Tensor | None = None,
    advantage: torch.Tensor | None = None,
    cond_dropout: float = DEFAULT_CFGRL_DROPOUT,
    use_advantage_labels: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Binary CFGRL labels: uncond clones ã, cond clones positive executed chunks.

    Failures / A<0 are mapped to uncond rather than a dedicated o=NEG head so
    the shared backbone is not trained mostly on unsuccessful actions.
    """

    batch_size = a_data.shape[0]
    device = a_data.device
    if use_advantage_labels:
        if advantage is None:
            raise ValueError("advantage is required when use_advantage_labels=True")
        adv = advantage.reshape(batch_size).to(device=device)
        positive = adv >= 0
        adv_mean = float(adv.mean().detach())
    else:
        if success is None:
            success = torch.zeros(batch_size, device=device)
        success = success.reshape(batch_size).to(device=device, dtype=a_data.dtype)
        positive = success > 0.5
        adv_mean = float(success.mean().detach())
    pos_frac = float(positive.float().mean().detach())
    o = torch.where(
        positive,
        torch.full((batch_size,), CFGRL_O_POS, device=device, dtype=torch.long),
        torch.full((batch_size,), CFGRL_O_UNCOND, device=device, dtype=torch.long),
    )
    drop = torch.rand(batch_size, device=device) < float(cond_dropout)
    o = torch.where(
        drop,
        torch.full((batch_size,), CFGRL_O_UNCOND, device=device, dtype=torch.long),
        o,
    )
    a_star = torch.where(o.view(batch_size, 1, 1) == CFGRL_O_POS, a_data, a_ref)
    return o, a_star, {
        "cfgrl_pos_frac": pos_frac,
        "cfgrl_uncond_frac": float((o == CFGRL_O_UNCOND).float().mean().detach()),
        "cfgrl_use_advantage": float(use_advantage_labels),
        "cfgrl_adv_mean": adv_mean,
    }


@torch.no_grad()
def cfgrl_endpoint_diagnostics(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Endpoint MSE of w=0 (uncond) and w=1 (cond) samples vs ã and executed a."""

    if not model.is_flow:
        raise RuntimeError("cfgrl_endpoint_diagnostics requires cf_mode=flow")
    was_training = bool(model.training)
    model.eval()
    state = _batch_state(model, batch, detach_token=True, use_target=False)
    a_data = model.normalize_action(batch["executed_actions"])
    a_ref = model.normalize_action(batch["reference_actions"])
    mask = batch["action_mask"].unsqueeze(-1)

    def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        return float(
            (((pred - target) * mask) ** 2).sum()
            / mask.sum().clamp_min(1.0)
            / pred.shape[-1]
        )

    uncond, _ = model.flow_sample(state, a_ref, cfg_w=0.0)
    cond, _ = model.flow_sample(state, a_ref, cfg_w=1.0)
    if was_training:
        model.train()
    return {
        "cfgrl_uncond_ref_mse": _masked_mse(uncond, a_ref),
        "cfgrl_cond_ref_mse": _masked_mse(cond, a_ref),
        "cfgrl_uncond_data_mse": _masked_mse(uncond, a_data),
        "cfgrl_cond_data_mse": _masked_mse(cond, a_data),
    }


def cfgrl_actor_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    cond_dropout: float = DEFAULT_CFGRL_DROPOUT,
    use_advantage_labels: bool = False,
    snapshot_critic: nn.Module | None = None,
) -> dict[str, float]:
    """CFGRL flow-matching: uncond clones ã, cond clones positive executed a.

    No Q-max actor term and no residual β. Labels come from stop-grad Q (A vs
    VLA reference) once the critic is healthy, otherwise from episode success.
    Negative / failed chunks are trained as uncond (clone ã), not as o=NEG BC.
    """
    if not model.is_flow:
        raise RuntimeError("cfgrl_actor_step requires cf_mode=flow")
    model.train()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
    a_data = model.normalize_action(batch["executed_actions"])
    a_ref = model.normalize_action(batch["reference_actions"])
    b = a_data.shape[0]
    device = a_data.device
    adv = None
    if use_advantage_labels:
        t_end = torch.ones(b, 1, device=device, dtype=a_data.dtype)
        critic_model = snapshot_critic if snapshot_critic is not None else model
        with torch.no_grad():
            q_a = critic_model.q_lower_tail_chunk(state, a_data, t=t_end)
            q_ref = critic_model.q_lower_tail_chunk(state, a_ref, t=t_end)
            adv = q_a - q_ref
    success = batch.get("success")
    if success is None:
        success = batch.get("mc_return")
    o, a_star, label_info = cfgrl_condition_and_target(
        a_data,
        a_ref,
        success=success,
        advantage=adv,
        cond_dropout=cond_dropout,
        use_advantage_labels=use_advantage_labels,
    )
    x0 = torch.randn_like(a_star)
    t = torch.rand(b, 1, device=device, dtype=a_star.dtype)
    x_t = (1.0 - t.view(b, 1, 1)) * x0 + t.view(b, 1, 1) * a_star
    target_v = a_star - x0
    v = model.flow_velocity(state, x_t, t, a_ref, o=o)
    mask = batch["action_mask"].unsqueeze(-1)
    bc = (((v - target_v.detach()) * mask) ** 2).sum() / mask.sum().clamp_min(1.0) / v.shape[-1]
    opt.zero_grad(set_to_none=True)
    bc.backward()
    nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
    opt.step()
    return {
        "actor_loss": float(bc.detach()),
        "bc_loss": float(bc.detach()),
        **label_info,
        "actor_q_coef": 0.0,
        "cfgrl_w": float(model.cfgrl_w),
    }


def _ae_actor_gradient_diagnostics(
    ae_backend: Any,
    opt: torch.optim.Optimizer,
) -> dict[str, float]:
    root = getattr(ae_backend, "model", ae_backend)
    named_parameters = getattr(root, "named_parameters", None)
    if not callable(named_parameters):
        raise RuntimeError(
            "Molmo AE backend does not expose named parameters for gradient diagnostics"
        )
    named_trainable = [
        (name, parameter)
        for name, parameter in named_parameters()
        if parameter.requires_grad
    ]
    if not named_trainable:
        raise RuntimeError("Molmo AE actor has no trainable parameters")
    optimizer_ids = {
        id(parameter)
        for group in opt.param_groups
        for parameter in group["params"]
    }

    def gradient_norm(named: list[tuple[str, torch.Tensor]]) -> float:
        squared = 0.0
        for _name, parameter in named:
            if parameter.grad is not None:
                norm = parameter.grad.detach().float().norm()
                squared += float((norm * norm).cpu())
        return squared**0.5

    diagnostics: dict[str, float] = {}
    projection_norms: list[float] = []
    for projection in ("context_k_proj", "context_v_proj"):
        selected = [
            (name, parameter)
            for name, parameter in named_trainable
            if projection in name
        ]
        if not selected:
            raise RuntimeError(
                f"Molmo AE actor has no trainable {projection} parameters"
            )
        if any(id(parameter) not in optimizer_ids for _name, parameter in selected):
            raise RuntimeError(
                f"Actor optimizer does not own every trainable {projection} parameter"
            )
        norm = gradient_norm(selected)
        if not torch.isfinite(torch.tensor(norm)) or norm <= 0.0:
            raise RuntimeError(
                f"Molmo AE actor produced zero/non-finite {projection} gradient; "
                "context projection must be created outside no_grad"
            )
        diagnostics[f"ae_{projection}_grad_norm"] = norm
        projection_norms.append(norm)

    total_norm = gradient_norm(named_trainable)
    if not torch.isfinite(torch.tensor(total_norm)) or total_norm <= 0.0:
        raise RuntimeError("Molmo AE actor produced zero/non-finite total gradient")
    owned = sum(
        parameter.numel()
        for _name, parameter in named_trainable
        if id(parameter) in optimizer_ids
    )
    trainable = sum(parameter.numel() for _name, parameter in named_trainable)
    diagnostics["ae_grad_norm"] = total_norm
    diagnostics["ae_context_proj_grad_norm"] = sum(
        norm * norm for norm in projection_norms
    ) ** 0.5
    diagnostics["ae_context_proj_grad_nonzero"] = 1.0
    diagnostics["ae_optimizer_param_coverage"] = owned / max(trainable, 1)
    return diagnostics


def ae_flow_actor_step(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    opt: torch.optim.Optimizer,
    alpha_opt: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    beta: float = 1.0,
    target_divergence: float = 0.0025,
    bc_coef: float = 1.0,
    bc_ref_coef: float = 1.0,
    endpoint_aux_coef: float = 0.5,
    endpoint_aux_steps: int = 4,
    zero_grad: bool = True,
    optimizer_step: bool = True,
    loss_scale: float = 1.0,
) -> dict[str, float]:
    """Joint AE actor in Molmo q01-q99 coordinates over the full horizon."""
    if not model.is_flow:
        raise RuntimeError("ae_flow_actor_step requires cf_mode=flow")
    del endpoint_aux_steps
    _require_ae_full_fields(
        batch,
        "full_reference_actions",
        "full_executed_actions",
        "source_native",
    )
    model.train()
    ae_backend.train(True)
    invalidate_cache = getattr(ae_backend, "invalidate_modulation_cache", None)
    if callable(invalidate_cache):
        invalidate_cache()

    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
    # encode_context freezes and detaches VLM KV internally.  It does not invoke
    # the action expert, so AE context projections below remain differentiable.
    contexts = _ae_batch_contexts(model, ae_backend, batch)
    a_data = _ae_normalized_full_actions(
        model,
        ae_backend,
        batch,
        "full_executed_actions",
    )
    a_ref = _ae_normalized_full_actions(
        model,
        ae_backend,
        batch,
        "full_reference_actions",
    )
    batch_size = a_ref.shape[0]
    device = a_ref.device
    ref_w = float(max(0.0, min(1.0, bc_ref_coef)))
    a_bc = _mask_ae_native(ref_w * a_ref + (1.0 - ref_w) * a_data)
    source_native = _ae_source_native(
        model,
        ae_backend,
        contexts,
        batch,
        "source_native",
        template=a_ref,
    )
    t = torch.rand(batch_size, 1, device=device, dtype=torch.float32)
    t_full = t.view(batch_size, 1, 1).to(dtype=a_bc.dtype)
    x_t = _mask_ae_native((1.0 - t_full) * source_native + t_full * a_bc)
    target_v = _mask_ae_native(a_bc - source_native)
    velocity = _ae_native_velocity(ae_backend, contexts, x_t, t)
    valid = (~_ae_pad_mask(batch_size, device=device)).expand_as(velocity)
    bc = (((velocity - target_v.detach()) * valid) ** 2).sum()
    bc = bc / valid.sum().clamp_min(1)

    dt = torch.minimum(
        torch.full_like(t, 1.0 / float(model.flow_steps)),
        1.0 - t,
    )
    x_plus_native = _mask_ae_native(
        x_t
        + velocity * dt.view(batch_size, 1, 1).to(dtype=velocity.dtype)
    )
    x_plus = _project_native_chunk(model, x_plus_native)
    x_t_compact = _project_native_chunk(model, x_t)
    t_plus = (t + 1.0 / float(model.flow_steps)).clamp(0.0, 1.0)
    q_look = model.q_lower_tail_chunk(state, x_plus, t=t_plus)
    with torch.no_grad():
        q_base = model.q_lower_tail_chunk(
            state,
            x_t_compact.detach(),
            t=t,
        )
    local_advantage = q_look - q_base

    endpoint_native = _ae_integrate_native(
        model,
        ae_backend,
        contexts,
        state,
        source_native,
        steps=model.flow_steps,
        apply_guide=False,
        detach_guide=True,
    )
    endpoint = _project_native_chunk(model, endpoint_native)
    t_end = torch.ones(
        batch_size,
        1,
        device=endpoint.device,
        dtype=endpoint.dtype,
    )
    q_end = model.q_lower_tail_chunk(state, endpoint, t=t_end)
    reference = _project_native_chunk(model, a_ref)
    with torch.no_grad():
        q_reference = model.q_lower_tail_chunk(state, reference, t=t_end)
    endpoint_advantage = q_end - q_reference
    endpoint_error = (endpoint_native - a_ref.detach()) * valid
    residual_mse = (endpoint_error**2).sum() / valid.sum().clamp_min(1)
    alpha = model.log_alpha()
    actor_loss = (
        -float(endpoint_aux_coef) * endpoint_advantage.mean()
        + float(bc_coef) * bc
        + float(beta) * residual_mse
        + alpha.detach() * residual_mse
    )
    if zero_grad:
        opt.zero_grad(set_to_none=True)
    (actor_loss * float(loss_scale)).backward()
    gradient_diagnostics: dict[str, float] = {}
    if optimizer_step:
        gradient_diagnostics = _ae_actor_gradient_diagnostics(ae_backend, opt)
        trainable_parameters = ae_backend.trainable_parameters()
        nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
        opt.step()
        if callable(invalidate_cache):
            invalidate_cache()

    constraint = (residual_mse - target_divergence).detach()
    alpha_loss = -model.log_alpha.log_alpha * constraint
    if zero_grad:
        alpha_opt.zero_grad(set_to_none=True)
    (alpha_loss * float(loss_scale)).backward()
    if optimizer_step:
        alpha_opt.step()

    return {
        "actor_loss": float(actor_loss.detach()),
        "actor_adv": float(endpoint_advantage.mean().detach()),
        "actor_local_adv": float(local_advantage.mean().detach()),
        "actor_endpoint_adv": float(endpoint_advantage.mean().detach()),
        "actor_q_look": float(q_look.mean().detach()),
        "actor_q_end": float(q_end.mean().detach()),
        "residual_mse": float(residual_mse.detach()),
        "bc_loss": float(bc.detach()),
        "bc_ref_coef": float(ref_w),
        "alpha": float(alpha.detach()),
        "native_horizon": float(AE_NATIVE_HORIZON),
        "endpoint_steps": float(model.flow_steps),
        "endpoint_t": float(t_end.mean().detach()),
        "v_source": 1.0,  # marker: molmo_ae
        **gradient_diagnostics,
    }


def ae_flow_guide_step(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    opt: torch.optim.Optimizer,
    batch: dict[str, Any],
    *,
    beta: float = 0.05,
    distill_coef: float = 1.0,
    target_delta_frac: float | None = None,
) -> dict[str, float]:
    """Distill a compact guide from full-horizon native AE flow states."""
    if model.guide is None:
        return {"guide_loss": 0.0, "guide_adv": 0.0}
    if not model.is_flow:
        return guide_step(model, opt, batch, beta=beta)
    del beta, target_delta_frac
    _require_ae_full_fields(
        batch,
        "full_reference_actions",
        "source_native",
    )
    model.train()
    ae_backend.eval()

    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
    contexts = _ae_batch_contexts(model, ae_backend, batch)
    reference_native = _ae_normalized_full_actions(
        model,
        ae_backend,
        batch,
        "full_reference_actions",
    )
    source_native = _ae_source_native(
        model,
        ae_backend,
        contexts,
        batch,
        "source_native",
        template=reference_native,
    )
    batch_size = reference_native.shape[0]
    t = 0.5 + 0.5 * torch.rand(
        batch_size,
        1,
        device=reference_native.device,
        dtype=torch.float32,
    )
    t_full = t.view(batch_size, 1, 1).to(dtype=reference_native.dtype)
    x_t_native = _mask_ae_native(
        (1.0 - t_full) * source_native + t_full * reference_native
    ).detach()
    x_t = _project_native_chunk(model, x_t_native)

    target, gradient_diagnostics = stochastic_target_critic_gradient(
        model,
        state,
        x_t,
        t=t,
        return_diagnostics=True,
    )
    if _guide_teacher_is_dead(gradient_diagnostics):
        return {
            "guide_loss": 0.0,
            "guide_adv": 0.0,
            "guide_update_skipped": 1.0,
            "guide_skip_tiny_critic_gradient": 1.0,
            "native_horizon": float(AE_NATIVE_HORIZON),
            **gradient_diagnostics,
        }

    with torch.no_grad():
        velocity_native = _ae_native_velocity(
            ae_backend,
            contexts,
            x_t_native,
            t,
        ).detach()
        velocity = _project_native_chunk(model, velocity_native)

    g_chunk, w_flat, diag = model.guide.guidance(
        state.detach(),
        x_t.detach(),
        t,
        velocity,
    )
    distill = F.mse_loss(w_flat, target)

    dt = torch.minimum(
        torch.full_like(t, 1.0 / float(model.flow_steps)),
        1.0 - t,
    )
    x_guided = x_t.detach() + (velocity + g_chunk) * dt.view(
        batch_size,
        1,
        1,
    )
    x_base = x_t.detach() + velocity * dt.view(batch_size, 1, 1)
    t_plus = (t + 1.0 / float(model.flow_steps)).clamp(0.0, 1.0)
    with torch.no_grad():
        q_g = model.q_lower_tail_chunk(state.detach(), x_guided, t=t_plus)
        q_b = model.q_lower_tail_chunk(state.detach(), x_base, t=t_plus)
    adv = q_g - q_b
    mag = (g_chunk**2).mean()
    loss = float(distill_coef) * distill
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.guide.parameters(), 1.0)
    opt.step()
    out = {
        "guide_loss": float(loss.detach()),
        "guide_adv": float(adv.mean().detach()),
        "guide_update_skipped": 0.0,
        "guide_skip_tiny_critic_gradient": 0.0,
        "guide_distill": float(distill.detach()),
        "guide_mse": float(mag.detach()),
        "w_norm": float(w_flat.norm(dim=-1).mean().detach()),
        "target_norm": float(target.norm(dim=-1).mean().detach()),
        "native_horizon": float(AE_NATIVE_HORIZON),
        **gradient_diagnostics,
    }
    for k, v_t in diag.items():
        out[f"guide_{k}"] = float(v_t.detach()) if torch.is_tensor(v_t) else float(v_t)
    return out


def flow_guide_step(
    model: MolmoAct2RLTCF,
    opt: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    beta: float = 0.05,
    distill_coef: float = 1.0,
    target_delta_frac: float | None = None,
    guide_on_reference: bool = False,
) -> dict[str, float]:
    """Distill stochastic target-critic gradients into raw W."""
    del target_delta_frac
    if model.guide is None:
        return {"guide_loss": 0.0, "guide_adv": 0.0}
    if not model.is_flow:
        return guide_step(
            model,
            opt,
            batch,
            beta=beta,
            guide_on_reference=guide_on_reference,
        )
    model.train()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        a1 = model.normalize_action(batch["reference_actions"])
    b = a1.shape[0]
    x0 = torch.randn_like(a1)
    # Bias BC times toward the endpoint where the fixed critic is informative.
    t = 0.5 + 0.5 * torch.rand(b, 1, device=a1.device, dtype=a1.dtype)
    x_t = ((1.0 - t.view(b, 1, 1)) * x0 + t.view(b, 1, 1) * a1).detach().requires_grad_(True)
    target, gradient_diagnostics = stochastic_target_critic_gradient(
        model,
        state,
        x_t,
        t=t,
        return_diagnostics=True,
    )
    if _guide_teacher_is_dead(gradient_diagnostics):
        return {
            "guide_loss": 0.0,
            "guide_adv": 0.0,
            "guide_update_skipped": 1.0,
            "guide_skip_tiny_critic_gradient": 1.0,
            "guide_on_reference": float(guide_on_reference),
            **gradient_diagnostics,
        }

    v = model.flow_velocity(state.detach(), x_t.detach(), t, a1).detach()
    g_chunk, w_flat, diag = model.guide.guidance(state.detach(), x_t.detach(), t, v)
    distill = F.mse_loss(w_flat, target)

    dt = torch.minimum(
        torch.full_like(t, 1.0 / float(model.flow_steps)),
        1.0 - t,
    )
    x_guided = x_t.detach() + (v + g_chunk) * dt.view(b, 1, 1)
    x_base = x_t.detach() + v * dt.view(b, 1, 1)
    t_plus = (t + 1.0 / float(model.flow_steps)).clamp(0.0, 1.0)
    with torch.no_grad():
        q_g = model.q_lower_tail_chunk(state.detach(), x_guided, t=t_plus)
        q_b = model.q_lower_tail_chunk(state.detach(), x_base, t=t_plus)
    adv = q_g - q_b
    mag = (g_chunk**2).mean()
    loss = float(distill_coef) * distill
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.guide.parameters(), 1.0)
    opt.step()
    out = {
        "guide_loss": float(loss.detach()),
        "guide_adv": float(adv.mean().detach()),
        "guide_update_skipped": 0.0,
        "guide_skip_tiny_critic_gradient": 0.0,
        "guide_distill": float(distill.detach()),
        "guide_mse": float(mag.detach()),
        "w_norm": float(w_flat.norm(dim=-1).mean().detach()),
        "target_norm": float(target.norm(dim=-1).mean().detach()),
        "guide_on_reference": float(guide_on_reference),
        **gradient_diagnostics,
    }
    for k, v_t in diag.items():
        out[f"guide_{k}"] = float(v_t.detach()) if torch.is_tensor(v_t) else float(v_t)
    return out
