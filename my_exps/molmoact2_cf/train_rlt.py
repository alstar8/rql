"""Training steps for RLT-Consensus CF-VLA (token recon, chunk TD, actor, CF guide)."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import nn

from rlt_models import (
    MolmoAct2RLTCF,
    bootstrap_scale,
    chunk_return,
    common_scale_normalize,
)

VelocityProvider = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


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
) -> torch.Tensor:
    """Sample one target critic per example and return common-scale z targets."""
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
    selected = qs[member, torch.arange(batch_size, device=action_leaf.device)].sum()
    grad = torch.autograd.grad(selected, action_leaf, retain_graph=False)[0].detach()
    return common_scale_normalize(grad.reshape(batch_size, -1))


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
            apply_guide=model.guide is not None,
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
    beta: float = 0.05,
    target_delta_frac: float | None = None,
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
        actor_mean, ainfo = model.actor_chunk(state, ref, deterministic=True, apply_guide=False)

    # One stochastic target-critic member per example avoids opposed-member
    # cancellation while preserving the common-scale target magnitude.
    target = stochastic_target_critic_gradient(
        model,
        state,
        actor_mean,
    )

    w_flat = model.guide.raw_w(state.detach(), ref)
    _guided_ref, g_delta = model.guide.guide(
        state.detach(), ref, actor_delta=ainfo["actor_delta"]
    )
    # Additive composition: guide rides on top of the actor residual.
    composed = actor_mean.detach() + g_delta
    distill = F.mse_loss(w_flat, target)
    align = 1.0 - F.cosine_similarity(
        w_flat, target, dim=-1
    ).mean()
    with torch.no_grad():
        q = model.q_min_chunk(state.detach(), composed)
        base_q = model.q_min_chunk(state.detach(), actor_mean.detach())
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
        "guide_align": float(align.detach()),
        "guide_distill": float(distill.detach()),
        "guide_mse": float(mag.detach()),
        "guide_delta_rms": float(mag.detach().sqrt()),
        "w_norm": float(w_flat.norm(dim=-1).mean().detach()),
        "target_norm": float(target.norm(dim=-1).mean().detach()),
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
    """q_min advantage of deploy action vs reference (matches actor/guide training).

    v8 gated on ensemble LCB (= mean-std), which stayed ≪ τ even when q_min
    actor advantage already cleared the threshold.
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
        q_act = model.q_min_chunk(state, act, t=t)
        q_ref = model.q_min_chunk(state, ref, t=t)
        return float((q_act - q_ref).mean().detach())


def predicted_guide_advantage(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
) -> float:
    """q_min advantage of guided actor vs unguided actor (residual CF guide gate)."""
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
        q_g = model.q_min_chunk(state, act_g, t=t)
        q_u = model.q_min_chunk(state, act_u, t=t)
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
        q0 = model.q_min_chunk(state, a, t=t)
        q1 = model.q_min_chunk(state, a + float(noise) * torch.randn_like(a), t=t)
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
    prefix = "next_" if next_state else ""
    external = batch[f"{prefix}external_cam"]
    wrist = batch[f"{prefix}wrist_cam"]
    instructions = batch[f"{prefix}instruction"]
    proprio = batch[f"{prefix}proprio"]
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
            action_horizon=int(model.chunk_size),
        )
        contexts.append(ctx)
    return contexts


def _ae_velocity_provider(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    contexts: list[Any | None],
) -> VelocityProvider:
    std = model.action_std.detach().clamp_min(1e-6)

    def velocity(
        _state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        _reference: torch.Tensor,
    ) -> torch.Tensor:
        x_raw = model.denormalize_action(x_t)
        rows: list[torch.Tensor] = []
        for i, context in enumerate(contexts):
            if context is None:
                rows.append(torch.zeros_like(x_raw[i : i + 1]))
                continue
            row = ae_backend.velocity(context, x_raw[i : i + 1], t[i : i + 1])
            rows.append(row.to(device=x_t.device, dtype=x_t.dtype))
        return torch.cat(rows, dim=0) / std.to(device=x_t.device, dtype=x_t.dtype)

    return velocity


def _flow_endpoint_with_provider(
    model: MolmoAct2RLTCF,
    state: torch.Tensor,
    reference_n: torch.Tensor,
    velocity_provider: VelocityProvider,
    *,
    apply_guide: bool,
    x0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the same velocity/guidance field used by reverse states."""
    s = state.detach()
    ref = reference_n.detach()
    x = torch.randn_like(ref) if x0 is None else x0
    b = x.shape[0]
    dt = 1.0 / float(model.flow_steps)
    for i in range(model.flow_steps):
        t = torch.full(
            (b, 1),
            i / float(model.flow_steps),
            device=x.device,
            dtype=x.dtype,
        )
        v = velocity_provider(s, x, t, ref)
        g = torch.zeros_like(v)
        if apply_guide and model.guide is not None:
            g, _, _ = model.guide.guidance(s, x, t, v.detach())
        x = x + (v + g) * dt
    return x


def ae_flow_gate_metrics(
    model: MolmoAct2RLTCF,
    ae_backend: Any,
    batch: dict[str, Any],
    *,
    sensitivity_noise: float = 0.03,
) -> dict[str, float]:
    """Gate diagnostics from actual trainable-AE flow predictions on images."""
    if not model.is_flow:
        raise RuntimeError("ae_flow_gate_metrics requires cf_mode=flow")
    model.eval()
    ae_backend.eval()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        reference = model.normalize_action(batch["reference_actions"])
        contexts = _ae_batch_contexts(model, ae_backend, batch)
        provider = _ae_velocity_provider(model, ae_backend, contexts)
        x0_raw = torch.randn_like(batch["reference_actions"])
        x0 = model.normalize_action(x0_raw)
        actor = _flow_endpoint_with_provider(
            model,
            state,
            reference,
            provider,
            apply_guide=False,
            x0=x0,
        )
        guided = actor
        if model.guide is not None:
            guided = _flow_endpoint_with_provider(
                model,
                state,
                reference,
                provider,
                apply_guide=True,
                x0=x0,
            )
        deploy = guided if model.guide is not None else actor
        t1 = torch.ones(
            deploy.shape[0],
            1,
            device=deploy.device,
            dtype=deploy.dtype,
        )
        qs_deploy = model.q_chunk(state, deploy, t=t1)
        qs_reference = model.q_chunk(state, reference, t=t1)
        paired = qs_deploy - qs_reference
        paired_lcb = paired.mean(dim=0) - paired.std(dim=0, unbiased=False)
        q_min_advantage = (
            model.q_min_chunk(state, deploy, t=t1)
            - model.q_min_chunk(state, reference, t=t1)
        )
        guide_advantage = deploy.new_zeros(deploy.shape[0])
        if model.guide is not None:
            guide_advantage = (
                model.q_min_chunk(state, guided, t=t1)
                - model.q_min_chunk(state, actor, t=t1)
            )
        perturbed = deploy + float(sensitivity_noise) * torch.randn_like(deploy)
        sensitivity = (
            model.q_min_chunk(state, deploy, t=t1)
            - model.q_min_chunk(state, perturbed, t=t1)
        ).abs()
    return {
        "paired_lcb": float(paired_lcb.mean().detach()),
        "q_min_advantage": float(q_min_advantage.mean().detach()),
        "guide_advantage": float(guide_advantage.mean().detach()),
        "sensitivity": float(sensitivity.mean().detach()),
    }


def flow_gate_metrics(
    model: MolmoAct2RLTCF,
    batch: dict[str, torch.Tensor],
    *,
    sensitivity_noise: float = 0.03,
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
        deploy = guided if model.guide is not None else actor
        t1 = torch.ones(
            deploy.shape[0],
            1,
            device=deploy.device,
            dtype=deploy.dtype,
        )
        qs_deploy = model.q_chunk(state, deploy, t=t1)
        qs_reference = model.q_chunk(state, reference, t=t1)
        paired = qs_deploy - qs_reference
        paired_lcb = paired.mean(dim=0) - paired.std(dim=0, unbiased=False)
        q_min_advantage = (
            model.q_min_chunk(state, deploy, t=t1)
            - model.q_min_chunk(state, reference, t=t1)
        )
        guide_advantage = deploy.new_zeros(deploy.shape[0])
        if model.guide is not None:
            guide_advantage = (
                model.q_min_chunk(state, guided, t=t1)
                - model.q_min_chunk(state, actor, t=t1)
            )
        perturbed = deploy + float(sensitivity_noise) * torch.randn_like(deploy)
        sensitivity = (
            model.q_min_chunk(state, deploy, t=t1)
            - model.q_min_chunk(state, perturbed, t=t1)
        ).abs()
    return {
        "paired_lcb": float(paired_lcb.mean().detach()),
        "q_min_advantage": float(q_min_advantage.mean().detach()),
        "guide_advantage": float(guide_advantage.mean().detach()),
        "sensitivity": float(sensitivity.mean().detach()),
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
            apply_guide=model.guide is not None,
            reference_present=present,
        )
        if target_noise > 0.0:
            next_act = (next_act + target_noise * torch.randn_like(next_act)).clamp(-2.0, 2.0)
        # Endpoint action-values live at t=1. Bootstrapping at t=0 (noise time)
        # evaluates Q(s', a', 0) which collapses to ~0 and starves TD targets.
        t1 = torch.ones(b, 1, device=next_state.device, dtype=next_state.dtype)
        q_next = model.q_min_chunk(next_state, next_act, target=True, t=t1)
        boot = bootstrap_scale(gamma, model.chunk_size, batch["action_mask"], batch["terminal"])
        r_sum = chunk_return(batch["rewards"], gamma, batch["action_mask"])
        y_td = (r_sum + boot * q_next).clamp(0.0, 1.0)
        y_mc = batch["mc_return"].clamp(0.0, 1.0)
        x_t, t = _flow_reverse_state(model, state, a, ref, apply_guide=model.guide is not None)

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
    q_pos = model.q_min_chunk(state, a, t=t_end)
    rank_base = a
    if rank_coef > 0.0 and rank_noise > 0.0:
        a_neg = rank_base.detach() + float(rank_noise) * torch.randn_like(rank_base)
        q_neg = model.q_min_chunk(state, a_neg, t=t_end)
        rank = rank + float(rank_coef) * (F.relu(q_neg - q_pos + float(rank_margin)) * w).mean()
        q_gap = q_gap + (q_pos - q_neg).detach().mean()
    if far_rank_coef > 0.0 and far_rank_noise > 0.0:
        a_far = rank_base.detach() + float(far_rank_noise) * torch.randn_like(rank_base)
        q_far = model.q_min_chunk(state, a_far, t=t_end)
        rank = rank + float(far_rank_coef) * (
            F.relu(q_far - q_pos + float(rank_margin)) * w
        ).mean()
        q_gap = q_gap + (q_pos - q_far).detach().mean()
    if shuffle_rank_coef > 0.0 and a.shape[0] > 1:
        perm = torch.randperm(a.shape[0], device=a.device)
        a_shuf = rank_base.detach()[perm]
        q_shuf = model.q_min_chunk(state, a_shuf, t=t_end)
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
) -> dict[str, float]:
    """Flow critic TD whose reverse and bootstrap fields come from Molmo AE."""
    del ref_dropout
    if not model.is_flow:
        raise RuntimeError("ae_flow_critic_td_step requires cf_mode=flow")
    model.train()
    ae_backend.eval()
    detach_token = not model.tune_token_online
    state = _batch_state(model, batch, detach_token=detach_token, use_target=False)
    actions = model.normalize_action(batch["executed_actions"])
    reference = model.normalize_action(batch["reference_actions"])
    contexts = _ae_batch_contexts(model, ae_backend, batch)
    provider = _ae_velocity_provider(model, ae_backend, contexts)

    with torch.no_grad():
        next_state = _batch_state(
            model,
            batch,
            next_state=True,
            detach_token=True,
            use_target=True,
        )
        next_reference = model.normalize_action(batch["next_reference_actions"])
        next_contexts = _ae_batch_contexts(
            model,
            ae_backend,
            batch,
            next_state=True,
        )
        next_provider = _ae_velocity_provider(model, ae_backend, next_contexts)
        next_x0_raw = torch.randn_like(batch["next_reference_actions"])
        next_x0 = model.normalize_action(next_x0_raw)
        next_action = _flow_endpoint_with_provider(
            model,
            next_state,
            next_reference,
            next_provider,
            apply_guide=model.guide is not None,
            x0=next_x0,
        )
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
        q_next = model.q_min_chunk(
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
        x_t, t = _flow_reverse_state(
            model,
            state,
            actions,
            reference,
            apply_guide=model.guide is not None,
            velocity_provider=provider,
        )

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
    q_pos = model.q_min_chunk(state, actions, t=t_end)
    if rank_coef > 0.0 and rank_noise > 0.0:
        action_neg = actions.detach() + float(rank_noise) * torch.randn_like(actions)
        q_neg = model.q_min_chunk(state, action_neg, t=t_end)
        rank = rank + float(rank_coef) * (
            F.relu(q_neg - q_pos + float(rank_margin)) * weight
        ).mean()
        q_gap = q_gap + (q_pos - q_neg).detach().mean()
    if far_rank_coef > 0.0 and far_rank_noise > 0.0:
        action_far = actions.detach() + float(far_rank_noise) * torch.randn_like(actions)
        q_far = model.q_min_chunk(state, action_far, t=t_end)
        rank = rank + float(far_rank_coef) * (
            F.relu(q_far - q_pos + float(rank_margin)) * weight
        ).mean()
        q_gap = q_gap + (q_pos - q_far).detach().mean()
    if shuffle_rank_coef > 0.0 and actions.shape[0] > 1:
        perm = torch.randperm(actions.shape[0], device=actions.device)
        action_shuffled = actions.detach()[perm]
        q_shuffled = model.q_min_chunk(state, action_shuffled, t=t_end)
        rank = rank + float(shuffle_rank_coef) * (
            F.relu(q_shuffled - q_pos + float(rank_margin)) * weight
        ).mean()
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
) -> dict[str, float]:
    """Paper-faithful joint CF actor: BC on reference + Q lookahead through v_θ.

    ConsensusFlow trains the base flow v_θ jointly with G_φ:
      L = L_BC(ã - x0) + L_actor(-Q(s, x^+, t^+)) + β||a_end - ã||^2
    with x^+ = x_t + (v_θ + sg(G)) Δ.  Absolute Q (not adv vs BC velocity)
    routes value into the trainable base Action Expert (FlowVelocityActor).

    v12: BC targets the VLA reference by default (bc_ref_coef=1), not noisy
    executed actions — executed was corrupted by explore_residual_std.
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
        )
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

    # Paper lookahead: sg(G) so actor grads flow through v_θ only.
    g = torch.zeros_like(v)
    if model.guide is not None:
        g, _, _ = model.guide.guidance(state, x_t, t, v.detach())
        g = g.detach()
    dt = torch.minimum(
        torch.full_like(t, 1.0 / float(model.flow_steps)),
        1.0 - t,
    )
    x_plus = x_t + (v + g) * dt.view(b, 1, 1)
    t_plus = (t + 1.0 / float(model.flow_steps)).clamp(0.0, 1.0)
    # Absolute ensemble-mean Q at the guided one-step state (eq. cf-actor).
    q_look = model.q_chunk(state, x_plus, t=t_plus).mean(dim=0)
    with torch.no_grad():
        q_base = model.q_chunk(state, x_t.detach(), t=t).mean(dim=0)
    adv = q_look - q_base

    # Endpoint auxiliary: short guided unroll so endpoint Q shapes v_θ.
    n_aux = max(1, min(int(endpoint_aux_steps), int(model.flow_steps)))
    x_end = x_t
    t_end = t
    for _ in range(n_aux):
        v_i = model.flow_velocity(state, x_end, t_end, ref_in)
        g_i = torch.zeros_like(v_i)
        if model.guide is not None:
            g_i, _, _ = model.guide.guidance(state, x_end.detach(), t_end, v_i.detach())
            g_i = g_i.detach()
        dt_i = torch.minimum(
            torch.full_like(t_end, 1.0 / float(model.flow_steps)),
            1.0 - t_end,
        )
        x_end = x_end + (v_i + g_i) * dt_i.view(b, 1, 1)
        t_end = (t_end + 1.0 / float(model.flow_steps)).clamp(0.0, 1.0)
    q_end = model.q_chunk(state, x_end, t=t_end).mean(dim=0)

    # Trust region vs VLA reference (not explore-corrupted executed).
    endpoint_det, _ = model.flow_sample(state, ref_in, apply_guide=False)
    residual_mse = (
        ((endpoint_det - a_ref.detach()) * mask) ** 2
    ).sum() / mask.sum().clamp_min(1.0) / endpoint_det.shape[-1]
    alpha = model.log_alpha()
    actor_loss = (
        -q_look.mean()
        - float(endpoint_aux_coef) * q_end.mean()
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
        "actor_adv": float(adv.mean().detach()),
        "actor_q_look": float(q_look.mean().detach()),
        "actor_q_end": float(q_end.mean().detach()),
        "residual_mse": float(residual_mse.detach()),
        "bc_loss": float(bc.detach()),
        "bc_ref_coef": float(ref_w),
        "alpha": float(alpha.detach()),
    }


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
) -> dict[str, float]:
    """Joint CF actor with MolmoAct2 Action Expert as \(v_\\theta\).

    Requires ``batch`` keys from ImageChunkReplay (cameras + instruction) plus
    standard chunk fields.  VLM KV is stop-grad (knowledge insulation); LoRA AE
    receives BC + Q lookahead through ``v_AE + sg(G)``.

    v12: BC toward VLA/AE reference by default (not explore-corrupted executed).
    """
    if not model.is_flow:
        raise RuntimeError("ae_flow_actor_step requires cf_mode=flow")
    model.train()
    ae_backend.train(True)

    b = int(batch["executed_actions"].shape[0])
    device = batch["executed_actions"].device

    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        contexts = _ae_batch_contexts(model, ae_backend, batch)
    velocity_provider = _ae_velocity_provider(model, ae_backend, contexts)

    a_data_raw = batch["executed_actions"].to(device=device, dtype=torch.float32)
    a_ref_raw = batch["reference_actions"].to(device=device, dtype=torch.float32)
    ref_w = float(max(0.0, min(1.0, bc_ref_coef)))
    a_bc_raw = ref_w * a_ref_raw + (1.0 - ref_w) * a_data_raw
    a_data = model.normalize_action(a_data_raw)
    a_ref = model.normalize_action(a_ref_raw)
    a_bc = model.normalize_action(a_bc_raw)
    mask = batch["action_mask"].unsqueeze(-1)
    x0_raw = torch.randn_like(a_bc_raw)
    t = torch.rand(b, 1, device=device, dtype=torch.float32)
    x_t_raw = (1.0 - t.view(b, 1, 1)) * x0_raw + t.view(b, 1, 1) * a_bc_raw
    target_v_raw = a_bc_raw - x0_raw

    std = model.action_std.to(device=device, dtype=torch.float32).clamp_min(1e-6)
    x_t = model.normalize_action(x_t_raw)
    v_n = velocity_provider(state, x_t, t, a_ref)
    target_v = target_v_raw / std
    bc = (((v_n - target_v.detach()) * mask) ** 2).sum() / mask.sum().clamp_min(1.0) / v_n.shape[-1]

    g = torch.zeros_like(v_n)
    if model.guide is not None:
        g, _, _ = model.guide.guidance(state, x_t, t, v_n.detach())
        g = g.detach()
    dt = torch.minimum(
        torch.full_like(t, 1.0 / float(model.flow_steps)),
        1.0 - t,
    )
    x_plus = x_t + (v_n + g) * dt.view(b, 1, 1)
    t_plus = (t + 1.0 / float(model.flow_steps)).clamp(0.0, 1.0)
    q_look = model.q_chunk(state, x_plus, t=t_plus).mean(dim=0)
    with torch.no_grad():
        q_base = model.q_chunk(state, x_t.detach(), t=t).mean(dim=0)
    adv = q_look - q_base

    n_aux = max(1, min(int(endpoint_aux_steps), int(model.flow_steps)))
    x_end = x_t
    t_end = t
    for _ in range(n_aux):
        v_i = velocity_provider(state, x_end, t_end, a_ref)
        g_i = torch.zeros_like(v_i)
        if model.guide is not None:
            g_i, _, _ = model.guide.guidance(state, x_end.detach(), t_end, v_i.detach())
            g_i = g_i.detach()
        dt_i = torch.minimum(
            torch.full_like(t_end, 1.0 / float(model.flow_steps)),
            1.0 - t_end,
        )
        x_end = x_end + (v_i + g_i) * dt_i.view(b, 1, 1)
        t_end = (t_end + 1.0 / float(model.flow_steps)).clamp(0.0, 1.0)
    q_end = model.q_chunk(state, x_end, t=t_end).mean(dim=0)

    residual_mse = (
        ((x_end - a_ref.detach()) * mask) ** 2
    ).sum() / mask.sum().clamp_min(1.0) / x_end.shape[-1]
    alpha = model.log_alpha()
    actor_loss = (
        -q_look.mean()
        - float(endpoint_aux_coef) * q_end.mean()
        + float(bc_coef) * bc
        + float(beta) * residual_mse
        + alpha.detach() * residual_mse
    )
    opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    grad_norm = 0.0
    for p in ae_backend.trainable_parameters():
        if p.grad is not None:
            grad_norm += float(p.grad.detach().float().norm().cpu())
    nn.utils.clip_grad_norm_(ae_backend.trainable_parameters(), 1.0)
    opt.step()

    constraint = (residual_mse - target_divergence).detach()
    alpha_loss = -model.log_alpha.log_alpha * constraint
    alpha_opt.zero_grad(set_to_none=True)
    alpha_loss.backward()
    alpha_opt.step()

    return {
        "actor_loss": float(actor_loss.detach()),
        "actor_adv": float(adv.mean().detach()),
        "actor_q_look": float(q_look.mean().detach()),
        "actor_q_end": float(q_end.mean().detach()),
        "residual_mse": float(residual_mse.detach()),
        "bc_loss": float(bc.detach()),
        "bc_ref_coef": float(ref_w),
        "alpha": float(alpha.detach()),
        "ae_grad_norm": float(grad_norm),
        "v_source": 1.0,  # marker: molmo_ae
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
    """CF guide distill with base velocity from MolmoAct AE (detached)."""
    del beta, target_delta_frac
    if model.guide is None:
        return {"guide_loss": 0.0, "guide_adv": 0.0}
    if not model.is_flow:
        return guide_step(model, opt, batch, beta=beta)
    model.train()
    ae_backend.eval()

    b = int(batch["reference_actions"].shape[0])
    device = batch["reference_actions"].device

    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        a1_raw = batch["reference_actions"].to(device=device, dtype=torch.float32)
        a1 = model.normalize_action(a1_raw)
        contexts = _ae_batch_contexts(model, ae_backend, batch)
    velocity_provider = _ae_velocity_provider(model, ae_backend, contexts)

    x0_raw = torch.randn_like(a1_raw)
    t = 0.5 + 0.5 * torch.rand(b, 1, device=device, dtype=torch.float32)
    x_t_raw = ((1.0 - t.view(b, 1, 1)) * x0_raw + t.view(b, 1, 1) * a1_raw).detach()
    x_t = model.normalize_action(x_t_raw).detach().requires_grad_(True)

    target = stochastic_target_critic_gradient(
        model,
        state,
        x_t,
        t=t,
    )

    with torch.no_grad():
        v = velocity_provider(state, x_t.detach(), t, a1).detach()

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
        q_g = model.q_min_chunk(state.detach(), x_guided, t=t_plus)
        q_b = model.q_min_chunk(state.detach(), x_base, t=t_plus)
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
        "guide_distill": float(distill.detach()),
        "guide_mse": float(mag.detach()),
        "w_norm": float(w_flat.norm(dim=-1).mean().detach()),
        "target_norm": float(target.norm(dim=-1).mean().detach()),
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
) -> dict[str, float]:
    """Distill stochastic target-critic gradients into raw W."""
    del beta, target_delta_frac
    if model.guide is None:
        return {"guide_loss": 0.0, "guide_adv": 0.0}
    if not model.is_flow:
        return guide_step(model, opt, batch, beta=beta)
    model.train()
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True, use_target=False)
        a1 = model.normalize_action(batch["reference_actions"])
    b = a1.shape[0]
    x0 = torch.randn_like(a1)
    # Bias BC times toward the endpoint where the fixed critic is informative.
    t = 0.5 + 0.5 * torch.rand(b, 1, device=a1.device, dtype=a1.dtype)
    x_t = ((1.0 - t.view(b, 1, 1)) * x0 + t.view(b, 1, 1) * a1).detach().requires_grad_(True)
    target = stochastic_target_critic_gradient(
        model,
        state,
        x_t,
        t=t,
    )

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
        q_g = model.q_min_chunk(state.detach(), x_guided, t=t_plus)
        q_b = model.q_min_chunk(state.detach(), x_base, t=t_plus)
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
        "guide_distill": float(distill.detach()),
        "guide_mse": float(mag.detach()),
        "w_norm": float(w_flat.norm(dim=-1).mean().detach()),
        "target_norm": float(target.norm(dim=-1).mean().detach()),
    }
    for k, v_t in diag.items():
        out[f"guide_{k}"] = float(v_t.detach()) if torch.is_tensor(v_t) else float(v_t)
    return out
