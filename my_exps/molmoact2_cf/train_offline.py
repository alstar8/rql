"""Offline Phase 1 (CQL critic) + Phase 2 (endpoint G RL) for MolmoAct2 CF."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from buffer import StratifiedReplay, load_buffer
from models import MolmoAct2CF


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann–Whitney AUROC without sklearn."""
    labels = labels.astype(np.float64)
    scores = scores.astype(np.float64)
    pos = scores[labels > 0.5]
    neg = scores[labels <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(tie)
    # Use ranking
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg
        i = j
    sum_pos_ranks = ranks[labels > 0.5].sum()
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _batch_state(
    model: MolmoAct2CF,
    batch: dict[str, torch.Tensor],
    *,
    detach: bool = False,
) -> torch.Tensor:
    """Encode x=(z, s) when VLA features are present; else use flat states."""
    if "features" in batch and "proprio" in batch and model.use_vla_features:
        state = model.encode_state(batch["features"], batch["proprio"])
    else:
        state = batch["states"]
    return state.detach() if detach else state


def build_optimizers(
    model: MolmoAct2CF,
    *,
    lr_q: float,
    lr_g: float,
    lr_alpha: float,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer]:
    """Create disjoint optimizers.

    The feature projector is critic-owned. Updating it from both Adam
    optimizers gave the first VLA run conflicting optimizer moments and let G
    reshape the critic representation to manufacture advantage.
    """
    q_params = list(model.critic.parameters())
    if model.projector is not None:
        q_params += list(model.projector.parameters())
    opt_q = torch.optim.Adam(q_params, lr=lr_q)
    opt_g = torch.optim.Adam(model.refiner.parameters(), lr=lr_g)
    opt_alpha = torch.optim.Adam(model.log_alpha.parameters(), lr=lr_alpha)
    return opt_q, opt_g, opt_alpha


def critic_is_healthy(info: dict[str, float], max_mc_loss: float) -> bool:
    """Gate G on a finite, calibrated bounded critic."""
    required = ("critic_loss", "mc_loss", "q_min", "q_max", "cql_loss")
    if any(key not in info or not np.isfinite(info[key]) for key in required):
        return False
    return (
        info["mc_loss"] <= max_mc_loss
        and -1e-6 <= info["q_min"]
        and info["q_max"] <= 1.0 + 1e-6
        and info["cql_loss"] >= -1e-6
    )


def critic_step(
    model: MolmoAct2CF,
    batch: dict[str, torch.Tensor],
    opt: torch.optim.Optimizer,
    *,
    cql_coef: float,
    cql_n_actions: int,
    cql_action_radius: float = 0.05,
    cql_margin: float = 0.0,
    cql_far_scale: float = 1.0,
) -> dict[str, float]:
    model.train()
    s = _batch_state(model, batch)
    # ``actions`` must be the deployed behavior action. ``base_actions`` is
    # retained separately for the residual update.
    a = batch["actions"]
    # Full episodes are available, so discounted sparse MC returns give much
    # faster credit assignment than a 500-step one-step bootstrap.
    y = batch["returns"].clamp(0.0, 1.0)
    qs = model.critic(s, a)  # (E, B)
    mc = ((qs - y.unsqueeze(0)) ** 2).mean()
    cql, cql_info = model.critic.cql_penalty(
        s,
        a,
        n_actions=cql_n_actions,
        coef=cql_coef,
        action_radius=cql_action_radius,
        margin=cql_margin,
        far_scale=cql_far_scale,
    )
    loss = mc + cql
    if not torch.isfinite(loss):
        raise FloatingPointError(
            f"non-finite critic loss: mc={float(mc.detach())} cql={float(cql.detach())}"
        )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = list(model.critic.parameters())
    if model.projector is not None:
        params += list(model.projector.parameters())
    grad_norm = nn.utils.clip_grad_norm_(params, 5.0)
    opt.step()
    model.soft_update_target(0.005)
    return {
        "critic_loss": float(loss.item()),
        "mc_loss": float(mc.item()),
        # Backward-compatible metric name for existing aggregation scripts.
        "td_loss": float(mc.item()),
        "q_mean": float(qs.mean().item()),
        "q_min": float(qs.min().item()),
        "q_max": float(qs.max().item()),
        "q_std": float(qs.std().item()),
        "cql_loss": float(cql_info["cql_loss"].item()),
        "cql_gap": float(cql_info["cql_gap"].item()),
        "target_mean": float(y.mean().item()),
        "grad_norm": float(grad_norm.item()),
    }


def refiner_step(
    model: MolmoAct2CF,
    batch: dict[str, torch.Tensor],
    opt_g: torch.optim.Optimizer,
    opt_alpha: torch.optim.Optimizer,
    *,
    target_divergence: float,
) -> dict[str, float]:
    model.train()
    # Projector/critic are fixed for the G step; only G sees gradients.
    s = _batch_state(model, batch, detach=True)
    a_v = batch.get("base_actions", batch["actions"])
    refined, delta = model.refiner.refine(
        s,
        a_v,
        delta_clip=model.refiner.max_delta,
    )
    # Maximize advantage vs frozen base action. Absolute Q is nearly flat after
    # MC regression on sparse returns, so -Q alone left G inert (adv≈2e-4).
    # Live ensemble mean is sharper than the Polyak target for this tiny actor.
    q = model.critic.q_mean(s, refined)
    with torch.no_grad():
        base_q = model.critic.q_mean(s, a_v)
    adv = q - base_q
    q_loss = -adv.mean()
    residual_mse = (delta**2).mean()
    alpha = model.log_alpha()
    trust = alpha.detach() * residual_mse
    constraint = (residual_mse - target_divergence).detach()
    alpha_loss = -model.log_alpha.log_alpha * constraint
    g_loss = q_loss + trust

    opt_g.zero_grad(set_to_none=True)
    g_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.refiner.parameters(), 5.0)
    opt_g.step()

    opt_alpha.zero_grad(set_to_none=True)
    alpha_loss.backward()
    opt_alpha.step()

    return {
        "refiner_loss": float(g_loss.item()),
        "refiner_q_mean": float(q.mean().item()),
        "base_q_mean": float(base_q.mean().item()),
        "predicted_advantage": float(adv.mean().item()),
        "residual_mse": float(residual_mse.item()),
        "residual_rms": float(residual_mse.sqrt().item()),
        "alpha": float(alpha.item()),
        "grad_norm": float(grad_norm.item()),
    }


def evaluate_critic_auroc(model: MolmoAct2CF, replay: StratifiedReplay, device: torch.device) -> float:
    model.eval()
    if "features" in replay.arrays and model.use_vla_features:
        features = torch.from_numpy(replay.arrays["features"]).to(device)
        proprio = torch.from_numpy(replay.arrays["proprio"]).to(device)
        with torch.no_grad():
            states = model.encode_state(features, proprio)
    else:
        states = torch.from_numpy(replay.arrays["states"]).to(device)
    actions = torch.from_numpy(replay.arrays["actions"]).to(device)
    labels = replay.arrays["successes"]
    with torch.no_grad():
        q = model.critic.q_mean(states, actions).cpu().numpy()
    return _auroc(q, labels)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    arrays = load_buffer(Path(args.buffer))
    replay = StratifiedReplay(arrays, pos_frac=args.pos_frac, seed=args.seed)
    use_vla = bool(args.use_vla_features) and ("features" in arrays or args.force_vla)
    model = MolmoAct2CF(
        action_dim=8,
        hidden=args.hidden,
        n_critics=args.n_critics,
        initial_alpha=args.initial_alpha,
        use_vla_features=use_vla,
    ).to(device)
    if "proprio_mean" in arrays:
        model.set_norm_stats(
            arrays["proprio_mean"],
            arrays["proprio_std"],
            arrays["action_mean"],
            arrays["action_std"],
            arrays.get("feature_mean"),
            arrays.get("feature_std"),
        )
    elif "state_mean" in arrays:
        # Legacy proprio-only buffer.
        model.set_norm_stats(
            arrays["state_mean"],
            arrays["state_std"],
            arrays["action_mean"],
            arrays["action_std"],
        )

    opt_q, opt_g, opt_alpha = build_optimizers(
        model,
        lr_q=args.lr_q,
        lr_g=args.lr_g,
        lr_alpha=args.lr_alpha,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    print(f"[phase1] critic steps={args.phase1_steps} buffer_n={len(replay)}")
    t0 = time.time()
    for step in range(1, args.phase1_steps + 1):
        batch = {k: v.to(device) for k, v in replay.sample_batch(args.batch_size).items()}
        info = critic_step(
            model,
            batch,
            opt_q,
            cql_coef=args.cql_coef,
            cql_n_actions=args.cql_n_actions,
            cql_action_radius=args.cql_action_radius,
            cql_margin=args.cql_margin,
            cql_far_scale=args.cql_far_scale,
        )
        if step % args.log_every == 0 or step == args.phase1_steps:
            auroc = evaluate_critic_auroc(model, replay, device)
            row = {"phase": 1, "step": step, "auroc": auroc, **info}
            history.append(row)
            print(
                f"  p1 step {step:5d} loss={info['critic_loss']:.4f} "
                f"td={info['td_loss']:.4f} cql={info['cql_loss']:.4f} "
                f"q={info['q_mean']:.4f} auroc={auroc:.3f}"
            )

    # Optional AE BC placeholder: smoke default is 0 (frozen MolmoAct2).
    if args.ae_bc_steps > 0:
        print(
            f"[phase1] ae_bc_steps={args.ae_bc_steps} requested but deferred "
            "(smoke uses frozen MolmoAct2; set ae_bc_steps=0). Skipping."
        )

    print(f"[phase2] refiner steps={args.phase2_steps} target_div={args.target_divergence}")
    for step in range(1, args.phase2_steps + 1):
        batch = {k: v.to(device) for k, v in replay.sample_batch(args.batch_size).items()}
        # Keep critic warm lightly
        if step % 2 == 0:
            critic_step(
                model,
                batch,
                opt_q,
                cql_coef=args.cql_coef * 0.5,
                cql_n_actions=args.cql_n_actions,
                cql_action_radius=args.cql_action_radius,
                cql_margin=args.cql_margin,
                cql_far_scale=args.cql_far_scale,
            )
        info = refiner_step(
            model,
            batch,
            opt_g,
            opt_alpha,
            target_divergence=args.target_divergence,
        )
        if step % args.log_every == 0 or step == args.phase2_steps:
            row = {"phase": 2, "step": step, **info}
            history.append(row)
            print(
                f"  p2 step {step:5d} adv={info['predicted_advantage']:.4f} "
                f"rms={info['residual_rms']:.4f} alpha={info['alpha']:.4f} "
                f"q={info['refiner_q_mean']:.4f}"
            )

    ckpt = out_dir / "molmoact2_cf.pt"
    meta = {
        "buffer": str(args.buffer),
        "phase1_steps": args.phase1_steps,
        "phase2_steps": args.phase2_steps,
        "target_divergence": args.target_divergence,
        "ae_bc_steps": args.ae_bc_steps,
        "elapsed_sec": time.time() - t0,
        "final": history[-1] if history else {},
    }
    model.save(str(ckpt), meta=meta)
    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved {ckpt}")
    print(json.dumps(meta, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--buffer", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="runs/molmoact2_cf_smoke")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n_critics", type=int, default=2)
    p.add_argument("--phase1_steps", type=int, default=2000)
    p.add_argument("--phase2_steps", type=int, default=2000)
    p.add_argument("--ae_bc_steps", type=int, default=0)
    p.add_argument("--lr_q", type=float, default=3e-4)
    p.add_argument("--lr_g", type=float, default=3e-4)
    p.add_argument("--lr_alpha", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--cql_coef", type=float, default=0.1)
    p.add_argument("--cql_n_actions", type=int, default=8)
    p.add_argument(
        "--cql_action_radius",
        type=float,
        default=0.05,
        help="Inner residual ball excluded from far-OOD CQL samples",
    )
    p.add_argument(
        "--cql_far_scale",
        type=float,
        default=1.0,
        help="Std of far-OOD Gaussian noise for CQL (outside residual ball)",
    )
    p.add_argument("--cql_margin", type=float, default=0.0)
    p.add_argument("--target_divergence", type=float, default=2.5e-3)
    p.add_argument("--initial_alpha", type=float, default=0.3)
    p.add_argument("--pos_frac", type=float, default=0.4)
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_vla_features", action="store_true", default=True)
    p.add_argument("--no_vla_features", action="store_true", default=False)
    p.add_argument("--force_vla", action="store_true", default=False)
    args = p.parse_args()
    if args.no_vla_features:
        args.use_vla_features = False
    return args


if __name__ == "__main__":
    train(parse_args())
