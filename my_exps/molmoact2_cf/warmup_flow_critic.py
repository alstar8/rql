"""Offline pretrain for flow-time ConsensusFlow critic (+ BC actor) on demo chunks.

Starts from a residual/token AE checkpoint, builds time-dependent Q_k(s,x,t),
FlowVelocityActor, and FlowCFGuide, then runs reverse-state TD + BC updates.
Guide remains randomly initialized (distilled online), matching residual pretrain.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from chunk_replay import ChunkReplay  # noqa: E402
from rlt_models import MolmoAct2RLTCF  # noqa: E402
from train_rlt import (  # noqa: E402
    action_sensitivity,
    build_rlt_optimizers,
    cfgrl_actor_step,
    critic_is_healthy,
    flow_actor_step,
    flow_critic_td_step,
)

log = logging.getLogger("molmoact2_cf.warmup_flow_critic")


def warmup(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    chunk_replay = ChunkReplay.load_npz(args.chunk_replay)
    log.info("loaded chunks=%d episodes=%d", len(chunk_replay), chunk_replay.n_episodes)

    model = MolmoAct2RLTCF.from_token_ckpt_as_flow(
        args.rlt_ckpt,
        map_location=device,
        use_cf_guide=True,
        n_critics=args.n_critics,
        flow_steps=args.flow_steps,
        guidance_coef=args.guidance_coef,
        hidden=args.hidden,
        n_hidden_actor=args.n_hidden_actor,
        n_hidden_critic=args.n_hidden_critic,
        z_expand_dim=args.z_expand_dim,
        layernorm_heads=args.layernorm_heads,
        use_cfgrl=args.use_cfgrl,
        cfgrl_o_dim=args.cfgrl_o_dim,
        cfgrl_w=args.cfgrl_w,
    ).to(device)
    model.freeze_token_encoder()
    log.info(
        "built flow CF from %s | params critic=%d actor=%d guide=%d cfgrl=%s",
        args.rlt_ckpt,
        sum(p.numel() for p in model.critic.parameters()),
        sum(p.numel() for p in model.actor.parameters()),
        sum(p.numel() for p in model.guide.parameters()) if model.guide is not None else 0,
        model.use_cfgrl,
    )
    if args.use_cfgrl and not model.use_cfgrl:
        raise RuntimeError("born-CFGRL pretrain requested but model is not CFGRL")

    optimizers = build_rlt_optimizers(model, lr_critic=args.lr, lr_actor=args.lr_actor)
    metrics_path = Path(args.out_ckpt).with_name("flow_critic_warmup_metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    last_q: dict[str, float] = {}
    last_actor: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        batch = chunk_replay.sample(args.batch_size, device=device)
        last_q = flow_critic_td_step(
            model,
            optimizers["critic"],
            batch,
            gamma=args.gamma,
            mc_coef=args.mc_coef,
            cql_coef=args.cql_coef,
            cql_n_actions=args.cql_n_actions,
            rank_coef=args.rank_coef,
            rank_margin=args.rank_margin,
            rank_noise=args.rank_noise,
            far_rank_coef=args.far_rank_coef,
            far_rank_noise=args.far_rank_noise,
            shuffle_rank_coef=args.shuffle_rank_coef,
            target_noise=args.target_noise,
        )
        if step % args.actor_every == 0:
            if model.use_cfgrl:
                # Born-CFGRL pretrain: success-labeled flow matching with
                # condition dropout, so the o-conditional actor the online
                # run inherits is already trained (no handoff re-init).
                last_actor = cfgrl_actor_step(
                    model,
                    optimizers["actor"],
                    batch,
                    cond_dropout=args.cfgrl_dropout,
                    ref_dropout=args.cfgrl_ref_dropout,
                    use_advantage_labels=False,
                )
            else:
                last_actor = flow_actor_step(
                    model,
                    optimizers["actor"],
                    optimizers["alpha"],
                    batch,
                    beta=args.actor_beta,
                    target_divergence=args.target_divergence,
                    ref_dropout=args.ref_dropout,
                    bc_coef=args.bc_coef,
                    q_coef=float(getattr(args, "actor_q_coef", 0.0)),
                )
        if step % args.log_every == 0 or step == args.steps:
            sens = action_sensitivity(model, batch, noise=args.rank_noise)
            healthy = critic_is_healthy(model, batch)
            row = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "action_sensitivity": float(sens),
                "critic_healthy": bool(healthy),
                **{k: float(v) for k, v in last_q.items()},
                **{f"actor_{k}": float(v) for k, v in last_actor.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            log.info(
                "step=%d/%d td=%.4f bc=%.4f rank_gap=%.4f q_std=%.4f sens=%.4f healthy=%s",
                step,
                args.steps,
                row.get("q_td_loss", 0.0),
                row.get("actor_bc_loss", 0.0),
                row.get("q_rank_gap", 0.0),
                row.get("q_std", 0.0),
                sens,
                healthy,
            )

    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    model.save(
        str(temporary),
        meta={
            "flow_critic_warmup_steps": args.steps,
            "chunk_transitions": len(chunk_replay),
            "chunk_replay": args.chunk_replay,
            "token_ckpt": args.rlt_ckpt,
            "cf_mode": "flow",
            "metrics": {**last_q, **last_actor},
        },
    )
    os.replace(temporary, out)
    log.info("saved %s", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlt_ckpt", required=True, help="Token AE (or residual) checkpoint")
    parser.add_argument("--chunk_replay", required=True, help="Prefer reencoded NPZ with z filled")
    parser.add_argument("--out_ckpt", required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_critics", type=int, default=10)
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--guidance_coef", type=float, default=0.5)
    # Head architecture: must match the online run so the handoff is a pure
    # load (as_cfgrl fast-path), not a silent re-init.
    parser.add_argument("--hidden", type=int, default=1_024)
    parser.add_argument("--n_hidden_actor", type=int, default=10)
    parser.add_argument("--n_hidden_critic", type=int, default=5)
    parser.add_argument("--z_expand_dim", type=int, default=512)
    parser.add_argument(
        "--layernorm_heads",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use_cfgrl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build a born-CFGRL actor (o-conditioned) and pretrain it with CFGRL labels",
    )
    parser.add_argument("--cfgrl_o_dim", type=int, default=16)
    parser.add_argument("--cfgrl_w", type=float, default=1.0)
    parser.add_argument("--cfgrl_dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--actor_every", type=int, default=1)
    parser.add_argument("--bc_coef", type=float, default=1.0)
    parser.add_argument("--actor_beta", type=float, default=1.0)
    parser.add_argument(
        "--actor_q_coef",
        type=float,
        default=0.0,
        help="Q weight for flow actor during offline warmup (0 = pure BC).",
    )
    parser.add_argument("--target_divergence", type=float, default=0.0025)
    parser.add_argument("--ref_dropout", type=float, default=0.0)
    parser.add_argument(
        "--cfgrl_ref_dropout",
        type=float,
        default=0.5,
        help="reference-action dropout for the CFGRL actor; matches the "
        "critic bootstrap regime and the V20 online phase",
    )
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mc_coef", type=float, default=0.1)
    parser.add_argument("--cql_coef", type=float, default=0.1)
    parser.add_argument("--cql_n_actions", type=int, default=8)
    parser.add_argument("--rank_coef", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--rank_noise", type=float, default=0.08)
    parser.add_argument("--far_rank_coef", type=float, default=0.5)
    parser.add_argument("--far_rank_noise", type=float, default=0.35)
    parser.add_argument("--shuffle_rank_coef", type=float, default=0.5)
    parser.add_argument("--target_noise", type=float, default=0.02)
    return parser.parse_args()


if __name__ == "__main__":
    warmup(parse_args())
