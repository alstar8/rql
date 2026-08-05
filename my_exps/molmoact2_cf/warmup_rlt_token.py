"""Offline warmup for the MolmoAct2 RLT token autoencoder."""

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

from chunk_replay import TokenReplay  # noqa: E402
from rlt_models import MolmoAct2RLTCF  # noqa: E402
from train_rlt import build_rlt_optimizers, token_step  # noqa: E402

log = logging.getLogger("molmoact2_cf.warmup_rlt_token")


def _save_model(
    model: MolmoAct2RLTCF,
    path: Path,
    meta: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    model.save(str(temporary), meta=meta)
    os.replace(temporary, path)


def warmup(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    replay_path = Path(args.token_replay)
    if not replay_path.is_file():
        raise FileNotFoundError(
            f"TokenReplay does not exist: {replay_path}. "
            "This script intentionally does not synthesize token data."
        )
    replay = TokenReplay.load_npz(str(replay_path))
    if len(replay) == 0:
        raise ValueError(f"TokenReplay is empty: {replay_path}")

    device = torch.device(args.device)
    if args.rlt_ckpt:
        checkpoint = Path(args.rlt_ckpt)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RLT checkpoint not found: {checkpoint}")
        model = MolmoAct2RLTCF.load(str(checkpoint), map_location=device).to(device)
        log.info("Loaded checkpoint %s", checkpoint)
    else:
        model = MolmoAct2RLTCF(
            feature_dim=replay.token_dim,
            use_cf_guide=args.use_cf_guide,
            tune_token_online=True,
            n_critics=args.n_critics,
        ).to(device)
        log.info("Initialized a fresh RLT model with n_critics=%d", args.n_critics)
    if model.feature_dim != replay.token_dim:
        raise ValueError(
            f"Replay token_dim={replay.token_dim} does not match "
            f"model feature_dim={model.feature_dim}"
        )
    if args.use_cf_guide and model.guide is None:
        raise ValueError("Checkpoint has no CF guide but --use_cf_guide was requested")
    if not args.use_cf_guide:
        model.guide = None
        model.use_cf_guide = False
    model.unfreeze_token_encoder()

    optimizers = build_rlt_optimizers(model, lr_token=args.lr)
    output = Path(args.out_ckpt)
    metrics_path = output.parent / "token_warmup_metrics.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    last_info: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        batch = replay.sample(args.batch_size, device=device)
        last_info = token_step(model, optimizers["token"], batch)
        if step % args.log_every == 0 or step == args.steps:
            row = {
                "step": step,
                "steps": args.steps,
                "token_sequences": len(replay),
                "elapsed_sec": time.time() - start,
                **last_info,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            log.info(
                "step=%d/%d recon=%.6f z_norm=%.4f",
                step,
                args.steps,
                last_info.get("token_recon_loss", 0.0),
                last_info.get("z_norm", 0.0),
            )

    # TD bootstrapping must start from the warmed encoder, not its random init.
    model.target_token_encoder.load_state_dict(model.token_ae.encoder.state_dict())
    meta: dict[str, object] = {
        "warmup_steps": args.steps,
        "token_sequences": len(replay),
        "token_replay": str(replay_path),
        "elapsed_sec": time.time() - start,
        "metrics": last_info,
    }
    _save_model(model, output, meta)
    log.info("Saved warmed RLT checkpoint to %s", output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token_replay",
        required=True,
        help="Required TokenReplay .npz produced by collect_token_replay.py",
    )
    parser.add_argument(
        "--out_ckpt",
        type=str,
        default=str(_HERE / "runs/rlt_token_warmup/rlt_token.pt"),
    )
    parser.add_argument("--rlt_ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_critics",
        type=int,
        default=10,
        help="Critic ensemble size stored in the warmed checkpoint (CF default K=10)",
    )
    guide_group = parser.add_mutually_exclusive_group()
    guide_group.add_argument(
        "--use_cf_guide",
        dest="use_cf_guide",
        action="store_true",
    )
    guide_group.add_argument(
        "--no_cf_guide",
        dest="use_cf_guide",
        action="store_false",
    )
    parser.set_defaults(use_cf_guide=True)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.log_every <= 0:
        parser.error("--log_every must be positive")
    return args


if __name__ == "__main__":
    warmup(parse_args())
