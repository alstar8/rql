"""Offline RLT critic pretrain on ChunkReplay with warmed RL-token encoder."""

from __future__ import annotations

import argparse
import gc
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

from chunk_replay import ChunkReplay, TokenReplay  # noqa: E402
from rlt_models import MolmoAct2RLTCF, Z_DIM  # noqa: E402
from train_rlt import (  # noqa: E402
    action_sensitivity,
    build_rlt_optimizers,
    critic_is_healthy,
    critic_td_step,
)

log = logging.getLogger("molmoact2_cf.warmup_rlt_critic")


def _episode_runs(chunk_replay: ChunkReplay) -> list[list[int]]:
    """Split buffer into contiguous episode runs (handles colliding episode_ids)."""
    n = len(chunk_replay)
    if n == 0:
        return []
    runs: list[list[int]] = []
    cur = [0]
    for i in range(1, n):
        prev = chunk_replay.rows[i - 1]
        row = chunk_replay.rows[i]
        new_ep = int(row.episode_id) != int(prev.episode_id) or int(row.start_step) <= int(
            prev.start_step
        )
        if new_ep:
            runs.append(cur)
            cur = [i]
        else:
            cur.append(i)
    runs.append(cur)
    return runs


@torch.no_grad()
def _encode_token_arrays(
    model: MolmoAct2RLTCF,
    tokens_obj: np.ndarray,
    masks_obj: np.ndarray,
    *,
    offset: int,
    zs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> None:
    n = len(tokens_obj)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        max_s = max(int(np.asarray(masks_obj[i]).shape[0]) for i in range(start, end))
        token_dim = int(np.asarray(tokens_obj[start]).shape[-1])
        tok = np.zeros((end - start, max_s, token_dim), dtype=np.float32)
        mask = np.zeros((end - start, max_s), dtype=np.float32)
        for bi, i in enumerate(range(start, end)):
            t = np.asarray(tokens_obj[i], dtype=np.float32)
            m = np.asarray(masks_obj[i], dtype=np.float32)
            s = min(int(m.shape[0]), int(t.shape[0]), max_s)
            tok[bi, :s] = t[:s]
            mask[bi, :s] = m[:s]
        z = model.token_ae.encode(
            torch.as_tensor(tok, device=device),
            torch.as_tensor(mask, device=device),
        )
        zs[offset + start : offset + end] = z.detach().float().cpu().numpy()
        if (start // batch_size) % 50 == 0:
            log.info("encode offset=%d local=%d/%d", offset, end, n)


@torch.no_grad()
def reencode_chunk_zs_from_paths(
    model: MolmoAct2RLTCF,
    chunk_replay: ChunkReplay,
    token_paths: list[Path],
    *,
    batch_size: int = 16,
    device: torch.device,
) -> None:
    """Fill z / next_z by streaming shard NPZs (avoids loading full merged token replay)."""
    n = len(chunk_replay)
    if n == 0:
        raise RuntimeError("empty chunk replay")
    model.eval()
    zs = np.zeros((n, Z_DIM), dtype=np.float32)
    offset = 0
    for path in token_paths:
        log.info("loading token shard %s", path)
        data = np.load(str(path), allow_pickle=True)
        tokens_obj = data["tokens"]
        masks_obj = data["masks"]
        n_part = len(tokens_obj)
        if offset + n_part > n:
            raise RuntimeError(
                f"token shards overflow buffer: offset={offset} part={n_part} n={n} path={path}"
            )
        _encode_token_arrays(
            model,
            tokens_obj,
            masks_obj,
            offset=offset,
            zs=zs,
            batch_size=batch_size,
            device=device,
        )
        offset += n_part
        del data, tokens_obj, masks_obj
        gc.collect()
    if offset != n:
        raise RuntimeError(f"token shard total {offset} != chunk_replay {n}")

    for idxs in _episode_runs(chunk_replay):
        for local, gi in enumerate(idxs):
            row = chunk_replay.rows[gi]
            row.z = zs[gi]
            if local + 1 < len(idxs):
                row.next_z = zs[idxs[local + 1]]
                row.terminal = False
            else:
                row.next_z = np.zeros(Z_DIM, dtype=np.float32)
                row.terminal = True


@torch.no_grad()
def reencode_chunk_zs(
    model: MolmoAct2RLTCF,
    chunk_replay: ChunkReplay,
    chunk_token_replay: TokenReplay,
    *,
    batch_size: int = 16,
    device: torch.device,
) -> None:
    """Fill z / next_z from an in-memory TokenReplay (small buffers only)."""
    n = len(chunk_replay)
    if n == 0:
        raise RuntimeError("empty chunk replay")
    if len(chunk_token_replay) != n:
        raise RuntimeError(
            f"chunk_token_replay length {len(chunk_token_replay)} != chunk_replay {n}"
        )
    model.eval()
    zs = np.zeros((n, Z_DIM), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        max_s = max(int(chunk_token_replay.masks[i].shape[0]) for i in range(start, end))
        tok = np.zeros((end - start, max_s, chunk_token_replay.token_dim), dtype=np.float32)
        mask = np.zeros((end - start, max_s), dtype=np.float32)
        for bi, i in enumerate(range(start, end)):
            s = int(chunk_token_replay.masks[i].shape[0])
            tok[bi, :s] = chunk_token_replay.tokens[i].astype(np.float32)
            mask[bi, :s] = chunk_token_replay.masks[i].astype(np.float32)
        z = model.token_ae.encode(
            torch.as_tensor(tok, device=device),
            torch.as_tensor(mask, device=device),
        )
        zs[start:end] = z.detach().float().cpu().numpy()

    for idxs in _episode_runs(chunk_replay):
        for local, gi in enumerate(idxs):
            row = chunk_replay.rows[gi]
            row.z = zs[gi]
            if local + 1 < len(idxs):
                row.next_z = zs[idxs[local + 1]]
                row.terminal = False
            else:
                row.next_z = np.zeros(Z_DIM, dtype=np.float32)
                row.terminal = True


def _resolve_token_paths(args: argparse.Namespace) -> list[Path] | None:
    if args.chunk_token_shards:
        paths = sorted(Path(p) for p in args.chunk_token_shards)
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"missing token shards: {missing}")
        return paths
    if args.chunk_token_glob:
        root = Path(args.chunk_token_glob)
        if root.is_dir():
            paths = sorted(root.glob("chunk_token_replay_*_s*.npz"))
        else:
            paths = sorted(Path().glob(args.chunk_token_glob))
            if not paths:
                paths = sorted(Path(args.chunk_token_glob).parent.glob(Path(args.chunk_token_glob).name))
        # Prefer shard files; never include merged.
        paths = [p for p in paths if "_s" in p.stem and "merged" not in p.name]
        if not paths:
            raise FileNotFoundError(f"no shard matches for {args.chunk_token_glob}")
        return paths
    return None


def warmup(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    chunk_replay = ChunkReplay.load_npz(args.chunk_replay)
    log.info("loaded chunks=%d episodes=%d", len(chunk_replay), chunk_replay.n_episodes)

    model = MolmoAct2RLTCF.load(args.rlt_ckpt, map_location=device).to(device)
    model.freeze_token_encoder()

    token_paths = _resolve_token_paths(args)
    if token_paths is not None:
        log.info("streaming re-encode from %d token shards", len(token_paths))
        for p in token_paths:
            log.info("  shard %s", p)
        reencode_chunk_zs_from_paths(
            model,
            chunk_replay,
            token_paths,
            batch_size=args.encode_batch_size,
            device=device,
        )
    else:
        log.info("loading full chunk_token_replay %s", args.chunk_token_replay)
        chunk_token_replay = TokenReplay.load_npz(args.chunk_token_replay)
        log.info("loaded chunk_tokens=%d", len(chunk_token_replay))
        reencode_chunk_zs(
            model,
            chunk_replay,
            chunk_token_replay,
            batch_size=args.encode_batch_size,
            device=device,
        )
        del chunk_token_replay
        gc.collect()

    reenc_path = Path(args.out_ckpt).with_name("chunk_replay_reencoded.npz")
    chunk_replay.save_npz(str(reenc_path))
    log.info("wrote re-encoded chunk replay %s", reenc_path)

    optimizers = build_rlt_optimizers(model, lr_critic=args.lr)
    metrics_path = Path(args.out_ckpt).with_name("critic_warmup_metrics.jsonl")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    last: dict[str, float] = {}

    for step in range(1, args.steps + 1):
        batch = chunk_replay.sample(args.batch_size, device=device)
        last = critic_td_step(
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
        if step % args.log_every == 0 or step == args.steps:
            sens = action_sensitivity(
                model,
                batch,
                noise=args.rank_noise,
            )
            healthy = critic_is_healthy(model, batch)
            row = {
                "step": step,
                "elapsed_sec": time.time() - start,
                "action_sensitivity": float(sens),
                "critic_healthy": bool(healthy),
                **{k: float(v) for k, v in last.items()},
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            log.info(
                "step=%d/%d td=%.4f rank_gap=%.4f q_std=%.4f sens=%.4f healthy=%s",
                step,
                args.steps,
                row.get("q_td_loss", 0.0),
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
            "critic_warmup_steps": args.steps,
            "chunk_transitions": len(chunk_replay),
            "chunk_replay": str(reenc_path),
            "token_ckpt": args.rlt_ckpt,
            "metrics": last,
        },
    )
    os.replace(temporary, out)
    log.info("saved %s", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlt_ckpt", required=True, help="Token-warmed RLT checkpoint")
    parser.add_argument("--chunk_replay", required=True)
    parser.add_argument(
        "--chunk_token_replay",
        default="",
        help="Merged token replay (avoid for large demos; prefer --chunk_token_glob)",
    )
    parser.add_argument(
        "--chunk_token_glob",
        default="",
        help="Dir or glob of shard files chunk_token_replay_*_s*.npz",
    )
    parser.add_argument(
        "--chunk_token_shards",
        nargs="*",
        default=None,
        help="Explicit ordered list of token shard NPZs",
    )
    parser.add_argument("--out_ckpt", required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--encode_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    args = parser.parse_args()
    if not args.chunk_token_shards and not args.chunk_token_glob and not args.chunk_token_replay:
        parser.error("need --chunk_token_shards, --chunk_token_glob, or --chunk_token_replay")
    return args


if __name__ == "__main__":
    warmup(parse_args())
