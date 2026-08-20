"""Re-encode a chunk replay's z / next_z with a warmed token encoder.

Standalone version of the re-encode stage that used to live inside the
residual-critic warmup: the V20 pretrain chain is collect -> token warmup ->
re-encode -> born-CFGRL flow pretrain, with no residual stage in between.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from chunk_replay import ChunkReplay  # noqa: E402
from rlt_models import MolmoAct2RLTCF  # noqa: E402
from warmup_rlt_critic import (  # noqa: E402
    _resolve_token_paths,
    reencode_chunk_zs_from_paths,
)

log = logging.getLogger("molmoact2_cf.reencode_chunk_replay")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlt_ckpt", required=True, help="warmed token checkpoint")
    parser.add_argument("--chunk_replay", required=True)
    parser.add_argument(
        "--chunk_token_glob",
        default="",
        help="dir or glob with chunk_token_replay_*_s*.npz shards",
    )
    parser.add_argument(
        "--chunk_token_shards",
        nargs="*",
        default=None,
        help="explicit shard paths (overrides --chunk_token_glob)",
    )
    parser.add_argument("--out_replay", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--encode_batch_size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    chunk_replay = ChunkReplay.load_npz(args.chunk_replay)
    log.info(
        "loaded chunks=%d episodes=%d", len(chunk_replay), chunk_replay.n_episodes
    )
    model = MolmoAct2RLTCF.load(args.rlt_ckpt, map_location=device).to(device)
    model.freeze_token_encoder()
    model.eval()

    token_paths = _resolve_token_paths(args)
    if not token_paths:
        raise RuntimeError(
            "no token shards found; pass --chunk_token_glob or --chunk_token_shards"
        )
    log.info("streaming re-encode from %d token shards", len(token_paths))
    reencode_chunk_zs_from_paths(
        model,
        chunk_replay,
        token_paths,
        batch_size=int(args.encode_batch_size),
        device=device,
    )
    out = Path(args.out_replay)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp.npz")
    chunk_replay.save_npz(str(temporary))
    temporary.replace(out)
    log.info("wrote re-encoded chunk replay %s", out)


if __name__ == "__main__":
    main()
