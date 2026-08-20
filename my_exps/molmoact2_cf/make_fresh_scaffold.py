"""Create a fresh (random-init) RLT scaffold checkpoint for data collection.

The V20 pretrain chain must not reuse any previously trained artifact. The
collector (`train_rlt_online.py --actor_mode vla_only`) only needs a scaffold
to export VLA tokens and record chunks; the token AE inside it is re-trained
by warmup afterwards and every chunk's z is re-encoded, so random weights are
correct here. Norm stats are identity (mean 0 / std 1) so nothing is silently
scaled by stale statistics.
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

from rlt_models import MolmoAct2RLTCF  # noqa: E402

log = logging.getLogger("molmoact2_cf.make_fresh_scaffold")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_ckpt", required=True)
    parser.add_argument("--z_dim", type=int, default=256)
    parser.add_argument("--token_d_model", type=int, default=512)
    parser.add_argument("--token_layers", type=int, default=4)
    parser.add_argument("--token_heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(int(args.seed))
    model = MolmoAct2RLTCF(
        z_dim=int(args.z_dim),
        token_d_model=int(args.token_d_model),
        token_layers=int(args.token_layers),
        token_heads=int(args.token_heads),
        use_cf_guide=False,
        tune_token_online=False,
    )
    # Identity norm stats: the scaffold must not smuggle in stale statistics.
    model.set_norm_stats(
        torch.zeros(model.proprio_dim),
        torch.ones(model.proprio_dim),
        torch.zeros(model.action_dim),
        torch.ones(model.action_dim),
        torch.zeros(model.feature_dim),
        torch.ones(model.feature_dim),
    )
    out = Path(args.out_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.tmp")
    model.save(
        str(temporary),
        meta={
            "fresh_scaffold": True,
            "identity_norm_stats": True,
            "seed": int(args.seed),
            "purpose": "vla_only token-export scaffold for offline collection",
        },
    )
    temporary.replace(out)
    log.info("wrote fresh scaffold %s", out)


if __name__ == "__main__":
    main()
