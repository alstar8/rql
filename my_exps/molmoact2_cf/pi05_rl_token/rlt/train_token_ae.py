"""Phase 1: train the RL token autoencoder (L_ro). The VLA stays frozen.

    python -m rlt.train_token_ae \
        --token_replay '<runs>/rlt_pretrain_demo1k/token_replay_*_s*.npz' \
        --out runs/rlt/token_ae.pt --steps 8000
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .cli import describe, parse_into, save_config
from .config import TokenAEConfig
from .data import TokenSequences, resolve
from .logging_utils import RunLogger, default_tb_dir
from .token_ae import RLTokenAE


def main() -> None:
    cfg = parse_into(TokenAEConfig)
    print(describe(cfg))

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = torch.device(cfg.device)

    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out.with_suffix(".config.json"))
    metrics_path = out.with_suffix(".metrics.jsonl")
    tb_dir = default_tb_dir(out, cfg.tb_dir)

    if cfg.token_cache:
        print(f"mmapping token cache {cfg.token_cache}")
        data = TokenSequences.from_cache(Path(cfg.token_cache))
    else:
        paths = resolve(cfg.token_replay)
        print(f"loading {len(paths)} shard(s) ...")
        data = TokenSequences.load(paths, max_sequences=cfg.max_sequences or None)
    print("data:", json.dumps(data.stats(), indent=2))

    ae = RLTokenAE(
        z_dim=cfg.z_dim, d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers
    ).to(device)
    n_params = sum(p.numel() for p in ae.parameters())
    print(f"AE parameters: {n_params / 1e6:.2f} M")

    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr)
    start = time.time()

    with RunLogger(metrics_path, tb_dir, run_name=out.stem) as logger:
        logger.add_text("config", describe(cfg))
        for step in range(1, cfg.steps + 1):
            tokens, mask = data.sample(cfg.batch_size, device, rng)
            loss, stats = ae.reconstruction_loss(tokens, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(ae.parameters(), cfg.grad_clip)
            opt.step()

            if step % cfg.log_every == 0 or step == 1:
                stats["elapsed_sec"] = round(time.time() - start, 1)
                stats["grad_norm"] = float(grad_norm)
                logger.log(step, stats)
                print(
                    f"step {step:6d}  recon {stats['recon_loss']:8.4f}  "
                    f"|z| {stats['z_norm']:6.2f}  grad {float(grad_norm):6.2f}"
                )

    ae.save(str(out))
    print(f"\nsaved   {out}\nconfig  {out.with_suffix('.config.json')}\nmetrics {metrics_path}")


if __name__ == "__main__":
    main()
