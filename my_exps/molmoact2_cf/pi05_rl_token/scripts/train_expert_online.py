#!/usr/bin/env python
"""Online student action expert: flow-match QUORUM chunks while RL is running.

The frozen expert stays in the HTTP servers and is the reference RLT conditions on.
This process holds a second copy of the action expert and trains it on the deployed
QUORUM chunk (`teacher` in the distill shards). The learner asks for a save every
N episodes; after that save the frozen servers copy these weights.

    python scripts/train_expert_online.py \
        --data <distill_out> --init <base_or_student> --out <student_dir> \
        --control <control_dir>
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.expert_io import freeze_backbone  # noqa: E402
from scripts.train_expert_distill import (  # noqa: E402
    BASE_CHECKPOINT,
    collate,
    load_shard,
    make_target,
    masked_flow_loss,
    save_checkpoint,
    to_model_inputs,
    build_policy,
)

log = logging.getLogger("train_expert_online")


class LiveShardStreamer:
    """Like ShardStreamer, but re-globs the data dir at the start of every epoch."""

    def __init__(self, data_dir: str | Path, seed: int = 0) -> None:
        self.data_dir = Path(data_dir)
        self.rng = np.random.default_rng(seed)

    def paths(self) -> list[Path]:
        return sorted(self.data_dir.glob("distill_*.npz"))

    def iter_epoch(self):
        paths = self.paths()
        if not paths:
            return
        order = self.rng.permutation(len(paths))
        for shard_idx in order:
            path = paths[shard_idx]
            try:
                shard = load_shard(path)
            except (FileNotFoundError, OSError, ValueError):
                continue
            n = shard["teacher"].shape[0]
            for row in self.rng.permutation(n):
                raw_obs = {
                    "observation/exterior_image_1_left": shard["external_cam"][row],
                    "observation/wrist_image_left": shard["wrist_cam"][row],
                    "observation/joint_position": shard["state"][row][:7],
                    "observation/gripper_position": shard["state"][row][7:8],
                    "prompt": str(shard["instruction"][row]),
                }
                yield raw_obs, shard["teacher"][row], shard["reference"][row]


def read_command(control_dir: Path) -> str:
    path = control_dir / "command"
    if not path.exists():
        return "run"
    text = path.read_text().strip()
    return text or "run"


def write_status(control_dir: Path, payload: dict) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    dest = control_dir / "status.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(dest)


def wait_for_data(streamer: LiveShardStreamer, control_dir: Path) -> None:
    """Block until collectors write the first shard, or the learner asks us to stop."""
    while True:
        cmd = read_command(control_dir)
        if cmd == "stop":
            raise SystemExit(0)
        if streamer.paths():
            return
        time.sleep(2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--init", default=BASE_CHECKPOINT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

    control_dir = Path(args.control)
    control_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / "command").write_text("run")

    device = "cuda"
    init = args.init
    out_weights = Path(args.out) / "model.safetensors"
    if out_weights.exists():
        init = str(args.out)
        log.info("resuming student weights from %s", init)
    policy, model = build_policy(init)
    n_train, n_frozen = freeze_backbone(model)
    log.info("trainable %dM (expert+proj), frozen %dM (backbone)", n_train // 10**6, n_frozen // 10**6)

    streamer = LiveShardStreamer(args.data, seed=args.seed)
    log.info("waiting for distill shards in %s", args.data)
    wait_for_data(streamer, control_dir)
    log.info("first shards: %d files", len(streamer.paths()))

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, eps=1e-8
    )

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        return args.lr

    prev: dict = {}
    status_path = control_dir / "status.json"
    if status_path.exists():
        try:
            prev = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            prev = {}

    model.train()
    epoch = streamer.iter_epoch()
    batch: list = []
    running: list[float] = []
    started = time.time()
    step = int(prev.get("step") or 0)
    generation = int(prev.get("generation") or 0)
    last_loss = float(prev.get("loss") if prev.get("loss") is not None else float("nan"))
    if generation or step:
        log.info("resuming student loop at step %d generation %d", step, generation)

    def status(**extra) -> dict:
        payload = {
            "step": step,
            "generation": generation,
            "loss": last_loss,
            "shards": len(streamer.paths()),
            "hours": round((time.time() - started) / 3600, 3),
            "saved": False,
            **extra,
        }
        write_status(control_dir, payload)
        return payload

    status()
    while True:
        cmd = read_command(control_dir)
        if cmd == "stop":
            save_checkpoint(model, args.init, args.out)
            generation += 1
            status(saved=True)
            log.info("stop: saved generation %d at step %d", generation, step)
            return
        if cmd == "save":
            save_checkpoint(model, args.init, args.out)
            generation += 1
            status(saved=True)
            log.info("save: generation %d step %d loss=%.4f shards=%d",
                     generation, step, last_loss, len(streamer.paths()))
            while read_command(control_dir) == "save":
                time.sleep(0.5)
            continue

        while len(batch) < args.batch_size:
            try:
                raw_obs, teacher, reference = next(epoch)
            except StopIteration:
                epoch = streamer.iter_epoch()
                if not streamer.paths():
                    time.sleep(1.0)
                break
            batch.append(to_model_inputs(policy, raw_obs, make_target(teacher, reference)))
        if len(batch) < args.batch_size:
            continue

        step += 1
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        observation, actions = collate(batch, device)
        batch = []
        loss = masked_flow_loss(model, observation, actions)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        last_loss = float(loss)
        running.append(last_loss)
        if step % args.log_every == 0:
            log.info(
                "step %d loss=%.4f lr=%.2e shards=%d (%.1f it/s)",
                step, float(np.mean(running)), lr_at(step), len(streamer.paths()),
                step / max(time.time() - started, 1e-9),
            )
            running = []
            status()


if __name__ == "__main__":
    main()
