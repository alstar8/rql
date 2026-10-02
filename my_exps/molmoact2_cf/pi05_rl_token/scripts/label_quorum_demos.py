#!/usr/bin/env python
"""Label encoded Pick-demo observations with greedy QUORUM chunks.

The obs shards and the replay buffer are the same decisions in the same order.
`agent.act(..., explore=False)` is the deployed chunk. The tail of the 16-step
target stays the frozen pi0.5 reference, which is what lowtime distillation trains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import RLConfig  # noqa: E402
from pi05.train import build  # noqa: E402

CHUNK = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--buffer", type=Path, required=True)
    parser.add_argument("--obs", type=Path, required=True)
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--token-ae", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    cfg = RLConfig(
        scene="desk_mug",
        token_ae=args.token_ae,
        algorithm="v22_24",
        beta=100.0,
        cf_actor_coef=0.0,
        cf_ref_conditioned=True,
        gate_step=0,
        gate_frac=0.0,
        out_dir=str(args.out),
        ae_finetune=False,
    )
    _, agent, _, _, _, _ = build(cfg, args.out / "build", phase="pretrain", with_runner=False)
    agent.load(str(args.agent))

    replay = np.load(args.buffer)
    states = replay["state"]
    refs = replay["reference"]
    shards = sorted(args.obs.glob("obs_*.npz"))
    args.out.mkdir(parents=True, exist_ok=True)
    cursor = 0
    for shard_path in shards:
        shard = np.load(shard_path)
        n = int(shard["reference"].shape[0])
        teachers = []
        for start in range(0, n, args.batch_size):
            stop = min(start + args.batch_size, n)
            pred = agent.act(
                states[cursor + start : cursor + stop],
                refs[cursor + start : cursor + stop],
                explore=False,
            )
            teachers.append(np.asarray(pred, dtype=np.float32).reshape(-1, CHUNK, 8))
        teacher = np.concatenate(teachers)
        if teacher.shape[0] != n:
            raise SystemExit(f"{shard_path.name}: labeled {teacher.shape[0]} of {n}")
        dest = args.out / f"distill_{shard_path.stem}.npz"
        np.savez(
            dest,
            external_cam=shard["external_cam"],
            wrist_cam=shard["wrist_cam"],
            state=shard["state"],
            instruction=shard["instruction"],
            reference=shard["reference"],
            teacher=teacher,
            executed=teacher,
        )
        cursor += n
        print(f"wrote {dest.name} ({n})", flush=True)
    if cursor != len(states):
        raise SystemExit(f"labeled {cursor} decisions, buffer has {len(states)}")
    print(f"labeled {cursor} decisions", flush=True)


if __name__ == "__main__":
    main()
