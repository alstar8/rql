#!/usr/bin/env python
"""Build a tiny but *realistic* distillation shard for smoke-testing the V25 trainer.

The self-check in train_expert_distill.py only means something if the recorded
`reference` chunks are genuine samples of the frozen model. Random actions are not, so
this generator runs `policy.infer` on synthetic observations and uses the model's own
output as the reference; the `teacher` is that reference's committed 8 steps plus a small
perturbation (a stand-in for the V+G correction). The self-check should then report
reference loss < teacher loss.

    CUDA_VISIBLE_DEVICES=0 HF_HOME=... python scripts/_distill_smoke_data.py --out /tmp/v25_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_expert_distill import BASE_CHECKPOINT, build_policy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    policy, _ = build_policy(BASE_CHECKPOINT)
    rng = np.random.default_rng(args.seed)

    externals, wrists, states, instrs, refs, teas = [], [], [], [], [], []
    for i in range(args.n):
        ext = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        wri = rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
        state = (rng.standard_normal(8) * 0.1).astype(np.float32)
        raw = {
            "observation/exterior_image_1_left": ext,
            "observation/wrist_image_left": wri,
            "observation/joint_position": state[:7],
            "observation/gripper_position": state[7:8],
            "prompt": "pick up the object",
        }
        actions = np.asarray(policy.infer(raw)["actions"], dtype=np.float32)
        reference = actions.reshape(actions.shape[0], -1)[:, :8]  # (16, 8) physical
        teacher = reference[:8] + 0.05 * rng.standard_normal((8, 8)).astype(np.float32)
        externals.append(ext)
        wrists.append(wri)
        states.append(state)
        instrs.append("pick up the object")
        refs.append(reference.astype(np.float32))
        teas.append(teacher.astype(np.float32))
        if (i + 1) % 8 == 0:
            print(f"generated {i + 1}/{args.n}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "distill_smoke_pid0_s0000.npz"
    np.savez(
        path,
        external_cam=np.stack(externals),
        wrist_cam=np.stack(wrists),
        state=np.stack(states),
        instruction=np.asarray(instrs),
        reference=np.stack(refs),
        teacher=np.stack(teas),
    )
    print(f"wrote {path} reference={np.stack(refs).shape} teacher={np.stack(teas).shape}")


if __name__ == "__main__":
    main()
