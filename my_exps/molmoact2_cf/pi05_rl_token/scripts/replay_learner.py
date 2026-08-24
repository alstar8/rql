"""Replay a saved buffer through the learner, with no environment involved.

The online run diverged: Q reached 1e9 during warmup, on data the frozen VLA
produced. That is a property of the update rule, not of the rollouts, so it can
be reproduced -- and candidate fixes compared -- straight from `buffer.npz` in
seconds instead of hours.

    python scripts/replay_learner.py --buffer runs/online_134/buffer.npz --iterations 12000

Each variant starts from the same seed and sees the same data. What matters is
whether Q stays inside [0, 1], which is where the true return has to be: the
reward is a single +1 on a terminal step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlt.agent import RLTokenAgent  # noqa: E402
from rlt.config import ACTION_DIM, CHUNK, PROPRIO_DIM, OnlineConfig  # noqa: E402
from rlt.replay import ChunkReplay  # noqa: E402

VARIANTS: dict[str, dict] = {
    "paper_literal": {},
    "target_actor": {"target_actor": True},
    "clip_target": {"clip_target": True},
    "target_actor+clip": {"target_actor": True, "clip_target": True},
    "layer_norm": {"critic_layer_norm": True},
    "layer_norm+target_actor": {"critic_layer_norm": True, "target_actor": True},
}


def load_buffer(path: str, seed: int) -> ChunkReplay:
    data = np.load(path)
    size = int(data["size"])
    state_dim = data["state"].shape[1]
    chunk_dim = data["action"].shape[1]
    buffer = ChunkReplay(size, state_dim, chunk_dim, seed)
    buffer.load(path)
    print(f"{path}: {size} rows, {int((data['reward'] > 0).sum())} rewarded, "
          f"{int(data['done'].sum())} terminal, state_dim={state_dim}")
    return buffer


def run(name: str, overrides: dict, buffer: ChunkReplay, args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = OnlineConfig(episode_idx=128, device=args.device, **overrides)
    agent = RLTokenAgent(cfg, buffer.state.shape[1], buffer.action.shape[1], CHUNK)

    trace = []
    for step in range(args.iterations):
        for _ in range(cfg.critic_updates_per_actor):
            agent.critic_step(buffer.sample(cfg.batch_size))
        agent.actor_step(buffer.sample(cfg.batch_size))
        if (step + 1) % max(args.iterations // 10, 1) == 0:
            probe = agent.probe(buffer.sample(512))
            trace.append(
                {
                    "iteration": step + 1,
                    "q_actor": probe["probe/q_actor"],
                    "q_reference": probe["probe/q_reference"],
                    "gap": probe["probe/q_gap"],
                    "deviation": probe["probe/actor_deviation"],
                }
            )
            last = trace[-1]
            print(f"  {name:<24} it {step + 1:>6}  q_actor {last['q_actor']:>12.4g}  "
                  f"q_ref {last['q_reference']:>12.4g}  gap {last['gap']:>11.4g}  "
                  f"dev {last['deviation']:>10.4g}")
            if not np.isfinite(last["q_actor"]) or abs(last["q_actor"]) > 1e6:
                print(f"  {name:<24} diverged, stopping early")
                break
    return {"variant": name, "overrides": overrides, "trace": trace}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--buffer", required=True)
    ap.add_argument("--iterations", type=int, default=12000, help="2 critic + 1 actor each")
    ap.add_argument("--variants", default="", help="comma-separated subset of the known variants")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    buffer = load_buffer(args.buffer, args.seed)
    names = [n.strip() for n in args.variants.split(",") if n.strip()] or list(VARIANTS)
    results = [run(name, VARIANTS[name], buffer, args) for name in names]

    print(f"\n{'variant':<24} {'q_actor':>12} {'q_reference':>12} {'gap':>11} {'deviation':>10}  verdict")
    for result in results:
        if not result["trace"]:
            print(f"{result['variant']:<24}  no trace")
            continue
        last = result["trace"][-1]
        bounded = 0.0 <= last["q_actor"] <= 1.0 and 0.0 <= last["q_reference"] <= 1.0
        print(f"{result['variant']:<24} {last['q_actor']:>12.4g} {last['q_reference']:>12.4g} "
              f"{last['gap']:>11.4g} {last['deviation']:>10.4g}  "
              + ("stayed in [0,1]" if bounded else "left [0,1]"))

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
