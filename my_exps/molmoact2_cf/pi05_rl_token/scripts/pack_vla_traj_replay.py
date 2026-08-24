#!/usr/bin/env python
"""Rebuild a ChunkReplay npz from dump_vla_traj episode shards.

The V21 mug AC pretrain dumps one npz per frozen-VLA episode. Offline AC only
needs the closed chunk transitions (state, action, reference, reward). Tokens
are re-encoded with the same frozen AE that collect used, then dropped — the
saved replay never stored them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import ACTION_DIM, PROPRIO_DIM  # noqa: E402
from rlt.replay import ChunkReplay, Decision, Rollout, build_transitions  # noqa: E402
from rlt.vla import TokenEncoder  # noqa: E402


def _split_tokens(packed: np.ndarray, lengths: np.ndarray) -> list[np.ndarray]:
    out = []
    cursor = 0
    for length in lengths.astype(int):
        out.append(np.asarray(packed[cursor : cursor + length]))
        cursor += int(length)
    if cursor != packed.shape[0]:
        raise ValueError(f"token concat {packed.shape[0]} != sum(dec_len) {cursor}")
    return out


def pack_dump(dump_dir: Path, ae_path: Path, out_path: Path, *, chunk: int, gamma: float, device: str) -> dict:
    encoder = TokenEncoder(str(ae_path), device)
    state_dim = encoder.z_dim + PROPRIO_DIM
    chunk_dim = chunk * ACTION_DIM
    paths = sorted(dump_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz in {dump_dir}")

    rows: list[dict] = []
    successes = 0
    for path in paths:
        data = np.load(path)
        steps = int(data["steps"])
        terminal = int(data["terminal_step"])
        success = bool(data["success"])
        successes += int(success)
        dec_step = np.asarray(data["dec_step"])
        proprio = np.asarray(data["dec_proprio"], dtype=np.float32)
        reference = np.asarray(data["dec_reference"], dtype=np.float32)
        lengths = np.asarray(data["dec_len"])
        tokens = _split_tokens(np.asarray(data["dec_tokens"]), lengths)
        masks = _split_tokens(np.asarray(data["dec_mask"]), lengths)
        committed = np.asarray(data["committed"], dtype=np.float32)
        rewards = np.asarray(data["rewards"], dtype=np.float32)[:steps]
        if committed.ndim != 2 or committed.shape[1] != ACTION_DIM:
            raise ValueError(f"{path.name} committed {committed.shape}")
        if committed.shape[0] < steps:
            raise ValueError(f"{path.name} committed {committed.shape[0]} < steps {steps}")

        z_batch = []
        for tok, msk in zip(tokens, masks):
            z_batch.append(encoder.encode(tok, msk))
        decisions = []
        for i, step in enumerate(dec_step.tolist()):
            state = np.concatenate([z_batch[i], proprio[i]], axis=0).astype(np.float32)
            decisions.append(
                Decision(step=int(step), state=state, reference=reference[i].reshape(-1))
            )
        rollout = Rollout(
            steps=steps,
            decisions=decisions,
            committed=list(committed),
            rewards=rewards,
            terminal_step=None if terminal < 0 else terminal,
            success=success,
        )
        rows.extend(build_transitions(rollout, chunk, gamma))

    if not rows:
        raise RuntimeError(f"no closed transitions from {dump_dir}")
    buffer = ChunkReplay(len(rows), state_dim, chunk_dim, seed=0)
    buffer.extend(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buffer.save(str(out_path))
    return {
        "episodes": len(paths),
        "successes": successes,
        "rows": len(buffer),
        "reward_rows": buffer.reward_rows(),
        "state_dim": state_dim,
        "out": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--token-ae", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    stats = pack_dump(args.dump, args.token_ae, args.out, chunk=args.chunk, gamma=args.gamma, device=args.device)
    print(stats)


if __name__ == "__main__":
    main()
