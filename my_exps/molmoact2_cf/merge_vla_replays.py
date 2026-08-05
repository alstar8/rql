"""Merge G=0 VLA replay shards and fit global normalization statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def merge_replays(paths: list[Path]) -> dict[str, np.ndarray]:
    if not paths:
        raise ValueError("at least one replay path is required")
    shards = [_load(path) for path in paths]
    required = {
        "features",
        "proprio",
        "actions_raw",
        "base_actions_raw",
        "returns",
        "successes",
        "rewards",
        "dones",
    }
    for path, shard in zip(paths, shards):
        missing = required - shard.keys()
        if missing:
            raise ValueError(f"{path} missing replay keys: {sorted(missing)}")

    features = np.concatenate([s["features"] for s in shards], axis=0).astype(np.float16)
    proprio = np.concatenate([s["proprio"] for s in shards], axis=0).astype(np.float32)
    actions_raw = np.concatenate([s["actions_raw"] for s in shards], axis=0).astype(np.float32)
    base_raw = np.concatenate([s["base_actions_raw"] for s in shards], axis=0).astype(np.float32)

    proprio_mean = proprio.mean(axis=0, dtype=np.float64).astype(np.float32)
    proprio_std = proprio.std(axis=0, dtype=np.float64).clip(min=1e-3).astype(np.float32)
    action_mean = base_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    action_std = base_raw.std(axis=0, dtype=np.float64).clip(min=1e-3).astype(np.float32)
    feature_mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = features.std(axis=0, dtype=np.float64).clip(min=1e-3).astype(np.float32)

    action_std_safe = np.clip(action_std, 1e-3, None)
    merged = {
        "features": features,
        "proprio": proprio,
        "actions": ((actions_raw - action_mean) / action_std_safe).astype(np.float32),
        "base_actions": ((base_raw - action_mean) / action_std_safe).astype(np.float32),
        "actions_raw": actions_raw,
        "base_actions_raw": base_raw,
        "returns": np.concatenate([s["returns"] for s in shards]).astype(np.float32),
        "successes": np.concatenate([s["successes"] for s in shards]).astype(np.float32),
        "rewards": np.concatenate([s["rewards"] for s in shards]).astype(np.float32),
        "dones": np.concatenate([s["dones"] for s in shards]).astype(np.float32),
        "proprio_mean": proprio_mean,
        "proprio_std": proprio_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "n_transitions": np.asarray([len(proprio)], dtype=np.int64),
        "n_episodes": np.asarray(
            [sum(int(s.get("n_episodes", np.asarray([0]))[0]) for s in shards)],
            dtype=np.int64,
        ),
    }
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    merged = merge_replays(args.paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **merged)
    summary = {
        "out": str(args.out),
        "shards": len(args.paths),
        "transitions": int(merged["n_transitions"][0]),
        "episodes": int(merged["n_episodes"][0]),
        "success_fraction": float(merged["successes"].mean()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
