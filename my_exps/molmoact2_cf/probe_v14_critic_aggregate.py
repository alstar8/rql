"""Probe robust critic aggregates on immutable V13 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlt_models import MolmoAct2RLTCF


DEFAULT_VARIANTS = (
    "residual_rlt_actor",
    "residual_rlt_cf",
    "flow_rlt_actor",
    "flow_rlt_cf",
    "molmo_ae_lora_actor",
    "molmo_ae_lora_cf",
)


def _balanced_indices(success: np.ndarray, limit: int) -> np.ndarray:
    positive = np.flatnonzero(success > 0.5)
    negative = np.flatnonzero(success <= 0.5)
    if not len(positive) or not len(negative):
        return np.arange(min(len(success), limit), dtype=np.int64)
    per_outcome = min(len(positive), len(negative), max(1, limit // 2))
    return np.concatenate((positive[-per_outcome:], negative[-per_outcome:]))


def _aggregate(values: torch.Tensor, heads: int) -> torch.Tensor:
    if heads >= values.shape[0]:
        return values.mean(dim=0)
    return torch.topk(values, k=heads, dim=0, largest=False).values.mean(dim=0)


def probe_variant(run_dir: Path, variant: str, limit: int) -> dict[str, Any]:
    checkpoint = (
        run_dir
        / variant
        / "snapshots"
        / "ep_000400"
        / "rlt_cf.pt"
    )
    replay_path = run_dir / variant / "chunk_replay.npz"
    model = MolmoAct2RLTCF.load(str(checkpoint), map_location="cpu")
    model.eval()
    with np.load(replay_path, allow_pickle=False) as replay:
        success_all = np.asarray(replay["success"], dtype=np.float32)
        indices = _balanced_indices(success_all, limit)
        z = torch.from_numpy(
            np.asarray(replay["z"][indices], dtype=np.float32)
        )
        proprio = torch.from_numpy(
            np.asarray(replay["proprio"][indices], dtype=np.float32)
        )
        actions_raw = torch.from_numpy(
            np.asarray(replay["executed_actions"][indices], dtype=np.float32)
        )
        outcomes = torch.from_numpy(success_all[indices]) > 0.5
    with torch.no_grad():
        state = model.encode_state_from_z(z, proprio)
        actions = model.normalize_action(actions_raw)
        flow_time = None
        if model.is_flow:
            flow_time = torch.ones(
                actions.shape[0],
                1,
                dtype=actions.dtype,
            )
        values = model.q_chunk(state, actions, t=flow_time)
        pattern = torch.arange(
            actions.numel(),
            dtype=actions.dtype,
        ).reshape_as(actions)
        perturbed = actions + 0.03 * torch.where(
            torch.sin(pattern + 1.0) >= 0.0,
            torch.ones_like(actions),
            -torch.ones_like(actions),
        )
        perturbed_values = model.q_chunk(state, perturbed, t=flow_time)

    per_head_sensitivity = (values - perturbed_values).abs().mean(dim=1)
    positive_values = values[:, outcomes]
    negative_values = values[:, ~outcomes]
    if positive_values.shape[1] and negative_values.shape[1]:
        per_head_separation = (
            positive_values.mean(dim=1) - negative_values.mean(dim=1)
        )
    else:
        per_head_separation = torch.full(
            (values.shape[0],),
            float("nan"),
        )
    min_owner = torch.bincount(
        values.argmin(dim=0),
        minlength=values.shape[0],
    ).float()
    aggregates: dict[str, Any] = {}
    for heads in (1, 2, 3, 4, values.shape[0]):
        name = "mean" if heads == values.shape[0] else f"bottom_{heads}"
        current = _aggregate(values, heads)
        current_perturbed = _aggregate(perturbed_values, heads)
        separation = float("nan")
        if outcomes.any() and (~outcomes).any():
            separation = float(
                current[outcomes].mean() - current[~outcomes].mean()
            )
        aggregates[name] = {
            "action_sensitivity": float(
                (current - current_perturbed).abs().mean()
            ),
            "outcome_separation": separation,
            "mean": float(current.mean()),
            "std": float(current.std(unbiased=False)),
        }
    return {
        "variant": variant,
        "samples": int(values.shape[1]),
        "positive_samples": int(outcomes.sum()),
        "negative_samples": int((~outcomes).sum()),
        "per_head_sensitivity": [
            float(value) for value in per_head_sensitivity
        ],
        "per_head_outcome_separation": [
            float(value) for value in per_head_separation
        ],
        "min_head_owner_fraction": [
            float(value / max(values.shape[1], 1)) for value in min_owner
        ],
        "aggregates": aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/rlt_cf_v13_controlled"),
    )
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    variants = [
        probe_variant(args.run_dir.resolve(), variant, args.limit)
        for variant in DEFAULT_VARIANTS
    ]
    report = {
        "selected_aggregate": {
            "name": "bottom_k_mean",
            "fraction": 0.25,
            "minimum_heads": 2,
            "effective_heads_for_v14_ensemble": 3,
            "reason": (
                "Retains pessimism while preventing one dead hard-min head "
                "from monopolizing actor, rank, guide, and gate gradients."
            ),
        },
        "variants": variants,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
