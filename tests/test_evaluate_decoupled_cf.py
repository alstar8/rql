"""Unit tests for Decoupled ConsensusFlow aggregation metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_decoupled_cf import (  # noqa: E402
    final_training_diagnostics,
    latest_run,
    normalized_auc,
    paired_bootstrap,
    steps_to_threshold,
    success_at,
)


def test_exact_success_auc_and_steps_to_90():
    rows = [
        {"step": 1_000_001.0, "evaluation/success": 0.2},
        {"step": 1_200_000.0, "evaluation/success": 0.4},
        {"step": 1_300_000.0, "evaluation/success": 0.9},
        {"step": 1_400_000.0, "evaluation/success": 1.0},
    ]
    assert success_at(rows, 1_200_000) == 0.4
    assert success_at(rows, 2_000_000) is None
    expected_auc = (
        (0.4 + 0.9) * 0.5 * 100_000
        + (0.9 + 1.0) * 0.5 * 100_000
    ) / 200_000
    np.testing.assert_allclose(
        normalized_auc(rows, 1_200_000, 1_400_000),
        expected_auc,
    )
    assert steps_to_threshold(rows, 0.9, 1_200_000) == 1_300_000


def test_training_diagnostics_endpoint_and_latent_aliases():
    endpoint = final_training_diagnostics(
        [
            {
                "step": 1_400_000.0,
                "training/critic_loss": 0.25,
                "training/q_mean": 2.0,
                "training/target_q_mean": 1.5,
                "training/residual_rms": 0.1,
                "training/alpha": 3.0,
            }
        ],
        1_400_000,
    )
    assert endpoint["critic_bellman_rmse"] == 0.5
    assert endpoint["critic_q_bias"] == 0.5
    assert endpoint["residual_rms"] == 0.1
    assert endpoint["dual_alpha"] == 3.0

    latent = final_training_diagnostics(
        [
            {
                "step": 1_400_000.0,
                "training/latent_critic_loss": 0.09,
                "training/latent_q_mean": 0.5,
                "training/latent_target_q_mean": 0.4,
                "training/latent_kl": 0.2,
            }
        ],
        1_400_000,
    )
    np.testing.assert_allclose(latent["critic_bellman_rmse"], 0.3)
    np.testing.assert_allclose(latent["critic_q_bias"], 0.1)
    assert latent["latent_kl"] == 0.2


def _write_eval(path: Path, steps):
    path.parent.mkdir(parents=True)
    lines = ["step,evaluation/success"]
    lines.extend(f"{step},0.5" for step in steps)
    path.write_text("\n".join(lines) + "\n")


def test_latest_run_prefers_progress_over_timestamp(tmp_path):
    group_root = tmp_path / "rql" / "group"
    older_complete = group_root / "sd000_20260101_000000"
    newer_incomplete = group_root / "sd000_20260102_000000"
    _write_eval(older_complete / "eval.csv", [1_200_000, 1_400_000])
    _write_eval(newer_incomplete / "eval.csv", [1_200_000])

    assert latest_run(tmp_path, "group", 0) == older_complete


def test_paired_bootstrap_preserves_pair_differences():
    result = paired_bootstrap(
        [0.8, 0.6, 0.9],
        [0.7, 0.5, 0.8],
        n_boot=100,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_allclose(result["mean_delta"], 0.1)
    np.testing.assert_allclose(result["ci_low"], 0.1)
    np.testing.assert_allclose(result["ci_high"], 0.1)
    assert result["wins"] == 3
