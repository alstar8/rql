"""Tests for ConsensusFlow ablation switches and stats aggregation helpers."""

from __future__ import annotations

import importlib.util
import csv
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

RQL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = RQL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(RQL_ROOT))


def _load_stats_module():
    path = SCRIPTS / "aggregate_consensusflow_stats.py"
    name = "aggregate_consensusflow_stats"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # dataclasses require the module to be present in sys.modules during exec.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_conflict_power_zero_disables_kill():
    """conflict_power=0 → kill_frac = 1 - trust^0 = 0 for all trust in [0,1]."""
    for trust in (0.0, 0.01, 0.5, 1.0):
        kill = 1.0 - jnp.power(jnp.clip(trust, 0.0, 1.0), 0.0)
        assert float(kill) == pytest.approx(0.0, abs=1e-6)


def test_dflrql9_behavior_safe_conflict_power_zero():
    from agents.dflrql9 import DFLRQL9Agent, get_config

    cfg = dict(get_config())
    cfg["conflict_power"] = 0.0
    cfg["residual_coef"] = 0.0
    agent = object.__new__(DFLRQL9Agent)
    object.__setattr__(agent, "config", cfg)
    # Anti-BC direction should be retained when conflict_power=0.
    w = jnp.array([[0.0, -1.0]])
    behavior = jnp.array([[0.0, 1.0]])
    safe_w, diag = DFLRQL9Agent._behavior_safe_direction(agent, w, behavior)
    assert float(diag["conflict_kill_frac"].mean()) == pytest.approx(0.0, abs=1e-5)
    # Direction should remain anti-behavior (negative y).
    assert float(safe_w[0, 1]) < 0.0


def test_dflrql9_defaults_match_full_method():
    from agents.dflrql9 import get_config

    cfg = get_config()
    assert cfg.guidance_coef == 0.5
    assert cfg.consensus_floor == 0.01
    assert cfg.conflict_power == 2.0
    assert cfg.residual_coef == 0.25
    assert cfg.ensemble_ct == 10


def test_bootstrap_and_paired_helpers():
    mod = _load_stats_module()
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, size=50)
    ci = mod.bootstrap_ci(x, n_boot=1000, seed=1)
    assert ci["n"] == 50
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    a = np.array([0.6, 0.7, 0.8])
    b = np.array([0.5, 0.65, 0.75])
    paired = mod.paired_tests(a, b)
    assert paired["n_paired"] == 3
    assert paired["mean_delta"] == pytest.approx(0.0666666667, rel=1e-6)
    assert paired["win_loss_tie"]["win"] == 3


def test_final_checkpoint_from_run_uses_only_exact_step(tmp_path):
    mod = _load_stats_module()
    eval_csv = tmp_path / "eval.csv"
    with eval_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "evaluation/success",
                "evaluation/episode.return",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "step": 800_000,
                    "evaluation/success": 0.1,
                    "evaluation/episode.return": 1.0,
                },
                {
                    "step": 900_000,
                    "evaluation/success": 0.2,
                    "evaluation/episode.return": 2.0,
                },
                {
                    "step": 1_000_000,
                    "evaluation/success": 0.9,
                    "evaluation/episode.return": 9.0,
                },
            ]
        )
    metric = mod.final_checkpoint_from_run(tmp_path, 1_000_000)
    assert metric.mode == "final_checkpoint"
    assert metric.success == pytest.approx(0.9)

    missing = mod.final_checkpoint_from_run(tmp_path, 2_000_000)
    assert missing.mode == "missing_final_checkpoint"
    assert missing.success is None
    assert missing.max_step == 1_000_000


def test_ablation_specs_document_switches():
    mod = _load_stats_module()
    assert set(mod.ABLATION_SPECS) == {
        "full",
        "no_guidance",
        "lambda02",
        "lambda10",
        "no_conflict",
        "no_residual",
        "no_floor",
        "no_crf",
        "nocrf_k2",
        "nocrf_k5",
        "nocrf_k20",
        "single_critic",
    }
    assert mod.ABLATION_SPECS["no_guidance"]["flags"]["guidance_coef"] == 0.0
    assert mod.ABLATION_SPECS["lambda02"]["flags"]["guidance_coef"] == 0.2
    assert mod.ABLATION_SPECS["lambda02"]["flags"] == {
        "guidance_coef": 0.2,
        "distill_coef": 1.0,
        "consensus_floor": 0.0,
        "conflict_power": 0.0,
        "residual_coef": 0.0,
        "ensemble_ct": 10,
    }
    assert mod.ABLATION_SPECS["lambda10"]["flags"]["guidance_coef"] == 1.0
    assert mod.ABLATION_SPECS["lambda10"]["flags"]["conflict_power"] == 0.0
    assert mod.ABLATION_SPECS["lambda10"]["flags"]["residual_coef"] == 0.0
    assert mod.ABLATION_SPECS["lambda10"]["flags"]["consensus_floor"] == 0.0
    assert mod.ABLATION_SPECS["no_conflict"]["flags"]["conflict_power"] == 0.0
    assert mod.ABLATION_SPECS["no_residual"]["flags"]["residual_coef"] == 0.0
    assert mod.ABLATION_SPECS["no_floor"]["flags"]["consensus_floor"] == 0.0
    assert mod.ABLATION_SPECS["no_crf"]["flags"] == {
        "guidance_coef": 0.5,
        "distill_coef": 1.0,
        "consensus_floor": 0.0,
        "conflict_power": 0.0,
        "residual_coef": 0.0,
        "ensemble_ct": 10,
    }
    assert mod.ABLATION_SPECS["nocrf_k2"]["flags"]["ensemble_ct"] == 2
    assert mod.ABLATION_SPECS["nocrf_k5"]["flags"]["ensemble_ct"] == 5
    assert mod.ABLATION_SPECS["nocrf_k20"]["flags"]["ensemble_ct"] == 20
    for name in ("nocrf_k2", "nocrf_k5", "nocrf_k20"):
        assert mod.ABLATION_SPECS[name]["flags"]["conflict_power"] == 0.0
        assert mod.ABLATION_SPECS[name]["flags"]["residual_coef"] == 0.0
        assert mod.ABLATION_SPECS[name]["flags"]["consensus_floor"] == 0.0
    assert mod.ABLATION_SPECS["single_critic"]["flags"]["ensemble_ct"] == 1
    assert mod.ABLATION_SPECS["full"]["reuse_from_2m"] is True


def test_flags_match_expected():
    mod = _load_stats_module()
    flags = {
        "agent": {
            "guidance_coef": 0.5,
            "consensus_floor": 0.01,
            "conflict_power": 2.0,
            "residual_coef": 0.25,
            "ensemble_ct": 10,
            "distill_coef": 1.0,
        }
    }
    assert mod.flags_match_expected(flags, mod.ABLATION_SPECS["full"]["flags"])
    flags["agent"]["guidance_coef"] = 0.0
    assert not mod.flags_match_expected(flags, mod.ABLATION_SPECS["full"]["flags"])


def test_final_checkpoint_csv_loader_if_present():
    mod = _load_stats_module()
    csv_path = RQL_ROOT / "my_exps" / "ogbench50_all50_metrics_2m.csv"
    if not csv_path.is_file():
        pytest.skip("source CSV missing")
    agg = mod.load_source_csv_task_means(csv_path)
    assert agg["baseline"]["n_tasks"] == 50
    assert abs(agg["baseline"]["grand_mean"] - 0.5386666666666666) < 1e-6
    assert abs(agg["v9"]["grand_mean"] - 0.5937333333333332) < 1e-6
