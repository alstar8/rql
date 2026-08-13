"""Focused tests for V15 helpers and harness wiring."""

from __future__ import annotations

import math
from pathlib import Path

import v15_harness as h
from v15_helpers import (
    EmpiricalGateTracker,
    actor_phase_for_episode,
    empirical_delta_lcb,
    wilson_lower,
)


HERE = Path(__file__).resolve().parent


def test_v15_variant_matrix() -> None:
    assert len(h.VARIANTS) == 8
    names = [variant.name for variant in h.VARIANTS]
    assert names == [
        "residual_vla_baseline",
        "residual_rlt_actor",
        "residual_vla_cf",
        "residual_rlt_cf",
        "flow_vla_baseline",
        "flow_rlt_actor",
        "flow_rlt_cf",
        "molmo_ae_lora_actor",
    ]
    assert h.VARIANT_BY_NAME["residual_vla_cf"].use_guide
    assert h.VARIANT_BY_NAME["residual_vla_cf"].server_port == 8702
    assert h.VARIANT_BY_NAME["flow_rlt_cf"].server_port == 8706
    assert h.evaluation_policies(h.VARIANT_BY_NAME["residual_vla_cf"]) == (
        "reference",
        "actor_guide",
    )


def test_wilson_and_empirical_lcb() -> None:
    assert 0.0 <= wilson_lower(5, 20) <= 0.25
    lcb = empirical_delta_lcb(8, 20, 4, 20)
    assert math.isfinite(lcb)
    tracker = EmpiricalGateTracker()
    for _ in range(16):
        tracker.record(used_actor=True, success=True)
        tracker.record(used_actor=False, success=False)
    metrics = tracker.metrics()
    assert metrics["empirical_actor_episodes"] == 16
    assert metrics["empirical_lcb"] > 0.0


def test_actor_phase_schedule() -> None:
    warm = actor_phase_for_episode(10, bc_episodes=50)
    assert warm.phase == "bc_warmup"
    assert warm.q_coef == 0.0
    later = actor_phase_for_episode(80, bc_episodes=50)
    assert later.phase == "clipped_q"
    assert later.q_coef == 1.0
    assert later.ref_dropout == 0.0


def test_train_command_includes_v15_flags() -> None:
    run_dir = (HERE / "runs" / "rlt_cf_v15_controlled").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = h.build_train_command(
        python_executable="python",
        root=HERE,
        run_dir=run_dir,
        benchmark_train=HERE / "runs" / "benchmarks" / "house0_kettle_v13" / "train",
        residual_checkpoint=HERE / "runs" / "rlt_pretrain_demo1k" / "x.pt",
        flow_checkpoint=HERE / "runs" / "rlt_pretrain_demo1k" / "y.pt",
        tmp_rollout_dir=HERE / "tmp",
        variant=h.VARIANT_BY_NAME["residual_rlt_actor"],
        fresh=True,
    )
    assert "--actor_mixture_prob" in cmd
    assert cmd[cmd.index("--actor_mixture_prob") + 1] == "0.25"
    assert "--require_empirical_gate" in cmd
    assert "--residual_clip" in cmd
    assert "--actor_cql_coef" in cmd

    cf = h.build_train_command(
        python_executable="python",
        root=HERE,
        run_dir=run_dir,
        benchmark_train=HERE / "runs" / "benchmarks" / "house0_kettle_v13" / "train",
        residual_checkpoint=HERE / "runs" / "rlt_pretrain_demo1k" / "x.pt",
        flow_checkpoint=HERE / "runs" / "rlt_pretrain_demo1k" / "y.pt",
        tmp_rollout_dir=HERE / "tmp",
        variant=h.VARIANT_BY_NAME["residual_vla_cf"],
        fresh=True,
    )
    assert "--guide_on_reference" in cf
    assert cf[cf.index("--actor_mixture_prob") + 1] == "0.0"


def test_validate_trainer_cli() -> None:
    report = h.validate_trainer_cli(HERE / "train_rlt_online.py")
    assert report["valid"], report["missing_options"]


def test_midrun_snapshots_include_100_200() -> None:
    assert 100 in h.SNAPSHOT_EPISODES
    assert 200 in h.SNAPSHOT_EPISODES
    assert 400 in h.SNAPSHOT_EPISODES
