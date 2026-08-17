"""Focused tests for V16 helpers and harness wiring."""

from __future__ import annotations

from pathlib import Path

import v16_harness as h
from v16_helpers import actor_phase_for_episode


HERE = Path(__file__).resolve().parent


def test_v16_variant_matrix() -> None:
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
    assert h.VARIANT_BY_NAME["residual_rlt_actor"].server_port == 8711
    assert h.HTTP_PORTS == tuple(range(8710, 8717))


def test_actor_phase_uses_paper_ref_dropout() -> None:
    warm = actor_phase_for_episode(10, bc_episodes=50)
    assert warm.phase == "bc_warmup"
    assert warm.q_coef == 0.0
    later = actor_phase_for_episode(80, bc_episodes=50)
    assert later.phase == "clipped_q"
    assert later.q_coef == 1.0
    assert later.ref_dropout == 0.5


def test_train_command_paper_collect_flags() -> None:
    run_dir = (HERE / "runs" / "rlt_cf_v16_controlled").resolve()
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
    assert "--always_collect_actor" in cmd
    assert cmd[cmd.index("--actor_mixture_prob") + 1] == "0.0"
    assert cmd[cmd.index("--actor_beta") + 1] == "2.0"
    assert cmd[cmd.index("--train_ref_dropout") + 1] == "0.5"
    assert cmd[cmd.index("--explore_residual_std") + 1] == "0.0"
    assert cmd[cmd.index("--residual_clip") + 1] == "0.02"

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
    assert "--always_collect_actor" not in cf
    assert cmd[cmd.index("--explore_residual_std") + 1] == "0.0"
    assert cf[cf.index("--explore_residual_std") + 1] == "0.0"
    assert cf[cf.index("--explore_deploy_std") + 1] == "0.0"

    ae = h.build_train_command(
        python_executable="python",
        root=HERE,
        run_dir=run_dir,
        benchmark_train=HERE / "runs" / "benchmarks" / "house0_kettle_v13" / "train",
        residual_checkpoint=HERE / "runs" / "rlt_pretrain_demo1k" / "x.pt",
        flow_checkpoint=HERE / "runs" / "rlt_pretrain_demo1k" / "y.pt",
        tmp_rollout_dir=HERE / "tmp",
        variant=h.VARIANT_BY_NAME["molmo_ae_lora_actor"],
        fresh=True,
    )
    assert ae[ae.index("--explore_residual_std") + 1] == "0.0"
    assert ae[ae.index("--explore_deploy_std") + 1] == "0.0"
    assert ae[ae.index("--max_update_sec_per_episode") + 1] == str(
        h.AE_MAX_UPDATE_SEC_PER_EPISODE
    )


def test_validate_trainer_cli() -> None:
    report = h.validate_trainer_cli(HERE / "train_rlt_online.py")
    assert report["valid"], report["missing_options"]


def test_midrun_snapshots_include_100_200() -> None:
    assert 100 in h.SNAPSHOT_EPISODES
    assert 200 in h.SNAPSHOT_EPISODES
    assert 400 in h.SNAPSHOT_EPISODES
