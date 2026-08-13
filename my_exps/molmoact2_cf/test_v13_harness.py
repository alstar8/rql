"""Focused CPU/static tests for the controlled V13 operational harness."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

import v13_harness
from plot_v13_controlled import build_paired_policy_report, generate_report
from snapshot_run_status import read_pid_record
from v13_harness import (
    FINAL_VALIDATION_SEEDS,
    HTTP_PORTS,
    INTERIM_VALIDATION_SEEDS,
    MAX_VALID_EPISODES,
    TARGET_ENV_STEPS,
    VARIANTS,
    build_eval_command,
    build_manifest,
    build_train_command,
    evaluation_complete,
)
from validate_v13_wiring import run_probe


HERE = Path(__file__).resolve().parent
SHELL_FILES = (
    HERE / "launch_v13_controlled.sh",
    HERE / "eval_v13_controlled.sh",
    HERE / "status_v13_controlled.sh",
)
PYTHON_FILES = (
    HERE / "v13_harness.py",
    HERE / "snapshot_run_status.py",
    HERE / "plot_v13_controlled.py",
    HERE / "validate_v13_wiring.py",
    Path(__file__),
)


def _argument_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_shell_syntax_and_static_forbidden_flags() -> None:
    for path in SHELL_FILES:
        subprocess.run(["bash", "-n", str(path)], check=True)

    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (*SHELL_FILES, HERE / "v13_harness.py")
    )
    for forbidden in (
        "--joint_cf",
        "--guide_target_delta_frac",
        "--g_min_guide_advantage",
        "--guide_beta",
    ):
        assert forbidden not in combined
    launch = (HERE / "launch_v13_controlled.sh").read_text(encoding="utf-8")
    assert "--validate-only" in launch
    assert "FRESH" in launch
    assert "--watchdog" in launch
    assert "rlt_cf_v12" not in launch
    evaluation = (HERE / "eval_v13_controlled.sh").read_text(encoding="utf-8")
    assert "V13_EVAL_VARIANTS" in evaluation
    assert "V13_EVAL_POLICIES" in evaluation
    assert "V13_ALLOW_UNSAFE_AE_FORCE_DEPLOY" in evaluation
    assert "POLICIES=(checkpoint_gate)" in evaluation


def test_ep400_paired_report_marks_vacuous_and_invalid_cells(
    tmp_path: Path,
) -> None:
    snapshot = (
        tmp_path
        / "residual_rlt_cf"
        / "snapshots"
        / "ep_000400"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "snapshot.json").write_text(
        json.dumps({"gate_deploy_actor": False}),
        encoding="utf-8",
    )

    def run(policy: str, successes: tuple[bool, bool]) -> dict:
        return {
            "config": "residual_rlt_cf",
            "snapshot": "ep_000400",
            "episode": 400,
            "policy": policy,
            "seed": 20260831,
            "complete": True,
            "summary": {},
            "results": [
                {
                    "valid": True,
                    "episode_idx": index,
                    "success": success,
                }
                for index, success in enumerate(successes)
            ],
        }

    report = build_paired_policy_report(
        [
            run("checkpoint_gate", (False, True)),
            run("actor", (True, True)),
            run("actor_guide", (False, True)),
        ],
        tmp_path,
    )
    comparisons = {
        (row["treatment_policy"], row["control_policy"]): row
        for row in report["comparisons"]
    }
    actor = comparisons[("actor", "checkpoint_gate")]
    guide = comparisons[("actor_guide", "actor")]
    assert actor["paired_rollouts"] == 2
    assert actor["paired_success_rate_delta"] == pytest.approx(0.5)
    assert actor["treatment_only_successes"] == 1
    assert guide["paired_success_rate_delta"] == pytest.approx(-0.5)
    assert any("frozen-reference" in warning for warning in report["warnings"])
    assert report["expected_safe_ep400_jobs"] == 64
    assert report["expected_safe_ep400_rollouts"] == 768
    assert {
        cell["config"]
        for cell in report["invalid_by_construction"]
    } == {"molmo_ae_lora_actor", "molmo_ae_lora_cf"}


def test_variant_matrix_is_exactly_one_arm_per_gpu() -> None:
    expected_names = [
        "residual_vla_baseline",
        "residual_rlt_actor",
        "residual_rlt_cf",
        "flow_vla_baseline",
        "flow_rlt_actor",
        "flow_rlt_cf",
        "molmo_ae_lora_actor",
        "molmo_ae_lora_cf",
    ]
    assert [variant.name for variant in VARIANTS] == expected_names
    assert [variant.gpu for variant in VARIANTS] == list(range(8))
    assert [variant.updates_per_episode for variant in VARIANTS] == [
        0,
        8,
        8,
        0,
        8,
        8,
        4,
        4,
    ]
    assert tuple(
        variant.server_port
        for variant in VARIANTS
        if variant.server_port is not None
    ) == HTTP_PORTS
    assert sum(variant.ae_trainable for variant in VARIANTS) == 2
    assert all(
        variant.server_port is None
        for variant in VARIANTS[-2:]
    )
    assert VARIANTS[0].policies == ("reference",)
    assert VARIANTS[1].policies == (
        "checkpoint_gate",
        "actor",
        "reference_noise",
    )
    assert VARIANTS[2].policies == (
        "checkpoint_gate",
        "actor",
        "actor_guide",
    )


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda item: item.name)
def test_training_and_eval_benchmark_paths_are_isolated(
    tmp_path: Path,
    variant: v13_harness.VariantSpec,
) -> None:
    root = tmp_path / "root"
    run_dir = tmp_path / "run"
    benchmark_root = tmp_path / "controlled"
    train_benchmark = benchmark_root / "train"
    val_benchmark = benchmark_root / "val"
    residual = tmp_path / "residual.pt"
    flow = tmp_path / "flow.pt"
    train = build_train_command(
        python_executable="/python",
        root=root,
        run_dir=run_dir,
        benchmark_train=train_benchmark,
        residual_checkpoint=residual,
        flow_checkpoint=flow,
        tmp_rollout_dir=tmp_path / "train_tmp",
        variant=variant,
        fresh=True,
    )
    assert Path(_argument_value(train, "--benchmark_dir")) == train_benchmark
    assert str(val_benchmark) not in train
    assert _argument_value(train, "--start_episode") == "0"
    assert _argument_value(train, "--shard_size") == "24"
    assert _argument_value(train, "--max_valid_episodes") == "400"
    assert int(_argument_value(train, "--target_env_steps")) == TARGET_ENV_STEPS
    assert TARGET_ENV_STEPS > MAX_VALID_EPISODES * 500
    assert "--no_resume" in train
    if variant.ae_trainable:
        assert _argument_value(train, "--ae_batch_size") == "2"

    policy = variant.policies[-1]
    evaluation = build_eval_command(
        python_executable="/python",
        root=root,
        run_dir=run_dir,
        benchmark_val=val_benchmark,
        tmp_rollout_dir=tmp_path / "eval_tmp",
        variant=variant,
        episode=100,
        policy=policy,
        seed=INTERIM_VALIDATION_SEEDS[0],
    )
    assert Path(_argument_value(evaluation, "--benchmark_dir")) == val_benchmark
    assert str(train_benchmark) not in evaluation
    assert "--eval_only" in evaluation
    assert _argument_value(evaluation, "--explore_residual_std") == "0"
    assert _argument_value(evaluation, "--explore_deploy_std") == "0"
    assert _argument_value(evaluation, "--max_valid_episodes") == "12"
    expected_policy = "reference" if policy == "reference_noise" else policy
    assert _argument_value(evaluation, "--deploy_policy") == expected_policy
    if policy == "reference_noise":
        assert _argument_value(evaluation, "--eval_reference_noise_std") == "0.02"
    assert _argument_value(evaluation, "--rlt_ckpt").endswith(
        f"{variant.name}/snapshots/ep_000100/rlt_cf.pt"
    )
    if variant.ae_trainable:
        assert _argument_value(evaluation, "--ae_trainable_ckpt").endswith(
            f"{variant.name}/snapshots/ep_000100/molmo_ae_lora.pt"
        )

    assert len(INTERIM_VALIDATION_SEEDS) == 1
    assert len(FINAL_VALIDATION_SEEDS) == 4


def test_manifest_generation_hashes_specs_commands_and_redacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "repo" / "submodules" / "rql" / "my_exps" / "molmoact2_cf"
    root.mkdir(parents=True)
    benchmark = tmp_path / "benchmark"
    (benchmark / "train").mkdir(parents=True)
    (benchmark / "val").mkdir()
    train_bytes = b"controlled train"
    val_bytes = b"held out val"
    (benchmark / "train" / "benchmark.json").write_bytes(train_bytes)
    (benchmark / "val" / "benchmark.json").write_bytes(val_bytes)
    (benchmark / "manifest.json").write_text("{}\n", encoding="utf-8")
    residual = tmp_path / "residual.pt"
    flow = tmp_path / "flow.pt"
    residual.write_bytes(b"residual checkpoint")
    flow.write_bytes(b"flow checkpoint")

    monkeypatch.setattr(v13_harness, "_gpu_inventory", lambda: {"available": False, "gpus": []})
    monkeypatch.setattr(
        v13_harness,
        "_git_state",
        lambda path: {"path": str(path), "sha": "abc", "dirty": False},
    )
    manifest = build_manifest(
        root=root,
        run_dir=tmp_path / "run",
        log_dir=tmp_path / "logs",
        benchmark_root=benchmark,
        residual_checkpoint=residual,
        flow_checkpoint=flow,
        python_executable="/python",
        serve_prefix=["/serve-python"],
        tmp_rollout_dir=tmp_path / "tmp_rollouts",
        egl_lock_dir=tmp_path / "egl",
        gpu_ids=[str(index) for index in range(8)],
        environment={
            "B1K_TMP": "/scratch",
            "HF_TOKEN": "must-not-appear",
            "FRESH": "1",
        },
    )
    assert manifest["benchmark"]["files"]["train"]["sha256"] == hashlib.sha256(
        train_bytes
    ).hexdigest()
    assert manifest["benchmark"]["files"]["val"]["sha256"] == hashlib.sha256(
        val_bytes
    ).hexdigest()
    assert manifest["environment"]["HF_TOKEN"] == "<redacted>"
    assert len(manifest["variants"]) == 8
    assert len(manifest["servers"]) == 6
    serialized = json.dumps(manifest)
    assert "must-not-appear" not in serialized
    assert "--no_resume" in manifest["variants"][0]["initial_fresh_command"]
    assert "--no_resume" not in manifest["variants"][0]["resume_command"]


def test_evaluation_completion_requires_one_clean_pass_per_pose(tmp_path: Path) -> None:
    out_dir = tmp_path / "evaluation"
    out_dir.mkdir()
    (out_dir / "validation_summary.json").write_text(
        json.dumps(
            {
                "eval_only": True,
                "valid_episodes": 12,
                "skipped_episodes": 0,
            }
        ),
        encoding="utf-8",
    )
    results = [
        {
            "valid": True,
            "episode_idx": episode,
            "success": episode % 2 == 0,
        }
        for episode in range(12)
    ]
    (out_dir / "validation_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results),
        encoding="utf-8",
    )
    assert evaluation_complete(out_dir)

    results[-1]["episode_idx"] = 0
    (out_dir / "validation_results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in results),
        encoding="utf-8",
    )
    assert not evaluation_complete(out_dir)


def test_plotting_survives_incomplete_running_data(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    metrics_dir = run_dir / VARIANTS[1].name
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "valid_episodes": 10,
                "env_steps": 1234,
                "cumulative_success_rate": 0.2,
                "window_success_rate": 0.2,
                "q_td_loss": 0.1,
                "q_mean": 0.3,
                "q_std": 0.02,
                "actor_adv": 0.01,
                "gate_paired_lcb": 0.001,
                "gate_sensitivity": 0.004,
            }
        )
        + "\n{truncated",
        encoding="utf-8",
    )
    incomplete = (
        run_dir
        / "validation"
        / VARIANTS[1].name
        / "ep_000100"
        / "actor"
        / f"seed_{INTERIM_VALIDATION_SEEDS[0]}"
    )
    incomplete.mkdir(parents=True)
    (incomplete / "validation_summary.json").write_text(
        json.dumps(
            {
                "eval_only": True,
                "valid_episodes": 2,
                "cumulative_success_rate": 0.5,
            }
        ),
        encoding="utf-8",
    )
    (incomplete / "validation_results.jsonl").write_text(
        json.dumps(
            {
                "valid": True,
                "episode_idx": 0,
                "success": True,
                "episode_steps": 100,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = tmp_path / "benchmark"
    (benchmark / "val").mkdir(parents=True)
    (benchmark / "val" / "benchmark.json").write_text(
        json.dumps(
            [
                {
                    "controlled": {
                        "pair_id": "val_k00_r00",
                        "kettle_pose_id": "val_k00",
                        "robot_pose_id": "val_r00",
                    },
                    "task": {},
                }
            ]
        ),
        encoding="utf-8",
    )

    paths = generate_report(run_dir, benchmark)
    assert all(path.is_file() for path in paths.values())
    assert all(path.stat().st_size > 0 for path in paths.values())
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    config = status["configs"][VARIANTS[1].name]
    assert config["training_metrics_rows"] == 1
    assert config["validation_complete_runs"] == 0


def test_wiring_probe_passes_all_eight_without_loading_real_ae_weights(
    tmp_path: Path,
) -> None:
    payload = run_probe(
        tmp_path / "unused_residual.pt",
        tmp_path / "unused_flow.pt",
        skip_checkpoints=True,
    )
    assert payload["passed"] is True
    assert payload["full_molmo_ae_weights_loaded"] is False
    assert len(payload["variants"]) == 8
    assert all(record["passed"] for record in payload["variants"])
    for record in payload["variants"]:
        membership = record["optimizer_membership"]
        assert membership["each_trainable_exactly_once"]
        assert membership["no_frozen_members"]
        assert record["synthetic_update"]["forbidden_unchanged"][
            "token_unchanged"
        ]


def test_pid_snapshot_reads_only_safe_first_field(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    pid_dir = run_dir / "pids"
    pid_dir.mkdir(parents=True)
    malformed = pid_dir / "bad.pid"
    malformed.write_text("not-a-pid metadata\n", encoding="utf-8")
    record = read_pid_record(malformed, run_dir)
    assert record["valid_pid"] is False
    assert record["alive"] is False

    current = pid_dir / "current.pid"
    current.write_text(f"{os.getpid()} metadata ignored\n", encoding="utf-8")
    record = read_pid_record(current, run_dir)
    assert record["pid"] == os.getpid()
    assert record["alive"] is True
    assert record["belongs_to_run"] is False


def test_new_python_files_have_no_inline_imports() -> None:
    for path in PYTHON_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            inline_imports = [
                child
                for child in ast.walk(node)
                if isinstance(child, (ast.Import, ast.ImportFrom))
            ]
            assert not inline_imports, f"Inline import found in {path}: {inline_imports}"
