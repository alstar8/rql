"""Focused CPU/static tests for the isolated controlled V14 harness."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import v13_harness
import v14_harness
from v14_harness import (
    AE_BATCH_SIZE,
    AE_IMAGE_REPLAY_CAPACITY,
    AE_MIN_SUCCESS_EPISODES,
    BENCHMARK_POSE_CYCLE,
    CRITIC_TARGET_POLICY,
    MAX_UPDATE_SEC_PER_EPISODE,
    Q_TAIL_FRACTION,
    Q_TAIL_MIN_HEADS,
    REQUIRED_AE_REPLAY_MEMBERS,
    REQUIRED_V14_TRAINER_OPTIONS,
    SNAPSHOT_EPISODES,
    VARIANTS,
    build_eval_command,
    build_manifest,
    build_train_command,
    evaluation_policies,
    resolve_native_action_contract,
    resume_state,
    validate_manifest,
    validate_trainer_cli,
)


HERE = Path(__file__).resolve().parent
SHELL_FILES = (
    HERE / "launch_v14_controlled.sh",
    HERE / "eval_v14_controlled.sh",
)
PYTHON_FILES = (
    HERE / "v14_harness.py",
    Path(__file__),
)


def _argument_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _write_fake_hf_contract(hf_home: Path) -> Path:
    revision = "v14-test-revision"
    model_cache = hf_home / "hub" / "models--allenai--MolmoAct2-DROID"
    snapshot = model_cache / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (model_cache / "refs").mkdir()
    (model_cache / "refs" / "main").write_text(revision + "\n", encoding="utf-8")
    (snapshot / "config.json").write_text(
        json.dumps({"n_obs_steps": 1, "max_action_dim": 32}) + "\n",
        encoding="utf-8",
    )
    (snapshot / "norm_stats.json").write_text(
        json.dumps(
            {
                "format": "molmoact2_norm_stats.v1",
                "norm_mode": "q01_q99",
                "metadata_by_tag": {
                    "franka_droid": {
                        "action_horizon": 15,
                        "n_action_steps": 15,
                        "normalize_gripper": False,
                        "control_mode": "absolute joint pose",
                        "setup_type": "single franka robotic arm in droid",
                        "action_stats": {
                            "names": [f"joint_{index}" for index in range(7)]
                            + ["gripper"],
                            "q01": [-0.5] * 8,
                            "q99": [0.5] * 8,
                            "mask": [True] * 7 + [False],
                        },
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot


def test_shell_syntax_and_fail_closed_static_contracts() -> None:
    for path in SHELL_FILES:
        subprocess.run(["bash", "-n", str(path)], check=True)

    launch = (HERE / "launch_v14_controlled.sh").read_text(encoding="utf-8")
    evaluation = (HERE / "eval_v14_controlled.sh").read_text(encoding="utf-8")
    harness = (HERE / "v14_harness.py").read_text(encoding="utf-8")
    combined = "\n".join((launch, evaluation, harness))

    assert "runs/rlt_cf_v13_controlled" not in combined
    assert "runs/rlt_cf_v14_controlled" in combined
    assert "--validate-only" in launch
    assert "validate-trainer-cli" in launch
    assert "validate-manifest" in launch
    assert "assert-gpu-ownership" in launch
    assert "port-free" in launch
    assert "--watchdog" in launch
    assert "V14_MODE" in launch
    assert "smoke)" in launch
    assert "full)" in launch
    assert "|| true" not in launch

    assert "V14_EVAL_VARIANTS" in evaluation
    assert "V14_EVAL_POLICIES" in evaluation
    assert "V14_EVAL_EPISODES" in evaluation
    assert "policies=(reference actor actor_guide)" in evaluation
    assert "policies=(reference reference_noise actor)" in evaluation
    assert "policies=(reference actor)" in evaluation
    assert "rm -rf" not in evaluation
    assert "eval_v13" not in evaluation


def test_shared_process_tools_recognize_v14_ownership() -> None:
    stop_source = (HERE / "stop_run.sh").read_text(encoding="utf-8")
    status_source = (HERE / "snapshot_run_status.py").read_text(
        encoding="utf-8"
    )
    assert "RLT_CF_V14_RUN_DIR=${RUN_DIR}" in stop_source
    assert '"RLT_CF_V14_RUN_DIR"' in status_source


def test_v14_preserves_exact_eight_arm_definitions() -> None:
    assert VARIANTS == tuple(v13_harness.VARIANTS)
    assert len(VARIANTS) == 8
    assert [variant.gpu for variant in VARIANTS] == list(range(8))
    assert [variant.name for variant in VARIANTS] == [
        "residual_vla_baseline",
        "residual_rlt_actor",
        "residual_rlt_cf",
        "flow_vla_baseline",
        "flow_rlt_actor",
        "flow_rlt_cf",
        "molmo_ae_lora_actor",
        "molmo_ae_lora_cf",
    ]
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


def test_smoke_mode_honors_episode_step_and_batch_overrides() -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "V14_MODE": "smoke",
            "V14_MAX_VALID_EPISODES": "3",
            "V14_TARGET_ENV_STEPS": "123",
            "V14_SNAPSHOT_EPISODES": "0,3",
            "V14_AE_BATCH_SIZE": "6",
            "V14_MAX_UPDATE_SEC_PER_EPISODE": "4.5",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, v14_harness as h; "
                "print(json.dumps([h.RUN_MODE, h.MAX_VALID_EPISODES, "
                "h.TARGET_ENV_STEPS, h.SNAPSHOT_EPISODES, h.AE_BATCH_SIZE, "
                "h.MAX_UPDATE_SEC_PER_EPISODE]))"
            ),
        ],
        cwd=HERE,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == ["smoke", 3, 123, [0, 3], 6, 4.5]


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda item: item.name)
def test_train_commands_are_v14_isolated_and_explicit(
    tmp_path: Path,
    variant: v14_harness.VariantSpec,
) -> None:
    run_dir = tmp_path / "rlt_cf_v14_controlled"
    command = build_train_command(
        python_executable="/python",
        root=HERE,
        run_dir=run_dir,
        benchmark_train=tmp_path / "benchmark" / "train",
        residual_checkpoint=tmp_path / "residual.pt",
        flow_checkpoint=tmp_path / "flow.pt",
        tmp_rollout_dir=tmp_path / "rollouts",
        variant=variant,
        fresh=True,
    )

    assert Path(_argument_value(command, "--out_dir")).is_relative_to(run_dir)
    assert "rlt_cf_v13_controlled" not in "\0".join(command)
    assert "--no_resume" in command
    assert "--no_critic_target_use_guide" in command
    assert "--critic_target_use_guide" not in command
    assert "--allow_legacy_ae_resume" not in command
    assert _argument_value(command, "--max_updates_per_episode") == str(
        variant.updates_per_episode
    )
    assert float(
        _argument_value(command, "--max_update_sec_per_episode")
    ) == pytest.approx(MAX_UPDATE_SEC_PER_EPISODE)
    assert _argument_value(command, "--snapshot_episodes") == ",".join(
        str(episode) for episode in SNAPSHOT_EPISODES
    )
    assert _argument_value(command, "--benchmark_pose_cycle") == str(
        BENCHMARK_POSE_CYCLE
    )
    if variant.ae_trainable:
        assert _argument_value(command, "--ae_batch_size") == str(AE_BATCH_SIZE)
        assert _argument_value(
            command,
            "--ae_image_replay_capacity",
        ) == str(AE_IMAGE_REPLAY_CAPACITY)
        assert _argument_value(command, "--ae_min_success_episodes") == str(
            AE_MIN_SUCCESS_EPISODES
        )


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda item: item.name)
def test_eval_commands_are_read_only_and_policy_complete(
    tmp_path: Path,
    variant: v14_harness.VariantSpec,
) -> None:
    run_dir = tmp_path / "rlt_cf_v14_controlled"
    expected = (
        ("reference",)
        if variant.is_baseline
        else (
            ("reference", "reference_noise", "actor")
            if variant.name in {"residual_rlt_actor", "flow_rlt_actor"}
            else (
                ("reference", "actor", "actor_guide")
                if variant.use_guide
                else ("reference", "actor")
            )
        )
    )
    assert evaluation_policies(variant) == expected
    for policy in expected:
        command = build_eval_command(
            python_executable="/python",
            root=HERE,
            run_dir=run_dir,
            benchmark_val=tmp_path / "benchmark" / "val",
            tmp_rollout_dir=tmp_path / "eval_rollouts",
            variant=variant,
            episode=SNAPSHOT_EPISODES[0],
            policy=policy,
            seed=20260831,
        )
        assert "--eval_only" in command
        assert "--no_resume" in command
        assert "--no_critic_target_use_guide" in command
        assert "--allow_legacy_ae_resume" not in command
        assert _argument_value(command, "--updates_per_episode") == "0"
        assert _argument_value(command, "--max_updates_per_episode") == "0"
        assert _argument_value(command, "--deploy_policy") == (
            "reference" if policy == "reference_noise" else policy
        )
        if policy == "reference_noise":
            assert float(
                _argument_value(command, "--eval_reference_noise_std")
            ) == pytest.approx(v14_harness.EXPLORE_STD)
        assert Path(_argument_value(command, "--out_dir")).is_relative_to(
            run_dir / "validation"
        )
        assert "rlt_cf_v13_controlled" not in "\0".join(command)


def test_native_action_contract_resolves_full_cached_provenance(
    tmp_path: Path,
) -> None:
    snapshot = _write_fake_hf_contract(tmp_path / "hf")
    contract = resolve_native_action_contract(tmp_path / "hf")
    assert contract["repo_id"] == "allenai/MolmoAct2-DROID"
    assert contract["resolved_revision"] == "v14-test-revision"
    assert contract["normalization_mode"] == "q01_q99"
    assert contract["normalization_tag"] == "franka_droid"
    assert contract["action_horizon"] == 15
    assert contract["n_action_steps"] == 15
    assert contract["n_obs_steps"] == 1
    assert contract["action_dim"] == 8
    assert contract["max_action_dim"] == 32
    assert contract["deployment_chunk_horizon"] == 8
    assert contract["invalid_action_policy"] == "fatal_no_fallback"
    assert contract["action_names"][-1] == "gripper"
    assert Path(contract["snapshot_path"]) == snapshot
    assert len(contract["config_sha256"]) == 64
    assert len(contract["norm_stats_sha256"]) == 64


def test_manifest_records_commands_contract_and_run_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = tmp_path / "benchmark"
    (benchmark / "train").mkdir(parents=True)
    (benchmark / "val").mkdir()
    (benchmark / "train" / "benchmark.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    (benchmark / "val" / "benchmark.json").write_text(
        "[]\n",
        encoding="utf-8",
    )
    (benchmark / "manifest.json").write_text("{}\n", encoding="utf-8")
    residual = tmp_path / "residual.pt"
    flow = tmp_path / "flow.pt"
    residual.write_bytes(b"residual")
    flow.write_bytes(b"flow")
    hf_home = tmp_path / "hf"
    _write_fake_hf_contract(hf_home)
    monkeypatch.setattr(
        v14_harness.v13,
        "_gpu_inventory",
        lambda: {"available": False, "gpus": []},
    )
    monkeypatch.setattr(
        v14_harness.v13,
        "_git_state",
        lambda path: {"path": str(path), "sha": "test", "dirty": False},
    )
    run_dir = tmp_path / "rlt_cf_v14_controlled"
    manifest = build_manifest(
        root=HERE,
        run_dir=run_dir,
        log_dir=tmp_path / "logs",
        benchmark_root=benchmark,
        residual_checkpoint=residual,
        flow_checkpoint=flow,
        python_executable="/python",
        serve_prefix=["/serve-python"],
        tmp_rollout_dir=tmp_path / "rollouts",
        egl_lock_dir=tmp_path / "egl",
        gpu_ids=[str(index) for index in range(8)],
        environment={
            "HF_HOME": str(hf_home),
            "HF_TOKEN": "must-not-appear",
            "V14_MODE": "full",
        },
    )

    assert manifest["schema_version"] == "v14-controlled-1"
    assert manifest["run"]["name"] == "rlt_cf_v14_controlled"
    fixed = manifest["fixed_parameters"]
    assert fixed["ae_image_replay_capacity"] == 2048
    assert fixed["ae_batch_size"] == AE_BATCH_SIZE
    assert fixed["ae_min_success_episodes"] == 3
    assert fixed["q_tail_fraction"] == Q_TAIL_FRACTION
    assert fixed["q_tail_min_heads"] == Q_TAIL_MIN_HEADS
    assert fixed["critic_target_policy"] == CRITIC_TARGET_POLICY
    assert fixed["critic_target_use_guide"] is False
    assert fixed["allow_legacy_ae_resume"] is False
    assert len(manifest["variants"]) == 8
    assert len(manifest["servers"]) == 6
    assert len(manifest["provenance"]["sources"]) == 8
    assert (
        manifest["provenance"]["native_action_contract"]["action_horizon"]
        == 15
    )
    serialized = json.dumps(manifest)
    assert "must-not-appear" not in serialized
    assert "rlt_cf_v13_controlled" not in serialized
    for record in manifest["variants"]:
        assert "--no_critic_target_use_guide" in record["initial_fresh_command"]
        assert "--allow_legacy_ae_resume" not in record["resume_command"]

    manifest_path = run_dir / "MANIFEST.json"
    v14_harness.v13.atomic_write_json(manifest_path, manifest)
    result = validate_manifest(manifest_path, expected_run_dir=run_dir)
    assert result["valid"], result["errors"]


def test_strict_ae_resume_rejects_legacy_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "rlt_cf_v14_controlled"
    out_dir = run_dir / "molmo_ae_lora_actor"
    out_dir.mkdir(parents=True)
    for name in (
        "rlt_cf_latest.pt",
        "chunk_replay.npz",
        "metrics.jsonl",
        "molmo_ae_lora_latest.pt",
    ):
        (out_dir / name).write_bytes(b"present")
    monkeypatch.setattr(
        v14_harness,
        "validate_manifest",
        lambda path, expected_run_dir=None: {"valid": True, "errors": []},
    )

    replay_path = out_dir / "ae_image_replay.npz"
    with zipfile.ZipFile(replay_path, "w") as archive:
        archive.writestr("reference_actions.npy", b"legacy")
    assert resume_state(out_dir, ae_trainable=True) == "partial"

    with zipfile.ZipFile(replay_path, "w") as archive:
        for member in REQUIRED_AE_REPLAY_MEMBERS:
            archive.writestr(member, b"v14")
    assert resume_state(out_dir, ae_trainable=True) == "complete"


def test_cli_contract_reports_concurrent_source_state_exactly() -> None:
    train_script = HERE / "train_rlt_online.py"
    result = validate_trainer_cli(train_script)
    source = train_script.read_text(encoding="utf-8")
    expected_missing = [
        option
        for option in REQUIRED_V14_TRAINER_OPTIONS
        if option not in source
    ]
    assert result["missing_options"] == expected_missing
    assert result["valid"] is (not expected_missing)
    assert set(result["required_options"]) == set(REQUIRED_V14_TRAINER_OPTIONS)


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
