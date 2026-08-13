"""Thin, fail-closed specifications for the controlled V14 harness."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import v13_harness as v13


RUN_NAME = "rlt_cf_v14_controlled"
BENCHMARK_NAME = v13.BENCHMARK_NAME
TRAIN_EPISODES = v13.TRAIN_EPISODES
VALIDATION_EPISODES = v13.VALIDATION_EPISODES
HORIZON = v13.HORIZON
TRAIN_SEED = v13.TRAIN_SEED
INTERIM_VALIDATION_SEEDS = v13.INTERIM_VALIDATION_SEEDS
FINAL_VALIDATION_SEEDS = v13.FINAL_VALIDATION_SEEDS
N_CRITICS = v13.N_CRITICS
FLOW_STEPS = v13.FLOW_STEPS
GUIDANCE_COEF = v13.GUIDANCE_COEF
PAIRED_LCB_THRESHOLD = v13.PAIRED_LCB_THRESHOLD
ACTION_SENSITIVITY_THRESHOLD = v13.ACTION_SENSITIVITY_THRESHOLD
EXPLORE_STD = v13.EXPLORE_STD
BC_REF_COEF = v13.BC_REF_COEF
AE_LORA_RANK = v13.AE_LORA_RANK
AE_LORA_ALPHA = v13.AE_LORA_ALPHA

MAX_VALID_EPISODES = int(os.environ.get("V14_MAX_VALID_EPISODES", "400"))
TARGET_ENV_STEPS = int(os.environ.get("V14_TARGET_ENV_STEPS", "250000"))
SNAPSHOT_EPISODES = tuple(
    int(token.strip())
    for token in os.environ.get(
        "V14_SNAPSHOT_EPISODES",
        "0,100,200,400",
    ).split(",")
    if token.strip()
)
AE_IMAGE_REPLAY_CAPACITY = 2048
AE_BATCH_SIZE = int(os.environ.get("V14_AE_BATCH_SIZE", "16"))
AE_MICROBATCH_SIZE = int(os.environ.get("V14_AE_MICROBATCH_SIZE", "4"))
AE_MIN_SUCCESS_EPISODES = int(
    os.environ.get("V14_AE_MIN_SUCCESS_EPISODES", "3")
)
BENCHMARK_POSE_CYCLE = TRAIN_EPISODES
MAX_UPDATE_SEC_PER_EPISODE = float(
    os.environ.get("V14_MAX_UPDATE_SEC_PER_EPISODE", "30")
)
Q_TAIL_FRACTION = 0.25
Q_TAIL_MIN_HEADS = 2
CRITIC_TARGET_POLICY = "actor"
RUN_MODE = os.environ.get("V14_MODE", "full")
HTTP_PORTS = v13.HTTP_PORTS
VARIANTS = tuple(v13.VARIANTS)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}
VariantSpec = v13.VariantSpec

MOLMO_AE_REPO_ID = "allenai/MolmoAct2-DROID"
MOLMO_AE_NORM_TAG = "franka_droid"
EXPECTED_NATIVE_ACTION_CONTRACT = {
    "normalization_mode": "q01_q99",
    "normalization_tag": MOLMO_AE_NORM_TAG,
    "action_horizon": 15,
    "n_action_steps": 15,
    "n_obs_steps": 1,
    "action_dim": 8,
    "max_action_dim": 32,
    "deployment_chunk_horizon": 8,
    "raw_action_scope": "replay_and_environment_boundaries_only",
    "invalid_action_policy": "fatal_no_fallback",
}
REQUIRED_V14_TRAINER_OPTIONS = (
    "--benchmark_pose_cycle",
    "--ae_microbatch_size",
    "--ae_min_success_episodes",
    "--max_update_sec_per_episode",
    "--max_updates_per_episode",
    "--allow_legacy_ae_resume",
    "--critic_target_use_guide",
    "--no_critic_target_use_guide",
)
REQUIRED_AE_REPLAY_MEMBERS = frozenset(
    {
        "full_reference_actions.npy",
        "full_executed_actions.npy",
        "next_full_reference_actions.npy",
        "source_native.npy",
        "next_source_native.npy",
        "rng_state_json.npy",
    }
)


if len(VARIANTS) != 8 or VARIANTS != tuple(v13.VARIANTS):
    raise RuntimeError("V14 must preserve the exact eight V13 arm definitions")
if MAX_VALID_EPISODES < 1:
    raise ValueError("V14_MAX_VALID_EPISODES must be positive")
if TARGET_ENV_STEPS < 1:
    raise ValueError("V14_TARGET_ENV_STEPS must be positive")
if AE_BATCH_SIZE < 2:
    raise ValueError("V14_AE_BATCH_SIZE must be at least 2")
if AE_MICROBATCH_SIZE < 1:
    raise ValueError("V14_AE_MICROBATCH_SIZE must be positive")
if AE_MIN_SUCCESS_EPISODES < 1:
    raise ValueError("V14_AE_MIN_SUCCESS_EPISODES must be positive")
if MAX_UPDATE_SEC_PER_EPISODE <= 0.0 or not math.isfinite(
    MAX_UPDATE_SEC_PER_EPISODE
):
    raise ValueError("V14_MAX_UPDATE_SEC_PER_EPISODE must be finite and positive")
if not SNAPSHOT_EPISODES or any(episode < 0 for episode in SNAPSHOT_EPISODES):
    raise ValueError("V14_SNAPSHOT_EPISODES must contain non-negative milestones")
if len(set(SNAPSHOT_EPISODES)) != len(SNAPSHOT_EPISODES):
    raise ValueError("V14_SNAPSHOT_EPISODES must not contain duplicates")
if 0 not in SNAPSHOT_EPISODES or MAX_VALID_EPISODES not in SNAPSHOT_EPISODES:
    raise ValueError(
        "V14_SNAPSHOT_EPISODES must include episode 0 and the final episode"
    )
if RUN_MODE not in {"full", "smoke"}:
    raise ValueError("V14_MODE must be full or smoke")


def _assert_v14_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    if resolved.name != RUN_NAME:
        raise ValueError(
            f"V14 run directory basename must be {RUN_NAME!r}, got {resolved}"
        )
    if "rlt_cf_v13_controlled" in resolved.parts:
        raise ValueError(f"V14 refuses a V13 run path: {resolved}")
    return resolved


def _argument_value(command: Sequence[str], flag: str) -> str:
    try:
        index = command.index(flag)
    except ValueError as error:
        raise ValueError(f"Generated command is missing required flag {flag}") from error
    if index + 1 >= len(command):
        raise ValueError(f"Generated command has no value after {flag}")
    return str(command[index + 1])


def _set_argument(command: list[str], flag: str, value: object) -> None:
    index = command.index(flag)
    if index + 1 >= len(command):
        raise ValueError(f"Generated command has no value after {flag}")
    command[index + 1] = str(value)


def _append_option(command: list[str], flag: str, value: object | None = None) -> None:
    if flag in command:
        raise ValueError(f"Generated command already contains {flag}")
    command.append(flag)
    if value is not None:
        command.append(str(value))


def training_output_dir(run_dir: Path, variant: VariantSpec) -> Path:
    return _assert_v14_run_dir(run_dir) / variant.name


def snapshot_dir(run_dir: Path, variant: VariantSpec, episode: int) -> Path:
    return training_output_dir(run_dir, variant) / "snapshots" / f"ep_{episode:06d}"


def validation_output_dir(
    run_dir: Path,
    variant: VariantSpec,
    episode: int,
    policy: str,
    seed: int,
) -> Path:
    return (
        _assert_v14_run_dir(run_dir)
        / "validation"
        / variant.name
        / f"ep_{episode:06d}"
        / policy
        / f"seed_{seed}"
    )


def evaluation_policies(variant: VariantSpec) -> tuple[str, ...]:
    if variant.is_baseline:
        return ("reference",)
    if variant.name in {"residual_rlt_actor", "flow_rlt_actor"}:
        return ("reference", "reference_noise", "actor")
    if variant.use_guide:
        return ("reference", "actor", "actor_guide")
    return ("reference", "actor")


def build_train_command(
    *,
    python_executable: str,
    root: Path,
    run_dir: Path,
    benchmark_train: Path,
    residual_checkpoint: Path,
    flow_checkpoint: Path,
    tmp_rollout_dir: Path,
    variant: VariantSpec,
    fresh: bool,
) -> list[str]:
    """Adapt the V13 command without inheriting any V13 output location."""

    run_dir = _assert_v14_run_dir(run_dir)
    command = v13.build_train_command(
        python_executable=python_executable,
        root=root,
        run_dir=run_dir,
        benchmark_train=benchmark_train,
        residual_checkpoint=residual_checkpoint,
        flow_checkpoint=flow_checkpoint,
        tmp_rollout_dir=tmp_rollout_dir,
        variant=variant,
        fresh=fresh,
    )
    _set_argument(command, "--out_dir", training_output_dir(run_dir, variant))
    _set_argument(command, "--max_valid_episodes", MAX_VALID_EPISODES)
    _set_argument(command, "--target_env_steps", TARGET_ENV_STEPS)
    _set_argument(
        command,
        "--snapshot_episodes",
        ",".join(str(episode) for episode in SNAPSHOT_EPISODES),
    )
    _set_argument(command, "--updates_per_episode", variant.updates_per_episode)
    _append_option(
        command,
        "--max_updates_per_episode",
        variant.updates_per_episode,
    )
    _append_option(
        command,
        "--max_update_sec_per_episode",
        MAX_UPDATE_SEC_PER_EPISODE,
    )
    _append_option(command, "--no_critic_target_use_guide")
    _append_option(
        command,
        "--benchmark_pose_cycle",
        BENCHMARK_POSE_CYCLE,
    )
    _append_option(
        command,
        "--ae_microbatch_size",
        AE_MICROBATCH_SIZE,
    )
    if variant.ae_trainable:
        _set_argument(command, "--ae_batch_size", AE_BATCH_SIZE)
        _set_argument(
            command,
            "--ae_image_replay_capacity",
            AE_IMAGE_REPLAY_CAPACITY,
        )
        _append_option(
            command,
            "--ae_min_success_episodes",
            AE_MIN_SUCCESS_EPISODES,
        )
    if "--allow_legacy_ae_resume" in command:
        raise ValueError("V14 commands must never enable legacy AE resume")
    return command


def build_server_command(
    *,
    serve_prefix: Sequence[str],
    root: Path,
    variant: VariantSpec,
    checkpoint: Path,
) -> list[str]:
    return v13.build_server_command(
        serve_prefix=serve_prefix,
        root=root,
        variant=variant,
        checkpoint=checkpoint,
    )


def build_eval_command(
    *,
    python_executable: str,
    root: Path,
    run_dir: Path,
    benchmark_val: Path,
    tmp_rollout_dir: Path,
    variant: VariantSpec,
    episode: int,
    policy: str,
    seed: int,
) -> list[str]:
    """Build a validation-only command rooted under the V14 validation tree."""

    if policy not in evaluation_policies(variant):
        raise ValueError(f"Policy {policy!r} is not valid for {variant.name}")
    run_dir = _assert_v14_run_dir(run_dir)
    delegated_policy = (
        "reference_noise"
        if policy == "reference_noise"
        else (
            "checkpoint_gate"
            if policy == "reference" and not variant.is_baseline
            else policy
        )
    )
    command = v13.build_eval_command(
        python_executable=python_executable,
        root=root,
        run_dir=run_dir,
        benchmark_val=benchmark_val,
        tmp_rollout_dir=tmp_rollout_dir,
        variant=variant,
        episode=episode,
        policy=delegated_policy,
        seed=seed,
    )
    _set_argument(
        command,
        "--out_dir",
        validation_output_dir(run_dir, variant, episode, policy, seed),
    )
    runtime_policy = "reference" if policy == "reference_noise" else policy
    _set_argument(command, "--deploy_policy", runtime_policy)
    if policy == "reference_noise":
        if "--eval_reference_noise_std" in command:
            _set_argument(
                command,
                "--eval_reference_noise_std",
                EXPLORE_STD,
            )
        else:
            _append_option(
                command,
                "--eval_reference_noise_std",
                EXPLORE_STD,
            )
    _set_argument(command, "--max_valid_episodes", VALIDATION_EPISODES)
    _set_argument(command, "--updates_per_episode", 0)
    _append_option(command, "--max_updates_per_episode", 0)
    _append_option(
        command,
        "--max_update_sec_per_episode",
        MAX_UPDATE_SEC_PER_EPISODE,
    )
    _append_option(command, "--no_critic_target_use_guide")
    if variant.ae_trainable:
        _set_argument(command, "--ae_batch_size", AE_BATCH_SIZE)
        _append_option(
            command,
            "--ae_microbatch_size",
            AE_MICROBATCH_SIZE,
        )
        _set_argument(
            command,
            "--ae_image_replay_capacity",
            AE_IMAGE_REPLAY_CAPACITY,
        )
        _append_option(
            command,
            "--ae_min_success_episodes",
            AE_MIN_SUCCESS_EPISODES,
        )
    if "--eval_only" not in command or "--no_resume" not in command:
        raise ValueError("V14 evaluation must be eval-only and non-resuming")
    if "--allow_legacy_ae_resume" in command:
        raise ValueError("V14 evaluation must never enable legacy AE resume")
    return command


def _hf_model_cache_dir(hf_home: Path) -> Path:
    return hf_home / "hub" / "models--allenai--MolmoAct2-DROID"


def resolve_native_action_contract(hf_home: Path) -> dict[str, Any]:
    """Resolve and verify the cached Molmo AE contract without loading weights."""

    model_cache = _hf_model_cache_dir(hf_home.resolve())
    revision_path = model_cache / "refs" / "main"
    if not revision_path.is_file():
        raise FileNotFoundError(
            f"Missing cached Molmo AE revision pointer: {revision_path}"
        )
    revision = revision_path.read_text(encoding="utf-8").strip()
    if not revision:
        raise ValueError(f"Empty Molmo AE revision pointer: {revision_path}")
    snapshot = model_cache / "snapshots" / revision
    config_path = snapshot / "config.json"
    norm_stats_path = snapshot / "norm_stats.json"
    if not config_path.is_file() or not norm_stats_path.is_file():
        raise FileNotFoundError(
            "Cached Molmo AE snapshot lacks config.json or norm_stats.json: "
            f"{snapshot}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    norm_stats = json.loads(norm_stats_path.read_text(encoding="utf-8"))
    tags = norm_stats.get("metadata_by_tag")
    if not isinstance(tags, dict) or MOLMO_AE_NORM_TAG not in tags:
        raise ValueError(
            f"Molmo AE norm stats lack tag {MOLMO_AE_NORM_TAG!r}"
        )
    tag = tags[MOLMO_AE_NORM_TAG]
    if not isinstance(tag, dict):
        raise TypeError("Molmo AE tag metadata must be a dictionary")
    action_stats = tag.get("action_stats")
    if not isinstance(action_stats, dict):
        raise TypeError("Molmo AE action_stats must be a dictionary")
    action_names = action_stats.get("names")
    q01 = action_stats.get("q01")
    q99 = action_stats.get("q99")
    if not isinstance(action_names, list) or not isinstance(q01, list) or not isinstance(q99, list):
        raise TypeError("Molmo AE action names/q01/q99 must be lists")
    observed = {
        "normalization_mode": norm_stats.get("norm_mode"),
        "normalization_tag": MOLMO_AE_NORM_TAG,
        "action_horizon": tag.get("action_horizon"),
        "n_action_steps": tag.get("n_action_steps"),
        "n_obs_steps": config.get("n_obs_steps"),
        "action_dim": len(action_names),
        "max_action_dim": config.get("max_action_dim"),
        "deployment_chunk_horizon": EXPECTED_NATIVE_ACTION_CONTRACT[
            "deployment_chunk_horizon"
        ],
        "raw_action_scope": EXPECTED_NATIVE_ACTION_CONTRACT["raw_action_scope"],
        "invalid_action_policy": EXPECTED_NATIVE_ACTION_CONTRACT[
            "invalid_action_policy"
        ],
    }
    if observed != EXPECTED_NATIVE_ACTION_CONTRACT:
        raise ValueError(
            "Molmo AE native action contract mismatch: "
            f"observed={observed}, expected={EXPECTED_NATIVE_ACTION_CONTRACT}"
        )
    if len(q01) != observed["action_dim"] or len(q99) != observed["action_dim"]:
        raise ValueError("Molmo AE q01/q99 widths do not match action_dim")
    return {
        **observed,
        "repo_id": MOLMO_AE_REPO_ID,
        "resolved_revision": revision,
        "snapshot_path": str(snapshot.resolve()),
        "config_path": str(config_path.resolve()),
        "config_sha256": v13.sha256_file(config_path),
        "norm_stats_path": str(norm_stats_path.resolve()),
        "norm_stats_sha256": v13.sha256_file(norm_stats_path),
        "action_names": action_names,
        "action_q01": q01,
        "action_q99": q99,
        "action_normalization_mask": action_stats.get("mask"),
        "normalize_gripper": tag.get("normalize_gripper"),
        "control_mode": tag.get("control_mode"),
        "setup_type": tag.get("setup_type"),
        "resolution_api": "huggingface_hub.snapshot_download(repo_id=repo_id)",
    }


def _source_provenance(root: Path) -> dict[str, dict[str, str]]:
    names = (
        "v14_harness.py",
        "launch_v14_controlled.sh",
        "eval_v14_controlled.sh",
        "train_rlt_online.py",
        "train_rlt.py",
        "rlt_models.py",
        "molmo_ae_backend.py",
        "chunk_replay.py",
    )
    records: dict[str, dict[str, str]] = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing V14 provenance source: {path}")
        records[name] = {
            "path": str(path.resolve()),
            "sha256": v13.sha256_file(path),
        }
    return records


def build_manifest(
    *,
    root: Path,
    run_dir: Path,
    log_dir: Path,
    benchmark_root: Path,
    residual_checkpoint: Path,
    flow_checkpoint: Path,
    python_executable: str,
    serve_prefix: Sequence[str],
    tmp_rollout_dir: Path,
    egl_lock_dir: Path,
    gpu_ids: Sequence[str],
    environment: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the immutable launch contract for all eight V14 arms."""

    root = root.resolve()
    run_dir = _assert_v14_run_dir(run_dir)
    log_dir = log_dir.resolve()
    benchmark_root = benchmark_root.resolve()
    residual_checkpoint = residual_checkpoint.resolve()
    flow_checkpoint = flow_checkpoint.resolve()
    environment = dict(os.environ if environment is None else environment)
    if len(gpu_ids) != len(VARIANTS):
        raise ValueError(f"Expected eight GPU IDs, got {len(gpu_ids)}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("V14 requires eight distinct physical GPU IDs")
    benchmark_files = {
        "train": benchmark_root / "train" / "benchmark.json",
        "val": benchmark_root / "val" / "benchmark.json",
        "manifest": benchmark_root / "manifest.json",
    }
    required = [
        *benchmark_files.values(),
        residual_checkpoint,
        flow_checkpoint,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Manifest inputs are missing: " + ", ".join(missing))
    hf_home = Path(
        environment.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    )
    action_contract = resolve_native_action_contract(hf_home)
    variant_records = []
    server_records = []
    for index, variant in enumerate(VARIANTS):
        gpu_id = str(gpu_ids[index])
        checkpoint = v13.variant_checkpoint(
            variant,
            residual_checkpoint,
            flow_checkpoint,
        )
        fresh_command = build_train_command(
            python_executable=python_executable,
            root=root,
            run_dir=run_dir,
            benchmark_train=benchmark_root / "train",
            residual_checkpoint=residual_checkpoint,
            flow_checkpoint=flow_checkpoint,
            tmp_rollout_dir=tmp_rollout_dir,
            variant=variant,
            fresh=True,
        )
        resume_command = build_train_command(
            python_executable=python_executable,
            root=root,
            run_dir=run_dir,
            benchmark_train=benchmark_root / "train",
            residual_checkpoint=residual_checkpoint,
            flow_checkpoint=flow_checkpoint,
            tmp_rollout_dir=tmp_rollout_dir,
            variant=variant,
            fresh=False,
        )
        record = dataclasses.asdict(variant)
        record.update(
            {
                "physical_gpu": gpu_id,
                "checkpoint": str(checkpoint),
                "initial_fresh_command": v13.redact_command(fresh_command),
                "resume_command": v13.redact_command(resume_command),
                "evaluation_policies": list(evaluation_policies(variant)),
                "snapshot_paths": [
                    str(snapshot_dir(run_dir, variant, episode))
                    for episode in SNAPSHOT_EPISODES
                ],
                "max_updates_per_episode": variant.updates_per_episode,
                "critic_target_policy": CRITIC_TARGET_POLICY,
                "allow_legacy_ae_resume": False,
            }
        )
        variant_records.append(record)
        if variant.server_port is not None:
            server_records.append(
                {
                    "variant": variant.name,
                    "physical_gpu": gpu_id,
                    "port": variant.server_port,
                    "command": v13.redact_command(
                        build_server_command(
                            serve_prefix=serve_prefix,
                            root=root,
                            variant=variant,
                            checkpoint=checkpoint,
                        )
                    ),
                }
            )
    relevant_environment = {
        key: v13.redact_value(key, value)
        for key, value in environment.items()
        if key.startswith("V14_")
        or key
        in {
            "B1K_ROOT",
            "B1K_TMP",
            "HF_HOME",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "MLSPACES_ASSETS_DIR",
            "RLT_EGL_MAX_CONCURRENT",
            "RLT_EGL_PER_GPU",
            "RLT_EGL_COOLDOWN_SEC",
            "RLT_IO_RETRY_ATTEMPTS",
            "RLT_IO_RETRY_BASE_SEC",
            "GPU_IDS",
            "FRESH",
            "MUJOCO_GL",
            "PYOPENGL_PLATFORM",
        }
    }
    return {
        "schema_version": "v14-controlled-1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "hostname": v13.socket.gethostname(),
            "fqdn": v13.socket.getfqdn(),
            "platform": v13.platform.platform(),
            "python": sys.version,
        },
        "gpu_inventory": v13._gpu_inventory(),
        "repositories": {
            name: v13._git_state(path)
            for name, path in {
                "rql": root.parents[1],
                "molmoact2": root.parents[2] / "molmoact2",
                "molmospaces": root.parents[2] / "molmospaces",
                "b1k_airi": root.parents[3],
            }.items()
        },
        "run": {
            "name": RUN_NAME,
            "mode": RUN_MODE,
            "run_dir": str(run_dir),
            "log_dir": str(log_dir),
            "tmp_rollout_dir": str(tmp_rollout_dir.resolve()),
            "egl_lock_dir": str(egl_lock_dir.resolve()),
            "v13_artifacts_read_only": True,
            "fresh_semantics": (
                "FRESH=1 deletes only the eight V14 arm directories and V14 "
                "validation outputs after owned-PID checks; watchdog restarts "
                "never use --no_resume."
            ),
        },
        "benchmark": {
            "name": BENCHMARK_NAME,
            "root": str(benchmark_root),
            "train_indices": [0, TRAIN_EPISODES - 1],
            "validation_indices": [0, VALIDATION_EPISODES - 1],
            "files": {
                label: {
                    "path": str(path.resolve()),
                    "sha256": v13.sha256_file(path),
                }
                for label, path in benchmark_files.items()
            },
        },
        "checkpoints": {
            "residual": {
                "path": str(residual_checkpoint),
                "sha256": v13.sha256_file(residual_checkpoint),
            },
            "flow": {
                "path": str(flow_checkpoint),
                "sha256": v13.sha256_file(flow_checkpoint),
            },
        },
        "provenance": {
            "sources": _source_provenance(root),
            "native_action_contract": action_contract,
            "trainer_cli_contract": list(REQUIRED_V14_TRAINER_OPTIONS),
        },
        "fixed_parameters": {
            "controlled_train_episodes": TRAIN_EPISODES,
            "controlled_validation_episodes": VALIDATION_EPISODES,
            "max_valid_episodes": MAX_VALID_EPISODES,
            "target_env_steps": TARGET_ENV_STEPS,
            "horizon": HORIZON,
            "snapshot_episodes": list(SNAPSHOT_EPISODES),
            "train_seed": TRAIN_SEED,
            "interim_validation_seeds": list(INTERIM_VALIDATION_SEEDS),
            "final_validation_seeds": list(FINAL_VALIDATION_SEEDS),
            "ae_image_replay_capacity": AE_IMAGE_REPLAY_CAPACITY,
            "ae_batch_size": AE_BATCH_SIZE,
            "ae_microbatch_size": AE_MICROBATCH_SIZE,
            "ae_accumulation_steps": int(
                math.ceil(AE_BATCH_SIZE / AE_MICROBATCH_SIZE)
            ),
            "ae_min_success_episodes": AE_MIN_SUCCESS_EPISODES,
            "benchmark_pose_cycle": BENCHMARK_POSE_CYCLE,
            "ae_critic_action_coordinates": "molmo_native_q01_q99_identity",
            "ae_legacy_critic_reuse": False,
            "max_update_sec_per_episode": MAX_UPDATE_SEC_PER_EPISODE,
            "max_updates_per_episode_by_variant": {
                variant.name: variant.updates_per_episode
                for variant in VARIANTS
            },
            "q_tail_fraction": Q_TAIL_FRACTION,
            "q_tail_min_heads": Q_TAIL_MIN_HEADS,
            "critic_target_policy": CRITIC_TARGET_POLICY,
            "critic_target_use_guide": False,
            "allow_legacy_ae_resume": False,
            "deploy_policy": "gated",
            "paired_lcb_threshold": PAIRED_LCB_THRESHOLD,
            "action_sensitivity_threshold": ACTION_SENSITIVITY_THRESHOLD,
            "exploration_std": EXPLORE_STD,
            "bc_ref_coef": BC_REF_COEF,
            "frozen_token": True,
            "n_critics": N_CRITICS,
            "flow_steps": FLOW_STEPS,
            "guidance_coef": GUIDANCE_COEF,
            "http_ports": list(HTTP_PORTS),
            "ae_lora_rank": AE_LORA_RANK,
            "ae_lora_alpha": AE_LORA_ALPHA,
        },
        "environment": relevant_environment,
        "servers": server_records,
        "variants": variant_records,
    }


def _validate_hashed_record(
    record: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    path = Path(str(record.get("path", "")))
    expected = str(record.get("sha256", ""))
    if not path.is_file():
        errors.append(f"{label} is missing: {path}")
        return
    if not expected or v13.sha256_file(path) != expected:
        errors.append(f"{label} sha256 changed: {path}")


def validate_manifest(
    manifest_path: Path,
    *,
    expected_run_dir: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read V14 manifest {manifest_path}: {error}") from error
    if payload.get("schema_version") != "v14-controlled-1":
        errors.append("schema_version is not v14-controlled-1")
    run = payload.get("run")
    if not isinstance(run, dict) or run.get("name") != RUN_NAME:
        errors.append(f"run.name is not {RUN_NAME}")
    recorded_run_dir = Path(str(run.get("run_dir", ""))) if isinstance(run, dict) else Path()
    if recorded_run_dir.name != RUN_NAME:
        errors.append("recorded run_dir is not V14-isolated")
    if expected_run_dir is not None and recorded_run_dir.resolve() != _assert_v14_run_dir(
        expected_run_dir
    ):
        errors.append("recorded run_dir does not match the requested V14 run")
    fixed = payload.get("fixed_parameters")
    if not isinstance(fixed, dict):
        errors.append("fixed_parameters is missing")
    else:
        expected_fixed = {
            "ae_image_replay_capacity": AE_IMAGE_REPLAY_CAPACITY,
            "ae_batch_size": AE_BATCH_SIZE,
            "ae_microbatch_size": AE_MICROBATCH_SIZE,
            "ae_min_success_episodes": AE_MIN_SUCCESS_EPISODES,
            "benchmark_pose_cycle": BENCHMARK_POSE_CYCLE,
            "ae_critic_action_coordinates": "molmo_native_q01_q99_identity",
            "ae_legacy_critic_reuse": False,
            "q_tail_fraction": Q_TAIL_FRACTION,
            "q_tail_min_heads": Q_TAIL_MIN_HEADS,
            "critic_target_policy": CRITIC_TARGET_POLICY,
            "critic_target_use_guide": False,
            "allow_legacy_ae_resume": False,
        }
        for key, expected in expected_fixed.items():
            if fixed.get(key) != expected:
                errors.append(f"fixed_parameters.{key} != {expected!r}")
    records = payload.get("variants")
    if not isinstance(records, list) or len(records) != len(VARIANTS):
        errors.append("manifest must contain exactly eight variants")
        records = []
    else:
        for variant, record in zip(VARIANTS, records):
            if not isinstance(record, dict) or record.get("name") != variant.name:
                errors.append(f"variant order/name mismatch at {variant.name}")
                continue
            if record.get("allow_legacy_ae_resume") is not False:
                errors.append(f"{variant.name} permits legacy AE resume")
            for command_key in ("initial_fresh_command", "resume_command"):
                command = record.get(command_key)
                if not isinstance(command, list):
                    errors.append(f"{variant.name}.{command_key} is missing")
                    continue
                rendered = "\0".join(str(token) for token in command)
                if "rlt_cf_v13_controlled" in rendered:
                    errors.append(f"{variant.name}.{command_key} references V13")
                if "--allow_legacy_ae_resume" in command:
                    errors.append(f"{variant.name}.{command_key} enables legacy AE")
                if "--no_critic_target_use_guide" not in command:
                    errors.append(f"{variant.name}.{command_key} lacks target policy")
    for group_name in ("checkpoints",):
        group = payload.get(group_name)
        if not isinstance(group, dict):
            errors.append(f"{group_name} is missing")
            continue
        for label, record in group.items():
            if isinstance(record, dict):
                _validate_hashed_record(record, f"{group_name}.{label}", errors)
            else:
                errors.append(f"{group_name}.{label} is invalid")
    benchmark = payload.get("benchmark")
    files = benchmark.get("files") if isinstance(benchmark, dict) else None
    if not isinstance(files, dict):
        errors.append("benchmark.files is missing")
    else:
        for label, record in files.items():
            if isinstance(record, dict):
                _validate_hashed_record(record, f"benchmark.{label}", errors)
            else:
                errors.append(f"benchmark.{label} is invalid")
    provenance = payload.get("provenance")
    sources = provenance.get("sources") if isinstance(provenance, dict) else None
    if not isinstance(sources, dict):
        errors.append("provenance.sources is missing")
    else:
        for label, record in sources.items():
            if isinstance(record, dict):
                _validate_hashed_record(record, f"source.{label}", errors)
            else:
                errors.append(f"source.{label} is invalid")
    action_contract = (
        provenance.get("native_action_contract")
        if isinstance(provenance, dict)
        else None
    )
    if not isinstance(action_contract, dict):
        errors.append("native action contract provenance is missing")
    else:
        for key, expected in EXPECTED_NATIVE_ACTION_CONTRACT.items():
            if action_contract.get(key) != expected:
                errors.append(f"native_action_contract.{key} != {expected!r}")
        for label in ("config", "norm_stats"):
            record = {
                "path": action_contract.get(f"{label}_path"),
                "sha256": action_contract.get(f"{label}_sha256"),
            }
            _validate_hashed_record(
                record,
                f"native_action_contract.{label}",
                errors,
            )
    return {
        "valid": not errors,
        "manifest": str(manifest_path),
        "errors": errors,
        "variants": len(records),
    }


def manifest_gpu_ids(manifest_path: Path) -> list[str]:
    result = validate_manifest(manifest_path)
    if not result["valid"]:
        raise ValueError("Invalid V14 manifest: " + "; ".join(result["errors"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_name = {
        str(record["name"]): str(record["physical_gpu"])
        for record in payload["variants"]
    }
    gpu_ids = [by_name.get(variant.name, "") for variant in VARIANTS]
    if any(not gpu_id for gpu_id in gpu_ids):
        raise ValueError("Manifest is missing one or more V14 GPU assignments")
    return gpu_ids


def manifest_run_settings(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "v14-controlled-1":
        raise ValueError("Run settings require a v14-controlled-1 manifest")
    run = payload.get("run")
    fixed = payload.get("fixed_parameters")
    if not isinstance(run, dict) or run.get("name") != RUN_NAME:
        raise ValueError(f"Run settings manifest is not for {RUN_NAME}")
    if not isinstance(fixed, dict):
        raise ValueError("Run settings manifest lacks fixed_parameters")
    return {
        "mode": str(run["mode"]),
        "max_valid_episodes": int(fixed["max_valid_episodes"]),
        "target_env_steps": int(fixed["target_env_steps"]),
        "snapshot_episodes": [
            int(episode) for episode in fixed["snapshot_episodes"]
        ],
        "ae_batch_size": int(fixed["ae_batch_size"]),
        "ae_microbatch_size": int(fixed["ae_microbatch_size"]),
        "ae_min_success_episodes": int(
            fixed["ae_min_success_episodes"]
        ),
        "max_update_sec_per_episode": float(
            fixed["max_update_sec_per_episode"]
        ),
    }


def _pid_belongs_to_run(pid: int, run_dir: Path) -> bool:
    environ = Path(f"/proc/{pid}/environ")
    try:
        entries = environ.read_bytes().split(b"\0")
    except OSError:
        return False
    marker = f"RLT_CF_V14_RUN_DIR={run_dir.resolve()}".encode("utf-8")
    return marker in entries


def assert_gpu_ownership(gpu_ids: Sequence[str], run_dir: Path) -> dict[str, Any]:
    run_dir = _assert_v14_run_dir(run_dir)
    if len(gpu_ids) != 8 or len(set(gpu_ids)) != 8:
        raise ValueError("GPU ownership check requires eight distinct GPU IDs")
    inventory = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    uuid_by_index: dict[str, str] = {}
    for line in inventory.splitlines():
        values = [value.strip() for value in line.split(",", 1)]
        if len(values) == 2:
            uuid_by_index[values[0]] = values[1]
    missing = [gpu_id for gpu_id in gpu_ids if gpu_id not in uuid_by_index]
    if missing:
        raise ValueError(f"Requested GPU IDs are unavailable: {missing}")
    applications = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    selected_uuids = {uuid_by_index[gpu_id] for gpu_id in gpu_ids}
    collisions = []
    owned = []
    for line in applications.splitlines():
        values = [value.strip() for value in line.split(",", 1)]
        if len(values) != 2 or values[1] not in selected_uuids:
            continue
        try:
            pid = int(values[0])
        except ValueError:
            collisions.append({"pid": values[0], "gpu_uuid": values[1]})
            continue
        record = {"pid": pid, "gpu_uuid": values[1]}
        if _pid_belongs_to_run(pid, run_dir):
            owned.append(record)
        else:
            collisions.append(record)
    if collisions:
        raise RuntimeError(
            "Selected GPUs have compute processes not owned by this V14 run: "
            + json.dumps(collisions, sort_keys=True)
        )
    return {
        "gpu_ids": list(gpu_ids),
        "gpu_uuids": [uuid_by_index[gpu_id] for gpu_id in gpu_ids],
        "owned_processes": owned,
        "collisions": [],
    }


def _strict_ae_replay_is_v14(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return REQUIRED_AE_REPLAY_MEMBERS.issubset(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def resume_state(out_dir: Path, ae_trainable: bool) -> str:
    state = v13.resume_state(out_dir, ae_trainable)
    if state != "complete" or not ae_trainable:
        return state
    manifest_path = out_dir.parent / "MANIFEST.json"
    try:
        validation = validate_manifest(
            manifest_path,
            expected_run_dir=out_dir.parent,
        )
    except (OSError, TypeError, ValueError):
        return "partial"
    if not validation["valid"]:
        return "partial"
    if not _strict_ae_replay_is_v14(out_dir / "ae_image_replay.npz"):
        return "partial"
    return "complete"


def validate_trainer_cli(train_script: Path) -> dict[str, Any]:
    source = train_script.read_text(encoding="utf-8")
    missing = [
        option
        for option in REQUIRED_V14_TRAINER_OPTIONS
        if option not in source
    ]
    return {
        "valid": not missing,
        "train_script": str(train_script),
        "required_options": list(REQUIRED_V14_TRAINER_OPTIONS),
        "missing_options": missing,
    }


def _variants_tsv() -> str:
    return v13._variants_tsv()


def _add_command_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=["nul", "shell", "json"], default="nul")


def _emit_command(command: Sequence[str], output_format: str) -> None:
    if output_format == "nul":
        sys.stdout.buffer.write(
            b"\0".join(token.encode("utf-8") for token in command)
        )
        sys.stdout.buffer.write(b"\0")
        return
    if output_format == "shell":
        print(shlex.join(command))
        return
    if output_format == "json":
        print(json.dumps(list(command)))
        return
    raise ValueError(f"Unsupported command output format: {output_format}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    variants_parser = subparsers.add_parser("variants")
    variants_parser.add_argument("--format", choices=["json", "tsv"], default="tsv")

    train_parser = subparsers.add_parser("train-command")
    train_parser.add_argument("--variant", choices=sorted(VARIANT_BY_NAME), required=True)
    train_parser.add_argument("--root", type=Path, required=True)
    train_parser.add_argument("--run-dir", type=Path, required=True)
    train_parser.add_argument("--benchmark-train", type=Path, required=True)
    train_parser.add_argument("--residual-checkpoint", type=Path, required=True)
    train_parser.add_argument("--flow-checkpoint", type=Path, required=True)
    train_parser.add_argument("--python-executable", required=True)
    train_parser.add_argument("--tmp-rollout-dir", type=Path, required=True)
    train_parser.add_argument("--fresh", action="store_true")
    _add_command_output_args(train_parser)

    server_parser = subparsers.add_parser("server-command")
    server_parser.add_argument("--variant", choices=sorted(VARIANT_BY_NAME), required=True)
    server_parser.add_argument("--root", type=Path, required=True)
    server_parser.add_argument("--checkpoint", type=Path, required=True)
    server_parser.add_argument("--serve-prefix", action="append", default=[])
    _add_command_output_args(server_parser)

    eval_parser = subparsers.add_parser("eval-command")
    eval_parser.add_argument("--variant", choices=sorted(VARIANT_BY_NAME), required=True)
    eval_parser.add_argument("--root", type=Path, required=True)
    eval_parser.add_argument("--run-dir", type=Path, required=True)
    eval_parser.add_argument("--benchmark-val", type=Path, required=True)
    eval_parser.add_argument("--tmp-rollout-dir", type=Path, required=True)
    eval_parser.add_argument("--python-executable", required=True)
    eval_parser.add_argument("--episode", type=int, choices=SNAPSHOT_EPISODES, required=True)
    eval_parser.add_argument(
        "--policy",
        choices=["reference", "reference_noise", "actor", "actor_guide"],
        required=True,
    )
    eval_parser.add_argument("--seed", type=int, required=True)
    _add_command_output_args(eval_parser)

    manifest_parser = subparsers.add_parser("manifest")
    manifest_parser.add_argument("--output", type=Path, required=True)
    manifest_parser.add_argument("--root", type=Path, required=True)
    manifest_parser.add_argument("--run-dir", type=Path, required=True)
    manifest_parser.add_argument("--log-dir", type=Path, required=True)
    manifest_parser.add_argument("--benchmark-root", type=Path, required=True)
    manifest_parser.add_argument("--residual-checkpoint", type=Path, required=True)
    manifest_parser.add_argument("--flow-checkpoint", type=Path, required=True)
    manifest_parser.add_argument("--python-executable", required=True)
    manifest_parser.add_argument("--serve-prefix", action="append", default=[])
    manifest_parser.add_argument("--tmp-rollout-dir", type=Path, required=True)
    manifest_parser.add_argument("--egl-lock-dir", type=Path, required=True)
    manifest_parser.add_argument("--gpu-ids", required=True)

    validate_manifest_parser = subparsers.add_parser("validate-manifest")
    validate_manifest_parser.add_argument("--manifest", type=Path, required=True)
    validate_manifest_parser.add_argument("--run-dir", type=Path)

    manifest_gpu_parser = subparsers.add_parser("manifest-gpu-ids")
    manifest_gpu_parser.add_argument("--manifest", type=Path, required=True)

    manifest_settings_parser = subparsers.add_parser("manifest-run-settings")
    manifest_settings_parser.add_argument("--manifest", type=Path, required=True)
    manifest_settings_parser.add_argument(
        "--format",
        choices=["json", "tsv"],
        default="json",
    )

    resume_parser = subparsers.add_parser("resume-state")
    resume_parser.add_argument("--out-dir", type=Path, required=True)
    resume_parser.add_argument("--ae", action="store_true")

    training_parser = subparsers.add_parser("training-complete")
    training_parser.add_argument("--out-dir", type=Path, required=True)
    training_parser.add_argument(
        "--expected-episodes",
        type=int,
        default=MAX_VALID_EPISODES,
    )

    port_parser = subparsers.add_parser("port-free")
    port_parser.add_argument("--port", type=int, required=True)

    evaluation_parser = subparsers.add_parser("evaluation-complete")
    evaluation_parser.add_argument("--out-dir", type=Path, required=True)
    evaluation_parser.add_argument(
        "--expected-episodes",
        type=int,
        default=VALIDATION_EPISODES,
    )

    gpu_parser = subparsers.add_parser("assert-gpu-ownership")
    gpu_parser.add_argument("--gpu-ids", required=True)
    gpu_parser.add_argument("--run-dir", type=Path, required=True)

    cli_parser = subparsers.add_parser("validate-trainer-cli")
    cli_parser.add_argument("--train-script", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "variants":
        if args.format == "json":
            print(
                json.dumps(
                    [dataclasses.asdict(variant) for variant in VARIANTS],
                    indent=2,
                )
            )
        else:
            print(_variants_tsv())
        return 0
    if args.command == "train-command":
        command = build_train_command(
            python_executable=args.python_executable,
            root=args.root.resolve(),
            run_dir=args.run_dir,
            benchmark_train=args.benchmark_train.resolve(),
            residual_checkpoint=args.residual_checkpoint.resolve(),
            flow_checkpoint=args.flow_checkpoint.resolve(),
            tmp_rollout_dir=args.tmp_rollout_dir.resolve(),
            variant=VARIANT_BY_NAME[args.variant],
            fresh=args.fresh,
        )
        _emit_command(command, args.format)
        return 0
    if args.command == "server-command":
        if not args.serve_prefix:
            raise ValueError("At least one --serve-prefix token is required")
        command = build_server_command(
            serve_prefix=args.serve_prefix,
            root=args.root.resolve(),
            variant=VARIANT_BY_NAME[args.variant],
            checkpoint=args.checkpoint.resolve(),
        )
        _emit_command(command, args.format)
        return 0
    if args.command == "eval-command":
        command = build_eval_command(
            python_executable=args.python_executable,
            root=args.root.resolve(),
            run_dir=args.run_dir,
            benchmark_val=args.benchmark_val.resolve(),
            tmp_rollout_dir=args.tmp_rollout_dir.resolve(),
            variant=VARIANT_BY_NAME[args.variant],
            episode=args.episode,
            policy=args.policy,
            seed=args.seed,
        )
        _emit_command(command, args.format)
        return 0
    if args.command == "manifest":
        if not args.serve_prefix:
            raise ValueError("At least one --serve-prefix token is required")
        payload = build_manifest(
            root=args.root,
            run_dir=args.run_dir,
            log_dir=args.log_dir,
            benchmark_root=args.benchmark_root,
            residual_checkpoint=args.residual_checkpoint,
            flow_checkpoint=args.flow_checkpoint,
            python_executable=args.python_executable,
            serve_prefix=list(args.serve_prefix),
            tmp_rollout_dir=args.tmp_rollout_dir,
            egl_lock_dir=args.egl_lock_dir,
            gpu_ids=[
                item.strip()
                for item in args.gpu_ids.split(",")
                if item.strip()
            ],
        )
        v13.atomic_write_json(args.output, payload)
        print(json.dumps({"manifest": str(args.output), "variants": len(VARIANTS)}))
        return 0
    if args.command == "validate-manifest":
        result = validate_manifest(
            args.manifest,
            expected_run_dir=args.run_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    if args.command == "manifest-gpu-ids":
        print(",".join(manifest_gpu_ids(args.manifest)))
        return 0
    if args.command == "manifest-run-settings":
        settings = manifest_run_settings(args.manifest)
        if args.format == "json":
            print(json.dumps(settings, sort_keys=True))
        else:
            print(
                "\t".join(
                    [
                        settings["mode"],
                        str(settings["max_valid_episodes"]),
                        str(settings["target_env_steps"]),
                        ",".join(
                            str(episode)
                            for episode in settings["snapshot_episodes"]
                        ),
                        str(settings["ae_batch_size"]),
                        str(settings["ae_microbatch_size"]),
                        str(settings["ae_min_success_episodes"]),
                        str(settings["max_update_sec_per_episode"]),
                    ]
                )
            )
        return 0
    if args.command == "resume-state":
        state = resume_state(args.out_dir, args.ae)
        print(state)
        return 2 if state == "partial" else 0
    if args.command == "training-complete":
        complete = v13.training_complete(
            args.out_dir,
            args.expected_episodes,
        )
        print("complete" if complete else "incomplete")
        return 0 if complete else 1
    if args.command == "port-free":
        available = v13.port_is_available(args.port)
        print("free" if available else "in-use")
        return 0 if available else 1
    if args.command == "evaluation-complete":
        complete = v13.evaluation_complete(
            args.out_dir,
            args.expected_episodes,
        )
        print("complete" if complete else "incomplete")
        return 0 if complete else 1
    if args.command == "assert-gpu-ownership":
        result = assert_gpu_ownership(
            [
                item.strip()
                for item in args.gpu_ids.split(",")
                if item.strip()
            ],
            args.run_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "validate-trainer-cli":
        result = validate_trainer_cli(args.train_script)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["valid"] else 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
