"""Shared, side-effect-light specifications for the controlled V13 harness."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


RUN_NAME = "rlt_cf_v13_controlled"
BENCHMARK_NAME = "house0_kettle_v13"
TRAIN_EPISODES = 24
VALIDATION_EPISODES = 12
MAX_VALID_EPISODES = 400
HORIZON = 500
TARGET_ENV_STEPS = 250_000
SNAPSHOT_EPISODES = (0, 100, 200, 400)
TRAIN_SEED = 20260813
INTERIM_VALIDATION_SEEDS = (20260831,)
FINAL_VALIDATION_SEEDS = (20260831, 20260832, 20260833, 20260834)
HTTP_PORTS = tuple(range(8700, 8706))
N_CRITICS = 10
FLOW_STEPS = 10
GUIDANCE_COEF = 0.5
PAIRED_LCB_THRESHOLD = 0.003
ACTION_SENSITIVITY_THRESHOLD = 0.003
EXPLORE_STD = 0.02
BC_REF_COEF = 1.0
AE_IMAGE_REPLAY_CAPACITY = 128
AE_LORA_RANK = 16
AE_LORA_ALPHA = 32


@dataclass(frozen=True)
class VariantSpec:
    """One controlled arm and its fixed GPU/model assignment."""

    name: str
    gpu: int
    cf_mode: str
    actor_mode: str
    use_guide: bool
    ae_trainable: bool
    checkpoint_kind: str
    updates_per_episode: int
    server_port: int | None
    actor_class: str
    critic_class: str
    guide_class: str | None

    @property
    def is_baseline(self) -> bool:
        return self.actor_mode == "vla_only"

    @property
    def policies(self) -> tuple[str, ...]:
        if self.is_baseline:
            return ("reference",)
        if self.use_guide:
            return ("checkpoint_gate", "actor", "actor_guide")
        return ("checkpoint_gate", "actor")


VARIANTS = (
    VariantSpec(
        "residual_vla_baseline",
        0,
        "residual",
        "vla_only",
        False,
        False,
        "residual",
        0,
        8700,
        "ChunkGaussianActor",
        "EnsembleCQL",
        None,
    ),
    VariantSpec(
        "residual_rlt_actor",
        1,
        "residual",
        "rlt",
        False,
        False,
        "residual",
        8,
        8701,
        "ChunkGaussianActor",
        "EnsembleCQL",
        None,
    ),
    VariantSpec(
        "residual_rlt_cf",
        2,
        "residual",
        "rlt",
        True,
        False,
        "residual",
        8,
        8702,
        "ChunkGaussianActor",
        "EnsembleCQL",
        "CFGradientGuide",
    ),
    VariantSpec(
        "flow_vla_baseline",
        3,
        "flow",
        "vla_only",
        False,
        False,
        "flow",
        0,
        8703,
        "FlowVelocityActor",
        "EnsembleTimeCQL",
        None,
    ),
    VariantSpec(
        "flow_rlt_actor",
        4,
        "flow",
        "rlt",
        False,
        False,
        "flow",
        8,
        8704,
        "FlowVelocityActor",
        "EnsembleTimeCQL",
        None,
    ),
    VariantSpec(
        "flow_rlt_cf",
        5,
        "flow",
        "rlt",
        True,
        False,
        "flow",
        8,
        8705,
        "FlowVelocityActor",
        "EnsembleTimeCQL",
        "FlowCFGuide",
    ),
    VariantSpec(
        "molmo_ae_lora_actor",
        6,
        "flow",
        "rlt",
        False,
        True,
        "flow",
        4,
        None,
        "MolmoAEBackend(AE-LoRA)",
        "EnsembleTimeCQL",
        None,
    ),
    VariantSpec(
        "molmo_ae_lora_cf",
        7,
        "flow",
        "rlt",
        True,
        True,
        "flow",
        4,
        None,
        "MolmoAEBackend(AE-LoRA)",
        "EnsembleTimeCQL",
        "FlowCFGuide",
    ),
)

VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}

_SECRET_KEY_RE = re.compile(
    r"(?:token|secret|password|passwd|credential|api[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)
_URI_CREDENTIAL_RE = re.compile(r"(https?://)([^/@\s:]+):([^/@\s]+)@")


def variant_checkpoint(
    variant: VariantSpec,
    residual_checkpoint: Path,
    flow_checkpoint: Path,
) -> Path:
    if variant.checkpoint_kind == "residual":
        return residual_checkpoint
    if variant.checkpoint_kind == "flow":
        return flow_checkpoint
    raise ValueError(f"Unsupported checkpoint kind: {variant.checkpoint_kind}")


def training_output_dir(run_dir: Path, variant: VariantSpec) -> Path:
    return run_dir / variant.name


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
        run_dir
        / "validation"
        / variant.name
        / f"ep_{episode:06d}"
        / policy
        / f"seed_{seed}"
    )


def _common_model_args(variant: VariantSpec) -> list[str]:
    args = [
        "--n_critics",
        str(N_CRITICS),
        "--cf_mode",
        variant.cf_mode,
        "--flow_steps",
        str(FLOW_STEPS),
        "--guidance_coef",
        str(GUIDANCE_COEF),
        "--actor_mode",
        variant.actor_mode,
        "--freeze_token",
    ]
    args.append("--use_cf_guide" if variant.use_guide else "--no_cf_guide")
    if variant.ae_trainable:
        args.extend(
            [
                "--ae_trainable",
                "--ae_lora",
                "--ae_lora_rank",
                str(AE_LORA_RANK),
                "--ae_lora_alpha",
                str(AE_LORA_ALPHA),
                "--ae_batch_size",
                "1",
                "--ae_image_replay_capacity",
                str(AE_IMAGE_REPLAY_CAPACITY),
            ]
        )
    return args


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
    """Return the exact trainer argv for one arm."""

    out_dir = training_output_dir(run_dir, variant)
    checkpoint = variant_checkpoint(
        variant,
        residual_checkpoint,
        flow_checkpoint,
    )
    command = [
        python_executable,
        str(root / "train_rlt_online.py"),
        "--device",
        "cuda:0",
        "--out_dir",
        str(out_dir),
        "--config_name",
        variant.name,
        "--benchmark_dir",
        str(benchmark_train),
        "--target_env_steps",
        str(TARGET_ENV_STEPS),
        "--max_valid_episodes",
        str(MAX_VALID_EPISODES),
        "--start_episode",
        "0",
        "--shard_size",
        str(TRAIN_EPISODES),
        "--horizon",
        str(HORIZON),
        "--log_every_episodes",
        "10",
        "--ckpt_every_episodes",
        "10",
        "--snapshot_episodes",
        ",".join(str(episode) for episode in SNAPSHOT_EPISODES),
        "--window_episodes",
        "50",
        "--replay_out",
        str(out_dir / "chunk_replay.npz"),
        "--seed",
        str(TRAIN_SEED),
        "--deploy_policy",
        "gated",
        "--g_min_advantage",
        str(PAIRED_LCB_THRESHOLD),
        "--g_min_action_sensitivity",
        str(ACTION_SENSITIVITY_THRESHOLD),
        "--gate_sensitivity_noise",
        "0.08",
        "--explore_residual_std",
        str(EXPLORE_STD),
        "--explore_deploy_std",
        str(EXPLORE_STD),
        "--explore_warmup_mult",
        "1.0",
        "--bc_ref_coef",
        str(BC_REF_COEF),
        "--rank_coef",
        "1.0",
        "--rank_margin",
        "0.05",
        "--rank_noise",
        "0.08",
        "--far_rank_coef",
        "0.5",
        "--far_rank_noise",
        "0.35",
        "--shuffle_rank_coef",
        "0.5",
        "--target_noise",
        "0.02",
        "--cql_n_actions",
        "8",
        "--tmp_rollout_dir",
        str(tmp_rollout_dir),
        "--rlt_ckpt",
        str(checkpoint),
        "--updates_per_episode",
        str(variant.updates_per_episode),
    ]
    command.extend(_common_model_args(variant))
    if variant.ae_trainable:
        command.extend(
            [
                "--ae_image_replay_out",
                str(out_dir / "ae_image_replay.npz"),
                "--server_port",
                "0",
            ]
        )
    else:
        if variant.server_port is None:
            raise ValueError(f"HTTP variant {variant.name} has no server port")
        command.extend(
            [
                "--server_host",
                "localhost",
                "--server_port",
                str(variant.server_port),
            ]
        )
    if fresh:
        command.append("--no_resume")
    return command


def build_server_command(
    *,
    serve_prefix: Sequence[str],
    root: Path,
    variant: VariantSpec,
    checkpoint: Path,
) -> list[str]:
    if variant.server_port is None:
        raise ValueError(f"{variant.name} is in-process and has no HTTP server")
    feature_mode = "tokens" if variant.is_baseline else "rl_token"
    command = [
        *serve_prefix,
        str(root / "serve.py"),
        "--host",
        "0.0.0.0",
        "--port",
        str(variant.server_port),
        "--device",
        "cuda:0",
        "--dtype",
        "bfloat16",
        "--disable_g",
        "--feature_mode",
        feature_mode,
    ]
    if feature_mode == "rl_token":
        command.extend(["--rlt_ckpt", str(checkpoint)])
    return command


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
    """Return one immutable-snapshot, validation-only evaluation argv."""

    if policy not in variant.policies:
        raise ValueError(f"Policy {policy!r} is not valid for {variant.name}")
    source_dir = snapshot_dir(run_dir, variant, episode)
    out_dir = validation_output_dir(run_dir, variant, episode, policy, seed)
    command = [
        python_executable,
        str(root / "train_rlt_online.py"),
        "--eval_only",
        "--no_resume",
        "--device",
        "cuda:0",
        "--out_dir",
        str(out_dir),
        "--config_name",
        variant.name,
        "--benchmark_dir",
        str(benchmark_val),
        "--rlt_ckpt",
        str(source_dir / "rlt_cf.pt"),
        "--target_env_steps",
        "10000",
        "--max_valid_episodes",
        str(VALIDATION_EPISODES),
        "--start_episode",
        "0",
        "--shard_size",
        str(VALIDATION_EPISODES),
        "--horizon",
        str(HORIZON),
        "--log_every_episodes",
        str(VALIDATION_EPISODES),
        "--ckpt_every_episodes",
        str(VALIDATION_EPISODES),
        "--seed",
        str(seed),
        "--deploy_policy",
        policy,
        "--g_min_advantage",
        str(PAIRED_LCB_THRESHOLD),
        "--g_min_action_sensitivity",
        str(ACTION_SENSITIVITY_THRESHOLD),
        "--explore_residual_std",
        "0",
        "--explore_deploy_std",
        "0",
        "--explore_warmup_mult",
        "1.0",
        "--bc_ref_coef",
        str(BC_REF_COEF),
        "--updates_per_episode",
        "0",
        "--tmp_rollout_dir",
        str(tmp_rollout_dir),
    ]
    command.extend(_common_model_args(variant))
    if variant.ae_trainable:
        command.extend(
            [
                "--ae_trainable_ckpt",
                str(source_dir / "molmo_ae_lora.pt"),
                "--server_port",
                "0",
            ]
        )
    else:
        if variant.server_port is None:
            raise ValueError(f"HTTP variant {variant.name} has no server port")
        command.extend(
            [
                "--server_host",
                "localhost",
                "--server_port",
                str(variant.server_port),
            ]
        )
    return command


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        result.update({"sha": revision, "dirty": bool(status.strip())})
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        result["unavailable_reason"] = str(error)
    return result


def _gpu_inventory() -> dict[str, Any]:
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        rows = []
        for line in output.splitlines():
            values = [value.strip() for value in line.split(",")]
            if len(values) >= 5:
                rows.append(
                    {
                        "index": values[0],
                        "uuid": values[1],
                        "name": values[2],
                        "memory_total_mib": values[3],
                        "driver_version": values[4],
                    }
                )
        return {"available": True, "gpus": rows}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"available": False, "unavailable_reason": str(error), "gpus": []}


def redact_value(key: str, value: Any) -> Any:
    if _SECRET_KEY_RE.search(key):
        return "<redacted>"
    if isinstance(value, str):
        return _URI_CREDENTIAL_RE.sub(r"\1<redacted>@", value)
    if isinstance(value, dict):
        return {
            str(child_key): redact_value(str(child_key), child_value)
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_value(key, child) for child in value]
    return value


def redact_command(command: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for token in command:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        rendered = str(token)
        if rendered.startswith("--") and _SECRET_KEY_RE.search(rendered):
            redacted.append(rendered)
            hide_next = "=" not in rendered
            if "=" in rendered:
                redacted[-1] = rendered.split("=", 1)[0] + "=<redacted>"
            continue
        if "=" in rendered:
            key, value = rendered.split("=", 1)
            if _SECRET_KEY_RE.search(key):
                redacted.append(f"{key}=<redacted>")
                continue
            rendered = f"{key}={redact_value(key, value)}"
        redacted.append(_URI_CREDENTIAL_RE.sub(r"\1<redacted>@", rendered))
    return redacted


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
    """Build the reproducibility manifest without launching any process."""

    root = root.resolve()
    run_dir = run_dir.resolve()
    log_dir = log_dir.resolve()
    benchmark_root = benchmark_root.resolve()
    residual_checkpoint = residual_checkpoint.resolve()
    flow_checkpoint = flow_checkpoint.resolve()
    benchmark_files = {
        "train": benchmark_root / "train" / "benchmark.json",
        "val": benchmark_root / "val" / "benchmark.json",
        "manifest": benchmark_root / "manifest.json",
    }
    required = [
        benchmark_files["train"],
        benchmark_files["val"],
        residual_checkpoint,
        flow_checkpoint,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Manifest inputs are missing: " + ", ".join(missing))
    if len(gpu_ids) != len(VARIANTS):
        raise ValueError(f"Expected eight GPU IDs, got {len(gpu_ids)}")

    environment = dict(os.environ if environment is None else environment)
    relevant_keys = (
        "B1K_ROOT",
        "B1K_TMP",
        "HF_HOME",
        "MLSPACES_ASSETS_DIR",
        "RLT_EGL_MAX_CONCURRENT",
        "RLT_EGL_PER_GPU",
        "RLT_EGL_COOLDOWN_SEC",
        "RLT_IO_RETRY_ATTEMPTS",
        "RLT_IO_RETRY_BASE_SEC",
        "CUDA_VISIBLE_DEVICES",
        "GPU_IDS",
        "FRESH",
    )
    env_snapshot = {
        key: redact_value(key, environment[key])
        for key in relevant_keys
        if key in environment
    }
    repositories = {
        "rql": root.parents[1],
        "molmoact2": root.parents[2] / "molmoact2",
        "molmospaces": root.parents[2] / "molmospaces",
        "b1k_airi": root.parents[3],
    }
    variant_records = []
    server_records = []
    for index, variant in enumerate(VARIANTS):
        gpu_id = str(gpu_ids[index])
        checkpoint = variant_checkpoint(
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
                "initial_fresh_command": redact_command(fresh_command),
                "resume_command": redact_command(resume_command),
                "snapshot_paths": [
                    str(snapshot_dir(run_dir, variant, episode))
                    for episode in SNAPSHOT_EPISODES
                ],
            }
        )
        variant_records.append(record)
        if variant.server_port is not None:
            server_records.append(
                {
                    "variant": variant.name,
                    "physical_gpu": gpu_id,
                    "port": variant.server_port,
                    "command": redact_command(
                        build_server_command(
                            serve_prefix=serve_prefix,
                            root=root,
                            variant=variant,
                            checkpoint=checkpoint,
                        )
                    ),
                }
            )

    benchmark_hashes = {
        label: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for label, path in benchmark_files.items()
        if path.is_file()
    }
    return {
        "schema_version": "v13-controlled-1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "gpu_inventory": _gpu_inventory(),
        "repositories": {
            name: _git_state(path) for name, path in repositories.items()
        },
        "run": {
            "name": RUN_NAME,
            "run_dir": str(run_dir),
            "log_dir": str(log_dir),
            "tmp_rollout_dir": str(tmp_rollout_dir),
            "egl_lock_dir": str(egl_lock_dir),
            "fresh_semantics": (
                "FRESH=1 clears only the eight V13 training output directories "
                "after live-PID checks and adds --no_resume only to the initial starts; "
                "watchdog restarts preserve artifacts and omit --no_resume."
            ),
        },
        "benchmark": {
            "name": BENCHMARK_NAME,
            "root": str(benchmark_root),
            "train_indices": [0, TRAIN_EPISODES - 1],
            "validation_indices": [0, VALIDATION_EPISODES - 1],
            "files": benchmark_hashes,
        },
        "checkpoints": {
            "residual": {
                "path": str(residual_checkpoint),
                "sha256": sha256_file(residual_checkpoint),
            },
            "flow": {
                "path": str(flow_checkpoint),
                "sha256": sha256_file(flow_checkpoint),
            },
        },
        "fixed_parameters": {
            "max_valid_episodes": MAX_VALID_EPISODES,
            "horizon": HORIZON,
            "target_env_steps": TARGET_ENV_STEPS,
            "snapshot_episodes": list(SNAPSHOT_EPISODES),
            "train_seed": TRAIN_SEED,
            "interim_validation_seeds": list(INTERIM_VALIDATION_SEEDS),
            "final_validation_seeds": list(FINAL_VALIDATION_SEEDS),
            "deploy_policy": "gated",
            "paired_lcb_threshold": PAIRED_LCB_THRESHOLD,
            "action_sensitivity_threshold": ACTION_SENSITIVITY_THRESHOLD,
            "extra_guide_advantage_threshold": None,
            "exploration_std": EXPLORE_STD,
            "bc_ref_coef": BC_REF_COEF,
            "frozen_token": True,
            "ae_instances_per_gpu": 1,
            "ae_image_replay_capacity": AE_IMAGE_REPLAY_CAPACITY,
        },
        "environment": env_snapshot,
        "servers": server_records,
        "variants": variant_records,
    }


def latest_jsonl_row(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    latest: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                latest = candidate
    return latest


def resume_state(out_dir: Path, ae_trainable: bool) -> str:
    """Return empty, complete, or partial for a trainer's safe-resume bundle."""

    required = [
        out_dir / "rlt_cf_latest.pt",
        out_dir / "chunk_replay.npz",
        out_dir / "metrics.jsonl",
    ]
    markers = [
        *required,
        out_dir / "LATEST_CKPT.txt",
        out_dir / "summary.json",
        out_dir / "snapshots",
    ]
    if ae_trainable:
        required.extend(
            [
                out_dir / "molmo_ae_lora_latest.pt",
                out_dir / "ae_image_replay.npz",
            ]
        )
        markers.extend(required[-2:])
    present_required = [path.exists() for path in required]
    any_marker = any(path.exists() for path in markers)
    if all(present_required):
        return "complete"
    if not any_marker:
        return "empty"
    return "partial"


def evaluation_complete(out_dir: Path, expected_episodes: int = VALIDATION_EPISODES) -> bool:
    summary_path = out_dir / "validation_summary.json"
    results_path = out_dir / "validation_results.jsonl"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        bool(summary.get("eval_only"))
        and int(summary.get("valid_episodes", 0) or 0) == expected_episodes
        and results_path.is_file()
    )


def training_complete(out_dir: Path, expected_episodes: int = MAX_VALID_EPISODES) -> bool:
    row = latest_jsonl_row(out_dir / "metrics.jsonl")
    if row is None:
        try:
            row = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
    return int(row.get("valid_episodes", 0) or 0) >= expected_episodes


def _variants_tsv() -> str:
    lines = []
    for variant in VARIANTS:
        lines.append(
            "|".join(
                [
                    variant.name,
                    str(variant.gpu),
                    variant.cf_mode,
                    variant.actor_mode,
                    "1" if variant.use_guide else "0",
                    "1" if variant.ae_trainable else "0",
                    variant.checkpoint_kind,
                    str(variant.updates_per_episode),
                    "" if variant.server_port is None else str(variant.server_port),
                ]
            )
        )
    return "\n".join(lines)


def _add_command_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--format", choices=["nul", "shell", "json"], default="nul")


def _emit_command(command: Sequence[str], output_format: str) -> None:
    if output_format == "nul":
        sys.stdout.buffer.write(b"\0".join(token.encode("utf-8") for token in command))
        sys.stdout.buffer.write(b"\0")
    elif output_format == "shell":
        print(shlex.join(command))
    elif output_format == "json":
        print(json.dumps(list(command)))
    else:
        raise ValueError(f"Unsupported command output format: {output_format}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    variants_parser = subparsers.add_parser("variants")
    variants_parser.add_argument("--format", choices=["json", "tsv"], default="tsv")

    train_command_parser = subparsers.add_parser("train-command")
    train_command_parser.add_argument("--variant", choices=sorted(VARIANT_BY_NAME), required=True)
    train_command_parser.add_argument("--root", type=Path, required=True)
    train_command_parser.add_argument("--run-dir", type=Path, required=True)
    train_command_parser.add_argument("--benchmark-train", type=Path, required=True)
    train_command_parser.add_argument("--residual-checkpoint", type=Path, required=True)
    train_command_parser.add_argument("--flow-checkpoint", type=Path, required=True)
    train_command_parser.add_argument("--python-executable", required=True)
    train_command_parser.add_argument("--tmp-rollout-dir", type=Path, required=True)
    train_command_parser.add_argument("--fresh", action="store_true")
    _add_command_output_args(train_command_parser)

    server_command_parser = subparsers.add_parser("server-command")
    server_command_parser.add_argument("--variant", choices=sorted(VARIANT_BY_NAME), required=True)
    server_command_parser.add_argument("--root", type=Path, required=True)
    server_command_parser.add_argument("--checkpoint", type=Path, required=True)
    server_command_parser.add_argument("--serve-prefix", action="append", default=[])
    _add_command_output_args(server_command_parser)

    eval_command_parser = subparsers.add_parser("eval-command")
    eval_command_parser.add_argument("--variant", choices=sorted(VARIANT_BY_NAME), required=True)
    eval_command_parser.add_argument("--root", type=Path, required=True)
    eval_command_parser.add_argument("--run-dir", type=Path, required=True)
    eval_command_parser.add_argument("--benchmark-val", type=Path, required=True)
    eval_command_parser.add_argument("--tmp-rollout-dir", type=Path, required=True)
    eval_command_parser.add_argument("--python-executable", required=True)
    eval_command_parser.add_argument("--episode", type=int, choices=SNAPSHOT_EPISODES, required=True)
    eval_command_parser.add_argument(
        "--policy",
        choices=["reference", "checkpoint_gate", "actor", "actor_guide"],
        required=True,
    )
    eval_command_parser.add_argument("--seed", type=int, required=True)
    _add_command_output_args(eval_command_parser)

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

    gpu_parser = subparsers.add_parser("manifest-gpu-ids")
    gpu_parser.add_argument("--manifest", type=Path, required=True)

    resume_parser = subparsers.add_parser("resume-state")
    resume_parser.add_argument("--out-dir", type=Path, required=True)
    resume_parser.add_argument("--ae", action="store_true")

    training_parser = subparsers.add_parser("training-complete")
    training_parser.add_argument("--out-dir", type=Path, required=True)
    training_parser.add_argument("--expected-episodes", type=int, default=MAX_VALID_EPISODES)

    eval_parser = subparsers.add_parser("evaluation-complete")
    eval_parser.add_argument("--out-dir", type=Path, required=True)
    eval_parser.add_argument("--expected-episodes", type=int, default=VALIDATION_EPISODES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "variants":
        if args.format == "json":
            print(json.dumps([dataclasses.asdict(variant) for variant in VARIANTS], indent=2))
        else:
            print(_variants_tsv())
        return 0
    if args.command == "train-command":
        command = build_train_command(
            python_executable=args.python_executable,
            root=args.root.resolve(),
            run_dir=args.run_dir.resolve(),
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
            run_dir=args.run_dir.resolve(),
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
        serve_prefix: Iterable[str] = args.serve_prefix
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
            serve_prefix=list(serve_prefix),
            tmp_rollout_dir=args.tmp_rollout_dir,
            egl_lock_dir=args.egl_lock_dir,
            gpu_ids=[item.strip() for item in args.gpu_ids.split(",") if item.strip()],
        )
        atomic_write_json(args.output, payload)
        print(json.dumps({"manifest": str(args.output), "variants": len(VARIANTS)}))
        return 0
    if args.command == "manifest-gpu-ids":
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        records = payload.get("variants", [])
        by_name = {
            str(record.get("name")): str(record.get("physical_gpu"))
            for record in records
            if isinstance(record, dict)
        }
        gpu_ids = [by_name.get(variant.name, "") for variant in VARIANTS]
        if any(not gpu_id for gpu_id in gpu_ids):
            raise ValueError("Manifest is missing one or more V13 physical GPU assignments")
        print(",".join(gpu_ids))
        return 0
    if args.command == "resume-state":
        state = resume_state(args.out_dir, args.ae)
        print(state)
        return 2 if state == "partial" else 0
    if args.command == "training-complete":
        complete = training_complete(args.out_dir, args.expected_episodes)
        print("complete" if complete else "incomplete")
        return 0 if complete else 1
    if args.command == "evaluation-complete":
        complete = evaluation_complete(args.out_dir, args.expected_episodes)
        print("complete" if complete else "incomplete")
        return 0 if complete else 1
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
