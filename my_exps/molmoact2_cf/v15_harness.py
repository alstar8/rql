"""Controlled V15 harness: mixture collection, conservative actor, empirical gate.

Delegates launch/eval CLI machinery to v14_harness after installing V15
constants and build helpers.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Sequence

import v13_harness as v13
import v14_harness as v14
from v13_harness import VariantSpec


RUN_NAME = "rlt_cf_v15_controlled"
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

MAX_VALID_EPISODES = int(os.environ.get("V15_MAX_VALID_EPISODES", "400"))
TARGET_ENV_STEPS = int(os.environ.get("V15_TARGET_ENV_STEPS", "250000"))
SNAPSHOT_EPISODES = tuple(
    int(token.strip())
    for token in os.environ.get(
        "V15_SNAPSHOT_EPISODES",
        "0,100,200,400",
    ).split(",")
    if token.strip()
)
AE_IMAGE_REPLAY_CAPACITY = v14.AE_IMAGE_REPLAY_CAPACITY
AE_BATCH_SIZE = int(os.environ.get("V15_AE_BATCH_SIZE", str(v14.AE_BATCH_SIZE)))
AE_MICROBATCH_SIZE = int(
    os.environ.get("V15_AE_MICROBATCH_SIZE", str(v14.AE_MICROBATCH_SIZE))
)
AE_MIN_SUCCESS_EPISODES = int(
    os.environ.get("V15_AE_MIN_SUCCESS_EPISODES", str(v14.AE_MIN_SUCCESS_EPISODES))
)
BENCHMARK_POSE_CYCLE = TRAIN_EPISODES
MAX_UPDATE_SEC_PER_EPISODE = float(
    os.environ.get("V15_MAX_UPDATE_SEC_PER_EPISODE", "30")
)
RUN_MODE = os.environ.get("V15_MODE", "full")

ACTOR_MIXTURE_PROB = float(os.environ.get("V15_ACTOR_MIXTURE_PROB", "0.25"))
ACTOR_BC_EPISODES = int(os.environ.get("V15_ACTOR_BC_EPISODES", "50"))
RESIDUAL_CLIP = float(os.environ.get("V15_RESIDUAL_CLIP", "0.02"))
ADVANTAGE_CLIP = float(os.environ.get("V15_ADVANTAGE_CLIP", "0.05"))
ENDPOINT_REF_MSE_MAX = float(os.environ.get("V15_ENDPOINT_REF_MSE_MAX", "0.01"))
ACTOR_CQL_COEF = float(os.environ.get("V15_ACTOR_CQL_COEF", "0.1"))
EMPIRICAL_MIN_EPISODES = int(os.environ.get("V15_EMPIRICAL_MIN_EPISODES", "16"))
G_MIN_EMPIRICAL = float(os.environ.get("V15_G_MIN_EMPIRICAL", "0.0"))

HTTP_PORTS = tuple(range(8700, 8707))

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
    # Guide-on-frozen-VLA isolation arm (no learned actor deploy).
    VariantSpec(
        "residual_vla_cf",
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
        "residual_rlt_cf",
        3,
        "residual",
        "rlt",
        True,
        False,
        "residual",
        8,
        8703,
        "ChunkGaussianActor",
        "EnsembleCQL",
        "CFGradientGuide",
    ),
    VariantSpec(
        "flow_vla_baseline",
        4,
        "flow",
        "vla_only",
        False,
        False,
        "flow",
        0,
        8704,
        "FlowVelocityActor",
        "EnsembleTimeCQL",
        None,
    ),
    VariantSpec(
        "flow_rlt_actor",
        5,
        "flow",
        "rlt",
        False,
        False,
        "flow",
        8,
        8705,
        "FlowVelocityActor",
        "EnsembleTimeCQL",
        None,
    ),
    VariantSpec(
        "flow_rlt_cf",
        6,
        "flow",
        "rlt",
        True,
        False,
        "flow",
        8,
        8706,
        "FlowVelocityActor",
        "EnsembleTimeCQL",
        "FlowCFGuide",
    ),
    # Provisional AE arm: V14 held-out canary incomplete at V15 authoring time.
    VariantSpec(
        "molmo_ae_lora_actor",
        7,
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
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}

REQUIRED_V15_TRAINER_OPTIONS = tuple(
    dict.fromkeys(
        (
            *v14.REQUIRED_V14_TRAINER_OPTIONS,
            "--actor_mixture_prob",
            "--require_empirical_gate",
            "--guide_on_reference",
            "--no_guide_on_reference",
            "--actor_bc_episodes",
            "--residual_clip",
            "--advantage_clip",
            "--endpoint_ref_mse_max",
            "--actor_cql_coef",
            "--g_min_empirical_advantage",
            "--empirical_min_episodes",
            "--train_ref_dropout",
        )
    )
)


def _assert_v15_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    if resolved.name != RUN_NAME:
        raise ValueError(
            f"V15 run directory basename must be {RUN_NAME!r}, got {resolved}"
        )
    for forbidden in ("rlt_cf_v13_controlled", "rlt_cf_v14_controlled"):
        if forbidden in resolved.parts:
            raise ValueError(f"V15 refuses prior controlled run path: {resolved}")
    return resolved


def _append_option(command: list[str], flag: str, value: object | None = None) -> None:
    if flag in command:
        return
    command.append(flag)
    if value is not None:
        command.append(str(value))


def _set_argument(command: list[str], flag: str, value: object) -> None:
    index = command.index(flag)
    command[index + 1] = str(value)


def training_output_dir(run_dir: Path, variant: VariantSpec) -> Path:
    return _assert_v15_run_dir(run_dir) / variant.name


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
        _assert_v15_run_dir(run_dir)
        / "validation"
        / variant.name
        / f"ep_{episode:06d}"
        / policy
        / f"seed_{seed}"
    )


def evaluation_policies(variant: VariantSpec) -> tuple[str, ...]:
    if variant.name == "residual_vla_cf":
        return ("reference", "actor_guide")
    if variant.is_baseline:
        return ("reference",)
    if variant.name in {"residual_rlt_actor", "flow_rlt_actor"}:
        return ("reference", "reference_noise", "actor")
    if variant.use_guide:
        return ("reference", "actor", "actor_guide")
    return ("reference", "actor")


def _apply_v15_flags(command: list[str], variant: VariantSpec) -> list[str]:
    _set_argument(command, "--max_valid_episodes", MAX_VALID_EPISODES)
    _set_argument(command, "--target_env_steps", TARGET_ENV_STEPS)
    _set_argument(
        command,
        "--snapshot_episodes",
        ",".join(str(episode) for episode in SNAPSHOT_EPISODES),
    )
    _set_argument(command, "--updates_per_episode", variant.updates_per_episode)
    _set_argument(command, "--ref_dropout", "0.0")
    if "--max_updates_per_episode" in command:
        _set_argument(command, "--max_updates_per_episode", variant.updates_per_episode)
    else:
        _append_option(command, "--max_updates_per_episode", variant.updates_per_episode)
    if "--max_update_sec_per_episode" in command:
        _set_argument(
            command,
            "--max_update_sec_per_episode",
            MAX_UPDATE_SEC_PER_EPISODE,
        )
    else:
        _append_option(
            command,
            "--max_update_sec_per_episode",
            MAX_UPDATE_SEC_PER_EPISODE,
        )
    _append_option(command, "--no_critic_target_use_guide")
    _append_option(command, "--benchmark_pose_cycle", BENCHMARK_POSE_CYCLE)

    if variant.actor_mode == "rlt" and not variant.ae_trainable:
        if variant.name == "residual_vla_cf":
            command.append("--guide_on_reference")
            _append_option(command, "--actor_mixture_prob", 0.0)
        else:
            _append_option(command, "--actor_mixture_prob", ACTOR_MIXTURE_PROB)
            if "--require_empirical_gate" not in command:
                command.append("--require_empirical_gate")
            _append_option(command, "--g_min_empirical_advantage", G_MIN_EMPIRICAL)
            _append_option(command, "--empirical_min_episodes", EMPIRICAL_MIN_EPISODES)
        _append_option(command, "--actor_bc_episodes", ACTOR_BC_EPISODES)
        _append_option(command, "--residual_clip", RESIDUAL_CLIP)
        _append_option(command, "--advantage_clip", ADVANTAGE_CLIP)
        _append_option(command, "--endpoint_ref_mse_max", ENDPOINT_REF_MSE_MAX)
        _append_option(command, "--actor_cql_coef", ACTOR_CQL_COEF)
        _append_option(command, "--train_ref_dropout", 0.0)

    if variant.ae_trainable:
        _set_argument(command, "--ae_batch_size", AE_BATCH_SIZE)
        if "--ae_image_replay_capacity" in command:
            _set_argument(
                command,
                "--ae_image_replay_capacity",
                AE_IMAGE_REPLAY_CAPACITY,
            )
        else:
            _append_option(
                command,
                "--ae_image_replay_capacity",
                AE_IMAGE_REPLAY_CAPACITY,
            )
        _append_option(command, "--ae_microbatch_size", AE_MICROBATCH_SIZE)
        _append_option(command, "--ae_min_success_episodes", AE_MIN_SUCCESS_EPISODES)
    return command


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
    run_dir = _assert_v15_run_dir(run_dir)
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
    return _apply_v15_flags(command, variant)


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
    if policy not in evaluation_policies(variant):
        raise ValueError(f"Policy {policy!r} is not valid for {variant.name}")
    run_dir = _assert_v15_run_dir(run_dir)
    # Use a policy accepted by VariantSpec.policies for the v13 builder.
    if policy == "reference_noise":
        builder_policy = "reference_noise"
    elif policy == "reference" and not variant.is_baseline:
        builder_policy = "checkpoint_gate"
    else:
        builder_policy = policy
    # residual_vla_cf: VariantSpec.policies includes actor_guide via use_guide.
    if variant.name == "residual_vla_cf" and policy == "reference":
        builder_policy = "checkpoint_gate"
    command = v13.build_eval_command(
        python_executable=python_executable,
        root=root,
        run_dir=run_dir,
        benchmark_val=benchmark_val,
        tmp_rollout_dir=tmp_rollout_dir,
        variant=variant,
        episode=episode,
        policy=builder_policy,
        seed=seed,
    )
    _set_argument(
        command,
        "--out_dir",
        validation_output_dir(run_dir, variant, episode, policy, seed),
    )
    runtime = "reference" if policy in {"reference", "reference_noise"} else policy
    if policy == "reference" and not variant.is_baseline:
        runtime = "checkpoint_gate" if variant.name != "residual_vla_cf" else "reference"
    _set_argument(command, "--deploy_policy", runtime)
    _set_argument(command, "--max_valid_episodes", VALIDATION_EPISODES)
    _set_argument(command, "--updates_per_episode", 0)
    if "--max_updates_per_episode" in command:
        _set_argument(command, "--max_updates_per_episode", 0)
    else:
        _append_option(command, "--max_updates_per_episode", 0)
    _append_option(command, "--no_critic_target_use_guide")
    if variant.name == "residual_vla_cf":
        if "--guide_on_reference" not in command:
            command.append("--guide_on_reference")
    if policy == "reference_noise":
        if "--eval_reference_noise_std" in command:
            _set_argument(command, "--eval_reference_noise_std", EXPLORE_STD)
        else:
            _append_option(command, "--eval_reference_noise_std", EXPLORE_STD)
    if variant.ae_trainable:
        _set_argument(command, "--ae_batch_size", AE_BATCH_SIZE)
        _append_option(command, "--ae_microbatch_size", AE_MICROBATCH_SIZE)
        if "--ae_image_replay_capacity" in command:
            _set_argument(
                command,
                "--ae_image_replay_capacity",
                AE_IMAGE_REPLAY_CAPACITY,
            )
        _append_option(command, "--ae_min_success_episodes", AE_MIN_SUCCESS_EPISODES)
    return command


def validate_trainer_cli(train_script: Path) -> dict[str, Any]:
    source = train_script.read_text(encoding="utf-8")
    missing = [
        option
        for option in REQUIRED_V15_TRAINER_OPTIONS
        if option not in source
    ]
    return {
        "valid": not missing,
        "train_script": str(train_script),
        "required_options": list(REQUIRED_V15_TRAINER_OPTIONS),
        "missing_options": missing,
    }


def _install_into_v14() -> None:
    """Point v14 CLI helpers at V15 constants/builders."""
    v14.RUN_NAME = RUN_NAME
    v14.VARIANTS = VARIANTS
    v14.VARIANT_BY_NAME = VARIANT_BY_NAME
    v14.MAX_VALID_EPISODES = MAX_VALID_EPISODES
    v14.TARGET_ENV_STEPS = TARGET_ENV_STEPS
    v14.SNAPSHOT_EPISODES = SNAPSHOT_EPISODES
    v14.AE_BATCH_SIZE = AE_BATCH_SIZE
    v14.AE_MICROBATCH_SIZE = AE_MICROBATCH_SIZE
    v14.AE_MIN_SUCCESS_EPISODES = AE_MIN_SUCCESS_EPISODES
    v14.AE_IMAGE_REPLAY_CAPACITY = AE_IMAGE_REPLAY_CAPACITY
    v14.BENCHMARK_POSE_CYCLE = BENCHMARK_POSE_CYCLE
    v14.MAX_UPDATE_SEC_PER_EPISODE = MAX_UPDATE_SEC_PER_EPISODE
    v14.RUN_MODE = RUN_MODE
    v14.HTTP_PORTS = HTTP_PORTS
    v14.REQUIRED_V14_TRAINER_OPTIONS = REQUIRED_V15_TRAINER_OPTIONS
    v14._assert_v14_run_dir = _assert_v15_run_dir  # type: ignore[attr-defined]
    v14.training_output_dir = training_output_dir
    v14.snapshot_dir = snapshot_dir
    v14.validation_output_dir = validation_output_dir
    v14.evaluation_policies = evaluation_policies
    v14.build_train_command = build_train_command
    v14.build_server_command = build_server_command
    v14.build_eval_command = build_eval_command
    v14.validate_trainer_cli = validate_trainer_cli

    # v14.variants TSV delegates to v13._variants_tsv which reads v13.VARIANTS.
    v13.VARIANTS = VARIANTS
    v13.VARIANT_BY_NAME = VARIANT_BY_NAME
    v13.HTTP_PORTS = HTTP_PORTS

    def _variants_tsv() -> str:
        rows = []
        for variant in VARIANTS:
            rows.append(
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
        return "\n".join(rows)

    v14._variants_tsv = _variants_tsv  # type: ignore[attr-defined]
    v13._variants_tsv = _variants_tsv  # type: ignore[attr-defined]

    original_pid = v14._pid_belongs_to_run

    def _pid_belongs_to_run(pid: int, run_dir: Path) -> bool:
        environ = Path(f"/proc/{pid}/environ")
        try:
            entries = environ.read_bytes().split(b"\0")
        except OSError:
            return False
        marker = f"RLT_CF_V15_RUN_DIR={run_dir.resolve()}".encode("utf-8")
        if marker in entries:
            return True
        return original_pid(pid, run_dir)

    v14._pid_belongs_to_run = _pid_belongs_to_run  # type: ignore[attr-defined]


def main(argv: Sequence[str] | None = None) -> int:
    _install_into_v14()
    return int(v14.main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
