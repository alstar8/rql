"""Controlled V17 harness: kettle-matched offline + safe collect + soft Q ramp.

Extends V16 improved (z=256/d=512/L4) with:
- episode-level actor mixture (no always-collect by default)
- stronger soft-β BC
- longer BC warmup + linear q_coef ramp
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import v13_harness as v13
import v16_harness as v16
from v13_harness import VariantSpec


RUN_NAME = "rlt_cf_v17_kettle"
BENCHMARK_NAME = v13.BENCHMARK_NAME
TRAIN_EPISODES = v13.TRAIN_EPISODES
VALIDATION_EPISODES = v13.VALIDATION_EPISODES
HORIZON = v13.HORIZON
TRAIN_SEED = v13.TRAIN_SEED
N_CRITICS = v13.N_CRITICS
FLOW_STEPS = v13.FLOW_STEPS
GUIDANCE_COEF = v13.GUIDANCE_COEF

MAX_VALID_EPISODES = int(os.environ.get("V17_MAX_VALID_EPISODES", "1000"))
TARGET_ENV_STEPS = int(os.environ.get("V17_TARGET_ENV_STEPS", "600000"))
SNAPSHOT_EPISODES = tuple(
    int(token.strip())
    for token in os.environ.get(
        "V17_SNAPSHOT_EPISODES",
        "0,100,200,400,700,1000",
    ).split(",")
    if token.strip()
)
MAX_UPDATE_SEC_PER_EPISODE = float(
    os.environ.get("V17_MAX_UPDATE_SEC_PER_EPISODE", "60")
)
UPDATES_PER_EPISODE = int(os.environ.get("V17_UPDATES_PER_EPISODE", "128"))
BENCHMARK_POSE_CYCLE = TRAIN_EPISODES

# V17 collect / loss defaults (fix V16 always-collect SR collapse).
ACTOR_MIXTURE_PROB = float(os.environ.get("V17_ACTOR_MIXTURE_PROB", "0.15"))
ACTOR_BC_EPISODES = int(os.environ.get("V17_ACTOR_BC_EPISODES", "150"))
Q_RAMP_EPISODES = int(os.environ.get("V17_Q_RAMP_EPISODES", "100"))
ACTOR_BETA = float(os.environ.get("V17_ACTOR_BETA", "5.0"))
# Delay paper always-collect until empirical gate can fire (effectively off).
ALWAYS_COLLECT_AFTER = int(os.environ.get("V17_ALWAYS_COLLECT_AFTER", "100000"))
ENABLE_ALWAYS_COLLECT = os.environ.get("V17_ALWAYS_COLLECT", "0") == "1"
RESIDUAL_CLIP = float(os.environ.get("V17_RESIDUAL_CLIP", "0.02"))
ADVANTAGE_CLIP = float(os.environ.get("V17_ADVANTAGE_CLIP", "0.05"))
ENDPOINT_REF_MSE_MAX = float(os.environ.get("V17_ENDPOINT_REF_MSE_MAX", "0.01"))
ACTOR_CQL_COEF = float(os.environ.get("V17_ACTOR_CQL_COEF", "0.1"))
EMPIRICAL_MIN_EPISODES = int(os.environ.get("V17_EMPIRICAL_MIN_EPISODES", "16"))
G_MIN_EMPIRICAL = float(os.environ.get("V17_G_MIN_EMPIRICAL", "0.0"))
TRAIN_REF_DROPOUT = float(os.environ.get("V17_TRAIN_REF_DROPOUT", "0.5"))
CRITIC_REF_DROPOUT = float(os.environ.get("V17_CRITIC_REF_DROPOUT", "0.5"))
EXPLORE_RESIDUAL_STD = float(os.environ.get("V17_EXPLORE_RESIDUAL_STD", "0.0"))

HTTP_PORTS = tuple(range(8730, 8738))

VARIANTS = (
    VariantSpec(
        "residual_rlt_cf",
        0,
        "residual",
        "rlt",
        True,
        False,
        "residual",
        UPDATES_PER_EPISODE,
        8730,
        "ChunkGaussianActor",
        "EnsembleCQL",
        "CFGradientGuide",
    ),
    VariantSpec(
        "flow_rlt_cf",
        1,
        "flow",
        "rlt",
        True,
        False,
        "flow",
        UPDATES_PER_EPISODE,
        8734,
        "FlowVelocityActor",
        "EnsembleTimeCQL",
        "FlowCFGuide",
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}

REQUIRED_V17_TRAINER_OPTIONS = tuple(
    dict.fromkeys(
        (
            *v16.REQUIRED_V16_TRAINER_OPTIONS,
            "--q_ramp_episodes",
            "--always_collect_after_episodes",
        )
    )
)


def _assert_v17_run_dir(run_dir: Path) -> Path:
    resolved = run_dir.resolve()
    if resolved.name != RUN_NAME:
        raise ValueError(
            f"V17 run directory basename must be {RUN_NAME!r}, got {resolved}"
        )
    for forbidden in (
        "rlt_cf_v13_controlled",
        "rlt_cf_v14_controlled",
        "rlt_cf_v15_controlled",
        "rlt_cf_v16_controlled",
        "rlt_cf_v16_rlt_improved",
    ):
        if forbidden in resolved.parts:
            raise ValueError(f"V17 refuses prior run path: {resolved}")
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


def _set_explore_std(command: list[str], value: float) -> None:
    if "--explore_residual_std" in command:
        _set_argument(command, "--explore_residual_std", value)
    else:
        _append_option(command, "--explore_residual_std", value)
    if "--explore_deploy_std" in command:
        _set_argument(command, "--explore_deploy_std", value)
    else:
        _append_option(command, "--explore_deploy_std", value)


def training_output_dir(run_dir: Path, variant: VariantSpec) -> Path:
    return _assert_v17_run_dir(run_dir) / variant.name


def _apply_v17_flags(command: list[str], variant: VariantSpec) -> list[str]:
    _set_argument(command, "--max_valid_episodes", MAX_VALID_EPISODES)
    _set_argument(command, "--target_env_steps", TARGET_ENV_STEPS)
    _set_argument(
        command,
        "--snapshot_episodes",
        ",".join(str(episode) for episode in SNAPSHOT_EPISODES),
    )
    _set_argument(command, "--updates_per_episode", variant.updates_per_episode)
    _set_argument(command, "--ref_dropout", CRITIC_REF_DROPOUT)
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

    if variant.actor_mode == "rlt":
        _append_option(command, "--actor_mixture_prob", ACTOR_MIXTURE_PROB)
        if ENABLE_ALWAYS_COLLECT:
            if "--always_collect_actor" not in command:
                command.append("--always_collect_actor")
            _append_option(command, "--always_collect_after_episodes", ALWAYS_COLLECT_AFTER)
        else:
            # Explicitly avoid V16 always-collect.
            while "--always_collect_actor" in command:
                command.remove("--always_collect_actor")
        if "--require_empirical_gate" not in command:
            command.append("--require_empirical_gate")
        _append_option(command, "--g_min_empirical_advantage", G_MIN_EMPIRICAL)
        _append_option(command, "--empirical_min_episodes", EMPIRICAL_MIN_EPISODES)
        _set_explore_std(command, EXPLORE_RESIDUAL_STD)
        _append_option(command, "--actor_bc_episodes", ACTOR_BC_EPISODES)
        _append_option(command, "--q_ramp_episodes", Q_RAMP_EPISODES)
        _append_option(command, "--residual_clip", RESIDUAL_CLIP)
        _append_option(command, "--advantage_clip", ADVANTAGE_CLIP)
        _append_option(command, "--endpoint_ref_mse_max", ENDPOINT_REF_MSE_MAX)
        _append_option(command, "--actor_cql_coef", ACTOR_CQL_COEF)
        if "--train_ref_dropout" in command:
            _set_argument(command, "--train_ref_dropout", TRAIN_REF_DROPOUT)
        else:
            _append_option(command, "--train_ref_dropout", TRAIN_REF_DROPOUT)
        if "--actor_beta" in command:
            _set_argument(command, "--actor_beta", ACTOR_BETA)
        else:
            _append_option(command, "--actor_beta", ACTOR_BETA)
    return command


def build_train_command(
    *,
    variant: VariantSpec | str,
    root: Path,
    run_dir: Path,
    benchmark_train: Path,
    residual_checkpoint: Path,
    flow_checkpoint: Path,
    python_executable: str,
    tmp_rollout_dir: Path,
    fresh: bool = False,
) -> list[str]:
    if isinstance(variant, str):
        variant = VARIANT_BY_NAME[variant]
    run_dir = _assert_v17_run_dir(run_dir)
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
    _set_argument(command, "--out_dir", str(run_dir / variant.name))
    return _apply_v17_flags(command, variant)


def build_server_command(
    *,
    variant: VariantSpec | str,
    root: Path,
    checkpoint: Path,
    serve_prefix: Sequence[str],
) -> list[str]:
    if isinstance(variant, str):
        variant = VARIANT_BY_NAME[variant]
    return v13.build_server_command(
        serve_prefix=serve_prefix,
        root=root,
        variant=variant,
        checkpoint=checkpoint,
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train-command")
    train.add_argument("--variant", required=True)
    train.add_argument("--root", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--benchmark-train", type=Path, required=True)
    train.add_argument("--residual-checkpoint", type=Path, required=True)
    train.add_argument("--flow-checkpoint", type=Path, required=True)
    train.add_argument("--python-executable", required=True)
    train.add_argument("--tmp-rollout-dir", type=Path, required=True)
    train.add_argument("--fresh", action="store_true")
    train.add_argument("--format", choices=("shell", "nul"), default="shell")

    server = sub.add_parser("server-command")
    server.add_argument("--variant", required=True)
    server.add_argument("--root", type=Path, required=True)
    server.add_argument("--checkpoint", type=Path, required=True)
    server.add_argument("--format", choices=("shell", "nul"), default="shell")
    server.add_argument("--serve-prefix", nargs="+", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "train-command":
        command = build_train_command(
            variant=args.variant,
            root=args.root,
            run_dir=args.run_dir,
            benchmark_train=args.benchmark_train,
            residual_checkpoint=args.residual_checkpoint,
            flow_checkpoint=args.flow_checkpoint,
            python_executable=args.python_executable,
            tmp_rollout_dir=args.tmp_rollout_dir,
            fresh=args.fresh,
        )
    else:
        command = build_server_command(
            variant=args.variant,
            root=args.root,
            checkpoint=args.checkpoint,
            serve_prefix=args.serve_prefix,
        )
    if args.format == "nul":
        sys.stdout.buffer.write(b"\0".join(c.encode() for c in command) + b"\0")
    else:
        print(" ".join(command))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
