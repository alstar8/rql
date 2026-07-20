#!/usr/bin/env python3
"""Checkpoint diagnostics: teacher-forced vs free-running AR for DiscreteARIQL.

Loads a DiscreteARIQL checkpoint (and optionally an OGBench dataset batch),
then reports teacher-forced vs free-running token/register/prefix/sequence
exact rates and decoded-action RMSE/correlation.

Does **not** run during training updates (avoids K transformer calls per step).

Examples:
  # From a finished run directory (reads flags.json + latest/params_*.pkl):
  python scripts/diagnose_discrete_ar_iql.py \\
    --run-dir exp/rql/humanoidmaze-large-dari-v6-2m/sd000_... \\
    --batch-size 64

  # Explicit checkpoint + tokenizer + env:
  python scripts/diagnose_discrete_ar_iql.py \\
    --checkpoint exp/rql/.../params_100000.pkl \\
    --tokenizer-path exp/ogbench_oattok/humanoidmaze-large_h1_d21.pkl \\
    --env-name humanoidmaze-large-navigate-singletask-v0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.discrete_ar_iql import DiscreteARIQLAgent, get_config
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent


def _latest_params_pkl(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("params_*.pkl"))
    if not cands:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    return cands[-1]


def _load_flags(run_dir: Path) -> dict:
    path = run_dir / "flags.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing flags.json in {run_dir}")
    return json.loads(path.read_text())


def _config_from_flags(flags: dict) -> dict:
    cfg = dict(get_config())
    agent_flags = flags.get("agent") or {}
    for key, value in agent_flags.items():
        if key in cfg or key in (
            "tokenizer_path",
            "actor_emb_dim",
            "actor_depth",
            "actor_num_heads",
            "actor_dropout",
            "num_registers",
            "h",
            "batch_size",
            "eval_sampling_temperature",
        ):
            cfg[key] = value
    return cfg


def _format_info(info: dict) -> str:
    lines = []
    keys = sorted(info.keys())
    for key in keys:
        val = float(np.asarray(info[key]))
        lines.append(f"  {key}: {val:.6f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teacher-forced vs free-running AR diagnostics for DiscreteARIQL."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exp run dir with flags.json and params_*.pkl.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to params_*.pkl (or omit if --run-dir has one).",
    )
    parser.add_argument(
        "--restore-epoch",
        type=int,
        default=None,
        help="Epoch when --checkpoint/--run-dir is a directory.",
    )
    parser.add_argument("--env-name", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument(
        "--ogbench-data-dir",
        type=str,
        default="/workspace-SR008.nfs2/users/staroverov/ogbench/data",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Actor selection (0=target_actor; >0=online). See agent docs.",
    )
    parser.add_argument(
        "--sample-freerun",
        action="store_true",
        help="Use sampling temperature for freerun (default: argmax freerun).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Average metrics over this many dataset batches.",
    )
    args = parser.parse_args()

    flags: dict = {}
    if args.run_dir is not None:
        flags = _load_flags(args.run_dir)
        if args.checkpoint is None:
            if args.restore_epoch is not None:
                args.checkpoint = args.run_dir
            else:
                args.checkpoint = _latest_params_pkl(args.run_dir)

    if args.checkpoint is None:
        raise SystemExit("Provide --checkpoint or --run-dir with params_*.pkl")

    env_name = args.env_name or flags.get("env_name")
    if not env_name:
        raise SystemExit("Provide --env-name or a --run-dir with flags.json")

    cfg = _config_from_flags(flags) if flags else dict(get_config())
    if args.tokenizer_path:
        cfg["tokenizer_path"] = args.tokenizer_path
    if not cfg.get("tokenizer_path"):
        raise SystemExit(
            "tokenizer_path missing: pass --tokenizer-path or use --run-dir "
            "whose flags.json has agent.tokenizer_path"
        )
    cfg["batch_size"] = int(args.batch_size)

    ogbench_data_dir = (
        args.ogbench_data_dir
        or flags.get("ogbench_data_dir")
        or "/workspace-SR008.nfs2/users/staroverov/ogbench/data"
    )

    print(f"env={env_name}")
    print(f"checkpoint={args.checkpoint}")
    print(f"tokenizer={cfg['tokenizer_path']}")
    print(f"batch_size={cfg['batch_size']} num_batches={args.num_batches}")

    _, _, train_dataset, _ = make_env_and_datasets(
        env_name,
        frame_stack=None,
        agent_config=cfg,
        dataset_dir=ogbench_data_dir,
    )
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=train_dataset.size + 1
    )
    train_dataset.config = cfg
    ex_batch = train_dataset.sample(1)

    agent = DiscreteARIQLAgent.create(
        args.seed,
        ex_batch["observations"],
        ex_batch["actions"],
        cfg,
    )
    restore_path = str(args.checkpoint)
    restore_epoch = args.restore_epoch
    if Path(restore_path).is_dir() and restore_epoch is None:
        raise SystemExit(
            "--restore-epoch required when checkpoint path is a directory"
        )
    agent = restore_agent(agent, restore_path, restore_epoch)

    force_argmax = not args.sample_freerun
    accum: dict[str, float] = {}
    n = 0
    for bi in range(int(args.num_batches)):
        batch = train_dataset.sample(int(args.batch_size))
        seed = jax.random.PRNGKey(args.seed + bi)
        info = agent.diagnose_teacher_vs_freerun(
            batch["observations"][0],
            batch["actions"],
            seed=seed,
            temperature=float(args.temperature),
            force_argmax=force_argmax,
        )
        for key, val in info.items():
            accum[key] = accum.get(key, 0.0) + float(np.asarray(val))
        n += 1

    mean_info = {k: v / n for k, v in accum.items()}
    print("teacher-forced vs free-running diagnostics (mean over batches):")
    print(_format_info(mean_info))
    print(
        f"summary: tf_token_acc={mean_info['tf_token_acc']:.4f} "
        f"fr_token_acc={mean_info['fr_token_acc']:.4f} "
        f"tf_seq_exact={mean_info['tf_seq_exact']:.4f} "
        f"fr_seq_exact={mean_info['fr_seq_exact']:.4f} "
        f"fr_action_rmse_gt={mean_info['fr_action_rmse_gt']:.4f} "
        f"fr_action_corr_gt={mean_info['fr_action_corr_gt']:.4f}"
    )


if __name__ == "__main__":
    main()
