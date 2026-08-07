#!/usr/bin/env python3
"""Aggregate Decoupled ConsensusFlow screening and scale-up runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_ARMS = (
    "endpoint_frozen",
    "endpoint_bc",
    "endpoint_trust",
    "latent_frozen",
    "latent_bc",
)
DEFAULT_TASKS = (
    "antmaze_giant_task1",
    "humanoidmaze_medium_task1",
    "antsoccer_arena_task4",
    "cube_double_task2",
    "puzzle_4x4_task4",
)
BASE_GROUPS = {
    "antmaze_giant_task1": "cftune-antmaze-C-rho0-lam1-o2o1m",
    "humanoidmaze_medium_task1": (
        "ogbench50-dflrql9-humanoidmaze_medium_task1-o2o1m"
    ),
    "antsoccer_arena_task4": "cftune-soccer-S3-std-a03-o2o1m",
    "cube_double_task2": "ogbench50-dflrql9-cube_double_task2-o2o1m",
    "puzzle_4x4_task4": "ogbench50-dflrql9-puzzle_4x4_task4-o2o1m",
}
SUCCESS_KEY = "evaluation/success"
RESIDUAL_OFF_SUCCESS_KEY = "evaluation/residual_off_success"
ROLE_GAP_KEY = "evaluation/role_gap_success"


def parse_csv(path: Path) -> list[dict[str, float]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            parsed: dict[str, float] = {}
            for key, value in raw.items():
                if value in (None, ""):
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    continue
            if parsed:
                rows.append(parsed)
    return rows


def deduplicate_steps(
    rows: Iterable[dict[str, float]],
) -> list[dict[str, float]]:
    by_step = {}
    for row in rows:
        if "step" in row:
            by_step[int(row["step"])] = row
    return [by_step[step] for step in sorted(by_step)]


def success_at(
    rows: Iterable[dict[str, float]],
    step: int,
    key: str = SUCCESS_KEY,
) -> float | None:
    for row in rows:
        if int(row.get("step", -1)) == int(step) and key in row:
            return float(row[key])
    return None


def normalized_auc(
    rows: Iterable[dict[str, float]],
    start_step: int,
    end_step: int,
    key: str = SUCCESS_KEY,
) -> float | None:
    points = [
        (int(row["step"]), float(row[key]))
        for row in rows
        if "step" in row
        and key in row
        and start_step <= int(row["step"]) <= end_step
    ]
    points.sort()
    if not points:
        return None
    if points[0][0] > start_step:
        points.insert(0, (start_step, points[0][1]))
    if points[-1][0] < end_step:
        points.append((end_step, points[-1][1]))
    x = np.asarray([point[0] for point in points], dtype=float)
    y = np.asarray([point[1] for point in points], dtype=float)
    duration = float(end_step - start_step)
    if duration <= 0:
        raise ValueError(
            f"end_step must exceed start_step: {start_step}, {end_step}"
        )
    return float(np.trapezoid(y, x) / duration)


def steps_to_threshold(
    rows: Iterable[dict[str, float]],
    threshold: float,
    start_step: int,
    key: str = SUCCESS_KEY,
) -> int | None:
    for row in sorted(rows, key=lambda item: item.get("step", float("inf"))):
        step = int(row.get("step", -1))
        if step >= start_step and row.get(key, -np.inf) >= threshold:
            return step
    return None


def latest_run(
    save_dir: Path,
    group: str,
    seed: int,
) -> Path | None:
    root = save_dir / "rql" / group
    candidates = []
    for run_dir in root.glob(f"sd{seed:03d}_*"):
        eval_rows = deduplicate_steps(parse_csv(run_dir / "eval.csv"))
        max_step = max(
            (int(row["step"]) for row in eval_rows if "step" in row),
            default=-1,
        )
        candidates.append((max_step, run_dir.stat().st_mtime, run_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def final_training_diagnostics(
    train_rows: list[dict[str, float]],
    report_step: int,
) -> dict[str, float | None]:
    eligible = [
        row
        for row in train_rows
        if int(row.get("step", -1)) <= report_step
    ]
    if not eligible:
        return {}
    row = max(eligible, key=lambda item: item.get("step", -1))

    def first(*keys):
        for key in keys:
            if key in row:
                return finite_or_none(row[key])
        return None

    critic_mse = first(
        "training/critic_loss",
        "training/latent_critic_loss",
    )
    q_mean = first("training/q_mean", "training/latent_q_mean")
    target_q_mean = first(
        "training/target_q_mean",
        "training/latent_target_q_mean",
    )
    q_bias = (
        finite_or_none(q_mean - target_q_mean)
        if q_mean is not None and target_q_mean is not None
        else None
    )
    return {
        "step": int(row["step"]),
        "residual_rms": first("training/residual_rms"),
        "dual_alpha": first("training/alpha"),
        "constraint_error": first("training/constraint_error"),
        "latent_kl": first("training/latent_kl"),
        "latent_mean_norm": first("training/latent_mean_norm"),
        "latent_std_mean": first("training/latent_std_mean"),
        "critic_bellman_mse": critic_mse,
        "critic_bellman_rmse": (
            finite_or_none(np.sqrt(max(critic_mse, 0.0)))
            if critic_mse is not None
            else None
        ),
        "critic_q_bias": q_bias,
        "online_replay_fraction": first(
            "training/online_replay_fraction"
        ),
    }


def run_metrics(
    run_dir: Path | None,
    *,
    milestones: tuple[int, ...],
    online_start: int,
    report_step: int,
) -> dict[str, Any]:
    if run_dir is None:
        return {
            "run_dir": None,
            "complete": False,
            "max_step": None,
            "success": {str(step): None for step in milestones},
            "online_auc": None,
            "steps_to_90": None,
            "role_gap": None,
            "diagnostics": {},
        }
    eval_rows = deduplicate_steps(parse_csv(run_dir / "eval.csv"))
    train_rows = deduplicate_steps(parse_csv(run_dir / "train.csv"))
    max_step = max(
        (int(row["step"]) for row in eval_rows if "step" in row),
        default=None,
    )
    report_success = success_at(eval_rows, report_step)
    residual_off = success_at(
        eval_rows,
        report_step,
        RESIDUAL_OFF_SUCCESS_KEY,
    )
    explicit_gap = success_at(eval_rows, report_step, ROLE_GAP_KEY)
    role_gap = explicit_gap
    if (
        role_gap is None
        and report_success is not None
        and residual_off is not None
    ):
        role_gap = report_success - residual_off
    return {
        "run_dir": str(run_dir),
        "complete": max_step is not None and max_step >= report_step,
        "max_step": max_step,
        "success": {
            str(step): success_at(eval_rows, step) for step in milestones
        },
        "online_auc": normalized_auc(
            eval_rows,
            online_start,
            report_step,
        ),
        "steps_to_90": steps_to_threshold(
            eval_rows,
            0.9,
            online_start,
        ),
        "residual_off_success": residual_off,
        "role_gap": role_gap,
        "diagnostics": final_training_diagnostics(
            train_rows,
            report_step,
        ),
    }


def bootstrap_mean(
    values: list[float],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
        }
    samples = rng.choice(
        array,
        size=(n_boot, array.size),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n": int(array.size),
        "n_boot": int(n_boot),
    }


def paired_bootstrap(
    method_values: list[float],
    base_values: list[float],
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    method = np.asarray(method_values, dtype=float)
    base = np.asarray(base_values, dtype=float)
    valid = np.isfinite(method) & np.isfinite(base)
    delta = method[valid] - base[valid]
    if delta.size == 0:
        return {
            "mean_delta": None,
            "ci_low": None,
            "ci_high": None,
            "n": 0,
        }
    samples = rng.choice(
        delta,
        size=(n_boot, delta.size),
        replace=True,
    ).mean(axis=1)
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "n": int(delta.size),
        "n_boot": int(n_boot),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
    }


def aggregate(args) -> dict[str, Any]:
    save_dir = Path(args.save_dir)
    tasks = tuple(args.tasks)
    arms = tuple(args.arms)
    seeds = tuple(args.seeds)
    milestones = tuple(sorted(set(args.milestones + [args.report_step])))
    rng = np.random.default_rng(args.bootstrap_seed)

    base: dict[str, dict[str, Any]] = {}
    for task in tasks:
        group = BASE_GROUPS[task]
        base[task] = {}
        for seed in seeds:
            run_dir = latest_run(save_dir, group, seed)
            base[task][str(seed)] = run_metrics(
                run_dir,
                milestones=milestones,
                online_start=args.online_start,
                report_step=args.report_step,
            )

    methods: dict[str, dict[str, Any]] = {}
    for arm in arms:
        methods[arm] = {}
        for task in tasks:
            group = f"dcf-{args.tag}-{arm}-{task}"
            methods[arm][task] = {}
            for seed in seeds:
                run_dir = latest_run(save_dir, group, seed)
                methods[arm][task][str(seed)] = run_metrics(
                    run_dir,
                    milestones=milestones,
                    online_start=args.online_start,
                    report_step=args.report_step,
                )

    summaries = {}
    base_success = []
    base_auc = []
    base_keys = []
    for task in tasks:
        for seed in seeds:
            metric = base[task][str(seed)]
            success = metric["success"][str(args.report_step)]
            auc = metric["online_auc"]
            if success is not None and auc is not None:
                base_keys.append((task, seed))
                base_success.append(success)
                base_auc.append(auc)

    for arm in arms:
        method_success = []
        paired_base_success = []
        method_auc = []
        paired_base_auc = []
        role_gaps = []
        completed = 0
        for task, seed in base_keys:
            metric = methods[arm][task][str(seed)]
            success = metric["success"][str(args.report_step)]
            auc = metric["online_auc"]
            if metric["complete"]:
                completed += 1
            if success is not None:
                method_success.append(success)
                paired_base_success.append(
                    base[task][str(seed)]["success"][
                        str(args.report_step)
                    ]
                )
            if auc is not None:
                method_auc.append(auc)
                paired_base_auc.append(base[task][str(seed)]["online_auc"])
            if metric.get("role_gap") is not None:
                role_gaps.append(metric["role_gap"])
        summaries[arm] = {
            "completed_pairs": completed,
            "success": bootstrap_mean(
                method_success,
                n_boot=args.n_boot,
                rng=rng,
            ),
            "success_vs_base": paired_bootstrap(
                method_success,
                paired_base_success,
                n_boot=args.n_boot,
                rng=rng,
            ),
            "online_auc": bootstrap_mean(
                method_auc,
                n_boot=args.n_boot,
                rng=rng,
            ),
            "online_auc_vs_base": paired_bootstrap(
                method_auc,
                paired_base_auc,
                n_boot=args.n_boot,
                rng=rng,
            ),
            "role_gap": bootstrap_mean(
                role_gaps,
                n_boot=args.n_boot,
                rng=rng,
            ),
        }

    ranking = sorted(
        arms,
        key=lambda arm: (
            summaries[arm]["success_vs_base"]["mean_delta"]
            if summaries[arm]["success_vs_base"]["mean_delta"] is not None
            else -np.inf,
            summaries[arm]["online_auc_vs_base"]["mean_delta"]
            if summaries[arm]["online_auc_vs_base"]["mean_delta"] is not None
            else -np.inf,
        ),
        reverse=True,
    )
    return {
        "protocol": {
            "tag": args.tag,
            "tasks": list(tasks),
            "seeds": list(seeds),
            "arms": list(arms),
            "online_start": args.online_start,
            "report_step": args.report_step,
            "milestones": list(milestones),
            "steps_to_threshold": 0.9,
            "success_key": SUCCESS_KEY,
        },
        "base": base,
        "methods": methods,
        "summaries": summaries,
        "ranking": ranking,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default="exp")
    parser.add_argument("--tag", default="seed0-o200k-n200k-v1")
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[
            "antmaze_giant_task1",
            "antsoccer_arena_task4",
            "cube_double_task2",
            "puzzle_4x4_task4",
        ],
        choices=DEFAULT_TASKS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--online-start", type=int, default=1_200_000)
    parser.add_argument("--report-step", type=int, default=1_400_000)
    parser.add_argument(
        "--milestones",
        nargs="+",
        type=int,
        default=[1_200_000, 1_400_000, 2_000_000],
    )
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="my_exps/decoupled_cf_screen.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result = aggregate(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")
    for rank, arm in enumerate(result["ranking"], start=1):
        summary = result["summaries"][arm]
        print(
            f"{rank}. {arm}: "
            f"success_delta={summary['success_vs_base']['mean_delta']} "
            f"auc_delta={summary['online_auc_vs_base']['mean_delta']} "
            f"completed={summary['completed_pairs']}"
        )


if __name__ == "__main__":
    main()
