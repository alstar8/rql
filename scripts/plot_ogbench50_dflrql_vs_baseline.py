#!/usr/bin/env python3
"""Plot DFL-RQL variants vs RQL baseline on the OGBench 50-task suite.

Optional --include-qdflrql9 discovers ogbench50-qdflrql9-* runs
(agent_name=quantized_dflrql9), same index path as v9.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ogbench50_config import (  # noqa: E402
    OGBench50Task,
    expand_tasks,
    exp_root_for_task,
    load_tasks_config,
)

PAPER_CHECKPOINT_OFFSETS = (200_000, 100_000, 0)

METHODS = {
    "RQL baseline": ("baseline", "#2563eb"),
    "DFL-RQL v6": ("dflrql6", "#dc2626"),
    "DFL-RQL v7": ("dflrql7", "#16a34a"),
    "DFL-RQL v8": ("dflrql8", "#7c3aed"),
    "DFL-RQL v9": ("dflrql9", "#0f766e"),
    "DFL-RQL v9 no-CRF": ("dflrql9_nocrf", "#ea580c"),
    "Quantized DFL-RQL v9": ("qdflrql9", "#e11d48"),
}

AGENT_FOR_METHOD = {
    "baseline": "rql",
    "dflrql6": "dflrql6",
    "dflrql7": "dflrql7",
    "dflrql8": "dflrql8",
    "dflrql9": "dflrql9",
    "dflrql9_nocrf": "dflrql9",
    "qdflrql9": "quantized_dflrql9",
}

METHOD_CSV_KEY = {
    "baseline": "baseline",
    "dflrql6": "v6",
    "dflrql7": "v7",
    "dflrql8": "v8",
    "dflrql9": "v9",
    "dflrql9_nocrf": "v9_nocrf",
    "qdflrql9": "qdflrql9",
}

# Fallback run-group prefixes when index miss (paper protocol prefers flags index).
METHOD_RUN_GROUP_PREFIX = {
    "baseline": "ogbench50-",
    "dflrql6": "ogbench50-dflrql6-",
    "dflrql7": "ogbench50-dflrql7-",
    "dflrql8": "ogbench50-dflrql8-",
    "dflrql9": "ogbench50-dflrql9-",
    "dflrql9_nocrf": "ogbench50-dflrql9-nocrf-",
    "qdflrql9": "ogbench50-qdflrql9-",
}

# Variant tags that share agent_name=dflrql9 but must not overwrite full CF.
_DFLRQL9_VARIANT_MARKERS = ("-nocrf-", "-noconflict-", "-nofloor-", "-noresidual-")


@dataclass
class RunRecord:
    run_dir: Path
    max_step: int
    has_curve: bool
    # Higher is better. Used to prefer task-level / 100M dataset runs over
    # stale env-level runs that share the same max eval step.
    quality: int = 0


def _run_quality(flags: dict, env_name: str) -> int:
    """Rank runs so paper-protocol plots pick the intended OGBench setup.

    Stale early baseline launches used env-level groups like
    ``ogbench50-cube-quadruple-play`` without ``ogbench_dataset_dir``. Later
    launches use per-task groups and the 100M shards for puzzle-4x4 /
    cube-quadruple. Prefer those so a zero-success stale curve does not win
    just because it also reached 2M steps.
    """
    score = 0
    run_group = str(flags.get("run_group", ""))
    dataset_dir = flags.get("ogbench_dataset_dir")
    if dataset_dir:
        score += 100
        if "100m" in str(dataset_dir).lower():
            score += 100
    # Per-task groups contain "_task" (e.g. ogbench50-cube_quadruple_task1).
    if "_task" in run_group:
        score += 50
    elif run_group.endswith(("-play", "-navigate")) or run_group in {
        f"ogbench50-{token}"
        for token in (
            "scene-play",
            "puzzle-3x3-play",
            "puzzle-4x4-play",
            "cube-double-play",
            "cube-triple-play",
            "cube-quadruple-play",
            "antmaze-large-navigate",
            "antmaze-giant-navigate",
            "humanoidmaze-medium-navigate",
            "humanoidmaze-large-navigate",
        )
    }:
        # Explicitly demote legacy env-level groups.
        score -= 20
    # Hard tasks without a dataset override are almost certainly the wrong run.
    if any(token in env_name for token in ("puzzle-4x4", "cube-quadruple")):
        if not dataset_dir:
            score -= 200
    return score


def _episode_return(row: dict[str, str]) -> float:
    for key in ("evaluation/episode.return", "evaluation/episode_return"):
        if key in row and row[key] not in (None, ""):
            return float(row[key])
    raise KeyError("evaluation/episode.return")


def parse_eval_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("step") in (None, ""):
                continue
            try:
                rows.append(
                    {
                        "step": int(float(row["step"])),
                        "success": float(row["evaluation/success"]),
                        "return": _episode_return(row),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def max_eval_step(eval_csv: Path) -> int:
    rows = parse_eval_csv(eval_csv)
    return max((row["step"] for row in rows), default=0)


def eval_has_step(eval_csv: Path, step: int) -> bool:
    rows = parse_eval_csv(eval_csv)
    return any(row["step"] == step for row in rows)


def report_checkpoints(report_step: int) -> tuple[int, int, int]:
    """Legacy last3 checkpoints: last three 100k-spaced evals ending at report_step."""
    return (
        report_step - PAPER_CHECKPOINT_OFFSETS[0],
        report_step - PAPER_CHECKPOINT_OFFSETS[1],
        report_step - PAPER_CHECKPOINT_OFFSETS[2],
    )


def checkpoint_label(report_step: int, *, last3: bool = False) -> str:
    """Human-readable checkpoint label for CSV / plot titles."""
    if last3:
        ckpts = report_checkpoints(report_step)
        return "/".join(f"{s // 1000}k" for s in ckpts)
    return f"{report_step // 1000}k"


def eval_has_all_checkpoints(
    eval_csv: Path,
    checkpoints: tuple[int, ...],
) -> bool:
    rows = parse_eval_csv(eval_csv)
    steps = {row["step"] for row in rows}
    return all(step in steps for step in checkpoints)


def mean_success_at_checkpoints(
    rows: list[dict[str, float]],
    checkpoints: tuple[int, ...],
) -> float | None:
    by_step = {row["step"]: row["success"] for row in rows}
    if not all(step in by_step for step in checkpoints):
        return None
    return float(np.mean([by_step[step] for step in checkpoints]))


def _paper_success_for_run(
    eval_csv: Path,
    checkpoints: tuple[int, ...] | None,
) -> float:
    """Mean success at paper checkpoints, else success at the final eval step."""
    rows = parse_eval_csv(eval_csv)
    if not rows:
        return float("-inf")
    by_step = {row["step"]: row["success"] for row in rows}
    if checkpoints is not None and all(step in by_step for step in checkpoints):
        return float(np.mean([by_step[step] for step in checkpoints]))
    return float(by_step[max(by_step)])


def build_result_index(
    save_dir: str,
    seed: int,
    prefer_step: int | None = None,
    prefer_checkpoints: tuple[int, ...] | None = None,
) -> dict[str, dict[str, RunRecord]]:
    """Map env_name -> method -> best run (handles per-task and env-level run groups)."""
    index: dict[str, dict[str, RunRecord]] = defaultdict(dict)
    rql_root = Path(save_dir) / "rql"
    if not rql_root.is_dir():
        return index

    for flags_path in rql_root.glob("**/flags.json"):
        if "ogbench50" not in str(flags_path):
            continue
        with flags_path.open() as f:
            flags = json.load(f)
        if flags.get("seed") != seed:
            continue
        agent_name = flags.get("agent", {}).get("agent_name")
        run_group = str(flags.get("run_group", ""))
        method = None
        if agent_name == "dflrql9":
            # Keep full CF and no-CRF (and other ablations) in separate buckets.
            if "-nocrf-" in run_group:
                method = "dflrql9_nocrf"
            elif any(marker in run_group for marker in _DFLRQL9_VARIANT_MARKERS):
                continue
            else:
                method = "dflrql9"
        else:
            for m, agent in AGENT_FOR_METHOD.items():
                if m in ("dflrql9", "dflrql9_nocrf"):
                    continue
                if agent_name == agent:
                    method = m
                    break
        if method is None:
            continue

        env_name = flags["env_name"]
        run_dir = flags_path.parent
        eval_csv = run_dir / "eval.csv"
        if not eval_csv.is_file():
            continue
        try:
            step = max_eval_step(eval_csv)
        except (KeyError, ValueError):
            continue
        if step <= 0:
            continue
        if prefer_checkpoints is not None:
            if not eval_has_all_checkpoints(eval_csv, prefer_checkpoints):
                continue
        elif prefer_step is not None and not eval_has_step(eval_csv, prefer_step):
            continue

        quality = _run_quality(flags, env_name)
        success = _paper_success_for_run(eval_csv, prefer_checkpoints)
        record = RunRecord(
            run_dir=run_dir,
            max_step=step,
            has_curve=True,
            quality=quality,
        )
        prev = index[env_name].get(method)
        if prev is None:
            index[env_name][method] = record
            continue
        prev_success = _paper_success_for_run(
            prev.run_dir / "eval.csv", prefer_checkpoints
        )
        if (record.quality, record.max_step, success) > (
            prev.quality,
            prev.max_step,
            prev_success,
        ):
            index[env_name][method] = record

    return index


def find_eval_csv(exp_root: Path, seed: int, env_name: str | None = None) -> Path | None:
    candidates: list[tuple[int, Path]] = []
    for run_dir in sorted(exp_root.glob(f"sd{seed:03d}_*")):
        eval_csv = run_dir / "eval.csv"
        if not eval_csv.is_file():
            continue
        if env_name is not None:
            flags_path = run_dir / "flags.json"
            if flags_path.is_file():
                with flags_path.open() as f:
                    flags = json.load(f)
                if flags.get("env_name") != env_name:
                    continue
        try:
            step = max_eval_step(eval_csv)
        except (KeyError, ValueError):
            continue
        candidates.append((step, eval_csv))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def parse_log_final_metrics(log_path: Path) -> tuple[float | None, float | None]:
    if not log_path.is_file():
        return None, None
    text = log_path.read_text(errors="replace")
    success_matches = re.findall(r"^success ([0-9.]+)", text, re.MULTILINE)
    return_matches = re.findall(r"^episode\.return ([-\d.]+)", text, re.MULTILINE)
    success = float(success_matches[-1]) if success_matches else None
    ret = float(return_matches[-1]) if return_matches else None
    return success, ret


def parse_log_final_success(log_path: Path) -> float | None:
    success, _ = parse_log_final_metrics(log_path)
    return success


def load_task_curve(
    save_dir: str,
    task: OGBench50Task,
    method: str,
    seed: int,
    index: dict[str, dict[str, RunRecord]],
) -> list[dict[str, float]] | None:
    record = index.get(task.env_name, {}).get(method)
    if record is not None:
        eval_csv = record.run_dir / "eval.csv"
        if eval_csv.is_file():
            return parse_eval_csv(eval_csv)

    stem = task.task_id.replace("-", "_")
    prefix = METHOD_RUN_GROUP_PREFIX.get(method, "ogbench50-")
    exp_root = Path(save_dir) / "rql" / f"{prefix}{stem}"
    eval_csv = find_eval_csv(exp_root, seed, env_name=task.env_name)
    if eval_csv is None:
        eval_csv = find_eval_csv(exp_root, seed)
    if eval_csv is None:
        return None
    return parse_eval_csv(eval_csv)


def final_metrics_for_task(
    save_dir: str,
    task: OGBench50Task,
    method: str,
    seed: int,
    index: dict[str, dict[str, RunRecord]],
    report_step: int = 1_000_000,
    log_roots: dict[str, Path] | None = None,
    require_exact_step: bool = False,
    use_log_fallback: bool = True,
) -> tuple[float | None, float | None]:
    rows = load_task_curve(save_dir, task, method, seed, index)
    if rows is not None:
        by_step_success = {row["step"]: row["success"] for row in rows}
        by_step_return = {row["step"]: row["return"] for row in rows}
        if report_step in by_step_success:
            return by_step_success[report_step], by_step_return.get(report_step)
        if not require_exact_step and by_step_success:
            last_step = max(by_step_success)
            return by_step_success[last_step], by_step_return.get(last_step)

    if use_log_fallback and log_roots is not None and method in log_roots:
        log_root = log_roots[method]
        for log_path in (
            log_root / f"seed{seed:03d}" / f"{task.env_name}.log",
            log_root / f"{task.env_name}.log",
        ):
            success, ret = parse_log_final_metrics(log_path)
            if success is not None:
                return success, ret
    return None, None


def paper_protocol_success_for_task(
    save_dir: str,
    task: OGBench50Task,
    method: str,
    seed: int,
    index: dict[str, dict[str, RunRecord]],
    report_step: int,
) -> float | None:
    """Authoritative paper protocol: exact success at ``report_step`` only."""
    success, _ = final_metrics_for_task(
        save_dir,
        task,
        method,
        seed,
        index,
        report_step=report_step,
        require_exact_step=True,
        use_log_fallback=False,
    )
    return success


def last3_protocol_success_for_task(
    save_dir: str,
    task: OGBench50Task,
    method: str,
    seed: int,
    index: dict[str, dict[str, RunRecord]],
    report_step: int,
) -> float | None:
    """Legacy: mean success over the last three 100k-spaced checkpoints."""
    checkpoints = report_checkpoints(report_step)
    rows = load_task_curve(save_dir, task, method, seed, index)
    if rows is None:
        return None
    return mean_success_at_checkpoints(rows, checkpoints)


def log_roots_for_seed(
    baseline_log_root: Path,
    v6_log_root: Path,
    seed: int,
    v7_log_root: Path | None = None,
    v8_log_root: Path | None = None,
    v9_log_root: Path | None = None,
    qdflrql9_log_root: Path | None = None,
) -> dict[str, Path]:
    """Per-seed log dirs; flat layout is only valid for seed 0 (legacy)."""
    roots: dict[str, Path] = {}
    mapping = [
        ("baseline", baseline_log_root),
        ("dflrql6", v6_log_root),
    ]
    if v7_log_root is not None:
        mapping.append(("dflrql7", v7_log_root))
    if v8_log_root is not None:
        mapping.append(("dflrql8", v8_log_root))
    if v9_log_root is not None:
        mapping.append(("dflrql9", v9_log_root))
    if qdflrql9_log_root is not None:
        mapping.append(("qdflrql9", qdflrql9_log_root))
    for method, base in mapping:
        seed_dir = base / f"seed{seed:03d}"
        if seed_dir.is_dir():
            roots[method] = seed_dir
        elif seed == 0:
            roots[method] = base
        else:
            roots[method] = seed_dir  # missing — log fallback will no-op
    return roots


def final_success_for_task(
    save_dir: str,
    task: OGBench50Task,
    method: str,
    seed: int,
    index: dict[str, dict[str, RunRecord]],
    report_step: int = 1_000_000,
    log_root: Path | None = None,
    log_roots: dict[str, Path] | None = None,
) -> float | None:
    roots = log_roots
    if roots is None and log_root is not None:
        roots = {"dflrql6": log_root}
    success, _ = final_metrics_for_task(
        save_dir, task, method, seed, index, report_step, roots
    )
    return success


def aggregate_curves(
    save_dir: str,
    tasks: list[OGBench50Task],
    method: str,
    seed: int,
    index: dict[str, dict[str, RunRecord]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    by_step_success: dict[int, list[float]] = defaultdict(list)
    by_step_return: dict[int, list[float]] = defaultdict(list)
    completed = 0

    for task in tasks:
        rows = load_task_curve(save_dir, task, method, seed, index)
        if rows is None:
            continue
        completed += 1
        for row in rows:
            by_step_success[row["step"]].append(row["success"])
            by_step_return[row["step"]].append(row["return"])

    if not by_step_success:
        empty = np.array([])
        return empty, empty, empty, empty, empty, 0

    steps = np.array(sorted(by_step_success))
    success_mean = np.array([np.mean(by_step_success[s]) for s in steps])
    success_std = np.array([np.std(by_step_success[s]) for s in steps])
    return_mean = np.array([np.mean(by_step_return[s]) for s in steps])
    return_std = np.array([np.std(by_step_return[s]) for s in steps])
    return steps, success_mean, success_std, return_mean, return_std, completed


def final_success_at_step(
    save_dir: str,
    tasks: list[OGBench50Task],
    method: str,
    seed: int,
    index: dict[str, dict[str, RunRecord]],
    report_step: int = 1_000_000,
    log_root: Path | None = None,
    log_roots: dict[str, Path] | None = None,
) -> dict[str, float | None]:
    domain_vals: dict[str, list[float]] = defaultdict(list)
    for task in tasks:
        val = final_success_for_task(
            save_dir,
            task,
            method,
            seed,
            index,
            report_step,
            log_root,
            log_roots,
        )
        domain_vals[task.domain_id].append(np.nan if val is None else val)

    return {
        domain_id: (
            float(np.nanmean(values))
            if values and not np.all(np.isnan(values))
            else None
        )
        for domain_id, values in domain_vals.items()
    }


def plot_learning_curves(
    save_dir: str,
    tasks: list[OGBench50Task],
    seed: int,
    index: dict[str, dict[str, RunRecord]],
    output_path: Path,
) -> dict[str, float | None]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    overall: dict[str, float | None] = {}

    for ax_idx, (metric, ylabel) in enumerate(
        [("success", "Success rate"), ("return", "Episode return")]
    ):
        ax = axes[ax_idx]
        for label, (method, color) in METHODS.items():
            steps, s_mean, s_std, r_mean, r_std, n_done = aggregate_curves(
                save_dir, tasks, method, seed, index
            )
            if metric == "success":
                y, y_std = s_mean, s_std
            else:
                y, y_std = r_mean, r_std
            if len(steps) == 0:
                continue
            ax.plot(steps, y, label=f"{label} ({n_done}/{len(tasks)} tasks)", color=color, lw=2)
            ax.fill_between(steps, y - y_std, y + y_std, color=color, alpha=0.15)
            if metric == "success" and steps[-1] >= 900_000:
                overall[label] = float(y[-1])
        ax.set_xlabel("Training steps")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        "OGBench 50-task: DFL-RQL v6 vs RQL baseline (mean ± std over tasks, seed 0)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return overall


def plot_domain_bars(
    save_dir: str,
    tasks: list[OGBench50Task],
    seed: int,
    index: dict[str, dict[str, RunRecord]],
    output_path: Path,
    log_roots: dict[str, Path] | None,
    report_step: int = 1_000_000,
) -> None:
    domain_ids = [d["id"] for d in load_tasks_config()["domains"]]
    x = np.arange(len(domain_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (label, (method, color)) in enumerate(METHODS.items()):
        domain_mean = final_success_at_step(
            save_dir, tasks, method, seed, index, report_step, log_roots=log_roots
        )
        values = [domain_mean.get(d, np.nan) for d in domain_ids]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
        for bar, val in zip(bars, values):
            if np.isnan(val):
                bar.set_hatch("xx")
                bar.set_alpha(0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(domain_ids, rotation=35, ha="right")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-domain success @ {report_step // 1000}k steps (mean over 5 tasks)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_per_task_bars(
    save_dir: str,
    tasks: list[OGBench50Task],
    seed: int,
    index: dict[str, dict[str, RunRecord]],
    output_path: Path,
    log_roots: dict[str, Path] | None,
    report_step: int = 1_000_000,
) -> None:
    labels = [task.task_id for task in tasks]
    x = np.arange(len(tasks))
    width = 0.38

    fig, ax = plt.subplots(figsize=(22, 6))
    for i, (method_label, (method, color)) in enumerate(METHODS.items()):
        values = [
            final_success_for_task(
                save_dir,
                task,
                method,
                seed,
                index,
                report_step,
                log_roots=log_roots,
            )
            for task in tasks
        ]
        plot_values = [np.nan if v is None else v for v in values]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, plot_values, width, label=method_label, color=color)
        for bar, val in zip(bars, plot_values):
            if val is None:
                bar.set_hatch("xx")
                bar.set_alpha(0.35)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(f"Success rate @ {report_step // 1000}k steps")
    ax.set_ylim(0, 1.05)
    ax.set_title("OGBench 50-task: DFL-RQL v6 vs RQL baseline (per task, seed 0)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def success_at_step(rows: list[dict[str, float]] | None, step: int) -> float | None:
    if not rows:
        return None
    by_step = {row["step"]: row["success"] for row in rows}
    return by_step.get(step)


def max_common_eval_step(
    curves: dict[str, list[dict[str, float]] | None],
    methods: list[str],
    min_step: int = 100_000,
) -> int | None:
    """Largest eval step present in every listed method's curve (at least min_step)."""
    step_sets: list[set[int]] = []
    for method in methods:
        rows = curves.get(method)
        if not rows:
            return None
        step_sets.append({row["step"] for row in rows if row["step"] >= min_step})
    common = set.intersection(*step_sets) if step_sets else set()
    return max(common) if common else None


def collect_common_max_step_metrics(
    save_dir: str,
    tasks: list[OGBench50Task],
    seeds: list[int],
    compare_methods: list[str],
    min_common_step: int = 100_000,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Fair interim compare: per (task, seed) use max shared eval step across methods."""
    indices = {seed: build_result_index(save_dir, seed) for seed in seeds}
    methods = list(compare_methods)
    rows: list[dict] = []
    # Per-seed aggregates only over tasks with a valid common step for that seed.
    per_seed_task_vals: dict[str, dict[int, list[float]]] = {
        METHOD_CSV_KEY[m]: {seed: [] for seed in seeds} for m in methods
    }
    common_steps_all: list[int] = []

    for task in tasks:
        row: dict = {
            "task_id": task.task_id,
            "domain_id": task.domain_id,
            "env_name": task.env_name,
            "report_mode": "common_max_step",
            "compare_methods": ",".join(methods),
            "min_common_step": min_common_step,
        }
        paired: dict[str, list[float]] = {METHOD_CSV_KEY[m]: [] for m in methods}
        step_vals: list[int] = []
        for seed in seeds:
            curves = {
                method: load_task_curve(
                    save_dir, task, method, seed, indices[seed]
                )
                for method in methods
            }
            common_step = max_common_eval_step(
                curves, methods, min_step=min_common_step
            )
            if common_step is None:
                row[f"common_step_seed{seed}"] = None
                continue
            row[f"common_step_seed{seed}"] = common_step
            step_vals.append(common_step)
            common_steps_all.append(common_step)
            for method in methods:
                key = METHOD_CSV_KEY[method]
                success = success_at_step(curves[method], common_step)
                if success is None:
                    continue
                paired[key].append(success)
                row[f"{key}_seed{seed}"] = success
                per_seed_task_vals[key][seed].append(success)

        for method in methods:
            key = METHOD_CSV_KEY[method]
            vals = paired[key]
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            row[f"{key}_n"] = len(vals)
        row["common_step_mean"] = float(np.mean(step_vals)) if step_vals else None
        row["common_step_min"] = int(min(step_vals)) if step_vals else None
        row["common_step_max"] = int(max(step_vals)) if step_vals else None
        row["common_step_n"] = len(step_vals)
        row["report_step"] = row["common_step_mean"]
        row["report_checkpoints"] = (
            f"common_max≈{int(row['common_step_mean']) // 1000}k"
            if row["common_step_mean"] is not None
            else "common_max=NA"
        )

        base_m = row.get("baseline_mean")
        if "dflrql6" in methods:
            v6_m = row.get("v6_mean")
            row["delta_mean"] = (
                v6_m - base_m
                if base_m is not None and v6_m is not None
                else None
            )
        if "dflrql7" in methods:
            v7_m = row.get("v7_mean")
            row["delta_v7_mean"] = (
                v7_m - base_m
                if base_m is not None and v7_m is not None
                else None
            )
        if "dflrql8" in methods:
            v8_m = row.get("v8_mean")
            row["delta_v8_mean"] = (
                v8_m - base_m
                if base_m is not None and v8_m is not None
                else None
            )
        if "dflrql9" in methods:
            v9_m = row.get("v9_mean")
            row["delta_v9_mean"] = (
                v9_m - base_m
                if base_m is not None and v9_m is not None
                else None
            )
        if "qdflrql9" in methods:
            qd_m = row.get("qdflrql9_mean")
            row["delta_qdflrql9_mean"] = (
                qd_m - base_m
                if base_m is not None and qd_m is not None
                else None
            )
        rows.append(row)

    per_seed_aggregate: dict[str, list[float]] = {
        METHOD_CSV_KEY[m]: [] for m in methods
    }
    for method in methods:
        key = METHOD_CSV_KEY[method]
        for seed in seeds:
            vals = per_seed_task_vals[key][seed]
            if vals:
                per_seed_aggregate[key].append(float(np.mean(vals)))

    if rows:
        rows[0]["_global_common_step_mean"] = (
            float(np.mean(common_steps_all)) if common_steps_all else None
        )
        rows[0]["_global_common_step_min"] = (
            int(min(common_steps_all)) if common_steps_all else None
        )
        rows[0]["_global_common_step_max"] = (
            int(max(common_steps_all)) if common_steps_all else None
        )
        rows[0]["_global_common_n"] = len(common_steps_all)

    return rows, {k: np.array(v) for k, v in per_seed_aggregate.items()}


def collect_multiseed_metrics(
    save_dir: str,
    tasks: list[OGBench50Task],
    seeds: list[int],
    baseline_log_root: Path,
    v6_log_root: Path,
    report_step: int = 1_000_000,
    paper_protocol: bool = True,
    require_exact_step: bool = False,
    last3_average: bool = False,
    v7_log_root: Path | None = None,
    include_v7: bool = False,
    v8_log_root: Path | None = None,
    include_v8: bool = False,
    v9_log_root: Path | None = None,
    include_v9: bool = False,
    include_nocrf: bool = False,
    qdflrql9_log_root: Path | None = None,
    include_qdflrql9: bool = False,
    common_max_step: bool = False,
    compare_methods: list[str] | None = None,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Per-task mean/std over seeds + per-seed 50-task aggregates for headline stats.

    Paper protocol (default): exact ``evaluation/success`` at ``report_step``
    only (final checkpoint). Pass ``last3_average=True`` for the RQL-style
    mean of the last three 100k-spaced checkpoints.
    """
    if common_max_step:
        if compare_methods is not None:
            methods = compare_methods
        elif include_qdflrql9:
            methods = ["baseline", "qdflrql9"]
        elif include_nocrf:
            methods = ["baseline", "dflrql9", "dflrql9_nocrf"]
        elif include_v9:
            methods = ["baseline", "dflrql9"]
        elif include_v8:
            methods = ["baseline", "dflrql8"]
        elif include_v7:
            methods = ["baseline", "dflrql7"]
        else:
            methods = ["baseline", "dflrql6"]
        return collect_common_max_step_metrics(save_dir, tasks, seeds, methods)

    use_final_checkpoint = paper_protocol and not last3_average
    use_last3 = paper_protocol and last3_average
    if use_last3:
        checkpoints = report_checkpoints(report_step)
        indices = {
            seed: build_result_index(
                save_dir, seed, prefer_checkpoints=checkpoints
            )
            for seed in seeds
        }
    elif use_final_checkpoint or require_exact_step:
        indices = {
            seed: build_result_index(
                save_dir,
                seed,
                prefer_step=report_step,
            )
            for seed in seeds
        }
    else:
        indices = {
            seed: build_result_index(save_dir, seed)
            for seed in seeds
        }
    methods = ["baseline", "dflrql6"]
    if include_v7:
        methods.append("dflrql7")
    if include_v8:
        methods.append("dflrql8")
    if include_v9:
        methods.append("dflrql9")
    if include_nocrf:
        methods.append("dflrql9_nocrf")
    if include_qdflrql9:
        methods.append("qdflrql9")
    rows: list[dict] = []
    per_seed_aggregate: dict[str, list[float]] = {
        METHOD_CSV_KEY[m]: [] for m in methods
    }
    use_log_fallback = not paper_protocol and not require_exact_step

    def task_success(
        task: OGBench50Task,
        method: str,
        seed: int,
    ) -> float | None:
        if use_last3:
            return last3_protocol_success_for_task(
                save_dir, task, method, seed, indices[seed], report_step
            )
        if use_final_checkpoint:
            return paper_protocol_success_for_task(
                save_dir, task, method, seed, indices[seed], report_step
            )
        roots = log_roots_for_seed(
            baseline_log_root,
            v6_log_root,
            seed,
            v7_log_root=v7_log_root,
            v8_log_root=v8_log_root,
            v9_log_root=v9_log_root,
            qdflrql9_log_root=qdflrql9_log_root,
        )
        success, _ = final_metrics_for_task(
            save_dir,
            task,
            method,
            seed,
            indices[seed],
            report_step,
            roots,
            require_exact_step=require_exact_step,
            use_log_fallback=use_log_fallback,
        )
        return success

    ckpt_label = checkpoint_label(report_step, last3=use_last3)
    for task in tasks:
        row: dict = {
            "task_id": task.task_id,
            "domain_id": task.domain_id,
            "env_name": task.env_name,
            "report_step": report_step,
            "report_checkpoints": ckpt_label,
        }
        for method in methods:
            key = METHOD_CSV_KEY[method]
            vals: list[float] = []
            for seed in seeds:
                success = task_success(task, method, seed)
                if success is not None:
                    vals.append(success)
                    row[f"{key}_seed{seed}"] = success
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            row[f"{key}_n"] = len(vals)

        base_m = row.get("baseline_mean")
        v6_m = row.get("v6_mean")
        row["delta_mean"] = (
            v6_m - base_m
            if base_m is not None and v6_m is not None
            else None
        )
        if include_v7:
            v7_m = row.get("v7_mean")
            row["delta_v7_mean"] = (
                v7_m - base_m
                if base_m is not None and v7_m is not None
                else None
            )
        if include_v8:
            v8_m = row.get("v8_mean")
            row["delta_v8_mean"] = (
                v8_m - base_m
                if base_m is not None and v8_m is not None
                else None
            )
        if include_v9:
            v9_m = row.get("v9_mean")
            row["delta_v9_mean"] = (
                v9_m - base_m
                if base_m is not None and v9_m is not None
                else None
            )
        if include_nocrf:
            nocrf_m = row.get("v9_nocrf_mean")
            row["delta_v9_nocrf_mean"] = (
                nocrf_m - base_m
                if base_m is not None and nocrf_m is not None
                else None
            )
        if include_qdflrql9:
            qd_m = row.get("qdflrql9_mean")
            row["delta_qdflrql9_mean"] = (
                qd_m - base_m
                if base_m is not None and qd_m is not None
                else None
            )
        rows.append(row)

    for seed in seeds:
        for method in methods:
            key = METHOD_CSV_KEY[method]
            task_vals: list[float] = []
            for task in tasks:
                success = task_success(task, method, seed)
                if success is not None:
                    task_vals.append(success)
            if task_vals:
                per_seed_aggregate[key].append(float(np.mean(task_vals)))

    return rows, {k: np.array(v) for k, v in per_seed_aggregate.items()}


def aggregate_headline_stats(
    per_seed_aggregate: dict[str, np.ndarray],
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key in per_seed_aggregate:
        vals = per_seed_aggregate.get(key, np.array([]))
        if len(vals) == 0:
            out[key] = (float("nan"), float("nan"))
        else:
            out[key] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def plot_all_tasks_metrics(
    save_dir: str,
    tasks: list[OGBench50Task],
    seeds: list[int],
    output_path: Path,
    baseline_log_root: Path,
    v6_log_root: Path,
    report_step: int = 1_000_000,
    paper_protocol: bool = True,
    require_exact_step: bool = False,
    last3_average: bool = False,
    v7_log_root: Path | None = None,
    include_v7: bool = False,
    v8_log_root: Path | None = None,
    include_v8: bool = False,
    v9_log_root: Path | None = None,
    include_v9: bool = False,
    include_nocrf: bool = False,
    qdflrql9_log_root: Path | None = None,
    include_qdflrql9: bool = False,
    common_max_step: bool = False,
    compare_methods: list[str] | None = None,
) -> list[dict]:
    metrics, per_seed_agg = collect_multiseed_metrics(
        save_dir,
        tasks,
        seeds,
        baseline_log_root,
        v6_log_root,
        report_step,
        paper_protocol=paper_protocol,
        require_exact_step=require_exact_step,
        last3_average=last3_average,
        v7_log_root=v7_log_root,
        include_v7=include_v7,
        v8_log_root=v8_log_root,
        include_v8=include_v8,
        v9_log_root=v9_log_root,
        include_v9=include_v9,
        include_nocrf=include_nocrf,
        qdflrql9_log_root=qdflrql9_log_root,
        include_qdflrql9=include_qdflrql9,
        common_max_step=common_max_step,
        compare_methods=compare_methods,
    )
    if common_max_step:
        step_vals = [
            m["common_step_mean"]
            for m in metrics
            if m.get("common_step_mean") is not None
        ]
        if step_vals:
            protocol_note = (
                f"at per-env max common eval step "
                f"(mean≈{int(np.mean(step_vals)) // 1000}k, "
                f"range {int(min(m['common_step_min'] for m in metrics if m.get('common_step_min') is not None)) // 1000}k"
                f"–{int(max(m['common_step_max'] for m in metrics if m.get('common_step_max') is not None)) // 1000}k)"
            )
        else:
            protocol_note = "at per-env max common eval step (no paired data)"
    else:
        ckpt_label = checkpoint_label(report_step, last3=last3_average)
        if paper_protocol and last3_average:
            protocol_note = f"mean @ {ckpt_label} (legacy last3)"
        elif paper_protocol:
            protocol_note = f"@ {ckpt_label} final checkpoint (paper protocol)"
        else:
            protocol_note = f"@ {report_step // 1000}k steps"
    headline = aggregate_headline_stats(per_seed_agg)

    def _env_caption(task: OGBench50Task) -> str:
        """Readable env caption for x-axis (domain + task id)."""
        # e.g. humanoidmaze-large-navigate-singletask-task3-v0
        #   -> humanoidmaze-large-navigate task3
        name = task.env_name
        if name.endswith("-v0"):
            name = name[:-3]
        if "-singletask-" in name:
            domain, task_part = name.split("-singletask-", 1)
            return f"{domain} {task_part}"
        return name

    labels = [_env_caption(task) for task in tasks]
    x = np.arange(len(tasks))
    if compare_methods is not None:
        allowed = set(compare_methods)
    else:
        allowed = {"baseline", "dflrql6"}
        if include_v7:
            allowed.add("dflrql7")
        if include_v8:
            allowed.add("dflrql8")
        if include_v9:
            allowed.add("dflrql9")
        if include_nocrf:
            allowed.add("dflrql9_nocrf")
        if include_qdflrql9:
            allowed.add("qdflrql9")
    # Drop methods with no aggregate data (e.g. v6 when comparing only baseline+v7).
    active_methods = [
        (label, method, color)
        for label, (method, color) in METHODS.items()
        if method in allowed and len(per_seed_agg.get(METHOD_CSV_KEY[method], [])) > 0
    ]
    if not active_methods:
        active_methods = [
            (label, method, color)
            for label, (method, color) in METHODS.items()
            if method in allowed
        ]
    n_methods = len(active_methods)
    width = 0.8 / max(n_methods, 1)
    n_seeds_requested = len(seeds)
    show_v7_delta = any(m == "dflrql7" for _, m, _ in active_methods)
    show_v6_delta = any(m == "dflrql6" for _, m, _ in active_methods)
    show_v8_delta = any(m == "dflrql8" for _, m, _ in active_methods)
    show_v9_delta = any(m == "dflrql9" for _, m, _ in active_methods)
    show_nocrf_delta = any(m == "dflrql9_nocrf" for _, m, _ in active_methods)
    show_qdflrql9_delta = any(m == "qdflrql9" for _, m, _ in active_methods)

    fig, axes = plt.subplots(
        3, 1, figsize=(24, 14), sharex=False, gridspec_kw={"height_ratios": [3.2, 2.2, 1.4]}
    )

    ax = axes[0]
    for i, (method_label, method, color) in enumerate(active_methods):
        key = METHOD_CSV_KEY[method]
        means = [
            m[f"{key}_mean"] if m.get(f"{key}_mean") is not None else np.nan
            for m in metrics
        ]
        stds = [m[f"{key}_std"] if m.get(f"{key}_n") else np.nan for m in metrics]
        n_tasks = sum(1 for m in metrics if m.get(f"{key}_n", 0) > 0)
        n_seed_m = len(per_seed_agg.get(key, []))
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=2,
            label=f"{method_label} ({n_tasks}/50 tasks, {n_seed_m}/{n_seeds_requested} seeds)",
            color=color,
            error_kw={"elinewidth": 0.8, "alpha": 0.7},
        )
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Success {protocol_note} (mean ± std over seeds)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    delta_series: list[tuple[str, str, str, str]] = []
    if show_v6_delta:
        delta_series.append(("v6", "delta_mean", "#dc2626", "v6 − baseline"))
    if show_v7_delta:
        delta_series.append(("v7", "delta_v7_mean", "#16a34a", "v7 − baseline"))
    if show_v8_delta:
        delta_series.append(("v8", "delta_v8_mean", "#7c3aed", "v8 − baseline"))
    if show_v9_delta:
        delta_series.append(("v9", "delta_v9_mean", "#0f766e", "v9 − baseline"))
    if show_nocrf_delta:
        delta_series.append(
            ("v9_nocrf", "delta_v9_nocrf_mean", "#ea580c", "v9-noCRF − baseline")
        )
    if show_qdflrql9_delta:
        delta_series.append(
            ("qdflrql9", "delta_qdflrql9_mean", "#e11d48", "qdflrql9 − baseline")
        )
    n_delta = max(len(delta_series), 1)
    dwidth = 0.7 / n_delta
    for di, (_key, col, color, ylabel) in enumerate(delta_series):
        vals = [m[col] if m.get(col) is not None else np.nan for m in metrics]
        offset = (di - (n_delta - 1) / 2) * dwidth
        bar_colors = [
            color if (not np.isnan(d) and d >= 0) else "#94a3b8" for d in vals
        ]
        ax.bar(
            x + offset,
            vals,
            width=dwidth,
            color=bar_colors,
            alpha=0.85,
            label=ylabel,
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Δ success vs baseline")
    ax.set_title("Per-task Δ success (mean over paired seeds at common step)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    ax.axis("off")
    lines = [
        f"50-task aggregate {protocol_note} (mean ± std over seeds)",
        "",
    ]
    for method_label, method, _color in active_methods:
        key = METHOD_CSV_KEY[method]
        mean, std = headline.get(key, (float("nan"), float("nan")))
        n_seed = len(per_seed_agg.get(key, []))
        lines.append(
            f"{method_label:24s} {mean:.3f} ± {std:.3f}  ({n_seed} seed aggregates)"
        )
    base_mean, _ = headline.get("baseline", (float("nan"), float("nan")))
    if show_v7_delta:
        v7_mean, _ = headline.get("v7", (float("nan"), float("nan")))
        if not np.isnan(base_mean) and not np.isnan(v7_mean):
            lines.append(f"Δ (v7 − baseline) = {v7_mean - base_mean:+.3f}")
    if show_v8_delta:
        v8_mean, _ = headline.get("v8", (float("nan"), float("nan")))
        if not np.isnan(base_mean) and not np.isnan(v8_mean):
            lines.append(f"Δ (v8 − baseline) = {v8_mean - base_mean:+.3f}")
    if show_v9_delta:
        v9_mean, _ = headline.get("v9", (float("nan"), float("nan")))
        if not np.isnan(base_mean) and not np.isnan(v9_mean):
            lines.append(f"Δ (v9 − baseline) = {v9_mean - base_mean:+.3f}")
    if show_qdflrql9_delta:
        qd_mean, _ = headline.get("qdflrql9", (float("nan"), float("nan")))
        if not np.isnan(base_mean) and not np.isnan(qd_mean):
            lines.append(f"Δ (qdflrql9 − baseline) = {qd_mean - base_mean:+.3f}")
    if show_v6_delta:
        v6_mean, _ = headline.get("v6", (float("nan"), float("nan")))
        if not np.isnan(base_mean) and not np.isnan(v6_mean):
            lines.append(f"Δ (v6 − baseline) = {v6_mean - base_mean:+.3f}")
        if show_v7_delta:
            v7_mean, _ = headline.get("v7", (float("nan"), float("nan")))
            if not np.isnan(v6_mean) and not np.isnan(v7_mean):
                lines.append(f"Δ (v7 − v6)       = {v7_mean - v6_mean:+.3f}")
        if show_v8_delta:
            v8_mean, _ = headline.get("v8", (float("nan"), float("nan")))
            if not np.isnan(v6_mean) and not np.isnan(v8_mean):
                lines.append(f"Δ (v8 − v6)       = {v8_mean - v6_mean:+.3f}")
        if show_v9_delta:
            v9_mean, _ = headline.get("v9", (float("nan"), float("nan")))
            if not np.isnan(v6_mean) and not np.isnan(v9_mean):
                lines.append(f"Δ (v9 − v6)       = {v9_mean - v6_mean:+.3f}")
    if show_v7_delta and show_v8_delta:
        v7_mean, _ = headline.get("v7", (float("nan"), float("nan")))
        v8_mean, _ = headline.get("v8", (float("nan"), float("nan")))
        if not np.isnan(v7_mean) and not np.isnan(v8_mean):
            lines.append(f"Δ (v8 − v7)       = {v8_mean - v7_mean:+.3f}")
    if show_v8_delta and show_v9_delta:
        v8_mean, _ = headline.get("v8", (float("nan"), float("nan")))
        v9_mean, _ = headline.get("v9", (float("nan"), float("nan")))
        if not np.isnan(v8_mean) and not np.isnan(v9_mean):
            lines.append(f"Δ (v9 − v8)       = {v9_mean - v8_mean:+.3f}")
    if show_v9_delta and show_qdflrql9_delta:
        v9_mean, _ = headline.get("v9", (float("nan"), float("nan")))
        qd_mean, _ = headline.get("qdflrql9", (float("nan"), float("nan")))
        if not np.isnan(v9_mean) and not np.isnan(qd_mean):
            lines.append(f"Δ (qdflrql9 − v9) = {qd_mean - v9_mean:+.3f}")
    if common_max_step:
        paired_n = sum(1 for m in metrics if m.get("common_step_n", 0) > 0)
        lines.append(f"paired tasks with common step: {paired_n}/50")
    summary = "\n".join(lines)
    ax.text(
        0.5,
        0.5,
        summary,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )

    # Env-name captions on both metric panels (summary panel is text-only).
    for ax_i in (0, 1):
        axes[ax_i].set_xticks(x)
        axes[ax_i].set_xticklabels(labels, rotation=80, ha="right", fontsize=6)
    axes[0].tick_params(labelbottom=False)
    title_methods = " vs ".join(label for label, _, _ in active_methods)
    fig.suptitle(
        f"OGBench 50-task {protocol_note}: {title_methods} "
        f"(seeds {seeds}, error bars = seed std)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18, hspace=0.35)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-dir",
        default=str(SCRIPT_DIR.parent / "exp"),
    )
    parser.add_argument("--seed", type=int, default=0, help="Single seed (legacy; use --seeds for multi-seed).")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds for multi-seed metrics plot (default: --seed only).",
    )
    parser.add_argument(
        "--report-step",
        type=int,
        default=1_000_000,
        help="Report final success at this training step (default: 1M).",
    )
    parser.add_argument(
        "--log-root",
        default=str(SCRIPT_DIR.parent / "exp" / "dflrql6_50"),
        help="V6 stdout logs from run_train_rql_dfl6_ogbench50.sh",
    )
    parser.add_argument(
        "--baseline-log-root",
        default=str(SCRIPT_DIR.parent / "exp" / "baseline_50"),
        help="Baseline stdout logs from run_train_rql_dfl6_ogbench50.sh",
    )
    parser.add_argument(
        "--output",
        default=str(SCRIPT_DIR.parent / "my_exps" / "ogbench50_dflrql6_vs_baseline.png"),
    )
    parser.add_argument(
        "--domain-output",
        default=str(SCRIPT_DIR.parent / "my_exps" / "ogbench50_dflrql6_vs_baseline_domains.png"),
    )
    parser.add_argument(
        "--tasks-output",
        default=str(SCRIPT_DIR.parent / "my_exps" / "ogbench50_dflrql6_vs_baseline_tasks.png"),
    )
    parser.add_argument(
        "--metrics-output",
        default=str(SCRIPT_DIR.parent / "my_exps" / "ogbench50_all50_metrics.png"),
    )
    parser.add_argument(
        "--csv-output",
        default=str(SCRIPT_DIR.parent / "my_exps" / "ogbench50_all50_metrics.csv"),
    )
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Only write the multi-seed all50 metrics plot/CSV (skip learning curves).",
    )
    parser.add_argument(
        "--paper-protocol",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Authoritative paper protocol: exact evaluation/success at --report-step "
            "(final checkpoint only). Use --last3-average for the superseded "
            "1.8M/1.9M/2.0M mean."
        ),
    )
    parser.add_argument(
        "--last3-average",
        action="store_true",
        help=(
            "RQL validation protocol: mean success over the last three "
            "100k-spaced checkpoints ending at --report-step "
            "(e.g. 800k/900k/1M or 1.8M/1.9M/2.0M)."
        ),
    )
    parser.add_argument(
        "--v7-log-root",
        default=str(SCRIPT_DIR.parent / "exp" / "dflrql7_50_2000000"),
        help="V7 stdout logs from run_train_rql_dfl7_ogbench50.sh",
    )
    parser.add_argument(
        "--include-v7",
        action="store_true",
        help="Include DFL-RQL v7 bars/deltas in the all50 metrics plot.",
    )
    parser.add_argument(
        "--v8-log-root",
        default=str(SCRIPT_DIR.parent / "exp" / "dflrql8_50_2000000"),
        help="V8 stdout logs from run_train_rql_dfl8_ogbench50.sh",
    )
    parser.add_argument(
        "--include-v8",
        action="store_true",
        help="Include DFL-RQL v8 bars/deltas in the all50 metrics plot.",
    )
    parser.add_argument(
        "--v9-log-root",
        default=str(SCRIPT_DIR.parent / "exp" / "dflrql9_50_2000000"),
        help="V9 stdout logs from run_train_rql_dfl9_ogbench50.sh",
    )
    parser.add_argument(
        "--include-v9",
        action="store_true",
        help="Include DFL-RQL v9 / ConsensusFlow bars/deltas in the all50 metrics plot.",
    )
    parser.add_argument(
        "--include-nocrf",
        action="store_true",
        help="Include DFL-RQL v9 no-CRF (ConsensusFlow without CRF) bars/deltas.",
    )
    parser.add_argument(
        "--qdflrql9-log-root",
        default=str(SCRIPT_DIR.parent / "exp" / "qdflrql9_50_2000000"),
        help="Quantized DFL-RQL v9 stdout logs from run_train_quantized_dflrql9_ogbench50.sh",
    )
    parser.add_argument(
        "--include-qdflrql9",
        action="store_true",
        help="Include Quantized DFL-RQL v9 bars/deltas in the all50 metrics plot.",
    )
    parser.add_argument(
        "--common-max-step",
        action="store_true",
        help=(
            "Fair interim compare: for each (task, seed), evaluate all methods "
            "at the maximum eval step present in every compared method."
        ),
    )
    parser.add_argument(
        "--compare-methods",
        nargs="+",
        default=None,
        choices=[
            "baseline",
            "dflrql6",
            "dflrql7",
            "dflrql8",
            "dflrql9",
            "dflrql9_nocrf",
            "qdflrql9",
        ],
        help="Methods to include when using --common-max-step.",
    )
    args = parser.parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    compare_methods = args.compare_methods
    if args.common_max_step and compare_methods is None:
        if args.include_qdflrql9:
            compare_methods = ["baseline", "qdflrql9"]
        elif args.include_nocrf:
            compare_methods = ["baseline", "dflrql9", "dflrql9_nocrf"]
        elif args.include_v9:
            compare_methods = ["baseline", "dflrql9"]
        elif args.include_v8:
            compare_methods = ["baseline", "dflrql8"]
        elif args.include_v7:
            compare_methods = ["baseline", "dflrql7"]
        else:
            compare_methods = ["baseline", "dflrql6"]
    include_v7 = args.include_v7 or (
        compare_methods is not None and "dflrql7" in compare_methods
    )
    include_v8 = args.include_v8 or (
        compare_methods is not None and "dflrql8" in compare_methods
    )
    include_v9 = args.include_v9 or (
        compare_methods is not None and "dflrql9" in compare_methods
    )
    include_nocrf = args.include_nocrf or (
        compare_methods is not None and "dflrql9_nocrf" in compare_methods
    )
    include_qdflrql9 = args.include_qdflrql9 or (
        compare_methods is not None and "qdflrql9" in compare_methods
    )

    tasks = expand_tasks()
    baseline_log_root = Path(args.baseline_log_root)
    v6_log_root = Path(args.log_root)
    log_roots = log_roots_for_seed(
        baseline_log_root,
        v6_log_root,
        args.seed,
        v7_log_root=Path(args.v7_log_root),
        v8_log_root=Path(args.v8_log_root),
        v9_log_root=Path(args.v9_log_root),
        qdflrql9_log_root=Path(args.qdflrql9_log_root),
    )
    index = build_result_index(args.save_dir, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.metrics_only:
        overall = plot_learning_curves(args.save_dir, tasks, args.seed, index, output_path)
        plot_domain_bars(
            args.save_dir,
            tasks,
            args.seed,
            index,
            Path(args.domain_output),
            log_roots,
            report_step=args.report_step,
        )
        plot_per_task_bars(
            args.save_dir,
            tasks,
            args.seed,
            index,
            Path(args.tasks_output),
            log_roots,
            report_step=args.report_step,
        )

    metrics = plot_all_tasks_metrics(
        args.save_dir,
        tasks,
        seeds,
        Path(args.metrics_output),
        baseline_log_root,
        v6_log_root,
        report_step=args.report_step,
        paper_protocol=False if args.common_max_step else args.paper_protocol,
        require_exact_step=not args.paper_protocol and not args.common_max_step,
        last3_average=args.last3_average,
        v7_log_root=Path(args.v7_log_root),
        include_v7=include_v7,
        v8_log_root=Path(args.v8_log_root),
        include_v8=include_v8,
        v9_log_root=Path(args.v9_log_root),
        include_v9=include_v9,
        include_nocrf=include_nocrf,
        qdflrql9_log_root=Path(args.qdflrql9_log_root),
        include_qdflrql9=include_qdflrql9,
        common_max_step=args.common_max_step,
        compare_methods=compare_methods,
    )

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics:
        # Drop internal summary keys from CSV.
        fieldnames = [k for k in metrics[0].keys() if not k.startswith("_")]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metrics)

    print(f"Wrote {args.metrics_output}")
    print(f"Wrote {csv_path}")
    if not args.metrics_only:
        print(f"Wrote {output_path}")
        print(f"Wrote {args.domain_output}")
        print(f"Wrote {args.tasks_output}")

    _, per_seed_agg = collect_multiseed_metrics(
        args.save_dir,
        tasks,
        seeds,
        baseline_log_root,
        v6_log_root,
        args.report_step,
        paper_protocol=False if args.common_max_step else args.paper_protocol,
        require_exact_step=not args.paper_protocol and not args.common_max_step,
        last3_average=args.last3_average,
        v7_log_root=Path(args.v7_log_root),
        include_v7=include_v7,
        v8_log_root=Path(args.v8_log_root),
        include_v8=include_v8,
        v9_log_root=Path(args.v9_log_root),
        include_v9=include_v9,
        include_nocrf=include_nocrf,
        qdflrql9_log_root=Path(args.qdflrql9_log_root),
        include_qdflrql9=include_qdflrql9,
        common_max_step=args.common_max_step,
        compare_methods=compare_methods,
    )
    headline = aggregate_headline_stats(per_seed_agg)
    base_mean, base_std = headline.get("baseline", (float("nan"), float("nan")))
    if args.common_max_step:
        step_vals = [
            m["common_step_mean"]
            for m in metrics
            if m.get("common_step_mean") is not None
        ]
        proto = (
            f"common_max≈{int(np.mean(step_vals)) // 1000}k"
            if step_vals
            else "common_max"
        )
    elif args.paper_protocol:
        proto = checkpoint_label(args.report_step, last3=args.last3_average)
    else:
        proto = str(args.report_step)
    parts = [f"baseline {base_mean:.3f}±{base_std:.3f}"]
    if "v6" in headline and len(per_seed_agg.get("v6", [])):
        v6_mean, v6_std = headline["v6"]
        parts.append(f"v6 {v6_mean:.3f}±{v6_std:.3f}")
        if not np.isnan(base_mean) and not np.isnan(v6_mean):
            parts.append(f"Δv6-base={v6_mean - base_mean:+.3f}")
    if include_v7 and "v7" in headline and len(per_seed_agg.get("v7", [])):
        v7_mean, v7_std = headline["v7"]
        parts.append(f"v7 {v7_mean:.3f}±{v7_std:.3f}")
        if not np.isnan(base_mean) and not np.isnan(v7_mean):
            parts.append(f"Δv7-base={v7_mean - base_mean:+.3f}")
    if include_v8 and "v8" in headline and len(per_seed_agg.get("v8", [])):
        v8_mean, v8_std = headline["v8"]
        parts.append(f"v8 {v8_mean:.3f}±{v8_std:.3f}")
        if not np.isnan(base_mean) and not np.isnan(v8_mean):
            parts.append(f"Δv8-base={v8_mean - base_mean:+.3f}")
    if include_v9 and "v9" in headline and len(per_seed_agg.get("v9", [])):
        v9_mean, v9_std = headline["v9"]
        parts.append(f"v9 {v9_mean:.3f}±{v9_std:.3f}")
        if not np.isnan(base_mean) and not np.isnan(v9_mean):
            parts.append(f"Δv9-base={v9_mean - base_mean:+.3f}")
    if include_nocrf and "v9_nocrf" in headline and len(per_seed_agg.get("v9_nocrf", [])):
        nocrf_mean, nocrf_std = headline["v9_nocrf"]
        parts.append(f"v9_nocrf {nocrf_mean:.3f}±{nocrf_std:.3f}")
        if not np.isnan(base_mean) and not np.isnan(nocrf_mean):
            parts.append(f"Δnocrf-base={nocrf_mean - base_mean:+.3f}")
    if include_qdflrql9 and "qdflrql9" in headline and len(per_seed_agg.get("qdflrql9", [])):
        qd_mean, qd_std = headline["qdflrql9"]
        parts.append(f"qdflrql9 {qd_mean:.3f}±{qd_std:.3f}")
        if not np.isnan(base_mean) and not np.isnan(qd_mean):
            parts.append(f"Δqd-base={qd_mean - base_mean:+.3f}")
    print(f"  Aggregate ({proto}): " + ", ".join(parts))


if __name__ == "__main__":
    main()
