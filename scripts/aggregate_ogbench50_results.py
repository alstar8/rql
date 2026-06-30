#!/usr/bin/env python3
"""Aggregate OGBench 50-task benchmark results and plot Figure 5-style charts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ogbench50_config import (  # noqa: E402
    BASELINES_PATH,
    OGBench50Task,
    expand_tasks,
    exp_root_for_task,
    load_tasks_config,
)

REPORT_STEPS = (800_000, 900_000, 1_000_000)


def parse_eval_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "step": int(float(row["step"])),
                    "success": float(row["evaluation/success"]),
                }
            )
    return rows


def last_three_success(rows: list[dict[str, float]]) -> float | None:
    by_step = {row["step"]: row["success"] for row in rows}
    values = [by_step.get(step) for step in REPORT_STEPS]
    if any(v is None for v in values):
        return None
    return float(np.mean(values)) * 100.0


def find_run_dirs(exp_root: Path, seed: int) -> list[Path]:
    return sorted(exp_root.glob(f"sd{seed:03d}_*"))


def collect_results(save_dir: str, tasks: list[OGBench50Task], seeds: list[int]) -> dict:
    per_run: list[dict] = []
    for task in tasks:
        exp_root = exp_root_for_task(save_dir, task)
        for seed in seeds:
            success_pct = None
            run_dir = None
            for candidate in find_run_dirs(exp_root, seed):
                eval_csv = candidate / "eval.csv"
                if not eval_csv.is_file():
                    continue
                success_pct = last_three_success(parse_eval_csv(eval_csv))
                if success_pct is not None:
                    run_dir = str(candidate)
                    break
            per_run.append(
                {
                    "task_id": task.task_id,
                    "domain_id": task.domain_id,
                    "task_num": task.task_num,
                    "env_name": task.env_name,
                    "seed": seed,
                    "success_pct": success_pct,
                    "run_dir": run_dir,
                }
            )

    by_task: dict[str, list[float]] = defaultdict(list)
    by_domain: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []
    for row in per_run:
        if row["success_pct"] is None:
            continue
        by_task[row["task_id"]].append(row["success_pct"])
        by_domain[row["domain_id"]].append(row["success_pct"])
        all_values.append(row["success_pct"])

    task_mean = {k: float(np.mean(v)) for k, v in by_task.items()}
    domain_mean = {k: float(np.mean(v)) for k, v in by_domain.items()}
    overall = float(np.mean(all_values)) if all_values else None

    completed = sum(1 for row in per_run if row["success_pct"] is not None)
    total = len(per_run)

    return {
        "per_run": per_run,
        "task_mean": task_mean,
        "domain_mean": domain_mean,
        "overall_mean": overall,
        "completed_runs": completed,
        "total_runs": total,
    }


def load_paper_baselines(path: Path | None = None) -> dict:
    with open(path or BASELINES_PATH) as f:
        return json.load(f)


def plot_figure5(
    domain_mean: dict[str, float],
    output_path: Path,
    method_label: str = "RQL (ours)",
    overlay_methods: list[str] | None = None,
) -> None:
    baselines = load_paper_baselines()
    domain_ids = [d["id"] for d in load_tasks_config()["domains"]]
    overlay_methods = overlay_methods or ["RQL (paper)", "TFQL", "FQL"]

    x = np.arange(len(domain_ids))
    series: list[tuple[str, list[float | None]]] = []
    for method in overlay_methods:
        if method.startswith("RQL (paper"):
            key = "RQL"
        else:
            key = method
        values = [
            baselines["domain_agg_percent"].get(domain_id, {}).get(key)
            for domain_id in domain_ids
        ]
        series.append((method, values))
    if domain_mean:
        series.append(
            (
                method_label,
                [domain_mean.get(domain_id) for domain_id in domain_ids],
            )
        )

    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (label, values) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label)
        if label == method_label:
            for bar in bars:
                bar.set_hatch("//")
                bar.set_edgecolor("black")

    ax.set_xticks(x)
    ax.set_xticklabels(domain_ids, rotation=35, ha="right")
    ax.set_ylabel("Success rate (%)")
    ax.set_title("OGBench 50-task benchmark (domain aggregates)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_overall_bar(
    overall_mean: float | None,
    output_path: Path,
    method_label: str = "RQL (ours)",
    top_n: int = 10,
) -> None:
    baselines = load_paper_baselines()
    methods = baselines["methods"]
    sorted_methods = sorted(methods.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels = [m for m, _ in sorted_methods]
    values = [v for _, v in sorted_methods]
    if overall_mean is not None:
        labels.append(method_label)
        values.append(overall_mean)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["C0" if lbl != method_label else "C3" for lbl in labels]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Success rate (%) on 50 tasks")
    ax.set_title("OGBench 50-task aggregate (paper Figure 5 style)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-dir",
        default=str(SCRIPT_DIR.parent / "exp"),
        help="RQL save_dir root (contains rql/<run_group>/...).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR.parent / "benchmarks" / "results"),
        help="Directory for CSV/JSON/plots.",
    )
    parser.add_argument("--method-label", default="RQL (ours)")
    args = parser.parse_args()

    tasks = expand_tasks()
    results = collect_results(args.save_dir, tasks, args.seeds)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "ogbench50_per_run.json").open("w") as f:
        json.dump(results["per_run"], f, indent=2)

    summary = {
        "overall_mean": results["overall_mean"],
        "domain_mean": results["domain_mean"],
        "task_mean": results["task_mean"],
        "completed_runs": results["completed_runs"],
        "total_runs": results["total_runs"],
        "report_steps": list(REPORT_STEPS),
        "seeds": args.seeds,
    }
    with (out_dir / "ogbench50_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "ogbench50_per_run.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "domain_id",
                "task_num",
                "env_name",
                "seed",
                "success_pct",
                "run_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(results["per_run"])

    print(
        f"Completed {results['completed_runs']}/{results['total_runs']} runs. "
        f"Overall mean success: {results['overall_mean']}"
    )
    print("Domain aggregates:")
    for domain_id in [d["id"] for d in load_tasks_config()["domains"]]:
        val = results["domain_mean"].get(domain_id)
        print(f"  {domain_id}: {val if val is not None else 'pending'}")

    if results["domain_mean"]:
        plot_figure5(
            results["domain_mean"],
            out_dir / "figure5_domains.png",
            method_label=args.method_label,
        )
    if results["overall_mean"] is not None:
        plot_overall_bar(
            results["overall_mean"],
            out_dir / "figure5_overall.png",
            method_label=args.method_label,
        )
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
