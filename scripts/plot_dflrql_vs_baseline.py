#!/usr/bin/env python3
"""Plot humanoidmaze-large DFL-RQL variants vs RQL baseline.

Reads eval.csv under exp/rql/<run_group>/sd*/ and writes
my_exps/dflrql_vs_baseline.png (success + return, mean ± std over seeds).

V7 (singletask-v0), piecewise seed normalization:
  - steps in [0, 1M]: mean ± std over all 8 seeds from humanoidmaze-large-dflrql7
  - steps in (1M, 2M]: mean ± std over seeds 0–2 from humanoidmaze-large-dflrql7-2m
    (only where those runs have evals; no extrapolation)

V8 is read directly from humanoidmaze-large-dflrql8-2m and is never
extrapolated beyond the last available evaluation for each seed.

V9 is read directly from humanoidmaze-large-dflrql9-2m.

Default SERIES: baseline + v9 + ConsensusLatentFlow v4 + CDF v3 smokes
(historical categorical CDF kept distinguishable from continuous CLF).
Override CLF run group with --clf-run-group (launcher passes RQL_RUN_GROUP).
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_CLF_RUN_GROUP = "humanoidmaze-large-clf-v4-2m"
CLF_V4_COLOR = "#C2185B"

# (label, run_group, color, linestyle) — CLF group filled in at runtime.
_SERIES_HEAD = [
    ("RQL baseline", "humanoidmaze-large-rql-tuned-2m", "#4C72B0", "-"),
    ("DFL-RQL v9", "humanoidmaze-large-dflrql9-2m", "#0F766E", "-"),
]
_SERIES_CDF_TAIL = [
    (
        "CDF v3 guided",
        "humanoidmaze-large-cdf-v3-guided-smoke300k",
        "#D55E00",
        "-",
    ),
    (
        "CDF v3 unguided",
        "humanoidmaze-large-cdf-v3-unguided-smoke300k",
        "#E69F00",
        "--",
    ),
]

# Full historical series (use --all-methods); CLF inserted before CDF tail.
_SERIES_ALL_HEAD = [
    ("RQL baseline", "humanoidmaze-large-rql-tuned-2m", "#4C72B0", "-"),
    ("RQL baseline (1M)", "humanoidmaze-large-rql-tuned", "#4C72B0", ":"),
    ("DFL-RQL v1", "humanoidmaze-large-dflrql", "#55A868", "-"),
    ("DFL-RQL v2", "humanoidmaze-large-dflrql2", "#C44E52", "-"),
    ("DFL-RQL v3", "humanoidmaze-large-dflrql3", "#8172B2", "-"),
    ("DFL-RQL v4", "humanoidmaze-large-dflrql4", "#CCB974", "-"),
    ("DFL-RQL v5", "humanoidmaze-large-dflrql5", "#64B5CD", "-"),
    ("DFL-RQL v6", "humanoidmaze-large-dflrql6", "#8C8C8C", "--"),
    ("DFL-RQL v7", "__v7__", "#E24A33", "-"),
    ("DFL-RQL v8", "humanoidmaze-large-dflrql8-2m", "#7C3AED", "-"),
    ("DFL-RQL v9", "humanoidmaze-large-dflrql9-2m", "#0F766E", "-"),
    ("ConsensusDiscreteFlow v2", "humanoidmaze-large-cdf-2m-v2", "#999999", ":"),
]


def build_series(clf_run_group: str, all_methods: bool) -> list[tuple[str, str, str, str]]:
    clf = ("CLF v4", clf_run_group, CLF_V4_COLOR, "-")
    if all_methods:
        return [*_SERIES_ALL_HEAD, clf, *_SERIES_CDF_TAIL]
    return [*_SERIES_HEAD, clf, *_SERIES_CDF_TAIL]

V7_1M_GROUP = "humanoidmaze-large-dflrql7"
V7_2M_GROUP = "humanoidmaze-large-dflrql7-2m"
V7_SPLIT_STEP = 1_000_000
V7_2M_SEEDS = (0, 1, 2)


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
                        "return": float(row["evaluation/episode.return"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def seed_from_run_dir(run_dir: Path) -> int | None:
    m = re.match(r"sd(\d+)_", run_dir.name)
    return int(m.group(1)) if m else None


def load_group_by_seed(
    exp_root: Path, run_group: str
) -> dict[int, list[dict[str, float]]]:
    group_dir = exp_root / "rql" / run_group
    if not group_dir.is_dir():
        return {}
    out: dict[int, list[dict[str, float]]] = {}
    for run_dir in sorted(group_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        seed = seed_from_run_dir(run_dir)
        if seed is None:
            continue
        eval_csv = run_dir / "eval.csv"
        if not eval_csv.is_file():
            continue
        rows = parse_eval_csv(eval_csv)
        if not rows:
            continue
        prev = out.get(seed)
        if prev is None or max(r["step"] for r in rows) >= max(r["step"] for r in prev):
            out[seed] = rows
    return out


def load_group(exp_root: Path, run_group: str) -> list[list[dict[str, float]]]:
    by_seed = load_group_by_seed(exp_root, run_group)
    return [by_seed[s] for s in sorted(by_seed)]


def max_step(curve: list[dict[str, float]]) -> int:
    return max(int(r["step"]) for r in curve) if curve else 0


def aggregate(
    seed_curves: list[list[dict[str, float]]], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return steps, mean, std, n_per_step (no extrapolation past last eval)."""
    if not seed_curves:
        return None
    all_steps = sorted({row["step"] for curve in seed_curves for row in curve})
    if not all_steps:
        return None
    steps = np.asarray(all_steps, dtype=float)
    values = []
    for curve in seed_curves:
        xs = np.asarray([r["step"] for r in curve], dtype=float)
        ys = np.asarray([r[metric] for r in curve], dtype=float)
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        interp = np.interp(steps, xs, ys, left=np.nan, right=np.nan)
        values.append(interp)
    arr = np.vstack(values)
    mean = np.nanmean(arr, axis=0)
    # Sample std over seeds that have data at this step.
    std = np.nanstd(arr, axis=0, ddof=1)
    std = np.where(np.sum(~np.isnan(arr), axis=0) > 1, std, 0.0)
    n_per = np.sum(~np.isnan(arr), axis=0).astype(float)
    valid = ~np.isnan(mean)
    if not np.any(valid):
        return None
    return steps[valid], mean[valid], std[valid], n_per[valid]


def aggregate_v7_piecewise(
    exp_root: Path, metric: str, plot_max_step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    """0–1M over 8 seeds (1M group); 1M–2M over seeds 0–2 (2M group)."""
    c1 = load_group_by_seed(exp_root, V7_1M_GROUP)
    c2 = load_group_by_seed(exp_root, V7_2M_GROUP)
    curves_1m = [c1[s] for s in sorted(c1)]
    curves_2m = [
        c2[s]
        for s in V7_2M_SEEDS
        if s in c2 and max_step(c2[s]) > V7_SPLIT_STEP
    ]

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    n_lo = len(curves_1m)
    n_hi = len(curves_2m)

    if curves_1m:
        agg = aggregate(curves_1m, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (steps <= V7_SPLIT_STEP) & (steps <= plot_max_step)
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if curves_2m:
        agg = aggregate(curves_2m, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (steps > V7_SPLIT_STEP) & (steps <= plot_max_step)
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if not parts:
        return None
    steps = np.concatenate([p[0] for p in parts])
    mean = np.concatenate([p[1] for p in parts])
    std = np.concatenate([p[2] for p in parts])
    n_per = np.concatenate([p[3] for p in parts])
    return steps, mean, std, n_per, n_lo, n_hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "exp",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "my_exps" / "dflrql_vs_baseline.png",
    )
    parser.add_argument("--max-step", type=int, default=2_000_000)
    parser.add_argument(
        "--clf-run-group",
        type=str,
        default=DEFAULT_CLF_RUN_GROUP,
        help=(
            "Run group for ConsensusLatentFlow v4 "
            f"(default: {DEFAULT_CLF_RUN_GROUP})."
        ),
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Plot the full historical method suite instead of baseline/v9/CLF/CDF.",
    )
    args = parser.parse_args()

    series = build_series(args.clf_run_group, args.all_methods)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    plotted = 0
    has_2m_baseline = bool(load_group(args.save_dir, "humanoidmaze-large-rql-tuned-2m"))
    for label, run_group, color, ls in series:
        if label == "RQL baseline (1M)" and has_2m_baseline:
            continue
        if label == "RQL baseline" and not has_2m_baseline:
            continue

        if run_group == "__v7__":
            succ_agg = aggregate_v7_piecewise(
                args.save_dir, "success", args.max_step
            )
            if succ_agg is None:
                print("skip DFL-RQL v7: no runs")
                continue
            steps_s, mean_s, std_s, n_per_s, n_lo, n_hi = succ_agg
            legend = f"{label} (n={n_lo}@≤1M"
            legend += f", n={n_hi}@>1M)" if n_hi else ")"
            for metric, ax, ylabel in [
                ("success", axes[0], "Eval success"),
                ("return", axes[1], "Eval return"),
            ]:
                agg = (
                    succ_agg
                    if metric == "success"
                    else aggregate_v7_piecewise(
                        args.save_dir, metric, args.max_step
                    )
                )
                if agg is None:
                    continue
                steps, mean, std, n_per, _, _ = agg
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    linestyle=ls,
                    label=legend if metric == "success" else None,
                )
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
                ax.set_xlabel("Offline steps")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                if args.max_step > V7_SPLIT_STEP:
                    ax.axvline(
                        V7_SPLIT_STEP, color="#bbbbbb", linestyle=":", linewidth=1
                    )
            plotted += 1
            idx_1m = int(np.argmin(np.abs(steps_s - V7_SPLIT_STEP)))
            print(
                f"{label}: n={int(n_per_s[idx_1m])}/{n_lo} success@{int(steps_s[idx_1m])}"
                f"={mean_s[idx_1m]:.3f}±{std_s[idx_1m]:.3f}"
            )
            if n_hi and float(steps_s[-1]) > V7_SPLIT_STEP:
                print(
                    f"{label}: n={int(n_per_s[-1])}/{n_hi} success@{int(steps_s[-1])}"
                    f"={mean_s[-1]:.3f}±{std_s[-1]:.3f}"
                )
            continue

        curves = load_group(args.save_dir, run_group)
        if not curves:
            print(f"skip {label}: no runs in {run_group}")
            continue
        for metric, ax, ylabel in [
            ("success", axes[0], "Eval success"),
            ("return", axes[1], "Eval return"),
        ]:
            agg = aggregate(curves, metric)
            if agg is None:
                continue
            steps, mean, std, n_per = agg
            mask = steps <= args.max_step
            steps, mean, std, n_per = (
                steps[mask],
                mean[mask],
                std[mask],
                n_per[mask],
            )
            n_at_end = int(n_per[-1]) if len(n_per) else 0
            ax.plot(
                steps,
                mean,
                color=color,
                linestyle=ls,
                label=f"{label} (n={n_at_end}/{len(curves)})",
            )
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
            ax.set_xlabel("Offline steps")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            if args.max_step > V7_SPLIT_STEP:
                ax.axvline(V7_SPLIT_STEP, color="#bbbbbb", linestyle=":", linewidth=1)
        plotted += 1
        final = aggregate(curves, "success")
        if final is not None:
            steps, mean, std, n_per = final
            target = min(args.max_step, float(steps[-1]))
            idx = int(np.argmin(np.abs(steps - target)))
            print(
                f"{label}: n={int(n_per[idx])}/{len(curves)} success@{int(steps[idx])}"
                f"={mean[idx]:.3f}±{std[idx]:.3f}"
                f" (data to {int(max(max_step(c) for c in curves))})"
            )

    if plotted == 0:
        raise SystemExit("No series found to plot")

    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("humanoidmaze-large-navigate-singletask-v0")
    axes[1].set_title("Episode return")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
