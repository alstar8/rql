#!/usr/bin/env python3
"""Plot offline→online Q-Flow transfer curves (success + return, mean ± std).

Writes my_exps/offline_to_online.png and a companion aggregate/provenance JSON.
Absolute step axes; no extrapolation. Tolerates missing / incomplete phases and
plots whatever eval data exists.

Protocol: 1M offline (phase 1) + 1M online (phase 2). X-axis ends at 2M.
Single phase marker at 1M.

Default series:
  1. RQL                     offline-1m (≤1M) + online-2m (>1M)
  2. CF                      offline scratch ≤1M + online-2m (>1M)
  3. CF no-CRF               offline-1m (≤1M) + online-2m (>1M)
  4. Q-Flow RQL warmstart   warmstart-2m (≤1M) + online-2m (>1M)
  5. Pure Q-Flow            offline-1m (≤1M) + online-2m (>1M)
  6. AR-QDFL + FastSAC      single group (absolute 0..2M; warmup hidden)

The legacy RQL→Q-Flow 2M+2M arm is omitted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_dflrql_vs_baseline import (
    AR_QDFL_FASTSAC_PLACEHOLDER,
    AR_QDFL_FASTSAC_WARMUP_POLICY,
    CF_NOCRF_PLACEHOLDER,
    CF_PLACEHOLDER,
    DEFAULT_AR_QDFL_FASTSAC_GROUP,
    DEFAULT_CF_NOCRF_PHASE1_GROUP,
    DEFAULT_CF_NOCRF_PHASE2_GROUP,
    DEFAULT_CF_PHASE1_GROUP,
    DEFAULT_CF_PHASE2_GROUP,
    DEFAULT_ONLINE_PLOT_MAX_STEP,
    DEFAULT_PURE_QFLOW_PHASE1_GROUP,
    DEFAULT_PURE_QFLOW_PHASE2_GROUP,
    DEFAULT_QFLOW_V2_PHASE1_GROUP,
    DEFAULT_QFLOW_V2_PHASE2_GROUP,
    DEFAULT_RQL_ONLINE_PHASE1_GROUP,
    DEFAULT_RQL_ONLINE_PHASE2_GROUP,
    ONLINE_PHASE_SPLIT_STEP,
    PURE_QFLOW_PLACEHOLDER,
    QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
    RQL_ONLINE_PLACEHOLDER,
    aggregate_single_group,
    aggregate_two_phase_piecewise,
    curve_arrays_to_lists,
    per_seed_endpoints,
)

ONLINE_SERIES = [
    (
        "RQL",
        RQL_ONLINE_PLACEHOLDER,
        "#4C72B0",
        "-",
    ),
    (
        "CF",
        CF_PLACEHOLDER,
        "#0F766E",
        "-",
    ),
    (
        "CF no-CRF",
        CF_NOCRF_PLACEHOLDER,
        "#D55E00",
        "-",
    ),
    (
        "Q-Flow RQL warmstart",
        QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
        "#134E4A",
        "-",
    ),
    (
        "Pure Q-Flow",
        PURE_QFLOW_PLACEHOLDER,
        "#BE123C",
        "-",
    ),
    (
        "AR-QDFL + FastSAC",
        AR_QDFL_FASTSAC_PLACEHOLDER,
        "#7C3AED",
        "-",
    ),
]

_WARMUP_NONE = {
    "mode": "none",
    "included_in_eval_csv": True,
    "note": "two_phase_piecewise_join; no hidden warmup on plot axis",
}


def build_online_series() -> list[tuple[str, str, str, str]]:
    """Default offline→online series (1M+1M absolute-step piecewise joins)."""
    return list(ONLINE_SERIES)


def _metric_block(
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_per: np.ndarray,
) -> dict[str, list[float]]:
    return curve_arrays_to_lists(steps, mean, std, n_per)


def build_series_provenance(
    *,
    label: str,
    placeholder: str,
    exp_root: Path,
    max_step: int,
    aggregation: str,
    warmup_policy: dict[str, Any],
    run_groups: dict[str, str],
    split_step: int | None = None,
) -> dict[str, Any]:
    """Machine-readable aggregate entry; omit invented unfinished-run values."""
    entry: dict[str, Any] = {
        "label": label,
        "placeholder": placeholder,
        "aggregation": aggregation,
        "run_groups": run_groups,
        "warmup_policy": warmup_policy,
        "split_step": split_step,
        "plotted": False,
        "n_seeds": {},
        "per_seed_endpoints": {},
        "success": None,
        "return": None,
    }

    if aggregation == "single_group":
        group = run_groups["run_group"]
        entry["per_seed_endpoints"] = {
            group: per_seed_endpoints(exp_root, group, max_step)
        }
        succ = aggregate_single_group(exp_root, "success", max_step, group)
        ret = aggregate_single_group(exp_root, "return", max_step, group)
        if succ is None:
            return entry
        steps_s, mean_s, std_s, n_per_s, n_seeds = succ
        entry["n_seeds"] = {"run_group": int(n_seeds)}
        entry["success"] = _metric_block(steps_s, mean_s, std_s, n_per_s)
        if ret is not None:
            steps_r, mean_r, std_r, n_per_r, _ = ret
            entry["return"] = _metric_block(steps_r, mean_r, std_r, n_per_r)
        entry["plotted"] = True
        return entry

    # two_phase_piecewise
    p1 = run_groups["phase1"]
    p2 = run_groups["phase2"]
    assert split_step is not None
    entry["per_seed_endpoints"] = {
        "phase1": per_seed_endpoints(exp_root, p1, min(max_step, split_step)),
        "phase2": per_seed_endpoints(exp_root, p2, max_step),
    }
    succ = aggregate_two_phase_piecewise(
        exp_root, "success", max_step, p1, p2, split_step
    )
    ret = aggregate_two_phase_piecewise(
        exp_root, "return", max_step, p1, p2, split_step
    )
    if succ is None:
        return entry
    steps_s, mean_s, std_s, n_per_s, n_lo, n_hi = succ
    entry["n_seeds"] = {"phase1": int(n_lo), "phase2": int(n_hi)}
    entry["success"] = _metric_block(steps_s, mean_s, std_s, n_per_s)
    if ret is not None:
        steps_r, mean_r, std_r, n_per_r, _, _ = ret
        entry["return"] = _metric_block(steps_r, mean_r, std_r, n_per_r)
    entry["plotted"] = True
    return entry


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
        default=(
            Path(__file__).resolve().parents[1] / "my_exps" / "offline_to_online.png"
        ),
    )
    parser.add_argument(
        "--aggregate-json",
        type=Path,
        default=None,
        help=(
            "Provenance/aggregate JSON path "
            "(default: sibling of --out with _aggregate.json)."
        ),
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=DEFAULT_ONLINE_PLOT_MAX_STEP,
        help=(
            "X-axis / filter max step "
            f"(default: {DEFAULT_ONLINE_PLOT_MAX_STEP})."
        ),
    )
    parser.add_argument(
        "--rql-phase1-run-group",
        type=str,
        default=DEFAULT_RQL_ONLINE_PHASE1_GROUP,
    )
    parser.add_argument(
        "--rql-phase2-run-group",
        type=str,
        default=DEFAULT_RQL_ONLINE_PHASE2_GROUP,
    )
    parser.add_argument(
        "--qflow-v2-phase1-run-group",
        type=str,
        default=DEFAULT_QFLOW_V2_PHASE1_GROUP,
    )
    parser.add_argument(
        "--qflow-v2-phase2-run-group",
        type=str,
        default=DEFAULT_QFLOW_V2_PHASE2_GROUP,
    )
    parser.add_argument(
        "--pure-qflow-phase1-run-group",
        type=str,
        default=DEFAULT_PURE_QFLOW_PHASE1_GROUP,
    )
    parser.add_argument(
        "--pure-qflow-phase2-run-group",
        type=str,
        default=DEFAULT_PURE_QFLOW_PHASE2_GROUP,
    )
    parser.add_argument(
        "--cf-phase1-run-group",
        type=str,
        default=DEFAULT_CF_PHASE1_GROUP,
    )
    parser.add_argument(
        "--cf-phase2-run-group",
        type=str,
        default=DEFAULT_CF_PHASE2_GROUP,
    )
    parser.add_argument(
        "--cf-nocrf-phase1-run-group",
        type=str,
        default=DEFAULT_CF_NOCRF_PHASE1_GROUP,
    )
    parser.add_argument(
        "--cf-nocrf-phase2-run-group",
        type=str,
        default=DEFAULT_CF_NOCRF_PHASE2_GROUP,
    )
    parser.add_argument(
        "--ar-qdfl-fastsac-run-group",
        type=str,
        default=DEFAULT_AR_QDFL_FASTSAC_GROUP,
    )
    parser.add_argument(
        "--title",
        type=str,
        default="humanoidmaze-large-navigate-singletask-v0",
        help="Left-panel title (env name).",
    )
    args = parser.parse_args()

    series = build_online_series()
    two_phase = {
        RQL_ONLINE_PLACEHOLDER: (
            args.rql_phase1_run_group,
            args.rql_phase2_run_group,
            ONLINE_PHASE_SPLIT_STEP,
        ),
        CF_PLACEHOLDER: (
            args.cf_phase1_run_group,
            args.cf_phase2_run_group,
            ONLINE_PHASE_SPLIT_STEP,
        ),
        CF_NOCRF_PLACEHOLDER: (
            args.cf_nocrf_phase1_run_group,
            args.cf_nocrf_phase2_run_group,
            ONLINE_PHASE_SPLIT_STEP,
        ),
        QFLOW_RQL_WARMSTART_V2_PLACEHOLDER: (
            args.qflow_v2_phase1_run_group,
            args.qflow_v2_phase2_run_group,
            ONLINE_PHASE_SPLIT_STEP,
        ),
        PURE_QFLOW_PLACEHOLDER: (
            args.pure_qflow_phase1_run_group,
            args.pure_qflow_phase2_run_group,
            ONLINE_PHASE_SPLIT_STEP,
        ),
    }
    single_group = {
        AR_QDFL_FASTSAC_PLACEHOLDER: args.ar_qdfl_fastsac_run_group,
    }
    warmup_by_placeholder = {
        RQL_ONLINE_PLACEHOLDER: dict(_WARMUP_NONE),
        CF_PLACEHOLDER: dict(_WARMUP_NONE),
        CF_NOCRF_PLACEHOLDER: dict(_WARMUP_NONE),
        QFLOW_RQL_WARMSTART_V2_PLACEHOLDER: dict(_WARMUP_NONE),
        PURE_QFLOW_PLACEHOLDER: dict(_WARMUP_NONE),
        AR_QDFL_FASTSAC_PLACEHOLDER: dict(AR_QDFL_FASTSAC_WARMUP_POLICY),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    plotted = 0
    phase_marker_drawn = False
    provenance_series: list[dict[str, Any]] = []

    def draw_markers(ax: plt.Axes) -> None:
        nonlocal phase_marker_drawn
        if args.max_step >= ONLINE_PHASE_SPLIT_STEP:
            ax.axvline(
                ONLINE_PHASE_SPLIT_STEP,
                color="#444444",
                linestyle=":",
                linewidth=1.2,
                label="online @1M" if not phase_marker_drawn else None,
            )
            phase_marker_drawn = True

    for label, run_group, color, ls in series:
        if run_group in single_group:
            group_name = single_group[run_group]
            prov = build_series_provenance(
                label=label,
                placeholder=run_group,
                exp_root=args.save_dir,
                max_step=args.max_step,
                aggregation="single_group",
                warmup_policy=warmup_by_placeholder[run_group],
                run_groups={"run_group": group_name},
            )
            provenance_series.append(prov)
            succ_agg = aggregate_single_group(
                args.save_dir, "success", args.max_step, group_name
            )
            if succ_agg is None:
                print(f"skip {label}: no runs in {group_name}")
                continue
            steps_s, mean_s, std_s, n_per_s, n_seeds = succ_agg
            legend = f"{label} (n={n_seeds})"
            for metric, ax, ylabel in [
                ("success", axes[0], "Eval success"),
                ("return", axes[1], "Eval return"),
            ]:
                agg = (
                    succ_agg
                    if metric == "success"
                    else aggregate_single_group(
                        args.save_dir, metric, args.max_step, group_name
                    )
                )
                if agg is None:
                    continue
                steps, mean, std, n_per, _ = agg
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    linestyle=ls,
                    label=legend if metric == "success" else None,
                )
                lo, hi = mean - std, mean + std
                if metric == "success":
                    lo = np.clip(lo, 0.0, 1.0)
                    hi = np.clip(hi, 0.0, 1.0)
                ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
                ax.set_xlabel("Training steps")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                draw_markers(ax)
            plotted += 1
            print(
                f"{label}: n={int(n_per_s[-1])}/{n_seeds} "
                f"success@{int(steps_s[-1])}"
                f"={mean_s[-1]:.3f}±{std_s[-1]:.3f}"
                f" (partial to {int(steps_s[-1])})"
            )
            continue

        if run_group not in two_phase:
            print(f"skip {label}: unknown series {run_group}")
            continue
        phase1_group, phase2_group, split_step = two_phase[run_group]
        prov = build_series_provenance(
            label=label,
            placeholder=run_group,
            exp_root=args.save_dir,
            max_step=args.max_step,
            aggregation="two_phase_piecewise",
            warmup_policy=warmup_by_placeholder[run_group],
            run_groups={"phase1": phase1_group, "phase2": phase2_group},
            split_step=split_step,
        )
        provenance_series.append(prov)
        succ_agg = aggregate_two_phase_piecewise(
            args.save_dir,
            "success",
            args.max_step,
            phase1_group,
            phase2_group,
            split_step,
        )
        if succ_agg is None:
            print(f"skip {label}: no runs in {phase1_group} / {phase2_group}")
            continue
        steps_s, mean_s, std_s, n_per_s, n_lo, n_hi = succ_agg
        split_m = split_step // 1_000_000
        legend = f"{label} (n={n_lo}@≤{split_m}M"
        legend += f", n={n_hi}@>{split_m}M)" if n_hi else ")"
        for metric, ax, ylabel in [
            ("success", axes[0], "Eval success"),
            ("return", axes[1], "Eval return"),
        ]:
            agg = (
                succ_agg
                if metric == "success"
                else aggregate_two_phase_piecewise(
                    args.save_dir,
                    metric,
                    args.max_step,
                    phase1_group,
                    phase2_group,
                    split_step,
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
            lo, hi = mean - std, mean + std
            # Success is in [0, 1]; clip the ±std band so high variance
            # cannot draw above 100% / below 0%.
            if metric == "success":
                lo = np.clip(lo, 0.0, 1.0)
                hi = np.clip(hi, 0.0, 1.0)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.15)
            ax.set_xlabel("Training steps")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            draw_markers(ax)
        plotted += 1
        idx_split = int(
            np.argmin(np.abs(steps_s - min(float(steps_s[-1]), float(split_step))))
        )
        print(
            f"{label}: n={int(n_per_s[idx_split])}/{n_lo} "
            f"success@{int(steps_s[idx_split])}"
            f"={mean_s[idx_split]:.3f}±{std_s[idx_split]:.3f}"
            f" (partial to {int(steps_s[-1])})"
        )
        if n_hi and float(steps_s[-1]) > split_step:
            print(
                f"{label}: n={int(n_per_s[-1])}/{n_hi} success@{int(steps_s[-1])}"
                f"={mean_s[-1]:.3f}±{std_s[-1]:.3f}"
            )

    agg_path = args.aggregate_json
    if agg_path is None:
        agg_path = args.out.with_name(args.out.stem + "_aggregate.json")
    payload = {
        "plot": "offline_to_online",
        "title": args.title,
        "save_dir": str(args.save_dir),
        "out": str(args.out),
        "max_step": int(args.max_step),
        "phase_split_step": int(ONLINE_PHASE_SPLIT_STEP),
        "series": provenance_series,
        "notes": [
            "No extrapolation past each seed's last eval.csv row.",
            "Unfinished seeds without eval rows are omitted from per_seed_endpoints.",
            "AR-QDFL FastSAC warmup is hidden from eval.csv (absolute 0..2M axis).",
        ],
    }
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    print(f"wrote {agg_path}")

    if plotted == 0:
        raise SystemExit("No series found to plot")

    for ax in axes:
        ax.set_xlim(0, args.max_step)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title(args.title)
    axes[1].set_title("Episode return")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
