#!/usr/bin/env python3
"""Plot offline→online Q-Flow transfer curves (success + return, mean ± std).

Writes my_exps/offline_to_online.png. Absolute step axes; no extrapolation.
Tolerates missing / incomplete phases and plots whatever eval data exists.

Default series:
  1. Legacy RQL→Q-Flow     ready-2m (≤2M) + online-4m (>2M)
  2. Actor-freeze 100k     ready-2m (≤2M) + actorfreeze-2p1m (>2M)
  3. Q-Flow RQL warmstart v2  warmstart (≤2M) + bridge (2M, 2.1M] + online (>2.1M)
  4. Pure Q-Flow           offline-1m (≤1M) + online-2m (>1M)

Phase markers: pure online @1M, legacy/v2 bridge @2M, v2 online @2.1M.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_dflrql_vs_baseline import (
    BRIDGE_END_STEP,
    DEFAULT_ONLINE_PLOT_MAX_STEP,
    DEFAULT_PURE_QFLOW_PHASE1_GROUP,
    DEFAULT_PURE_QFLOW_PHASE2_GROUP,
    DEFAULT_QFLOW_V2_PHASE1_GROUP,
    DEFAULT_QFLOW_V2_PHASE2_GROUP,
    DEFAULT_QFLOW_V2_PHASE3_GROUP,
    DEFAULT_RQL_QFLOW_ACTORFREEZE_GROUP,
    DEFAULT_RQL_QFLOW_PHASE1_GROUP,
    DEFAULT_RQL_QFLOW_PHASE2_GROUP,
    ONLINE_START_STEP,
    PURE_QFLOW_ONLINE_START_STEP,
    PURE_QFLOW_PLACEHOLDER,
    QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
    RQL_QFLOW_ACTORFREEZE_PLACEHOLDER,
    RQL_QFLOW_ONLINE_PLACEHOLDER,
    WARMSTART_END_STEP,
    aggregate_three_phase_piecewise,
    aggregate_two_phase_piecewise,
)

ONLINE_SERIES = [
    (
        "RQL→Q-Flow online",
        RQL_QFLOW_ONLINE_PLACEHOLDER,
        "#7C2D12",
        "-",
    ),
    (
        "RQL→Q-Flow actor-freeze",
        RQL_QFLOW_ACTORFREEZE_PLACEHOLDER,
        "#A16207",
        "--",
    ),
    (
        "Q-Flow RQL warmstart v2",
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
]


def build_online_series() -> list[tuple[str, str, str, str]]:
    """Default offline→online series (absolute-step piecewise joins)."""
    return list(ONLINE_SERIES)


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
        "--max-step",
        type=int,
        default=DEFAULT_ONLINE_PLOT_MAX_STEP,
        help=(
            "X-axis / filter max step "
            f"(default: {DEFAULT_ONLINE_PLOT_MAX_STEP})."
        ),
    )
    parser.add_argument(
        "--rql-qflow-phase1-run-group",
        type=str,
        default=DEFAULT_RQL_QFLOW_PHASE1_GROUP,
    )
    parser.add_argument(
        "--rql-qflow-phase2-run-group",
        type=str,
        default=DEFAULT_RQL_QFLOW_PHASE2_GROUP,
    )
    parser.add_argument(
        "--rql-qflow-actorfreeze-run-group",
        type=str,
        default=DEFAULT_RQL_QFLOW_ACTORFREEZE_GROUP,
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
        "--qflow-v2-phase3-run-group",
        type=str,
        default=DEFAULT_QFLOW_V2_PHASE3_GROUP,
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
        "--no-actor-freeze",
        action="store_true",
        help="Omit the actor-freeze 100k diagnostic series.",
    )
    args = parser.parse_args()

    series = build_online_series()
    if args.no_actor_freeze:
        series = [
            s for s in series if s[1] != RQL_QFLOW_ACTORFREEZE_PLACEHOLDER
        ]

    two_phase = {
        RQL_QFLOW_ONLINE_PLACEHOLDER: (
            args.rql_qflow_phase1_run_group,
            args.rql_qflow_phase2_run_group,
            ONLINE_START_STEP,
        ),
        RQL_QFLOW_ACTORFREEZE_PLACEHOLDER: (
            args.rql_qflow_phase1_run_group,
            args.rql_qflow_actorfreeze_run_group,
            ONLINE_START_STEP,
        ),
        PURE_QFLOW_PLACEHOLDER: (
            args.pure_qflow_phase1_run_group,
            args.pure_qflow_phase2_run_group,
            PURE_QFLOW_ONLINE_START_STEP,
        ),
    }
    qflow_v2_groups = (
        args.qflow_v2_phase1_run_group,
        args.qflow_v2_phase2_run_group,
        args.qflow_v2_phase3_run_group,
    )
    series_groups = {run_group for _, run_group, _, _ in series}
    draw_online_start = bool(
        series_groups
        & {RQL_QFLOW_ONLINE_PLACEHOLDER, RQL_QFLOW_ACTORFREEZE_PLACEHOLDER}
    )
    draw_qflow_v2_markers = QFLOW_RQL_WARMSTART_V2_PLACEHOLDER in series_groups
    draw_pure_qflow_marker = PURE_QFLOW_PLACEHOLDER in series_groups

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    plotted = 0
    online_marker_drawn = False
    bridge_marker_drawn = False
    v2_online_marker_drawn = False
    pure_qflow_marker_drawn = False

    def draw_markers(ax: plt.Axes) -> None:
        nonlocal online_marker_drawn, bridge_marker_drawn
        nonlocal v2_online_marker_drawn, pure_qflow_marker_drawn
        if draw_pure_qflow_marker and args.max_step >= PURE_QFLOW_ONLINE_START_STEP:
            ax.axvline(
                PURE_QFLOW_ONLINE_START_STEP,
                color="#BE123C",
                linestyle=":",
                linewidth=1.2,
                label=(
                    "pure Q-Flow online @1M"
                    if not pure_qflow_marker_drawn
                    else None
                ),
            )
            pure_qflow_marker_drawn = True
        if draw_online_start and args.max_step >= ONLINE_START_STEP:
            ax.axvline(
                ONLINE_START_STEP,
                color="#7C2D12",
                linestyle="-.",
                linewidth=1.2,
                label="legacy online @2M" if not online_marker_drawn else None,
            )
            online_marker_drawn = True
        if draw_qflow_v2_markers and args.max_step >= WARMSTART_END_STEP:
            # Share 2M with legacy when both present; still label v2 bridge.
            if not draw_online_start:
                ax.axvline(
                    WARMSTART_END_STEP,
                    color="#134E4A",
                    linestyle="--",
                    linewidth=1.2,
                    label="v2 bridge @2M" if not bridge_marker_drawn else None,
                )
                bridge_marker_drawn = True
            elif not bridge_marker_drawn:
                # Legacy already drew @2M; add a legend-only note for v2.
                ax.plot(
                    [],
                    [],
                    color="#134E4A",
                    linestyle="--",
                    linewidth=1.2,
                    label="v2 bridge @2M (same)",
                )
                bridge_marker_drawn = True
        if draw_qflow_v2_markers and args.max_step >= BRIDGE_END_STEP:
            ax.axvline(
                BRIDGE_END_STEP,
                color="#134E4A",
                linestyle="-.",
                linewidth=1.2,
                label="v2 online @2.1M" if not v2_online_marker_drawn else None,
            )
            v2_online_marker_drawn = True

    for label, run_group, color, ls in series:
        if run_group == QFLOW_RQL_WARMSTART_V2_PLACEHOLDER:
            p1_g, p2_g, p3_g = qflow_v2_groups
            succ_agg = aggregate_three_phase_piecewise(
                args.save_dir,
                "success",
                args.max_step,
                p1_g,
                p2_g,
                p3_g,
                WARMSTART_END_STEP,
                BRIDGE_END_STEP,
            )
            if succ_agg is None:
                print(f"skip {label}: no runs in {p1_g} / {p2_g} / {p3_g}")
                continue
            steps_s, mean_s, std_s, n_per_s, n_lo, n_mid, n_hi = succ_agg
            legend = (
                f"{label} (n={n_lo}@≤2M, n={n_mid}@≤2.1M"
                f"{f', n={n_hi}@>2.1M' if n_hi else ''})"
            )
            for metric, ax, ylabel in [
                ("success", axes[0], "Eval success"),
                ("return", axes[1], "Eval return"),
            ]:
                agg = (
                    succ_agg
                    if metric == "success"
                    else aggregate_three_phase_piecewise(
                        args.save_dir,
                        metric,
                        args.max_step,
                        p1_g,
                        p2_g,
                        p3_g,
                        WARMSTART_END_STEP,
                        BRIDGE_END_STEP,
                    )
                )
                if agg is None:
                    continue
                steps, mean, std, n_per, _, _, _ = agg
                ax.plot(
                    steps,
                    mean,
                    color=color,
                    linestyle=ls,
                    label=legend if metric == "success" else None,
                )
                ax.fill_between(
                    steps, mean - std, mean + std, color=color, alpha=0.15
                )
                ax.set_xlabel("Training steps")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                draw_markers(ax)
            plotted += 1
            idx_ws = int(np.argmin(np.abs(steps_s - min(float(steps_s[-1]), WARMSTART_END_STEP))))
            print(
                f"{label}: n={int(n_per_s[idx_ws])}/{n_lo} "
                f"success@{int(steps_s[idx_ws])}"
                f"={mean_s[idx_ws]:.3f}±{std_s[idx_ws]:.3f}"
                f" (partial to {int(steps_s[-1])})"
            )
            if n_mid and float(steps_s[-1]) > WARMSTART_END_STEP:
                idx_br = int(np.argmin(np.abs(steps_s - BRIDGE_END_STEP)))
                print(
                    f"{label}: n={int(n_per_s[idx_br])}/{n_mid} "
                    f"success@{int(steps_s[idx_br])}"
                    f"={mean_s[idx_br]:.3f}±{std_s[idx_br]:.3f}"
                )
            if n_hi and float(steps_s[-1]) > BRIDGE_END_STEP:
                print(
                    f"{label}: n={int(n_per_s[-1])}/{n_hi} success@{int(steps_s[-1])}"
                    f"={mean_s[-1]:.3f}±{std_s[-1]:.3f}"
                )
            continue

        if run_group not in two_phase:
            print(f"skip {label}: unknown series {run_group}")
            continue
        phase1_group, phase2_group, split_step = two_phase[run_group]
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
            ax.fill_between(
                steps, mean - std, mean + std, color=color, alpha=0.15
            )
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

    if plotted == 0:
        raise SystemExit("No series found to plot")

    for ax in axes:
        ax.set_xlim(0, args.max_step)
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("humanoidmaze-large-navigate-singletask-v0")
    axes[1].set_title("Episode return")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
