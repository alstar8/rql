#!/usr/bin/env python3
"""Plot humanoidmaze-large offline DFL-RQL variants vs RQL baseline.

Reads eval.csv under exp/rql/<run_group>/sd*/ and writes
my_exps/dflrql_vs_baseline.png (success + return, mean ± std over seeds).

Default core SERIES (--core-only): offline baselines/ablations only —
RQL baseline + six from-scratch 3-seed × 2M actor runs (direct 0..2M curves).
Offline→online Q-Flow arms live in plot_offline_to_online.py; this module
still exports their constants and piecewise aggregators for --all-methods
and shared tests:

  1. DFL-RQL v9          humanoidmaze-large-dflrql9-scratch-2m
  2. Quantized DFL-RQL v9 humanoidmaze-large-quantized-dflrql9-scratch-2m
  3. DARI (AR OAT)        humanoidmaze-large-dari-scratch-2m
  4. CDF (discrete FM)    humanoidmaze-large-cdf-scratch-2m
  5. AR QDFL student      humanoidmaze-large-discrete-ar-qdfl-distill-2m
  6. DD QDFL student      humanoidmaze-large-discrete-diffusion-qdfl-distill-2m

Historical / all-methods options (preserved):
  V7 piecewise 1M→2M; Quantized V9 piecewise 400k restore path
  (`__quantized_v9__`); paired QDFL students piecewise joint+freeze
  (`__ar_qdfl_distill__` / `__dd_qdfl_distill__`); legacy RQL→Q-Flow,
  Q-Flow RQL warmstart v2, and pure Q-Flow placeholders.

Pass --with-extras for CLF v4 / historical CDF smokes / DCMI / DARI v6;
--all-methods for the full historical suite. Override primary CLF/DCMI/DARI
with --clf-run-group / --dcmi-run-group / --dari-run-group; override student
phase groups with --ar-qdfl-*-run-group / --dd-qdfl-*-run-group; override
Q-Flow phase groups with --rql-qflow-*-run-group /
--qflow-v2-*-run-group / --pure-qflow-*-run-group.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Best CLF-v4 arm from 300k smokes (immediate actor RL). Override with --clf-run-group.
DEFAULT_CLF_RUN_GROUP = "humanoidmaze-large-clf-v4-immediate-actor-smoke300k"
CLF_V4_COLOR = "#C2185B"
CLF_V4_DELAYED_GROUP = "humanoidmaze-large-clf-v4-alpha1-smoke300k"
CLF_V4_DELAYED_COLOR = "#F06292"

# DiscreteCoordMaskIQL v5. Override with --dcmi-run-group.
DEFAULT_DCMI_RUN_GROUP = "humanoidmaze-large-dcmi-v5-2m"
DCMI_V5_COLOR = "#2E7D32"

# Historical DiscreteARIQL v6 (extras / all-methods). Override with --dari-run-group.
DEFAULT_DARI_RUN_GROUP = "humanoidmaze-large-dari-v6-2m"
DARI_V6_COLOR = "#6A1B9A"

# From-scratch 2M core groups (direct curves).
DEFAULT_DFLRQL9_SCRATCH_GROUP = "humanoidmaze-large-dflrql9-scratch-2m"
DEFAULT_QUANTIZED_DFLRQL9_SCRATCH_GROUP = (
    "humanoidmaze-large-quantized-dflrql9-scratch-2m"
)
DEFAULT_DARI_SCRATCH_GROUP = "humanoidmaze-large-dari-scratch-2m"
DEFAULT_CDF_SCRATCH_GROUP = "humanoidmaze-large-cdf-scratch-2m"
DEFAULT_AR_QDFL_PHASE1_GROUP = "humanoidmaze-large-discrete-ar-qdfl-distill-2m"
DEFAULT_DD_QDFL_PHASE1_GROUP = (
    "humanoidmaze-large-discrete-diffusion-qdfl-distill-2m"
)
DEFAULT_RQL_QFLOW_PHASE1_GROUP = "humanoidmaze-large-rql-qflow-ready-2m"
DEFAULT_RQL_QFLOW_PHASE2_GROUP = "humanoidmaze-large-rql-qflow-online-4m"
DEFAULT_RQL_QFLOW_ACTORFREEZE_GROUP = (
    "humanoidmaze-large-rql-qflow-online-actorfreeze-2p1m"
)
RQL_QFLOW_ONLINE_PLACEHOLDER = "__rql_qflow_online__"
RQL_QFLOW_ACTORFREEZE_PLACEHOLDER = "__rql_qflow_actorfreeze__"
ONLINE_START_STEP = 2_000_000

DEFAULT_QFLOW_V2_PHASE1_GROUP = "humanoidmaze-large-qflow-rql-warmstart-2m"
DEFAULT_QFLOW_V2_PHASE2_GROUP = "humanoidmaze-large-qflow-rql-online-2m"
# Legacy three-phase groups (bridge + 2M online) kept for historical helpers/tests.
DEFAULT_QFLOW_V2_PHASE3_GROUP = "humanoidmaze-large-qflow-rql-online-4p1m"
DEFAULT_QFLOW_V2_BRIDGE_GROUP = "humanoidmaze-large-qflow-rql-bridge-2p1m"
QFLOW_RQL_WARMSTART_V2_PLACEHOLDER = "__qflow_rql_warmstart_v2__"
WARMSTART_END_STEP = 2_000_000
BRIDGE_END_STEP = 2_100_000
# Offline core and offline→online plots both end at 2M (1M offline + 1M online).
DEFAULT_PLOT_MAX_STEP = 2_000_000
DEFAULT_ONLINE_PLOT_MAX_STEP = 2_000_000
ONLINE_PHASE_SPLIT_STEP = 1_000_000

DEFAULT_PURE_QFLOW_PHASE1_GROUP = "humanoidmaze-large-qflow-offline-1m"
DEFAULT_PURE_QFLOW_PHASE2_GROUP = "humanoidmaze-large-qflow-online-2m"
PURE_QFLOW_PLACEHOLDER = "__pure_qflow__"
PURE_QFLOW_ONLINE_START_STEP = ONLINE_PHASE_SPLIT_STEP

DEFAULT_RQL_ONLINE_PHASE1_GROUP = "humanoidmaze-large-rql-offline-1m"
DEFAULT_RQL_ONLINE_PHASE2_GROUP = "humanoidmaze-large-rql-online-2m"
RQL_ONLINE_PLACEHOLDER = "__rql_online__"

# ConsensusFlow full (dflrql9 defaults: floor/conflict/residual on).
# Phase 1 reuses the scratch offline run (plot uses ≤1M); phase 2 is online.
DEFAULT_CF_PHASE1_GROUP = "humanoidmaze-large-dflrql9-scratch-2m"
DEFAULT_CF_PHASE2_GROUP = "humanoidmaze-large-cf-online-2m"
CF_PLACEHOLDER = "__cf__"

# ConsensusFlow no-CRF (dflrql9): reuse HL ablation offline-1M as phase 1.
DEFAULT_CF_NOCRF_PHASE1_GROUP = "humanoidmaze-large-cf-ablation-nocrf-1m"
DEFAULT_CF_NOCRF_PHASE2_GROUP = "humanoidmaze-large-cf-nocrf-online-2m"
CF_NOCRF_PLACEHOLDER = "__cf_nocrf__"

# AR-QDFL FastSAC: single run group with absolute plot steps 0..2M.
# Critic warmup is hidden from eval.csv (see run_ar_qdfl_fast_sac.py).
DEFAULT_AR_QDFL_FASTSAC_GROUP = "humanoidmaze-large-ar-qdfl-fastsac-2m"
AR_QDFL_FASTSAC_PLACEHOLDER = "__ar_qdfl_fastsac__"
AR_QDFL_FASTSAC_WARMUP_UPDATES = 100_000
AR_QDFL_FASTSAC_WARMUP_POLICY = {
    "mode": "hidden",
    "warmup_updates": AR_QDFL_FASTSAC_WARMUP_UPDATES,
    "included_in_eval_csv": False,
    "plot_axis": "absolute_0_to_2m",
    # Post-warmup eval is logged at offline_steps+1 (or remapped there when
    # older runs wrote a duplicate row at offline_steps).
    "immediate_eval_at_offline_end": True,
    "post_warmup_eval_step_offset": 1,
    "source": "scripts/run_ar_qdfl_fast_sac.py",
}

CORE_SCRATCH_GROUPS = (
    DEFAULT_DFLRQL9_SCRATCH_GROUP,
    DEFAULT_QUANTIZED_DFLRQL9_SCRATCH_GROUP,
    DEFAULT_DARI_SCRATCH_GROUP,
    DEFAULT_CDF_SCRATCH_GROUP,
    DEFAULT_AR_QDFL_PHASE1_GROUP,
    DEFAULT_DD_QDFL_PHASE1_GROUP,
)

# Offline core only (label, run_group, color, linestyle).
# Q-Flow offline→online arms are plotted by plot_offline_to_online.py.
_SERIES_HEAD = [
    ("RQL baseline", "humanoidmaze-large-rql-tuned-2m", "#4C72B0", "-"),
    ("DFL-RQL v9", DEFAULT_DFLRQL9_SCRATCH_GROUP, "#0F766E", "-"),
    (
        "Quantized DFL-RQL v9",
        DEFAULT_QUANTIZED_DFLRQL9_SCRATCH_GROUP,
        "#E11D48",
        "-",
    ),
    ("DARI (AR OAT)", DEFAULT_DARI_SCRATCH_GROUP, "#6A1B9A", "-"),
    ("CDF (discrete FM)", DEFAULT_CDF_SCRATCH_GROUP, "#D55E00", "-"),
    ("AR QDFL student", DEFAULT_AR_QDFL_PHASE1_GROUP, "#F59E0B", "-"),
    ("DD QDFL student", DEFAULT_DD_QDFL_PHASE1_GROUP, "#0284C7", "-"),
]
_SERIES_CDF_TAIL = [
    (
        "CLF v4 delayed",
        CLF_V4_DELAYED_GROUP,
        CLF_V4_DELAYED_COLOR,
        "--",
    ),
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

# Full historical series (use --all-methods); CLF/DCMI/DARI inserted before CDF tail.
# Quantized V9 and QDFL students use piecewise placeholders here.
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
    (
        "Quantized DFL-RQL v9 (400k restore)",
        "__quantized_v9__",
        "#E11D48",
        "-",
    ),
    (
        "DFL-RQL v9 scratch",
        DEFAULT_DFLRQL9_SCRATCH_GROUP,
        "#0F766E",
        "--",
    ),
    (
        "Quantized DFL-RQL v9 scratch",
        DEFAULT_QUANTIZED_DFLRQL9_SCRATCH_GROUP,
        "#E11D48",
        "--",
    ),
    (
        "DARI scratch",
        DEFAULT_DARI_SCRATCH_GROUP,
        "#6A1B9A",
        "--",
    ),
    (
        "CDF scratch",
        DEFAULT_CDF_SCRATCH_GROUP,
        "#D55E00",
        "--",
    ),
    (
        "AR QDFL student (phase1+2)",
        "__ar_qdfl_distill__",
        "#F59E0B",
        "-",
    ),
    (
        "DD QDFL student (phase1+2)",
        "__dd_qdfl_distill__",
        "#0284C7",
        "-",
    ),
    (
        "Q-Flow RQL warmstart (phase1+2)",
        QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
        "#134E4A",
        "-",
    ),
    (
        "Pure Q-Flow (phase1+2)",
        PURE_QFLOW_PLACEHOLDER,
        "#BE123C",
        "-",
    ),
    ("ConsensusDiscreteFlow v2", "humanoidmaze-large-cdf-2m-v2", "#999999", ":"),
]

QUANTIZED_V9_600K_GROUP = "humanoidmaze-large-quantized-dflrql9-resume400k-to600k"
QUANTIZED_V9_2M_GROUP = "humanoidmaze-large-quantized-dflrql9-2m"
QUANTIZED_V9_PARENT_400K_GROUP = "humanoidmaze-large-dflrql9-400k-ckpt"
QUANTIZED_V9_PARENT_2M_GROUP = "humanoidmaze-large-dflrql9-2m"
QUANTIZED_V9_RESTORE_STEP = 400_000
QUANTIZED_V9_SPLIT_STEP = 600_000

DEFAULT_AR_QDFL_PHASE2_GROUP = "humanoidmaze-large-discrete-ar-qdfl-distill-2p1m"
DEFAULT_DD_QDFL_PHASE2_GROUP = "humanoidmaze-large-discrete-diffusion-qdfl-distill-2p1m"
TEACHER_FREEZE_STEP = 2_000_000


def build_series(
    clf_run_group: str,
    all_methods: bool,
    dcmi_run_group: str = DEFAULT_DCMI_RUN_GROUP,
    dari_run_group: str = DEFAULT_DARI_RUN_GROUP,
    core_only: bool = True,
    with_extras: bool = False,
) -> list[tuple[str, str, str, str]]:
    clf = ("CLF v4 immediate", clf_run_group, CLF_V4_COLOR, "-")
    dcmi = ("DCMI v5", dcmi_run_group, DCMI_V5_COLOR, "-")
    dari = ("DARI v6", dari_run_group, DARI_V6_COLOR, "-")
    if all_methods:
        return [*_SERIES_ALL_HEAD, clf, dcmi, dari, *_SERIES_CDF_TAIL]
    if core_only and not with_extras:
        # Offline only: RQL baseline + six from-scratch 2M actors.
        return list(_SERIES_HEAD)
    # Avoid duplicating the delayed series if the primary CLF group is already that run.
    # Also skip historical CDF smokes that collide with the core CDF scratch label color.
    tail = [
        s
        for s in _SERIES_CDF_TAIL
        if not (s[1] == clf_run_group and s[0].startswith("CLF"))
    ]
    return [*_SERIES_HEAD, clf, dcmi, dari, *tail]
V7_1M_GROUP = "humanoidmaze-large-dflrql7"
V7_2M_GROUP = "humanoidmaze-large-dflrql7-2m"
V7_SPLIT_STEP = 1_000_000
V7_2M_SEEDS = (0, 1, 2)


def disambiguate_duplicate_eval_steps(
    rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Keep chronological duplicate steps by bumping later copies to step+k.

    AR-QDFL FastSAC historically wrote both the offline-end eval and the
    post-warmup eval at ``offline_steps``. Without remapping, last-wins merge
    drops the true offline endpoint. Bumping the 2nd occurrence of step S to
    S+1 matches the two-phase online-start convention (e.g. 1_000_001).
    """
    seen: dict[int, int] = {}
    out: list[dict[str, float]] = []
    for row in rows:
        step = int(row["step"])
        n = seen.get(step, 0)
        seen[step] = n + 1
        if n == 0:
            out.append(row)
        else:
            out.append({**row, "step": step + n})
    return out


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
    return disambiguate_duplicate_eval_steps(rows)


def seed_from_run_dir(run_dir: Path) -> int | None:
    m = re.match(r"sd(\d+)_", run_dir.name)
    return int(m.group(1)) if m else None


def merge_seed_curves(
    curves: list[list[dict[str, float]]],
) -> list[dict[str, float]]:
    """Union eval rows across resume dirs; later dirs win on duplicate steps."""
    by_step: dict[int, dict[str, float]] = {}
    for rows in curves:
        for row in rows:
            by_step[int(row["step"])] = row
    return [by_step[s] for s in sorted(by_step)]


def load_group_by_seed(
    exp_root: Path, run_group: str
) -> dict[int, list[dict[str, float]]]:
    group_dir = exp_root / "rql" / run_group
    if not group_dir.is_dir():
        return {}
    per_seed: dict[int, list[list[dict[str, float]]]] = {}
    for run_dir in sorted(group_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        seed = seed_from_run_dir(run_dir)
        if seed is None:
            continue
        # Prefer the eval curve with the most steps. Live FastSAC writers can
        # keep appending into an NFS ``.nfs*`` orphan after rsync replaces
        # ``eval.csv`` while the file is open.
        candidates = []
        eval_csv = run_dir / "eval.csv"
        if eval_csv.is_file():
            candidates.append(eval_csv)
        for orphan in run_dir.glob(".nfs*"):
            try:
                head = orphan.read_text(errors="ignore")[:120]
            except OSError:
                continue
            if "evaluation/success" in head or head.startswith("evaluation/"):
                candidates.append(orphan)
        best_rows: list[dict[str, float]] = []
        for path in candidates:
            rows = parse_eval_csv(path)
            if not rows:
                continue
            if not best_rows or max(int(r["step"]) for r in rows) > max(
                int(r["step"]) for r in best_rows
            ):
                best_rows = rows
        if not best_rows:
            continue
        per_seed.setdefault(seed, []).append(best_rows)
    return {seed: merge_seed_curves(curves) for seed, curves in per_seed.items()}


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


def load_quantized_v9_parent_by_seed(exp_root: Path) -> dict[int, list[dict[str, float]]]:
    """Continuous DFLRQL9 curves used as Quantized V9 init (prefer 400k-ckpt)."""
    parent = load_group_by_seed(exp_root, QUANTIZED_V9_PARENT_400K_GROUP)
    if parent:
        return parent
    return load_group_by_seed(exp_root, QUANTIZED_V9_PARENT_2M_GROUP)


def aggregate_quantized_v9_piecewise(
    exp_root: Path, metric: str, plot_max_step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    """Pre-400k from continuous parent; (400k,600k] resume; >600k from 2M."""
    c_parent = load_quantized_v9_parent_by_seed(exp_root)
    c_lo = load_group_by_seed(exp_root, QUANTIZED_V9_600K_GROUP)
    c_hi = load_group_by_seed(exp_root, QUANTIZED_V9_2M_GROUP)

    # Prefer seed-matched parent curves for seeds that enter quantized fine-tune.
    quantized_seeds = sorted(set(c_lo) | set(c_hi))
    if quantized_seeds and c_parent:
        curves_parent = [
            c_parent[s] for s in quantized_seeds if s in c_parent
        ]
        # If some quantized seeds lack a parent run, fall back to all parent seeds.
        if not curves_parent:
            curves_parent = [c_parent[s] for s in sorted(c_parent)]
    else:
        curves_parent = [c_parent[s] for s in sorted(c_parent)]

    curves_lo = [c_lo[s] for s in sorted(c_lo)]
    curves_hi = [
        c_hi[s]
        for s in sorted(c_hi)
        if max_step(c_hi[s]) > QUANTIZED_V9_SPLIT_STEP
    ]

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    n_parent = len(curves_parent)
    n_lo = len(curves_lo)
    n_hi = len(curves_hi)

    if curves_parent:
        agg = aggregate(curves_parent, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (steps <= QUANTIZED_V9_RESTORE_STEP) & (steps <= plot_max_step)
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if curves_lo:
        agg = aggregate(curves_lo, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (
                (steps > QUANTIZED_V9_RESTORE_STEP)
                & (steps <= QUANTIZED_V9_SPLIT_STEP)
                & (steps <= plot_max_step)
            )
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if curves_hi:
        agg = aggregate(curves_hi, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (steps > QUANTIZED_V9_SPLIT_STEP) & (steps <= plot_max_step)
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if not parts:
        return None
    steps = np.concatenate([p[0] for p in parts])
    mean = np.concatenate([p[1] for p in parts])
    std = np.concatenate([p[2] for p in parts])
    n_per = np.concatenate([p[3] for p in parts])
    # Legend uses pre-restore parent n and post-600k n.
    return steps, mean, std, n_per, n_parent or n_lo, n_hi


def aggregate_two_phase_piecewise(
    exp_root: Path,
    metric: str,
    plot_max_step: int,
    phase1_group: str,
    phase2_group: str,
    split_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int] | None:
    """Join phase-1 (≤split) and phase-2 (>split) curves; no extrapolation."""
    c1 = load_group_by_seed(exp_root, phase1_group)
    c2 = load_group_by_seed(exp_root, phase2_group)
    curves_1 = [c1[s] for s in sorted(c1)]
    curves_2 = [
        c2[s] for s in sorted(c2) if max_step(c2[s]) > split_step
    ]

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    n_lo = len(curves_1)
    n_hi = len(curves_2)

    if curves_1:
        agg = aggregate(curves_1, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (steps <= split_step) & (steps <= plot_max_step)
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if curves_2:
        agg = aggregate(curves_2, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (steps > split_step) & (steps <= plot_max_step)
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if not parts:
        return None
    steps = np.concatenate([p[0] for p in parts])
    mean = np.concatenate([p[1] for p in parts])
    std = np.concatenate([p[2] for p in parts])
    n_per = np.concatenate([p[3] for p in parts])
    return steps, mean, std, n_per, n_lo, n_hi


def aggregate_single_group(
    exp_root: Path,
    metric: str,
    plot_max_step: int,
    run_group: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None:
    """Mean/std over seeds for one run group; no extrapolation past last eval.

    Used by AR-QDFL FastSAC whose eval.csv already spans absolute plot steps
    0..2M with warmup omitted. Partial runs contribute only through their
    last available eval step (nan beyond that seed's last row).
    """
    curves = load_group(exp_root, run_group)
    if not curves:
        return None
    agg = aggregate(curves, metric)
    if agg is None:
        return None
    steps, mean, std, n_per = agg
    mask = steps <= plot_max_step
    if not np.any(mask):
        return None
    return steps[mask], mean[mask], std[mask], n_per[mask], len(curves)


def per_seed_endpoints(
    exp_root: Path,
    run_group: str,
    plot_max_step: int,
) -> dict[str, dict[str, float | int]]:
    """Last observed eval ≤ plot_max_step per seed; omit seeds with no rows."""
    by_seed = load_group_by_seed(exp_root, run_group)
    out: dict[str, dict[str, float | int]] = {}
    for seed in sorted(by_seed):
        rows = [r for r in by_seed[seed] if int(r["step"]) <= plot_max_step]
        if not rows:
            continue
        last = rows[-1]
        out[str(seed)] = {
            "step": int(last["step"]),
            "success": float(last["success"]),
            "return": float(last["return"]),
        }
    return out


def curve_arrays_to_lists(
    steps: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_per: np.ndarray,
) -> dict[str, list[float]]:
    return {
        "steps": [float(x) for x in steps],
        "mean": [float(x) for x in mean],
        "std": [float(x) for x in std],
        "n": [float(x) for x in n_per],
    }


def aggregate_three_phase_piecewise(
    exp_root: Path,
    metric: str,
    plot_max_step: int,
    phase1_group: str,
    phase2_group: str,
    phase3_group: str,
    warmstart_end: int,
    bridge_end: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int] | None:
    """Join warmstart (≤ws), bridge (ws, bridge], online (>bridge); no offset."""
    c1 = load_group_by_seed(exp_root, phase1_group)
    c2 = load_group_by_seed(exp_root, phase2_group)
    c3 = load_group_by_seed(exp_root, phase3_group)
    curves_1 = [c1[s] for s in sorted(c1)]
    curves_2 = [
        c2[s]
        for s in sorted(c2)
        if any(warmstart_end < int(r["step"]) <= bridge_end for r in c2[s])
    ]
    curves_3 = [
        c3[s] for s in sorted(c3) if max_step(c3[s]) > bridge_end
    ]

    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    n_lo = len(curves_1)
    n_mid = len(curves_2)
    n_hi = len(curves_3)

    if curves_1:
        agg = aggregate(curves_1, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (steps <= warmstart_end) & (steps <= plot_max_step)
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if curves_2:
        agg = aggregate(curves_2, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (
                (steps > warmstart_end)
                & (steps <= bridge_end)
                & (steps <= plot_max_step)
            )
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if curves_3:
        agg = aggregate(curves_3, metric)
        if agg is not None:
            steps, mean, std, n_per = agg
            mask = (steps > bridge_end) & (steps <= plot_max_step)
            if np.any(mask):
                parts.append((steps[mask], mean[mask], std[mask], n_per[mask]))

    if not parts:
        return None
    steps = np.concatenate([p[0] for p in parts])
    mean = np.concatenate([p[1] for p in parts])
    std = np.concatenate([p[2] for p in parts])
    n_per = np.concatenate([p[3] for p in parts])
    return steps, mean, std, n_per, n_lo, n_mid, n_hi


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
    parser.add_argument(
        "--max-step",
        type=int,
        default=DEFAULT_PLOT_MAX_STEP,
        help=(
            "X-axis / filter max step "
            f"(default: {DEFAULT_PLOT_MAX_STEP}; offline core ends at 2M)."
        ),
    )
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
        "--dcmi-run-group",
        type=str,
        default=DEFAULT_DCMI_RUN_GROUP,
        help=(
            "Run group for DiscreteCoordMaskIQL v5 "
            f"(default: {DEFAULT_DCMI_RUN_GROUP})."
        ),
    )
    parser.add_argument(
        "--dari-run-group",
        type=str,
        default=DEFAULT_DARI_RUN_GROUP,
        help=(
            "Run group for historical DiscreteARIQL v6 (extras/all-methods) "
            f"(default: {DEFAULT_DARI_RUN_GROUP})."
        ),
    )
    parser.add_argument(
        "--ar-qdfl-phase1-run-group",
        type=str,
        default=DEFAULT_AR_QDFL_PHASE1_GROUP,
        help=(
            "Joint co-train run group for AR QDFL student piecewise "
            f"(default: {DEFAULT_AR_QDFL_PHASE1_GROUP})."
        ),
    )
    parser.add_argument(
        "--ar-qdfl-phase2-run-group",
        type=str,
        default=DEFAULT_AR_QDFL_PHASE2_GROUP,
        help=(
            "Frozen-teacher run group for AR QDFL student piecewise "
            f"(default: {DEFAULT_AR_QDFL_PHASE2_GROUP})."
        ),
    )
    parser.add_argument(
        "--dd-qdfl-phase1-run-group",
        type=str,
        default=DEFAULT_DD_QDFL_PHASE1_GROUP,
        help=(
            "Joint co-train run group for discrete-diffusion QDFL student "
            f"piecewise (default: {DEFAULT_DD_QDFL_PHASE1_GROUP})."
        ),
    )
    parser.add_argument(
        "--dd-qdfl-phase2-run-group",
        type=str,
        default=DEFAULT_DD_QDFL_PHASE2_GROUP,
        help=(
            "Frozen-teacher run group for discrete-diffusion QDFL student "
            f"piecewise (default: {DEFAULT_DD_QDFL_PHASE2_GROUP})."
        ),
    )
    parser.add_argument(
        "--rql-qflow-phase1-run-group",
        type=str,
        default=DEFAULT_RQL_QFLOW_PHASE1_GROUP,
        help=(
            "Offline ready run group for RQL→Q-Flow online piecewise "
            f"(default: {DEFAULT_RQL_QFLOW_PHASE1_GROUP})."
        ),
    )
    parser.add_argument(
        "--rql-qflow-phase2-run-group",
        type=str,
        default=DEFAULT_RQL_QFLOW_PHASE2_GROUP,
        help=(
            "Online continuation run group for RQL→Q-Flow online piecewise "
            f"(default: {DEFAULT_RQL_QFLOW_PHASE2_GROUP})."
        ),
    )
    parser.add_argument(
        "--qflow-v2-phase1-run-group",
        type=str,
        default=DEFAULT_QFLOW_V2_PHASE1_GROUP,
        help=(
            "Warmstart run group for Q-Flow RQL warmstart v2 piecewise "
            f"(default: {DEFAULT_QFLOW_V2_PHASE1_GROUP})."
        ),
    )
    parser.add_argument(
        "--qflow-v2-phase2-run-group",
        type=str,
        default=DEFAULT_QFLOW_V2_PHASE2_GROUP,
        help=(
            "Bridge run group for Q-Flow RQL warmstart v2 piecewise "
            f"(default: {DEFAULT_QFLOW_V2_PHASE2_GROUP})."
        ),
    )
    parser.add_argument(
        "--qflow-v2-phase3-run-group",
        type=str,
        default=DEFAULT_QFLOW_V2_PHASE3_GROUP,
        help=(
            "Online continuation run group for Q-Flow RQL warmstart v2 "
            f"piecewise (default: {DEFAULT_QFLOW_V2_PHASE3_GROUP})."
        ),
    )
    parser.add_argument(
        "--pure-qflow-phase1-run-group",
        type=str,
        default=DEFAULT_PURE_QFLOW_PHASE1_GROUP,
        help=(
            "Offline run group for pure Q-Flow piecewise "
            f"(default: {DEFAULT_PURE_QFLOW_PHASE1_GROUP})."
        ),
    )
    parser.add_argument(
        "--pure-qflow-phase2-run-group",
        type=str,
        default=DEFAULT_PURE_QFLOW_PHASE2_GROUP,
        help=(
            "Online continuation run group for pure Q-Flow piecewise "
            f"(default: {DEFAULT_PURE_QFLOW_PHASE2_GROUP})."
        ),
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help=(
            "Plot the full historical method suite (piecewise Quantized V9 "
            "restore + QDFL phase1+2) instead of the six scratch core groups."
        ),
    )
    parser.add_argument(
        "--core-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Plot offline RQL baseline + six from-scratch 2M actors "
            "(default: true). Q-Flow offline→online arms are in "
            "plot_offline_to_online.py."
        ),
    )
    parser.add_argument(
        "--with-extras",
        action="store_true",
        help="Include CLF/historical CDF smokes/DCMI/DARI v6 alongside core.",
    )
    args = parser.parse_args()

    series = build_series(
        args.clf_run_group,
        args.all_methods,
        dcmi_run_group=args.dcmi_run_group,
        dari_run_group=args.dari_run_group,
        core_only=args.core_only,
        with_extras=args.with_extras,
    )
    student_phase_groups = {
        "__ar_qdfl_distill__": (
            args.ar_qdfl_phase1_run_group,
            args.ar_qdfl_phase2_run_group,
            TEACHER_FREEZE_STEP,
        ),
        "__dd_qdfl_distill__": (
            args.dd_qdfl_phase1_run_group,
            args.dd_qdfl_phase2_run_group,
            TEACHER_FREEZE_STEP,
        ),
        RQL_QFLOW_ONLINE_PLACEHOLDER: (
            args.rql_qflow_phase1_run_group,
            args.rql_qflow_phase2_run_group,
            ONLINE_START_STEP,
        ),
        QFLOW_RQL_WARMSTART_V2_PLACEHOLDER: (
            args.qflow_v2_phase1_run_group,
            args.qflow_v2_phase2_run_group,
            ONLINE_PHASE_SPLIT_STEP,
        ),
        PURE_QFLOW_PLACEHOLDER: (
            args.pure_qflow_phase1_run_group,
            args.pure_qflow_phase2_run_group,
            PURE_QFLOW_ONLINE_START_STEP,
        ),
    }
    series_groups = {run_group for _, run_group, _, _ in series}
    draw_teacher_freeze = bool(
        series_groups & {"__ar_qdfl_distill__", "__dd_qdfl_distill__"}
    )
    draw_online_start = RQL_QFLOW_ONLINE_PLACEHOLDER in series_groups
    draw_phase_split = bool(
        series_groups
        & {QFLOW_RQL_WARMSTART_V2_PLACEHOLDER, PURE_QFLOW_PLACEHOLDER}
    )
    draw_v7_split = "__v7__" in series_groups
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    plotted = 0
    has_2m_baseline = bool(load_group(args.save_dir, "humanoidmaze-large-rql-tuned-2m"))
    freeze_marker_drawn = False
    online_marker_drawn = False
    phase_marker_drawn = False

    def draw_markers(ax: plt.Axes) -> None:
        nonlocal freeze_marker_drawn, online_marker_drawn, phase_marker_drawn
        if draw_v7_split and args.max_step > V7_SPLIT_STEP:
            ax.axvline(V7_SPLIT_STEP, color="#bbbbbb", linestyle=":", linewidth=1)
        if draw_teacher_freeze and args.max_step >= TEACHER_FREEZE_STEP:
            ax.axvline(
                TEACHER_FREEZE_STEP,
                color="#333333",
                linestyle="--",
                linewidth=1.2,
                label="teacher freeze @2M" if not freeze_marker_drawn else None,
            )
            freeze_marker_drawn = True
        if draw_phase_split and args.max_step >= ONLINE_PHASE_SPLIT_STEP:
            ax.axvline(
                ONLINE_PHASE_SPLIT_STEP,
                color="#444444",
                linestyle=":",
                linewidth=1.2,
                label="online @1M" if not phase_marker_drawn else None,
            )
            phase_marker_drawn = True
        if draw_online_start and args.max_step >= ONLINE_START_STEP:
            ax.axvline(
                ONLINE_START_STEP,
                color="#7C2D12",
                linestyle="-.",
                linewidth=1.2,
                label="online start @2M" if not online_marker_drawn else None,
            )
            online_marker_drawn = True

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
                draw_markers(ax)
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

        if run_group == "__quantized_v9__":
            succ_agg = aggregate_quantized_v9_piecewise(
                args.save_dir, "success", args.max_step
            )
            if succ_agg is None:
                print("skip Quantized DFL-RQL v9: no runs")
                continue
            steps_s, mean_s, std_s, n_per_s, n_lo, n_hi = succ_agg
            legend = f"{label} (n={n_lo}@≤400k"
            legend += f", n={n_hi}@>600k)" if n_hi else ")"
            for metric, ax, ylabel in [
                ("success", axes[0], "Eval success"),
                ("return", axes[1], "Eval return"),
            ]:
                agg = (
                    succ_agg
                    if metric == "success"
                    else aggregate_quantized_v9_piecewise(
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
                draw_markers(ax)
            plotted += 1
            idx_split = int(np.argmin(np.abs(steps_s - QUANTIZED_V9_SPLIT_STEP)))
            print(
                f"{label}: n={int(n_per_s[idx_split])}/{n_lo} "
                f"success@{int(steps_s[idx_split])}"
                f"={mean_s[idx_split]:.3f}±{std_s[idx_split]:.3f}"
            )
            if n_hi and float(steps_s[-1]) > QUANTIZED_V9_SPLIT_STEP:
                print(
                    f"{label}: n={int(n_per_s[-1])}/{n_hi} success@{int(steps_s[-1])}"
                    f"={mean_s[-1]:.3f}±{std_s[-1]:.3f}"
                )
            continue

        if run_group in student_phase_groups:
            phase1_group, phase2_group, split_step = student_phase_groups[run_group]
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
            legend = f"{label} (n={n_lo}@≤{split_step // 1_000_000}M"
            legend += f", n={n_hi}@>{split_step // 1_000_000}M)" if n_hi else ")"
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
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
                ax.set_xlabel("Training steps")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.3)
                draw_markers(ax)
            plotted += 1
            idx_split = int(np.argmin(np.abs(steps_s - split_step)))
            print(
                f"{label}: n={int(n_per_s[idx_split])}/{n_lo} "
                f"success@{int(steps_s[idx_split])}"
                f"={mean_s[idx_split]:.3f}±{std_s[idx_split]:.3f}"
            )
            if n_hi and float(steps_s[-1]) > split_step:
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
            ax.set_xlabel("Training steps")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            draw_markers(ax)
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

    for ax in axes:
        ax.set_xlim(0, args.max_step)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].set_title("humanoidmaze-large-navigate-singletask-v0")
    axes[1].set_title("Episode return")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
