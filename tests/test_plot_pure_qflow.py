"""Plot wiring for pure Q-Flow piecewise series (1M offline → +1M online)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_dflrql_vs_baseline import (  # noqa: E402
    DEFAULT_ONLINE_PLOT_MAX_STEP,
    DEFAULT_PLOT_MAX_STEP,
    DEFAULT_PURE_QFLOW_PHASE1_GROUP,
    DEFAULT_PURE_QFLOW_PHASE2_GROUP,
    PURE_QFLOW_ONLINE_START_STEP,
    PURE_QFLOW_PLACEHOLDER,
    aggregate_two_phase_piecewise,
    build_series,
)
from plot_offline_to_online import build_online_series  # noqa: E402


def _write_eval(run_dir: Path, rows: list[tuple[int, float, float]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "eval.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "evaluation/success",
                "evaluation/episode.return",
            ],
        )
        w.writeheader()
        for step, succ, ret in rows:
            w.writerow(
                {
                    "step": step,
                    "evaluation/success": succ,
                    "evaluation/episode.return": ret,
                }
            )


def test_pure_qflow_series_separation_and_defaults():
    offline = build_series("clf-group", all_methods=False)
    offline_labels = {s[0] for s in offline}
    assert "Pure Q-Flow" not in offline_labels
    online = {s[0]: s[1] for s in build_online_series()}
    assert online["Pure Q-Flow"] == PURE_QFLOW_PLACEHOLDER
    assert DEFAULT_PURE_QFLOW_PHASE1_GROUP.endswith("qflow-offline-1m")
    assert DEFAULT_PURE_QFLOW_PHASE2_GROUP.endswith("qflow-online-2m")
    assert PURE_QFLOW_ONLINE_START_STEP == 1_000_000
    assert DEFAULT_PLOT_MAX_STEP == 2_000_000
    assert DEFAULT_ONLINE_PLOT_MAX_STEP == 2_000_000


def test_pure_qflow_piecewise_join_at_1m(tmp_path: Path):
    exp = tmp_path
    p1 = exp / "rql" / DEFAULT_PURE_QFLOW_PHASE1_GROUP / "sd000_a"
    p2 = exp / "rql" / DEFAULT_PURE_QFLOW_PHASE2_GROUP / "sd000_b"
    _write_eval(
        p1,
        [
            (500_000, 0.2, 10.0),
            (1_000_000, 0.4, 20.0),
        ],
    )
    _write_eval(
        p2,
        [
            (1_000_000, 0.4, 20.0),
            (1_500_000, 0.6, 30.0),
            (2_000_000, 0.8, 40.0),
        ],
    )

    agg = aggregate_two_phase_piecewise(
        exp,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        phase1_group=DEFAULT_PURE_QFLOW_PHASE1_GROUP,
        phase2_group=DEFAULT_PURE_QFLOW_PHASE2_GROUP,
        split_step=PURE_QFLOW_ONLINE_START_STEP,
    )
    assert agg is not None
    steps, mean, _std, _n_per, n_lo, n_hi = agg
    assert n_lo == 1 and n_hi == 1
    assert float(steps[0]) == 500_000
    assert float(steps[-1]) == 2_000_000
    # Phase-1 contributes ≤1M; phase-2 contributes >1M only (no duplicate join).
    assert set(steps.astype(int).tolist()) == {
        500_000,
        1_000_000,
        1_500_000,
        2_000_000,
    }
    assert float(mean[np.where(steps == 1_000_000)[0][0]]) == 0.4
    assert float(mean[np.where(steps == 1_500_000)[0][0]]) == 0.6

    # Cap at offline end: online-only points filtered out.
    agg_1m = aggregate_two_phase_piecewise(
        exp,
        "success",
        plot_max_step=PURE_QFLOW_ONLINE_START_STEP,
        phase1_group=DEFAULT_PURE_QFLOW_PHASE1_GROUP,
        phase2_group=DEFAULT_PURE_QFLOW_PHASE2_GROUP,
        split_step=PURE_QFLOW_ONLINE_START_STEP,
    )
    assert agg_1m is not None
    steps_1m, _, _, _, _, n_hi_1m = agg_1m
    assert n_hi_1m == 1  # phase-2 seed exists but points filtered by max_step
    assert float(steps_1m[-1]) == 1_000_000


def test_pure_qflow_incomplete_online_phase(tmp_path: Path):
    """Offline-only partial data still plots; missing online is tolerated."""
    exp = tmp_path
    p1 = exp / "rql" / DEFAULT_PURE_QFLOW_PHASE1_GROUP / "sd000_a"
    _write_eval(p1, [(50_000, 0.1, 5.0), (100_000, 0.2, 8.0)])
    agg = aggregate_two_phase_piecewise(
        exp,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        phase1_group=DEFAULT_PURE_QFLOW_PHASE1_GROUP,
        phase2_group=DEFAULT_PURE_QFLOW_PHASE2_GROUP,
        split_step=PURE_QFLOW_ONLINE_START_STEP,
    )
    assert agg is not None
    steps, mean, _std, _n_per, n_lo, n_hi = agg
    assert n_lo == 1 and n_hi == 0
    assert float(steps[-1]) == 100_000
    assert float(mean[-1]) == 0.2
