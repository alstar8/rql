"""Plot wiring for RQL→Q-Flow online piecewise series."""

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
    DEFAULT_RQL_QFLOW_ACTORFREEZE_GROUP,
    DEFAULT_RQL_QFLOW_PHASE1_GROUP,
    DEFAULT_RQL_QFLOW_PHASE2_GROUP,
    ONLINE_START_STEP,
    RQL_QFLOW_ACTORFREEZE_PLACEHOLDER,
    RQL_QFLOW_ONLINE_PLACEHOLDER,
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


def test_rql_qflow_series_separation_and_defaults():
    offline = {s[0] for s in build_series("clf-group", all_methods=False)}
    assert "RQL→Q-Flow online" not in offline
    online = {s[0]: s[1] for s in build_online_series()}
    assert online["RQL→Q-Flow online"] == RQL_QFLOW_ONLINE_PLACEHOLDER
    assert online["RQL→Q-Flow actor-freeze"] == RQL_QFLOW_ACTORFREEZE_PLACEHOLDER
    assert DEFAULT_RQL_QFLOW_PHASE1_GROUP.endswith("rql-qflow-ready-2m")
    assert DEFAULT_RQL_QFLOW_PHASE2_GROUP.endswith("rql-qflow-online-4m")
    assert DEFAULT_RQL_QFLOW_ACTORFREEZE_GROUP.endswith("actorfreeze-2p1m")
    assert ONLINE_START_STEP == 2_000_000
    assert DEFAULT_PLOT_MAX_STEP == 2_000_000
    assert DEFAULT_ONLINE_PLOT_MAX_STEP == 4_100_000


def test_rql_qflow_piecewise_join_at_2m(tmp_path: Path):
    exp = tmp_path
    p1 = exp / "rql" / DEFAULT_RQL_QFLOW_PHASE1_GROUP / "sd000_a"
    p2 = exp / "rql" / DEFAULT_RQL_QFLOW_PHASE2_GROUP / "sd000_b"
    _write_eval(
        p1,
        [
            (1_000_000, 0.2, 10.0),
            (2_000_000, 0.4, 20.0),
        ],
    )
    _write_eval(
        p2,
        [
            (2_000_000, 0.4, 20.0),
            (3_000_000, 0.6, 30.0),
            (4_000_000, 0.8, 40.0),
        ],
    )

    agg = aggregate_two_phase_piecewise(
        exp,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        phase1_group=DEFAULT_RQL_QFLOW_PHASE1_GROUP,
        phase2_group=DEFAULT_RQL_QFLOW_PHASE2_GROUP,
        split_step=ONLINE_START_STEP,
    )
    assert agg is not None
    steps, mean, _std, _n_per, n_lo, n_hi = agg
    assert n_lo == 1 and n_hi == 1
    assert float(steps[0]) == 1_000_000
    assert float(steps[-1]) == 4_000_000
    # Phase-1 contributes ≤2M; phase-2 contributes >2M only (no duplicate join).
    assert set(steps.astype(int).tolist()) == {
        1_000_000,
        2_000_000,
        3_000_000,
        4_000_000,
    }
    assert float(mean[np.where(steps == 2_000_000)[0][0]]) == 0.4
    assert float(mean[np.where(steps == 3_000_000)[0][0]]) == 0.6
    # Prior 2M methods naturally stop: filtering at 2M drops online-only points.
    agg_2m = aggregate_two_phase_piecewise(
        exp,
        "success",
        plot_max_step=2_000_000,
        phase1_group=DEFAULT_RQL_QFLOW_PHASE1_GROUP,
        phase2_group=DEFAULT_RQL_QFLOW_PHASE2_GROUP,
        split_step=ONLINE_START_STEP,
    )
    assert agg_2m is not None
    steps_2m, _, _, _, _, n_hi_2m = agg_2m
    assert n_hi_2m == 1  # phase-2 seed exists but all points filtered by max_step
    assert float(steps_2m[-1]) == 2_000_000


def test_actorfreeze_joins_ready_without_offset(tmp_path: Path):
    """Actor-freeze diagnostic reuses ready-2m ≤2M then 2.1M point absolute."""
    exp = tmp_path
    p1 = exp / "rql" / DEFAULT_RQL_QFLOW_PHASE1_GROUP / "sd000_a"
    p2 = exp / "rql" / DEFAULT_RQL_QFLOW_ACTORFREEZE_GROUP / "sd000_b"
    _write_eval(p1, [(2_000_000, 0.6, 25.0)])
    _write_eval(p2, [(2_100_000, 0.61, 26.0)])
    agg = aggregate_two_phase_piecewise(
        exp,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        phase1_group=DEFAULT_RQL_QFLOW_PHASE1_GROUP,
        phase2_group=DEFAULT_RQL_QFLOW_ACTORFREEZE_GROUP,
        split_step=ONLINE_START_STEP,
    )
    assert agg is not None
    steps, mean, _std, _n_per, n_lo, n_hi = agg
    assert n_lo == 1 and n_hi == 1
    assert set(steps.astype(int).tolist()) == {2_000_000, 2_100_000}
    assert float(mean[0]) == 0.6
    assert float(mean[1]) == 0.61
