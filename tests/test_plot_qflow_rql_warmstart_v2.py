"""Plot wiring for Q-Flow RQL warmstart 1M+1M piecewise series."""

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
    DEFAULT_QFLOW_V2_PHASE1_GROUP,
    DEFAULT_QFLOW_V2_PHASE2_GROUP,
    ONLINE_PHASE_SPLIT_STEP,
    QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
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


def test_qflow_v2_series_separation_and_defaults():
    offline = {s[0] for s in build_series("clf-group", all_methods=False)}
    assert "Q-Flow RQL warmstart" not in offline
    online = {s[0]: s[1] for s in build_online_series()}
    assert online["Q-Flow RQL warmstart"] == QFLOW_RQL_WARMSTART_V2_PLACEHOLDER
    assert DEFAULT_QFLOW_V2_PHASE1_GROUP.endswith("qflow-rql-warmstart-2m")
    assert DEFAULT_QFLOW_V2_PHASE2_GROUP.endswith("qflow-rql-online-2m")
    assert ONLINE_PHASE_SPLIT_STEP == 1_000_000
    assert DEFAULT_PLOT_MAX_STEP == 2_000_000
    assert DEFAULT_ONLINE_PLOT_MAX_STEP == 2_000_000


def test_qflow_v2_piecewise_join_at_1m(tmp_path: Path):
    exp = tmp_path
    p1 = exp / "rql" / DEFAULT_QFLOW_V2_PHASE1_GROUP / "sd000_a"
    p2 = exp / "rql" / DEFAULT_QFLOW_V2_PHASE2_GROUP / "sd000_b"
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
    steps, mean, std, n_per, n_lo, n_hi = aggregate_two_phase_piecewise(
        exp,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        phase1_group=DEFAULT_QFLOW_V2_PHASE1_GROUP,
        phase2_group=DEFAULT_QFLOW_V2_PHASE2_GROUP,
        split_step=ONLINE_PHASE_SPLIT_STEP,
    )
    assert n_lo == 1 and n_hi == 1
    assert float(steps[0]) == 500_000
    assert float(steps[-1]) == 2_000_000
    assert np.isclose(mean[0], 0.2)
    assert np.isclose(mean[-1], 0.8)


def test_qflow_v2_incomplete_phase1_partial(tmp_path: Path):
    """Missing online still returns warmstart partial curve."""
    exp = tmp_path
    p1 = exp / "rql" / DEFAULT_QFLOW_V2_PHASE1_GROUP / "sd000_a"
    _write_eval(p1, [(200_000, 0.1, 5.0), (500_000, 0.25, 12.0)])
    steps, mean, std, n_per, n_lo, n_hi = aggregate_two_phase_piecewise(
        exp,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        phase1_group=DEFAULT_QFLOW_V2_PHASE1_GROUP,
        phase2_group=DEFAULT_QFLOW_V2_PHASE2_GROUP,
        split_step=ONLINE_PHASE_SPLIT_STEP,
    )
    assert n_lo == 1 and n_hi == 0
    assert float(steps[-1]) == 500_000
    assert np.isclose(mean[-1], 0.25)
