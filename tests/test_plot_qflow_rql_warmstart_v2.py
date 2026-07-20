"""Plot wiring for Q-Flow RQL warmstart v2 three-phase piecewise series."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_dflrql_vs_baseline import (  # noqa: E402
    BRIDGE_END_STEP,
    DEFAULT_ONLINE_PLOT_MAX_STEP,
    DEFAULT_PLOT_MAX_STEP,
    DEFAULT_QFLOW_V2_PHASE1_GROUP,
    DEFAULT_QFLOW_V2_PHASE2_GROUP,
    DEFAULT_QFLOW_V2_PHASE3_GROUP,
    QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
    WARMSTART_END_STEP,
    aggregate_three_phase_piecewise,
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
    assert "Q-Flow RQL warmstart v2" not in offline
    online = {s[0]: s[1] for s in build_online_series()}
    assert online["Q-Flow RQL warmstart v2"] == QFLOW_RQL_WARMSTART_V2_PLACEHOLDER
    assert DEFAULT_QFLOW_V2_PHASE1_GROUP.endswith("qflow-rql-warmstart-2m")
    assert DEFAULT_QFLOW_V2_PHASE2_GROUP.endswith("qflow-rql-bridge-2p1m")
    assert DEFAULT_QFLOW_V2_PHASE3_GROUP.endswith("qflow-rql-online-4p1m")
    assert WARMSTART_END_STEP == 2_000_000
    assert BRIDGE_END_STEP == 2_100_000
    assert DEFAULT_PLOT_MAX_STEP == 2_000_000
    assert DEFAULT_ONLINE_PLOT_MAX_STEP == 4_100_000


def test_qflow_v2_piecewise_segments(tmp_path: Path):
    exp = tmp_path
    p1 = exp / "rql" / DEFAULT_QFLOW_V2_PHASE1_GROUP / "sd000_a"
    p2 = exp / "rql" / DEFAULT_QFLOW_V2_PHASE2_GROUP / "sd000_b"
    p3 = exp / "rql" / DEFAULT_QFLOW_V2_PHASE3_GROUP / "sd000_c"
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
            (2_100_000, 0.45, 22.0),
        ],
    )
    _write_eval(
        p3,
        [
            (2_100_000, 0.45, 22.0),
            (3_100_000, 0.6, 30.0),
            (4_100_000, 0.8, 40.0),
        ],
    )

    agg = aggregate_three_phase_piecewise(
        exp,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        phase1_group=DEFAULT_QFLOW_V2_PHASE1_GROUP,
        phase2_group=DEFAULT_QFLOW_V2_PHASE2_GROUP,
        phase3_group=DEFAULT_QFLOW_V2_PHASE3_GROUP,
        warmstart_end=WARMSTART_END_STEP,
        bridge_end=BRIDGE_END_STEP,
    )
    assert agg is not None
    steps, mean, _std, _n_per, n_lo, n_mid, n_hi = agg
    assert n_lo == 1 and n_mid == 1 and n_hi == 1
    assert float(steps[0]) == 1_000_000
    assert float(steps[-1]) == 4_100_000
    # Warmstart ≤2M; bridge (2M, 2.1M]; online >2.1M (no duplicate joins).
    assert set(steps.astype(int).tolist()) == {
        1_000_000,
        2_000_000,
        2_100_000,
        3_100_000,
        4_100_000,
    }
    assert float(mean[np.where(steps == 2_000_000)[0][0]]) == 0.4
    assert float(mean[np.where(steps == 2_100_000)[0][0]]) == 0.45
    assert float(mean[np.where(steps == 3_100_000)[0][0]]) == 0.6

    # Cap at warmstart end: bridge/online points filtered out.
    agg_2m = aggregate_three_phase_piecewise(
        exp,
        "success",
        plot_max_step=WARMSTART_END_STEP,
        phase1_group=DEFAULT_QFLOW_V2_PHASE1_GROUP,
        phase2_group=DEFAULT_QFLOW_V2_PHASE2_GROUP,
        phase3_group=DEFAULT_QFLOW_V2_PHASE3_GROUP,
        warmstart_end=WARMSTART_END_STEP,
        bridge_end=BRIDGE_END_STEP,
    )
    assert agg_2m is not None
    steps_2m, _, _, _, _, n_mid_2m, n_hi_2m = agg_2m
    assert n_mid_2m == 1  # bridge seed exists but points filtered by max_step
    assert n_hi_2m == 1
    assert float(steps_2m[-1]) == 2_000_000

    # Cap through bridge: online-only points filtered; 2.1M retained from bridge.
    agg_bridge = aggregate_three_phase_piecewise(
        exp,
        "success",
        plot_max_step=BRIDGE_END_STEP,
        phase1_group=DEFAULT_QFLOW_V2_PHASE1_GROUP,
        phase2_group=DEFAULT_QFLOW_V2_PHASE2_GROUP,
        phase3_group=DEFAULT_QFLOW_V2_PHASE3_GROUP,
        warmstart_end=WARMSTART_END_STEP,
        bridge_end=BRIDGE_END_STEP,
    )
    assert agg_bridge is not None
    steps_br, _, _, _, _, _, n_hi_br = agg_bridge
    assert n_hi_br == 1
    assert set(steps_br.astype(int).tolist()) == {
        1_000_000,
        2_000_000,
        2_100_000,
    }


def test_qflow_v2_incomplete_phases_partial_warmstart(tmp_path: Path):
    """Missing bridge/online still returns warmstart partial curve."""
    exp = tmp_path
    p1 = exp / "rql" / DEFAULT_QFLOW_V2_PHASE1_GROUP / "sd000_a"
    _write_eval(p1, [(100_000, 0.05, 2.0), (200_000, 0.1, 4.0)])
    agg = aggregate_three_phase_piecewise(
        exp,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        phase1_group=DEFAULT_QFLOW_V2_PHASE1_GROUP,
        phase2_group=DEFAULT_QFLOW_V2_PHASE2_GROUP,
        phase3_group=DEFAULT_QFLOW_V2_PHASE3_GROUP,
        warmstart_end=WARMSTART_END_STEP,
        bridge_end=BRIDGE_END_STEP,
    )
    assert agg is not None
    steps, mean, _std, _n_per, n_lo, n_mid, n_hi = agg
    assert n_lo == 1 and n_mid == 0 and n_hi == 0
    assert set(steps.astype(int).tolist()) == {100_000, 200_000}
    assert float(mean[-1]) == 0.1
