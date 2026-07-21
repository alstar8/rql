"""Plot wiring for AR-QDFL FastSAC single-group offline→online series."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_dflrql_vs_baseline import (  # noqa: E402
    AR_QDFL_FASTSAC_PLACEHOLDER,
    AR_QDFL_FASTSAC_WARMUP_POLICY,
    DEFAULT_AR_QDFL_FASTSAC_GROUP,
    DEFAULT_ONLINE_PLOT_MAX_STEP,
    ONLINE_PHASE_SPLIT_STEP,
    aggregate_single_group,
    per_seed_endpoints,
)
from plot_offline_to_online import (  # noqa: E402
    build_online_series,
    build_series_provenance,
)


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


def test_fastsac_duplicate_offline_end_rows_are_disambiguated(tmp_path: Path):
    """Offline-end + post-warmup both at 1M must both survive (→ 1M and 1M+1)."""
    from plot_dflrql_vs_baseline import parse_eval_csv

    group = DEFAULT_AR_QDFL_FASTSAC_GROUP
    run_dir = tmp_path / "rql" / group / "sd000_a"
    run_dir.mkdir(parents=True)
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
        w.writerow(
            {
                "step": 1_000_000,
                "evaluation/success": 0.68,
                "evaluation/episode.return": -1700.0,
            }
        )
        w.writerow(
            {
                "step": 1_000_000,
                "evaluation/success": 0.50,
                "evaluation/episode.return": -1800.0,
            }
        )
        w.writerow(
            {
                "step": 1_100_000,
                "evaluation/success": 0.55,
                "evaluation/episode.return": -1750.0,
            }
        )

    rows = parse_eval_csv(run_dir / "eval.csv")
    assert [int(r["step"]) for r in rows] == [1_000_000, 1_000_001, 1_100_000]
    assert rows[0]["success"] == 0.68
    assert rows[1]["success"] == 0.50

    agg = aggregate_single_group(
        tmp_path,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        run_group=group,
    )
    assert agg is not None
    steps, mean, *_rest = agg
    assert 1_000_000 in steps
    assert 1_000_001 in steps
    assert float(mean[list(steps).index(1_000_000)]) == pytest.approx(0.68)
    assert float(mean[list(steps).index(1_000_001)]) == pytest.approx(0.50)


def test_fastsac_series_defaults_and_warmup_policy():
    online = {s[0]: s[1] for s in build_online_series()}
    assert online["AR-QDFL + FastSAC"] == AR_QDFL_FASTSAC_PLACEHOLDER
    assert DEFAULT_AR_QDFL_FASTSAC_GROUP.endswith("ar-qdfl-fastsac-2m")
    assert AR_QDFL_FASTSAC_WARMUP_POLICY["mode"] == "hidden"
    assert AR_QDFL_FASTSAC_WARMUP_POLICY["included_in_eval_csv"] is False
    assert AR_QDFL_FASTSAC_WARMUP_POLICY["warmup_updates"] == 100_000
    assert AR_QDFL_FASTSAC_WARMUP_POLICY["immediate_eval_at_offline_end"] is True
    assert ONLINE_PHASE_SPLIT_STEP == 1_000_000
    assert DEFAULT_ONLINE_PLOT_MAX_STEP == 2_000_000


def test_fastsac_single_group_mean_std_three_seeds(tmp_path: Path):
    group = DEFAULT_AR_QDFL_FASTSAC_GROUP
    # Absolute plot steps already hide warmup; include offline + online points.
    _write_eval(
        tmp_path / "rql" / group / "sd000_a",
        [
            (500_000, 0.2, 10.0),
            (1_000_000, 0.4, 20.0),
            (1_500_000, 0.6, 30.0),
            (2_000_000, 0.8, 40.0),
        ],
    )
    _write_eval(
        tmp_path / "rql" / group / "sd001_b",
        [
            (500_000, 0.4, 12.0),
            (1_000_000, 0.6, 22.0),
            (1_500_000, 0.8, 32.0),
            (2_000_000, 1.0, 42.0),
        ],
    )
    _write_eval(
        tmp_path / "rql" / group / "sd002_c",
        [
            (500_000, 0.0, 8.0),
            (1_000_000, 0.2, 18.0),
            (1_500_000, 0.4, 28.0),
            (2_000_000, 0.6, 38.0),
        ],
    )

    agg = aggregate_single_group(
        tmp_path,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        run_group=group,
    )
    assert agg is not None
    steps, mean, std, n_per, n_seeds = agg
    assert n_seeds == 3
    assert float(steps[0]) == 500_000
    assert float(steps[-1]) == 2_000_000
    idx_1m = int(np.where(steps == 1_000_000)[0][0])
    idx_2m = int(np.where(steps == 2_000_000)[0][0])
    assert float(mean[idx_1m]) == pytest.approx(0.4)
    assert float(mean[idx_2m]) == pytest.approx(0.8)
    assert float(n_per[idx_2m]) == 3.0
    # Sample std over 3 seeds at 2M: {0.8, 1.0, 0.6} -> std ≈ 0.2
    assert float(std[idx_2m]) == pytest.approx(0.2)

    endpoints = per_seed_endpoints(tmp_path, group, DEFAULT_ONLINE_PLOT_MAX_STEP)
    assert set(endpoints) == {"0", "1", "2"}
    assert endpoints["0"]["step"] == 2_000_000
    assert endpoints["0"]["success"] == 0.8


def test_fastsac_partial_runs_no_invented_endpoints(tmp_path: Path):
    """Partial seed stops early; unfinished seed with no eval is omitted."""
    group = DEFAULT_AR_QDFL_FASTSAC_GROUP
    _write_eval(
        tmp_path / "rql" / group / "sd000_a",
        [(100_000, 0.1, 5.0), (200_000, 0.2, 8.0)],
    )
    _write_eval(
        tmp_path / "rql" / group / "sd001_b",
        [(100_000, 0.3, 6.0)],
    )
    # Seed 2 dir exists but has no eval.csv yet (unfinished).
    (tmp_path / "rql" / group / "sd002_c").mkdir(parents=True)

    agg = aggregate_single_group(
        tmp_path,
        "success",
        plot_max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        run_group=group,
    )
    assert agg is not None
    steps, mean, std, n_per, n_seeds = agg
    assert n_seeds == 2
    assert float(steps[-1]) == 200_000
    assert float(mean[-1]) == 0.2
    assert float(n_per[np.where(steps == 100_000)[0][0]]) == 2.0
    assert float(n_per[np.where(steps == 200_000)[0][0]]) == 1.0
    # Only one seed at 200k => sample std forced to 0.
    assert float(std[np.where(steps == 200_000)[0][0]]) == 0.0

    endpoints = per_seed_endpoints(tmp_path, group, DEFAULT_ONLINE_PLOT_MAX_STEP)
    assert set(endpoints) == {"0", "1"}
    assert "2" not in endpoints
    assert endpoints["0"]["step"] == 200_000
    assert endpoints["1"]["step"] == 100_000


def test_fastsac_provenance_json_shape(tmp_path: Path):
    group = DEFAULT_AR_QDFL_FASTSAC_GROUP
    _write_eval(
        tmp_path / "rql" / group / "sd000_a",
        [(1_000_000, 0.5, 25.0)],
    )
    entry = build_series_provenance(
        label="AR-QDFL + FastSAC",
        placeholder=AR_QDFL_FASTSAC_PLACEHOLDER,
        exp_root=tmp_path,
        max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        aggregation="single_group",
        warmup_policy=dict(AR_QDFL_FASTSAC_WARMUP_POLICY),
        run_groups={"run_group": group},
    )
    assert entry["plotted"] is True
    assert entry["aggregation"] == "single_group"
    assert entry["run_groups"]["run_group"] == group
    assert entry["warmup_policy"]["mode"] == "hidden"
    assert entry["n_seeds"]["run_group"] == 1
    assert entry["per_seed_endpoints"][group]["0"]["success"] == 0.5
    assert entry["success"] is not None
    assert entry["success"]["steps"] == [1_000_000.0]
    # Round-trip JSON (no NaN / invented fields).
    blob = json.dumps({"series": [entry]})
    loaded = json.loads(blob)
    assert loaded["series"][0]["return"]["mean"] == [25.0]


def test_empty_group_provenance_does_not_invent(tmp_path: Path):
    group = DEFAULT_AR_QDFL_FASTSAC_GROUP
    (tmp_path / "rql" / group / "sd000_a").mkdir(parents=True)
    entry = build_series_provenance(
        label="AR-QDFL + FastSAC",
        placeholder=AR_QDFL_FASTSAC_PLACEHOLDER,
        exp_root=tmp_path,
        max_step=DEFAULT_ONLINE_PLOT_MAX_STEP,
        aggregation="single_group",
        warmup_policy=dict(AR_QDFL_FASTSAC_WARMUP_POLICY),
        run_groups={"run_group": group},
    )
    assert entry["plotted"] is False
    assert entry["success"] is None
    assert entry["return"] is None
    assert entry["per_seed_endpoints"][group] == {}
    assert entry["n_seeds"] == {}
