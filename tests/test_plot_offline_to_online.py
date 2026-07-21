"""Focused tests for offline vs offline→online plot series separation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_dflrql_vs_baseline import (  # noqa: E402
    AR_QDFL_FASTSAC_PLACEHOLDER,
    CF_NOCRF_PLACEHOLDER,
    CF_PLACEHOLDER,
    CORE_SCRATCH_GROUPS,
    DEFAULT_AR_QDFL_FASTSAC_GROUP,
    DEFAULT_ONLINE_PLOT_MAX_STEP,
    DEFAULT_PLOT_MAX_STEP,
    ONLINE_PHASE_SPLIT_STEP,
    PURE_QFLOW_PLACEHOLDER,
    QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
    RQL_ONLINE_PLACEHOLDER,
    RQL_QFLOW_ACTORFREEZE_PLACEHOLDER,
    RQL_QFLOW_ONLINE_PLACEHOLDER,
    build_series,
)
from plot_offline_to_online import ONLINE_SERIES, build_online_series  # noqa: E402


def test_offline_core_excludes_qflow_online_arms():
    core = build_series("clf-group", all_methods=False)
    labels = [s[0] for s in core]
    groups = [s[1] for s in core]
    assert labels == [
        "RQL baseline",
        "DFL-RQL v9",
        "Quantized DFL-RQL v9",
        "DARI (AR OAT)",
        "CDF (discrete FM)",
        "AR QDFL student",
        "DD QDFL student",
    ]
    assert groups[1:] == list(CORE_SCRATCH_GROUPS)
    for placeholder in (
        RQL_QFLOW_ONLINE_PLACEHOLDER,
        RQL_QFLOW_ACTORFREEZE_PLACEHOLDER,
        QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
        PURE_QFLOW_PLACEHOLDER,
        AR_QDFL_FASTSAC_PLACEHOLDER,
    ):
        assert placeholder not in groups
    assert DEFAULT_PLOT_MAX_STEP == 2_000_000


def test_online_series_is_1m_plus_1m_only():
    online = build_online_series()
    by_label = {s[0]: s[1] for s in online}
    assert "RQL→Q-Flow online" not in by_label
    assert "RQL→Q-Flow actor-freeze" not in by_label
    assert RQL_QFLOW_ONLINE_PLACEHOLDER not in by_label.values()
    assert RQL_QFLOW_ACTORFREEZE_PLACEHOLDER not in by_label.values()
    assert by_label["RQL"] == RQL_ONLINE_PLACEHOLDER
    assert by_label["CF"] == CF_PLACEHOLDER
    assert by_label["CF no-CRF"] == CF_NOCRF_PLACEHOLDER
    assert by_label["Q-Flow RQL warmstart"] == QFLOW_RQL_WARMSTART_V2_PLACEHOLDER
    assert by_label["Pure Q-Flow"] == PURE_QFLOW_PLACEHOLDER
    assert by_label["AR-QDFL + FastSAC"] == AR_QDFL_FASTSAC_PLACEHOLDER
    assert online == list(ONLINE_SERIES)
    assert len(online) == 6
    assert DEFAULT_ONLINE_PLOT_MAX_STEP == 2_000_000
    assert ONLINE_PHASE_SPLIT_STEP == 1_000_000
    assert DEFAULT_AR_QDFL_FASTSAC_GROUP == "humanoidmaze-large-ar-qdfl-fastsac-2m"


def test_all_methods_keeps_1m_qflow_placeholders():
    all_m = build_series("clf-group", all_methods=True)
    all_groups = {s[0]: s[1] for s in all_m}
    assert "RQL→Q-Flow online (phase1+2)" not in all_groups
    assert (
        all_groups["Q-Flow RQL warmstart (phase1+2)"]
        == QFLOW_RQL_WARMSTART_V2_PLACEHOLDER
    )
    assert all_groups["Pure Q-Flow (phase1+2)"] == PURE_QFLOW_PLACEHOLDER
    # FastSAC is offline→online only; not injected into offline all-methods.
    assert AR_QDFL_FASTSAC_PLACEHOLDER not in all_groups.values()
