"""Focused tests for offline vs offline→online plot series separation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from plot_dflrql_vs_baseline import (  # noqa: E402
    CORE_SCRATCH_GROUPS,
    DEFAULT_ONLINE_PLOT_MAX_STEP,
    DEFAULT_PLOT_MAX_STEP,
    PURE_QFLOW_PLACEHOLDER,
    QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
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
    ):
        assert placeholder not in groups
    assert DEFAULT_PLOT_MAX_STEP == 2_000_000


def test_online_series_includes_legacy_freeze_v2_pure():
    online = build_online_series()
    by_label = {s[0]: s[1] for s in online}
    assert by_label["RQL→Q-Flow online"] == RQL_QFLOW_ONLINE_PLACEHOLDER
    assert by_label["RQL→Q-Flow actor-freeze"] == RQL_QFLOW_ACTORFREEZE_PLACEHOLDER
    assert by_label["Q-Flow RQL warmstart v2"] == QFLOW_RQL_WARMSTART_V2_PLACEHOLDER
    assert by_label["Pure Q-Flow"] == PURE_QFLOW_PLACEHOLDER
    assert online == list(ONLINE_SERIES)
    assert DEFAULT_ONLINE_PLOT_MAX_STEP == 4_100_000


def test_all_methods_keeps_qflow_placeholders():
    all_m = build_series("clf-group", all_methods=True)
    all_groups = {s[0]: s[1] for s in all_m}
    assert all_groups["RQL→Q-Flow online (phase1+2)"] == RQL_QFLOW_ONLINE_PLACEHOLDER
    assert (
        all_groups["Q-Flow RQL warmstart v2 (3-phase)"]
        == QFLOW_RQL_WARMSTART_V2_PLACEHOLDER
    )
    assert all_groups["Pure Q-Flow (phase1+2)"] == PURE_QFLOW_PLACEHOLDER
