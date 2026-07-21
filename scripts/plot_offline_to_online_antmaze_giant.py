#!/usr/bin/env python3
"""Plot Antmaze-giant offline→online Q-Flow curves (success + return).

Writes my_exps/offline_to_online_antmaze_giant.png.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT = SCRIPT_DIR.parent / "my_exps" / "offline_to_online_antmaze_giant.png"


def main() -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "plot_offline_to_online.py"),
        "--out",
        str(OUT),
        "--title",
        "antmaze-giant-navigate-singletask-v0",
        # Prefer Antmaze-specific groups (skip missing series gracefully).
        "--rql-phase1-run-group",
        "antmaze-giant-rql-offline-1m",
        "--rql-phase2-run-group",
        "antmaze-giant-rql-online-2m",
        # Full CF defaults to HumanoidMaze-large; do not mix that into Antmaze.
        # These groups are intentionally absent until matched Antmaze CF exists.
        "--cf-phase1-run-group",
        "antmaze-giant-cf-offline-1m",
        "--cf-phase2-run-group",
        "antmaze-giant-cf-online-2m",
        "--cf-nocrf-phase1-run-group",
        "antmaze-giant-cf-nocrf-offline-1m",
        "--cf-nocrf-phase2-run-group",
        "antmaze-giant-cf-nocrf-online-2m",
        "--qflow-v2-phase1-run-group",
        "antmaze-giant-qflow-rql-warmstart-1m",
        "--qflow-v2-phase2-run-group",
        "antmaze-giant-qflow-rql-online-2m",
        "--pure-qflow-phase1-run-group",
        "antmaze-giant-qflow-offline-1m",
        "--pure-qflow-phase2-run-group",
        "antmaze-giant-qflow-online-2m",
        # Do not mix the HumanoidMaze-only FastSAC experiment into Antmaze.
        # This group is intentionally absent until a matched Antmaze run exists.
        "--ar-qdfl-fastsac-run-group",
        "antmaze-giant-ar-qdfl-fastsac-2m",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
