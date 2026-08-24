#!/usr/bin/env python
"""Mug/kettle-style jittered benches for every Pick-v1.1 object category.

One val-benchmark episode per category (first in the held-out 0-127 slice), 48 train
repeats (seed 0) and 48 eval repeats (seed 1000, unjittered template dropped so the
halves are disjoint), ±2 cm XY jitter. Mug (ep 9) and kettle (ep 0) already have this
construction; this script builds the other 16.

    python scripts/make_pick_object_benches.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import VAL_BENCHMARK, VENV_SIM  # noqa: E402

OUT = ROOT / "assets/benches/pick_objects"
REPEATS = 48
JITTER = 0.02

# First episode of each metadata category in val 0-127. mug/kettle skipped: those
# benches already live under assets/benches (desk_mug, house0_kettle_v21_pose0).
OBJECTS = [
    {"name": "remote", "val_episode": 1, "task": "pick up the remote."},
    {"name": "ladle", "val_episode": 2, "task": "pick up the ladle."},
    {"name": "tissue", "val_episode": 3, "task": "pick up the tissue."},
    {"name": "spoon", "val_episode": 5, "task": "pick up the spoon."},
    {"name": "spatula", "val_episode": 6, "task": "pick up the spatula."},
    {"name": "pot", "val_episode": 10, "task": "pick up the pot."},
    {"name": "soap_dispenser", "val_episode": 12, "task": "pick up the bottle."},
    {"name": "spray_bottle", "val_episode": 14, "task": "pick up the bottle."},
    {"name": "cup", "val_episode": 18, "task": "pick up the cup."},
    {"name": "shaker", "val_episode": 20, "task": "pick up the pepper."},
    {"name": "fork", "val_episode": 29, "task": "pick up the fork."},
    {"name": "bottle", "val_episode": 41, "task": "pick up the bottle."},
    {"name": "fruit", "val_episode": 43, "task": "pick up the apple."},
    {"name": "bowl", "val_episode": 44, "task": "pick up the bowl."},
    {"name": "knife", "val_episode": 54, "task": "pick up the knife."},
    {"name": "box", "val_episode": 55, "task": "pick up the box."},
]


def build(episode: int, out: Path, seed: int, extra: int = 0) -> None:
    subprocess.run(
        [
            str(VENV_SIM),
            str(ROOT / "scripts/make_repeat_benchmark.py"),
            "--source", str(VAL_BENCHMARK),
            "--episode-index", str(episode),
            "--repeats", str(REPEATS + extra),
            "--jitter", str(JITTER),
            "--seed", str(seed),
            "--out", str(out),
        ],
        check=True,
    )
    if extra:
        path = out / "benchmark.json"
        episodes = json.loads(path.read_text())
        path.write_text(json.dumps(episodes[extra:]))


def main() -> None:
    val = json.loads((VAL_BENCHMARK / "benchmark.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in OBJECTS:
        ep = val[spec["val_episode"]]
        house = int(ep["house_index"])
        asset = ep["task"]["pickup_obj_name"]
        lang = ep["language"]["task_description"]
        assert lang.strip().lower().rstrip(".") + "." == spec["task"], (lang, spec["task"])
        dest = OUT / spec["name"]
        train, evald = dest / "train", dest / "eval"
        print(f"\n=== {spec['name']} ep={spec['val_episode']} house={house} {lang!r} ===")
        if not (train / "benchmark.json").exists():
            build(spec["val_episode"], train, seed=0)
        else:
            print(f"  skip existing {train}")
        if not (evald / "benchmark.json").exists():
            build(spec["val_episode"], evald, seed=1000, extra=1)
        else:
            print(f"  skip existing {evald}")
        records.append(
            {
                **spec,
                "house": house,
                "asset": asset,
                    "gate_step": 0,
                "horizon": 500,
                "note": (
                    f"Pick-v1.1 object dataset, mug/kettle construction: first held-out "
                    f"episode (val {spec['val_episode']}, house {house}), ±2 cm XY jitter, "
                    f"48 train / 48 eval. Frozen pi0.5 collect of 100 train trajectories. "
                    f"RL from env step 0 (set gate_step=N for a frozen-VLA prefix)."
                ),
            }
        )
    manifest = {
        "source": str(VAL_BENCHMARK),
        "jitter_m": JITTER,
        "train_repeats": REPEATS,
        "eval_repeats": REPEATS,
        "train_seed": 0,
        "eval_seed": 1000,
        "skipped": ["mug (desk_mug ep 9)", "kettle (house0 ep 0)"],
        "objects": records,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {OUT / 'manifest.json'} ({len(records)} objects)")


if __name__ == "__main__":
    main()
