#!/usr/bin/env python
"""Turn a measured candidate into a scene: build its benchmarks, print its config entry.

    scripts/promote_scene.py --episode 25 --name sink_dispenser \
        --note "3/12 = 25% probed; reaches the dispenser and stalls without closing"

A candidate only becomes a scene once it has been measured in the current regime and its
failure mode has been looked at. This does the mechanical half -- two jittered benchmarks
and the dict entry -- and refuses to invent the half that has to come from measurement:
the note is required, because a scene whose entry does not say what was measured is how
the last selection became unreproducible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import ROOT as DATA_ROOT  # noqa: E402
from pi05.config import VAL_BENCHMARK, VENV_SIM  # noqa: E402


def episode_meta(index: int) -> dict:
    episodes = json.loads((VAL_BENCHMARK / "benchmark.json").read_text())
    entry = episodes[index]
    return {
        "task": entry["language"]["task_description"],
        "house": entry.get("house_index", entry.get("scene", {}).get("house_index", -1)),
        "object": entry["task"].get("pickup_obj_name", "?"),
    }


def build(episode: int, out: Path, repeats: int, seed: int, extra: int = 0) -> None:
    import os

    subprocess.run(
        [
            str(VENV_SIM), str(ROOT / "scripts/make_repeat_benchmark.py"),
            "--episode-index", str(episode),
            "--repeats", str(repeats + extra),
            "--jitter", "0.02",
            "--seed", str(seed),
            "--out", str(out),
        ],
        check=True, capture_output=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if extra:
        # Repeat 0 is the unjittered template and is identical in every seed, so the
        # eval half is built one long and that one dropped, leaving the two disjoint.
        path = out / "benchmark.json"
        episodes = json.loads(path.read_text())
        path.write_text(json.dumps(episodes[extra:]))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--episode", type=int, required=True, help="index into the val benchmark")
    ap.add_argument("--name", required=True, help="scene name, e.g. sink_dispenser")
    ap.add_argument("--note", required=True, help="what was measured, and how it fails")
    ap.add_argument("--repeats", type=int, default=48)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = episode_meta(args.episode)
    train = DATA_ROOT / f"bench_{args.name}_train{args.repeats}"
    evald = DATA_ROOT / f"bench_{args.name}_eval{args.repeats}"

    print(f"episode {args.episode}: {meta['task']!r}")
    print(f"  house  {meta['house']}")
    print(f"  object {meta['object']}")
    print(f"  train  {train}")
    print(f"  eval   {evald}")

    if not args.dry_run:
        build(args.episode, train, args.repeats, seed=0)
        build(args.episode, evald, args.repeats, seed=1000, extra=1)
        print("  benchmarks built")

    task = meta["task"].strip().lower()
    if not task.endswith("."):
        task += "."
    print("\npaste into config.SCENES:\n")
    print(f'    "{args.name}": Scene(')
    print(f'        name="{args.name}",')
    print(f"        val_episode={args.episode},")
    print(f"        house={meta['house']},")
    print(f'        task="{task}",')
    print(f'        benchmark_train=ROOT / "{train.name}",')
    print(f'        benchmark_eval=ROOT / "{evald.name}",')
    print(f'        note="{args.note}",')
    print("    ),")


if __name__ == "__main__":
    main()
