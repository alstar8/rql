"""Build a benchmark that repeats several candidate episodes, to measure their success rates.

Choosing an RL scene by a single rollout is guesswork: the sampler is stochastic, so one
success tells you almost nothing about the rate. This repeats each candidate with the same
small positional jitter used for the RL scene, so the measured rate reflects what RL would
actually face.

Each variant records which candidate it came from, so the rollouts can be grouped again
afterwards even though the evaluator reorganises everything by house.

    python scripts/make_profile_benchmark.py --episodes 3 17 42 --repeats 12 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

VAL_BENCHMARK = Path(
    "/home/jovyan/users/staroverov/B1K/mlspaces/cache/benchmarks/molmospaces-bench-v1/20260408"
    "/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=VAL_BENCHMARK)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--jitter", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    source = json.loads((args.source / "benchmark.json").read_text())
    rng = np.random.default_rng(args.seed)

    variants = []
    manifest = []
    for episode_index in args.episodes:
        template = source[episode_index]
        task = template["task"]
        start = list(task["pickup_obj_start_pose"])
        print(f"episode {episode_index}: house {template['house_index']:>3}  "
              f"{template['language']['task_description']!r}")

        for repeat in range(args.repeats):
            variant = json.loads(json.dumps(template))
            offset = np.zeros(2) if repeat == 0 else rng.uniform(-args.jitter, args.jitter, size=2)
            pose = list(start)
            pose[0] = float(start[0] + offset[0])
            pose[1] = float(start[1] + offset[1])
            variant["task"]["pickup_obj_start_pose"] = pose

            scene = variant.get("scene_modifications") or {}
            poses = scene.get("object_poses") or {}
            name = task["pickup_obj_name"]
            if name in poses:
                existing = list(poses[name])
                existing[0], existing[1] = pose[0], pose[1]
                poses[name] = existing
                scene["object_poses"] = poses
                variant["scene_modifications"] = scene

            manifest.append(
                {
                    "variant": len(variants),
                    "source_episode": episode_index,
                    "house_index": template["house_index"],
                    "task": template["language"]["task_description"],
                    "object": name,
                }
            )
            variants.append(variant)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(variants))
    (args.out / "profile_manifest.json").write_text(json.dumps(manifest, indent=2))
    src_meta = args.source / "benchmark_metadata.json"
    if src_meta.exists():
        shutil.copy(src_meta, args.out / "benchmark_metadata.json")

    print(f"\nwrote {len(variants)} variants "
          f"({len(args.episodes)} candidates x {args.repeats} repeats) to {args.out}")


if __name__ == "__main__":
    main()
