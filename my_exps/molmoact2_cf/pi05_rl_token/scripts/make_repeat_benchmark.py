"""Build a benchmark of one episode repeated many times with the object nudged slightly.

RL Token needs many rollouts of the same scene, and identical repeats would only sample
the policy's own stochasticity. A small positional jitter on the pickup object varies the
task itself without changing what it is, which is what makes the collected data useful
for learning a grasp correction rather than memorising one pose.

Only the horizontal position moves. Height is left alone so the object stays on its
surface, and orientation is left alone so an elongated object keeps the approach
direction the policy was trained to expect.

    python scripts/make_repeat_benchmark.py --episode-index 117 --repeats 64 --jitter 0.02
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
    ap.add_argument("--episode-index", type=int, required=True, help="index into the val benchmark")
    ap.add_argument("--repeats", type=int, default=64)
    ap.add_argument("--jitter", type=float, default=0.02, help="metres, uniform in x and y")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    episodes = json.loads((args.source / "benchmark.json").read_text())
    template = episodes[args.episode_index]
    task = template["task"]
    start = list(task["pickup_obj_start_pose"])

    print(f"episode {args.episode_index}: {template['language']['task_description']!r}")
    print(f"  house {template['house_index']}  object {task['pickup_obj_name']}")
    print(f"  start pose {[round(v, 3) for v in start[:3]]}")
    print(f"  jitter +-{args.jitter} m in x,y; z and orientation unchanged")

    rng = np.random.default_rng(args.seed)
    variants = []
    for i in range(args.repeats):
        variant = json.loads(json.dumps(template))  # deep copy
        if i == 0:
            offset = np.zeros(2)  # keep one unperturbed run as the reference
        else:
            offset = rng.uniform(-args.jitter, args.jitter, size=2)
        pose = list(start)
        pose[0] = float(start[0] + offset[0])
        pose[1] = float(start[1] + offset[1])
        variant["task"]["pickup_obj_start_pose"] = pose
        scene = variant.get("scene_modifications") or {}
        # object_poses is what the sampler actually applies when placing the scene, so the
        # jitter has to land there too or the object would spawn back at its original spot.
        poses = scene.get("object_poses") or {}
        name = task["pickup_obj_name"]
        if name in poses:
            existing = list(poses[name])
            existing[0] = pose[0]
            existing[1] = pose[1]
            poses[name] = existing
            scene["object_poses"] = poses
            variant["scene_modifications"] = scene
        elif i == 1:
            print(f"  note: {name} absent from scene_modifications.object_poses "
                  f"({len(poses)} entries); jitter applies through task spec only")
        variants.append(variant)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark.json").write_text(json.dumps(variants))
    for extra in ("benchmark_metadata.json",):
        src = args.source / extra
        if src.exists():
            shutil.copy(src, args.out / extra)

    spread = np.array([v["task"]["pickup_obj_start_pose"][:2] for v in variants])
    print(f"\nwrote {len(variants)} variants to {args.out}")
    print(f"  x range {spread[:,0].min():.3f}..{spread[:,0].max():.3f}")
    print(f"  y range {spread[:,1].min():.3f}..{spread[:,1].max():.3f}")


if __name__ == "__main__":
    main()
