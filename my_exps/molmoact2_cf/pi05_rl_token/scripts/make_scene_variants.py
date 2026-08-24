"""One scene, many starting conditions: a benchmark of variants of one episode.

Training on 872 different scenes gave the critic nothing to compare, because a
state is never revisited. Training on one fixed episode gave it plenty, but only
for that exact configuration. This builds the middle case: the same house, the
same object, the same camera placement, with the object and the arm starting
somewhere different every episode.

Both ranges come from MolmoSpaces itself, not from a guess:

* the object is placed by `place_object_near`, the same routine that built the
  benchmark, with the Pick sampler's own numbers -- 0.15 to 0.5 m from where it
  started, no further than 0.7 m from the robot base, constrained to the
  supporting surface, with a random yaw;
* the arm starts within +-0.05 rad per joint of the benchmark's home pose, which
  is exactly the spread measured across the benchmark's own 1000 episodes
  (uniform, std 0.029 = 0.1/sqrt(12)).

    python scripts/make_scene_variants.py --episode_idx 134 --variants 1200 \
        --out runs/variants/ep134

Writes a benchmark directory that `--benchmark_dir` accepts, holding `variants`
episodes that differ only in those two things.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlt.mlspaces_env import configure_assets  # noqa: E402

# The generator's own placement parameters, from PickTaskSamplerConfig.
MIN_DIST = 0.15
MAX_DIST = 0.5
MAX_DIST_TO_ROBOT = 0.7
MAX_TRIES = 100
Z_EPS = 0.003
# Measured across the benchmark's 1000 episodes: uniform, +-0.05 rad per joint.
ARM_JITTER = 0.05


def sample_arm(home: np.ndarray, rng: np.random.Generator) -> list[float]:
    return (home + rng.uniform(-ARM_JITTER, ARM_JITTER, size=home.shape)).tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode_idx", type=int, required=True, help="benchmark episode to vary")
    ap.add_argument("--variants", type=int, default=1200)
    ap.add_argument("--out", required=True, help="directory for the new benchmark")
    ap.add_argument("--benchmark_dir", default="")
    ap.add_argument("--assets_dir", default="/home/jovyan/users/staroverov/B1K/mlspaces/assets")
    ap.add_argument("--cache_dir", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    configure_assets(args.assets_dir, args.cache_dir)

    from molmo_spaces.env.object_manager import ObjectManager  # noqa: F401  (import order)
    from molmo_spaces.evaluation.benchmark_schema import EpisodeSpec
    from molmo_spaces.tasks.json_eval_task_sampler import JsonEvalTaskSampler
    from molmo_spaces.utils.mj_model_and_data_utils import body_base_pos
    from molmo_spaces.utils.mujoco_scene_utils import get_supporting_geom, place_object_near

    from rlt.eval_config import RLTokenEvalConfig, default_benchmark_dir

    bench = Path(args.benchmark_dir) if args.benchmark_dir else default_benchmark_dir()
    episodes = json.loads((bench / "benchmark.json").read_text())
    spec_dict = episodes[args.episode_idx]
    spec = EpisodeSpec(**spec_dict)
    object_name = spec.task["pickup_obj_name"]
    print(f"episode {args.episode_idx}: house {spec.house_index}, object {object_name}")

    exp_config = RLTokenEvalConfig()
    exp_config.benchmark_path = bench
    sampler = JsonEvalTaskSampler(exp_config, spec)
    task = sampler.sample_task(house_index=spec.house_index)
    if task is None:
        raise RuntimeError("the scene could not be built; nothing to vary")

    env = task.env
    manager = env.object_managers[env.current_batch_index]
    body_id = manager.get_object_body_id(object_name)
    support = get_supporting_geom(env.current_data, body_id)
    if support is None:
        raise RuntimeError(
            f"{object_name} rests on nothing the heuristic recognises, so its surface is unknown"
        )

    start = np.asarray(spec.task["pickup_obj_start_pose"], dtype=float)
    robot_base = np.asarray(spec.task["robot_base_pose"][:3], dtype=float)
    home = np.asarray(spec.robot.init_qpos["arm"], dtype=float)
    rng = np.random.default_rng(args.seed)

    variants: list[dict] = []
    failures = 0
    for i in range(args.variants):
        try:
            place_object_near(
                data=env.current_data,
                object_id=body_id,
                placement_point=start[:3].copy(),
                min_dist=MIN_DIST,
                max_dist=MAX_DIST,
                max_tries=MAX_TRIES,
                reference_pos=robot_base,
                max_dist_to_reference=MAX_DIST_TO_ROBOT,
                supporting_geom_id=support,
                z_eps=Z_EPS,
            )
        except Exception as error:  # noqa: BLE001 -- ObjectPlacementError and friends
            failures += 1
            if failures <= 3:
                print(f"  placement {i} failed: {error}")
            continue

        placed = manager.get_object(object_name)
        pose = np.concatenate([placed.position, placed.quat]).tolist()
        base_z = float(body_base_pos(env.current_data, body_id)[2])

        variant = json.loads(json.dumps(spec_dict))  # deep copy of the original episode
        variant["scene_modifications"]["object_poses"][object_name] = pose
        variant["task"]["pickup_obj_start_pose"] = pose
        variant["robot"]["init_qpos"]["arm"] = sample_arm(home, rng)
        variant["seed"] = int(rng.integers(0, 2**31 - 1))
        variant["source"] = {**(spec_dict.get("source") or {}), "variant_of": args.episode_idx}
        variants.append(variant)

        if i < 3 or i == args.variants - 1:
            moved = float(np.linalg.norm(np.asarray(pose[:3]) - start[:3]))
            print(f"  variant {i}: moved {moved:.3f} m, base z {base_z:.4f} (was {start[2]:.4f})")

    if not variants:
        raise RuntimeError("no placement succeeded; the surface may be too small for this object")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(variants))
    (out / "benchmark_metadata.json").write_text(
        json.dumps(
            {
                "description": f"variants of episode {args.episode_idx} of {bench.name}",
                "num_episodes": len(variants),
                "source_episode": args.episode_idx,
                "house_index": spec.house_index,
                "object": object_name,
                "placement": {
                    "min_dist": MIN_DIST,
                    "max_dist": MAX_DIST,
                    "max_dist_to_robot": MAX_DIST_TO_ROBOT,
                    "arm_jitter_rad": ARM_JITTER,
                },
                "seed": args.seed,
            },
            indent=2,
        )
    )
    print(f"wrote {len(variants)} variants to {out} ({failures} placements failed)")


if __name__ == "__main__":
    main()
