"""Reconstruct the experiment_config.pkl that create_json_benchmark.py expects.

The bulk downloader ships the trajectories but not the experiment config, and without it
the benchmark builder refuses to start. It reads only four things from that file:
scene_dataset, data_split, the camera system's class name and its image resolution.

None of them are invented here. scene_dataset and data_split come from PickBaseConfig,
which is the class the demonstrations were generated with; the camera config is the real
object taken out of a trajectory's own frozen_config, so its class name and resolution
are whatever actually produced the recordings.

    python scripts/make_experiment_config.py --houses-dir <dir of house_* dirs>
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
import types
from pathlib import Path

import h5py

MOLMOSPACES = Path("/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces")
sys.path.insert(0, str(MOLMOSPACES / "scripts" / "benchmarks"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--houses-dir", type=Path, required=True)
    ap.add_argument("--scene-dataset", default=None, help="default: read from PickBaseConfig")
    ap.add_argument("--data-split", default=None, help="default: read from PickBaseConfig")
    args = ap.parse_args()

    from create_json_benchmark import extract_frozen_config

    from molmo_spaces.configs.base_pick_config import PickBaseConfig

    fields = PickBaseConfig.model_fields
    scene_dataset = args.scene_dataset or fields["scene_dataset"].default
    data_split = args.data_split or fields["data_split"].default

    h5_files = sorted(glob.glob(str(args.houses_dir / "house_*" / "trajectories_batch_*.h5")))
    if not h5_files:
        raise SystemExit(f"no trajectory files under {args.houses_dir}")

    with h5py.File(h5_files[0], "r") as f:
        traj_key = next(k for k in f if k.startswith("traj_"))
        scene = json.loads(f[traj_key]["obs_scene"][()].decode("utf-8"))
    frozen = extract_frozen_config(scene)
    camera_config = frozen.camera_config

    resolution = getattr(camera_config, "img_resolution", None)
    if resolution is None:
        raise SystemExit("camera_config has no img_resolution; refusing to guess one")

    exp_config = types.SimpleNamespace(
        scene_dataset=scene_dataset,
        data_split=data_split,
        camera_config=camera_config,
    )

    out = args.houses_dir / "experiment_config.pkl"
    with out.open("wb") as handle:
        pickle.dump(exp_config, handle)

    print(f"scene_dataset     {scene_dataset}   (from PickBaseConfig)")
    print(f"data_split        {data_split}   (from PickBaseConfig)")
    print(f"camera system     {type(camera_config).__name__}   (from a trajectory's frozen_config)")
    print(f"img_resolution    {tuple(resolution)}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
