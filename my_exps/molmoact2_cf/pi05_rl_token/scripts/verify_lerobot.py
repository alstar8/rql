"""Cross-check a converted LeRobot dataset against the demonstrations it came from.

Reads samples back through the same loader openpi will use and compares them, value by
value, with the raw h5. This is the step that catches a shifted frame, a flipped delta
sign or a mis-scaled gripper -- none of which show up in a training curve.

    python scripts/verify_lerobot.py --dataset <dir> --source <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.action_space import GRIPPER_CMD_MAX, absolute_arm_from_delta, normalize_instruction  # noqa: E402
from pi05.demo_reader import iter_episodes  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--houses", nargs="*", default=None)
    ap.add_argument("--dump-images", type=Path, default=None)
    args = ap.parse_args()

    ds = LeRobotDataset("verify/pick", root=str(args.dataset))
    print(f"dataset: {ds.num_episodes} episodes, {ds.num_frames} frames, fps={ds.fps}")
    print(f"features: {sorted(ds.features)}\n")

    episodes = list(iter_episodes(args.source, houses=args.houses))
    print(f"source episodes available: {len(episodes)}\n")

    # The converter writes episodes in iteration order, so episode i of the dataset is
    # episodes[i] of the source.
    worst_state = worst_action = worst_grip = 0.0
    checked = 0

    for ep_index in range(min(3, ds.num_episodes)):
        src = episodes[ep_index]
        start = int(ds.episode_data_index["from"][ep_index])
        end = int(ds.episode_data_index["to"][ep_index])
        n = end - start
        print(f"--- episode {ep_index}: {src.episode_id}")
        print(f"    frames in dataset {n}, usable steps in source {src.n_steps}")
        assert n == src.n_steps, "episode length disagrees with the source trajectory"

        expected_prompt = normalize_instruction(src.instruction)
        print(f"    instruction: {src.instruction!r} -> {expected_prompt!r}")

        for offset in (0, n // 2, n - 1):
            item = ds[start + offset]
            task = item.get("task")
            assert task == expected_prompt, f"prompt mismatch: {task!r} != {expected_prompt!r}"

            raw_arm = np.asarray(src.qpos[offset]["arm"][:7], dtype=np.float32)
            raw_cmd = np.asarray(src.commands[offset]["arm"][:7], dtype=np.float32)
            raw_grip_cmd = float(src.commands[offset]["gripper"][0])

            state = np.asarray(item["joint_position"], dtype=np.float32)
            action = np.asarray(item["actions"], dtype=np.float32)

            worst_state = max(worst_state, float(np.abs(state - raw_arm).max()))
            # The delta must reconstruct exactly the absolute target the sim executed.
            rebuilt = absolute_arm_from_delta(action, state)
            worst_action = max(worst_action, float(np.abs(rebuilt - raw_cmd).max()))
            worst_grip = max(
                worst_grip, abs(float(action[7]) * GRIPPER_CMD_MAX - raw_grip_cmd)
            )
            checked += 1

            img = np.asarray(item["exterior_image_1_left"])
            print(
                f"    step {offset:3d}: state[3]={state[3]:+.4f} (raw {raw_arm[3]:+.4f})  "
                f"delta[3]={action[3]:+.5f}  grip={action[7]:.2f} (raw {raw_grip_cmd:.0f})  "
                f"img{tuple(img.shape)} {img.dtype}"
            )

            if args.dump_images is not None and offset == n // 2:
                args.dump_images.mkdir(parents=True, exist_ok=True)
                arr = img
                if arr.ndim == 3 and arr.shape[0] == 3:  # loader may hand back CHW floats
                    arr = np.transpose(arr, (1, 2, 0))
                if arr.dtype != np.uint8:
                    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(arr).save(args.dump_images / f"ep{ep_index}_exo.png")
                wrist = np.asarray(item["wrist_image_left"])
                if wrist.ndim == 3 and wrist.shape[0] == 3:
                    wrist = np.transpose(wrist, (1, 2, 0))
                if wrist.dtype != np.uint8:
                    wrist = (np.clip(wrist, 0, 1) * 255).astype(np.uint8)
                Image.fromarray(wrist).save(args.dump_images / f"ep{ep_index}_wrist.png")
        print()

    print(f"checked {checked} frames")
    print(f"  max |state - raw qpos|                 = {worst_state:.2e}")
    print(f"  max |delta + state - raw command|      = {worst_action:.2e}")
    print(f"  max |gripper*255 - raw gripper command| = {worst_grip:.2e}")
    ok = worst_state < 1e-5 and worst_action < 1e-4 and worst_grip < 1e-2
    print("\nVERDICT:", "consistent with the source" if ok else "MISMATCH -- do not train on this")


if __name__ == "__main__":
    main()
