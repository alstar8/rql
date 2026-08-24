"""Replay demonstrations through the running policy server and compare the actions.

This is the wiring test without the simulator. It exercises the whole chain the model
depends on -- the same image preprocessing, state layout, prompt wording and delta
convention used at training, through the websocket server and its delta-to-absolute
conversion -- and checks the result against the absolute joint targets the simulator
actually executed in the demonstration.

A model trained to memorise these episodes must reproduce them here. If it does, the
data path and the action space agree end to end; if it does not, no amount of simulator
debugging will help.

    python scripts/replay_check.py --house-dir <house_0 dir> --episodes 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openpi_client import websocket_client_policy  # noqa: E402

from pi05.action_space import GRIPPER_CMD_MAX, normalize_instruction, state_from_qpos  # noqa: E402
from pi05.convert import _resize  # noqa: E402
from pi05.demo_reader import iter_episodes, read_video  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="dir containing house_* dirs")
    ap.add_argument("--houses", nargs="*", default=["house_0"])
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=12, help="steps sampled per episode")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"connected to {args.host}:{args.port}\n")

    arm_errors, grip_errors = [], []

    for count, episode in enumerate(iter_episodes(args.source, houses=args.houses)):
        if count >= args.episodes:
            break
        exo = read_video(episode.exo_video, episode.n_steps + 1)
        wrist = read_video(episode.wrist_video, episode.n_steps + 1)
        prompt = normalize_instruction(episode.instruction)
        print(f"--- {episode.episode_id}: {prompt!r}")

        # Skip the first transition: it is the jump from the reset pose and was excluded
        # from training, so the model was never shown it.
        offsets = np.linspace(1, episode.n_steps - 1, args.steps, dtype=int)
        for t in offsets:
            state = state_from_qpos(episode.qpos[t])
            observation = {
                "observation/exterior_image_1_left": _resize(exo[t]),
                "observation/wrist_image_left": _resize(wrist[t]),
                "observation/joint_position": state[:7].astype(np.float32),
                "observation/gripper_position": state[7:].astype(np.float32),
                "prompt": prompt,
            }
            predicted = np.asarray(client.infer(observation)["actions"])[0]

            recorded_arm = np.asarray(episode.commands[t]["arm"][:7], dtype=np.float32)
            recorded_grip = float(episode.commands[t]["gripper"][0]) / GRIPPER_CMD_MAX

            arm_error = float(np.abs(predicted[:7] - recorded_arm).max())
            grip_error = abs(float(predicted[7]) - recorded_grip)
            arm_errors.append(arm_error)
            grip_errors.append(grip_error)

        print(f"    steps {len(offsets)}: median arm error "
              f"{np.median(arm_errors[-len(offsets):]):.4f} rad")

    arm_errors = np.array(arm_errors)
    grip_errors = np.array(grip_errors)
    print(f"\nchecked {len(arm_errors)} steps")
    print(f"  arm  |predicted - executed|  median {np.median(arm_errors):.4f}  "
          f"p90 {np.quantile(arm_errors, 0.9):.4f}  max {arm_errors.max():.4f} rad")
    print(f"  grip |predicted - executed|  median {np.median(grip_errors):.4f}")
    print("\nFor scale: consecutive demonstration steps differ by about 0.023 rad, and a")
    print("policy that ignored its input entirely would sit around 0.3-1.2 rad off.")


if __name__ == "__main__":
    main()
