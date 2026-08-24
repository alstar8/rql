#!/usr/bin/env python
"""Does the policy go for the object it was asked for?

    scripts/check_target.py --run <run>/eval_output

A scene at ~0% is only a useful RL target when the GRASP is what fails. If the policy
reaches for the wrong object instead, no correction applied to the action chunk can fix
it -- the model decided what to look at before the chunk existed.

Judging that by eye does not work, and this exists because it already went wrong: the
`wine_bottle` candidate was accepted on the strength of a wrist frame showing a gold
object squarely between the fingers. That object was not the wine bottle; the bottle was
the dark green thing at the very edge of the same frame.

MolmoSpaces records the ground truth, and it was there the whole time. Every trajectory
carries, per step and per camera:

    obs/extra/object_image_points/pickup_obj/<camera>/points   (T, 10, 2), normalised
    obs/extra/object_image_points/gripper/<camera>/points      the same for the hand

Two numbers per rollout, over the last quarter of the episode where the outcome settles:

    visible   fraction of steps the target appears in the wrist view
    gap       median distance between target and hand centroids, normalised
              (0 = same spot, ~1.4 = opposite corners)

CALIBRATION -- the thresholds below are measured, not guessed. The first version of this
script used a guessed 0.35 and flagged successful grasps as wrong-object. A success is
proof the policy reached what it was asked for, so successes define "on target":

    desk_mug        38 successes: gap median 0.43 (p10 0.39, p90 0.52), visible 1.00
    sink_dispenser  14 successes: gap median 0.37 (p90 0.44),           visible 1.00

The gap never reaches zero: the hand's keypoints sit below the object in the wrist view,
so ~0.4 IS contact. What marks a miss is the target leaving the view, or the hand settling
further away than any success ever does.

This flags; it does not conclude. "NOT AT OBJ" covers both a wrong object and a grasp that
failed and drifted, and only a marked frame separates them -- draw the recorded target
points onto the wrist view at the step the fingers close, and look at what is there.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE = "obs/extra/object_image_points"

GAP_LIMIT = 0.55      # above the p90 of every measured success
VISIBLE_LIMIT = 0.90  # every measured success keeps the target in view throughout

#: A rollout shorter than this ended early, which under end_on_success means it won.
HORIZON_MARGIN = 495


def centroids(group, who: str, camera: str):
    counts = np.asarray(group[f"{BASE}/{who}/{camera}/num_points"]).reshape(-1)
    points = np.asarray(group[f"{BASE}/{who}/{camera}/points"])
    out = np.full((len(counts), 2), np.nan, dtype=np.float64)
    for step, count in enumerate(counts):
        if count > 0:
            out[step] = points[step, : int(count)].mean(axis=0)
    return out, counts > 0


def real_length(group) -> int:
    """Where the episode actually stopped.

    The recording is padded to the horizon, so a win looks full-length until the still
    tail is trimmed. Trimming on joint motion is what makes `won` trustworthy.
    """
    qpos = np.asarray(group["obs/agent/qpos"])[:, :7]
    moved = np.abs(np.diff(qpos, axis=0)).sum(axis=1)
    still = int(np.argmax(moved[::-1] > 1e-6))
    return len(qpos) - still


def judge(group, camera: str = "wrist_camera", tail: float = 0.25) -> dict:
    target, target_seen = centroids(group, "pickup_obj", camera)
    hand, hand_seen = centroids(group, "gripper", camera)

    steps = max(real_length(group), 2)
    window = slice(int(steps * (1 - tail)), steps)
    both = (target_seen & hand_seen)[window]

    gaps = np.linalg.norm(target[window] - hand[window], axis=1)[both]
    visible = float(target_seen[window].mean())
    gap = float(np.median(gaps)) if gaps.size else float("nan")

    if not gaps.size or visible < VISIBLE_LIMIT or gap > GAP_LIMIT:
        verdict = "NOT AT OBJ"
    else:
        verdict = "ON TARGET"
    return {
        "steps": steps,
        "won": steps < HORIZON_MARGIN,
        "visible": visible,
        "gap": gap,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", type=Path, required=True, help="a run's eval_output")
    ap.add_argument("--camera", default="wrist_camera")
    ap.add_argument("--limit", type=int, default=12, help="rollouts to list; 0 = summary only")
    args = ap.parse_args()

    files = sorted(args.run.rglob("trajectories_*.h5"))
    if not files:
        raise SystemExit(f"no trajectories under {args.run}")

    rows = []
    for path in files:
        with h5py.File(path, "r") as handle:
            for name in handle:
                if name.startswith("traj"):
                    row = judge(handle[name], args.camera)
                    row["rollout"] = f"{path.stem.split('_')[-3]}/{name}"
                    rows.append(row)

    if args.limit:
        header = "%18s %6s %4s %8s %7s  verdict" % (
            "rollout", "steps", "won", "visible", "gap")
        print(header)
        for row in rows[: args.limit]:
            print("%18s %6d %4s %7.0f%% %7.2f  %s" % (
                row["rollout"], row["steps"], "yes" if row["won"] else "-",
                100 * row["visible"], row["gap"], row["verdict"]))
        print()

    at_target = sum(1 for r in rows if r["verdict"] == "ON TARGET")
    wins = sum(1 for r in rows if r["won"])
    print("%d rollouts | ended at the target %d | succeeded %d" % (len(rows), at_target, wins))

    if wins:
        print("A success proves the policy reaches the object it was asked for: what "
              "fails here is the grasp, which is what RL is meant to fix.")
    elif at_target == 0:
        print("No success, and the hand never ends up at the target. Confirm with a "
              "marked frame before rejecting the scene.")
    else:
        print("No success, but the hand does end up at the target: consistent with a "
              "grasp that never closes. Worth a marked frame to be sure.")


if __name__ == "__main__":
    main()
