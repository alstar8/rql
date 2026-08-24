"""Where does the policy actually fail on a repeated scene: reaching, or grasping?

RL Token is worth pointing at the grasp only if the policy reliably gets to the object
and then loses it. This separates those cases across many rollouts of the same scene:

  never touched      the reach failed, and a grasp correction cannot help
  touched, not held  reached but failed to secure the object -- the case RL targets
  held, not success  grasped and then lost or mislifted it
  success            nothing to fix

Distance to the object centre is reported too, but with a caveat that matters for
elongated objects: the centre can stay far while the gripper is right at a graspable end,
so contact is the more honest signal of "arrived".

    python scripts/analyze_repeats.py --eval-root <dir with shard_*/>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.proximity import GATE_THRESHOLD_M, first_entry, gripper_object_distance  # noqa: E402


def unpack(row) -> dict:
    raw = bytes(np.asarray(row, dtype=np.uint8).tobytes()).rstrip(b"\x00")
    return json.loads(raw.decode("utf-8")) if raw else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=GATE_THRESHOLD_M)
    args = ap.parse_args()

    rows = []
    for h5_path in sorted(args.eval_root.rglob("trajectories_batch_*.h5")):
        with h5py.File(h5_path, "r") as f:
            for key in sorted((k for k in f if k.startswith("traj_")), key=lambda k: int(k.split("_")[1])):
                traj = f[key]
                states = traj["obs/extra/grasp_state_pickup_obj"]
                grasp = [unpack(states[i]) for i in range(states.shape[0])]
                touching = np.array([g.get("gripper", {}).get("touching", False) for g in grasp])
                held = np.array([g.get("gripper", {}).get("held", False) for g in grasp])
                distance = gripper_object_distance(
                    np.asarray(traj["obs/extra/tcp_pose"]),
                    np.asarray(traj["obs/extra/robot_base_pose"]),
                    np.asarray(traj["obs/extra/obj_start"]),
                )
                rows.append(
                    {
                        "success": bool(np.asarray(traj["success"])[-1]),
                        "touched": bool(touching.any()),
                        "held": bool(held.any()),
                        "touch_steps": int(touching.sum()),
                        "held_steps": int(held.sum()),
                        "min_distance": float(distance.min()),
                        "steps": int(len(distance)),
                        "gate_step": first_entry(distance, args.threshold),
                    }
                )

    if not rows:
        raise SystemExit(f"no trajectories under {args.eval_root}")

    n = len(rows)
    success = sum(r["success"] for r in rows)
    touched = sum(r["touched"] for r in rows)
    held = sum(r["held"] for r in rows)
    gated = sum(r["gate_step"] is not None for r in rows)

    print(f"rollouts: {n}\n")
    print(f"  reached contact      {touched:>3}/{n}  ({100*touched/n:.0f}%)")
    print(f"  ever held            {held:>3}/{n}  ({100*held/n:.0f}%)")
    print(f"  success              {success:>3}/{n}  ({100*success/n:.0f}%)")
    print(f"  gate opened (<={args.threshold} m to centre)  {gated:>3}/{n}  ({100*gated/n:.0f}%)")

    never = [r for r in rows if not r["touched"]]
    touched_not_held = [r for r in rows if r["touched"] and not r["held"]]
    held_not_success = [r for r in rows if r["held"] and not r["success"]]

    print("\nfailure breakdown:")
    print(f"  never touched        {len(never):>3}  reach failed; a grasp fix cannot help")
    print(f"  touched, not held    {len(touched_not_held):>3}  <- the case RL Token targets")
    print(f"  held, not success    {len(held_not_success):>3}  secured then lost or mislifted")
    print(f"  success              {success:>3}")

    d = np.array([r["min_distance"] for r in rows])
    print("\nclosest approach to object centre (m):")
    print(f"  min {d.min():.3f}   median {np.median(d):.3f}   max {d.max():.3f}")
    for label, subset in (("touched", [r for r in rows if r["touched"]]),
                          ("never touched", never)):
        if subset:
            dd = np.array([r["min_distance"] for r in subset])
            print(f"  {label:<14} median {np.median(dd):.3f}  range {dd.min():.3f}..{dd.max():.3f}")

    if touched:
        contact_d = np.array([r["min_distance"] for r in rows if r["touched"]])
        print(f"\nGate check: episodes that made contact had centre distance "
              f"{contact_d.min():.3f}..{contact_d.max():.3f} m.")
        if contact_d.max() > args.threshold:
            print(f"  A {args.threshold} m gate on centre distance would MISS some of them, "
                  "which is the elongated-object problem: contact happens while the centre "
                  "is still far.")


if __name__ == "__main__":
    main()
