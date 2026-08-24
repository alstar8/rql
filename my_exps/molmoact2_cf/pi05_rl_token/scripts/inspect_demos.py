"""Inspect MolmoSpaces Pick demonstrations before writing any converter.

Answers, on real files and numerically, the three questions that silently ruin a
fine-tune: which action field the environment actually executes, where the language
instruction lives, and how mp4 frames line up with trajectory steps. A wrong guess
here still produces a falling loss, so it cannot be caught later by watching training.

Read-only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np


def unpack_json(arr: np.ndarray):
    """Most string fields are JSON padded into fixed-length uint8 rows."""
    raw = bytes(np.asarray(arr, dtype=np.uint8).tobytes()).rstrip(b"\x00")
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as exc:
        return f"<unparsed {exc}: {raw[:120]!r}>"


def walk(group, prefix="", depth=0, max_depth=3):
    for key in group:
        item = group[key]
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Group):
            if depth < max_depth:
                walk(item, path, depth + 1, max_depth)
            else:
                print(f"    {path}/  (group, {len(item)} keys)")
        else:
            print(f"    {path}  shape={item.shape} dtype={item.dtype}")


def main(h5_path: Path) -> None:
    print(f"file: {h5_path}\n")
    with h5py.File(h5_path, "r") as f:
        root = list(f.keys())
        trajs = sorted(k for k in root if k.startswith("traj_"))
        print(f"root keys: {[k for k in root if not k.startswith('traj_')]}")
        print(f"trajectories: {len(trajs)}  ({trajs[:3]} ...)\n")

        if "valid_traj_mask" in f:
            mask = np.asarray(f["valid_traj_mask"])
            print(f"valid_traj_mask: shape={mask.shape} dtype={mask.dtype} "
                  f"sum={mask.sum()} of {mask.size}\n")

        print("=" * 78)
        print("== structure of traj_0")
        print("=" * 78)
        walk(f[trajs[0]], prefix=trajs[0])

        print()
        print("=" * 78)
        print("== action fields vs achieved state  (medians over |residual|)")
        print("=" * 78)
        for name in trajs[:4]:
            t = f[name]
            acts = t["actions"]
            qpos = np.asarray(t["obs/agent/qpos"])
            print(f"\n--- {name}: qpos shape {qpos.shape}")
            for key in acts:
                a = np.asarray(acts[key])
                print(f"    actions/{key}: shape={a.shape} dtype={a.dtype} "
                      f"min={np.min(a):+.4f} max={np.max(a):+.4f}")

            arm_q = qpos[:, :7]
            n = len(arm_q)
            for key in ("joint_pos", "joint_pos_rel", "commanded_action"):
                if key not in acts:
                    continue
                a = np.asarray(acts[key])
                if a.ndim != 2 or a.shape[1] < 7:
                    continue
                cmd = a[:, :7]
                m = min(len(cmd), n - 1)
                # hypothesis A: command is an absolute target reached at t+1
                res_next = np.abs(cmd[:m] - arm_q[1:m + 1])
                # hypothesis B: command merely echoes the current state
                res_now = np.abs(cmd[:m] - arm_q[:m])
                # hypothesis C: command is a delta added to the current state
                res_delta = np.abs(cmd[:m] + arm_q[:m] - arm_q[1:m + 1])
                print(f"    {key:18s} |cmd - q[t+1]|={np.median(res_next):.5f}  "
                      f"|cmd - q[t]|={np.median(res_now):.5f}  "
                      f"|cmd + q[t] - q[t+1]|={np.median(res_delta):.5f}")

            # gripper: what is in qpos beyond the 7 arm joints
            if qpos.shape[1] > 7:
                g = qpos[:, 7:]
                print(f"    qpos[7:] (gripper): shape={g.shape} "
                      f"min={g.min():+.5f} max={g.max():+.5f}")

        print()
        print("=" * 78)
        print("== episode-level fields of traj_0")
        print("=" * 78)
        t = f[trajs[0]]
        for key in ("success", "rewards", "terminated", "truncated"):
            if key in t:
                a = np.asarray(t[key])
                print(f"  {key}: shape={a.shape} dtype={a.dtype} "
                      f"first={a.flat[0]} last={a.flat[-1]} sum={a.sum() if a.dtype != bool else a.astype(int).sum()}")

        print()
        print("=" * 78)
        print("== language: obs_scene / task_info of traj_0")
        print("=" * 78)
        for key in ("obs_scene", "obs/extra/task_info"):
            if key not in t:
                print(f"  {key}: absent")
                continue
            item = t[key]
            if isinstance(item, h5py.Group):
                print(f"  {key}/ is a group: {list(item.keys())[:20]}")
                continue
            arr = np.asarray(item)
            print(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
            payload = unpack_json(arr[0] if arr.ndim > 1 else arr)
            if isinstance(payload, dict):
                print(f"    top-level keys: {list(payload.keys())}")
                blob = json.dumps(payload)[:1500]
                print(f"    {blob}")
            else:
                print(f"    {str(payload)[:600]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
