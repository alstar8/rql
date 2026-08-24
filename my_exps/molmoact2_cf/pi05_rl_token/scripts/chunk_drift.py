"""How much does open-loop execution of a delta-action chunk drift?

Delta actions are stored per frame as q_cmd[t] - q_state[t]. At serving we only know
the state at the start of the chunk, so element k is replayed as q_state[t] + delta[t+k]
while the truth is q_state[t+k] + delta[t+k]. The gap grows with k. This measures it on
real demonstrations so the chunk size is chosen from numbers rather than taste.

Reported in radians of joint error against the recorded absolute command.
Read-only.
"""

from __future__ import annotations

import glob
import json

import h5py
import numpy as np


def unpack(row) -> dict:
    raw = bytes(np.asarray(row, dtype=np.uint8).tobytes()).rstrip(b"\x00")
    return json.loads(raw.decode("utf-8"))


def main() -> None:
    root = "/workspace-SR008.nfs2/users/staroverov/pi05_molmospaces/mbdata_pick/FrankaPickOmniCamConfig/part0/train"
    files = sorted(glob.glob(root + "/house_*/trajectories_batch_*.h5"))[:30]

    # per chunk offset k -> list of max-joint errors
    errs: dict[int, list[float]] = {k: [] for k in range(16)}
    n_traj = 0

    for path in files:
        try:
            f = h5py.File(path, "r")
        except Exception:
            continue
        for key in f:
            if not key.startswith("traj_"):
                continue
            traj = f[key]
            cmd_ds, q_ds = traj["actions/joint_pos"], traj["obs/agent/qpos"]
            n = cmd_ds.shape[0] - 1  # last action row is empty
            if n < 20:
                continue
            n_traj += 1
            cmd = np.array([unpack(cmd_ds[i])["arm"] for i in range(n)])
            state = np.array([unpack(q_ds[i])["arm"] for i in range(n)])
            delta = cmd - state  # what we store per frame

            for t in range(0, n - 16, 3):
                base = state[t]  # the only state the policy knows at inference
                for k in range(16):
                    replayed = base + delta[t + k]
                    truth = cmd[t + k]
                    errs[k].append(float(np.abs(replayed - truth).max()))
        f.close()

    print(f"trajectories: {n_traj}\n")
    print(" k   median max-joint err   p90      p99     (radians)")
    for k in range(16):
        e = np.array(errs[k])
        print(f"{k:2d}   {np.median(e):.4f}            {np.quantile(e,0.9):.4f}   {np.quantile(e,0.99):.4f}")

    print("\nFor reference, the median per-step joint motion |q[t+1]-q[t]| sets the scale")
    print("of what counts as a large error; a chunk is safe while drift stays well under")
    print("the motion the policy is trying to produce.")


if __name__ == "__main__":
    main()
