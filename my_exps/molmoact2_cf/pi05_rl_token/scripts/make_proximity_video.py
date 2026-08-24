"""Render eval episodes with the gripper-object distance drawn on every frame.

The point is to make the proximity gate checkable by eye. A green border appears exactly
on the steps where the distance is at or below the threshold -- the steps where RL would
take over -- and the distance itself is printed, so the threshold can be chosen from what
the video shows rather than picked in advance.

Distance comes from pi05/proximity.py, which transforms the TCP out of the robot base
frame into world coordinates before comparing with the object.

    python scripts/make_proximity_video.py --eval-dir <house_N dir> --out <dir> --threshold 0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.proximity import first_entry, gripper_object_distance  # noqa: E402

GREEN = (60, 220, 60)
GREY = (90, 90, 95)
BORDER = 14


def unpack_json_row(row) -> dict:
    raw = bytes(np.asarray(row, dtype=np.uint8).tobytes()).rstrip(b"\x00")
    return json.loads(raw.decode("utf-8")) if raw else {}


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def render(h5_path: Path, out_dir: Path, threshold: float, fps: int) -> list[dict]:
    house_dir = h5_path.parent
    summaries = []

    with h5py.File(h5_path, "r") as f:
        traj_keys = sorted((k for k in f if k.startswith("traj_")), key=lambda k: int(k.split("_")[1]))
        for index, key in enumerate(traj_keys):
            traj = f[key]
            distance = gripper_object_distance(
                np.asarray(traj["obs/extra/tcp_pose"]),
                np.asarray(traj["obs/extra/robot_base_pose"]),
                np.asarray(traj["obs/extra/obj_start"]),
            )
            success = bool(np.asarray(traj["success"])[-1]) if "success" in traj else False

            exo = read_frames(house_dir / f"episode_{index:08d}_exo_camera_1_batch_1_of_1.mp4")
            wrist = read_frames(house_dir / f"episode_{index:08d}_wrist_camera_batch_1_of_1.mp4")
            if not exo or not wrist:
                print(f"  {key}: videos missing, skipped")
                continue

            steps = min(len(exo), len(wrist), len(distance))
            h = max(exo[0].shape[0], wrist[0].shape[0])
            w_exo = int(exo[0].shape[1] * h / exo[0].shape[0])
            w_wrist = int(wrist[0].shape[1] * h / wrist[0].shape[0])
            width, height = w_exo + w_wrist, h + 64

            out_path = out_dir / f"{house_dir.name}_{key}_thr{threshold:.2f}.mp4"
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

            entry = first_entry(distance[:steps], threshold)
            for t in range(steps):
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                canvas[:h, :w_exo] = cv2.resize(exo[t], (w_exo, h))
                canvas[:h, w_exo:] = cv2.resize(wrist[t], (w_wrist, h))

                close = distance[t] <= threshold
                if close:
                    cv2.rectangle(canvas, (0, 0), (width - 1, h - 1), GREEN, BORDER)

                colour = GREEN if close else GREY
                cv2.putText(canvas, f"step {t:3d}/{steps}", (10, h + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 235), 1, cv2.LINE_AA)
                cv2.putText(canvas, f"dist {distance[t]:.3f} m", (150, h + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
                cv2.putText(canvas, f"threshold {threshold:.2f} m", (330, h + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 190), 1, cv2.LINE_AA)
                cv2.putText(canvas, "RL WOULD TAKE OVER" if close else "approach",
                            (10, h + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)
                cv2.putText(canvas, f"success={success}", (330, h + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 210), 1, cv2.LINE_AA)

                # A strip showing distance over the whole episode, with the current step
                # marked, so the approach is legible without scrubbing.
                bar_y, bar_x0, bar_w = h + 58, 470, width - 490
                if bar_w > 40:
                    span = max(distance[:steps].max(), threshold * 1.5)
                    for i in range(0, bar_w):
                        s = int(i * steps / bar_w)
                        val = distance[s]
                        c = GREEN if val <= threshold else (120, 120, 130)
                        cv2.line(canvas, (bar_x0 + i, bar_y), (bar_x0 + i, bar_y - int(24 * (1 - val / span))), c, 1)
                    cur = bar_x0 + int(t * bar_w / steps)
                    cv2.line(canvas, (cur, bar_y + 2), (cur, bar_y - 26), (255, 255, 255), 1)

                writer.write(canvas)
            writer.release()

            summary = {
                "episode": index,
                "traj": key,
                "steps": steps,
                "success": success,
                "distance_start": float(distance[0]),
                "distance_min": float(distance[:steps].min()),
                "distance_min_step": int(distance[:steps].argmin()),
                "first_within_threshold": entry,
                "video": str(out_path),
            }
            summaries.append(summary)
            print(f"  {key}: success={success} steps={steps} "
                  f"start={distance[0]:.3f} min={distance[:steps].min():.3f} "
                  f"first<= {threshold}: {entry}")
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, required=True, help="house_N directory from an eval run")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.10, help="metres")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    h5_files = sorted(args.eval_dir.glob("trajectories_batch_*.h5"))
    if not h5_files:
        raise SystemExit(f"no trajectory h5 under {args.eval_dir}")

    summaries = []
    for h5_path in h5_files:
        summaries += render(h5_path, args.out, args.threshold, args.fps)

    (args.out / "proximity_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {len(summaries)} videos to {args.out}")


if __name__ == "__main__":
    main()
