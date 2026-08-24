"""Tile several rollouts of the same scene with a large step counter on each.

The object pose recorded by the evaluator is the benchmark's spec pose, not the object's
live position, so a distance-based gate cannot be trusted here -- the object moves and
the number does not. Until there is a live pose, the gate is chosen by watching: this
renders rollouts side by side with the step index visible, so a handover step can be read
off directly.

    python scripts/make_step_video.py --eval-root <dir with shard_*/> --out <dir> --count 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

GREEN = (60, 220, 60)
RED = (60, 60, 220)


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


def episode_success(h5_path: Path, index: int) -> bool | None:
    try:
        with h5py.File(h5_path, "r") as f:
            key = f"traj_{index}"
            if key in f and "success" in f[key]:
                return bool(np.asarray(f[key]["success"])[-1])
    except OSError:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--count", type=int, default=6, help="rollouts to include")
    ap.add_argument("--camera", default="exo_camera_1")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()

    videos = sorted(args.eval_root.rglob(f"*{args.camera}*.mp4"))[: args.count]
    if not videos:
        raise SystemExit(f"no {args.camera} videos under {args.eval_root}")

    clips, labels = [], []
    for path in videos:
        frames = read_frames(path)
        if not frames:
            continue
        index = int(path.name.split("_")[1])
        h5_path = next(iter(path.parent.glob("trajectories_batch_*.h5")), None)
        ok = episode_success(h5_path, index) if h5_path else None
        clips.append(frames)
        labels.append((path.parent.parent.name, index, ok, len(frames)))

    longest = max(len(c) for c in clips)
    cell_h, cell_w = 240, 420
    cols = min(args.cols, len(clips))
    rows = (len(clips) + cols - 1) // cols
    width, height = cols * cell_w, rows * (cell_h + 34)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"repeats_{args.camera}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))

    for t in range(longest):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        for i, (frames, (run, index, ok, n)) in enumerate(zip(clips, labels, strict=True)):
            r, c = divmod(i, cols)
            y0, x0 = r * (cell_h + 34), c * cell_w
            frame = frames[min(t, len(frames) - 1)]
            canvas[y0 : y0 + cell_h, x0 : x0 + cell_w] = cv2.resize(frame, (cell_w, cell_h))

            done = t >= n
            colour = GREEN if ok else RED
            if ok is not None:
                cv2.rectangle(canvas, (x0, y0), (x0 + cell_w - 2, y0 + cell_h - 2), colour, 3)
            step_text = f"step {min(t, n - 1):3d}/{n - 1}" + ("  (ended)" if done else "")
            cv2.putText(canvas, step_text, (x0 + 8, y0 + cell_h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 250), 2, cv2.LINE_AA)
            tag = "success" if ok else ("fail" if ok is not None else "?")
            cv2.putText(canvas, f"{run}/ep{index}  {tag}", (x0 + 210, y0 + cell_h + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
        writer.write(canvas)
    writer.release()

    print(f"wrote {out_path}  ({len(clips)} rollouts, {longest} frames, {width}x{height})")
    for run, index, ok, n in labels:
        print(f"  {run}/ep{index}: {n} steps, success={ok}")


if __name__ == "__main__":
    main()
