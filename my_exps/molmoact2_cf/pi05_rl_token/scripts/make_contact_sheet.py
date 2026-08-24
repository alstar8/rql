#!/usr/bin/env python
"""Tile an episode's frames into one labelled image, for choosing the RL gate step.

    scripts/make_contact_sheet.py --eval-dir <run>/eval_output --episodes 0,1,2,3 \
        --every 10 --out sheets/

The gate is the step where RL takes control, and it is picked by looking: before it the
arm is still travelling toward the object and the VLA is good at that; after it the
gripper is in the neighbourhood and the grasp is what fails. A number chosen from a
description rather than from frames is a guess.

One frame is one env step. MolmoSpaces drives the policy at `policy_dt_ms = 66`, the
videos are written at 15.15 fps, and the writer emits one frame per policy step, so the
frame index IS the step index -- which is what makes a contact sheet readable as a
timeline rather than as a set of pictures.

Both cameras are tiled for the same steps: the exterior view shows the approach, the
wrist view shows whether the object is actually between the fingers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LABEL_H = 18
PAD = 3


def read_frames(video: Path, steps: list[int]) -> dict[int, np.ndarray]:
    """Grab the requested frames. Sequential reads: seeking is unreliable on these files."""
    capture = cv2.VideoCapture(str(video))
    wanted = set(steps)
    out: dict[int, np.ndarray] = {}
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            out[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
        if index > max(wanted, default=0):
            break
    capture.release()
    return out


def total_frames(video: Path) -> int:
    capture = cv2.VideoCapture(str(video))
    n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return n


def strip(frames: dict[int, np.ndarray], steps: list[int], scale: float, title: str) -> Image.Image:
    """One labelled row: the same view at each requested step."""
    present = [s for s in steps if s in frames]
    if not present:
        raise ValueError(f"no frames for {title}")
    h, w = frames[present[0]].shape[:2]
    tw, th = int(w * scale), int(h * scale)

    sheet = Image.new("RGB", (len(present) * (tw + PAD), th + LABEL_H + PAD), (16, 16, 18))
    draw = ImageDraw.Draw(sheet)
    for column, step in enumerate(present):
        tile = Image.fromarray(frames[step]).resize((tw, th), Image.BILINEAR)
        x = column * (tw + PAD)
        sheet.paste(tile, (x, LABEL_H))
        draw.text((x + 2, 3), f"{title} t={step}", fill=(235, 235, 235))
    return sheet


def stack(images: list[Image.Image]) -> Image.Image:
    width = max(i.width for i in images)
    height = sum(i.height for i in images)
    out = Image.new("RGB", (width, height), (16, 16, 18))
    y = 0
    for image in images:
        out.paste(image, (0, y))
        y += image.height
    return out


def episode_videos(eval_dir: Path) -> dict[str, dict[str, Path]]:
    """Map rollout id -> {"exo": path, "wrist": path}.

    MolmoSpaces may run one episode spec several times and names the results
    `episode_N_<camera>_batch_i_of_n.mp4`. Those batches are independent rollouts, not
    segments of one -- two batches of the same spec end at different steps -- so each is
    its own entry. Keying on the episode index alone would silently keep only the last
    batch, which is how a failure could hide behind a success of the same spec.
    """
    found: dict[str, dict[str, Path]] = {}
    for video in sorted(eval_dir.rglob("episode_*.mp4")):
        name = video.stem
        parts = name.split("_")
        try:
            index = int(parts[1])
        except (IndexError, ValueError):
            continue
        batch = ""
        if "batch" in parts:
            position = parts.index("batch")
            batch = f"b{parts[position + 1]}" if position + 1 < len(parts) else ""
        key = "wrist" if "wrist" in name else "exo"
        found.setdefault(f"{index:03d}{('_' + batch) if batch else ''}", {})[key] = video
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eval-dir", type=Path, required=True, help="a run's eval_output")
    ap.add_argument("--episodes", default="", help="comma list; default: the first --limit")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--every", type=int, default=10, help="step interval between tiles")
    ap.add_argument("--max-step", type=int, default=0, help="0 = to the end of the episode")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True, help="directory for the sheets")
    args = ap.parse_args()

    videos = episode_videos(args.eval_dir)
    if not videos:
        raise SystemExit(f"no episode videos under {args.eval_dir}")

    if args.episodes:
        keys = set(args.episodes.split(","))
        wanted = [k for k in sorted(videos) if k in keys or k.split("_")[0] in keys]
    else:
        wanted = sorted(videos)[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    for index in wanted:
        pair = videos.get(index)
        if not pair or "exo" not in pair:
            print(f"episode {index}: no video, skipping")
            continue

        length = total_frames(pair["exo"])
        last = min(args.max_step, length - 1) if args.max_step else length - 1
        steps = list(range(0, last + 1, args.every))
        if steps[-1] != last:
            steps.append(last)  # the final frame is where the outcome is visible

        rows = []
        for key in ("exo", "wrist"):
            if key in pair:
                rows.append(strip(read_frames(pair[key], steps), steps, args.scale, key))
        sheet = stack(rows)
        path = args.out / f"episode_{index}_steps.png"
        sheet.save(path)
        print(f"episode {index}: {length} frames -> {path}")


if __name__ == "__main__":
    main()
