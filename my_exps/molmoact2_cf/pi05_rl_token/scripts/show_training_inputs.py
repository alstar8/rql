"""Render what the model actually receives, straight out of the training dataset.

Reads the merged LeRobot parquet -- the same bytes the dataloader serves -- and lays out
the two camera views side by side with the instruction, state and action for each frame.
Reading the parquet rather than the source h5 is the point: this shows what training
consumes, so a mistake anywhere in conversion would be visible here.

    python scripts/show_training_inputs.py --dataset <merged> --out frames.png
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw

CHUNK_SIZE = 1000
PANEL = 224
PAD = 10
LABEL_H = 46


def episode_path(root: Path, index: int) -> Path:
    return root / "data" / f"chunk-{index // CHUNK_SIZE:03d}" / f"episode_{index:06d}.parquet"


def load_tasks(root: Path) -> dict[int, str]:
    tasks = {}
    for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            tasks[row["task_index"]] = row["task"]
    return tasks


def decode(cell) -> Image.Image:
    return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=6, help="how many frames to show")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    info = json.loads((args.dataset / "meta" / "info.json").read_text())
    tasks = load_tasks(args.dataset)
    total_episodes = info["total_episodes"]

    rng = random.Random(args.seed)
    picks = []
    # Spread across the dataset rather than clustering in the first chunk, so this also
    # samples episodes contributed by different converter workers.
    while len(picks) < args.rows:
        ep = rng.randrange(total_episodes)
        path = episode_path(args.dataset, ep)
        if not path.exists():
            continue
        table = pq.read_table(path)
        if table.num_rows < 3:
            continue
        frame = rng.randrange(table.num_rows)
        picks.append((ep, frame, table))

    width = PANEL * 2 + PAD * 3
    height = (PANEL + LABEL_H + PAD) * len(picks) + PAD
    canvas = Image.new("RGB", (width, height), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)

    for row, (ep, frame_idx, table) in enumerate(picks):
        exterior = decode(table.column("exterior_image_1_left")[frame_idx].as_py())
        wrist = decode(table.column("wrist_image_left")[frame_idx].as_py())
        state = table.column("joint_position")[frame_idx].as_py()
        gripper = table.column("gripper_position")[frame_idx].as_py()
        action = table.column("actions")[frame_idx].as_py()
        task = tasks.get(table.column("task_index")[frame_idx].as_py(), "?")

        top = PAD + row * (PANEL + LABEL_H + PAD)
        canvas.paste(exterior.resize((PANEL, PANEL)), (PAD, top))
        canvas.paste(wrist.resize((PANEL, PANEL)), (PAD * 2 + PANEL, top))

        grip = gripper[0] if isinstance(gripper, list) else gripper
        draw.text(
            (PAD, top + PANEL + 4),
            f'ep {ep}  frame {frame_idx}/{table.num_rows}   "{task}"',
            fill=(235, 235, 240),
        )
        draw.text(
            (PAD, top + PANEL + 18),
            "state j0..j6 " + " ".join(f"{v:+.2f}" for v in state) + f"  grip {grip:.2f}",
            fill=(150, 200, 255),
        )
        draw.text(
            (PAD, top + PANEL + 32),
            "action delta " + " ".join(f"{v:+.3f}" for v in action[:7]) + f"  grip {action[7]:.2f}",
            fill=(255, 200, 140),
        )

    draw.text((PAD, 1), "left: exterior_image_1_left    right: wrist_image_left", fill=(140, 140, 150))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"wrote {args.out}  ({canvas.size[0]}x{canvas.size[1]})")
    for ep, frame_idx, table in picks:
        task = tasks.get(table.column("task_index")[frame_idx].as_py(), "?")
        print(f"  episode {ep:>6}  frame {frame_idx:>4}  {task}")


if __name__ == "__main__":
    main()
