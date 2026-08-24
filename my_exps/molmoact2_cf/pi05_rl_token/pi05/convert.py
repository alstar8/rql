"""Build a LeRobot dataset for openpi out of MolmoSpaces Pick demonstrations.

Feature names follow the DROID convention because that is what openpi's stock
``pi05_droid_finetune`` config repacks; using it means no edits anywhere outside this
directory. ``exterior_image_2_left`` is written as a copy of the shoulder view: the
repack transform looks the key up unconditionally and would raise KeyError without it,
while ``DroidInputs`` never reads it, so the duplicate costs disk and nothing else.

Images are stored already resized to 224x224 by the same resize-with-pad the eval client
uses, so training and inference see identical pixels. It is reimplemented here on PIL
rather than imported from openpi because openpi's copy pulls in jax, and importing jax
makes this CPU-only job seize a GPU it never uses.

Conversion is per-shard and resumable: every written trajectory is recorded in a
manifest so a later shard can be appended without rebuilding what exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from .action_space import action_from_command, normalize_instruction, state_from_qpos
from .demo_reader import Episode, iter_episodes, read_video

# Demonstrations are recorded at policy_dt_ms=66, and DROID's LeRobot convention is 15.
FPS = 15
IMAGE_SIZE = 224
MANIFEST_NAME = "converted_manifest.json"

# Every trajectory opens with two frames that teach the wrong thing. At t0 the planner
# emits a command of 1.0-2.3 rad that the controller never executes -- the arm does not
# move at all -- which is roughly 15 sigma on the DROID action scale. At t1 the command
# equals the current pose exactly, a hold. Real motion begins at t2, by which point the
# arm is still at the reset pose. Keeping t0 would let one frame in seventy dominate the
# loss; keeping t1 would teach "at the reset pose, output zero", which is exactly where
# every evaluation episode starts. Both are dropped.
DROP_FIRST_STEPS = 2


def _features(duplicate_exterior2: bool) -> dict:
    image_spec = {
        "dtype": "image",
        "shape": (IMAGE_SIZE, IMAGE_SIZE, 3),
        "names": ["height", "width", "channel"],
    }
    features = {
        "exterior_image_1_left": dict(image_spec),
        "wrist_image_left": dict(image_spec),
        "joint_position": {"dtype": "float32", "shape": (7,), "names": ["joint_position"]},
        "gripper_position": {"dtype": "float32", "shape": (1,), "names": ["gripper_position"]},
        "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
    }
    if duplicate_exterior2:
        features["exterior_image_2_left"] = dict(image_spec)
    return features


def _resize(frame: np.ndarray) -> np.ndarray:
    """Scale to fit inside IMAGE_SIZE without distortion, then centre on black padding.

    Mirrors molmo_spaces' resize_with_pad, which is what runs at eval time.
    """
    image = Image.fromarray(frame)
    cur_width, cur_height = image.size
    if (cur_width, cur_height) == (IMAGE_SIZE, IMAGE_SIZE):
        return np.asarray(image)

    ratio = max(cur_width / IMAGE_SIZE, cur_height / IMAGE_SIZE)
    resized = image.resize(
        (int(cur_width / ratio), int(cur_height / ratio)), resample=Image.BILINEAR
    )
    canvas = Image.new(resized.mode, (IMAGE_SIZE, IMAGE_SIZE), 0)
    canvas.paste(
        resized,
        (max(0, (IMAGE_SIZE - resized.width) // 2), max(0, (IMAGE_SIZE - resized.height) // 2)),
    )
    return np.asarray(canvas)


def load_manifest(root: Path) -> set[str]:
    path = root / MANIFEST_NAME
    if not path.exists():
        return set()
    return set(json.loads(path.read_text())["episodes"])


def save_manifest(root: Path, done: set[str]) -> None:
    path = root / MANIFEST_NAME
    path.write_text(json.dumps({"episodes": sorted(done)}, indent=2))


def open_dataset(repo_id: str, root: Path, *, duplicate_exterior2: bool) -> LeRobotDataset:
    """Create the dataset, or reopen it so a new shard appends to what is there."""
    if (root / "meta").exists():
        dataset = LeRobotDataset(repo_id, root=str(root))
        dataset.start_image_writer(num_processes=5, num_threads=10)
        return dataset
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=str(root),
        robot_type="panda",
        fps=FPS,
        features=_features(duplicate_exterior2),
        image_writer_threads=10,
        image_writer_processes=5,
    )


def write_episode(
    dataset: LeRobotDataset,
    episode: Episode,
    *,
    duplicate_exterior2: bool,
    drop_first_steps: int = DROP_FIRST_STEPS,
) -> int:
    """Append one demonstration; returns the number of frames written."""
    exo = read_video(episode.exo_video, episode.n_steps + 1)
    wrist = read_video(episode.wrist_video, episode.n_steps + 1)
    instruction = normalize_instruction(episode.instruction)
    placeholder = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    for t in range(drop_first_steps, episode.n_steps):
        exo_frame = _resize(exo[t])
        frame = {
            "exterior_image_1_left": exo_frame,
            "wrist_image_left": _resize(wrist[t]),
            "joint_position": np.asarray(episode.qpos[t]["arm"][:7], dtype=np.float32),
            "gripper_position": state_from_qpos(episode.qpos[t])[7:].astype(np.float32),
            "actions": action_from_command(episode.commands[t], episode.qpos[t]),
            "task": instruction,
        }
        if duplicate_exterior2:
            frame["exterior_image_2_left"] = placeholder
        dataset.add_frame(frame)

    dataset.save_episode()
    return episode.n_steps - drop_first_steps


def convert(
    source: Path,
    out_root: Path,
    repo_id: str,
    *,
    houses: list[str] | None = None,
    max_episodes: int | None = None,
    duplicate_exterior2: bool = True,
    require_success: bool = True,
    drop_first_steps: int = DROP_FIRST_STEPS,
    verbose: bool = True,
) -> dict:
    """Convert demonstrations under ``source`` into a LeRobot dataset at ``out_root``."""
    out_root = Path(out_root)
    # Only the parent: LeRobotDataset.create insists on making the dataset directory
    # itself, and fails if it already exists.
    out_root.parent.mkdir(parents=True, exist_ok=True)
    done = load_manifest(out_root)

    dataset = open_dataset(repo_id, out_root, duplicate_exterior2=duplicate_exterior2)

    written = skipped = frames = 0
    for episode in iter_episodes(source, houses=houses, require_success=require_success):
        if episode.episode_id in done:
            skipped += 1
            continue
        if max_episodes is not None and written >= max_episodes:
            break
        try:
            frames += write_episode(
                dataset,
                episode,
                duplicate_exterior2=duplicate_exterior2,
                drop_first_steps=drop_first_steps,
            )
        except ValueError as exc:
            # A frame/step mismatch means this episode's video is unusable; record
            # nothing for it and keep going rather than poisoning the dataset.
            if verbose:
                print(f"  skipping {episode.episode_id}: {exc}")
            continue
        done.add(episode.episode_id)
        save_manifest(out_root, done)
        written += 1
        if verbose and written % 20 == 0:
            print(f"  {written} episodes, {frames} frames")

    save_manifest(out_root, done)
    return {"written": written, "skipped": skipped, "frames": frames, "total": len(done)}
