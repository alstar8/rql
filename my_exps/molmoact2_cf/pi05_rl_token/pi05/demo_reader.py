"""Reading MolmoSpaces Pick demonstrations: h5 trajectories plus their mp4 frames.

Everything the converter needs to know about the raw format lives here, including the
two details that fail silently if guessed:

  * the last row of every trajectory carries an empty action ``{}``, so a trajectory of
    N rows yields N-1 usable transitions;
  * frames and steps correspond one-to-one, and the mp4 filename is stored inside the
    h5 rather than being derivable from the trajectory index.

Most fields are JSON padded into fixed-length uint8 rows and have to be unpacked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import h5py
import numpy as np

# The two views the eval benchmark also renders; the other three demo cameras never
# appear at eval and are ignored.
EXO_CAMERA = "droid_shoulder_light_randomization"
WRIST_CAMERA = "wrist_camera_zed_mini"


def unpack_json_row(row) -> dict:
    """Fields are JSON blobs zero-padded to a fixed width."""
    raw = bytes(np.asarray(row, dtype=np.uint8).tobytes()).rstrip(b"\x00")
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def read_scene(traj: h5py.Group) -> dict:
    blob = traj["obs_scene"][()]
    if isinstance(blob, bytes):
        return json.loads(blob.decode("utf-8"))
    return json.loads(str(blob))


def camera_filename(traj: h5py.Group, camera: str) -> str:
    """The mp4 name is stored as bytes in obs/sensor_data/<camera>."""
    raw = np.asarray(traj[f"obs/sensor_data/{camera}"])
    return bytes(raw.tobytes()).rstrip(b"\x00").decode("utf-8")


def read_video(path: Path, expected_frames: int) -> np.ndarray:
    """Decode a whole episode's frames as RGB, refusing any length mismatch.

    A silent off-by-one between frames and steps would train the model on the wrong
    image for every transition, so the mismatch is raised rather than trimmed.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[..., ::-1])  # BGR -> RGB
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    if len(frames) != expected_frames:
        raise ValueError(
            f"{path.name}: {len(frames)} frames for {expected_frames} trajectory rows; "
            "frames and steps must correspond one-to-one"
        )
    return np.stack(frames)


@dataclass
class Episode:
    """One demonstration, already aligned and trimmed to usable transitions."""

    house: str
    h5_path: Path
    traj_key: str
    instruction: str
    n_steps: int  # usable transitions, i.e. rows - 1
    qpos: list[dict]
    commands: list[dict]
    exo_video: Path
    wrist_video: Path
    success: bool

    @property
    def episode_id(self) -> str:
        return f"{self.house}/{self.h5_path.name}/{self.traj_key}"


def iter_episodes(
    root: Path,
    *,
    houses: list[str] | None = None,
    require_success: bool = True,
) -> Iterator[Episode]:
    """Walk ``<root>/house_*/trajectories_batch_*.h5`` yielding valid demonstrations.

    Trajectories flagged by ``valid_traj_mask`` are skipped; so are ones whose recorded
    videos are missing, since a partially downloaded shard should not silently shrink
    the dataset.
    """
    root = Path(root)
    house_dirs = sorted(p for p in root.glob("house_*") if p.is_dir())
    if houses is not None:
        wanted = set(houses)
        house_dirs = [p for p in house_dirs if p.name in wanted]

    for house_dir in house_dirs:
        for h5_path in sorted(house_dir.glob("trajectories_batch_*.h5")):
            try:
                handle = h5py.File(h5_path, "r")
            except OSError:
                continue
            with handle as f:
                mask = np.asarray(f["valid_traj_mask"]) if "valid_traj_mask" in f else None
                for traj_key in sorted(
                    (k for k in f if k.startswith("traj_")),
                    key=lambda k: int(k.split("_")[1]),
                ):
                    index = int(traj_key.split("_")[1])
                    if mask is not None and index < len(mask) and not bool(mask[index]):
                        continue

                    traj = f[traj_key]
                    rows = traj["actions/joint_pos"].shape[0]
                    n_steps = rows - 1  # final row's action is {}
                    if n_steps < 2:
                        continue

                    succeeded = bool(np.asarray(traj["success"])[-1])
                    if require_success and not succeeded:
                        continue

                    exo = house_dir / camera_filename(traj, EXO_CAMERA)
                    wrist = house_dir / camera_filename(traj, WRIST_CAMERA)
                    if not exo.exists() or not wrist.exists():
                        continue

                    scene = read_scene(traj)
                    description = scene.get("task_description")
                    if not description:
                        continue

                    yield Episode(
                        house=house_dir.name,
                        h5_path=h5_path,
                        traj_key=traj_key,
                        instruction=description,
                        n_steps=n_steps,
                        qpos=[unpack_json_row(traj["obs/agent/qpos"][i]) for i in range(rows)],
                        commands=[
                            unpack_json_row(traj["actions/joint_pos"][i]) for i in range(n_steps)
                        ],
                        exo_video=exo,
                        wrist_video=wrist,
                        success=succeeded,
                    )
