"""Reading demonstrations, tested on the boundaries that go wrong without a sound.

Built on a synthetic h5/mp4 pair rather than the real corpus so the test runs anywhere
and pins the exact conventions: the trailing empty action, the validity mask, and the
one-to-one correspondence between frames and steps.
"""

import json

import cv2
import h5py
import numpy as np
import pytest

from pi05.demo_reader import (
    EXO_CAMERA,
    WRIST_CAMERA,
    iter_episodes,
    read_video,
    unpack_json_row,
)

WIDTH, HEIGHT = 64, 48


def _pack(payload: dict, width: int = 2000) -> np.ndarray:
    raw = json.dumps(payload).encode("utf-8")
    row = np.zeros(width, dtype=np.uint8)
    row[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return row


def _pack_name(name: str) -> np.ndarray:
    raw = name.encode("utf-8")
    row = np.zeros(100, dtype=np.uint8)
    row[: len(raw)] = np.frombuffer(raw, dtype=np.uint8)
    return row


def _write_video(path, n_frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (WIDTH, HEIGHT))
    for i in range(n_frames):
        writer.write(np.full((HEIGHT, WIDTH, 3), i % 255, dtype=np.uint8))
    writer.release()


def _build(tmp_path, *, rows=6, valid=(True,), n_frames=None, success=True):
    house = tmp_path / "house_0"
    house.mkdir(exist_ok=True)
    h5_path = house / "trajectories_batch_1_of_2.h5"
    n_frames = rows if n_frames is None else n_frames

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("valid_traj_mask", data=np.array(valid, dtype=bool))
        for index in range(len(valid)):
            traj = f.create_group(f"traj_{index}")
            arm = [[0.1 * t + 0.01 * j for j in range(7)] for t in range(rows)]
            traj.create_dataset(
                "obs/agent/qpos",
                data=np.stack([_pack({"arm": arm[t], "base": [], "gripper": [0.0, 0.0]}) for t in range(rows)]),
            )
            # Final action row is empty, exactly as the real corpus records it.
            actions = [{"arm": arm[t], "gripper": [0.0]} for t in range(rows - 1)] + [{}]
            traj.create_dataset(
                "actions/joint_pos", data=np.stack([_pack(a) for a in actions])
            )
            traj.create_dataset("success", data=np.array([False] * (rows - 1) + [success]))
            exo_name = f"episode_{index:08d}_{EXO_CAMERA}_batch_1_of_2.mp4"
            wrist_name = f"episode_{index:08d}_{WRIST_CAMERA}_batch_1_of_2.mp4"
            traj.create_dataset(f"obs/sensor_data/{EXO_CAMERA}", data=_pack_name(exo_name))
            traj.create_dataset(f"obs/sensor_data/{WRIST_CAMERA}", data=_pack_name(wrist_name))
            traj.create_dataset(
                "obs_scene",
                data=json.dumps({"task_description": "Pick up the green cone"}).encode("utf-8"),
            )
            _write_video(house / exo_name, n_frames)
            _write_video(house / wrist_name, n_frames)
    return tmp_path


def test_unpack_ignores_the_zero_padding():
    assert unpack_json_row(_pack({"arm": [1.0]})) == {"arm": [1.0]}


def test_empty_final_action_row_unpacks_to_nothing():
    assert unpack_json_row(_pack({})) == {}


def test_trajectory_yields_one_fewer_step_than_rows(tmp_path):
    root = _build(tmp_path, rows=6)
    episodes = list(iter_episodes(root))
    assert len(episodes) == 1
    # Six recorded rows, five usable transitions: the last action is empty.
    assert episodes[0].n_steps == 5
    assert len(episodes[0].commands) == 5
    assert len(episodes[0].qpos) == 6
    assert all("arm" in c for c in episodes[0].commands)


def test_invalid_trajectories_are_dropped(tmp_path):
    root = _build(tmp_path, rows=5, valid=(True, False))
    assert [e.traj_key for e in iter_episodes(root)] == ["traj_0"]


def test_failed_trajectories_are_dropped_unless_asked_for(tmp_path):
    root = _build(tmp_path, rows=5, success=False)
    assert list(iter_episodes(root)) == []
    assert len(list(iter_episodes(root, require_success=False))) == 1


def test_house_filter_selects_nothing_when_the_house_is_absent(tmp_path):
    root = _build(tmp_path, rows=5)
    assert list(iter_episodes(root, houses=["house_7"])) == []


def test_frame_count_mismatch_is_an_error_not_a_silent_trim(tmp_path):
    root = _build(tmp_path, rows=6, n_frames=9)
    episode = next(iter_episodes(root))
    with pytest.raises(ValueError, match="one-to-one"):
        read_video(episode.exo_video, episode.n_steps + 1)


def test_video_decodes_to_the_recorded_length(tmp_path):
    root = _build(tmp_path, rows=6)
    episode = next(iter_episodes(root))
    frames = read_video(episode.exo_video, episode.n_steps + 1)
    assert frames.shape == (6, HEIGHT, WIDTH, 3)


def test_instruction_comes_from_obs_scene(tmp_path):
    root = _build(tmp_path, rows=5)
    assert next(iter_episodes(root)).instruction == "Pick up the green cone"
