"""Frame writing, checked without building a real LeRobot dataset.

The interesting behaviour is which steps get written and what lands in each frame, so a
recording stand-in for the dataset is enough and keeps the test fast.
"""

import json

import numpy as np
import pytest

from pi05.action_space import GRIPPER_CMD_MAX
from pi05.convert import DROP_FIRST_STEPS, write_episode
from pi05.demo_reader import Episode


class RecordingDataset:
    def __init__(self):
        self.frames = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


@pytest.fixture
def episode(tmp_path, monkeypatch):
    n_steps = 5
    arm = [[0.1 * t + 0.01 * j for j in range(7)] for t in range(n_steps + 1)]
    ep = Episode(
        house="house_0",
        h5_path=tmp_path / "trajectories_batch_1_of_2.h5",
        traj_key="traj_0",
        instruction="Pick up the green cone",
        n_steps=n_steps,
        qpos=[{"arm": a, "base": [], "gripper": [0.0, 0.0]} for a in arm],
        commands=[{"arm": arm[t + 1], "gripper": [GRIPPER_CMD_MAX]} for t in range(n_steps)],
        exo_video=tmp_path / "exo.mp4",
        wrist_video=tmp_path / "wrist.mp4",
        success=True,
    )
    # Frames are read through read_video; substitute a plain array of the right length.
    fake = np.zeros((n_steps + 1, 48, 64, 3), dtype=np.uint8)
    monkeypatch.setattr("pi05.convert.read_video", lambda path, expected: fake)
    return ep


def test_first_transition_is_dropped_by_default(episode):
    ds = RecordingDataset()
    written = write_episode(ds, episode, duplicate_exterior2=True)
    assert DROP_FIRST_STEPS == 1
    assert written == episode.n_steps - 1
    assert len(ds.frames) == episode.n_steps - 1
    assert ds.saved == 1


def test_first_transition_can_be_kept(episode):
    ds = RecordingDataset()
    written = write_episode(ds, episode, duplicate_exterior2=True, drop_first_steps=0)
    assert written == episode.n_steps
    assert len(ds.frames) == episode.n_steps


def test_dropping_shifts_which_step_lands_first(episode):
    kept = RecordingDataset()
    write_episode(kept, episode, duplicate_exterior2=True, drop_first_steps=0)
    dropped = RecordingDataset()
    write_episode(dropped, episode, duplicate_exterior2=True, drop_first_steps=1)
    # The dropped run must begin at the second step, not merely be one frame shorter.
    np.testing.assert_allclose(
        dropped.frames[0]["joint_position"], kept.frames[1]["joint_position"]
    )


def test_frame_carries_the_normalised_instruction(episode):
    ds = RecordingDataset()
    write_episode(ds, episode, duplicate_exterior2=True)
    assert ds.frames[0]["task"] == "pick up the green cone."


def test_exterior2_is_a_placeholder_and_can_be_omitted(episode):
    # The key must exist for openpi's repack transform, but the model never sees it, so
    # it carries no pixels rather than a costly copy of the shoulder view.
    with_dup = RecordingDataset()
    write_episode(with_dup, episode, duplicate_exterior2=True)
    placeholder = with_dup.frames[0]["exterior_image_2_left"]
    assert placeholder.shape == (224, 224, 3)
    assert not placeholder.any()

    without = RecordingDataset()
    write_episode(without, episode, duplicate_exterior2=False)
    assert "exterior_image_2_left" not in without.frames[0]


def test_written_action_reconstructs_the_recorded_command(episode):
    ds = RecordingDataset()
    write_episode(ds, episode, duplicate_exterior2=True, drop_first_steps=0)
    frame = ds.frames[2]
    rebuilt = frame["actions"][:7] + frame["joint_position"]
    np.testing.assert_allclose(rebuilt, episode.commands[2]["arm"], atol=1e-5)


def test_images_are_resized_to_the_model_resolution(episode):
    ds = RecordingDataset()
    write_episode(ds, episode, duplicate_exterior2=True)
    assert ds.frames[0]["exterior_image_1_left"].shape == (224, 224, 3)
