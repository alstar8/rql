"""Grid labelling: the border has to match the clip it is drawn around.

A green border on a failed rollout is worse than no border at all -- it is the
one part of a demo video nobody double-checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from make_grid_video import border, build_grid, find_videos, pad_to, success_from_name  # noqa: E402


def test_outcome_is_read_from_the_rollout_filename():
    assert success_from_name(Path("rollout_007_ok_exo_camera_1.mp4")) is True
    assert success_from_name(Path("rollout_012_fail_exo_camera_1.mp4")) is False
    assert success_from_name(Path("episode_00000003_exo_camera_1_batch_1_of_1.mp4")) is None


def test_rollout_videos_win_over_episode_videos(tmp_path):
    """A directory holding both must use the per-rollout files: those carry the
    outcome, the episode-named ones would all share one key."""
    (tmp_path / "house_1").mkdir()
    (tmp_path / "house_1" / "episode_00000000_exo_camera_1_batch_1_of_1.mp4").touch()
    for i in range(3):
        (tmp_path / f"rollout_{i:03d}_ok_exo_camera_1.mp4").touch()

    found = find_videos(tmp_path, "exo_camera_1")
    assert [p.name for p in found] == [f"rollout_{i:03d}_ok_exo_camera_1.mp4" for i in range(3)]


def test_border_colour_follows_the_outcome():
    frames = np.zeros((2, 20, 20, 3), np.uint8)
    green = border(frames, True)
    red = border(frames, False)
    assert tuple(green[0, 0, 0]) == (60, 200, 60)
    assert tuple(red[0, 0, 0]) == (200, 60, 60)
    assert np.array_equal(border(frames, None), frames)
    assert np.array_equal(green[:, 10, 10], frames[:, 10, 10]), "the middle must stay untouched"


def test_short_clips_hold_their_last_frame():
    frames = np.stack([np.full((4, 4, 3), i, np.uint8) for i in range(3)])
    held = pad_to(frames, 5)
    assert len(held) == 5
    assert np.array_equal(held[3], frames[-1]) and np.array_equal(held[4], frames[-1])


def test_grid_pads_a_short_batch_and_keeps_even_dimensions():
    clips = [np.zeros((3, 10, 10, 3), np.uint8) for _ in range(5)]
    grid = build_grid(clips, nrow=4, ncol=4)
    assert grid.shape[1] % 2 == 0 and grid.shape[2] % 2 == 0
    assert grid.shape[1:3] == (40, 40)
