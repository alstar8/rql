"""Which checkpoint gets kept as the best one.

A run's reported number comes from `agent_best.pt`, so choosing it wrongly is not a
cosmetic bug -- it decides what gets evaluated and published. Two ways it went wrong:

* keeping only the last checkpoint, when RL peaks and then degrades;
* naming a best while the frozen VLA was still driving, which saves an untrained actor.

The second one actually happened to every run in flight on 2026-08-21: the multi-task
actor recorded 0.90 at episode 10 with warmup 60, so its `agent_best.pt` held a policy
that had never been in control.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05.train import checkpoint, scoreable  # noqa: E402


class FakeAgent:
    """Records where it was told to save, so the test can see which files were written."""

    def __init__(self) -> None:
        self.saved: list[str] = []

    def save(self, path: str) -> None:
        self.saved.append(Path(path).name)
        Path(path).write_text("weights")


class FakeBuffer:
    def save(self, path: str) -> None:
        Path(path).write_text("rows")


@pytest.mark.parametrize(
    "episodes_done, warmup, window, expected",
    [
        (10, 60, 10, False),   # deep in warmup: the multi-task bug
        (60, 60, 10, False),   # warmup just ended; the window is still all VLA
        (69, 60, 10, False),   # window straddles the handover
        (70, 60, 10, True),    # first window that is entirely the actor's
        (200, 40, 10, True),
    ],
)
def test_a_best_actor_can_only_be_named_once_the_window_is_all_actor(
    episodes_done, warmup, window, expected
):
    assert scoreable(episodes_done, warmup, window) is expected


def test_an_unscored_checkpoint_leaves_the_best_alone(tmp_path):
    """Warmup checkpoints must save the agent without competing for 'best'."""
    agent, buffer = FakeAgent(), FakeBuffer()

    checkpoint(agent, buffer, tmp_path, 10, 0.9)          # a real score
    assert json.loads((tmp_path / "best.json").read_text()) == {"sr": 0.9, "episode": 10}

    agent.saved.clear()
    checkpoint(agent, buffer, tmp_path, 20, None)          # warmup: no score
    assert agent.saved == ["agent.pt"], "an unscored checkpoint must not touch agent_best"
    assert json.loads((tmp_path / "best.json").read_text()) == {"sr": 0.9, "episode": 10}


def test_the_best_copy_survives_a_collapse(tmp_path):
    """The peak has to outlive the run that walked away from it."""
    agent, buffer = FakeAgent(), FakeBuffer()

    checkpoint(agent, buffer, tmp_path, 240, 0.80)
    checkpoint(agent, buffer, tmp_path, 300, 0.10)

    best = json.loads((tmp_path / "best.json").read_text())
    assert best == {"sr": 0.80, "episode": 240}
    assert (tmp_path / "agent_best.pt").exists()
