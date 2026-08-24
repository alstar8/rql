"""The learner loop end to end, including the branches that run on a timer.

Publishing weights, the console line and checkpointing fire every few seconds to
ten minutes, so a full test suite can pass while one of them is broken -- which
is exactly what happened: a stale variable name in the checkpoint branch killed
a run ten minutes in. Here the intervals are set to zero so every branch runs on
the first pass.
"""

from __future__ import annotations

import json
import queue as queue_module

import numpy as np
import pytest

from rlt import train_parallel
from rlt.agent import RLTokenAgent
from rlt.config import OnlineConfig
from rlt.logging_utils import RunLogger
from rlt.replay import ChunkReplay
from rlt.shared import WeightLink

STATE_DIM, CHUNK, ACTION_DIM = 6, 4, 3
CHUNK_DIM = CHUNK * ACTION_DIM


class FakeQueue:
    """Hands out a scripted sequence, then behaves like an empty queue."""

    def __init__(self, items):
        self.items = list(items)

    def get(self, timeout=None):
        if not self.items:
            raise queue_module.Empty
        return self.items.pop(0)

    def qsize(self):
        return len(self.items)


def row(seed: int, reward: float = 0.0) -> dict:
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=CHUNK_DIM).astype(np.float32)
    return {
        "state": rng.normal(size=STATE_DIM).astype(np.float32),
        "action": reference + rng.normal(scale=0.05, size=CHUNK_DIM).astype(np.float32),
        "reference": reference,
        "next_state": rng.normal(size=STATE_DIM).astype(np.float32),
        "next_reference": rng.normal(size=CHUNK_DIM).astype(np.float32),
        "reward": reward,
        "done": 1.0,
    }


def episode_message(index: int, success: float) -> tuple:
    return (
        "episode",
        {
            "episode": index,
            "benchmark_idx": 134,
            "collector": index % 2,
            "phase": 1,
            "success": success,
            "steps": 100,
            "sec_per_episode": 40.0,
            "vla_ms_per_call": 420.0,
        },
    )


def test_learner_absorbs_logs_publishes_and_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(train_parallel, "PUBLISH_EVERY", 0.0)
    monkeypatch.setattr(train_parallel, "CONSOLE_EVERY", 0.0)
    monkeypatch.setattr(train_parallel, "CHECKPOINT_EVERY", 0.0)

    cfg = OnlineConfig(episode_idx=134, device="cpu", hidden_dim=16, batch_size=8, utd=5)
    agent = RLTokenAgent(cfg, STATE_DIM, CHUNK_DIM, CHUNK)
    buffer = ChunkReplay(1024, STATE_DIM, CHUNK_DIM, seed=0)

    rows = [("rows", [row(i, reward=float(i % 2)) for i in range(16)])]
    messages = rows + [episode_message(0, 1.0), episode_message(1, 0.0), ("done", 0)]

    logger = RunLogger(tmp_path / "metrics.jsonl", None)
    episodes_log = RunLogger(tmp_path / "episodes.jsonl", None)
    train_parallel.learn(
        cfg, agent, buffer, FakeQueue(messages), WeightLink(tmp_path / "actor.pt"),
        logger, episodes_log, tmp_path, n_workers=1,
    )
    logger.close()
    episodes_log.close()

    assert len(buffer) == 16
    # utd iterations per stored row, each with critic_updates_per_actor critic steps
    assert agent.critic_steps == cfg.utd * 16 * cfg.critic_updates_per_actor
    assert agent.actor_steps == cfg.utd * 16

    for name in ("agent.pt", "buffer.npz", "progress.json", "actor.pt"):
        assert (tmp_path / name).exists(), f"{name} was not written"
    assert json.loads((tmp_path / "progress.json").read_text())["episodes_done"] == 2

    metrics = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [m["step"] for m in metrics] == [1, 2], "one monotonic point per finished episode"
    assert metrics[-1]["sr_total"] == pytest.approx(0.5)
    assert metrics[-1]["updates_per_row"] == pytest.approx(cfg.utd)
    assert "success" not in metrics[-1], "raw per-episode fields belong in episodes.jsonl"

    episodes = [json.loads(line) for line in (tmp_path / "episodes.jsonl").read_text().splitlines()]
    assert [e["success"] for e in episodes] == [1.0, 0.0]
