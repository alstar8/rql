"""Collector weight reload must survive a torn actor_live.pt write."""

from __future__ import annotations

from pathlib import Path

from pi05.collectors import load_live_weights


class _FakeAgent:
    def __init__(self, fails: int) -> None:
        self.fails = fails
        self.calls = 0

    def load(self, path: str) -> None:
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError("PytorchStreamReader failed reading file data/259")


def test_load_live_weights_retries_torn_read(tmp_path, monkeypatch):
    monkeypatch.setattr("pi05.collectors.time.sleep", lambda _s: None)
    path = tmp_path / "actor_live.pt"
    path.write_bytes(b"ckpt")
    agent = _FakeAgent(fails=2)
    version = load_live_weights(agent, path, -1)
    assert agent.calls == 3
    assert version == path.stat().st_mtime_ns


def test_load_live_weights_skips_unchanged_mtime(tmp_path):
    path = tmp_path / "actor_live.pt"
    path.write_bytes(b"ckpt")
    seen = path.stat().st_mtime_ns
    agent = _FakeAgent(fails=0)
    assert load_live_weights(agent, path, seen) == seen
    assert agent.calls == 0
