"""Skip-collect pretrain: load a saved replay npz and attach token shards for L_ro."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05.config import RLConfig  # noqa: E402
from pi05.train import attach_token_replay, is_replay_file  # noqa: E402
from rlt.config import VLA_TOKEN_DIM  # noqa: E402


def test_is_replay_file_false_when_empty_or_missing(tmp_path):
    assert not is_replay_file(RLConfig())
    assert not is_replay_file(RLConfig(vla_traj=str(tmp_path / "missing.npz")))
    assert not is_replay_file(RLConfig(vla_traj=str(tmp_path)))


def test_is_replay_file_true_for_an_existing_npz(tmp_path):
    path = tmp_path / "buffer.npz"
    path.write_bytes(b"not empty")
    assert is_replay_file(RLConfig(vla_traj=str(path)))


def test_attach_token_replay_loads_shards(tmp_path):
    rng = np.random.default_rng(0)
    n, length = 4, 6
    tokens = np.array(
        [rng.standard_normal((length, VLA_TOKEN_DIM)).astype(np.float16) for _ in range(n)],
        dtype=object,
    )
    masks = np.array([np.ones(length, dtype=np.uint8) for _ in range(n)], dtype=object)
    shard = tmp_path / "token_replay_s0.npz"
    np.savez_compressed(shard, tokens=tokens, masks=masks)

    agent = SimpleNamespace(token_corpus=None)
    loaded = attach_token_replay(agent, str(tmp_path / "*.npz"), max_sequences=None)
    assert loaded == n
    assert len(agent.token_corpus) == n


def test_attach_token_replay_missing_glob_fails(tmp_path):
    agent = SimpleNamespace(token_corpus=None)
    with pytest.raises(FileNotFoundError, match="matched nothing"):
        attach_token_replay(agent, str(tmp_path / "nope_*.npz"))
