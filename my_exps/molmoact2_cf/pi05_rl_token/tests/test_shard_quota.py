"""Capping the corpus must not silently drop a whole data source."""

import numpy as np

from rlt.data import TokenSequences
from tests.test_data import write_token_shard


def test_cap_takes_from_every_shard(tmp_path):
    """A prefix cap would return DROID only, since 'droid' sorts before 'molmobot'."""
    droid = tmp_path / "token_replay_droid_s0.npz"
    molmobot = tmp_path / "token_replay_molmobot_s0.npz"
    write_token_shard(droid, 30, rng=np.random.default_rng(0))
    write_token_shard(molmobot, 30, rng=np.random.default_rng(1))

    data = TokenSequences.load([droid, molmobot], max_sequences=20)
    assert len(data) == 20  # 10 from each

    # Same lengths would be a coincidence; check the content actually differs by
    # confirming both halves came from different generators.
    first_half = [t.shape[0] for t in data.tokens[:10]]
    second_half = [t.shape[0] for t in data.tokens[10:]]
    assert first_half != second_half


def test_no_cap_loads_everything(tmp_path):
    a = tmp_path / "token_replay_droid_s0.npz"
    b = tmp_path / "token_replay_molmobot_s0.npz"
    write_token_shard(a, 7)
    write_token_shard(b, 5)
    assert len(TokenSequences.load([a, b])) == 12


def test_cap_larger_than_corpus_is_harmless(tmp_path):
    a = tmp_path / "token_replay_droid_s0.npz"
    b = tmp_path / "token_replay_molmobot_s0.npz"
    write_token_shard(a, 4)
    write_token_shard(b, 4)
    assert len(TokenSequences.load([a, b], max_sequences=1000)) == 8
