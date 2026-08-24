"""Data loading and z re-encoding, on synthetic npz in the real on-disk format.

The alignment between chunk rows and token rows is the one place where a silent
mistake would poison every downstream number, so it is tested explicitly.
"""

import numpy as np
import pytest

from rlt.config import VLA_TOKEN_DIM
from rlt.data import TokenSequences, collate_tokens, resolve

C, A, P, Z = 8, 8, 8, 16


def write_token_shard(path, n, dim=VLA_TOKEN_DIM, rng=None):
    rng = rng or np.random.default_rng(0)
    lengths = rng.integers(4, 12, size=n)
    tokens = np.array([rng.standard_normal((int(s), dim)).astype(np.float16) for s in lengths], dtype=object)
    masks = np.array([np.ones(int(s), dtype=np.uint8) for s in lengths], dtype=object)
    np.savez_compressed(path, tokens=tokens, masks=masks, token_dim=dim, max_seq=512)


def test_resolve_raises_on_no_match(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve(str(tmp_path / "nope_*.npz"))


def test_collate_pads_and_masks():
    tok, msk = collate_tokens(
        [np.ones((3, 4), np.float16), np.ones((5, 4), np.float16)],
        [np.ones(3, np.uint8), np.ones(5, np.uint8)],
        "cpu",
    )
    assert tok.shape == (2, 5, 4) and msk.shape == (2, 5)
    assert msk[0].tolist() == [1, 1, 1, 0, 0]
    assert tok[0, 3:].abs().sum() == 0


def test_token_sequences_load_and_cap(tmp_path):
    p = tmp_path / "token_replay_s0.npz"
    write_token_shard(p, 20)
    assert len(TokenSequences.load([p])) == 20
    assert len(TokenSequences.load([p], max_sequences=7)) == 7

    data = TokenSequences.load([p])
    tokens, mask = data.sample(4, "cpu", np.random.default_rng(0))
    assert tokens.shape[0] == 4 and tokens.shape[2] == VLA_TOKEN_DIM
    assert data.stats()["sequences"] == 20


def test_token_dim_mismatch_fails_loudly(tmp_path):
    p = tmp_path / "bad.npz"
    write_token_shard(p, 3, dim=64)
    with pytest.raises(ValueError, match="expected"):
        TokenSequences.load([p])
