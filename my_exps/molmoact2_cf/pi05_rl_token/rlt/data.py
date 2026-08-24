"""Readers for the pre-collected demo1k data.

Three npz families, all written by molmoact2_cf/chunk_replay.py:

  token_replay_*_s*.npz        VLA token sequences -> the RL-token autoencoder

The chunk-transition readers that used to live here belonged to the actor/critic
stages and were removed with them.

Nothing here trains; `scripts/validate_data.py` prints what these contain so a
bad path fails loudly before a training run rather than 20 minutes into one.
"""

from __future__ import annotations

from glob import glob
from pathlib import Path

import numpy as np
import torch

from .config import VLA_TOKEN_DIM


def resolve(pattern: str) -> list[Path]:
    """Glob -> sorted paths, with a clear error when nothing matches."""
    paths = sorted(Path(p) for p in glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no files match {pattern!r}")
    return paths


def collate_tokens(
    tokens: list[np.ndarray], masks: list[np.ndarray], device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad ragged (S_i, D) sequences into (B, S_max, D) + (B, S_max) mask."""
    b = len(tokens)
    s_max = max(int(m.shape[0]) for m in masks)
    dim = int(tokens[0].shape[1])
    tok = np.zeros((b, s_max, dim), dtype=np.float32)
    msk = np.zeros((b, s_max), dtype=np.float32)
    for i, (t, m) in enumerate(zip(tokens, masks)):
        s = int(m.shape[0])
        tok[i, :s] = t[:s]
        msk[i, :s] = m[:s]
    return torch.as_tensor(tok, device=device), torch.as_tensor(msk, device=device)


class TokenSequences:
    """Ragged VLA token sequences, kept float16 so the corpus fits in RAM.

    The full merged set is 24,888 sequences of ~481x2560 -- ~61 GB decompressed.
    `max_sequences` caps it; the AE converges long before it needs all of them.
    """

    def __init__(self, tokens: list[np.ndarray], masks: list[np.ndarray]) -> None:
        self.tokens = tokens
        self.masks = masks
        self._memmap: np.ndarray | None = None  # set by from_cache()
        self._lengths: np.ndarray | None = None

    def __len__(self) -> int:
        return self._memmap.shape[0] if self._memmap is not None else len(self.tokens)

    @classmethod
    def load(cls, paths: list[Path], max_sequences: int | None = None) -> "TokenSequences":
        """Load shards, taking an equal quota from each when capping.

        The cap must not be a prefix of the concatenated shards: they sort as
        token_replay_droid_* before token_replay_molmobot_*, so a prefix would
        quietly train the AE on DROID only and never show a MolmoBot frame.
        """
        quota = None if not max_sequences else max(1, max_sequences // len(paths))

        tokens: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for path in paths:
            taken = 0
            with np.load(path, allow_pickle=True) as data:
                for t, m in zip(data["tokens"], data["masks"]):
                    if quota is not None and taken >= quota:
                        break
                    t = np.asarray(t, dtype=np.float16)
                    if t.ndim != 2 or t.shape[1] != VLA_TOKEN_DIM:
                        raise ValueError(f"{path}: expected (S,{VLA_TOKEN_DIM}) tokens, got {t.shape}")
                    tokens.append(t)
                    masks.append(np.asarray(m, dtype=np.uint8))
                    taken += 1
            print(f"  {path.name}: {taken} sequences")

        if not tokens:
            raise RuntimeError(f"no token sequences in {paths}")
        return cls(tokens, masks)

    @classmethod
    def from_cache(cls, cache_dir: Path) -> "TokenSequences":
        """Load the memmap written by scripts/prepare_token_cache.py.

        Rows stay on disk; the OS page cache serves every sweep process from one
        copy instead of each decompressing its own.
        """
        tokens = np.load(cache_dir / "tokens.npy", mmap_mode="r")
        lengths = np.load(cache_dir / "lengths.npy")
        if tokens.shape[0] != lengths.shape[0]:
            raise ValueError(f"{cache_dir}: {tokens.shape[0]} rows but {lengths.shape[0]} lengths")
        obj = cls([], [])
        obj._memmap = tokens
        obj._lengths = lengths
        return obj

    def sample(self, batch_size: int, device, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        if self._memmap is not None:
            i = rng.integers(0, self._memmap.shape[0], size=batch_size)
            lens = self._lengths[i]
            s = int(lens.max())
            tok = np.ascontiguousarray(self._memmap[i, :s], dtype=np.float32)
            msk = (np.arange(s)[None, :] < lens[:, None]).astype(np.float32)
            return torch.as_tensor(tok, device=device), torch.as_tensor(msk, device=device)

        i = rng.integers(0, len(self.tokens), size=batch_size)
        return collate_tokens([self.tokens[k] for k in i], [self.masks[k] for k in i], device)

    def stats(self) -> dict:
        if self._memmap is not None:
            lengths = self._lengths
            return {
                "sequences": int(lengths.shape[0]),
                "len_min": int(lengths.min()),
                "len_mean": float(lengths.mean()),
                "len_max": int(lengths.max()),
                "gib_float16": float(self._memmap.nbytes / 2**30),
                "source": "memmap cache",
            }
        lengths = np.array([int(m.shape[0]) for m in self.masks])
        return {
            "sequences": len(self),
            "len_min": int(lengths.min()),
            "len_mean": float(lengths.mean()),
            "len_max": int(lengths.max()),
            "gib_float16": float(lengths.sum() * VLA_TOKEN_DIM * 2 / 2**30),
        }
