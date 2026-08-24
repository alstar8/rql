"""Record the frozen VLA's token sequences while episodes run, for phase-1 training.

The phase-1 autoencoder needs a corpus of token sequences, one per control step. Rather
than build a separate collection path, this wraps the policy the evaluator already talks
to: every inference is served normally and its tokens are appended to a shard.

Shards match what rlt/data.py reads -- npz with ``tokens`` and ``masks``, both ragged
object arrays -- and are written in float16, which is what makes the corpus fit: at
968x2048 a step costs 4 MB in half precision against 8 MB in single.

Each worker writes its own shards, named with the process id, because several evaluators
run at once and must not collide.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

from .action_space import ARM_DOF

log = logging.getLogger(__name__)


class TokenShardWriter:
    """Buffers token sequences and writes them as npz shards rlt/data.py can read.

    Used by the HTTP server (`scripts/serve_pi05_http.py --record-tokens`), which is the
    current collection path: recording there means the corpus is produced by the same
    server, in the same execution regime, as the run that measures the policy -- so the
    encoder is always fit to the state distribution the actor will actually meet.

    Provenance is written alongside the shards. The first corpus had none, and the regime
    it was collected under had to be reverse-engineered from the serving code months
    later; `manifest.json` exists so that is never necessary again.
    """

    def __init__(
        self,
        record_dir: Path,
        *,
        shard_size: int = 256,
        tag: str = "tokens",
        max_sequences: int | None = None,
        provenance: dict | None = None,
    ) -> None:
        self.dir = Path(record_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._shard_size = shard_size
        self._tag = tag
        self._max_sequences = max_sequences
        self._tokens: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._shard_index = 0
        self._written = 0
        self._lock = threading.Lock()
        if provenance:
            import json

            (self.dir / f"manifest_{tag}_pid{os.getpid()}.json").write_text(
                json.dumps(provenance, indent=2, default=str)
            )
        log.info("recording token sequences to %s (tag %s)", self.dir, tag)

    @property
    def written(self) -> int:
        return self._written

    @property
    def full(self) -> bool:
        if self._max_sequences is None:
            return False
        return self._written + len(self._tokens) >= self._max_sequences

    def append(self, tokens: np.ndarray, mask: np.ndarray) -> None:
        """Buffer one sequence. float16 halves the corpus and is what the AE trains on."""
        if self.full:
            return
        with self._lock:
            self._tokens.append(np.asarray(tokens).astype(np.float16))
            self._masks.append(np.asarray(mask).astype(np.float16))
            if len(self._tokens) >= self._shard_size:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._tokens:
            return
        path = self.dir / f"token_replay_{self._tag}_pid{os.getpid()}_s{self._shard_index:04d}.npz"
        # Stack rather than wrap in an object array: pi0.5's prefix length is fixed, and
        # np.array(list, dtype=object) on uniform shapes builds a 3-D object array of
        # Python floats at 8 bytes each -- four times the size of float16.
        np.savez(
            path,
            tokens=np.stack(self._tokens).astype(np.float16),
            masks=np.stack(self._masks).astype(np.float16),
        )
        self._written += len(self._tokens)
        log.info("wrote %s (%d sequences, %d total)", path.name, len(self._tokens), self._written)
        self._tokens.clear()
        self._masks.clear()
        self._shard_index += 1

    def flush(self) -> int:
        """Write whatever is buffered. Called on shutdown so a partial shard survives."""
        with self._lock:
            self._flush_locked()
        return self._written


class TokenRecordingPolicy:
    """Serves actions as absolute joint targets and records the tokens behind them.

    SUPERSEDED by `TokenShardWriter` plus `scripts/serve_pi05_http.py --record-tokens`.
    This drives the websocket path, which converts deltas server-side (plan_time) and so
    can only collect in the pre-refactor regime. Kept to reproduce the first corpus.
    """

    def __init__(
        self,
        frozen,
        record_dir: Path | None = None,
        *,
        shard_size: int = 256,
        tag: str = "tokens",
        max_sequences: int | None = None,
    ) -> None:
        self._frozen = frozen
        self._record_dir = Path(record_dir) if record_dir else None
        self._shard_size = shard_size
        self._tag = tag
        self._tokens: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._shard_index = 0
        self._written = 0
        # A hard cap keeps a long collection run inside the disk budget even if more
        # episodes are queued than planned.
        self._max_sequences = max_sequences
        self._lock = threading.Lock()
        if self._record_dir:
            self._record_dir.mkdir(parents=True, exist_ok=True)
            log.info("recording token sequences to %s", self._record_dir)

    def infer(self, obs: dict) -> dict:
        out = self._frozen.predict(obs)

        if self._record_dir is not None and (
            self._max_sequences is None or self._written + len(self._tokens) < self._max_sequences
        ):
            # float16 halves the corpus; the autoencoder trains on these values, and the
            # reconstruction target is the same precision, so nothing is lost relative to
            # what it is asked to reproduce.
            with self._lock:
                self._tokens.append(out["token_features"].astype(np.float16))
                self._masks.append(out["token_attention_mask"].astype(np.float16))
                if len(self._tokens) >= self._shard_size:
                    self._flush_locked()

        actions = np.array(out["actions"], dtype=np.float32, copy=True)
        state = np.asarray(obs["observation/joint_position"], dtype=np.float32).reshape(-1)[:ARM_DOF]
        # The model speaks deltas; MolmoSpaces executes absolute joint targets.
        actions[..., :ARM_DOF] += state
        return {"actions": actions}

    def _flush_locked(self) -> None:
        if not self._tokens:
            return
        path = self._record_dir / f"token_replay_{self._tag}_pid{os.getpid()}_s{self._shard_index:04d}.npz"
        # Stack rather than wrap in an object array. pi0.5's prefix length is fixed, so
        # the shapes are uniform; np.array(list, dtype=object) on uniform shapes builds a
        # 3-D object array whose elements are Python floats at 8 bytes each -- four times
        # the size of float16, which would have blown the disk budget. The loader iterates
        # the first axis either way.
        np.savez(
            path,
            tokens=np.stack(self._tokens).astype(np.float16),
            masks=np.stack(self._masks).astype(np.float16),
        )
        self._written += len(self._tokens)
        log.info("wrote %s (%d sequences, %d total)", path.name, len(self._tokens), self._written)
        self._tokens.clear()
        self._masks.clear()
        self._shard_index += 1

    def flush(self) -> int:
        """Write whatever is buffered; returns the total number of sequences recorded."""
        if self._record_dir is None:
            return 0
        with self._lock:
            self._flush_locked()
        return self._written

    def reset(self) -> None:
        reset = getattr(self._frozen, "reset", None)
        if callable(reset):
            reset()
