"""Non-method helpers for the AR-QDFL FastSAC three-phase runner.

Covers stratified offline/online batch mixing, plot-step mapping that hides
critic warmup from ``eval.csv``, and an append-only online transition journal
for exact resume of the online replay partition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from utils.datasets import ReplayBuffer, get_size

PHASE_OFFLINE = "offline"
PHASE_WARMUP = "warmup"
PHASE_ONLINE = "online"
VALID_PHASES = (PHASE_OFFLINE, PHASE_WARMUP, PHASE_ONLINE)

DEFAULT_OFFLINE_STEPS = 1_000_000
DEFAULT_WARMUP_UPDATES = 100_000
DEFAULT_ONLINE_STEPS = 1_000_000


def online_replay_fraction(
    online_env_step: int,
    *,
    ramp_steps: int,
    fraction_max: float,
) -> float:
    """Linear ramp of the online-batch share toward ``fraction_max``.

    Fraction is 0 at ``online_env_step == 0`` and reaches ``fraction_max`` after
    ``ramp_steps`` environment interactions (clamped).
    """
    if ramp_steps <= 0:
        return float(np.clip(fraction_max, 0.0, 1.0))
    progress = min(1.0, max(0.0, float(online_env_step) / float(ramp_steps)))
    return float(np.clip(progress * float(fraction_max), 0.0, 1.0))


def stratified_batch_counts(
    batch_size: int,
    online_fraction: float,
    *,
    online_size: int,
) -> tuple[int, int]:
    """Return ``(n_offline, n_online)`` summing to ``batch_size``.

    When the online partition is empty, the entire batch is offline. Online
    count is floored so offline always receives the remainder.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if online_size <= 0 or online_fraction <= 0.0:
        return int(batch_size), 0
    n_online = int(np.floor(float(batch_size) * float(online_fraction)))
    n_online = min(n_online, int(online_size), int(batch_size))
    n_offline = int(batch_size) - n_online
    return n_offline, n_online


def merge_batches(
    offline_batch: Mapping[str, np.ndarray],
    online_batch: Optional[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Concatenate trajectory-style batches along the batch axis (axis=1).

    Offline-only batches are returned as a shallow copy of arrays.
    """
    if online_batch is None or not online_batch:
        return {k: np.asarray(v) for k, v in offline_batch.items()}
    keys = set(offline_batch.keys())
    if keys != set(online_batch.keys()):
        raise ValueError(
            f"batch key mismatch: offline={sorted(keys)} "
            f"online={sorted(online_batch.keys())}"
        )
    merged: dict[str, np.ndarray] = {}
    for key in offline_batch:
        off = np.asarray(offline_batch[key])
        on = np.asarray(online_batch[key])
        if off.ndim < 2 or on.ndim < 2:
            raise ValueError(
                f"expected traj layout (H+1, B, ...) for key={key}, "
                f"got offline={off.shape} online={on.shape}"
            )
        if off.shape[0] != on.shape[0] or off.shape[2:] != on.shape[2:]:
            raise ValueError(
                f"incompatible shapes for key={key}: "
                f"offline={off.shape} online={on.shape}"
            )
        merged[key] = np.concatenate([off, on], axis=1)
    return merged


def effective_online_traj_size(online_replay: Any) -> int:
    """Online size usable by RQL ``sample_traj`` (needs ≥1 finished episode)."""
    size = int(getattr(online_replay, "size", 0) or 0)
    if size <= 0:
        return 0
    terminal_locs = getattr(online_replay, "terminal_locs", None)
    if terminal_locs is None or len(terminal_locs) == 0:
        return 0
    return size


def sample_stratified_batch(
    offline_replay: Any,
    online_replay: Any,
    batch_size: int,
    *,
    online_env_step: int,
    ramp_steps: int,
    fraction_max: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Sample a stratified offline/online batch and return diagnostics."""
    online_size = effective_online_traj_size(online_replay)
    fraction = online_replay_fraction(
        online_env_step,
        ramp_steps=ramp_steps,
        fraction_max=fraction_max,
    )
    n_offline, n_online = stratified_batch_counts(
        batch_size, fraction, online_size=online_size
    )
    if n_offline > 0:
        offline_batch = offline_replay.sample(n_offline)
    else:
        offline_batch = None
    online_batch = online_replay.sample(n_online) if n_online > 0 else None
    if offline_batch is None:
        if online_batch is None:
            raise RuntimeError("both offline and online batch partitions empty")
        batch = {k: np.asarray(v) for k, v in online_batch.items()}
    else:
        batch = merge_batches(offline_batch, online_batch)
    info = {
        "online_replay_fraction": fraction,
        "n_offline": n_offline,
        "n_online": n_online,
        "online_replay_size": int(getattr(online_replay, "size", 0) or 0),
        "online_traj_sample_size": online_size,
    }
    return batch, info


def plot_step(
    phase: str,
    *,
    offline_update_count: int = 0,
    online_env_step: int = 0,
    offline_steps: int = DEFAULT_OFFLINE_STEPS,
) -> Optional[int]:
    """Map phase counters to the absolute eval/plot x-axis.

    - offline: ``offline_update_count`` (0..offline_steps)
    - warmup: ``None`` (hidden from eval.csv)
    - online: ``offline_steps + online_env_step``
    """
    if phase == PHASE_OFFLINE:
        return int(offline_update_count)
    if phase == PHASE_WARMUP:
        return None
    if phase == PHASE_ONLINE:
        return int(offline_steps) + int(online_env_step)
    raise ValueError(f"unknown phase {phase!r}; expected one of {VALID_PHASES}")


def should_write_eval(phase: str) -> bool:
    """Warmup diagnostics belong in warmup.csv, never eval.csv."""
    return phase != PHASE_WARMUP


class OnlineTransitionJournal:
    """Append-only on-disk journal of online transitions for exact replay rebuild."""

    def __init__(self, root: str | Path, *, shard_size: int = 10_000):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self._pending: list[dict[str, np.ndarray]] = []
        self._meta_path = self.root / "journal_meta.json"
        self._count = 0
        self._shard_idx = 0
        if self._meta_path.is_file():
            meta = json.loads(self._meta_path.read_text())
            self._count = int(meta.get("count", 0))
            self._shard_idx = int(meta.get("shard_idx", 0))

    @property
    def count(self) -> int:
        return int(self._count) + len(self._pending)

    def _shard_path(self, shard_idx: int) -> Path:
        return self.root / f"shard_{shard_idx:06d}.npz"

    def append(self, transition: Mapping[str, Any]) -> None:
        stored = {
            key: np.asarray(value, copy=True) for key, value in transition.items()
        }
        self._pending.append(stored)
        if len(self._pending) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            self._write_meta()
            return
        keys = list(self._pending[0].keys())
        payload = {
            key: np.stack([row[key] for row in self._pending], axis=0)
            for key in keys
        }
        path = self._shard_path(self._shard_idx)
        np.savez_compressed(path, **payload)
        self._count += len(self._pending)
        self._pending.clear()
        self._shard_idx += 1
        self._write_meta()

    def _write_meta(self) -> None:
        meta = {
            "count": int(self._count),
            "shard_idx": int(self._shard_idx),
            "pending": len(self._pending),
            "shard_size": int(self.shard_size),
        }
        self._meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))

    def iter_transitions(self):
        """Yield stored transitions in append order (flushes pending first)."""
        self.flush()
        for shard_idx in range(self._shard_idx):
            path = self._shard_path(shard_idx)
            if not path.is_file():
                raise FileNotFoundError(f"missing journal shard: {path}")
            with np.load(path, allow_pickle=False) as data:
                keys = list(data.keys())
                n = int(data[keys[0]].shape[0])
                arrays = {key: data[key] for key in keys}
            for i in range(n):
                yield {key: arrays[key][i] for key in keys}

    def rebuild_replay(
        self,
        example_transition: Mapping[str, Any],
        max_size: int,
        *,
        config: Optional[Mapping[str, Any]] = None,
        p_aug=None,
        frame_stack=None,
    ) -> ReplayBuffer:
        """Rebuild an online-only ``ReplayBuffer`` from the journal."""
        replay = ReplayBuffer.create(dict(example_transition), size=int(max_size))
        replay.config = config
        replay.p_aug = p_aug
        replay.frame_stack = frame_stack
        for transition in self.iter_transitions():
            replay.add_transition(transition)
        return replay

    def save_snapshot(self, path: str | Path) -> None:
        """Write a single-file snapshot of all journaled transitions."""
        self.flush()
        transitions = list(self.iter_transitions())
        path = Path(path)
        if not transitions:
            np.savez_compressed(path, __empty__=np.asarray(0, dtype=np.int32))
            return
        keys = list(transitions[0].keys())
        payload = {
            key: np.stack([row[key] for row in transitions], axis=0)
            for key in keys
        }
        np.savez_compressed(path, **payload)


def save_runner_state(path: str | Path, state: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(state)
    for key in ("numpy_rng_state", "observation"):
        if key in serializable and serializable[key] is not None:
            # Handled by companion .npz; drop from JSON.
            serializable.pop(key, None)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True))


def load_runner_state(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def save_resume_blobs(
    directory: str | Path,
    *,
    numpy_rng: np.random.RandomState,
    observation: Optional[np.ndarray],
    online_rng_key: Optional[Sequence[int]] = None,
) -> None:
    """Persist non-JSON resume blobs next to ``runner_state.json``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng_state = numpy_rng.get_state()
    # RandomState.get_state() -> ('MT19937', key_array, pos, has_gauss, cached)
    np.savez_compressed(
        directory / "numpy_rng.npz",
        key=np.asarray(rng_state[1], dtype=np.uint32),
        pos=np.asarray(rng_state[2], dtype=np.int64),
        has_gauss=np.asarray(rng_state[3], dtype=np.int64),
        cached_gaussian=np.asarray(rng_state[4], dtype=np.float64),
    )
    if observation is not None:
        np.savez_compressed(
            directory / "observation.npz",
            observation=np.asarray(observation),
        )
    if online_rng_key is not None:
        np.savez_compressed(
            directory / "online_rng.npz",
            key=np.asarray(online_rng_key, dtype=np.uint32),
        )


def load_resume_blobs(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory)
    out: dict[str, Any] = {}
    rng_path = directory / "numpy_rng.npz"
    if rng_path.is_file():
        with np.load(rng_path) as data:
            state = (
                "MT19937",
                np.asarray(data["key"], dtype=np.uint32),
                int(data["pos"]),
                int(data["has_gauss"]),
                float(data["cached_gaussian"]),
            )
            rng = np.random.RandomState()
            rng.set_state(state)
            out["numpy_rng"] = rng
    obs_path = directory / "observation.npz"
    if obs_path.is_file():
        with np.load(obs_path) as data:
            out["observation"] = np.asarray(data["observation"])
    key_path = directory / "online_rng.npz"
    if key_path.is_file():
        with np.load(key_path) as data:
            out["online_rng_key"] = np.asarray(data["key"], dtype=np.uint32)
    return out


def example_transition_from_dataset(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Build a single-step transition dict for ``ReplayBuffer.create``."""
    size = get_size(dataset)
    if size <= 0:
        raise ValueError("cannot build example transition from empty dataset")
    return {
        "observations": np.asarray(dataset["observations"][0]),
        "actions": np.asarray(dataset["actions"][0]),
        "rewards": np.asarray(dataset["rewards"][0]),
        "terminals": np.asarray(dataset["terminals"][0]),
        "masks": np.asarray(dataset["masks"][0]),
        "next_observations": np.asarray(dataset["next_observations"][0]),
    }


def env_action_from_agent(action: np.ndarray) -> np.ndarray:
    """Squeeze horizon-1 AR actions for Gymnasium ``env.step``."""
    action = np.asarray(action)
    if action.ndim == 2 and action.shape[0] == 1:
        return action[0].copy()
    return action.copy()
