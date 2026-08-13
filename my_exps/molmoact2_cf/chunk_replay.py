"""Chunk-aligned replay for RLT-style MolmoAct2 CF training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rlt_models import ACTION_DIM, CHUNK_SIZE, FEATURE_DIM, Z_DIM


FULL_ACTION_HORIZON = 15
PADDED_ACTION_DIM = 32


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _rng_state_json(rng: np.random.Generator) -> str:
    return json.dumps(
        rng.bit_generator.state,
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )


def _restore_rng_state(
    rng: np.random.Generator,
    encoded_state: np.ndarray,
) -> np.random.Generator:
    value = np.asarray(encoded_state).item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    state = json.loads(str(value))
    if not isinstance(state, dict):
        raise ValueError("replay RNG state must decode to a JSON object")
    bit_generator_name = state.get("bit_generator")
    if not isinstance(bit_generator_name, str):
        raise ValueError("replay RNG state is missing bit_generator")
    if rng.bit_generator.__class__.__name__ != bit_generator_name:
        bit_generator_type = getattr(np.random, bit_generator_name, None)
        if bit_generator_type is None:
            raise ValueError(f"unsupported numpy bit generator {bit_generator_name!r}")
        try:
            rng = np.random.Generator(bit_generator_type())
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"unsupported numpy bit generator {bit_generator_name!r}"
            ) from error
    rng.bit_generator.state = state
    return rng


def _outcome_counts(rows: list[Any]) -> tuple[int, int]:
    positive = sum(float(row.success) > 0.5 for row in rows)
    return int(positive), int(len(rows) - positive)


def _successful_episode_count(rows: list[Any]) -> int:
    return len(
        {
            int(row.episode_id)
            for row in rows
            if float(row.success) > 0.5
        }
    )


def _storage_nbytes(rows: list[Any]) -> int:
    """Count ndarray payload bytes once per shared ndarray object."""
    seen: set[int] = set()
    total = 0
    for row in rows:
        for value in vars(row).values():
            if not isinstance(value, np.ndarray):
                continue
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            total += int(value.nbytes)
    return total


def _sample_indices(
    rows: list[Any],
    batch_size: int,
    pos_frac: float,
    rng: np.random.Generator,
    *,
    require_both_outcomes: bool,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not rows:
        raise RuntimeError("empty replay")
    success = np.asarray([float(row.success) > 0.5 for row in rows])
    positive = np.flatnonzero(success)
    negative = np.flatnonzero(~success)
    if require_both_outcomes:
        if batch_size < 2:
            raise ValueError(
                "require_both_outcomes needs batch_size of at least 2"
            )
        if not len(positive) or not len(negative):
            raise RuntimeError(
                "require_both_outcomes needs a replay with both outcomes"
            )

    if len(positive) and len(negative):
        positive_count = int(round(batch_size * pos_frac))
        if require_both_outcomes:
            positive_count = min(max(positive_count, 1), batch_size - 1)
        else:
            positive_count = min(positive_count, batch_size, len(positive))
    elif len(positive):
        positive_count = batch_size
    else:
        positive_count = 0
    negative_count = batch_size - positive_count

    sampled: list[int] = []
    if positive_count:
        sampled.extend(
            rng.choice(
                positive,
                size=positive_count,
                replace=positive_count > len(positive),
            ).tolist()
        )
    if negative_count:
        sampled.extend(
            rng.choice(
                negative,
                size=negative_count,
                replace=negative_count > len(negative),
            ).tolist()
        )
    indices = np.asarray(sampled, dtype=np.int64)
    rng.shuffle(indices)
    return indices


def _episode_balanced_indices(
    rows: list[Any],
    batch_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not rows:
        raise RuntimeError("empty replay")
    rows_by_episode: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        rows_by_episode.setdefault(int(row.episode_id), []).append(index)

    episode_ids = np.asarray(list(rows_by_episode), dtype=np.int64)
    episode_order = rng.permutation(episode_ids)
    base_count, extra_count = divmod(batch_size, len(episode_ids))
    sampled: list[int] = []
    for position, episode_id in enumerate(episode_order.tolist()):
        count = base_count + int(position < extra_count)
        if count == 0:
            continue
        pool = np.asarray(rows_by_episode[int(episode_id)], dtype=np.int64)
        sampled.extend(
            rng.choice(
                pool,
                size=count,
                replace=count > len(pool),
            ).tolist()
        )
    indices = np.asarray(sampled, dtype=np.int64)
    rng.shuffle(indices)
    return indices


def _outcome_retention_indices(
    success: np.ndarray,
    capacity: int,
    pos_frac: float,
) -> np.ndarray:
    """Keep the newest quota from each outcome, then restore row order."""
    count = int(len(success))
    if capacity >= count:
        return np.arange(count, dtype=np.int64)
    if capacity <= 0:
        return np.empty((0,), dtype=np.int64)

    positive = np.flatnonzero(np.asarray(success, dtype=np.float32) > 0.5)
    negative = np.flatnonzero(np.asarray(success, dtype=np.float32) <= 0.5)
    if not len(positive):
        return negative[-capacity:]
    if not len(negative):
        return positive[-capacity:]

    target_positive = int(round(capacity * pos_frac))
    if capacity >= 2:
        target_positive = min(max(target_positive, 1), capacity - 1)
    else:
        target_positive = min(max(target_positive, 0), 1)
    positive_count = min(len(positive), target_positive)
    negative_count = min(len(negative), capacity - target_positive)
    remaining = capacity - positive_count - negative_count
    if remaining:
        add_positive = min(len(positive) - positive_count, remaining)
        positive_count += add_positive
        remaining -= add_positive
    if remaining:
        negative_count += min(len(negative) - negative_count, remaining)

    kept = np.concatenate(
        (
            positive[-positive_count:] if positive_count else positive[:0],
            negative[-negative_count:] if negative_count else negative[:0],
        )
    )
    return np.sort(kept)


def _pose_outcome_retention_indices(
    success: np.ndarray,
    episode_ids: np.ndarray,
    capacity: int,
    pos_frac: float,
    pose_cycle: int,
) -> np.ndarray:
    """Retain recent rows evenly across outcome and benchmark-pose strata."""
    if pose_cycle <= 0 or capacity >= len(success):
        return _outcome_retention_indices(success, capacity, pos_frac)
    outcome_only = _outcome_retention_indices(success, capacity, pos_frac)
    success_mask = np.asarray(success, dtype=np.float32) > 0.5
    positive_quota = int(success_mask[outcome_only].sum())
    quotas = ((True, positive_quota), (False, len(outcome_only) - positive_quota))
    kept: list[int] = []
    episode_array = np.asarray(episode_ids, dtype=np.int64)
    for positive, quota in quotas:
        if quota <= 0:
            continue
        outcome_indices = np.flatnonzero(success_mask == positive)
        by_pose: dict[int, list[int]] = {}
        for index in outcome_indices.tolist():
            pose_id = int(episode_array[index]) % int(pose_cycle)
            by_pose.setdefault(pose_id, []).append(int(index))
        depth = 0
        selected = 0
        while selected < quota:
            candidates = [
                indices[-1 - depth]
                for indices in by_pose.values()
                if depth < len(indices)
            ]
            if not candidates:
                break
            for index in sorted(candidates, reverse=True):
                if selected >= quota:
                    break
                kept.append(index)
                selected += 1
            depth += 1
    return np.asarray(sorted(kept), dtype=np.int64)


def _expand_action_horizon(
    compact_actions: np.ndarray,
    full_action_horizon: int,
) -> np.ndarray:
    """Copy compact actions and repeat the final action through the full horizon."""
    actions = np.asarray(compact_actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be rank 2, got shape {actions.shape}")
    if actions.shape[0] > full_action_horizon:
        return actions[:full_action_horizon].copy()
    if actions.shape[0] == full_action_horizon:
        return actions
    if actions.shape[0] == 0:
        raise ValueError("cannot expand an empty action sequence")
    expanded = np.empty(
        (full_action_horizon, actions.shape[1]),
        dtype=np.float32,
    )
    expanded[: actions.shape[0]] = actions
    expanded[actions.shape[0] :] = actions[-1]
    return expanded


@dataclass
class ChunkTransition:
    z: np.ndarray  # (Z,)
    proprio: np.ndarray  # (8,)
    reference_actions: np.ndarray  # (C,8)
    executed_actions: np.ndarray  # (C,8)
    rewards: np.ndarray  # (C,)
    action_mask: np.ndarray  # (C,)
    next_z: np.ndarray  # (Z,)
    next_proprio: np.ndarray  # (8,)
    next_reference_actions: np.ndarray  # (C,8)
    terminal: bool
    mc_return: float
    success: float
    episode_id: int
    start_step: int


class ChunkReplay:
    """In-memory chunk transition buffer with success-stratified sampling."""

    def __init__(
        self,
        max_transitions: int = 50_000,
        chunk_size: int = CHUNK_SIZE,
        action_dim: int = ACTION_DIM,
        z_dim: int = Z_DIM,
        pos_frac: float = 0.4,
        seed: int = 0,
        benchmark_pose_cycle: int = 0,
    ) -> None:
        self.max_transitions = int(max_transitions)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        self.pos_frac = float(pos_frac)
        self.benchmark_pose_cycle = int(benchmark_pose_cycle)
        if self.max_transitions < 0:
            raise ValueError("max_transitions must be non-negative")
        if not np.isfinite(self.pos_frac) or not 0.0 <= self.pos_frac <= 1.0:
            raise ValueError("pos_frac must be in [0, 1]")
        if self.benchmark_pose_cycle < 0:
            raise ValueError("benchmark_pose_cycle must be non-negative")
        self.rng = np.random.default_rng(seed)
        self.rows: list[ChunkTransition] = []
        self.n_episodes = 0

    def __len__(self) -> int:
        return len(self.rows)

    def outcome_counts(self) -> tuple[int, int]:
        """Return ``(successful_rows, failed_rows)``."""
        return _outcome_counts(self.rows)

    def successful_episode_count(self) -> int:
        return _successful_episode_count(self.rows)

    def storage_nbytes(self) -> int:
        return _storage_nbytes(self.rows)

    def add(self, tr: ChunkTransition) -> None:
        if tr.reference_actions.shape != (self.chunk_size, self.action_dim):
            raise ValueError(f"bad reference shape {tr.reference_actions.shape}")
        if tr.executed_actions.shape != (self.chunk_size, self.action_dim):
            raise ValueError(f"bad executed shape {tr.executed_actions.shape}")
        self.rows.append(tr)
        indices = _pose_outcome_retention_indices(
            np.asarray([row.success for row in self.rows], dtype=np.float32),
            np.asarray([row.episode_id for row in self.rows], dtype=np.int64),
            self.max_transitions,
            self.pos_frac,
            self.benchmark_pose_cycle,
        )
        if len(indices) != len(self.rows):
            self.rows = [self.rows[int(index)] for index in indices]

    def add_episode_chunks(
        self,
        zs: list[np.ndarray],
        proprios: list[np.ndarray],
        references: list[np.ndarray],
        executed: list[np.ndarray],
        rewards: list[np.ndarray],
        masks: list[np.ndarray],
        success: bool,
        gamma: float,
        episode_id: int | None = None,
    ) -> int:
        """Add non-overlapping chunk transitions from one episode.

        Each list entry is one chunk boundary. ``zs[i]`` is the state at the
        start of chunk i; after the last chunk, we bootstrap with a zero next
        state and terminal=True.
        """
        n = len(zs)
        if n == 0:
            return 0
        if not (len(proprios) == len(references) == len(executed) == len(rewards) == len(masks) == n):
            raise ValueError("episode chunk list length mismatch")
        if episode_id is None:
            episode_id = self.n_episodes
        self.n_episodes += 1
        # Per-step sparse terminal reward already folded into rewards arrays.
        # MC return for chunk i: discounted remaining success from chunk start.
        total_steps = int(sum(int(m.sum()) for m in masks))
        added = 0
        step = 0
        for i in range(n):
            k = int(np.asarray(masks[i]).sum())
            steps_to_end = max(total_steps - step - 1, 0)
            mc = (gamma**steps_to_end) * float(success)
            if i + 1 < n:
                next_z = zs[i + 1]
                next_p = proprios[i + 1]
                next_ref = references[i + 1]
                terminal = False
            else:
                next_z = np.zeros(self.z_dim, dtype=np.float32)
                next_p = np.zeros(8, dtype=np.float32)
                next_ref = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
                terminal = True
            self.add(
                ChunkTransition(
                    z=np.asarray(zs[i], dtype=np.float32),
                    proprio=np.asarray(proprios[i], dtype=np.float32),
                    reference_actions=np.asarray(references[i], dtype=np.float32),
                    executed_actions=np.asarray(executed[i], dtype=np.float32),
                    rewards=np.asarray(rewards[i], dtype=np.float32),
                    action_mask=np.asarray(masks[i], dtype=np.float32),
                    next_z=np.asarray(next_z, dtype=np.float32),
                    next_proprio=np.asarray(next_p, dtype=np.float32),
                    next_reference_actions=np.asarray(next_ref, dtype=np.float32),
                    terminal=terminal,
                    mc_return=float(mc),
                    success=float(success),
                    episode_id=int(episode_id),
                    start_step=int(step),
                )
            )
            added += 1
            step += k
        return added

    def has_both_outcomes(self) -> bool:
        positive, negative = self.outcome_counts()
        return positive > 0 and negative > 0

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        *,
        require_both_outcomes: bool = False,
    ) -> dict[str, torch.Tensor]:
        indices = _sample_indices(
            self.rows,
            batch_size,
            self.pos_frac,
            self.rng,
            require_both_outcomes=require_both_outcomes,
        )
        batch = [self.rows[int(index)] for index in indices]
        return self._collate(batch, device)

    def sample_natural(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Sample episodes evenly, then rows within each episode."""
        indices = _episode_balanced_indices(self.rows, batch_size, self.rng)
        batch = [self.rows[int(index)] for index in indices]
        return self._collate(batch, device)

    def _collate(self, batch: list[ChunkTransition], device: torch.device | str) -> dict[str, torch.Tensor]:
        def stack(key: str, dtype=np.float32):
            return torch.as_tensor(
                np.stack([getattr(r, key) for r in batch], axis=0).astype(dtype),
                device=device,
            )

        return {
            "z": stack("z"),
            "proprio": stack("proprio"),
            "reference_actions": stack("reference_actions"),
            "executed_actions": stack("executed_actions"),
            "rewards": stack("rewards"),
            "action_mask": stack("action_mask"),
            "next_z": stack("next_z"),
            "next_proprio": stack("next_proprio"),
            "next_reference_actions": stack("next_reference_actions"),
            "terminal": torch.as_tensor(
                np.asarray([float(r.terminal) for r in batch], dtype=np.float32),
                device=device,
            ),
            "mc_return": torch.as_tensor(
                np.asarray([r.mc_return for r in batch], dtype=np.float32),
                device=device,
            ),
            "success": torch.as_tensor(
                np.asarray([r.success for r in batch], dtype=np.float32),
                device=device,
            ),
            "episode_id": torch.as_tensor(
                np.asarray([r.episode_id for r in batch], dtype=np.int64),
                device=device,
            ),
            "start_step": torch.as_tensor(
                np.asarray([r.start_step for r in batch], dtype=np.int64),
                device=device,
            ),
        }

    def save_npz(self, path: str) -> None:
        if not self.rows:
            raise RuntimeError("no transitions to save")
        payload: dict[str, Any] = {
            "z": np.stack([r.z for r in self.rows]),
            "proprio": np.stack([r.proprio for r in self.rows]),
            "reference_actions": np.stack([r.reference_actions for r in self.rows]),
            "executed_actions": np.stack([r.executed_actions for r in self.rows]),
            "rewards": np.stack([r.rewards for r in self.rows]),
            "action_mask": np.stack([r.action_mask for r in self.rows]),
            "next_z": np.stack([r.next_z for r in self.rows]),
            "next_proprio": np.stack([r.next_proprio for r in self.rows]),
            "next_reference_actions": np.stack([r.next_reference_actions for r in self.rows]),
            "terminal": np.asarray([r.terminal for r in self.rows], dtype=np.bool_),
            "mc_return": np.asarray([r.mc_return for r in self.rows], dtype=np.float32),
            "success": np.asarray([r.success for r in self.rows], dtype=np.float32),
            "episode_id": np.asarray([r.episode_id for r in self.rows], dtype=np.int32),
            "start_step": np.asarray([r.start_step for r in self.rows], dtype=np.int32),
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "z_dim": self.z_dim,
            "n_episodes": self.n_episodes,
            "pos_frac": np.asarray(self.pos_frac, dtype=np.float64),
            "benchmark_pose_cycle": np.asarray(
                self.benchmark_pose_cycle,
                dtype=np.int64,
            ),
            "rng_state_json": np.asarray(_rng_state_json(self.rng), dtype=np.str_),
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: str, **kwargs: Any) -> "ChunkReplay":
        with np.load(path, allow_pickle=False) as data:
            # Materialize each array once — NpzFile.__getitem__ re-decompresses
            # on every access, which can OOM inside the row loop.
            arrays = {
                key: data[key]
                for key in (
                    "z",
                    "proprio",
                    "reference_actions",
                    "executed_actions",
                    "rewards",
                    "action_mask",
                    "next_z",
                    "next_proprio",
                    "next_reference_actions",
                    "terminal",
                    "mc_return",
                    "success",
                    "episode_id",
                    "start_step",
                )
            }
            load_kwargs = dict(kwargs)
            if "pos_frac" not in load_kwargs and "pos_frac" in data:
                load_kwargs["pos_frac"] = float(data["pos_frac"])
            if (
                "benchmark_pose_cycle" not in load_kwargs
                and "benchmark_pose_cycle" in data
            ):
                load_kwargs["benchmark_pose_cycle"] = int(
                    data["benchmark_pose_cycle"]
                )
            buf = cls(
                chunk_size=int(data["chunk_size"]),
                action_dim=int(data["action_dim"]),
                z_dim=int(data["z_dim"]),
                **load_kwargs,
            )
            count = len(arrays["z"])
            for key, values in arrays.items():
                if len(values) != count:
                    raise ValueError(
                        f"Chunk replay field {key} has {len(values)} rows; "
                        f"expected {count}"
                    )
            if "max_transitions" not in load_kwargs:
                buf.max_transitions = max(buf.max_transitions, count)
            indices = _pose_outcome_retention_indices(
                arrays["success"],
                arrays["episode_id"],
                buf.max_transitions,
                buf.pos_frac,
                buf.benchmark_pose_cycle,
            )
            copy_rows = len(indices) < count

            def row_array(key: str, index: int) -> np.ndarray:
                value = np.asarray(arrays[key][index])
                return value.copy() if copy_rows else value

            for index_value in indices:
                index = int(index_value)
                buf.rows.append(
                    ChunkTransition(
                        z=row_array("z", index),
                        proprio=row_array("proprio", index),
                        reference_actions=row_array("reference_actions", index),
                        executed_actions=row_array("executed_actions", index),
                        rewards=row_array("rewards", index),
                        action_mask=row_array("action_mask", index),
                        next_z=row_array("next_z", index),
                        next_proprio=row_array("next_proprio", index),
                        next_reference_actions=row_array(
                            "next_reference_actions",
                            index,
                        ),
                        terminal=bool(arrays["terminal"][index]),
                        mc_return=float(arrays["mc_return"][index]),
                        success=float(arrays["success"][index]),
                        episode_id=int(arrays["episode_id"][index]),
                        start_step=int(arrays["start_step"][index]),
                    )
                )
            buf.n_episodes = (
                int(data["n_episodes"]) if "n_episodes" in data else 0
            )
            if "rng_state_json" in data:
                buf.rng = _restore_rng_state(buf.rng, data["rng_state_json"])
        return buf


class TokenReplay:
    """Compact store of raw token sequences for RL-token pretraining."""

    def __init__(self, max_seq: int = 512, token_dim: int = FEATURE_DIM) -> None:
        self.max_seq = int(max_seq)
        self.token_dim = int(token_dim)
        self.tokens: list[np.ndarray] = []
        self.masks: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.tokens)

    def add(self, tokens: np.ndarray, mask: np.ndarray | None = None) -> None:
        t = np.asarray(tokens, dtype=np.float16)
        if t.ndim != 2 or t.shape[1] != self.token_dim:
            raise ValueError(f"tokens must be (S,{self.token_dim}), got {t.shape}")
        s = min(t.shape[0], self.max_seq)
        t = t[:s]
        if mask is None:
            m = np.ones((s,), dtype=np.uint8)
        else:
            m = np.asarray(mask, dtype=np.uint8)[:s]
        self.tokens.append(t)
        self.masks.append(m)

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        n = len(self.tokens)
        if n == 0:
            raise RuntimeError("empty token replay")
        idxs = np.random.randint(0, n, size=batch_size)
        max_s = max(int(self.masks[i].shape[0]) for i in idxs)
        tok = np.zeros((batch_size, max_s, self.token_dim), dtype=np.float32)
        mask = np.zeros((batch_size, max_s), dtype=np.float32)
        for bi, i in enumerate(idxs):
            s = self.masks[i].shape[0]
            tok[bi, :s] = self.tokens[i].astype(np.float32)
            mask[bi, :s] = self.masks[i].astype(np.float32)
        return {
            "tokens": torch.as_tensor(tok, device=device),
            "attention_mask": torch.as_tensor(mask, device=device),
        }

    def save_npz(self, path: str) -> None:
        # Ragged: store object arrays of variable-length sequences.
        np.savez_compressed(
            path,
            tokens=np.array(self.tokens, dtype=object),
            masks=np.array(self.masks, dtype=object),
            token_dim=self.token_dim,
            max_seq=self.max_seq,
        )

    @classmethod
    def load_npz(cls, path: str) -> "TokenReplay":
        data = np.load(path, allow_pickle=True)
        buf = cls(max_seq=int(data["max_seq"]), token_dim=int(data["token_dim"]))
        for t, m in zip(data["tokens"], data["masks"]):
            buf.add(np.asarray(t), np.asarray(m))
        return buf


@dataclass
class ImageChunkRow:
    """Full chunk transition plus current/next Molmo AE observations."""

    z: np.ndarray
    proprio: np.ndarray
    reference_actions: np.ndarray
    executed_actions: np.ndarray
    full_reference_actions: np.ndarray
    full_executed_actions: np.ndarray
    source_native: np.ndarray
    rewards: np.ndarray
    action_mask: np.ndarray
    next_z: np.ndarray
    next_proprio: np.ndarray
    next_reference_actions: np.ndarray
    next_full_reference_actions: np.ndarray
    next_source_native: np.ndarray
    terminal: bool
    mc_return: float
    external_cam: np.ndarray  # uint8 HxWx3
    wrist_cam: np.ndarray
    instruction: str
    next_external_cam: np.ndarray
    next_wrist_cam: np.ndarray
    next_instruction: str
    success: float
    episode_id: int
    start_step: int


class ImageChunkReplay:
    """Bounded image replay for Molmo action-expert CF updates."""

    def __init__(
        self,
        max_transitions: int = 512,
        chunk_size: int = CHUNK_SIZE,
        action_dim: int = ACTION_DIM,
        z_dim: int = Z_DIM,
        pos_frac: float = 0.4,
        seed: int = 0,
        full_action_horizon: int = FULL_ACTION_HORIZON,
        padded_action_dim: int = PADDED_ACTION_DIM,
        benchmark_pose_cycle: int = 0,
    ) -> None:
        self.max_transitions = int(max_transitions)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        self.pos_frac = float(pos_frac)
        self.full_action_horizon = int(full_action_horizon)
        self.padded_action_dim = int(padded_action_dim)
        self.benchmark_pose_cycle = int(benchmark_pose_cycle)
        if self.max_transitions < 0:
            raise ValueError("max_transitions must be non-negative")
        if not np.isfinite(self.pos_frac) or not 0.0 <= self.pos_frac <= 1.0:
            raise ValueError("pos_frac must be in [0, 1]")
        if self.full_action_horizon < self.chunk_size:
            raise ValueError(
                "full_action_horizon must be at least chunk_size"
            )
        if self.padded_action_dim < self.action_dim:
            raise ValueError("padded_action_dim must be at least action_dim")
        if self.benchmark_pose_cycle < 0:
            raise ValueError("benchmark_pose_cycle must be non-negative")
        self.rng = np.random.default_rng(seed)
        self.rows: list[ImageChunkRow] = []
        self.n_episodes = 0

    def __len__(self) -> int:
        return len(self.rows)

    def outcome_counts(self) -> tuple[int, int]:
        """Return ``(successful_rows, failed_rows)``."""
        return _outcome_counts(self.rows)

    def successful_episode_count(self) -> int:
        return _successful_episode_count(self.rows)

    def storage_nbytes(self) -> int:
        return _storage_nbytes(self.rows)

    def has_both_outcomes(self) -> bool:
        positive, negative = self.outcome_counts()
        return positive > 0 and negative > 0

    def _retain_outcomes(self) -> None:
        indices = _pose_outcome_retention_indices(
            np.asarray([row.success for row in self.rows], dtype=np.float32),
            np.asarray([row.episode_id for row in self.rows], dtype=np.int64),
            self.max_transitions,
            self.pos_frac,
            self.benchmark_pose_cycle,
        )
        if len(indices) != len(self.rows):
            self.rows = [self.rows[int(index)] for index in indices]

    def add_episode(
        self,
        *,
        zs: list[np.ndarray],
        proprios: list[np.ndarray],
        references: list[np.ndarray],
        executed: list[np.ndarray],
        rewards: list[np.ndarray],
        masks: list[np.ndarray],
        external_cams: list[np.ndarray],
        wrist_cams: list[np.ndarray],
        instructions: list[str],
        success: bool,
        gamma: float,
        episode_id: int | None = None,
        full_references: list[np.ndarray] | None = None,
        full_executed: list[np.ndarray] | None = None,
        sources_native: list[np.ndarray] | None = None,
    ) -> int:
        n = len(zs)
        if n == 0:
            return 0
        if not (
            len(proprios)
            == len(references)
            == len(executed)
            == len(rewards)
            == len(masks)
            == len(external_cams)
            == len(wrist_cams)
            == len(instructions)
            == n
        ):
            raise ValueError("image episode list length mismatch")
        if full_references is not None and len(full_references) != n:
            raise ValueError("full_references length mismatch")
        if full_executed is not None and len(full_executed) != n:
            raise ValueError("full_executed length mismatch")
        if sources_native is not None and len(sources_native) != n:
            raise ValueError("sources_native length mismatch")

        z_arrays = [np.asarray(value, dtype=np.float32) for value in zs]
        proprio_arrays = [
            np.asarray(value, dtype=np.float32) for value in proprios
        ]
        reference_arrays = [
            np.asarray(value, dtype=np.float32) for value in references
        ]
        executed_arrays = [
            np.asarray(value, dtype=np.float32) for value in executed
        ]
        reward_arrays = [
            np.asarray(value, dtype=np.float32) for value in rewards
        ]
        mask_arrays = [np.asarray(value, dtype=np.float32) for value in masks]
        external_arrays = [
            np.asarray(value, dtype=np.uint8) for value in external_cams
        ]
        wrist_arrays = [
            np.asarray(value, dtype=np.uint8) for value in wrist_cams
        ]
        for name, arrays in (
            ("references", reference_arrays),
            ("executed", executed_arrays),
        ):
            for value in arrays:
                expected = (self.chunk_size, self.action_dim)
                if value.shape != expected:
                    raise ValueError(
                        f"{name} entries must have shape {expected}, "
                        f"got {value.shape}"
                    )
        if sources_native is None:
            source_arrays = [
                np.zeros(
                    (self.full_action_horizon, self.padded_action_dim),
                    dtype=np.float32,
                )
                for _ in range(n)
            ]
        else:
            source_arrays = [
                np.asarray(value, dtype=np.float32)
                for value in sources_native
            ]
        source_shape = (self.full_action_horizon, self.padded_action_dim)
        for value in source_arrays:
            if value.shape != source_shape:
                raise ValueError(
                    "sources_native entries must have shape "
                    f"{source_shape}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError("sources_native entries must be finite")
        if full_references is None:
            full_reference_arrays = [
                _expand_action_horizon(value, self.full_action_horizon)
                for value in reference_arrays
            ]
        else:
            full_reference_arrays = [
                np.asarray(value, dtype=np.float32)
                for value in full_references
            ]
        if full_executed is None:
            full_executed_arrays = [
                _expand_action_horizon(value, self.full_action_horizon)
                for value in executed_arrays
            ]
        else:
            full_executed_arrays = [
                np.asarray(value, dtype=np.float32)
                for value in full_executed
            ]
        full_shape = (self.full_action_horizon, self.action_dim)
        for name, arrays in (
            ("full_references", full_reference_arrays),
            ("full_executed", full_executed_arrays),
        ):
            for value in arrays:
                if value.shape != full_shape:
                    raise ValueError(
                        f"{name} entries must have shape {full_shape}, "
                        f"got {value.shape}"
                    )
        if episode_id is None:
            episode_id = self.n_episodes
        self.n_episodes += 1
        total_steps = int(sum(int(mask.sum()) for mask in mask_arrays))
        added = 0
        step = 0
        for i in range(n):
            steps_to_end = max(total_steps - step - 1, 0)
            mc_return = (float(gamma) ** steps_to_end) * float(success)
            if i + 1 < n:
                next_z = z_arrays[i + 1]
                next_proprio = proprio_arrays[i + 1]
                next_reference = reference_arrays[i + 1]
                next_full_reference = full_reference_arrays[i + 1]
                next_source = source_arrays[i + 1]
                next_external = external_arrays[i + 1]
                next_wrist = wrist_arrays[i + 1]
                next_instruction = instructions[i + 1]
                terminal = False
            else:
                next_z = np.zeros(self.z_dim, dtype=np.float32)
                next_proprio = np.zeros_like(proprio_arrays[i])
                next_reference = np.zeros(
                    (self.chunk_size, self.action_dim),
                    dtype=np.float32,
                )
                next_full_reference = np.zeros(
                    (self.full_action_horizon, self.action_dim),
                    dtype=np.float32,
                )
                next_source = np.zeros(
                    (self.full_action_horizon, self.padded_action_dim),
                    dtype=np.float32,
                )
                next_external = np.zeros_like(external_arrays[i])
                next_wrist = np.zeros_like(wrist_arrays[i])
                next_instruction = ""
                terminal = True
            self.rows.append(
                ImageChunkRow(
                    z=z_arrays[i],
                    proprio=proprio_arrays[i],
                    reference_actions=reference_arrays[i],
                    executed_actions=executed_arrays[i],
                    full_reference_actions=full_reference_arrays[i],
                    full_executed_actions=full_executed_arrays[i],
                    source_native=source_arrays[i],
                    rewards=reward_arrays[i],
                    action_mask=mask_arrays[i],
                    next_z=next_z,
                    next_proprio=next_proprio,
                    next_reference_actions=next_reference,
                    next_full_reference_actions=next_full_reference,
                    next_source_native=next_source,
                    terminal=terminal,
                    mc_return=float(mc_return),
                    external_cam=external_arrays[i],
                    wrist_cam=wrist_arrays[i],
                    instruction=str(instructions[i]),
                    next_external_cam=next_external,
                    next_wrist_cam=next_wrist,
                    next_instruction=str(next_instruction),
                    success=float(success),
                    episode_id=int(episode_id),
                    start_step=int(step),
                )
            )
            added += 1
            step += int(mask_arrays[i].sum())
        self._retain_outcomes()
        return added

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
        *,
        require_both_outcomes: bool = False,
    ) -> dict[str, Any]:
        indices = _sample_indices(
            self.rows,
            batch_size,
            self.pos_frac,
            self.rng,
            require_both_outcomes=require_both_outcomes,
        )
        batch = [self.rows[int(index)] for index in indices]
        return self._collate(batch, device)

    def sample_natural(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> dict[str, Any]:
        """Sample episodes evenly without stratifying their outcomes."""
        indices = _episode_balanced_indices(self.rows, batch_size, self.rng)
        batch = [self.rows[int(index)] for index in indices]
        return self._collate(batch, device)

    def _collate(
        self,
        batch: list[ImageChunkRow],
        device: torch.device | str,
    ) -> dict[str, Any]:
        def stack(key: str, dtype: Any = np.float32) -> torch.Tensor:
            return torch.as_tensor(
                np.stack([getattr(r, key) for r in batch], axis=0).astype(dtype),
                device=device,
            )

        return {
            "z": stack("z"),
            "proprio": stack("proprio"),
            "reference_actions": stack("reference_actions"),
            "executed_actions": stack("executed_actions"),
            "full_reference_actions": stack("full_reference_actions"),
            "full_executed_actions": stack("full_executed_actions"),
            "source_native": stack("source_native"),
            "rewards": stack("rewards"),
            "action_mask": stack("action_mask"),
            "next_z": stack("next_z"),
            "next_proprio": stack("next_proprio"),
            "next_reference_actions": stack("next_reference_actions"),
            "next_full_reference_actions": stack(
                "next_full_reference_actions"
            ),
            "next_source_native": stack("next_source_native"),
            "terminal": torch.as_tensor(
                np.asarray([float(r.terminal) for r in batch], dtype=np.float32),
                device=device,
            ),
            "mc_return": torch.as_tensor(
                np.asarray([r.mc_return for r in batch], dtype=np.float32),
                device=device,
            ),
            "success": torch.as_tensor(
                np.asarray([r.success for r in batch], dtype=np.float32),
                device=device,
            ),
            "episode_id": torch.as_tensor(
                np.asarray([r.episode_id for r in batch], dtype=np.int64),
                device=device,
            ),
            "start_step": torch.as_tensor(
                np.asarray([r.start_step for r in batch], dtype=np.int64),
                device=device,
            ),
            "external_cam": [r.external_cam for r in batch],
            "wrist_cam": [r.wrist_cam for r in batch],
            "instruction": [r.instruction for r in batch],
            "next_external_cam": [r.next_external_cam for r in batch],
            "next_wrist_cam": [r.next_wrist_cam for r in batch],
            "next_instruction": [r.next_instruction for r in batch],
        }

    def save_npz(self, path: str) -> None:
        """Persist the complete AE replay so watchdog resume is behaviorally exact."""
        if not self.rows:
            return

        def stack(key: str, dtype: Any | None = None) -> np.ndarray:
            values = np.stack([getattr(row, key) for row in self.rows], axis=0)
            return values.astype(dtype) if dtype is not None else values

        np.savez_compressed(
            path,
            z=stack("z", np.float32),
            proprio=stack("proprio", np.float32),
            reference_actions=stack("reference_actions", np.float32),
            executed_actions=stack("executed_actions", np.float32),
            full_reference_actions=stack("full_reference_actions", np.float32),
            full_executed_actions=stack("full_executed_actions", np.float32),
            source_native=stack("source_native", np.float32),
            rewards=stack("rewards", np.float32),
            action_mask=stack("action_mask", np.float32),
            next_z=stack("next_z", np.float32),
            next_proprio=stack("next_proprio", np.float32),
            next_reference_actions=stack("next_reference_actions", np.float32),
            next_full_reference_actions=stack(
                "next_full_reference_actions",
                np.float32,
            ),
            next_source_native=stack("next_source_native", np.float32),
            terminal=np.asarray([row.terminal for row in self.rows], dtype=np.bool_),
            mc_return=np.asarray([row.mc_return for row in self.rows], dtype=np.float32),
            external_cam=stack("external_cam", np.uint8),
            wrist_cam=stack("wrist_cam", np.uint8),
            instruction=np.asarray([row.instruction for row in self.rows], dtype=np.str_),
            next_external_cam=stack("next_external_cam", np.uint8),
            next_wrist_cam=stack("next_wrist_cam", np.uint8),
            next_instruction=np.asarray(
                [row.next_instruction for row in self.rows],
                dtype=np.str_,
            ),
            success=np.asarray([row.success for row in self.rows], dtype=np.float32),
            episode_id=np.asarray([row.episode_id for row in self.rows], dtype=np.int64),
            start_step=np.asarray([row.start_step for row in self.rows], dtype=np.int64),
            n_episodes=np.asarray(self.n_episodes, dtype=np.int64),
            chunk_size=np.asarray(self.chunk_size, dtype=np.int64),
            action_dim=np.asarray(self.action_dim, dtype=np.int64),
            z_dim=np.asarray(self.z_dim, dtype=np.int64),
            pos_frac=np.asarray(self.pos_frac, dtype=np.float64),
            full_action_horizon=np.asarray(
                self.full_action_horizon,
                dtype=np.int64,
            ),
            padded_action_dim=np.asarray(
                self.padded_action_dim,
                dtype=np.int64,
            ),
            benchmark_pose_cycle=np.asarray(
                self.benchmark_pose_cycle,
                dtype=np.int64,
            ),
            rng_state_json=np.asarray(_rng_state_json(self.rng), dtype=np.str_),
        )

    @classmethod
    def load_npz(
        cls,
        path: str,
        *,
        max_transitions: int = 512,
        pos_frac: float | None = None,
        seed: int = 0,
        benchmark_pose_cycle: int | None = None,
    ) -> "ImageChunkReplay":
        """Restore a complete AE replay without synthetic next observations."""
        with np.load(path, allow_pickle=False) as data:
            required_keys = (
                "z",
                "proprio",
                "reference_actions",
                "executed_actions",
                "rewards",
                "action_mask",
                "next_z",
                "next_proprio",
                "next_reference_actions",
                "terminal",
                "mc_return",
                "external_cam",
                "wrist_cam",
                "instruction",
                "next_external_cam",
                "next_wrist_cam",
                "next_instruction",
                "success",
                "episode_id",
                "start_step",
            )
            arrays = {key: data[key] for key in required_keys}
            for key in (
                "full_reference_actions",
                "full_executed_actions",
                "next_full_reference_actions",
                "source_native",
                "next_source_native",
            ):
                if key in data:
                    arrays[key] = data[key]
            full_action_horizon = (
                int(data["full_action_horizon"])
                if "full_action_horizon" in data
                else FULL_ACTION_HORIZON
            )
            if pos_frac is None:
                resolved_pos_frac = (
                    float(data["pos_frac"])
                    if "pos_frac" in data
                    else 0.4
                )
            else:
                resolved_pos_frac = float(pos_frac)
            replay = cls(
                max_transitions=max_transitions,
                chunk_size=int(data["chunk_size"]),
                action_dim=int(data["action_dim"]),
                z_dim=int(data["z_dim"]),
                pos_frac=resolved_pos_frac,
                seed=seed,
                full_action_horizon=full_action_horizon,
                padded_action_dim=(
                    int(data["padded_action_dim"])
                    if "padded_action_dim" in data
                    else PADDED_ACTION_DIM
                ),
                benchmark_pose_cycle=(
                    int(data["benchmark_pose_cycle"])
                    if benchmark_pose_cycle is None
                    and "benchmark_pose_cycle" in data
                    else int(benchmark_pose_cycle or 0)
                ),
            )
            count = int(arrays["z"].shape[0])
            for key, values in arrays.items():
                if int(values.shape[0]) != count:
                    raise ValueError(
                        f"Image replay field {key} has {values.shape[0]} "
                        f"rows; expected {count}"
                    )
            compact_shape = (replay.chunk_size, replay.action_dim)
            for key in (
                "reference_actions",
                "executed_actions",
                "next_reference_actions",
            ):
                if tuple(arrays[key].shape[1:]) != compact_shape:
                    raise ValueError(
                        f"Image replay field {key} has action shape "
                        f"{arrays[key].shape[1:]}; expected {compact_shape}"
                    )
            full_shape = (
                replay.full_action_horizon,
                replay.action_dim,
            )
            for key in (
                "full_reference_actions",
                "full_executed_actions",
                "next_full_reference_actions",
            ):
                if key in arrays and tuple(arrays[key].shape[1:]) != full_shape:
                    raise ValueError(
                        f"Image replay field {key} has action shape "
                        f"{arrays[key].shape[1:]}; expected {full_shape}"
                    )
            source_shape = (
                replay.full_action_horizon,
                replay.padded_action_dim,
            )
            for key in ("source_native", "next_source_native"):
                if key in arrays and tuple(arrays[key].shape[1:]) != source_shape:
                    raise ValueError(
                        f"Image replay field {key} has source shape "
                        f"{arrays[key].shape[1:]}; expected {source_shape}"
                    )

            indices = _pose_outcome_retention_indices(
                arrays["success"],
                arrays["episode_id"],
                replay.max_transitions,
                replay.pos_frac,
                replay.benchmark_pose_cycle,
            )
            copy_rows = len(indices) < count

            def row_array(
                key: str,
                index: int,
                dtype: Any,
            ) -> np.ndarray:
                value = np.asarray(arrays[key][index], dtype=dtype)
                return value.copy() if copy_rows else value

            for index_value in indices:
                index = int(index_value)
                reference_actions = row_array(
                    "reference_actions",
                    index,
                    np.float32,
                )
                executed_actions = row_array(
                    "executed_actions",
                    index,
                    np.float32,
                )
                next_reference_actions = row_array(
                    "next_reference_actions",
                    index,
                    np.float32,
                )
                full_reference_actions = (
                    row_array(
                        "full_reference_actions",
                        index,
                        np.float32,
                    )
                    if "full_reference_actions" in arrays
                    else _expand_action_horizon(
                        reference_actions,
                        replay.full_action_horizon,
                    )
                )
                full_executed_actions = (
                    row_array(
                        "full_executed_actions",
                        index,
                        np.float32,
                    )
                    if "full_executed_actions" in arrays
                    else _expand_action_horizon(
                        executed_actions,
                        replay.full_action_horizon,
                    )
                )
                next_full_reference_actions = (
                    row_array(
                        "next_full_reference_actions",
                        index,
                        np.float32,
                    )
                    if "next_full_reference_actions" in arrays
                    else _expand_action_horizon(
                        next_reference_actions,
                        replay.full_action_horizon,
                    )
                )
                source_native = (
                    row_array(
                        "source_native",
                        index,
                        np.float32,
                    )
                    if "source_native" in arrays
                    else np.zeros(
                        (
                            replay.full_action_horizon,
                            replay.padded_action_dim,
                        ),
                        dtype=np.float32,
                    )
                )
                next_source_native = (
                    row_array(
                        "next_source_native",
                        index,
                        np.float32,
                    )
                    if "next_source_native" in arrays
                    else np.zeros(
                        (
                            replay.full_action_horizon,
                            replay.padded_action_dim,
                        ),
                        dtype=np.float32,
                    )
                )
                replay.rows.append(
                    ImageChunkRow(
                        z=row_array("z", index, np.float32),
                        proprio=row_array("proprio", index, np.float32),
                        reference_actions=reference_actions,
                        executed_actions=executed_actions,
                        full_reference_actions=full_reference_actions,
                        full_executed_actions=full_executed_actions,
                        source_native=source_native,
                        rewards=row_array("rewards", index, np.float32),
                        action_mask=row_array(
                            "action_mask",
                            index,
                            np.float32,
                        ),
                        next_z=row_array("next_z", index, np.float32),
                        next_proprio=row_array(
                            "next_proprio",
                            index,
                            np.float32,
                        ),
                        next_reference_actions=next_reference_actions,
                        next_full_reference_actions=(
                            next_full_reference_actions
                        ),
                        next_source_native=next_source_native,
                        terminal=bool(arrays["terminal"][index]),
                        mc_return=float(arrays["mc_return"][index]),
                        external_cam=row_array(
                            "external_cam",
                            index,
                            np.uint8,
                        ),
                        wrist_cam=row_array("wrist_cam", index, np.uint8),
                        instruction=str(arrays["instruction"][index]),
                        next_external_cam=row_array(
                            "next_external_cam",
                            index,
                            np.uint8,
                        ),
                        next_wrist_cam=row_array(
                            "next_wrist_cam",
                            index,
                            np.uint8,
                        ),
                        next_instruction=str(
                            arrays["next_instruction"][index]
                        ),
                        success=float(arrays["success"][index]),
                        episode_id=int(arrays["episode_id"][index]),
                        start_step=int(arrays["start_step"][index]),
                    )
                )
            replay.n_episodes = (
                int(data["n_episodes"]) if "n_episodes" in data else 0
            )
            if "rng_state_json" in data:
                replay.rng = _restore_rng_state(
                    replay.rng,
                    data["rng_state_json"],
                )
        return replay
