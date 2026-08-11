"""Chunk-aligned replay for RLT-style MolmoAct2 CF training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from rlt_models import ACTION_DIM, CHUNK_SIZE, FEATURE_DIM, Z_DIM


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
    ) -> None:
        self.max_transitions = int(max_transitions)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        self.pos_frac = float(pos_frac)
        self.rng = np.random.default_rng(seed)
        self.rows: list[ChunkTransition] = []
        self.n_episodes = 0

    def __len__(self) -> int:
        return len(self.rows)

    def add(self, tr: ChunkTransition) -> None:
        if tr.reference_actions.shape != (self.chunk_size, self.action_dim):
            raise ValueError(f"bad reference shape {tr.reference_actions.shape}")
        if tr.executed_actions.shape != (self.chunk_size, self.action_dim):
            raise ValueError(f"bad executed shape {tr.executed_actions.shape}")
        self.rows.append(tr)
        overflow = len(self.rows) - self.max_transitions
        if overflow > 0:
            self.rows = self.rows[overflow:]

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
        if not self.rows:
            return False
        succ = np.asarray([r.success for r in self.rows], dtype=np.float32)
        return bool(np.any(succ > 0.5) and np.any(succ <= 0.5))

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        n = len(self.rows)
        if n == 0:
            raise RuntimeError("empty chunk replay")
        succ = np.asarray([r.success for r in self.rows], dtype=np.float32)
        pos = np.flatnonzero(succ > 0.5)
        neg = np.flatnonzero(succ <= 0.5)
        n_pos = int(round(batch_size * self.pos_frac)) if len(pos) and len(neg) else 0
        n_pos = min(n_pos, batch_size, len(pos)) if len(pos) else 0
        n_neg = batch_size - n_pos
        idxs: list[int] = []
        if n_pos:
            idxs.extend(self.rng.choice(pos, size=n_pos, replace=len(pos) < n_pos).tolist())
        if n_neg:
            pool = neg if len(neg) else np.arange(n)
            idxs.extend(self.rng.choice(pool, size=n_neg, replace=len(pool) < n_neg).tolist())
        batch = [self.rows[i] for i in idxs]
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
        }
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: str, **kwargs: Any) -> "ChunkReplay":
        data = np.load(path, allow_pickle=False)
        # Materialize each array once — NpzFile.__getitem__ re-decompresses on
        # every access, which OOMs for large buffers if done inside the row loop.
        z = data["z"]
        proprio = data["proprio"]
        reference_actions = data["reference_actions"]
        executed_actions = data["executed_actions"]
        rewards = data["rewards"]
        action_mask = data["action_mask"]
        next_z = data["next_z"]
        next_proprio = data["next_proprio"]
        next_reference_actions = data["next_reference_actions"]
        terminal = data["terminal"]
        mc_return = data["mc_return"]
        success = data["success"]
        episode_id = data["episode_id"]
        start_step = data["start_step"]
        buf = cls(
            chunk_size=int(data["chunk_size"]),
            action_dim=int(data["action_dim"]),
            z_dim=int(data["z_dim"]),
            **kwargs,
        )
        n = len(z)
        if "max_transitions" not in kwargs:
            buf.max_transitions = max(buf.max_transitions, n)
        for i in range(n):
            buf.add(
                ChunkTransition(
                    z=z[i],
                    proprio=proprio[i],
                    reference_actions=reference_actions[i],
                    executed_actions=executed_actions[i],
                    rewards=rewards[i],
                    action_mask=action_mask[i],
                    next_z=next_z[i],
                    next_proprio=next_proprio[i],
                    next_reference_actions=next_reference_actions[i],
                    terminal=bool(terminal[i]),
                    mc_return=float(mc_return[i]),
                    success=float(success[i]),
                    episode_id=int(episode_id[i]),
                    start_step=int(start_step[i]),
                )
            )
        buf.n_episodes = int(data["n_episodes"]) if "n_episodes" in data else 0
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
    rewards: np.ndarray
    action_mask: np.ndarray
    next_z: np.ndarray
    next_proprio: np.ndarray
    next_reference_actions: np.ndarray
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
    """Small image-augmented buffer for AE LoRA CF updates (not disk-persisted)."""

    def __init__(
        self,
        max_transitions: int = 512,
        chunk_size: int = CHUNK_SIZE,
        action_dim: int = ACTION_DIM,
        z_dim: int = Z_DIM,
        pos_frac: float = 0.4,
        seed: int = 0,
    ) -> None:
        self.max_transitions = int(max_transitions)
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.z_dim = int(z_dim)
        self.pos_frac = float(pos_frac)
        self.rng = np.random.default_rng(seed)
        self.rows: list[ImageChunkRow] = []
        self.n_episodes = 0

    def __len__(self) -> int:
        return len(self.rows)

    def has_both_outcomes(self) -> bool:
        if not self.rows:
            return False
        success = np.asarray([row.success for row in self.rows], dtype=np.float32)
        return bool(np.any(success > 0.5) and np.any(success <= 0.5))

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
        if episode_id is None:
            episode_id = self.n_episodes
        self.n_episodes += 1
        total_steps = int(sum(int(np.asarray(mask).sum()) for mask in masks))
        added = 0
        step = 0
        for i in range(n):
            steps_to_end = max(total_steps - step - 1, 0)
            mc_return = (float(gamma) ** steps_to_end) * float(success)
            if i + 1 < n:
                next_z = zs[i + 1]
                next_proprio = proprios[i + 1]
                next_reference = references[i + 1]
                next_external = external_cams[i + 1]
                next_wrist = wrist_cams[i + 1]
                next_instruction = instructions[i + 1]
                terminal = False
            else:
                next_z = np.zeros(self.z_dim, dtype=np.float32)
                next_proprio = np.zeros_like(
                    np.asarray(proprios[i], dtype=np.float32)
                )
                next_reference = np.zeros(
                    (self.chunk_size, self.action_dim),
                    dtype=np.float32,
                )
                next_external = np.zeros_like(
                    np.asarray(external_cams[i], dtype=np.uint8)
                )
                next_wrist = np.zeros_like(
                    np.asarray(wrist_cams[i], dtype=np.uint8)
                )
                next_instruction = ""
                terminal = True
            self.rows.append(
                ImageChunkRow(
                    z=np.asarray(zs[i], dtype=np.float32),
                    proprio=np.asarray(proprios[i], dtype=np.float32),
                    reference_actions=np.asarray(references[i], dtype=np.float32),
                    executed_actions=np.asarray(executed[i], dtype=np.float32),
                    rewards=np.asarray(rewards[i], dtype=np.float32),
                    action_mask=np.asarray(masks[i], dtype=np.float32),
                    next_z=np.asarray(next_z, dtype=np.float32),
                    next_proprio=np.asarray(next_proprio, dtype=np.float32),
                    next_reference_actions=np.asarray(
                        next_reference,
                        dtype=np.float32,
                    ),
                    terminal=terminal,
                    mc_return=float(mc_return),
                    external_cam=np.asarray(external_cams[i], dtype=np.uint8),
                    wrist_cam=np.asarray(wrist_cams[i], dtype=np.uint8),
                    instruction=str(instructions[i]),
                    next_external_cam=np.asarray(next_external, dtype=np.uint8),
                    next_wrist_cam=np.asarray(next_wrist, dtype=np.uint8),
                    next_instruction=str(next_instruction),
                    success=float(success),
                    episode_id=int(episode_id),
                    start_step=int(step),
                )
            )
            added += 1
            step += int(np.asarray(masks[i]).sum())
        overflow = len(self.rows) - self.max_transitions
        if overflow > 0:
            self.rows = self.rows[overflow:]
        return added

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> dict[str, Any]:
        n = len(self.rows)
        if n == 0:
            raise RuntimeError("empty image chunk replay")
        succ = np.asarray([r.success for r in self.rows], dtype=np.float32)
        pos = np.flatnonzero(succ > 0.5)
        neg = np.flatnonzero(succ <= 0.5)
        n_pos = int(round(batch_size * self.pos_frac)) if len(pos) and len(neg) else 0
        n_pos = min(n_pos, batch_size, len(pos)) if len(pos) else 0
        n_neg = batch_size - n_pos
        idxs: list[int] = []
        if n_pos:
            idxs.extend(self.rng.choice(pos, size=n_pos, replace=len(pos) < n_pos).tolist())
        if n_neg:
            pool = neg if len(neg) else np.arange(n)
            idxs.extend(self.rng.choice(pool, size=n_neg, replace=len(pool) < n_neg).tolist())
        batch = [self.rows[i] for i in idxs]

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
            rewards=stack("rewards", np.float32),
            action_mask=stack("action_mask", np.float32),
            next_z=stack("next_z", np.float32),
            next_proprio=stack("next_proprio", np.float32),
            next_reference_actions=stack("next_reference_actions", np.float32),
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
        )

    @classmethod
    def load_npz(
        cls,
        path: str,
        *,
        max_transitions: int = 512,
        pos_frac: float = 0.4,
        seed: int = 0,
    ) -> "ImageChunkReplay":
        """Restore a complete AE replay without synthetic next observations."""
        with np.load(path, allow_pickle=False) as data:
            replay = cls(
                max_transitions=max_transitions,
                chunk_size=int(data["chunk_size"]),
                action_dim=int(data["action_dim"]),
                z_dim=int(data["z_dim"]),
                pos_frac=pos_frac,
                seed=seed,
            )
            count = int(data["z"].shape[0])
            keys = (
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
            for key in keys:
                if int(data[key].shape[0]) != count:
                    raise ValueError(
                        f"Image replay field {key} has {data[key].shape[0]} "
                        f"rows; expected {count}"
                    )
            start = max(0, count - replay.max_transitions)
            for index in range(start, count):
                replay.rows.append(
                    ImageChunkRow(
                        z=np.asarray(data["z"][index], dtype=np.float32),
                        proprio=np.asarray(data["proprio"][index], dtype=np.float32),
                        reference_actions=np.asarray(
                            data["reference_actions"][index],
                            dtype=np.float32,
                        ),
                        executed_actions=np.asarray(
                            data["executed_actions"][index],
                            dtype=np.float32,
                        ),
                        rewards=np.asarray(data["rewards"][index], dtype=np.float32),
                        action_mask=np.asarray(
                            data["action_mask"][index],
                            dtype=np.float32,
                        ),
                        next_z=np.asarray(data["next_z"][index], dtype=np.float32),
                        next_proprio=np.asarray(
                            data["next_proprio"][index],
                            dtype=np.float32,
                        ),
                        next_reference_actions=np.asarray(
                            data["next_reference_actions"][index],
                            dtype=np.float32,
                        ),
                        terminal=bool(data["terminal"][index]),
                        mc_return=float(data["mc_return"][index]),
                        external_cam=np.asarray(
                            data["external_cam"][index],
                            dtype=np.uint8,
                        ),
                        wrist_cam=np.asarray(data["wrist_cam"][index], dtype=np.uint8),
                        instruction=str(data["instruction"][index]),
                        next_external_cam=np.asarray(
                            data["next_external_cam"][index],
                            dtype=np.uint8,
                        ),
                        next_wrist_cam=np.asarray(
                            data["next_wrist_cam"][index],
                            dtype=np.uint8,
                        ),
                        next_instruction=str(data["next_instruction"][index]),
                        success=float(data["success"][index]),
                        episode_id=int(data["episode_id"][index]),
                        start_step=int(data["start_step"][index]),
                    )
                )
            replay.n_episodes = int(data["n_episodes"])
        return replay
