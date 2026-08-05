"""Build stratified CF replay from MolmoSpaces FrankaPick H5 packages."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

MLSPACES_GRIPPER_MAX_POS = 0.824033
GRIPPER_ACTION_SCALE = 255.0


def decode_json_bytes(raw: Any) -> Any:
    if isinstance(raw, (bytes, bytearray, np.bytes_)):
        s = bytes(raw).decode("utf-8").rstrip("\x00")
    else:
        s = np.asarray(raw).tobytes().decode("utf-8").rstrip("\x00")
    return json.loads(s)


@dataclass
class Transition:
    state: np.ndarray  # (8,)
    action: np.ndarray  # (8,)
    reward: float
    next_state: np.ndarray
    done: float
    success: float
    mc_return: float


def _state_from_qpos(qpos: dict) -> np.ndarray:
    joint = np.asarray(qpos["arm"], dtype=np.float32).reshape(7)
    grip = np.asarray([float(qpos["gripper"][0]) / MLSPACES_GRIPPER_MAX_POS], dtype=np.float32)
    return np.concatenate([joint, grip], axis=0)


def _action_from_joint(act: dict) -> np.ndarray:
    joint = np.asarray(act["arm"], dtype=np.float32).reshape(7)
    grip = np.asarray([float(act["gripper"][0]) / GRIPPER_ACTION_SCALE], dtype=np.float32)
    return np.concatenate([joint, grip], axis=0)


def episode_terminal_success(traj_group: h5py.Group) -> bool:
    """MolmoBot pick releases are mostly success-filtered; use last-step success."""
    if "success" not in traj_group:
        return False
    succ = np.asarray(traj_group["success"][()])
    return bool(succ[-1]) if len(succ) else False


def iter_h5_trajectories(data_dir: Path) -> Iterator[tuple[Path, str]]:
    for h5_path in sorted(data_dir.rglob("*.h5")):
        with h5py.File(h5_path, "r") as f:
            keys = [k for k in f.keys() if k.startswith("traj_")]
        for key in keys:
            yield h5_path, key


def extract_episode_transitions(
    h5_path: Path,
    traj_key: str,
    gamma: float = 0.99,
) -> list[Transition] | None:
    with h5py.File(h5_path, "r") as f:
        if traj_key not in f:
            return None
        tg = f[traj_key]
        if "actions/joint_pos" not in tg or "obs/agent/qpos" not in tg:
            return None
        n_act = int(tg["actions/joint_pos"].shape[0])
        n_obs = int(tg["obs/agent/qpos"].shape[0])
        # Drop dummy first action + last sentinel (MolmoSpaces convention).
        effective = min(n_act, n_obs) - 2
        if effective < 2:
            return None
        ep_success = episode_terminal_success(tg)
        step_success = np.asarray(tg["success"][:effective], dtype=bool)
        qpos = [decode_json_bytes(tg["obs/agent/qpos"][i]) for i in range(effective + 1)]
        acts = [decode_json_bytes(tg["actions/joint_pos"][i + 1]) for i in range(effective)]

    states = [_state_from_qpos(q) for q in qpos]
    actions = [_action_from_joint(a) for a in acts]
    transitions: list[Transition] = []
    for t in range(effective):
        done = float(t == effective - 1)
        # Sparse reward only at terminal if episode succeeded.
        reward = float(ep_success) if done else 0.0
        # Per-step success flag: True once object is grasped/held (use for stratification).
        transitions.append(
            Transition(
                state=states[t],
                action=actions[t],
                reward=reward,
                next_state=states[min(t + 1, len(states) - 1)],
                done=done,
                success=float(step_success[t]),
                mc_return=0.0,
            )
        )
    # Sparse MC return from terminal episode success.
    T = len(transitions)
    for t, tr in enumerate(transitions):
        steps_to_end = T - 1 - t
        tr.mc_return = (gamma**steps_to_end) * float(ep_success)
    return transitions


def build_stratified_arrays(
    data_dir: Path,
    max_episodes: int = 200,
    pos_frac: float = 0.4,
    gamma: float = 0.99,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build replay; stratify by per-step success (post-grasp) vs pre-success.

    MolmoBot pick shards are episode-success-filtered, so D+/D- uses
    in-episode ``success[t]`` (achieved) vs early fail/pre-success frames.
    """
    rng = np.random.default_rng(seed)
    episodes: list[list[Transition]] = []

    for h5_path, traj_key in iter_h5_trajectories(data_dir):
        if len(episodes) >= max_episodes:
            break
        trs = extract_episode_transitions(h5_path, traj_key, gamma=gamma)
        if not trs:
            continue
        episodes.append(trs)

    if not episodes:
        raise RuntimeError(f"No episodes found under {data_dir}")

    pos_trs: list[Transition] = []
    neg_trs: list[Transition] = []
    for ep in episodes:
        for tr in ep:
            if tr.success > 0.5:
                pos_trs.append(tr)
            else:
                neg_trs.append(tr)

    if not pos_trs or not neg_trs:
        raise RuntimeError(
            f"Need both pre- and post-success frames; got pos={len(pos_trs)} neg={len(neg_trs)}"
        )

    # Cap total transitions while keeping pos_frac.
    max_n = max(1000, sum(len(ep) for ep in episodes))
    n_pos = max(1, int(round(max_n * pos_frac)))
    n_neg = max(1, max_n - n_pos)
    n_pos = min(n_pos, len(pos_trs))
    n_neg = min(n_neg, len(neg_trs))
    # Rebalance if one pool is smaller.
    total = n_pos + n_neg
    n_pos = max(1, int(round(total * pos_frac)))
    n_neg = total - n_pos
    n_pos = min(n_pos, len(pos_trs))
    n_neg = min(n_neg, len(neg_trs))

    chosen_pos = [pos_trs[i] for i in rng.choice(len(pos_trs), size=n_pos, replace=False)]
    chosen_neg = [neg_trs[i] for i in rng.choice(len(neg_trs), size=n_neg, replace=False)]
    chosen = chosen_pos + chosen_neg
    rng.shuffle(chosen)

    states = np.stack([tr.state for tr in chosen]).astype(np.float32)
    actions = np.stack([tr.action for tr in chosen]).astype(np.float32)
    # Normalize actions/states for MLP stability (store stats for serve).
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0).clip(min=1e-3)
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0).clip(min=1e-3)
    states_n = (states - state_mean) / state_std
    actions_n = (actions - action_mean) / action_std

    out = {
        "states": states_n,
        "actions": actions_n,
        "states_raw": states,
        "actions_raw": actions,
        "rewards": np.asarray([tr.reward for tr in chosen], dtype=np.float32),
        "next_states": np.stack(
            [((tr.next_state - state_mean) / state_std).astype(np.float32) for tr in chosen]
        ),
        "dones": np.asarray([tr.done for tr in chosen], dtype=np.float32),
        "successes": np.asarray([tr.success for tr in chosen], dtype=np.float32),
        "returns": np.asarray([tr.mc_return for tr in chosen], dtype=np.float32),
        "state_mean": state_mean.astype(np.float32),
        "state_std": state_std.astype(np.float32),
        "action_mean": action_mean.astype(np.float32),
        "action_std": action_std.astype(np.float32),
        "n_pos_eps": np.asarray([n_pos], dtype=np.int32),
        "n_neg_eps": np.asarray([n_neg], dtype=np.int32),
        "n_transitions": np.asarray([len(chosen)], dtype=np.int32),
        "n_episodes": np.asarray([len(episodes)], dtype=np.int32),
    }
    return out


def save_buffer(arrays: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_buffer(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


class StratifiedReplay(Dataset):
    """Oversamples success transitions to keep batch pos fraction ~pos_frac."""

    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        pos_frac: float = 0.4,
        seed: int = 0,
    ) -> None:
        self.arrays = arrays
        self.pos_frac = pos_frac
        self.rng = np.random.default_rng(seed)
        succ = arrays["successes"] > 0.5
        self.pos_idx = np.where(succ)[0]
        self.neg_idx = np.where(~succ)[0]
        if len(self.pos_idx) == 0:
            self.pos_idx = np.arange(len(succ))
        if len(self.neg_idx) == 0:
            self.neg_idx = np.arange(len(succ))

    def __len__(self) -> int:
        if "states" in self.arrays:
            return int(self.arrays["states"].shape[0])
        if "proprio" in self.arrays:
            return int(self.arrays["proprio"].shape[0])
        return int(self.arrays["actions"].shape[0])

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        n_pos = max(1, int(round(batch_size * self.pos_frac)))
        n_neg = batch_size - n_pos
        pos = self.rng.choice(self.pos_idx, size=n_pos, replace=True)
        neg = self.rng.choice(self.neg_idx, size=n_neg, replace=True)
        idx = np.concatenate([pos, neg])
        self.rng.shuffle(idx)
        out: dict[str, torch.Tensor] = {
            "actions": torch.from_numpy(self.arrays["actions"][idx]).float(),
            "rewards": torch.from_numpy(self.arrays["rewards"][idx]).float(),
            "dones": torch.from_numpy(self.arrays["dones"][idx]).float(),
            "successes": torch.from_numpy(self.arrays["successes"][idx]).float(),
            "returns": torch.from_numpy(self.arrays["returns"][idx]).float(),
        }
        if "base_actions" in self.arrays:
            out["base_actions"] = torch.from_numpy(self.arrays["base_actions"][idx]).float()
        if "states" in self.arrays:
            out["states"] = torch.from_numpy(self.arrays["states"][idx]).float()
        if "next_states" in self.arrays:
            out["next_states"] = torch.from_numpy(self.arrays["next_states"][idx]).float()
        if "features" in self.arrays:
            out["features"] = torch.from_numpy(self.arrays["features"][idx]).float()
        if "proprio" in self.arrays:
            out["proprio"] = torch.from_numpy(self.arrays["proprio"][idx]).float()
        elif "states_raw" in self.arrays:
            out["proprio"] = torch.from_numpy(self.arrays["states_raw"][idx]).float()
        return out

    def heldout_scores(self) -> tuple[np.ndarray, np.ndarray]:
        """Success labels + returns for AUROC gate."""
        return self.arrays["successes"].copy(), self.arrays["returns"].copy()
