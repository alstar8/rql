"""Replay buffer over action-chunk transitions, and the rollout -> transition step.

A transition is <x, a_{1:C}, a_ref, r, x'> exactly as in Algorithm 1, plus the
next reference: the target needs a' ~ pi(.|x', a_ref') and the actor cannot be
queried without one.

Rows close in step order, C env steps after the decision that opened them, so
`close_transitions` can be called during the episode -- which is what lets the
learner run between env steps as Algorithm 1 does.

Two things here break quietly and are therefore tested:

* Truncation is not termination. An episode that hits the horizon has a last
  chunk with no successor. Successor links are resolved against the full list
  of decision points, and the danglers are dropped rather than bootstrapped off
  a zero state.
* The discount inside a chunk is per env step (gamma^i), the sum stops at the
  terminal step, and the bootstrap that follows carries gamma^C.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Decision:
    """A point where the VLA ran: the RL state and the reference chunk it proposed."""

    step: int  # env step index within the episode
    state: np.ndarray  # (state_dim,) = [z_rl, proprio]
    reference: np.ndarray  # (C * action_dim,) flattened
    tokens: np.ndarray | None = None  # (S, token_dim) float16, only when the encoder learns
    mask: np.ndarray | None = None  # (S,) 1=valid
    log_prob: float | None = None  # log π of the executed chunk; PPO stores this at act() time


@dataclass
class Rollout:
    """One episode as the policy recorded it; also usable while it is still running."""

    steps: int  # env steps executed so far
    decisions: list[Decision] = field(default_factory=list)
    committed: list[np.ndarray] = field(default_factory=list)  # per-step actions the policy committed to
    rewards: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))  # (steps,)
    terminal_step: int | None = None  # first step whose outcome was success; None if never
    success: bool = False


def _transition(rollout: Rollout, index: int, chunk: int, gamma: float) -> dict | None:
    """The row opened by decisions[index], or None if it has not closed (or never will)."""
    decision = rollout.decisions[index]
    t = decision.step
    if t + chunk > len(rollout.committed):
        return None  # the window runs past what the policy ever committed to

    terminal = rollout.terminal_step is not None and rollout.terminal_step < t + chunk
    successor = None
    for later in rollout.decisions[index + 1 :]:  # decision steps increase, so this stops early
        if later.step >= t + chunk:
            successor = later if later.step == t + chunk else None
            break
    if not terminal and successor is None:
        return None  # either not there yet, or a truncated tail with no successor

    last = min(t + chunk, rollout.steps) - t  # env steps of this window that actually ran
    if terminal:
        last = min(last, rollout.terminal_step - t + 1)
    discounts = gamma ** np.arange(last, dtype=np.float32)
    reward = float(np.dot(discounts, rollout.rewards[t : t + last]))

    action = np.asarray(rollout.committed[t : t + chunk], dtype=np.float32).reshape(-1)
    row = {
        "state": decision.state,
        "action": action,
        "reference": decision.reference,
        "reward": reward,
        "next_state": np.zeros_like(decision.state) if terminal else successor.state,
        "next_reference": np.zeros_like(decision.reference) if terminal else successor.reference,
        "done": float(terminal),
    }
    if decision.log_prob is not None:
        row["log_prob"] = float(decision.log_prob)
    if decision.tokens is not None:
        row["tokens"] = decision.tokens
        row["mask"] = decision.mask
        if terminal or successor is None or successor.tokens is None:
            row["next_tokens"] = np.zeros_like(decision.tokens)
            row["next_mask"] = np.zeros_like(decision.mask) if decision.mask is not None else None
        else:
            row["next_tokens"] = successor.tokens
            row["next_mask"] = successor.mask
    return row


def close_transitions(rollout: Rollout, chunk: int, gamma: float, first: int = 0) -> tuple[list[dict], int]:
    """Rows for decisions[first:] whose window has closed, plus the new `first`.

    Stops at the first decision that has not closed. Windows close in step
    order, so this loses nothing: call it again after the next decision, and
    once more when the episode ends and the terminal step is known.
    """
    rows: list[dict] = []
    index = first
    while index < len(rollout.decisions):
        row = _transition(rollout, index, chunk, gamma)
        if row is None:
            break
        rows.append(row)
        index += 1
    return rows, index


def build_transitions(rollout: Rollout, chunk: int, gamma: float) -> list[dict]:
    """All rows of a finished episode."""
    rows, _ = close_transitions(rollout, chunk, gamma)
    return rows


class ChunkReplay:
    """Fixed-capacity uniform replay. Nothing is evicted until it is full.

    Storage goes through `_alloc` and the cursor lives in an array rather than
    in plain attributes, so `SharedChunkReplay` can put both in shared memory
    without reimplementing anything.
    """

    def __init__(self, capacity: int, state_dim: int, chunk_dim: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self._cursor = self._alloc((2,), np.int64)  # [size, next]
        self.state = self._alloc((capacity, state_dim))
        self.action = self._alloc((capacity, chunk_dim))
        self.reference = self._alloc((capacity, chunk_dim))
        self.next_state = self._alloc((capacity, state_dim))
        self.next_reference = self._alloc((capacity, chunk_dim))
        self.reward = self._alloc((capacity,))
        self.done = self._alloc((capacity,))
        # Optional VLA tokens, allocated lazily. Full prefix sequences are large, so
        # they live as a list of float16 arrays on the filled slots only.
        self.tokens: list[np.ndarray | None] | None = None
        self.masks: list[np.ndarray | None] | None = None
        self.next_tokens: list[np.ndarray | None] | None = None
        self.next_masks: list[np.ndarray | None] | None = None

    def _alloc(self, shape: tuple[int, ...], dtype=np.float32) -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    @property
    def size(self) -> int:
        return int(self._cursor[0])

    @property
    def next(self) -> int:
        return int(self._cursor[1])

    def __len__(self) -> int:
        return self.size

    def add(self, row: dict) -> None:
        i = self.next
        self.state[i] = row["state"]
        self.action[i] = row["action"]
        self.reference[i] = row["reference"]
        self.next_state[i] = row["next_state"]
        self.next_reference[i] = row["next_reference"]
        self.reward[i] = row["reward"]
        self.done[i] = row["done"]
        if "tokens" in row and row["tokens"] is not None:
            if self.tokens is None:
                self.tokens = [None] * self.capacity
                self.masks = [None] * self.capacity
                self.next_tokens = [None] * self.capacity
                self.next_masks = [None] * self.capacity
            self.tokens[i] = np.asarray(row["tokens"], dtype=np.float16)
            self.masks[i] = None if row.get("mask") is None else np.asarray(row["mask"], dtype=np.float16)
            self.next_tokens[i] = (
                None if row.get("next_tokens") is None else np.asarray(row["next_tokens"], dtype=np.float16)
            )
            self.next_masks[i] = (
                None if row.get("next_mask") is None else np.asarray(row["next_mask"], dtype=np.float16)
            )
        # The row is complete before the cursor moves, so a reader that only
        # looks below `size` never sees a half-written row.
        self._cursor[1] = (i + 1) % self.capacity
        self._cursor[0] = min(self.size + 1, self.capacity)

    def extend(self, rows: list[dict]) -> None:
        for row in rows:
            self.add(row)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=batch_size)
        batch = {
            "state": self.state[idx],
            "action": self.action[idx],
            "reference": self.reference[idx],
            "next_state": self.next_state[idx],
            "next_reference": self.next_reference[idx],
            "reward": self.reward[idx],
            "done": self.done[idx],
        }
        if self.tokens is not None:
            batch["tokens"] = [self.tokens[int(i)] for i in idx]
            batch["mask"] = [self.masks[int(i)] for i in idx]
            batch["next_tokens"] = [self.next_tokens[int(i)] for i in idx]
            batch["next_mask"] = [self.next_masks[int(i)] for i in idx]
        return batch

    def action_coverage(self) -> float:
        """RMS distance between the executed action and the VLA reference.

        Zero means the buffer holds no counterfactual actions at all -- every
        state paired only with the action the VLA would have taken. A critic
        cannot separate "good action" from "good state" on such data, so this
        is checked before training rather than after.
        """
        if self.size == 0:
            return 0.0
        diff = self.action[: self.size] - self.reference[: self.size]
        return float(np.sqrt(np.mean(diff**2)))

    def reward_rows(self) -> int:
        return int((self.reward[: self.size] > 0).sum())

    FIELDS = ("state", "action", "reference", "next_state", "next_reference", "reward", "done")

    def save(self, path: str) -> None:
        """Only the filled rows, so a resumed run does not inherit zeros.

        `size` is read once: with collectors appending during the save, reading
        it per field would write fields of different lengths.
        """
        size = self.size
        np.savez(path, size=size, **{f: getattr(self, f)[:size] for f in self.FIELDS})

    def load(self, path: str) -> None:
        data = np.load(path)
        size = int(data["size"])
        if size > self.capacity:
            raise ValueError(f"{path} holds {size} rows, capacity is {self.capacity}")
        for field in self.FIELDS:
            getattr(self, field)[:size] = data[field]
        self._cursor[0] = size
        self._cursor[1] = size % self.capacity
