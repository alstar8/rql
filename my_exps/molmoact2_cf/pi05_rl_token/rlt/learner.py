"""The inner loop of Algorithm 1: store what closed, then run G updates.

Called at every decision point during an episode, so the actor improves while
the episode is still running -- the synchronous equivalent of the paper's
asynchronous learner. One iteration is two critic updates and one actor update;
`utd` iterations are run per transition that entered the buffer.
"""

from __future__ import annotations

from .agent import RLTokenAgent
from .config import OnlineConfig
from .replay import ChunkReplay, Rollout, close_transitions


class Learner:
    def __init__(
        self,
        cfg: OnlineConfig,
        agent: RLTokenAgent,
        buffer: ChunkReplay,
        chunk: int,
        updates: bool = True,
    ) -> None:
        self.cfg = cfg
        self.agent = agent
        self.buffer = buffer
        self.chunk = chunk
        # With several collectors the updates happen in the learner process, and
        # a collector only stores what it closed.
        self.updates = updates
        self.first = 0  # decisions[:first] have already become rows
        self.stats: dict[str, float] = {}
        self.pending_env_steps = 0
        # PPO GAE needs one contiguous episode. Mid-episode closes are buffered here
        # and consumed in finish_episode. Parallel training never uses this: the parent
        # calls absorb() with a finished-episode payload.
        self._on_policy_episode: list[dict] = []

    def _on_policy(self) -> bool:
        return getattr(self.agent, "consume_on_policy", None) is not None

    def start_episode(self) -> None:
        self.first = 0
        self._on_policy_episode = []

    def on_decision(self, policy) -> int:
        """Hook for RLTokenPolicy: absorb whatever the new decision closed."""
        rows, self.first = close_transitions(policy.rollout_view(), self.chunk, self.cfg.gamma, self.first)
        if self._on_policy():
            # Store, but do not GAE yet: a one-row close would turn GAE(λ) into TD(0).
            self.buffer.extend(rows)
            self._on_policy_episode.extend(rows)
            return len(rows)
        return self.absorb(rows)

    def finish_episode(self, rollout: Rollout) -> int:
        """Close the remaining rows now that the terminal step is known."""
        rows, self.first = close_transitions(rollout, self.chunk, self.cfg.gamma, self.first)
        if self._on_policy():
            self.buffer.extend(rows)
            self._on_policy_episode.extend(rows)
            n = len(rows)
            if self.updates and self._on_policy_episode:
                consume = self.agent.consume_on_policy
                self.stats.update(consume(self._on_policy_episode))
            self._on_policy_episode = []
            return n
        return self.absorb(rows)

    def absorb(self, rows: list[dict]) -> int:
        self.buffer.extend(rows)
        if rows and self.updates:
            consume = getattr(self.agent, "consume_on_policy", None)
            if consume is not None:
                # PPO: one contiguous trajectory per payload. UTD-on-replay does not apply.
                self.stats.update(consume(rows))
                return len(rows)
            cadence = int(getattr(self.cfg, "update_every_steps", 0) or 0)
            if cadence <= 0:
                self.update(len(rows))
            else:
                self.pending_env_steps += len(rows) * self.chunk
                if self.pending_env_steps >= cadence:
                    n_rows = max(1, self.pending_env_steps // self.chunk)
                    self.update(n_rows)
                    self.pending_env_steps = 0
        return len(rows)

    def update(self, n_rows: int) -> None:
        if len(self.buffer) < self.cfg.batch_size:
            return  # a batch is not available yet; the data is not lost, only unused
        for _ in range(self.cfg.utd * n_rows):
            for _ in range(self.cfg.critic_updates_per_actor):
                self.stats.update(self.agent.critic_step(self.buffer.sample(self.cfg.batch_size)))
            self.stats.update(self.agent.actor_step(self.buffer.sample(self.cfg.batch_size)))
