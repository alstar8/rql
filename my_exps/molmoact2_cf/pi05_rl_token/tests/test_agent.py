"""Actor and critic: the discount that reaches the target, and what beta does."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlt.agent import RLTokenAgent
from rlt.config import OnlineConfig
from rlt.replay import ChunkReplay

STATE_DIM = 6
CHUNK = 4
ACTION_DIM = 3
CHUNK_DIM = CHUNK * ACTION_DIM


def make_agent(**overrides) -> RLTokenAgent:
    cfg = OnlineConfig(episode_idx=128, device="cpu", hidden_dim=32, batch_size=16, **overrides)
    torch.manual_seed(0)
    return RLTokenAgent(cfg, STATE_DIM, CHUNK_DIM, CHUNK)


def make_buffer(rows: list[dict]) -> ChunkReplay:
    buffer = ChunkReplay(len(rows), STATE_DIM, CHUNK_DIM, seed=0)
    buffer.extend(rows)
    return buffer


def row(reward: float, done: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=CHUNK_DIM).astype(np.float32)
    return {
        "state": rng.normal(size=STATE_DIM).astype(np.float32),
        "action": reference + rng.normal(scale=0.05, size=CHUNK_DIM).astype(np.float32),
        "reference": reference,
        "next_state": rng.normal(size=STATE_DIM).astype(np.float32),
        "next_reference": rng.normal(size=CHUNK_DIM).astype(np.float32),
        "reward": reward,
        "done": done,
    }


def test_bootstrap_discount_is_gamma_to_the_chunk():
    agent = make_agent(gamma=0.99)
    assert agent.chunk_discount == pytest.approx(0.99**CHUNK)


def test_terminal_transitions_train_q_to_the_reward():
    """done=1 means no bootstrap, so Q must converge to the reward itself."""
    agent = make_agent(gamma=0.99)
    buffer = make_buffer([row(1.0, 1.0, seed=i) for i in range(8)])
    for _ in range(600):
        agent.critic_step(buffer.sample(8))
    q = agent.q_values(buffer.state[0], buffer.action[0])
    assert q == pytest.approx(1.0, abs=0.1)


def test_zero_reward_terminal_keeps_q_at_zero():
    agent = make_agent(gamma=0.99)
    buffer = make_buffer([row(0.0, 1.0, seed=i) for i in range(8)])
    for _ in range(600):
        agent.critic_step(buffer.sample(8))
    assert agent.q_values(buffer.state[0], buffer.action[0]) == pytest.approx(0.0, abs=0.1)


def test_large_beta_pins_the_actor_to_the_reference():
    agent = make_agent(beta=1000.0, reference_dropout=0.0)
    buffer = make_buffer([row(0.0, 1.0, seed=i) for i in range(8)])
    before = agent.probe(buffer.sample(8))["probe/actor_deviation"]
    for _ in range(2000):
        agent.actor_step(buffer.sample(8))
    after = agent.probe(buffer.sample(8))["probe/actor_deviation"]
    assert after < before / 5


def test_reference_dropout_hits_about_half_the_batch():
    torch.manual_seed(0)
    agent = make_agent(reference_dropout=0.5)
    kept = [
        (torch.rand(4096, 1) >= agent.cfg.reference_dropout).float().mean().item() for _ in range(5)
    ]
    assert 0.45 < float(np.mean(kept)) < 0.55


def test_clipped_target_cannot_exceed_the_reachable_return():
    """A critic pushed far above 1 must still produce a target inside [0, 1]:
    the reward is a single +1 on a terminal step, so nothing else is reachable."""
    agent = make_agent(clip_target=True)
    buffer = make_buffer([row(0.0, 0.0, seed=i) for i in range(8)])
    with torch.no_grad():  # pretend the critic already blew up
        for param in agent.critic_target.q1[-1].parameters():
            param.mul_(0.0).add_(1e6)
        for param in agent.critic_target.q2[-1].parameters():
            param.mul_(0.0).add_(1e6)
    stats = agent.critic_step(buffer.sample(8))
    assert stats["target_mean"] <= 1.0 + 1e-6


def test_unclipped_target_follows_the_critic_anywhere():
    """The same batch without the clip: this is the behaviour that diverged."""
    agent = make_agent(clip_target=False)
    buffer = make_buffer([row(0.0, 0.0, seed=i) for i in range(8)])
    with torch.no_grad():
        for param in agent.critic_target.q1[-1].parameters():
            param.mul_(0.0).add_(1e6)
        for param in agent.critic_target.q2[-1].parameters():
            param.mul_(0.0).add_(1e6)
    stats = agent.critic_step(buffer.sample(8))
    assert stats["target_mean"] > 1.0


def test_to_torch_leaves_token_lists_on_the_host():
    """store_decision_tokens puts ragged prefixes on the batch. They must not go
    through torch.as_tensor: that path is the CPU stall that left GPU 0 idle."""
    agent = make_agent()
    batch = {
        "state": np.zeros((4, STATE_DIM), np.float32),
        "action": np.zeros((4, CHUNK_DIM), np.float32),
        "reference": np.zeros((4, CHUNK_DIM), np.float32),
        "next_state": np.zeros((4, STATE_DIM), np.float32),
        "next_reference": np.zeros((4, CHUNK_DIM), np.float32),
        "reward": np.zeros(4, np.float32),
        "done": np.zeros(4, np.float32),
        "tokens": [np.zeros((8, 16), np.float16) for _ in range(4)],
        "mask": [np.ones(8, np.float16) for _ in range(4)],
        "next_tokens": [np.zeros((8, 16), np.float16) for _ in range(4)],
        "next_mask": [np.ones(8, np.float16) for _ in range(4)],
    }
    converted = agent._to_torch(batch)
    for key in ("state", "action", "reference", "next_state", "next_reference", "reward", "done"):
        assert torch.is_tensor(converted[key])
        assert converted[key].device == agent.device
    for key in ("tokens", "mask", "next_tokens", "next_mask"):
        assert converted[key] is batch[key]
    stats = agent.critic_step(batch)
    assert "critic_loss" in stats


def test_exploration_noise_only_when_asked():
    agent = make_agent(sigma=0.5)
    state = np.zeros(STATE_DIM, np.float32)
    reference = np.zeros(CHUNK_DIM, np.float32)
    greedy = [agent.act(state, reference, explore=False) for _ in range(3)]
    noisy = [agent.act(state, reference, explore=True) for _ in range(3)]
    assert np.allclose(greedy[0], greedy[1]) and np.allclose(greedy[1], greedy[2])
    assert not np.allclose(noisy[0], noisy[1])
