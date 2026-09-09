"""AWR: weighted regression onto replay actions. Not unweighted BC, not V21's -Q."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rlt.awr import AWRAgent, advantage_weights
from rlt.config import OnlineConfig
from rlt.consensusflow import make_agent
from rlt.networks import Actor, Value
from rlt.replay import ChunkReplay

PICK18_KETTLE_BUFFER = (
    Path(__file__).resolve().parents[2] / "runs" / "pick18" / "kettle" / "vla_buffer.npz"
)

STATE_DIM = 6
CHUNK = 4
ACTION_DIM = 3
CHUNK_DIM = CHUNK * ACTION_DIM


def make_cfg(**overrides) -> OnlineConfig:
    return OnlineConfig(
        episode_idx=128,
        device="cpu",
        hidden_dim=32,
        batch_size=16,
        algorithm="awr",
        reference_dropout=0.0,
        **overrides,
    )


def make_awr(**overrides) -> AWRAgent:
    torch.manual_seed(0)
    return make_agent(make_cfg(**overrides), STATE_DIM, CHUNK_DIM, CHUNK)


def row_batch(n: int, *, reward: float, action_offset: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    reference = rng.normal(size=(n, CHUNK_DIM)).astype(np.float32)
    return {
        "state": rng.normal(size=(n, STATE_DIM)).astype(np.float32),
        "action": (reference + action_offset).astype(np.float32),
        "reference": reference,
        "next_state": rng.normal(size=(n, STATE_DIM)).astype(np.float32),
        "next_reference": rng.normal(size=(n, CHUNK_DIM)).astype(np.float32),
        "reward": np.full(n, reward, np.float32),
        "done": np.ones(n, np.float32),
    }


def competing_batch(n: int = 64) -> dict:
    """Same state, two actions: +1 rewarded, -1 not. AWR must prefer +1."""
    half = n // 2
    state = np.zeros((n, STATE_DIM), np.float32)
    action = np.ones((n, CHUNK_DIM), np.float32)
    action[half:] = -1.0
    reference = np.zeros((n, CHUNK_DIM), np.float32)
    reward = np.ones(n, np.float32)
    reward[half:] = 0.0
    return {
        "state": state,
        "action": action,
        "reference": reference,
        "next_state": state.copy(),
        "next_reference": reference.copy(),
        "reward": reward,
        "done": np.ones(n, np.float32),
    }


def test_make_agent_picks_awr():
    agent = make_awr()
    assert isinstance(agent, AWRAgent)
    assert agent.awr_temp == pytest.approx(1.0)
    assert agent.awr_clip == pytest.approx(20.0)


def test_equal_advantages_are_unweighted_after_normalise():
    w = advantage_weights(torch.zeros(8), temp=1.0, clip=20.0)
    assert torch.allclose(w, torch.ones(8))


def test_larger_advantage_gets_strictly_larger_weight():
    adv = torch.tensor([0.0, 0.5, 1.0])
    w = advantage_weights(adv, temp=1.0, clip=20.0)
    assert w[0] < w[1] < w[2]
    assert w.mean().item() == pytest.approx(1.0, abs=1e-5)


def test_weights_respect_the_clip_before_normalising():
    adv = torch.tensor([0.0, 10.0])
    raw = torch.exp(adv / 0.05)
    assert raw[1].item() > 20.0
    w = advantage_weights(adv, temp=0.05, clip=5.0)
    # After clip, values are {1, 5}; mean-norm → {1, 5}/3.
    assert w[1].item() == pytest.approx(5.0 / 3.0, abs=1e-5)
    assert w[0].item() == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_gaussian_log_prob_peaks_at_the_mean():
    torch.manual_seed(0)
    actor = Actor(STATE_DIM, CHUNK_DIM, hidden_dim=16, n_layers=2, sigma=0.02)
    state = torch.zeros(4, STATE_DIM)
    ref = torch.zeros(4, CHUNK_DIM)
    mean = actor(state, ref)
    at_mean = actor.log_prob(state, ref, mean)
    far = actor.log_prob(state, ref, mean + 1.0)
    assert (at_mean > far).all()
    # Closed form: log N(mean; mean, σ²I) = -0.5 * D * log(2πσ²).
    expected = -0.5 * CHUNK_DIM * math.log(2.0 * math.pi * actor.sigma**2)
    assert at_mean.mean().item() == pytest.approx(expected, abs=1e-4)


def test_bootstrap_discount_is_gamma_to_the_chunk():
    agent = make_awr(gamma=0.99)
    assert agent.chunk_discount == pytest.approx(0.99**CHUNK)


def test_terminal_transitions_train_q_and_v_to_the_reward():
    agent = make_awr()
    batch = row_batch(16, reward=1.0, action_offset=0.0)
    for _ in range(400):
        agent.critic_step(batch)
    q = agent.q_values(batch["state"][0], batch["action"][0])
    v = float(agent.value(torch.as_tensor(batch["state"][:1])).item())
    assert q == pytest.approx(1.0, abs=0.15)
    assert v == pytest.approx(1.0, abs=0.15)


def test_clipped_target_cannot_exceed_the_reachable_return():
    agent = make_awr(clip_target=True)
    batch = row_batch(16, reward=0.0, action_offset=0.0)
    batch["done"][:] = 0.0
    with torch.no_grad():
        for param in agent.critic_target.q1[-1].parameters():
            param.mul_(0.0).add_(1e6)
        for param in agent.critic_target.q2[-1].parameters():
            param.mul_(0.0).add_(1e6)
        for param in agent.value_target.net[-1].parameters():
            param.mul_(0.0).add_(1e6)
    stats = agent.critic_step(batch)
    assert stats["target_mean"] <= 1.0 + 1e-6


def test_actor_step_does_not_touch_q_or_v():
    """AWR's actor is stop-grad on the critic. V21's -Q path is the thing we are not."""
    agent = make_awr()
    batch = competing_batch()
    agent.critic_step(batch)
    q_before = [p.detach().clone() for p in agent.critic.parameters()]
    v_before = [p.detach().clone() for p in agent.value.parameters()]
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    stats = agent.actor_step(batch)
    assert "awr_w_mean" in stats and "awr_adv_mean" in stats
    assert any(not torch.equal(p, b) for p, b in zip(agent.actor.parameters(), actor_before))
    for p, b in zip(agent.critic.parameters(), q_before):
        assert torch.equal(p, b)
    for p, b in zip(agent.value.parameters(), v_before):
        assert torch.equal(p, b)


def test_actor_step_never_samples():
    """The loss is NLL of the stored chunk. Sampling would be V21's reparameterized -Q."""
    agent = make_awr()
    calls = []
    original = agent.actor.sample

    def wrapped(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    agent.actor.sample = wrapped
    agent.actor_step(competing_batch())
    assert calls == []


def test_actor_regresses_onto_the_replay_action_not_the_reference():
    agent = make_awr()
    seen: list[torch.Tensor] = []
    original = agent.actor.log_prob

    def wrapped(state, reference, action):
        seen.append(action.detach().clone())
        return original(state, reference, action)

    agent.actor.log_prob = wrapped
    batch = competing_batch()
    agent.actor_step(batch)
    assert seen, "log_prob was not called"
    assert torch.allclose(seen[0].cpu(), torch.as_tensor(batch["action"]))


def test_v21_euclidean_beta_does_not_enter_the_actor():
    """AWR's constraint is the implicit KL to replay, not Eq. 5's β‖a − ã‖²."""
    batch = competing_batch()
    stats_a = make_awr(beta=1.0).actor_step(batch)
    stats_b = make_awr(beta=1e6).actor_step(batch)
    assert stats_a["actor_loss"] == pytest.approx(stats_b["actor_loss"], rel=0, abs=1e-6)


def test_value_is_a_state_baseline_and_never_sees_the_action():
    value = Value(STATE_DIM, hidden_dim=16, n_layers=2, layer_norm=False)
    out = value(torch.zeros(5, STATE_DIM))
    assert out.shape == (5,)
    assert value.net[0].in_features == STATE_DIM


def test_pretrain_hook_is_the_awr_update():
    agent = make_awr()
    stats = agent.actor_bc_step(competing_batch())
    assert stats["actor_bc_loss"] == pytest.approx(stats["actor_loss"])
    assert "actor_bc_rmse" in stats
    assert agent.actor_steps == 1


def test_awr_clones_the_better_of_two_actions_at_the_same_state():
    """Defining property: at one s, the rewarded replay action outranks the other.

    Unweighted BC on this batch would average +1 and -1 toward 0. V21's -Q sample
    never sees the replay action. AWR has to move μ toward +1.
    """
    torch.manual_seed(0)
    agent = make_awr(lr=3e-3, awr_temp=0.5)
    batch = competing_batch(128)
    for _ in range(250):
        agent.critic_step(batch)
    q_good = agent.critic.min_q(torch.zeros(1, STATE_DIM), torch.ones(1, CHUNK_DIM)).item()
    q_bad = agent.critic.min_q(torch.zeros(1, STATE_DIM), -torch.ones(1, CHUNK_DIM)).item()
    assert q_good > q_bad + 0.2

    for _ in range(400):
        agent.actor_step(batch)
    mean = agent.actor(torch.zeros(1, STATE_DIM), torch.zeros(1, CHUNK_DIM))
    assert float(mean.mean().item()) > 0.25


def test_exploration_noise_only_when_asked():
    agent = make_awr(sigma=0.5)
    state = np.zeros(STATE_DIM, np.float32)
    reference = np.zeros(CHUNK_DIM, np.float32)
    greedy = [agent.act(state, reference, explore=False) for _ in range(3)]
    noisy = [agent.act(state, reference, explore=True) for _ in range(3)]
    assert np.allclose(greedy[0], greedy[1]) and np.allclose(greedy[1], greedy[2])
    assert not np.allclose(noisy[0], noisy[1])


def test_to_torch_leaves_token_lists_on_the_host():
    agent = make_awr()
    batch = competing_batch(8)
    batch["tokens"] = [np.zeros((8, 16), np.float16) for _ in range(8)]
    batch["mask"] = [np.ones(8, np.float16) for _ in range(8)]
    batch["next_tokens"] = [np.zeros((8, 16), np.float16) for _ in range(8)]
    batch["next_mask"] = [np.ones(8, np.float16) for _ in range(8)]
    converted = agent._to_torch(batch)
    for key in ("tokens", "mask", "next_tokens", "next_mask"):
        assert converted[key] is batch[key]
    stats = agent.critic_step(batch)
    assert "critic_loss" in stats and "v_mean" in stats


def test_save_load_roundtrip(tmp_path):
    agent = make_awr()
    batch = competing_batch()
    agent.critic_step(batch)
    agent.actor_step(batch)
    ckpt = tmp_path / "awr.pt"
    agent.save(str(ckpt))
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "awr"
    assert "value" in payload and "value_target" in payload

    fresh = make_awr()
    fresh.load(str(ckpt))
    for p, q in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(agent.critic.parameters(), fresh.critic.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(agent.value.parameters(), fresh.value.parameters()):
        assert torch.equal(p, q)
    assert fresh.critic_steps == agent.critic_steps
    assert fresh.actor_steps == agent.actor_steps


def test_refuses_nonpositive_temperature():
    with pytest.raises(ValueError, match="awr_temp"):
        make_awr(awr_temp=0.0)


def test_refuses_nonpositive_clip():
    with pytest.raises(ValueError, match="awr_clip"):
        make_awr(awr_clip=0.0)


@pytest.mark.skipif(not PICK18_KETTLE_BUFFER.is_file(), reason="pick18 kettle buffer not on disk")
def test_real_pick18_buffer_step_is_finite_and_unmocked():
    """Same 272-d state / 64-d chunk the Pick-18 sweep trains on. No synthetic stand-in."""
    data = np.load(PICK18_KETTLE_BUFFER)
    n = int(data["size"])
    state_dim = int(data["state"].shape[1])
    chunk_dim = int(data["action"].shape[1])
    assert n >= 256
    assert state_dim == 272
    assert chunk_dim == 64
    buffer = ChunkReplay(capacity=n, state_dim=state_dim, chunk_dim=chunk_dim, seed=0)
    buffer.load(str(PICK18_KETTLE_BUFFER))
    torch.manual_seed(0)
    cfg = make_cfg()
    cfg.hidden_dim = 256
    cfg.n_layers = 2
    cfg.sigma = 0.02
    agent = AWRAgent(cfg, state_dim, chunk_dim, 8)
    assert sum(p.numel() for p in agent.actor.parameters()) > 0
    assert sum(p.numel() for p in agent.value.parameters()) > 0
    batch = buffer.sample(256)
    critic = agent.critic_step(batch)
    actor = agent.actor_step(batch)
    for key in ("critic_loss", "q_mean", "v_mean", "target_mean"):
        assert math.isfinite(critic[key])
    for key in ("actor_loss", "actor_ref_rmse", "awr_w_mean", "awr_adv_mean", "actor_q"):
        assert math.isfinite(actor[key])
    assert critic["target_mean"] <= 1.0 + 1e-5
    assert actor["awr_w_mean"] == pytest.approx(1.0, abs=1e-5)
