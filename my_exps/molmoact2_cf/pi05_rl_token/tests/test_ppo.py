"""PPO: clipped on-policy surrogate on sequential chunk rows. Not V21's -Q, not AWR."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from rlt.config import OnlineConfig
from rlt.consensusflow import make_agent
from rlt.learner import Learner
from rlt.networks import Value
from rlt.ppo import PPOAgent, generalized_advantage_estimate, ppo_surrogate, ppo_value_loss
from rlt.replay import ChunkReplay, Decision, Rollout, build_transitions

PICK18_KETTLE_BUFFER = (
    Path(__file__).resolve().parents[2] / "runs" / "pick18" / "kettle" / "vla_buffer.npz"
)

STATE_DIM = 6
CHUNK = 4
ACTION_DIM = 3
CHUNK_DIM = CHUNK * ACTION_DIM


def make_cfg(**overrides) -> OnlineConfig:
    values = dict(
        episode_idx=128,
        device="cpu",
        hidden_dim=32,
        batch_size=16,
        algorithm="ppo",
        reference_dropout=0.0,
        ppo_horizon=16,
        ppo_minibatch=8,
        ppo_epochs=2,
        ppo_norm_adv=False,
    )
    values.update(overrides)
    return OnlineConfig(**values)


def make_ppo(**overrides) -> PPOAgent:
    torch.manual_seed(0)
    np.random.seed(0)
    return make_agent(make_cfg(**overrides), STATE_DIM, CHUNK_DIM, CHUNK)


def on_policy_rows(agent: PPOAgent, n: int, *, reward: float, done: float = 1.0) -> list[dict]:
    """n independent one-step trajectories, actions and log π from the live actor."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(n):
        state = rng.normal(size=STATE_DIM).astype(np.float32)
        reference = rng.normal(size=CHUNK_DIM).astype(np.float32)
        action = agent.act(state, reference, explore=True)
        assert agent.last_log_prob is not None
        rows.append(
            {
                "state": state,
                "action": np.asarray(action, dtype=np.float32),
                "reference": reference,
                "next_state": rng.normal(size=STATE_DIM).astype(np.float32),
                "next_reference": rng.normal(size=CHUNK_DIM).astype(np.float32),
                "reward": np.float32(reward),
                "done": np.float32(done),
                "log_prob": float(agent.last_log_prob),
            }
        )
    return rows


def test_make_agent_picks_ppo():
    agent = make_ppo()
    assert isinstance(agent, PPOAgent)
    assert agent.ppo_clip == pytest.approx(0.2)
    assert agent.ppo_gae_lambda == pytest.approx(0.95)
    assert agent.ppo_horizon == 16


def test_gae_terminal_reward_is_the_advantage_when_v_is_zero():
    reward = torch.tensor([1.0])
    value = torch.zeros(1)
    next_value = torch.zeros(1)
    done = torch.ones(1)
    adv, ret = generalized_advantage_estimate(reward, value, next_value, done, gamma=0.99, lam=0.95)
    assert adv.item() == pytest.approx(1.0)
    assert ret.item() == pytest.approx(1.0)


def test_gae_with_lambda_one_is_the_discounted_return_minus_v():
    reward = torch.tensor([0.0, 1.0])
    value = torch.zeros(2)
    next_value = torch.zeros(2)
    done = torch.tensor([0.0, 1.0])
    adv, ret = generalized_advantage_estimate(reward, value, next_value, done, gamma=0.9, lam=1.0)
    assert adv[1].item() == pytest.approx(1.0)
    assert adv[0].item() == pytest.approx(0.9)
    assert ret[0].item() == pytest.approx(0.9)


def test_gae_does_not_carry_across_a_done_boundary():
    """Two one-step episodes concatenated. The second episode must not leak into the first."""
    reward = torch.tensor([1.0, 0.0])
    value = torch.zeros(2)
    next_value = torch.zeros(2)
    done = torch.tensor([1.0, 1.0])
    adv, _ = generalized_advantage_estimate(reward, value, next_value, done, gamma=0.99, lam=1.0)
    assert adv[0].item() == pytest.approx(1.0)
    assert adv[1].item() == pytest.approx(0.0)


def test_gae_three_step_lambda_one_is_the_discounted_return():
    reward = torch.tensor([0.0, 0.0, 1.0])
    value = torch.zeros(3)
    next_value = torch.zeros(3)
    done = torch.tensor([0.0, 0.0, 1.0])
    adv, ret = generalized_advantage_estimate(reward, value, next_value, done, gamma=0.9, lam=1.0)
    assert adv[2].item() == pytest.approx(1.0)
    assert adv[1].item() == pytest.approx(0.9)
    assert adv[0].item() == pytest.approx(0.81)
    assert ret[0].item() == pytest.approx(0.81)


def test_gae_ignores_next_value_on_a_terminal_row():
    """A timeout bootstraps; a terminal does not, even if V(s') is huge."""
    reward = torch.tensor([1.0])
    value = torch.zeros(1)
    next_value = torch.tensor([1e6])
    done = torch.ones(1)
    adv, _ = generalized_advantage_estimate(reward, value, next_value, done, gamma=0.99, lam=0.95)
    assert adv.item() == pytest.approx(1.0)


def test_clipped_value_loss_uses_the_clip_when_v_jumps():
    value = torch.tensor([10.0])
    old = torch.tensor([0.0])
    ret = torch.tensor([1.0])
    loss = ppo_value_loss(value, old, ret, clip=0.2)
    assert loss.item() == pytest.approx(0.5 * (10.0 - 1.0) ** 2)


def test_clipped_surrogate_uses_the_clip_when_the_ratio_is_huge():
    ratio = torch.tensor([10.0])
    adv = torch.tensor([1.0])
    unclipped = -(ratio * adv).mean()
    clipped = ppo_surrogate(ratio, adv, clip=0.2)
    assert unclipped.item() == pytest.approx(-10.0)
    assert clipped.item() == pytest.approx(-1.2)


def test_surrogate_is_unclipped_when_the_ratio_is_inside_the_band():
    ratio = torch.tensor([1.05, 0.95])
    adv = torch.tensor([1.0, -1.0])
    loss = ppo_surrogate(ratio, adv, clip=0.2)
    expected = -(ratio * adv).mean()
    assert loss.item() == pytest.approx(expected.item(), abs=1e-6)


def test_bootstrap_discount_is_gamma_to_the_chunk():
    agent = make_ppo(gamma=0.99)
    assert agent.chunk_discount == pytest.approx(0.99**CHUNK)


def test_gaussian_log_prob_at_act_matches_the_actor():
    agent = make_ppo(sigma=0.02)
    state = np.zeros(STATE_DIM, np.float32)
    reference = np.zeros(CHUNK_DIM, np.float32)
    action = agent.act(state, reference, explore=True)
    s = torch.zeros(1, STATE_DIM)
    r = torch.zeros(1, CHUNK_DIM)
    a = torch.as_tensor(action).unsqueeze(0)
    expected = float(agent.actor.log_prob(s, r, a).item())
    assert agent.last_log_prob == pytest.approx(expected, abs=1e-5)


def test_entropy_is_the_closed_form_constant_of_fixed_sigma():
    """σ is not learned. An entropy bonus would have gradient 0; we log the constant instead."""
    agent = make_ppo(sigma=0.02, ppo_horizon=8, ppo_minibatch=8, ppo_epochs=1)
    stats = agent.consume_on_policy(on_policy_rows(agent, 8, reward=1.0))
    expected = 0.5 * CHUNK_DIM * (1.0 + math.log(2.0 * math.pi * 0.02**2))
    assert stats["ppo_entropy"] == pytest.approx(expected, abs=1e-5)


def test_does_not_update_before_horizon():
    agent = make_ppo(ppo_horizon=32)
    stats = agent.consume_on_policy(on_policy_rows(agent, 8, reward=1.0))
    assert stats["ppo_pending"] == pytest.approx(8.0)
    assert agent.actor_steps == 0
    assert agent.critic_steps == 0
    assert len(agent._pending) == 8


def test_actor_step_on_replay_is_refused():
    agent = make_ppo()
    with pytest.raises(RuntimeError, match="on-policy"):
        agent.actor_step({"state": np.zeros((2, STATE_DIM), np.float32)})


def test_missing_log_prob_on_an_on_policy_payload_is_refused():
    """π_old must be the collector's log π. Recomputing it here would be a fake."""
    agent = make_ppo(ppo_horizon=8)
    rows = on_policy_rows(agent, 4, reward=1.0)
    del rows[1]["log_prob"]
    with pytest.raises(RuntimeError, match="log π"):
        agent.consume_on_policy(rows)


def test_warmup_rows_without_log_prob_are_not_a_ppo_update():
    """Frozen-VLA warmup never sampled this Gaussian, so there is no π_old."""
    agent = make_ppo(ppo_horizon=8)
    rows = on_policy_rows(agent, 4, reward=1.0)
    for row in rows:
        row.pop("log_prob")
    stats = agent.consume_on_policy(rows)
    assert stats["ppo_pending"] == pytest.approx(0.0)
    assert agent.actor_steps == 0
    assert agent._pending == []


def test_pretrain_hook_is_unweighted_bc_onto_the_reference():
    agent = make_ppo()
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(16, CHUNK_DIM)).astype(np.float32)
    batch = {
        "state": rng.normal(size=(16, STATE_DIM)).astype(np.float32),
        "action": reference.copy(),
        "reference": reference,
        "next_state": rng.normal(size=(16, STATE_DIM)).astype(np.float32),
        "next_reference": reference.copy(),
        "reward": np.zeros(16, np.float32),
        "done": np.ones(16, np.float32),
    }
    before = agent.actor(torch.as_tensor(batch["state"]), torch.as_tensor(batch["reference"]))
    stats = agent.actor_bc_step(batch)
    after = agent.actor(torch.as_tensor(batch["state"]), torch.as_tensor(batch["reference"]))
    assert "actor_bc_rmse" in stats
    err_before = (before - torch.as_tensor(reference)).pow(2).mean().sqrt().item()
    err_after = (after - torch.as_tensor(reference)).pow(2).mean().sqrt().item()
    assert err_after < err_before
    assert agent.actor_steps == 1


def test_terminal_transitions_train_v_to_the_reward():
    agent = make_ppo()
    rng = np.random.default_rng(1)
    batch = {
        "state": rng.normal(size=(16, STATE_DIM)).astype(np.float32),
        "action": rng.normal(size=(16, CHUNK_DIM)).astype(np.float32),
        "reference": rng.normal(size=(16, CHUNK_DIM)).astype(np.float32),
        "next_state": rng.normal(size=(16, STATE_DIM)).astype(np.float32),
        "next_reference": rng.normal(size=(16, CHUNK_DIM)).astype(np.float32),
        "reward": np.ones(16, np.float32),
        "done": np.ones(16, np.float32),
    }
    for _ in range(400):
        agent.critic_step(batch)
    v = float(agent.value(torch.as_tensor(batch["state"][:1])).item())
    assert v == pytest.approx(1.0, abs=0.15)


def test_clipped_td_target_cannot_exceed_the_reachable_return():
    agent = make_ppo(clip_target=True)
    rng = np.random.default_rng(0)
    batch = {
        "state": rng.normal(size=(16, STATE_DIM)).astype(np.float32),
        "action": rng.normal(size=(16, CHUNK_DIM)).astype(np.float32),
        "reference": rng.normal(size=(16, CHUNK_DIM)).astype(np.float32),
        "next_state": rng.normal(size=(16, STATE_DIM)).astype(np.float32),
        "next_reference": rng.normal(size=(16, CHUNK_DIM)).astype(np.float32),
        "reward": np.zeros(16, np.float32),
        "done": np.zeros(16, np.float32),
    }
    with torch.no_grad():
        for param in agent.value_target.net[-1].parameters():
            param.mul_(0.0).add_(1e6)
    stats = agent.critic_step(batch)
    assert stats["target_mean"] <= 1.0 + 1e-6


def test_surrogate_does_not_update_v_when_vf_coef_is_zero():
    agent = make_ppo(ppo_vf_coef=0.0, ppo_horizon=8, ppo_minibatch=8, ppo_epochs=1, ppo_norm_adv=False)
    rows = on_policy_rows(agent, 8, reward=1.0)
    v_before = [p.detach().clone() for p in agent.value.parameters()]
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    agent.consume_on_policy(rows)
    assert any(not torch.equal(p, b) for p, b in zip(agent.actor.parameters(), actor_before))
    for p, b in zip(agent.value.parameters(), v_before):
        assert torch.equal(p, b)


def test_v21_euclidean_beta_does_not_enter_ppo():
    rows = on_policy_rows(make_ppo(ppo_horizon=8, ppo_minibatch=8, ppo_epochs=1), 8, reward=1.0)

    def run(beta: float) -> dict:
        agent = make_ppo(beta=beta, ppo_horizon=8, ppo_minibatch=8, ppo_epochs=1)
        torch.manual_seed(123)
        return agent.consume_on_policy(rows)

    assert run(1.0)["actor_loss"] == pytest.approx(run(1e6)["actor_loss"], rel=0, abs=1e-5)


def test_ratio_uses_stored_log_prob_not_a_recompute_after_the_actor_moves():
    """Collector lag: π_old is the log π stored at act(), not the learner's current π."""
    agent = make_ppo(ppo_horizon=8, ppo_minibatch=8, ppo_epochs=1, ppo_norm_adv=False)
    rows = on_policy_rows(agent, 8, reward=1.0)
    with torch.no_grad():
        for param in agent.actor.parameters():
            param.add_(1.5)
    stats = agent.consume_on_policy(rows)
    assert stats["ppo_ratio_mean"] < 0.5
    assert stats["ppo_clip_frac"] > 0.5


def test_ppo_shifts_mean_toward_higher_reward_on_policy_samples():
    """Defining property: on-policy samples with reward = mean(a) must raise μ."""
    torch.manual_seed(0)
    np.random.seed(0)
    agent = make_ppo(
        lr=3e-3,
        sigma=0.5,
        ppo_horizon=64,
        ppo_minibatch=64,
        ppo_epochs=8,
        ppo_norm_adv=True,
        ppo_gae_lambda=0.0,
    )
    state = np.zeros(STATE_DIM, np.float32)
    reference = np.zeros(CHUNK_DIM, np.float32)
    rows = []
    for _ in range(64):
        action = agent.act(state, reference, explore=True)
        rows.append(
            {
                "state": state.copy(),
                "action": np.asarray(action, dtype=np.float32),
                "reference": reference.copy(),
                "next_state": state.copy(),
                "next_reference": reference.copy(),
                "reward": float(np.mean(action)),
                "done": np.float32(1.0),
                "log_prob": float(agent.last_log_prob),
            }
        )
    mean_before = float(agent.actor(torch.zeros(1, STATE_DIM), torch.zeros(1, CHUNK_DIM)).mean().item())
    agent.consume_on_policy(rows)
    mean_after = float(agent.actor(torch.zeros(1, STATE_DIM), torch.zeros(1, CHUNK_DIM)).mean().item())
    assert mean_after > mean_before + 0.02


def test_value_is_a_state_baseline_and_never_sees_the_action():
    value = Value(STATE_DIM, hidden_dim=16, n_layers=2, layer_norm=False)
    out = value(torch.zeros(5, STATE_DIM))
    assert out.shape == (5,)
    assert value.net[0].in_features == STATE_DIM


def test_exploration_noise_only_when_asked():
    agent = make_ppo(sigma=0.5)
    state = np.zeros(STATE_DIM, np.float32)
    reference = np.zeros(CHUNK_DIM, np.float32)
    greedy = [agent.act(state, reference, explore=False) for _ in range(3)]
    noisy = [agent.act(state, reference, explore=True) for _ in range(3)]
    assert np.allclose(greedy[0], greedy[1]) and np.allclose(greedy[1], greedy[2])
    assert not np.allclose(noisy[0], noisy[1])


def test_to_torch_leaves_token_lists_on_the_host():
    agent = make_ppo()
    rng = np.random.default_rng(0)
    batch = {
        "state": rng.normal(size=(8, STATE_DIM)).astype(np.float32),
        "action": rng.normal(size=(8, CHUNK_DIM)).astype(np.float32),
        "reference": rng.normal(size=(8, CHUNK_DIM)).astype(np.float32),
        "next_state": rng.normal(size=(8, STATE_DIM)).astype(np.float32),
        "next_reference": rng.normal(size=(8, CHUNK_DIM)).astype(np.float32),
        "reward": np.ones(8, np.float32),
        "done": np.ones(8, np.float32),
        "tokens": [np.zeros((8, 16), np.float16) for _ in range(8)],
        "mask": [np.ones(8, np.float16) for _ in range(8)],
        "next_tokens": [np.zeros((8, 16), np.float16) for _ in range(8)],
        "next_mask": [np.ones(8, np.float16) for _ in range(8)],
    }
    converted = agent._to_torch(batch)
    for key in ("tokens", "mask", "next_tokens", "next_mask"):
        assert converted[key] is batch[key]
    stats = agent.critic_step(batch)
    assert "critic_loss" in stats and "v_mean" in stats


def test_save_load_roundtrip(tmp_path):
    agent = make_ppo(ppo_horizon=8, ppo_minibatch=8, ppo_epochs=1)
    agent.consume_on_policy(on_policy_rows(agent, 8, reward=1.0))
    ckpt = tmp_path / "ppo.pt"
    agent.save(str(ckpt))
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload["algorithm"] == "ppo"
    assert "value" in payload and "value_target" in payload
    assert "critic" not in payload

    fresh = make_ppo(ppo_horizon=8, ppo_minibatch=8, ppo_epochs=1)
    fresh.load(str(ckpt))
    for p, q in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(agent.value.parameters(), fresh.value.parameters()):
        assert torch.equal(p, q)
    assert fresh.critic_steps == agent.critic_steps
    assert fresh.actor_steps == agent.actor_steps
    assert fresh._pending == []


def test_refuses_nonpositive_clip():
    with pytest.raises(ValueError, match="ppo_clip"):
        make_ppo(ppo_clip=0.0)


def test_refuses_horizon_below_one():
    with pytest.raises(ValueError, match="ppo_horizon"):
        make_ppo(ppo_horizon=0)


def test_log_prob_on_a_decision_lands_on_the_closed_row():
    committed = [np.full(ACTION_DIM, float(t), np.float32) for t in range(16)]
    decisions = [
        Decision(
            step=t,
            state=np.full(STATE_DIM, float(t), np.float32),
            reference=np.full(CHUNK_DIM, -float(t), np.float32),
            log_prob=-0.5 * t,
        )
        for t in range(0, 16, CHUNK)
    ]
    rewards = np.zeros(16, np.float32)
    rewards[7] = 1.0
    rollout = Rollout(
        steps=16,
        decisions=decisions,
        committed=committed,
        rewards=rewards,
        terminal_step=7,
        success=True,
    )
    rows = build_transitions(rollout, CHUNK, gamma=0.99)
    assert rows
    assert rows[0]["log_prob"] == pytest.approx(0.0)
    assert rows[1]["log_prob"] == pytest.approx(-2.0)


def test_learner_does_not_run_off_policy_utd_for_ppo():
    cfg = make_cfg(utd=5, critic_updates_per_actor=2, ppo_horizon=16, ppo_minibatch=8, ppo_epochs=2)
    torch.manual_seed(0)
    agent = PPOAgent(cfg, STATE_DIM, CHUNK_DIM, CHUNK)
    buffer = ChunkReplay(1024, STATE_DIM, CHUNK_DIM, seed=0)
    learner = Learner(cfg, agent, buffer, CHUNK)
    rows = on_policy_rows(agent, 16, reward=1.0)
    learner.absorb(rows)
    assert len(buffer) == 16
    # 2 epochs * (16/8) minibatches, not utd * 16 * critic_updates_per_actor
    assert agent.actor_steps == 2 * (16 // 8)
    assert agent.critic_steps == 2 * (16 // 8)


class _RolloutView:
    def __init__(self, rollout: Rollout) -> None:
        self._rollout = rollout

    def rollout_view(self) -> Rollout:
        return self._rollout


def _contiguous_success_rollout() -> Rollout:
    """Four chunk decisions, success on the last env step. log π is a dummy scalar."""
    steps = 16
    committed = [np.full(ACTION_DIM, float(t), np.float32) for t in range(steps)]
    decisions = [
        Decision(
            step=t,
            state=np.full(STATE_DIM, float(t), np.float32),
            reference=np.full(CHUNK_DIM, -float(t), np.float32),
            log_prob=0.0,
        )
        for t in range(0, steps, CHUNK)
    ]
    rewards = np.zeros(steps, np.float32)
    rewards[steps - 1] = 1.0
    return Rollout(
        steps=steps,
        decisions=decisions,
        committed=committed,
        rewards=rewards,
        terminal_step=steps - 1,
        success=True,
    )


def test_one_row_gae_is_not_the_episode_return():
    """Why the learner waits: GAE on each mid-episode close is TD(0), not GAE(λ=1)."""
    agent = make_ppo(ppo_horizon=64, ppo_gae_lambda=1.0, ppo_norm_adv=False)
    with torch.no_grad():
        for param in agent.value.parameters():
            param.zero_()
    rows = build_transitions(_contiguous_success_rollout(), CHUNK, gamma=0.99)
    assert len(rows) >= 2
    full = agent._gae_segment(rows)
    pieces = []
    for row in rows:
        pieces.extend(agent._gae_segment([row]))
    assert pieces[-1]["adv"] == pytest.approx(full[-1]["adv"], abs=1e-5)
    assert abs(pieces[0]["adv"] - full[0]["adv"]) > 0.05


def test_learner_computes_gae_on_the_finished_episode_not_on_mid_episode_closes():
    cfg = make_cfg(ppo_horizon=64, ppo_gae_lambda=1.0, ppo_norm_adv=False, gamma=0.99)
    torch.manual_seed(0)
    agent = PPOAgent(cfg, STATE_DIM, CHUNK_DIM, CHUNK)
    buffer = ChunkReplay(1024, STATE_DIM, CHUNK_DIM, seed=0)
    learner = Learner(cfg, agent, buffer, CHUNK)
    finished = _contiguous_success_rollout()
    one_shot = build_transitions(finished, CHUNK, gamma=0.99)

    mid = Rollout(
        steps=2 * CHUNK,
        decisions=finished.decisions[:2],
        committed=finished.committed[: 2 * CHUNK],
        rewards=np.zeros(2 * CHUNK, np.float32),
    )
    learner.start_episode()
    learner.on_decision(_RolloutView(mid))
    assert agent.actor_steps == 0
    assert agent._pending == []
    assert len(learner._on_policy_episode) == 1

    learner.finish_episode(finished)
    assert len(buffer) == len(one_shot)
    assert len(agent._pending) == len(one_shot)
    expected = agent._gae_segment(one_shot)
    for got, want in zip(agent._pending, expected):
        assert got["adv"] == pytest.approx(want["adv"], abs=1e-5)
        assert got["log_prob"] == pytest.approx(want["log_prob"], abs=1e-5)


@pytest.mark.skipif(not PICK18_KETTLE_BUFFER.is_file(), reason="pick18 kettle buffer not on disk")
def test_real_pick18_pretrain_and_on_policy_step_are_finite_and_unmocked():
    """Same 272-d state / 64-d chunk the Pick-18 sweep trains on. No synthetic stand-in.

    Pretrain is BC + TD on the frozen-VLA npz. Online PPO is consume_on_policy on
    actions the Gaussian actually sampled at those real states, with the stored
    log π from act().
    """
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
    cfg.ppo_horizon = 64
    cfg.ppo_minibatch = 32
    cfg.ppo_epochs = 2
    agent = PPOAgent(cfg, state_dim, chunk_dim, 8)
    assert sum(p.numel() for p in agent.actor.parameters()) > 0
    assert sum(p.numel() for p in agent.value.parameters()) > 0

    batch = buffer.sample(256)
    critic = agent.critic_step(batch)
    bc = agent.actor_bc_step(batch)
    for key in ("critic_loss", "q_mean", "v_mean", "target_mean"):
        assert math.isfinite(critic[key])
    for key in ("actor_bc_loss", "actor_bc_rmse"):
        assert math.isfinite(bc[key])
    assert critic["target_mean"] <= 1.0 + 1e-5

    rows = []
    for i in range(64):
        state = buffer.state[i]
        reference = buffer.reference[i]
        action = agent.act(state, reference, explore=True)
        rows.append(
            {
                "state": state,
                "action": np.asarray(action, dtype=np.float32),
                "reference": reference,
                "next_state": buffer.next_state[i],
                "next_reference": buffer.next_reference[i],
                "reward": float(buffer.reward[i]),
                "done": float(buffer.done[i]),
                "log_prob": float(agent.last_log_prob),
            }
        )
    ppo = agent.consume_on_policy(rows)
    for key in ("actor_loss", "critic_loss", "v_mean", "actor_ref_rmse", "ppo_ratio_mean", "ppo_clip_frac"):
        assert math.isfinite(ppo[key])
    assert ppo["ppo_batch"] == pytest.approx(64.0)
