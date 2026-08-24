"""flow_rlt (corrected V22): flow actor + frozen RL-Token critic."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rlt.agent import RLTokenAgent
from rlt.config import OnlineConfig
from rlt.consensusflow import make_agent
from rlt.flow_rlt import FlowRLTAgent
from rlt.token_ae import RLTokenAE

STATE_DIM = 8
CHUNK = 4
CHUNK_DIM = CHUNK * 2
Z_DIM = 4


def make_cfg(rlt_critic: str = "", **overrides) -> OnlineConfig:
    return OnlineConfig(
        episode_idx=128,
        device="cpu",
        batch_size=4,
        hidden_dim=16,
        algorithm="flow_rlt",
        rlt_critic=rlt_critic,
        cf_hidden_dim=16,
        cf_n_layers=2,
        cf_flow_steps=2,
        token_batch_size=2,
        z_dim=Z_DIM,
        **overrides,
    )


def make_rlt_critic(tmp_path) -> str:
    cfg = OnlineConfig(episode_idx=128, device="cpu", hidden_dim=16, n_layers=2)
    agent = RLTokenAgent(cfg, STATE_DIM, CHUNK_DIM, CHUNK)
    path = tmp_path / "rlt_agent.pt"
    agent.save(str(path))
    return str(path)


def make_batch(n: int = 32, tokens: bool = False) -> dict:
    rng = np.random.default_rng(0)
    state = rng.normal(size=(n, STATE_DIM)).astype(np.float32)
    # VLA-scale reference (std ~0.2) that depends on the state, so BC can fit it.
    reference = (0.2 * np.tanh(state[:, :1] * rng.normal(size=(1, CHUNK_DIM)))).astype(np.float32)
    batch = {
        "state": state,
        "action": reference.copy(),
        "reference": reference,
        "next_state": rng.normal(size=(n, STATE_DIM)).astype(np.float32),
        "next_reference": reference.copy(),
        "reward": rng.random(n).astype(np.float32),
        "done": np.zeros(n, np.float32),
    }
    if tokens:
        s, d = 6, 8
        batch["tokens"] = [rng.normal(size=(s, d)).astype(np.float16) for _ in range(n)]
        batch["mask"] = [np.ones(s, np.float16) for _ in range(n)]
    return batch


def test_make_agent_picks_flow_rlt(tmp_path):
    cfg = make_cfg(make_rlt_critic(tmp_path))
    agent = make_agent(cfg, STATE_DIM, CHUNK_DIM, CHUNK)
    assert isinstance(agent, FlowRLTAgent)


def test_compose_guidance_copies_reference():
    """CF composition: V = (ref - noise) + G. BC drives G -> 0, so the policy becomes
    an exact reference copy; the actor net is unconditioned (ref_dim=0)."""
    torch.manual_seed(0)
    # flow_actor_coef=0: the offline/pretrain regime (pure BC + anchor, no Q-ascent).
    agent = make_agent(make_cfg("", flow_compose=True, flow_actor_coef=0.0), STATE_DIM, CHUNK_DIM, CHUNK)
    assert agent.actor.ref_dim == 0
    batch = make_batch(256)
    ref_rms = float(np.sqrt((batch["reference"] ** 2).mean()))
    for _ in range(600):
        stats = agent.actor_bc_step(batch)
    # G driven to ~0 (bc_loss) and the online composed policy copies the reference
    # tightly (actor_ref_rmse). The EMA target lags in a short toy run, so assert on
    # the online copy metric rather than act().
    assert stats["bc_loss"] < 1e-3
    assert stats["actor_ref_rmse"] < 0.5 * ref_rms
    assert agent.act(batch["state"], batch["reference"]).shape == batch["reference"].shape


def test_learned_critic_trains_by_td():
    """rlt_critic empty -> a fresh critic that TD-trains (comparison arms 2/3)."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(""), STATE_DIM, CHUNK_DIM, CHUNK)
    assert not agent.critic_frozen
    before = [p.detach().clone() for p in agent.critic.parameters()]
    stats = agent.critic_step(make_batch())
    assert "critic_loss" in stats and agent.critic_steps == 1
    assert any(not torch.equal(p, b) for p, b in zip(agent.critic.parameters(), before))


def test_set_critic_frozen_stops_td():
    torch.manual_seed(0)
    agent = make_agent(make_cfg(""), STATE_DIM, CHUNK_DIM, CHUNK)
    agent.set_critic_frozen(True)
    before = [p.detach().clone() for p in agent.critic.parameters()]
    assert agent.critic_step(make_batch()) == {}
    for p, b in zip(agent.critic.parameters(), before):
        assert torch.equal(p, b)


def test_save_load_roundtrip_learned_critic(tmp_path):
    torch.manual_seed(0)
    agent = make_agent(make_cfg(""), STATE_DIM, CHUNK_DIM, CHUNK)
    agent.critic_step(make_batch())
    agent.update(make_batch())
    ckpt = tmp_path / "flow_rlt.pt"
    agent.save(str(ckpt))
    fresh = make_agent(make_cfg(""), STATE_DIM, CHUNK_DIM, CHUNK)
    fresh.load(str(ckpt))
    for p, q in zip(agent.critic.parameters(), fresh.critic.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert torch.equal(p, q)


def test_update_runs_and_critic_stays_frozen(tmp_path):
    torch.manual_seed(0)
    agent = make_agent(make_cfg(make_rlt_critic(tmp_path)), STATE_DIM, CHUNK_DIM, CHUNK)
    before = [p.detach().clone() for p in agent.critic.parameters()]
    for _ in range(5):
        stats = agent.update(make_batch())
    assert "bc_loss" in stats and "actor_q" in stats
    assert agent.actor_steps == 5
    assert agent.critic_steps == 0
    for p, b in zip(agent.critic.parameters(), before):
        assert torch.equal(p, b)


def test_bc_overfit_recovers_reference_scale(tmp_path):
    """The V22 failure was act() emitting ~1e9 deltas; here it must land near 0.2."""
    torch.manual_seed(0)
    cfg = make_cfg(make_rlt_critic(tmp_path), lr=5e-3, cf_ema=0.9)
    agent = make_agent(cfg, STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch()
    first = agent.update(batch)["bc_loss"]
    for _ in range(300):
        stats = agent.update(batch)
    # The beta anchor pulls the endpoint toward the reference, so the pure BC fit
    # cannot reach zero; a large drop from the ~1.0 start is the right check.
    assert stats["bc_loss"] < first * 0.3

    state = batch["state"]
    action = agent.act(state, batch["reference"], explore=False)
    # A reference-conditioned corrector copies the reference at BC convergence:
    # closer to the reference than to zero (the V22 failure was 77x the reference).
    ref_rms = float(np.sqrt((batch["reference"] ** 2).mean()))
    copy_rmse = float(np.sqrt(((action - batch["reference"]) ** 2).mean()))
    assert copy_rmse < ref_rms
    # A random-init critic is not TD-clipped to [0, 1] like the real one, but a
    # frozen bounded critic keeps actor_q small -- the V22 failure was actor_q ~ 1e5.
    assert abs(stats["actor_q"]) < 10.0


def test_act_shape_matches_chunk(tmp_path):
    agent = make_agent(make_cfg(make_rlt_critic(tmp_path)), STATE_DIM, CHUNK_DIM, CHUNK)
    action = agent.act(np.zeros(STATE_DIM, np.float32), np.zeros(CHUNK_DIM, np.float32))
    assert action.shape == (1, CHUNK_DIM)


def test_ae_finetune_changes_decoder_only_when_enabled(tmp_path):
    ae = RLTokenAE(token_dim=8, z_dim=Z_DIM, d_model=16, n_heads=2, n_layers=1)
    cfg = make_cfg(make_rlt_critic(tmp_path), ae_finetune=True)
    agent = make_agent(cfg, STATE_DIM, CHUNK_DIM, CHUNK, token_ae=ae)
    agent.freeze_token()
    agent.set_ae_finetune(True)
    before = agent.token_ae.decoder.out_proj.weight.detach().clone()
    agent.update(make_batch(tokens=True))
    assert not torch.equal(before, agent.token_ae.decoder.out_proj.weight.detach())

    ae2 = RLTokenAE(token_dim=8, z_dim=Z_DIM, d_model=16, n_heads=2, n_layers=1)
    cfg2 = make_cfg(make_rlt_critic(tmp_path), ae_finetune=False)
    agent2 = make_agent(cfg2, STATE_DIM, CHUNK_DIM, CHUNK, token_ae=ae2)
    agent2.freeze_token()
    before2 = agent2.token_ae.decoder.out_proj.weight.detach().clone()
    agent2.update(make_batch(tokens=True))
    assert torch.equal(before2, agent2.token_ae.decoder.out_proj.weight.detach())


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    critic_path = make_rlt_critic(tmp_path)
    agent = make_agent(make_cfg(critic_path), STATE_DIM, CHUNK_DIM, CHUNK)
    agent.update(make_batch())
    ckpt = tmp_path / "flow_rlt.pt"
    agent.save(str(ckpt))

    fresh = make_agent(make_cfg(critic_path), STATE_DIM, CHUNK_DIM, CHUNK)
    fresh.load(str(ckpt))
    for p, q in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(agent.critic.parameters(), fresh.critic.parameters()):
        assert torch.equal(p, q)
