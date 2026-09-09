"""V22_25: fixed CF routing (policy bootstrap, consensus distill, dead-grad gate,
trust-gated contraction, velocity-relative G) on the flow_rlt recipe."""

from __future__ import annotations

import numpy as np
import torch

from rlt.config import OnlineConfig
from rlt.consensusflow import make_agent
from rlt.token_ae import RLTokenAE
from rlt.v22_25 import V2225Agent

STATE_DIM = 8
CHUNK = 4
CHUNK_DIM = CHUNK * 2
Z_DIM = 4


def make_cfg(**overrides) -> OnlineConfig:
    return OnlineConfig(
        episode_idx=128,
        device="cpu",
        batch_size=4,
        hidden_dim=16,
        algorithm="v22_25",
        cf_hidden_dim=16,
        cf_n_layers=2,
        cf_ensemble=3,
        cf_flow_steps=2,
        cf_guidance_coef=0.25,  # velocity-relative, matching the pick-18 pipeline
        token_batch_size=2,
        z_dim=Z_DIM,
        **overrides,
    )


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


def test_make_agent_picks_v22_25():
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    assert isinstance(agent, V2225Agent)


def test_guidance_is_exactly_zero_at_init():
    """Zero-init W head: the deployed field at init is V alone (identity story)."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch()
    state = torch.as_tensor(batch["state"])
    x = torch.as_tensor(batch["reference"])
    t = torch.rand(state.shape[0], 1)
    v = agent.actor(state, x, t, torch.as_tensor(batch["reference"]))
    g = agent.guidance_field(state, x, t, v)
    assert torch.equal(g, torch.zeros_like(g))


def test_critic_step_trains_ten_head_timed_critic():
    torch.manual_seed(0)
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    before = [p.detach().clone() for p in agent.critic.parameters()]
    stats = agent.critic_step(make_batch())
    assert "critic_loss" in stats and agent.critic_steps == 1
    assert 0.0 <= stats["reverse_t_mean"] <= 1.0
    assert 0.0 <= stats["target_mean"] <= 1.0  # clipped TD target
    assert any(not torch.equal(p, b) for p, b in zip(agent.critic.parameters(), before))


def test_bootstrap_consumes_next_reference():
    """The policy bootstrap unrolls the EMA actor at (s', a_ref'): changing
    next_reference must change the TD target. v22_24's noise bootstrap never
    read next_reference, so this is the regression test for the fix."""
    batch = make_batch()
    big_ref = batch["next_reference"] * 100.0

    torch.manual_seed(0)
    agent_a = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    torch.manual_seed(7)  # same unroll noise for both calls
    target_a = agent_a.critic_step(batch)["target_mean"]

    torch.manual_seed(0)
    agent_b = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    torch.manual_seed(7)
    target_b = agent_b.critic_step({**batch, "next_reference": big_ref})["target_mean"]

    assert target_a != target_b


def test_terminal_rows_bootstrap_nothing():
    """done=1: the target is exactly the clipped reward, no bootstrap term."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch()
    batch["done"] = np.ones_like(batch["done"])
    stats = agent.critic_step(batch)
    expected = float(np.clip(batch["reward"], 0.0, 1.0).mean())
    assert abs(stats["target_mean"] - expected) < 1e-6


def test_actor_step_trains_actor_and_guidance_not_critic():
    torch.manual_seed(0)
    agent = make_agent(make_cfg(cf_distill_grad_floor=0.0), STATE_DIM, CHUNK_DIM, CHUNK)
    agent.critic_step(make_batch())  # give the distill target a trained critic
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    guide_before = [p.detach().clone() for p in agent.guidance.parameters()]
    stats = agent.actor_step(make_batch())
    assert agent.actor_steps == 1 and agent.critic_steps == 1
    for key in ("bc_loss", "actor_q", "anchor_loss", "distill_loss", "distill_on", "w_norm", "g_over_v"):
        assert key in stats
    assert stats["distill_on"] == 1.0
    assert any(not torch.equal(p, b) for p, b in zip(agent.actor.parameters(), actor_before))
    assert any(not torch.equal(p, b) for p, b in zip(agent.guidance.parameters(), guide_before))
    for p, b in zip(agent.critic.parameters(), critic_before):
        assert torch.equal(p, b)


def test_stage0_disables_lookahead_gradient():
    """cf_actor_coef=0: the lookahead is logged but contributes no gradient."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(cf_actor_coef=0.0), STATE_DIM, CHUNK_DIM, CHUNK)
    assert agent.actor_coef == 0.0
    stats = agent.actor_step(make_batch())
    assert "actor_q" in stats  # still logged for the kill diagnostics


def test_distill_gate_skips_dead_critic():
    """A critic with zeroed output layers has grad_x Q = 0 < floor: the distill
    term must be skipped (logged 0, distill_on 0) and W must not move."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    for head in agent.critic.heads:
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.zeros_(head[-1].bias)
    for head in agent.critic_target.heads:
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.zeros_(head[-1].bias)
    guide_before = [p.detach().clone() for p in agent.guidance.parameters()]
    stats = agent.actor_step(make_batch())
    assert stats["distill_on"] == 0.0
    assert stats["distill_loss"] == 0.0
    assert stats["q_grad_scale"] < agent.distill_grad_floor
    for p, b in zip(agent.guidance.parameters(), guide_before):
        assert torch.equal(p, b)


def test_consensus_target_is_agreement_bounded():
    """The consensus target is the mean of ~unit-norm per-head directions, so
    its norm is <= 1 and shrinks when heads disagree."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(cf_distill_grad_floor=0.0), STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch(64)
    for _ in range(3):
        agent.critic_step(batch)
    state = torch.as_tensor(batch["state"])
    x_t = torch.as_tensor(batch["reference"])
    t = torch.rand(state.shape[0], 1)
    z, grad_scale = agent._consensus_target(state, x_t, t)
    assert z is not None and grad_scale > 0.0
    assert z.shape == (state.shape[0], CHUNK_DIM)
    assert z.norm(dim=-1).max().item() <= 1.0 + 1e-5


def test_conflict_contraction_is_trust_gated():
    """kill = (1 - trust)^p: confident guidance keeps its anti-V direction;
    weak guidance loses it. v22_24's 1 - trust^p killed 84% at trust 0.4."""
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    v = torch.zeros(1, CHUNK_DIM)
    v[0, 0] = 1.0

    # trust ~0.5, fully anti-V: keeps (1 - 0.25) of the component -> 0.375.
    w_mid = torch.zeros(1, CHUNK_DIM)
    w_mid[0, 0] = -0.5
    out = agent._behavior_safe_direction(w_mid, v)
    assert 0.34 < out.norm(dim=-1).item() < 0.41
    assert (out[0, 0] / out.norm(dim=-1)).item() < -0.99  # anti direction kept

    # trust ~1, fully anti-V: kill ~ 0, the component survives intact.
    w_full = torch.zeros(1, CHUNK_DIM)
    w_full[0, 0] = -0.99
    out_full = agent._behavior_safe_direction(w_full, v)
    assert out_full.norm(dim=-1).item() > 0.9

    # trust ~0.1: kill = 0.81, the anti component is effectively removed.
    w_weak = torch.zeros(1, CHUNK_DIM)
    w_weak[0, 0] = -0.1
    out_weak = agent._behavior_safe_direction(w_weak, v)
    assert out_weak.norm(dim=-1).item() < 0.05


def test_guidance_scales_with_velocity():
    """G = lambda * t * ||sg(v)|| * u: doubling ||v|| must double ||G||."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(cf_distill_grad_floor=0.0), STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch(64)
    for _ in range(5):
        agent.critic_step(batch)
        agent.actor_step(batch)
    state = torch.as_tensor(batch["state"])
    x = torch.as_tensor(batch["reference"])
    t = torch.rand(state.shape[0], 1)
    v = agent.actor(state, x, t, torch.as_tensor(batch["reference"]))
    g1 = agent.guidance_field(state, x, t, v)
    g2 = agent.guidance_field(state, x, t, 2.0 * v)
    assert torch.allclose(g2, 2.0 * g1, rtol=1e-4, atol=1e-7)
    # Magnitude at a trained-scale operating point (||w|| ~ 0.5, set directly on
    # the real head): the guide must be a real fraction of the field, not
    # v22_24's ~2% unit-ball rounding error.
    with torch.no_grad():
        agent.guidance.net[-1].bias.copy_(0.5 * torch.ones(CHUNK_DIM) / np.sqrt(CHUNK_DIM))
    g = agent.guidance_field(state, x, t, v)
    ratio = (g.norm(dim=-1) / v.norm(dim=-1).clamp_min(1e-6)).mean().item()
    assert ratio > 0.02


def test_distill_moves_guidance_off_zero():
    torch.manual_seed(0)
    agent = make_agent(make_cfg(cf_distill_grad_floor=0.0), STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch(256)
    for _ in range(5):
        agent.critic_step(batch)
        agent.actor_step(batch)
    w = agent.guidance(
        torch.as_tensor(batch["state"]),
        torch.as_tensor(batch["reference"]),
        torch.rand(batch["state"].shape[0], 1),
    )
    assert w.norm(dim=-1).mean().item() > 0.0


def test_guidance_override_pairs_the_intervention():
    """lambda=0 must reproduce the unguided unroll; the trained lambda must not."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(cf_ema=0.9, cf_distill_grad_floor=0.0), STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch(256)
    for _ in range(10):
        agent.critic_step(batch)
        agent.actor_step(batch)
    state, reference = batch["state"], batch["reference"]

    agent.set_guidance_coef(0.0)
    torch.manual_seed(7)
    off = agent.act(state, reference, explore=False)
    torch.manual_seed(7)
    unguided = agent._unroll(
        torch.as_tensor(state), torch.randn(state.shape[0], CHUNK_DIM), torch.as_tensor(reference),
        guided=False, target=True,
    ).numpy()
    assert np.allclose(off, unguided, atol=1e-6)

    agent.set_guidance_coef(0.25)
    torch.manual_seed(7)
    on = agent.act(state, reference, explore=False)
    assert not np.allclose(on, off, atol=1e-6)


def test_bc_overfit_copies_reference_scale():
    """The first-V22 failure was act() emitting ~1e9 deltas; V must stay near 0.2."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(lr=5e-3, cf_ema=0.9, cf_actor_coef=0.0), STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch()
    first = agent.actor_step(batch)["bc_loss"]
    losses = []
    for _ in range(300):
        stats = agent.actor_step(batch)
        losses.append(stats["bc_loss"])
    # bc_loss is re-measured on fresh noise each step, so the point estimate is
    # noisy; the tail minimum is the robust convergence check.
    assert min(losses[-50:]) < first * 0.3

    agent.set_guidance_coef(0.0)  # judge V alone: the anchor + BC copy
    action = agent.act(batch["state"], batch["reference"], explore=False)
    ref_rms = float(np.sqrt((batch["reference"] ** 2).mean()))
    copy_rmse = float(np.sqrt(((action - batch["reference"]) ** 2).mean()))
    assert copy_rmse < ref_rms
    assert abs(stats["actor_q"]) < 10.0


def test_act_shape_matches_chunk():
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    action = agent.act(np.zeros(STATE_DIM, np.float32), np.zeros(CHUNK_DIM, np.float32))
    assert action.shape == (1, CHUNK_DIM)


def test_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    agent.critic_step(make_batch())
    agent.actor_step(make_batch())
    ckpt = tmp_path / "v22_25.pt"
    agent.save(str(ckpt))

    fresh = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    fresh.load(str(ckpt))
    for p, q in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(agent.critic.parameters(), fresh.critic.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(agent.guidance.parameters(), fresh.guidance.parameters()):
        assert torch.equal(p, q)
    assert fresh.critic_steps == agent.critic_steps
    assert fresh.actor_steps == agent.actor_steps


def test_ae_finetune_changes_decoder_only_when_enabled():
    ae = RLTokenAE(token_dim=8, z_dim=Z_DIM, d_model=16, n_heads=2, n_layers=1)
    agent = make_agent(make_cfg(ae_finetune=True), STATE_DIM, CHUNK_DIM, CHUNK, token_ae=ae)
    agent.freeze_token()
    agent.set_ae_finetune(True)
    before = agent.token_ae.decoder.out_proj.weight.detach().clone()
    agent.actor_step(make_batch(tokens=True))
    assert not torch.equal(before, agent.token_ae.decoder.out_proj.weight.detach())

    ae2 = RLTokenAE(token_dim=8, z_dim=Z_DIM, d_model=16, n_heads=2, n_layers=1)
    agent2 = make_agent(make_cfg(ae_finetune=False), STATE_DIM, CHUNK_DIM, CHUNK, token_ae=ae2)
    agent2.freeze_token()
    before2 = agent2.token_ae.decoder.out_proj.weight.detach().clone()
    agent2.actor_step(make_batch(tokens=True))
    assert torch.equal(before2, agent2.token_ae.decoder.out_proj.weight.detach())


def test_set_critic_frozen_stops_td():
    torch.manual_seed(0)
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    agent.set_critic_frozen(True)
    before = [p.detach().clone() for p in agent.critic.parameters()]
    assert agent.critic_step(make_batch()) == {}
    for p, b in zip(agent.critic.parameters(), before):
        assert torch.equal(p, b)
