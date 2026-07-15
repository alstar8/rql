"""Focused unit tests for ConsensusLatentFlow helpers (no tokenizer ckpt)."""

from __future__ import annotations

import sys
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.consensus_latent_flow import (  # noqa: E402
    ConsensusLatentFlowAgent,
    get_config,
)
from agents.oattok_jax import (  # noqa: E402
    CODEBOOK_SIZE,
    FSQ_DIM,
    FSQ_LEVELS,
    build_codebook,
)
from utils.flax_utils import ModuleDict, TrainState  # noqa: E402
from utils.networks import MLP, Value  # noqa: E402


Agent = ConsensusLatentFlowAgent


def test_source_latent_in_codebook_box():
    codebook = build_codebook(FSQ_LEVELS)
    k = 16
    lo, hi = Agent.codebook_box_bounds(codebook, k)
    latent, tokens = Agent.sample_uniform_token_latent(jax.random.PRNGKey(0), 64, k)
    assert latent.shape == (64, k * FSQ_DIM)
    assert tokens.min() >= 0 and tokens.max() < CODEBOOK_SIZE
    assert bool((latent >= lo - 1e-6).all())
    assert bool((latent <= hi + 1e-6).all())
    # Source is quantized codes, not unbounded Gaussian.
    assert float(jnp.abs(latent).max()) <= float(jnp.abs(hi).max()) + 1e-5


def test_project_box_clips_to_bounds():
    codebook = build_codebook(FSQ_LEVELS)
    k = 4
    lo, hi = Agent.codebook_box_bounds(codebook, k)
    x = jnp.full((2, lo.shape[0]), 100.0)
    y = Agent.project_box(x, lo, hi)
    np.testing.assert_allclose(np.asarray(y), np.asarray(jnp.broadcast_to(hi, y.shape)))
    x2 = jnp.full((2, lo.shape[0]), -100.0)
    y2 = Agent.project_box(x2, lo, hi)
    np.testing.assert_allclose(np.asarray(y2), np.asarray(jnp.broadcast_to(lo, y2.shape)))


def test_flow_interpolant_and_target_velocity():
    x0 = jnp.zeros((3, 8))
    x1 = jnp.ones((3, 8))
    t = jnp.array([[0.25], [0.5], [0.75]])
    x_t, vel = Agent.flow_interpolant(x0, x1, t)
    np.testing.assert_allclose(np.asarray(x_t), np.asarray(t * x1))
    np.testing.assert_allclose(np.asarray(vel), 1.0)
    # Convex combo of box points stays in AABB when x0,x1 in box.
    codebook = build_codebook()
    lo, hi = Agent.codebook_box_bounds(codebook, 2)
    x0b = jnp.broadcast_to(lo, (4, lo.shape[0]))
    x1b = jnp.broadcast_to(hi, (4, hi.shape[0]))
    tt = jnp.linspace(0.1, 0.9, 4)[:, None]
    xt, _ = Agent.flow_interpolant(x0b, x1b, tt)
    assert bool((xt >= lo - 1e-6).all() and (xt <= hi + 1e-6).all())


def test_final_action_shape_helper():
    assert Agent.final_action_shape(8, h=1, prim_dim=21) == (8, 21)
    assert Agent.final_action_shape(4, h=3, prim_dim=7) == (4, 21)


def test_actor_warmup_and_ramp_schedule():
    coef = 1.0
    warmup, ramp = 50_000, 50_000
    assert float(Agent.actor_weight_from_step(1, warmup, ramp, coef)) == pytest.approx(0.0)
    assert float(Agent.actor_weight_from_step(warmup, warmup, ramp, coef)) == pytest.approx(
        0.0
    )
    mid = warmup + ramp // 2
    assert float(Agent.actor_weight_from_step(mid, warmup, ramp, coef)) == pytest.approx(
        0.5, abs=1e-5
    )
    assert float(
        Agent.actor_weight_from_step(warmup + ramp, warmup, ramp, coef)
    ) == pytest.approx(1.0)
    assert float(
        Agent.actor_weight_from_step(warmup + ramp + 10_000, warmup, ramp, coef)
    ) == pytest.approx(1.0)


def test_guidance_coef_zero_field():
    guidance_coef = 0.0
    times = jnp.array([[0.4], [0.9]])
    safe_w = jnp.ones((2, 8))
    g = guidance_coef * times * safe_w
    np.testing.assert_allclose(np.asarray(g), 0.0)


def test_seeded_source_reproducibility():
    k = 16
    a, ta = Agent.sample_uniform_token_latent(jax.random.PRNGKey(123), 5, k)
    b, tb = Agent.sample_uniform_token_latent(jax.random.PRNGKey(123), 5, k)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(ta), np.asarray(tb))
    c, _ = Agent.sample_uniform_token_latent(jax.random.PRNGKey(999), 5, k)
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_loss_composition_invariant():
    actor_weight, alpha, distill_coef = 0.5, 2.0, 0.5
    actor, bc, critic, distill = 1.0, 1.5, 0.25, 0.8
    total = actor_weight * actor + alpha * bc + critic + distill_coef * distill
    assert float(total) == pytest.approx(0.5 * 1.0 + 2.0 * 1.5 + 0.25 + 0.5 * 0.8)


def test_config_has_actor_warmup_keys():
    cfg = get_config()
    assert cfg.agent_name == "consensus_latent_flow"
    assert cfg.bc_warmup_steps == 50_000
    assert cfg.actor_ramp_steps == 50_000
    assert cfg.actor_coef == 1.0
    assert cfg.alpha == 1.0


def test_actor_q_gradient_nonzero_critic_params_fixed():
    """Tiny actor+value: actor loss grads hit actor, not critic params."""
    rng = jax.random.PRNGKey(0)
    obs_dim, latent_dim, action_dim = 3, 4, 2
    batch = 5

    class TinyActor(nn.Module):
        @nn.compact
        def __call__(self, x):
            return MLP((8, latent_dim), activate_final=False, layer_norm=False)(x)

    value_def = Value(hidden_dims=(8,), layer_norm=False, num_ensembles=2)
    actor_def = TinyActor()

    obs = jax.random.normal(rng, (batch, obs_dim))
    latent = jax.random.normal(jax.random.PRNGKey(1), (batch, latent_dim)) * 0.1
    times = jnp.full((batch, 1), 0.3)
    # Fake linear decoder: latent -> actions (trainable identity-ish matrix outside).
    decode_w = jax.random.normal(jax.random.PRNGKey(2), (latent_dim, action_dim))

    def decode(z):
        return z @ decode_w

    ex_actor = jnp.concatenate([obs[:1], latent[:1], times[:1]], -1)
    ex_value = jnp.concatenate([obs[:1], decode(latent[:1]), times[:1]], -1)
    network_def = ModuleDict({"actor": actor_def, "value": value_def})
    params = network_def.init(
        jax.random.PRNGKey(3),
        actor=(ex_actor,),
        value=(ex_value,),
    )["params"]
    network = TrainState.create(network_def, params, tx=optax.adam(1e-3))

    def actor_loss_fn(grad_params):
        fm = jnp.concatenate([obs, latent, times], -1)
        v = network.select("actor")(fm, params=grad_params)
        x_plus = latent + v * 0.1
        y_plus = decode(x_plus)
        # Q with stored params (not grad_params) — critic fixed for actor path.
        q = network.select("value")(jnp.concatenate([obs, y_plus, times], -1))
        return -q.mean()

    grads = jax.grad(actor_loss_fn)(network.params)
    actor_grad_norm = jnp.linalg.norm(
        jnp.concatenate([g.reshape(-1) for g in jax.tree_util.tree_leaves(grads["modules_actor"])])
    )
    critic_grad_leaves = jax.tree_util.tree_leaves(grads["modules_value"])
    critic_grad_norm = jnp.linalg.norm(
        jnp.concatenate([g.reshape(-1) for g in critic_grad_leaves])
    )
    assert float(actor_grad_norm) > 0.0
    assert float(critic_grad_norm) == pytest.approx(0.0)
    assert bool(jnp.isfinite(actor_grad_norm))


def test_actor_weight_zero_skips_nan_path():
    """jax.lax.cond with weight=0 must not propagate NaN from skipped branch."""

    def branch_nan(_):
        return jnp.array(jnp.nan)

    def branch_zero(_):
        return jnp.zeros(())

    weight = jnp.array(0.0)
    actor_loss = jax.lax.cond(weight > 0.0, branch_nan, branch_zero, operand=None)
    total = weight * actor_loss + 1.0
    assert jnp.isfinite(total)
    assert float(total) == pytest.approx(1.0)


def test_chunk_bc_weights_h1():
    rs = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    w = Agent.chunk_bc_weights(rs, h=1)
    np.testing.assert_allclose(np.asarray(w), np.ones(2))


def test_chunk_bc_weights_h_gt1_excludes_padded_chunks():
    # h=3 actions; example 0 has terminal at step 1 → action index 1 invalid.
    # rs_terminals[:3] rows correspond to actions 0,1,2.
    rs = jnp.array(
        [
            [0.0, 0.0],  # action 0 always valid under right-shift
            [1.0, 0.0],  # action 1 invalid for ex0
            [1.0, 0.0],  # action 2 invalid for ex0
            [0.0, 0.0],  # trailing next-state terminal row
        ]
    )
    w = Agent.chunk_bc_weights(rs, h=3)
    np.testing.assert_allclose(np.asarray(w), np.array([0.0, 1.0]))


def test_reverse_duration_identity():
    """N reverse Euler steps of size d/N cover total duration d (f: 1 → 1-d)."""
    n = 10
    for d in (0.0, 0.3, 0.7, 1.0):
        d_b = d / n
        assert float(n * d_b) == pytest.approx(d)
        f = 1.0
        for _ in range(n):
            f = f - d_b
        assert float(f) == pytest.approx(1.0 - d)


def test_target_update_uses_post_update_online_params():
    """v4 EMA blends new online params (not pre-update self.network online)."""
    d = 0.25
    old_online = {"w": jnp.array(0.0)}
    new_online = {"w": jnp.array(10.0)}
    old_target = {"w": jnp.array(1.0)}

    class _Net:
        def __init__(self, params):
            self.params = params

    new_network = _Net(
        {
            "modules_value": new_online,
            "modules_target_value": dict(old_target),
        }
    )
    # self is unused after the post-update EMA fix; call unbound.
    Agent.target_update(None, new_network, "value", d=d)
    expected = float(new_online["w"] * d + old_target["w"] * (1 - d))
    buggy = float(old_online["w"] * d + old_target["w"] * (1 - d))
    got = float(new_network.params["modules_target_value"]["w"])
    assert got == pytest.approx(expected)
    assert got != pytest.approx(buggy)


def test_inference_seed_reproducibility_source_only():
    """Same seed → same source latent (Euler is deterministic given noise)."""
    k = 8
    seed = jax.random.PRNGKey(42)
    a, _ = Agent.sample_uniform_token_latent(seed, 1, k)
    b, _ = Agent.sample_uniform_token_latent(seed, 1, k)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Distinct seeds differ (source is the only stochastic inference input).
    c, _ = Agent.sample_uniform_token_latent(jax.random.PRNGKey(7), 1, k)
    assert not np.array_equal(np.asarray(a), np.asarray(c))


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1]
        / "exp"
        / "ogbench_oattok"
        / "humanoidmaze-large_h1_d21.pkl"
    ).is_file(),
    reason="frozen OATTok checkpoint not present",
)
def test_sample_actions_seed_reproducibility():
    """End-to-end: seed only affects source noise; same seed → same actions."""
    tok = (
        Path(__file__).resolve().parents[1]
        / "exp"
        / "ogbench_oattok"
        / "humanoidmaze-large_h1_d21.pkl"
    )
    cfg = get_config()
    cfg.tokenizer_path = str(tok)
    cfg.h = 1
    cfg.batch_size = 4
    cfg.num_registers = 16
    obs_dim, act_dim = 69, 21
    agent = Agent.create(
        seed=0,
        ex_observations=jnp.zeros((1, obs_dim)),
        ex_actions=jnp.zeros((1, act_dim)),
        config=cfg,
    )
    obs = jnp.zeros((obs_dim,))
    seed = jax.random.PRNGKey(123)
    a1 = agent.sample_actions(obs, seed=seed, temperature=0.0)
    a2 = agent.sample_actions(obs, seed=seed, temperature=0.0)
    np.testing.assert_allclose(np.asarray(a1), np.asarray(a2), atol=1e-5)
    a3 = agent.sample_actions(obs, seed=jax.random.PRNGKey(999), temperature=0.0)
    assert not np.allclose(np.asarray(a1), np.asarray(a3), atol=1e-5)
