"""Focused unit tests for QuantizedDFLRQL9 + DFLRQL8 actor q_pe hook."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dflrql9 import DFLRQL9Agent, get_config as get_dflrql9_config  # noqa: E402
from agents.quantized_dflrql9 import (  # noqa: E402
    QuantizedDFLRQL9Agent,
    get_config as get_quantized_config,
)
from utils.flax_utils import restore_agent  # noqa: E402

TOK_PATH = ROOT / "exp/ogbench_oattok/humanoidmaze-large_h1_d21.pkl"
CKPT_DIR = ROOT / "exp/rql/humanoidmaze-large-dflrql9-400k-ckpt"
OBS_DIM = 69
PRIM_DIM = 21
H = 1


def _tiny_dflrql9_config(**overrides):
    cfg = dict(get_dflrql9_config())
    cfg.update(
        {
            "h": H,
            "batch_size": 4,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (32, 32),
            "value_hidden_dims": (32, 32),
            "guidance_hidden_dims": (32, 32),
        }
    )
    cfg.update(overrides)
    return cfg


def _tiny_quantized_config(**overrides):
    cfg = dict(get_quantized_config())
    cfg.update(
        {
            "h": H,
            "batch_size": 4,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (32, 32),
            "value_hidden_dims": (32, 32),
            "guidance_hidden_dims": (32, 32),
            "tokenizer_path": str(TOK_PATH),
            "projection_enabled": True,
        }
    )
    cfg.update(overrides)
    return cfg


def _ex_obs_actions(batch_size=2):
    obs = jnp.zeros((batch_size, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((batch_size, PRIM_DIM), dtype=jnp.float32)
    return obs, actions


def _synthetic_batch(agent, batch_size=None):
    b = int(batch_size or agent.config["batch_size"])
    h = int(agent.config["h"])
    # DFLRQL batches use horizon-leading layouts: (h+1, B, ...) for obs/etc.
    obs = jax.random.normal(jax.random.PRNGKey(1), (h + 1, b, OBS_DIM))
    actions = jnp.clip(
        jax.random.normal(jax.random.PRNGKey(2), (h, b, PRIM_DIM)), -1, 1
    )
    rewards = jax.random.normal(jax.random.PRNGKey(3), (h + 1, b))
    terminals = jnp.zeros((h + 1, b), dtype=jnp.float32)
    masks = jnp.ones((h + 1, b), dtype=jnp.float32)
    return {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "masks": masks,
    }


def test_actor_q_action_identity_default():
    obs, actions = _ex_obs_actions()
    agent = DFLRQL9Agent.create(0, obs, actions, _tiny_dflrql9_config())
    x = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)
    out = agent._actor_q_action(x)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x))


def test_identity_hook_preserves_dflrql9_total_loss_numerics():
    """Identity hook must not change DFLRQL9 loss vs explicit lookahead path."""
    obs, actions = _ex_obs_actions()
    cfg = _tiny_dflrql9_config()
    agent = DFLRQL9Agent.create(0, obs, actions, cfg)
    batch = _synthetic_batch(agent)
    rng = jax.random.PRNGKey(11)

    loss, info = agent.total_loss(batch, agent.network.params, rng=rng)
    assert np.isfinite(float(np.asarray(loss)))
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    assert np.isfinite(float(np.asarray(info["critic_loss"])))

    # Recording identity subclass must match parent numerics exactly.
    seen = []

    class _Recording(DFLRQL9Agent):
        def _actor_q_action(self, flat_action):
            seen.append(flat_action)
            return flat_action

    agent2 = _Recording(
        rng=agent.rng,
        network=agent.network,
        config=agent.config,
    )
    loss2, info2 = agent2.total_loss(batch, agent2.network.params, rng=rng)
    np.testing.assert_allclose(
        float(np.asarray(loss)), float(np.asarray(loss2)), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        float(np.asarray(info["actor_loss"])),
        float(np.asarray(info2["actor_loss"])),
        rtol=0,
        atol=0,
    )
    assert len(seen) == 1


def test_fsq_projection_hard_forward_and_grad():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")
    obs, actions = _ex_obs_actions()
    agent = QuantizedDFLRQL9Agent.create(
        0, obs, actions, _tiny_quantized_config()
    )
    flat = jnp.clip(
        jax.random.normal(jax.random.PRNGKey(5), (3, H * PRIM_DIM)), -0.9, 0.9
    )
    projected = agent._project_flat_action(flat)

    # Explicit tokenizer roundtrip must match projector hard forward.
    btd = flat.reshape(3, H, PRIM_DIM)
    recons, _tokens, _quant = agent.tokenizer_def.apply(
        {"params": agent.tokenizer_params},
        btd,
        deterministic=True,
    )
    recons = jnp.clip(recons, -1.0, 1.0)
    explicit = recons.reshape(3, H * PRIM_DIM)
    np.testing.assert_allclose(
        np.asarray(projected), np.asarray(explicit), rtol=1e-5, atol=1e-5
    )

    # Gradients w.r.t. continuous input are nonzero and finite.
    def _sum_proj(x):
        return agent._project_flat_action(x).sum()

    g = jax.grad(_sum_proj)(flat)
    assert np.isfinite(np.asarray(g)).all()
    assert float(np.abs(np.asarray(g)).sum()) > 0.0


def test_param_tree_matches_dflrql9_and_registry():
    from agents import agents as agent_registry

    assert "quantized_dflrql9" in agent_registry
    assert agent_registry["quantized_dflrql9"] is QuantizedDFLRQL9Agent

    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")

    obs, actions = _ex_obs_actions()
    a9 = DFLRQL9Agent.create(0, obs, actions, _tiny_dflrql9_config())
    aq = QuantizedDFLRQL9Agent.create(
        0, obs, actions, _tiny_quantized_config()
    )
    paths9 = [
        tuple(map(str, p))
        for p, _ in jax.tree_util.tree_flatten_with_path(a9.network.params)[0]
    ]
    pathsq = [
        tuple(map(str, p))
        for p, _ in jax.tree_util.tree_flatten_with_path(aq.network.params)[0]
    ]
    assert paths9 == pathsq
    # Tokenizer is external nonpytree, not in opt params.
    assert aq.tokenizer_def is not None
    assert aq.tokenizer_params is not None


def test_hook_isolation_only_actor_q_projected():
    """Subclass mock: critic path sees unprojected actions; q_pe path projected."""
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")

    class _Spy(QuantizedDFLRQL9Agent):
        def _project_flat_action(self, flat_action):
            # Mark projected actions with a large constant shift for detection.
            return flat_action + 10.0

        def _actor_q_action(self, flat_action):
            return self._project_flat_action(flat_action)

    obs, actions = _ex_obs_actions()
    base = QuantizedDFLRQL9Agent.create(
        0, obs, actions, _tiny_quantized_config()
    )
    spy = _Spy(
        rng=base.rng,
        network=base.network,
        config=base.config,
        tokenizer_def=base.tokenizer_def,
        tokenizer_params=base.tokenizer_params,
        tokenizer_meta=base.tokenizer_meta,
    )

    # Identity parent vs spy: only actor_loss should move when projection
    # becomes a constant shift (critic uses behavior reverse, not hook).
    batch = _synthetic_batch(base)
    rng = jax.random.PRNGKey(21)

    class _IdentityQ(QuantizedDFLRQL9Agent):
        def _actor_q_action(self, flat_action):
            return flat_action

        def _project_flat_action(self, flat_action):
            return flat_action

    ident = _IdentityQ(
        rng=base.rng,
        network=base.network,
        config=base.config,
        tokenizer_def=base.tokenizer_def,
        tokenizer_params=base.tokenizer_params,
        tokenizer_meta=base.tokenizer_meta,
    )
    _, info_id = ident.total_loss(batch, ident.network.params, rng=rng)
    _, info_sp = spy.total_loss(batch, spy.network.params, rng=rng)

    # Critic loss identical (hook unused on critic path).
    np.testing.assert_allclose(
        float(np.asarray(info_id["critic_loss"])),
        float(np.asarray(info_sp["critic_loss"])),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        float(np.asarray(info_id["bc_loss"])),
        float(np.asarray(info_sp["bc_loss"])),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        float(np.asarray(info_id["distill_loss"])),
        float(np.asarray(info_sp["distill_loss"])),
        rtol=0,
        atol=0,
    )
    # Actor / q_pe must differ under the shifted projection.
    assert float(np.asarray(info_id["actor_loss"])) != pytest.approx(
        float(np.asarray(info_sp["actor_loss"])), abs=1e-6
    )


def test_sample_actions_equals_projector_deterministic():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")
    obs, actions = _ex_obs_actions()
    agent = QuantizedDFLRQL9Agent.create(
        0, obs, actions, _tiny_quantized_config()
    )
    o = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    seed = jax.random.PRNGKey(7)
    a1 = agent.sample_actions(o, seed=seed, temperature=0.0)
    a2 = agent.sample_actions(o, seed=seed, temperature=0.0)
    np.testing.assert_allclose(np.asarray(a1), np.asarray(a2), rtol=0, atol=0)
    assert a1.shape == (H, PRIM_DIM)

    # Same continuous flow, then the *jitted* projector (deploy path is XLA).
    # Eager vs XLA FSQ can disagree on half-integer round boundaries; compare
    # under the same transform used inside sample_actions.
    action_rng, n_rng = jax.random.split(seed)
    o2 = jnp.atleast_2d(o)[-1:]
    noise = jax.random.normal(n_rng, (1, agent.config["action_dim"]))
    raw_flat = agent.compute_flow_actions(
        o2, seed=action_rng, noise=noise, temperature=0.0
    )
    project_jit = jax.jit(lambda x: agent._project_flat_action(x))
    expected = project_jit(raw_flat).reshape(H, PRIM_DIM)
    np.testing.assert_allclose(
        np.asarray(a1), np.asarray(expected), rtol=1e-5, atol=1e-5
    )
    assert np.isfinite(np.asarray(a1)).all()
    assert float(np.max(np.abs(np.asarray(a1)))) <= 1.0 + 1e-5


def test_tokenizer_meta_mismatch_raises():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")
    obs, actions = _ex_obs_actions()
    cfg = _tiny_quantized_config(h=3)
    with pytest.raises(ValueError, match="sample_horizon"):
        QuantizedDFLRQL9Agent.create(0, obs, actions, cfg)

    cfg2 = _tiny_quantized_config()
    # Wrong prim dim via fake ex_actions width.
    bad_actions = jnp.zeros((2, 7), dtype=jnp.float32)
    with pytest.raises(ValueError, match="sample_dim"):
        QuantizedDFLRQL9Agent.create(0, obs, bad_actions, cfg2)

    cfg3 = _tiny_quantized_config(tokenizer_path="")
    with pytest.raises(ValueError, match="tokenizer_path"):
        QuantizedDFLRQL9Agent.create(0, obs, actions, cfg3)


def test_actor_grad_through_projection_reaches_actor_not_tokenizer():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")
    obs, actions = _ex_obs_actions()
    agent = QuantizedDFLRQL9Agent.create(
        0, obs, actions, _tiny_quantized_config()
    )
    batch = _synthetic_batch(agent)
    rng = jax.random.PRNGKey(33)

    def actor_loss_only(params):
        _, info = agent.total_loss(batch, params, rng=rng)
        return info["actor_loss"]

    grads = jax.grad(actor_loss_only)(agent.network.params)
    actor_leaves = jax.tree_util.tree_leaves(grads["modules_actor"])
    actor_norm = sum(float(np.sum(np.abs(np.asarray(x)))) for x in actor_leaves)
    assert actor_norm > 0.0
    assert all(np.isfinite(np.asarray(x)).all() for x in actor_leaves)

    # Tokenizer params are not in network.params / optimizer.
    assert "tokenizer" not in str(grads.keys())
    tok_leaves = jax.tree_util.tree_leaves(agent.tokenizer_params)
    assert len(tok_leaves) > 0


def test_restore_dflrql9_400k_checkpoint_compatible():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")
    ckpts = sorted(CKPT_DIR.glob("sd000_*/params_400000.pkl"))
    if not ckpts:
        pytest.skip(f"missing 400k ckpt under {CKPT_DIR}")
    ckpt = ckpts[0]

    # Full-size create matching HL flags (needed for restore shape match).
    cfg = dict(get_quantized_config())
    cfg.update(
        {
            "h": 1,
            "batch_size": 256,
            "ensemble_ct": 10,
            "alpha": 0.3,
            "discount": 0.995,
            "expectile": 0.5,
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.01,
            "conflict_power": 2.0,
            "residual_coef": 0.25,
            "tokenizer_path": str(TOK_PATH),
            "projection_enabled": True,
        }
    )
    obs = jnp.zeros((1, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((1, PRIM_DIM), dtype=jnp.float32)
    agent = QuantizedDFLRQL9Agent.create(0, obs, actions, cfg)
    restored = restore_agent(agent, str(ckpt))
    assert int(restored.network.step) >= 1
    # Tokenizer fields survive restore (nonpytree).
    assert restored.tokenizer_def is not None
    assert restored.tokenizer_params is not None
    # Sample works after restore.
    out = restored.sample_actions(
        obs[0], seed=jax.random.PRNGKey(0), temperature=0.0
    )
    assert out.shape == (1, PRIM_DIM)
    assert np.isfinite(np.asarray(out)).all()


def test_one_batch_update_smoke():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")
    obs, actions = _ex_obs_actions()
    agent = QuantizedDFLRQL9Agent.create(
        0, obs, actions, _tiny_quantized_config()
    )
    batch = _synthetic_batch(agent)
    new_agent, info = agent.update(batch)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert int(new_agent.network.step) == int(agent.network.step) + 1
