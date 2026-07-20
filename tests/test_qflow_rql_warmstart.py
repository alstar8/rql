"""Focused unit tests for QFlowRQLWarmstartAgent (paper-aligned Q-Flow v2)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from einops import rearrange

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.qflow_rql_warmstart import (  # noqa: E402
    DeterministicVectorField,
    FourierFeatures,
    FourierValue,
    QFlowRQLWarmstartAgent,
    TRAINING_PHASES,
    fourier_time_features,
    get_config,
)
from utils.flax_utils import restore_agent, save_agent  # noqa: E402
from utils.networks import Value  # noqa: E402


OBS_DIM = 8
PRIM_DIM = 4
H = 1


def rearrange_actions(batch, h):
    return rearrange(batch["actions"][:h], "h b d -> b (h d)")


def _tiny_config(**overrides):
    cfg = dict(get_config())
    cfg.update(
        {
            "h": H,
            "batch_size": 8,
            "ensemble_ct": 2,
            "inner_ensemble_ct": 2,
            "rql_ensemble_ct": 2,
            "flow_steps": 4,
            "actor_hidden_dims": (32, 32),
            "value_hidden_dims": (32, 32),
            "inner_value_hidden_dims": (32, 32),
            "actor_fourier_dim": 8,
            "inner_fourier_dim": 8,
            "alpha": 0.3,
            "expectile": 0.5,
            "discount": 0.995,
            "q_agg": "mean",
            "qflow_lambda": 1.0,
            "qflow_actor_coef": 1.0,
            "qflow_bridge_blend": 1.0,
            "training_phase": "rql_warmstart",
            "tau": 0.005,
            "terminal_q_tau": 0.005,
            "ema": 0.999,
            "rho": 0.0,
        }
    )
    cfg.update(overrides)
    return cfg


def _ex_obs_actions(batch_size=2):
    obs = jnp.zeros((batch_size, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((batch_size, PRIM_DIM), dtype=jnp.float32)
    return obs, actions


def _synthetic_batch(batch_size=8, h=H, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(h + 1, batch_size, OBS_DIM)).astype(np.float32)
    actions = np.clip(
        rng.normal(size=(h, batch_size, PRIM_DIM)).astype(np.float32), -1, 1
    )
    rewards = rng.normal(size=(h + 1, batch_size)).astype(np.float32)
    terminals = np.zeros((h + 1, batch_size), dtype=np.float32)
    masks = np.ones((h + 1, batch_size), dtype=np.float32)
    return {
        "observations": jnp.asarray(obs),
        "actions": jnp.asarray(actions),
        "rewards": jnp.asarray(rewards),
        "terminals": jnp.asarray(terminals),
        "masks": jnp.asarray(masks),
    }


def _flat_norm(tree):
    leaves = [np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(tree)]
    if not leaves:
        return 0.0
    return float(np.linalg.norm(np.concatenate(leaves)))


@pytest.fixture(scope="module")
def warmstart_agent():
    obs, actions = _ex_obs_actions()
    return QFlowRQLWarmstartAgent.create(0, obs, actions, _tiny_config())


@pytest.fixture(scope="module")
def bridge_agent():
    obs, actions = _ex_obs_actions()
    return QFlowRQLWarmstartAgent.create(
        1,
        obs,
        actions,
        _tiny_config(training_phase="qflow_bridge", qflow_bridge_blend=0.5),
    )


@pytest.fixture(scope="module")
def online_agent():
    obs, actions = _ex_obs_actions()
    return QFlowRQLWarmstartAgent.create(
        2, obs, actions, _tiny_config(training_phase="qflow_online")
    )


def test_config_defaults():
    cfg = get_config()
    assert cfg.agent_name == "qflow_rql_warmstart"
    assert cfg.training_phase == "rql_warmstart"
    assert cfg.h == 1
    assert cfg.alpha == 0.3
    assert cfg.discount == 0.995
    assert cfg.ensemble_ct == 2
    assert cfg.inner_ensemble_ct == 2
    assert cfg.inner_fourier_dim == 16
    assert cfg.flow_steps == 10
    assert cfg.qflow_bridge_blend == 1.0
    assert tuple(TRAINING_PHASES) == (
        "rql_warmstart",
        "qflow_bridge",
        "qflow_online",
    )


def test_registry_wiring():
    from agents import agents

    assert "qflow_rql_warmstart" in agents
    assert agents["qflow_rql_warmstart"] is QFlowRQLWarmstartAgent


def test_fourier_and_timefree_contracts():
    """Fourier features are even-dim; outer Q is time-free; VF is deterministic."""
    freqs = jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
    t = jnp.array([[0.0], [0.25], [0.5], [1.0]], dtype=jnp.float32)
    feats = fourier_time_features(t, freqs)
    assert feats.shape == (4, 8)
    # t=0 → sin=0, cos=1 for all freqs
    expected0 = jnp.concatenate([jnp.zeros(4), jnp.ones(4)], axis=0)
    np.testing.assert_allclose(np.asarray(feats[0]), np.asarray(expected0), atol=1e-5)

    key = jax.random.PRNGKey(0)
    ff = FourierFeatures(features=8)
    vars_ff = ff.init(key, t)
    out = ff.apply(vars_ff, t)
    assert out.shape == (4, 8)

    vf = DeterministicVectorField(
        hidden_dims=(16, 16), action_dim=PRIM_DIM, fourier_dim=8
    )
    obs = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    act = jnp.zeros((2, PRIM_DIM), dtype=jnp.float32)
    times = jnp.zeros((2, 1), dtype=jnp.float32)
    vars_vf = vf.init(key, obs, act, times)
    vel = vf.apply(vars_vf, obs, act, times)
    assert vel.shape == (2, PRIM_DIM)
    # No Gaussian / mode API.
    assert not hasattr(vf, "mode")

    # Time-free outer Q: Value(obs, actions) — no time channel.
    q_mod = Value(hidden_dims=(16, 16), num_ensembles=2, layer_norm=True)
    vars_q = q_mod.init(key, obs, act)
    qs = q_mod.apply(vars_q, obs, act)
    assert qs.shape == (2, 2)  # (ensemble, batch)

    iv = FourierValue(
        hidden_dims=(16, 16), fourier_dim=8, num_ensembles=2, layer_norm=True
    )
    vars_iv = iv.init(key, obs, act, times)
    vs = iv.apply(vars_iv, obs, act, times)
    assert vs.shape == (2, 2)


def test_module_tree(warmstart_agent):
    keys = set(warmstart_agent.network.params.keys())
    for name in (
        "modules_actor",
        "modules_target_actor",
        "modules_value",
        "modules_target_value",
        "modules_terminal_q",
        "modules_target_terminal_q",
        "modules_inner_value",
    ):
        assert name in keys
    # Outer Q is time-free (no Fourier submodule params).
    assert "time_features" not in warmstart_agent.network.params["modules_terminal_q"]
    # Actor is a deterministic VF MLP (Dense stack), not Gaussian Actor.
    actor_leaves = warmstart_agent.network.params["modules_actor"]
    assert "mlp" in actor_leaves
    assert "mean_net" not in actor_leaves
    assert "log_std_net" not in actor_leaves
    assert "log_stds" not in actor_leaves


def test_exact_rql_warmstart_gradient_isolation(warmstart_agent):
    """RQL loss touches actor/value only; aux Q/V are isolated."""
    batch = _synthetic_batch(batch_size=warmstart_agent.config["batch_size"], seed=3)
    rng = jax.random.PRNGKey(11)
    observations = batch["observations"][0]
    flat_actions = rearrange_actions(batch, warmstart_agent.config["h"])

    def rql_only(params):
        loss, _, _ = warmstart_agent.rql_actor_critic_loss(batch, params, rng)
        return loss

    rql_grads = jax.grad(rql_only)(warmstart_agent.network.params)
    assert _flat_norm(rql_grads["modules_actor"]) > 0.0
    assert _flat_norm(rql_grads["modules_value"]) > 0.0
    assert _flat_norm(rql_grads["modules_terminal_q"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(rql_grads["modules_inner_value"]) == pytest.approx(0.0, abs=1e-8)

    def tq_only(params):
        loss, _, _ = warmstart_agent.terminal_q_bellman_loss(batch, params, rng)
        return loss

    tq_grads = jax.grad(tq_only)(warmstart_agent.network.params)
    assert _flat_norm(tq_grads["modules_terminal_q"]) > 0.0
    assert _flat_norm(tq_grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(tq_grads["modules_value"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(tq_grads["modules_inner_value"]) == pytest.approx(0.0, abs=1e-8)

    def iv_only(params):
        loss, _ = warmstart_agent.inner_value_loss(
            observations, flat_actions, params, rng
        )
        return loss

    iv_grads = jax.grad(iv_only)(warmstart_agent.network.params)
    assert _flat_norm(iv_grads["modules_inner_value"]) > 0.0
    assert _flat_norm(iv_grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_value"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_terminal_q"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_target_terminal_q"]) == pytest.approx(
        0.0, abs=1e-8
    )


def test_current_policy_rollout_no_bptt(online_agent):
    """Rollout from arbitrary t uses current actor; no BPTT into actor for V."""
    b, d = 4, online_agent.config["action_dim"]
    obs = jax.random.normal(jax.random.PRNGKey(1), (b, OBS_DIM))
    x = jax.random.normal(jax.random.PRNGKey(2), (b, d))
    t = jnp.array([[0.0], [0.25], [0.5], [0.9]], dtype=jnp.float32)
    out = online_agent.roll_flow_to_terminal(obs, x, t, params=None)
    assert out.shape == (b, d)
    assert np.isfinite(np.asarray(out)).all()
    t1 = jnp.ones((b, 1), dtype=jnp.float32)
    out1 = online_agent.roll_flow_to_terminal(obs, x, t1, params=None)
    np.testing.assert_allclose(np.asarray(out1), np.asarray(x), atol=1e-5)

    batch = _synthetic_batch(batch_size=online_agent.config["batch_size"], seed=12)
    observations = batch["observations"][0]
    actions = rearrange_actions(batch, online_agent.config["h"])
    rng = jax.random.PRNGKey(13)

    def iv_only(params):
        loss, _ = online_agent.inner_value_loss(
            observations, actions, params, rng
        )
        return loss

    iv_grads = jax.grad(iv_only)(online_agent.network.params)
    assert _flat_norm(iv_grads["modules_inner_value"]) > 0.0
    assert _flat_norm(iv_grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)

    def actor_part(params):
        _, info = online_agent.qflow_policy_loss(
            batch,
            params,
            rng,
            blend=1.0,
            actor_coef=1.0,
        )
        return info["actor_loss"]

    actor_grads = jax.grad(actor_part)(online_agent.network.params)
    assert _flat_norm(actor_grads["modules_actor"]) > 0.0
    assert _flat_norm(actor_grads["modules_inner_value"]) == pytest.approx(
        0.0, abs=1e-7
    )


def test_exact_loss_decomposition_online(online_agent):
    """Online = critic + inner V + actor_coef * matching; blend forced to 1."""
    batch = _synthetic_batch(batch_size=online_agent.config["batch_size"], seed=20)
    rng = jax.random.PRNGKey(21)
    loss, info = online_agent.qflow_policy_loss(
        batch,
        online_agent.network.params,
        rng,
        blend=1.0,
        actor_coef=online_agent.config["qflow_actor_coef"],
    )
    expected = (
        float(info["critic_loss"])
        + float(info["inner_value_loss"])
        + float(info["actor_coef"]) * float(info["actor_loss"])
    )
    np.testing.assert_allclose(float(loss), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(float(info["qflow_bridge_blend"]), 1.0, atol=1e-6)
    # No RQL-only keys required online.
    assert "q_pe_mean" not in info


def test_bridge_blend_interpolates_target(bridge_agent):
    """blend=0 matches CFM; blend=1 matches full Q-Flow guidance."""
    batch = _synthetic_batch(batch_size=bridge_agent.config["batch_size"], seed=30)
    rng = jax.random.PRNGKey(31)
    params = bridge_agent.network.params

    _, info0 = bridge_agent.qflow_policy_loss(
        batch, params, rng, blend=0.0, actor_coef=1.0
    )
    _, info1 = bridge_agent.qflow_policy_loss(
        batch, params, rng, blend=1.0, actor_coef=1.0
    )
    # With blend=0, actor matches CFM ⇒ actor_loss ≈ bc_loss.
    np.testing.assert_allclose(
        float(info0["actor_loss"]),
        float(info0["bc_loss"]),
        rtol=1e-4,
        atol=1e-4,
    )
    # With nonzero inner grads, blend=1 diverges from pure CFM.
    if float(info1["inner_grad_norm"]) > 1e-6:
        assert float(info1["actor_loss"]) != pytest.approx(
            float(info1["bc_loss"]), abs=1e-5
        )


def test_terminal_q_target_polyak(warmstart_agent):
    batch = _synthetic_batch(batch_size=warmstart_agent.config["batch_size"], seed=19)
    params = dict(warmstart_agent.network.params)
    params["modules_target_terminal_q"] = jax.tree_util.tree_map(
        lambda x: x + 1.0, params["modules_target_terminal_q"]
    )
    agent = warmstart_agent.replace(
        network=warmstart_agent.network.replace(params=params)
    )
    before_online = agent.network.params["modules_terminal_q"]
    before_target = agent.network.params["modules_target_terminal_q"]
    new_agent, _ = agent.update(batch)
    after_target = new_agent.network.params["modules_target_terminal_q"]

    tau = float(agent.config["terminal_q_tau"])
    expected = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1.0 - tau),
        before_online,
        before_target,
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(after_target),
        jax.tree_util.tree_leaves(expected),
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-5)


def test_phase_restore_preserves_step(warmstart_agent):
    batch = _synthetic_batch(batch_size=warmstart_agent.config["batch_size"], seed=15)
    trained, _ = warmstart_agent.update(batch)
    with tempfile.TemporaryDirectory() as tmp:
        save_agent(trained, tmp, epoch=2000000)
        obs, actions = _ex_obs_actions()
        template = QFlowRQLWarmstartAgent.create(
            0, obs, actions, _tiny_config(training_phase="qflow_online")
        )
        restored = restore_agent(template, tmp, 2000000)

    assert restored.config["training_phase"] == "qflow_online"
    assert int(restored.network.step) == int(trained.network.step)
    for a, b in zip(
        jax.tree_util.tree_leaves(trained.network.params),
        jax.tree_util.tree_leaves(restored.network.params),
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    batch2 = _synthetic_batch(batch_size=restored.config["batch_size"], seed=16)
    restored2, info = restored.update(batch2)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert np.isfinite(float(np.asarray(info["inner_grad_to_cfm_ratio"])))
    assert int(restored2.network.step) == int(restored.network.step) + 1


def test_actor_coef0_freezes_actor(online_agent):
    obs, actions = _ex_obs_actions()
    agent = QFlowRQLWarmstartAgent.create(
        3,
        obs,
        actions,
        _tiny_config(training_phase="qflow_online", qflow_actor_coef=0.0),
    )
    batch = _synthetic_batch(batch_size=agent.config["batch_size"], seed=21)
    before_actor = agent.network.params["modules_actor"]
    before_tq = agent.network.params["modules_terminal_q"]
    before_iv = agent.network.params["modules_inner_value"]
    new_agent, info = agent.update(batch)

    for a, b in zip(
        jax.tree_util.tree_leaves(before_actor),
        jax.tree_util.tree_leaves(new_agent.network.params["modules_actor"]),
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    assert _flat_norm(
        jax.tree_util.tree_map(
            lambda a, b: a - b,
            before_tq,
            new_agent.network.params["modules_terminal_q"],
        )
    ) > 0.0
    assert _flat_norm(
        jax.tree_util.tree_map(
            lambda a, b: a - b,
            before_iv,
            new_agent.network.params["modules_inner_value"],
        )
    ) > 0.0
    assert float(np.asarray(info["actor_coef"])) == pytest.approx(0.0)

    rng = jax.random.PRNGKey(22)

    def total_only(params):
        loss, _ = agent.qflow_policy_loss(
            batch, params, rng, blend=1.0, actor_coef=0.0
        )
        return loss

    grads = jax.grad(total_only)(agent.network.params)
    assert _flat_norm(grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(grads["modules_terminal_q"]) > 0.0
    assert _flat_norm(grads["modules_inner_value"]) > 0.0


def test_finite_updates_and_sampling(warmstart_agent, bridge_agent, online_agent):
    for agent, seed in (
        (warmstart_agent, 40),
        (bridge_agent, 41),
        (online_agent, 42),
    ):
        batch = _synthetic_batch(batch_size=agent.config["batch_size"], seed=seed)
        new_agent, info = agent.update(batch)
        assert np.isfinite(float(np.asarray(info["total_loss"])))
        assert np.isfinite(float(np.asarray(info["terminal_q_action_grad_norm"])))
        if agent.config["training_phase"] != "rql_warmstart":
            for key in (
                "cfm_target_norm",
                "inner_grad_norm",
                "inner_grad_to_cfm_ratio",
                "pred_velocity_norm",
                "actor_coef",
                "qflow_bridge_blend",
            ):
                assert np.isfinite(float(np.asarray(info[key])))
        assert int(new_agent.network.step) == int(agent.network.step) + 1

        obs = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
        actions = agent.sample_actions(obs, seed=jax.random.PRNGKey(seed))
        assert actions.shape == (H, PRIM_DIM)
        assert np.isfinite(np.asarray(actions)).all()


def test_replace_training_phase(warmstart_agent):
    online = warmstart_agent.replace_training_phase("qflow_online")
    assert online.config["training_phase"] == "qflow_online"
    bridge = warmstart_agent.replace_training_phase("qflow_bridge")
    assert bridge.config["training_phase"] == "qflow_bridge"
    with pytest.raises(ValueError):
        warmstart_agent.replace_training_phase("rql_offline")
