"""Focused unit tests for pure QFlowAgent (paper Algorithm 1)."""

from __future__ import annotations

import inspect
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

from agents.qflow import (  # noqa: E402
    DeterministicVectorField,
    FourierFeatures,
    FourierValue,
    QFlowAgent,
    fourier_time_features,
    get_config,
)
from utils.flax_utils import restore_agent, save_agent  # noqa: E402
from utils.networks import Value  # noqa: E402


OBS_DIM = 8
PRIM_DIM = 4
H = 1

FORBIDDEN_CONFIG_KEYS = (
    "alpha",
    "expectile",
    "rho",
    "ema",
    "tau",
    "qflow_actor_coef",
    "qflow_bridge_blend",
    "rql_ensemble_ct",
    "training_phase",
)
FORBIDDEN_MODULES = (
    "modules_value",
    "modules_target_value",
    "modules_target_actor",
)
# Code-body patterns that must not appear (docstring may mention absences).
FORBIDDEN_CODE_PATTERNS = (
    "def expectile_loss",
    "q_pe",
    'select("value")',
    'select("target_value")',
    'select("target_actor")',
    "modules_target_actor",
    "modules_value",
    "rql_warmstart",
    "qflow_bridge",
    "qflow_actor_coef",
    "qflow_bridge_blend",
)


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
            "flow_steps": 4,
            "actor_hidden_dims": (32, 32),
            "value_hidden_dims": (32, 32),
            "inner_value_hidden_dims": (32, 32),
            "actor_fourier_dim": 8,
            "inner_fourier_dim": 8,
            "discount": 0.995,
            "q_agg": "mean",
            "qflow_lambda": 1.0,
            "terminal_q_tau": 0.005,
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
def agent():
    obs, actions = _ex_obs_actions()
    return QFlowAgent.create(0, obs, actions, _tiny_config())


def test_config_defaults_and_architecture_absence():
    cfg = get_config()
    assert cfg.agent_name == "qflow"
    assert cfg.h == 1
    assert cfg.batch_size == 256
    assert cfg.lr == 3e-4
    assert cfg.discount == 0.995
    assert cfg.qflow_lambda == 1.0
    assert cfg.q_agg == "mean"
    assert cfg.ensemble_ct == 2
    assert cfg.inner_ensemble_ct == 2
    assert cfg.inner_fourier_dim == 16
    assert cfg.flow_steps == 10
    assert cfg.terminal_q_tau == 0.005
    assert tuple(cfg.actor_hidden_dims) == (512, 512, 512, 512)
    for key in FORBIDDEN_CONFIG_KEYS:
        assert key not in cfg, f"unexpected config key {key}"

    src = inspect.getsource(QFlowAgent)
    # Strip the leading class docstring so absence mentions do not false-positive.
    body = src.split('"""', 2)[-1] if '"""' in src else src
    for token in FORBIDDEN_CODE_PATTERNS:
        assert token not in body, f"forbidden pattern {token!r} in QFlowAgent"


def test_registry_wiring():
    from agents import agents

    assert "qflow" in agents
    assert agents["qflow"] is QFlowAgent


def test_fourier_and_timefree_contracts():
    """Fourier features are even-dim; outer Q is time-free; VF is deterministic."""
    freqs = jnp.array([1.0, 2.0, 4.0, 8.0], dtype=jnp.float32)
    t = jnp.array([[0.0], [0.25], [0.5], [1.0]], dtype=jnp.float32)
    feats = fourier_time_features(t, freqs)
    assert feats.shape == (4, 8)
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
    assert not hasattr(vf, "mode")

    q_mod = Value(hidden_dims=(16, 16), num_ensembles=2, layer_norm=True)
    vars_q = q_mod.init(key, obs, act)
    qs = q_mod.apply(vars_q, obs, act)
    assert qs.shape == (2, 2)

    iv = FourierValue(
        hidden_dims=(16, 16), fourier_dim=8, num_ensembles=2, layer_norm=True
    )
    vars_iv = iv.init(key, obs, act, times)
    vs = iv.apply(vars_iv, obs, act, times)
    assert vs.shape == (2, 2)


def test_module_tree_absence(agent):
    keys = set(agent.network.params.keys())
    for name in (
        "modules_actor",
        "modules_terminal_q",
        "modules_target_terminal_q",
        "modules_inner_value",
    ):
        assert name in keys
    for name in FORBIDDEN_MODULES:
        assert name not in keys
    assert "time_features" not in agent.network.params["modules_terminal_q"]
    actor_leaves = agent.network.params["modules_actor"]
    assert "mlp" in actor_leaves
    assert "mean_net" not in actor_leaves
    assert "log_std_net" not in actor_leaves
    assert "log_stds" not in actor_leaves
    # FourierFeatures frequencies are fixed (non-trainable) so they do not
    # appear under params; modules are still DeterministicVectorField / FourierValue.
    mods = agent.network.model_def.modules
    assert type(mods["actor"]).__name__ == "DeterministicVectorField"
    assert type(mods["inner_value"]).__name__ == "FourierValue"
    assert type(mods["terminal_q"]).__name__ == "Value"


def test_exact_loss_decomposition(agent):
    """total = critic + inner V + actor matching."""
    batch = _synthetic_batch(batch_size=agent.config["batch_size"], seed=20)
    rng = jax.random.PRNGKey(21)
    loss, info = agent.total_loss(batch, agent.network.params, rng)
    expected = (
        float(info["critic_loss"])
        + float(info["inner_value_loss"])
        + float(info["actor_loss"])
    )
    np.testing.assert_allclose(float(loss), expected, rtol=1e-5, atol=1e-5)
    for forbidden in ("q_pe_mean", "qflow_bridge_blend", "actor_coef", "rql_loss"):
        assert forbidden not in info


def test_only_target_q_polyak(agent):
    batch = _synthetic_batch(batch_size=agent.config["batch_size"], seed=19)
    params = dict(agent.network.params)
    params["modules_target_terminal_q"] = jax.tree_util.tree_map(
        lambda x: x + 1.0, params["modules_target_terminal_q"]
    )
    agent_shifted = agent.replace(network=agent.network.replace(params=params))
    before_online = agent_shifted.network.params["modules_terminal_q"]
    before_target = agent_shifted.network.params["modules_target_terminal_q"]
    before_actor = agent_shifted.network.params["modules_actor"]
    before_iv = agent_shifted.network.params["modules_inner_value"]

    new_agent, _ = agent_shifted.update(batch)
    after_target = new_agent.network.params["modules_target_terminal_q"]

    tau = float(agent_shifted.config["terminal_q_tau"])
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

    # No other Polyak/EMA targets exist or change via a second target module.
    assert "modules_target_actor" not in new_agent.network.params
    assert "modules_target_value" not in new_agent.network.params
    # Actor / inner V are optimized (not Polyak copies).
    assert _flat_norm(
        jax.tree_util.tree_map(
            lambda a, b: a - b,
            before_actor,
            new_agent.network.params["modules_actor"],
        )
    ) > 0.0
    assert _flat_norm(
        jax.tree_util.tree_map(
            lambda a, b: a - b,
            before_iv,
            new_agent.network.params["modules_inner_value"],
        )
    ) > 0.0


def test_current_policy_rollout_no_bptt(agent):
    """Rollout uses current actor; no BPTT into actor for V."""
    b, d = 4, agent.config["action_dim"]
    obs = jax.random.normal(jax.random.PRNGKey(1), (b, OBS_DIM))
    x = jax.random.normal(jax.random.PRNGKey(2), (b, d))
    t = jnp.array([[0.0], [0.25], [0.5], [0.9]], dtype=jnp.float32)
    out = agent.roll_flow_to_terminal(obs, x, t, params=None)
    assert out.shape == (b, d)
    assert np.isfinite(np.asarray(out)).all()
    t1 = jnp.ones((b, 1), dtype=jnp.float32)
    out1 = agent.roll_flow_to_terminal(obs, x, t1, params=None)
    np.testing.assert_allclose(np.asarray(out1), np.asarray(x), atol=1e-5)

    batch = _synthetic_batch(batch_size=agent.config["batch_size"], seed=12)
    observations = batch["observations"][0]
    actions = rearrange_actions(batch, agent.config["h"])
    rng = jax.random.PRNGKey(13)

    def iv_only(params):
        loss, _ = agent.inner_value_loss(observations, actions, params, rng)
        return loss

    iv_grads = jax.grad(iv_only)(agent.network.params)
    assert _flat_norm(iv_grads["modules_inner_value"]) > 0.0
    assert _flat_norm(iv_grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_target_terminal_q"]) == pytest.approx(
        0.0, abs=1e-8
    )

    def actor_part(params):
        _, info = agent.total_loss(batch, params, rng)
        return info["actor_loss"]

    actor_grads = jax.grad(actor_part)(agent.network.params)
    assert _flat_norm(actor_grads["modules_actor"]) > 0.0
    assert _flat_norm(actor_grads["modules_inner_value"]) == pytest.approx(
        0.0, abs=1e-7
    )


def test_checkpoint_offline_online_same_tree(agent):
    """Offline→online restore keeps identical TrainState tree (no phase switch)."""
    batch = _synthetic_batch(batch_size=agent.config["batch_size"], seed=15)
    trained, _ = agent.update(batch)
    with tempfile.TemporaryDirectory() as tmp:
        save_agent(trained, tmp, epoch=1_000_000)
        obs, actions = _ex_obs_actions()
        # Same create() tree for "online" continuation (no training_phase).
        template = QFlowAgent.create(0, obs, actions, _tiny_config())
        restored = restore_agent(template, tmp, 1_000_000)

    assert set(restored.network.params.keys()) == set(trained.network.params.keys())
    assert "training_phase" not in restored.config
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


def test_finite_update_and_sampling(agent):
    batch = _synthetic_batch(batch_size=agent.config["batch_size"], seed=40)
    new_agent, info = agent.update(batch)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    for key in (
        "critic_loss",
        "inner_value_loss",
        "actor_loss",
        "terminal_q_action_grad_norm",
        "cfm_target_norm",
        "inner_grad_norm",
        "inner_grad_to_cfm_ratio",
        "pred_velocity_norm",
    ):
        assert np.isfinite(float(np.asarray(info[key])))
    assert int(new_agent.network.step) == int(agent.network.step) + 1

    obs = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    actions = agent.sample_actions(obs, seed=jax.random.PRNGKey(41))
    assert actions.shape == (H, PRIM_DIM)
    assert np.isfinite(np.asarray(actions)).all()


def test_create_ignores_training_phase():
    obs, actions = _ex_obs_actions()
    agent = QFlowAgent.create(
        7, obs, actions, _tiny_config(training_phase="should_be_dropped")
    )
    assert "training_phase" not in agent.config
