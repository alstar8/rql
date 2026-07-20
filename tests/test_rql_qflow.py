"""Focused unit tests for RQLQFlowAgent (offline isolation + online Q-Flow)."""

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

from agents.rql import RQLAgent  # noqa: E402
from agents.rql_qflow import RQLQFlowAgent, get_config  # noqa: E402
from utils.datasets import ReplayBuffer  # noqa: E402
from utils.flax_utils import restore_agent, save_agent  # noqa: E402


def rearrange_actions(batch, h):
    return rearrange(batch["actions"][:h], "h b d -> b (h d)")

OBS_DIM = 8
PRIM_DIM = 4
H = 1


def _tiny_config(**overrides):
    cfg = dict(get_config())
    cfg.update(
        {
            "h": H,
            "batch_size": 8,
            "ensemble_ct": 2,
            "inner_ensemble_ct": 1,
            "flow_steps": 4,
            "actor_hidden_dims": (32, 32),
            "value_hidden_dims": (32, 32),
            "inner_value_hidden_dims": (32, 32),
            "alpha": 0.3,
            "expectile": 0.5,
            "discount": 0.995,
            "q_agg": "mean",
            "qflow_lambda": 1.0,
            "training_phase": "rql_offline",
            "tau": 0.005,
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
def offline_agent():
    obs, actions = _ex_obs_actions()
    return RQLQFlowAgent.create(0, obs, actions, _tiny_config())


@pytest.fixture(scope="module")
def online_agent():
    obs, actions = _ex_obs_actions()
    return RQLQFlowAgent.create(
        1, obs, actions, _tiny_config(training_phase="qflow_online")
    )


def test_config_defaults():
    cfg = get_config()
    assert cfg.agent_name == "rql_qflow"
    assert cfg.training_phase == "rql_offline"
    assert cfg.qflow_lambda == 1.0
    assert cfg.qflow_actor_coef == 1.0
    assert cfg.q_agg == "mean"
    assert cfg.tau == 0.005
    assert tuple(cfg.inner_value_hidden_dims) == (512, 512, 512, 512)


def test_registry_wiring():
    from agents import agents

    assert "rql_qflow" in agents
    assert agents["rql_qflow"] is RQLQFlowAgent


def test_module_tree_has_inner_value(offline_agent):
    keys = set(offline_agent.network.params.keys())
    for name in (
        "modules_actor",
        "modules_value",
        "modules_target_actor",
        "modules_target_value",
        "modules_inner_value",
    ):
        assert name in keys


def test_phase_switch_helper(offline_agent):
    online = offline_agent.replace_training_phase("qflow_online")
    assert offline_agent.config["training_phase"] == "rql_offline"
    assert online.config["training_phase"] == "qflow_online"
    # Shared pytree params identical after phase-only replace.
    for a, b in zip(
        jax.tree_util.tree_leaves(offline_agent.network.params),
        jax.tree_util.tree_leaves(online.network.params),
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_offline_rql_loss_matches_vanilla_rql():
    """Shared RQL actor/critic loss matches RQLAgent under copied params."""
    obs, actions = _ex_obs_actions()
    cfg_q = _tiny_config(training_phase="rql_offline", batch_size=8)
    cfg_r = dict(cfg_q)
    cfg_r["agent_name"] = "rql"
    # RQL get_config keys only.
    cfg_r.pop("training_phase", None)
    cfg_r.pop("qflow_lambda", None)
    cfg_r.pop("qflow_actor_coef", None)
    cfg_r.pop("inner_value_hidden_dims", None)
    cfg_r.pop("inner_ensemble_ct", None)

    q_agent = RQLQFlowAgent.create(0, obs, actions, cfg_q)
    r_agent = RQLAgent.create(0, obs, actions, cfg_r)

    # Copy shared modules from q_agent into r_agent so losses are comparable.
    r_params = r_agent.network.params
    for key in (
        "modules_actor",
        "modules_value",
        "modules_target_actor",
        "modules_target_value",
    ):
        r_params[key] = q_agent.network.params[key]
    r_agent = r_agent.replace(
        network=r_agent.network.replace(params=r_params),
        rng=q_agent.rng,
    )

    batch = _synthetic_batch(batch_size=8, seed=3)
    rng = jax.random.PRNGKey(11)

    rql_loss, rql_info, _ = q_agent.rql_actor_critic_loss(
        batch, q_agent.network.params, rng
    )
    vanilla_loss, vanilla_info = r_agent.total_loss(
        batch, r_agent.network.params, rng=rng
    )

    np.testing.assert_allclose(
        float(rql_loss), float(vanilla_loss), rtol=1e-5, atol=1e-5
    )
    for key in ("actor_loss", "bc_loss", "critic_loss"):
        np.testing.assert_allclose(
            float(rql_info[key]),
            float(vanilla_info[key]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_offline_inner_value_zero_cross_grad(offline_agent):
    """Inner-value loss does not put grads into actor/critic; RQL loss skips V."""
    batch = _synthetic_batch(batch_size=offline_agent.config["batch_size"], seed=5)
    observations = batch["observations"][0]
    actions = rearrange_actions(batch, offline_agent.config["h"])
    rng = jax.random.PRNGKey(7)

    def iv_only(params):
        loss, _ = offline_agent.inner_value_loss(
            observations, actions, params, rng, rollout_module="target_actor"
        )
        return loss

    iv_grads = jax.grad(iv_only)(offline_agent.network.params)
    assert _flat_norm(iv_grads["modules_inner_value"]) > 0.0
    assert _flat_norm(iv_grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_value"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_target_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_target_value"]) == pytest.approx(0.0, abs=1e-8)

    def rql_only(params):
        loss, _, _ = offline_agent.rql_actor_critic_loss(batch, params, rng)
        return loss

    rql_grads = jax.grad(rql_only)(offline_agent.network.params)
    assert _flat_norm(rql_grads["modules_actor"]) > 0.0
    assert _flat_norm(rql_grads["modules_value"]) > 0.0
    assert _flat_norm(rql_grads["modules_inner_value"]) == pytest.approx(0.0, abs=1e-8)


def test_offline_update_finite(offline_agent):
    batch = _synthetic_batch(batch_size=offline_agent.config["batch_size"], seed=9)
    new_agent, info = offline_agent.update(batch)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert np.isfinite(float(np.asarray(info["inner_value_loss"])))
    assert np.isfinite(float(np.asarray(info["critic_loss"])))
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    assert int(new_agent.network.step) == int(offline_agent.network.step) + 1


def test_online_update_finite(online_agent):
    batch = _synthetic_batch(batch_size=online_agent.config["batch_size"], seed=10)
    new_agent, info = online_agent.update(batch)
    assert online_agent.config["training_phase"] == "qflow_online"
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert np.isfinite(float(np.asarray(info["inner_value_loss"])))
    assert np.isfinite(float(np.asarray(info["critic_loss"])))
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    for key in (
        "cfm_target_norm",
        "inner_grad_norm",
        "inner_grad_to_cfm_ratio",
        "pred_velocity_norm",
        "actor_coef",
    ):
        assert np.isfinite(float(np.asarray(info[key])))
    assert float(np.asarray(info["actor_coef"])) == pytest.approx(1.0)
    assert int(new_agent.network.step) == int(online_agent.network.step) + 1


def test_qflow_actor_coef0_freezes_actor_target_updates_critic_inner():
    """coef=0: actor/target_actor frozen; critic + inner_value update; diags finite."""
    obs, actions = _ex_obs_actions()
    agent = RQLQFlowAgent.create(
        2,
        obs,
        actions,
        _tiny_config(training_phase="qflow_online", qflow_actor_coef=0.0),
    )
    # Diverge target_actor from online so skipping EMA is observable.
    params = dict(agent.network.params)
    params["modules_target_actor"] = jax.tree_util.tree_map(
        lambda x: x + 0.05, params["modules_target_actor"]
    )
    agent = agent.replace(network=agent.network.replace(params=params))

    batch = _synthetic_batch(batch_size=agent.config["batch_size"], seed=21)
    before = agent.network.params
    new_agent, info = agent.update(batch)
    after = new_agent.network.params

    def _tree_equal(a, b):
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    _tree_equal(before["modules_actor"], after["modules_actor"])
    _tree_equal(before["modules_target_actor"], after["modules_target_actor"])
    assert _flat_norm(
        jax.tree_util.tree_map(
            lambda a, b: a - b,
            before["modules_value"],
            after["modules_value"],
        )
    ) > 0.0
    assert _flat_norm(
        jax.tree_util.tree_map(
            lambda a, b: a - b,
            before["modules_inner_value"],
            after["modules_inner_value"],
        )
    ) > 0.0

    assert float(np.asarray(info["actor_coef"])) == pytest.approx(0.0)
    for key in (
        "cfm_target_norm",
        "inner_grad_norm",
        "inner_grad_to_cfm_ratio",
        "pred_velocity_norm",
        "actor_coef",
        "actor_loss",
        "critic_loss",
        "inner_value_loss",
        "total_loss",
    ):
        assert np.isfinite(float(np.asarray(info[key])))

    # Gradients into actor must be exactly zero under coef=0 total loss.
    rng = jax.random.PRNGKey(22)

    def total_only(params):
        loss, _ = agent.qflow_online_loss(batch, params, rng)
        return loss

    grads = jax.grad(total_only)(agent.network.params)
    assert _flat_norm(grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(grads["modules_value"]) > 0.0
    assert _flat_norm(grads["modules_inner_value"]) > 0.0


def test_online_actor_target_stopgrad_no_inner_bptt(online_agent):
    """Velocity target uses stop-grad; actor grads must not hit inner_value."""
    batch = _synthetic_batch(batch_size=online_agent.config["batch_size"], seed=12)
    rng = jax.random.PRNGKey(13)

    def actor_part(params):
        # Isolate actor velocity loss by recomputing online loss pieces.
        loss, info = online_agent.qflow_online_loss(batch, params, rng)
        return info["actor_loss"]

    grads = jax.grad(actor_part)(online_agent.network.params)
    assert _flat_norm(grads["modules_actor"]) > 0.0
    assert _flat_norm(grads["modules_inner_value"]) == pytest.approx(0.0, abs=1e-7)
    assert _flat_norm(grads["modules_value"]) == pytest.approx(0.0, abs=1e-7)


def test_roll_to_terminal_from_arbitrary_t(offline_agent):
    b, d = 4, offline_agent.config["action_dim"]
    obs = jax.random.normal(jax.random.PRNGKey(1), (b, OBS_DIM))
    x = jax.random.normal(jax.random.PRNGKey(2), (b, d))
    t = jnp.array([[0.0], [0.25], [0.5], [0.9]], dtype=jnp.float32)
    out = offline_agent.roll_flow_to_terminal(obs, x, t, module_name="target_actor")
    assert out.shape == (b, d)
    assert np.isfinite(np.asarray(out)).all()
    # From t≈1, remaining horizon is tiny → nearly identity.
    t1 = jnp.ones((b, 1), dtype=jnp.float32)
    out1 = offline_agent.roll_flow_to_terminal(obs, x, t1, module_name="target_actor")
    np.testing.assert_allclose(np.asarray(out1), np.asarray(x), atol=1e-5)


def test_sample_actions_shape(offline_agent):
    obs = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    actions = offline_agent.sample_actions(
        obs, seed=jax.random.PRNGKey(0), temperature=0.0
    )
    assert actions.shape == (H, PRIM_DIM)


def test_checkpoint_restore_with_changed_phase(offline_agent):
    batch = _synthetic_batch(batch_size=offline_agent.config["batch_size"], seed=15)
    trained, _ = offline_agent.update(batch)
    with tempfile.TemporaryDirectory() as tmp:
        save_agent(trained, tmp, epoch=2000000)
        obs, actions = _ex_obs_actions()
        template = RQLQFlowAgent.create(
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

    # Online update after restore stays finite.
    batch2 = _synthetic_batch(batch_size=restored.config["batch_size"], seed=16)
    restored2, info = restored.update(batch2)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert int(restored2.network.step) == int(restored.network.step) + 1


def test_replay_buffer_grows_and_samples():
    """Online-style append increases size; samples remain finite-shaped."""
    n0 = 32
    obs = np.zeros((n0, OBS_DIM), dtype=np.float32)
    actions = np.zeros((n0, PRIM_DIM), dtype=np.float32)
    rewards = np.zeros((n0,), dtype=np.float32)
    terminals = np.zeros((n0,), dtype=np.float32)
    terminals[-1] = 1.0
    masks = np.ones((n0,), dtype=np.float32)
    next_obs = np.zeros((n0, OBS_DIM), dtype=np.float32)
    init = {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "masks": masks,
        "next_observations": next_obs,
    }
    buf = ReplayBuffer.create_from_initial_dataset(init, size=64)
    buf.config = {"h": 1}
    assert buf.size == n0

    for i in range(5):
        term = 1.0 if i == 4 else 0.0
        buf.add_transition(
            {
                "observations": np.zeros((OBS_DIM,), dtype=np.float32),
                "actions": np.zeros((PRIM_DIM,), dtype=np.float32),
                "rewards": np.array(0.0, dtype=np.float32),
                "terminals": np.array(term, dtype=np.float32),
                "masks": np.array(1.0 - term, dtype=np.float32),
                "next_observations": np.zeros((OBS_DIM,), dtype=np.float32),
            }
        )
    assert buf.size == n0 + 5
    batch = buf.sample(4)
    assert batch["observations"].shape[0] in (2, H + 1)  # traj layout when h set
    assert batch["actions"].shape[-1] == PRIM_DIM
