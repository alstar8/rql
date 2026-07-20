"""Focused unit tests for RQLQFlowTerminalAgent."""

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
from agents.rql_qflow_terminal import RQLQFlowTerminalAgent, get_config  # noqa: E402
from utils.flax_utils import restore_agent, save_agent  # noqa: E402


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
def offline_agent():
    obs, actions = _ex_obs_actions()
    return RQLQFlowTerminalAgent.create(0, obs, actions, _tiny_config())


@pytest.fixture(scope="module")
def online_agent():
    obs, actions = _ex_obs_actions()
    return RQLQFlowTerminalAgent.create(
        1, obs, actions, _tiny_config(training_phase="qflow_online")
    )


def test_config_defaults():
    cfg = get_config()
    assert cfg.agent_name == "rql_qflow_terminal"
    assert cfg.training_phase == "rql_offline"
    assert cfg.terminal_q_tau == 0.005
    assert cfg.ensemble_ct == 10
    assert tuple(cfg.value_hidden_dims) == (512, 512, 512, 512)


def test_registry_wiring():
    from agents import agents

    assert "rql_qflow_terminal" in agents
    assert agents["rql_qflow_terminal"] is RQLQFlowTerminalAgent


def test_module_tree_has_terminal_q(offline_agent):
    keys = set(offline_agent.network.params.keys())
    for name in (
        "modules_actor",
        "modules_value",
        "modules_target_actor",
        "modules_target_value",
        "modules_inner_value",
        "modules_terminal_q",
        "modules_target_terminal_q",
    ):
        assert name in keys


def test_offline_rql_actor_value_matches_vanilla_and_isolated():
    """RQL actor/value loss matches vanilla RQL; aux grads skip actor/value."""
    obs, actions = _ex_obs_actions()
    cfg_q = _tiny_config(training_phase="rql_offline", batch_size=8)
    cfg_r = dict(cfg_q)
    cfg_r["agent_name"] = "rql"
    for key in (
        "training_phase",
        "qflow_lambda",
        "qflow_actor_coef",
        "inner_value_hidden_dims",
        "inner_ensemble_ct",
        "terminal_q_tau",
    ):
        cfg_r.pop(key, None)

    q_agent = RQLQFlowTerminalAgent.create(0, obs, actions, cfg_q)
    r_agent = RQLAgent.create(0, obs, actions, cfg_r)

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

    observations = batch["observations"][0]
    flat_actions = rearrange_actions(batch, q_agent.config["h"])

    def tq_only(params):
        loss, _, _ = q_agent.terminal_q_bellman_loss(batch, params, rng)
        return loss

    tq_grads = jax.grad(tq_only)(q_agent.network.params)
    assert _flat_norm(tq_grads["modules_terminal_q"]) > 0.0
    assert _flat_norm(tq_grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(tq_grads["modules_value"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(tq_grads["modules_inner_value"]) == pytest.approx(0.0, abs=1e-8)

    def iv_only(params):
        loss, _ = q_agent.inner_value_loss(
            observations, flat_actions, params, rng, rollout_module="target_actor"
        )
        return loss

    iv_grads = jax.grad(iv_only)(q_agent.network.params)
    assert _flat_norm(iv_grads["modules_inner_value"]) > 0.0
    assert _flat_norm(iv_grads["modules_actor"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_value"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_terminal_q"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(iv_grads["modules_target_terminal_q"]) == pytest.approx(
        0.0, abs=1e-8
    )

    def rql_only(params):
        loss, _, _ = q_agent.rql_actor_critic_loss(batch, params, rng)
        return loss

    rql_grads = jax.grad(rql_only)(q_agent.network.params)
    assert _flat_norm(rql_grads["modules_actor"]) > 0.0
    assert _flat_norm(rql_grads["modules_value"]) > 0.0
    assert _flat_norm(rql_grads["modules_terminal_q"]) == pytest.approx(0.0, abs=1e-8)
    assert _flat_norm(rql_grads["modules_inner_value"]) == pytest.approx(0.0, abs=1e-8)


def test_terminal_q_updates_and_action_sensitivity_finite(offline_agent):
    batch = _synthetic_batch(batch_size=offline_agent.config["batch_size"], seed=9)
    before = offline_agent.network.params["modules_terminal_q"]
    new_agent, info = offline_agent.update(batch)

    assert np.isfinite(float(np.asarray(info["terminal_q_loss"])))
    assert np.isfinite(float(np.asarray(info["terminal_q_mean"])))
    assert np.isfinite(float(np.asarray(info["terminal_q_action_grad_norm"])))
    assert float(np.asarray(info["terminal_q_action_grad_norm"])) > 0.0

    delta = _flat_norm(
        jax.tree_util.tree_map(
            lambda a, b: a - b,
            before,
            new_agent.network.params["modules_terminal_q"],
        )
    )
    assert delta > 0.0


def test_inner_target_uses_terminal_head(offline_agent):
    """Perturbing target_terminal_q changes inner target; target_value does not."""
    batch = _synthetic_batch(batch_size=offline_agent.config["batch_size"], seed=17)
    observations = batch["observations"][0]
    actions = rearrange_actions(batch, offline_agent.config["h"])
    rng = jax.random.PRNGKey(18)

    _, base_info = offline_agent.inner_value_loss(
        observations, actions, offline_agent.network.params, rng
    )
    base_target = float(np.asarray(base_info["inner_target_q_mean"]))

    params_tq = dict(offline_agent.network.params)
    params_tq["modules_target_terminal_q"] = jax.tree_util.tree_map(
        lambda x: x + 0.5, params_tq["modules_target_terminal_q"]
    )
    agent_tq = offline_agent.replace(
        network=offline_agent.network.replace(params=params_tq)
    )
    _, tq_info = agent_tq.inner_value_loss(
        observations, actions, agent_tq.network.params, rng
    )
    assert float(np.asarray(tq_info["inner_target_q_mean"])) != pytest.approx(
        base_target, abs=1e-5
    )

    params_v = dict(offline_agent.network.params)
    params_v["modules_target_value"] = jax.tree_util.tree_map(
        lambda x: x + 0.5, params_v["modules_target_value"]
    )
    agent_v = offline_agent.replace(
        network=offline_agent.network.replace(params=params_v)
    )
    _, v_info = agent_v.inner_value_loss(
        observations, actions, agent_v.network.params, rng
    )
    np.testing.assert_allclose(
        float(np.asarray(v_info["inner_target_q_mean"])),
        base_target,
        rtol=1e-5,
        atol=1e-5,
    )


def test_terminal_target_polyak_update(offline_agent):
    batch = _synthetic_batch(batch_size=offline_agent.config["batch_size"], seed=19)
    # Diverge target from online so EMA is observable.
    params = dict(offline_agent.network.params)
    params["modules_target_terminal_q"] = jax.tree_util.tree_map(
        lambda x: x + 1.0, params["modules_target_terminal_q"]
    )
    agent = offline_agent.replace(
        network=offline_agent.network.replace(params=params)
    )
    before_online = agent.network.params["modules_terminal_q"]
    before_target = agent.network.params["modules_target_terminal_q"]
    new_agent, _ = agent.update(batch)
    after_online = new_agent.network.params["modules_terminal_q"]
    after_target = new_agent.network.params["modules_target_terminal_q"]

    # target_update mixes pre-update online params with pre-update target.
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

    assert _flat_norm(
        jax.tree_util.tree_map(lambda a, b: a - b, before_online, after_online)
    ) > 0.0
    assert _flat_norm(
        jax.tree_util.tree_map(lambda a, b: a - b, before_target, after_target)
    ) > 0.0


def test_checkpoint_phase_restore(offline_agent):
    batch = _synthetic_batch(batch_size=offline_agent.config["batch_size"], seed=15)
    trained, _ = offline_agent.update(batch)
    with tempfile.TemporaryDirectory() as tmp:
        save_agent(trained, tmp, epoch=2000000)
        obs, actions = _ex_obs_actions()
        template = RQLQFlowTerminalAgent.create(
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
    assert np.isfinite(float(np.asarray(info["terminal_q_loss"])))
    assert np.isfinite(float(np.asarray(info["inner_grad_to_cfm_ratio"])))
    assert int(restored2.network.step) == int(restored.network.step) + 1


def test_online_update_finite(online_agent):
    batch = _synthetic_batch(batch_size=online_agent.config["batch_size"], seed=10)
    new_agent, info = online_agent.update(batch)
    assert online_agent.config["training_phase"] == "qflow_online"
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert np.isfinite(float(np.asarray(info["terminal_q_loss"])))
    assert np.isfinite(float(np.asarray(info["terminal_q_mean"])))
    assert np.isfinite(float(np.asarray(info["terminal_q_action_grad_norm"])))
    assert np.isfinite(float(np.asarray(info["inner_value_loss"])))
    assert np.isfinite(float(np.asarray(info["actor_loss"])))
    assert np.isfinite(float(np.asarray(info["inner_grad_to_cfm_ratio"])))
    assert int(new_agent.network.step) == int(online_agent.network.step) + 1
