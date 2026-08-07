"""Tests for the π₀.₅ + CF fine-tune scaffold."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import agents as agent_registry  # noqa: E402
from agents.pi05_cf import Pi05CFAgent, get_config as get_pi05_cf_config  # noqa: E402

OBS_DIM = 7
ACTION_DIM = 3
BATCH_SIZE = 4


def _tiny_config(**overrides):
    config = dict(get_pi05_cf_config())
    config.update(
        {
            "h": 1,
            "batch_size": BATCH_SIZE,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (16, 16),
            "lora_hidden_dims": (16,),
            "lora_rank": 4,
            "value_hidden_dims": (16, 16),
            "refiner_hidden_dims": (16, 16),
            "actor_delay": 1,
            "target_policy_noise": 0.0,
            "cql_n_actions": 2,
            "train_phase": 2,
        }
    )
    config.update(overrides)
    return config


def _create_agent(**overrides):
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    return Pi05CFAgent.create(
        0,
        observations,
        actions,
        _tiny_config(**overrides),
    )


def _batch(*, with_successes: bool = False):
    batch = {
        "observations": jax.random.normal(
            jax.random.PRNGKey(1),
            (2, BATCH_SIZE, OBS_DIM),
        ),
        "actions": jnp.clip(
            jax.random.normal(
                jax.random.PRNGKey(2),
                (1, BATCH_SIZE, ACTION_DIM),
            ),
            -1,
            1,
        ),
        "rewards": jax.random.normal(
            jax.random.PRNGKey(3),
            (2, BATCH_SIZE),
        ),
        "terminals": jnp.zeros((2, BATCH_SIZE), dtype=jnp.float32),
        "masks": jnp.ones((2, BATCH_SIZE), dtype=jnp.float32),
    }
    if with_successes:
        batch["successes"] = jnp.asarray([1.0, 0.0, 1.0, 0.0], dtype=jnp.float32)
    return batch


def _tree_l1(tree) -> float:
    return sum(
        float(np.abs(np.asarray(leaf)).sum())
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_delta_l1(left, right) -> float:
    return _tree_l1(jax.tree_util.tree_map(lambda a, b: a - b, left, right))


def test_registry_and_defaults():
    config = get_pi05_cf_config()
    assert agent_registry["pi05_cf"] is Pi05CFAgent
    assert config.freeze_base_actor is True
    assert config.use_lora is True
    assert config.zero_init_refiner is True
    assert config.cql_coef > 0.0
    assert config.train_phase in (1, 2, 3)


def test_zero_refiner_matches_flow_policy():
    agent = _create_agent()
    seed = jax.random.PRNGKey(7)
    _, noise_rng = jax.random.split(seed)
    obs = jnp.arange(OBS_DIM, dtype=jnp.float32)
    noise = jax.random.normal(noise_rng, (1, ACTION_DIM))

    base = agent.compute_flow_actions(obs[None], noise, temperature=0.0)[0]
    refined = agent.sample_actions(obs, seed=seed, temperature=0.0).reshape(-1)
    np.testing.assert_allclose(
        np.asarray(refined),
        np.asarray(base),
        rtol=1e-5,
        atol=1e-6,
    )


def test_residual_off_bypasses_refiner():
    agent = _create_agent(disable_rl_policy=True)
    params = agent.network.params.copy()
    refiner_params = params["modules_target_refiner"].copy()
    output_params = refiner_params["Dense_0"].copy()
    output_params["bias"] = jnp.ones_like(output_params["bias"])
    refiner_params["Dense_0"] = output_params
    params["modules_target_refiner"] = refiner_params
    agent = agent.replace(network=agent.network.replace(params=params))

    seed = jax.random.PRNGKey(9)
    _, noise_rng = jax.random.split(seed)
    obs = jnp.arange(OBS_DIM, dtype=jnp.float32)
    noise = jax.random.normal(noise_rng, (1, ACTION_DIM))
    base = agent.compute_flow_actions(obs[None], noise, temperature=0.0)[0]
    residual_off = agent.sample_actions(
        obs,
        seed=seed,
        temperature=0.0,
    ).reshape(-1)
    np.testing.assert_array_equal(np.asarray(residual_off), np.asarray(base))


def test_bc_gradients_reach_lora_not_base_actor():
    agent = _create_agent()
    batch = _batch(with_successes=True)

    def loss_fn(params):
        loss, _ = agent._bc_loss(batch, params, jax.random.PRNGKey(11))
        return loss

    grads = jax.grad(loss_fn)(agent.network.params)
    assert _tree_l1(grads["modules_actor_lora"]) > 0.0
    assert _tree_l1(grads["modules_actor"]) == 0.0
    for module_name in (
        "target_actor",
        "target_actor_lora",
        "critic",
        "refiner",
        "log_alpha",
    ):
        assert _tree_l1(grads[f"modules_{module_name}"]) == 0.0


def test_refiner_gradients_isolated_from_expert():
    agent = _create_agent()
    batch = _batch()

    def loss_fn(params):
        loss, _ = agent._refiner_loss(batch, params, jax.random.PRNGKey(13))
        return loss

    grads = jax.grad(loss_fn)(agent.network.params)
    assert _tree_l1(grads["modules_refiner"]) > 0.0
    assert _tree_l1(grads["modules_log_alpha"]) > 0.0
    for module_name in (
        "actor",
        "actor_lora",
        "target_actor",
        "target_actor_lora",
        "critic",
        "target_critic",
        "target_refiner",
    ):
        assert _tree_l1(grads[f"modules_{module_name}"]) == 0.0


def test_phase1_disables_refiner_updates():
    agent = _create_agent(train_phase=1)
    before = agent.network.params
    updated, info = agent.update(_batch())
    assert float(np.asarray(info["refiner_weight"])) == 0.0
    assert (
        _tree_delta_l1(
            updated.network.params["modules_refiner"],
            before["modules_refiner"],
        )
        == 0.0
    )
    assert (
        _tree_delta_l1(
            updated.network.params["modules_actor"],
            before["modules_actor"],
        )
        == 0.0
    )
    assert (
        _tree_delta_l1(
            updated.network.params["modules_actor_lora"],
            before["modules_actor_lora"],
        )
        > 0.0
    )


def test_phase2_updates_refiner_and_keeps_base_frozen():
    agent = _create_agent(train_phase=2)
    before = agent.network.params
    updated, info = agent.update(_batch())
    assert float(np.asarray(info["refiner_weight"])) == 1.0
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert np.isfinite(float(np.asarray(info["cql_loss"])))
    assert (
        _tree_delta_l1(
            updated.network.params["modules_actor"],
            before["modules_actor"],
        )
        == 0.0
    )
    assert (
        _tree_delta_l1(
            updated.network.params["modules_refiner"],
            before["modules_refiner"],
        )
        > 0.0
    )


def test_success_bc_boost_changes_weights():
    agent = _create_agent(success_bc_boost=3.0)
    batch = _batch(with_successes=True)
    weights = np.asarray(agent._success_weights(batch))
    np.testing.assert_allclose(weights, [3.0, 1.0, 3.0, 1.0])
