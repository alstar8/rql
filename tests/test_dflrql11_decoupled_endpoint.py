"""Focused tests for endpoint Decoupled ConsensusFlow."""

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
from agents.dflrql11 import (  # noqa: E402
    DFLRQL11Agent,
    get_config as get_dflrql11_config,
)

OBS_DIM = 7
ACTION_DIM = 3
BATCH_SIZE = 4


def _tiny_config(**overrides):
    config = dict(get_dflrql11_config())
    config.update(
        {
            "h": 1,
            "batch_size": BATCH_SIZE,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (16, 16),
            "value_hidden_dims": (16, 16),
            "refiner_hidden_dims": (16, 16),
            "actor_delay": 1,
            "target_policy_noise": 0.0,
        }
    )
    config.update(overrides)
    return config


def _create_agent(**overrides):
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    return DFLRQL11Agent.create(
        0,
        observations,
        actions,
        _tiny_config(**overrides),
    )


def _batch():
    return {
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


def _tree_l1(tree) -> float:
    return sum(
        float(np.abs(np.asarray(leaf)).sum())
        for leaf in jax.tree_util.tree_leaves(tree)
    )


def _tree_delta_l1(left, right) -> float:
    return _tree_l1(jax.tree_util.tree_map(lambda a, b: a - b, left, right))


def test_registry_and_decoupled_defaults():
    config = get_dflrql11_config()
    assert agent_registry["dflrql11"] is DFLRQL11Agent
    assert config.freeze_v is True
    assert config.zero_init_refiner is True
    assert config.target_divergence > 0.0
    assert config.actor_delay >= 1


def test_zero_refiner_is_exact_base_policy():
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


def test_residual_off_bypasses_nonzero_refiner():
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


def test_bc_loss_gradients_reach_only_behavior_actor():
    agent = _create_agent(freeze_v=False)
    batch = _batch()

    def loss_fn(params):
        loss, _ = agent._bc_loss(batch, params, jax.random.PRNGKey(11))
        return loss

    grads = jax.grad(loss_fn)(agent.network.params)
    assert _tree_l1(grads["modules_actor"]) > 0.0
    for module_name in (
        "target_actor",
        "critic",
        "target_critic",
        "refiner",
        "target_refiner",
        "log_alpha",
    ):
        assert _tree_l1(grads[f"modules_{module_name}"]) == 0.0


def test_rl_loss_gradients_do_not_reach_behavior_actor_or_critic():
    agent = _create_agent()
    batch = _batch()

    def loss_fn(params):
        loss, _ = agent._refiner_loss(
            batch,
            params,
            jax.random.PRNGKey(13),
        )
        return loss

    grads = jax.grad(loss_fn)(agent.network.params)
    assert _tree_l1(grads["modules_refiner"]) > 0.0
    assert _tree_l1(grads["modules_log_alpha"]) > 0.0
    for module_name in (
        "actor",
        "target_actor",
        "critic",
        "target_critic",
        "target_refiner",
    ):
        assert _tree_l1(grads[f"modules_{module_name}"]) == 0.0


def test_frozen_update_preserves_v_and_updates_refiner_and_targets():
    agent = _create_agent()
    updated, info = agent.update(_batch())

    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert (
        _tree_delta_l1(
            updated.network.params["modules_actor"],
            agent.network.params["modules_actor"],
        )
        == 0.0
    )
    assert (
        _tree_delta_l1(
            updated.network.params["modules_target_actor"],
            agent.network.params["modules_target_actor"],
        )
        == 0.0
    )
    assert (
        _tree_delta_l1(
            updated.network.params["modules_refiner"],
            agent.network.params["modules_refiner"],
        )
        > 0.0
    )
    assert (
        _tree_delta_l1(
            updated.network.params["modules_target_critic"],
            agent.network.params["modules_target_critic"],
        )
        > 0.0
    )


def test_bc_online_update_changes_v_without_rl_gradient_leakage():
    agent = _create_agent(freeze_v=False)
    updated, _ = agent.update(_batch())

    assert (
        _tree_delta_l1(
            updated.network.params["modules_actor"],
            agent.network.params["modules_actor"],
        )
        > 0.0
    )


def test_lagrange_gradient_has_correct_direction():
    agent = _create_agent(target_divergence=0.01)
    batch = _batch()

    def alpha_gradient(current_agent):
        def loss_fn(params):
            _, info = current_agent._refiner_loss(
                batch,
                params,
                jax.random.PRNGKey(17),
            )
            return info["alpha_loss"]

        grads = jax.grad(loss_fn)(current_agent.network.params)
        return float(
            np.asarray(
                grads["modules_log_alpha"]["value"],
            )
        )

    # Zero residual lies below the target, so descent decreases log(alpha).
    assert alpha_gradient(agent) > 0.0

    params = agent.network.params.copy()
    refiner_params = params["modules_refiner"].copy()
    output_params = refiner_params["Dense_0"].copy()
    output_params["bias"] = jnp.ones_like(output_params["bias"])
    refiner_params["Dense_0"] = output_params
    params["modules_refiner"] = refiner_params
    high_residual_agent = agent.replace(
        network=agent.network.replace(params=params)
    )

    # A residual above the target makes descent increase log(alpha).
    assert alpha_gradient(high_residual_agent) < 0.0


def test_consensus_trust_shapes_and_losses_are_finite():
    agent = _create_agent(consensus_trust=True)
    loss, info = agent.total_loss(
        _batch(),
        agent.network.params,
        rng=jax.random.PRNGKey(19),
    )

    assert np.isfinite(float(np.asarray(loss)))
    assert np.isfinite(float(np.asarray(info["consensus_trust"])))
    assert 0.0 <= float(np.asarray(info["consensus_trust"])) <= 1.0
