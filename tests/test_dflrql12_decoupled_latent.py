"""Focused tests for latent-policy Decoupled ConsensusFlow."""

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
from agents.dflrql12 import (  # noqa: E402
    DFLRQL12Agent,
    get_config as get_dflrql12_config,
)

OBS_DIM = 7
ACTION_DIM = 3
BATCH_SIZE = 4


def _tiny_config(**overrides):
    config = dict(get_dflrql12_config())
    config.update(
        {
            "h": 1,
            "batch_size": BATCH_SIZE,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (16, 16),
            "value_hidden_dims": (16, 16),
            "latent_policy_hidden_dims": (16, 16),
            "actor_delay": 1,
            "target_candidates": 4,
            "deployment_candidates": 4,
        }
    )
    config.update(overrides)
    return config


def _create_agent(**overrides):
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    return DFLRQL12Agent.create(
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


def test_registry_and_latent_defaults():
    config = get_dflrql12_config()
    assert agent_registry["dflrql12"] is DFLRQL12Agent
    assert config.freeze_v is True
    assert config.target_candidates == 4
    assert config.deployment_candidates == 4
    assert config.latent_kl_beta > 0.0


def test_initial_policy_is_exact_unit_gaussian_with_zero_kl():
    agent = _create_agent()
    _, info = agent._latent_policy_loss(
        _batch(),
        agent.network.params,
        jax.random.PRNGKey(5),
    )

    assert float(np.asarray(info["latent_kl"])) == 0.0
    assert float(np.asarray(info["latent_mean_norm"])) == 0.0
    assert float(np.asarray(info["latent_std_mean"])) == 1.0


def test_residual_off_decodes_an_unselected_prior_sample():
    agent = _create_agent(disable_rl_policy=True)
    observations = jnp.zeros((1, OBS_DIM))
    seed = jax.random.PRNGKey(6)
    prior_latent = jax.random.normal(seed, (1, ACTION_DIM))
    expected = agent._flow_actions(
        observations,
        prior_latent,
        "target_actor",
    )[0]
    actual = agent.sample_actions(
        observations[0],
        seed=seed,
        temperature=0.0,
    ).reshape(-1)
    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=1e-5,
        atol=1e-6,
    )


def test_exact_gaussian_kl_formula():
    agent = _create_agent()
    params = agent.network.params.copy()
    policy_params = params["modules_latent_policy"].copy()
    mean_params = policy_params["Dense_0"].copy()
    std_params = policy_params["Dense_1"].copy()
    mean_params["bias"] = jnp.full_like(mean_params["bias"], 0.5)
    std_params["bias"] = jnp.full_like(std_params["bias"], jnp.log(1.5))
    policy_params["Dense_0"] = mean_params
    policy_params["Dense_1"] = std_params
    params["modules_latent_policy"] = policy_params
    modified = agent.replace(network=agent.network.replace(params=params))

    _, info = modified._latent_policy_loss(
        _batch(),
        modified.network.params,
        jax.random.PRNGKey(7),
    )
    expected_per_coordinate = 0.5 * (
        0.5**2 + 1.5**2 - 1.0 - 2.0 * np.log(1.5)
    )
    expected = ACTION_DIM * expected_per_coordinate
    np.testing.assert_allclose(
        float(np.asarray(info["latent_kl"])),
        expected,
        rtol=1e-5,
    )


def test_latent_rl_gradient_reaches_only_latent_policy():
    agent = _create_agent()
    batch = _batch()

    def loss_fn(params):
        loss, _ = agent._latent_policy_loss(
            batch,
            params,
            jax.random.PRNGKey(11),
        )
        return loss

    grads = jax.grad(loss_fn)(agent.network.params)
    assert _tree_l1(grads["modules_latent_policy"]) > 0.0
    for module_name in (
        "actor",
        "target_actor",
        "target_latent_policy",
        "latent_critic",
        "target_latent_critic",
    ):
        assert _tree_l1(grads[f"modules_{module_name}"]) == 0.0


def test_reverse_latent_critic_gradient_does_not_reach_flow():
    agent = _create_agent()
    batch = _batch()

    def loss_fn(params):
        loss, _ = agent._latent_critic_loss(
            batch,
            params,
            jax.random.PRNGKey(13),
        )
        return loss

    grads = jax.grad(loss_fn)(agent.network.params)
    assert _tree_l1(grads["modules_latent_critic"]) > 0.0
    for module_name in (
        "actor",
        "target_actor",
        "latent_policy",
        "target_latent_policy",
        "target_latent_critic",
    ):
        assert _tree_l1(grads[f"modules_{module_name}"]) == 0.0


def test_reverse_ode_is_identity_for_zero_vector_field():
    agent = _create_agent()
    params = agent.network.params.copy()
    zero_actor = jax.tree_util.tree_map(
        jnp.zeros_like,
        params["modules_actor"],
    )
    params["modules_actor"] = zero_actor
    params["modules_target_actor"] = zero_actor
    zero_agent = agent.replace(network=agent.network.replace(params=params))

    observations = jnp.zeros((BATCH_SIZE, OBS_DIM))
    actions = jnp.linspace(
        -0.5,
        0.5,
        BATCH_SIZE * ACTION_DIM,
    ).reshape(BATCH_SIZE, ACTION_DIM)
    latents = zero_agent.reverse_flow_latents(observations, actions)
    np.testing.assert_array_equal(np.asarray(latents), np.asarray(actions))


def test_frozen_update_preserves_v_and_updates_latent_policy_targets():
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
            updated.network.params["modules_latent_policy"],
            agent.network.params["modules_latent_policy"],
        )
        > 0.0
    )
    assert (
        _tree_delta_l1(
            updated.network.params["modules_target_latent_critic"],
            agent.network.params["modules_target_latent_critic"],
        )
        > 0.0
    )


def test_bc_online_changes_only_bc_flow_through_bc_objective():
    agent = _create_agent(freeze_v=False)
    batch = _batch()

    def bc_loss(params):
        return agent._bc_loss(
            batch,
            params,
            jax.random.PRNGKey(17),
        )[0]

    grads = jax.grad(bc_loss)(agent.network.params)
    assert _tree_l1(grads["modules_actor"]) > 0.0
    assert _tree_l1(grads["modules_latent_policy"]) == 0.0
    assert _tree_l1(grads["modules_latent_critic"]) == 0.0


def test_best_of_four_and_deployment_shapes_are_finite():
    agent = _create_agent()
    observations = jnp.zeros((BATCH_SIZE, OBS_DIM))
    best_q, candidates, candidate_q = agent._best_target_latent_q(
        observations,
        jax.random.PRNGKey(19),
    )
    assert best_q.shape == (BATCH_SIZE,)
    assert candidates.shape == (4, BATCH_SIZE, ACTION_DIM)
    assert candidate_q.shape == (4, BATCH_SIZE)
    np.testing.assert_allclose(
        np.asarray(best_q),
        np.asarray(candidate_q).max(axis=0),
    )

    action = agent.sample_actions(
        observations[0],
        seed=jax.random.PRNGKey(23),
        temperature=0.0,
    )
    assert action.shape == (1, ACTION_DIM)
    assert np.isfinite(np.asarray(action)).all()
