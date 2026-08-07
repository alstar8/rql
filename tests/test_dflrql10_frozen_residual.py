"""Tests for frozen-expert residual policy improvement."""

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
from agents.dflrql10 import (  # noqa: E402
    DFLRQL10Agent,
    get_config as get_dflrql10_config,
)

OBS_DIM = 7
ACTION_DIM = 3
BATCH_SIZE = 4


def _tiny_config(**overrides):
    config = dict(get_dflrql10_config())
    config.update(
        {
            "h": 1,
            "batch_size": BATCH_SIZE,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (16, 16),
            "value_hidden_dims": (16, 16),
            "guidance_hidden_dims": (16, 16),
            "distill_coef": 0.0,
        }
    )
    config.update(overrides)
    return config


def _create_agent(**overrides):
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    return DFLRQL10Agent.create(
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
    deltas = jax.tree_util.tree_map(lambda a, b: a - b, left, right)
    return _tree_l1(deltas)


def test_registry_and_frozen_expert_defaults():
    config = get_dflrql10_config()
    assert agent_registry["dflrql10"] is DFLRQL10Agent
    assert config.freeze_actor is True
    assert config.actor_q_coef == 0.0
    assert config.alpha == 0.0
    assert config.guidance_rollout_q_coef == 1.0
    assert config.guidance_use_advantage is True
    assert config.guidance_rl_bypass_safety is True
    assert config.guidance_energy_coef > 0.0


def test_rollout_q_gradient_only_reaches_guidance():
    agent = _create_agent(guidance_energy_coef=0.0)
    batch = _batch()
    rng = jax.random.PRNGKey(11)

    def rollout_loss(params):
        _, info = agent.total_loss(batch, params, rng=rng)
        return info["guidance_rollout_q_loss"]

    grads = jax.grad(rollout_loss)(agent.network.params)
    assert _tree_l1(grads["modules_guidance"]) > 0.0
    assert _tree_l1(grads["modules_actor"]) == 0.0
    assert _tree_l1(grads["modules_target_actor"]) == 0.0
    assert _tree_l1(grads["modules_value"]) == 0.0
    assert _tree_l1(grads["modules_target_value"]) == 0.0


def test_advantage_signal_is_logged():
    agent = _create_agent(guidance_energy_coef=0.0)
    _, info = agent.total_loss(
        _batch(),
        agent.network.params,
        rng=jax.random.PRNGKey(3),
    )
    assert float(np.asarray(info["guidance_use_advantage"])) == 1.0
    assert float(np.asarray(info["guidance_rl_bypass_safety"])) == 1.0
    assert np.isfinite(float(np.asarray(info["guidance_rollout_adv_mean"])))


def test_frozen_update_changes_guidance_but_not_actor():
    agent = _create_agent(guidance_energy_coef=0.01)
    updated, info = agent.update(_batch())

    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert np.isfinite(float(np.asarray(info["guidance_rollout_q_loss"])))
    assert float(np.asarray(info["freeze_actor"])) == 1.0
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
            updated.network.params["modules_guidance"],
            agent.network.params["modules_guidance"],
        )
        > 0.0
    )
