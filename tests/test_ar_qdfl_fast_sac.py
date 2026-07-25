"""Focused tests for ConsensusFlowRL's discrete FastSAC agent."""

from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ar_qdfl_fast_sac import (  # noqa: E402
    ARQDFLFastSACAgent,
    categorical_projection,
    get_config,
)
from agents.paired_qdfl_helpers import (  # noqa: E402
    finite_info,
    tree_leaves_allclose,
    tree_l2_distance,
)
from utils.flax_utils import restore_agent, save_agent  # noqa: E402

TOK_PATH = ROOT / "exp/ogbench_oattok/humanoidmaze-large_h1_d21.pkl"
OBS_DIM = 69
ACTION_DIM = 21


def _tiny_config(**overrides):
    config = dict(get_config())
    config.update(
        {
            "h": 1,
            "batch_size": 4,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (32, 32),
            "value_hidden_dims": (32, 32),
            "guidance_hidden_dims": (32, 32),
            "actor_emb_dim": 32,
            "actor_depth": 1,
            "actor_num_heads": 2,
            "critic_hidden_dims": (32, 16),
            "num_atoms": 11,
            "v_min": -10.0,
            "v_max": 0.0,
            "tokenizer_path": str(TOK_PATH),
        }
    )
    config.update(overrides)
    return config


def _create_agent(**overrides):
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    return ARQDFLFastSACAgent.create(
        0, observations, actions, _tiny_config(**overrides)
    )


def _batch(batch_size=4):
    return {
        "observations": jax.random.normal(
            jax.random.PRNGKey(1), (2, batch_size, OBS_DIM)
        ),
        "actions": jnp.clip(
            jax.random.normal(
                jax.random.PRNGKey(2), (2, batch_size, ACTION_DIM)
            ),
            -1.0,
            1.0,
        ),
        "rewards": -jnp.ones((2, batch_size), dtype=jnp.float32),
        "terminals": jnp.zeros((2, batch_size), dtype=jnp.float32),
        "masks": jnp.ones((2, batch_size), dtype=jnp.float32),
    }


@pytest.fixture(scope="module", autouse=True)
def tokenizer_available():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")


def test_categorical_projection_preserves_mass_and_terminal_target():
    support = jnp.linspace(-10.0, 0.0, 11)
    probabilities = jnp.zeros((2, 3, 11)).at[..., 5].set(1.0)
    rewards = jnp.asarray([-1.0, -4.0, 0.0])
    discounts = jnp.asarray([0.9, 0.0, 0.0])
    projected = categorical_projection(
        probabilities, rewards, discounts, support
    )
    np.testing.assert_allclose(np.asarray(projected.sum(-1)), 1.0, atol=1e-6)
    expected_terminal = jax.nn.one_hot(
        jnp.asarray([6, 10]), 11
    )  # rewards -4 and 0 on [-10,0].
    np.testing.assert_allclose(
        np.asarray(projected[0, 1:]),
        np.asarray(expected_terminal),
        atol=1e-6,
    )


def test_create_has_independent_critic_and_reference_actor():
    agent = _create_agent()
    assert "modules_q" in agent.critic.params
    assert "modules_target_q" in agent.critic.params
    assert tree_leaves_allclose(
        agent.critic.params["modules_q"],
        agent.critic.params["modules_target_q"],
    )
    assert tree_leaves_allclose(
        agent.reference_actor_params,
        agent.network.params["modules_target_actor"],
    )


def test_critic_warmup_does_not_change_actor_or_teacher():
    agent = _create_agent()
    actor_before = copy.deepcopy(agent.network.params)
    teacher_before = copy.deepcopy(agent.teacher.network.params)
    critic_before = copy.deepcopy(agent.critic.params["modules_q"])
    updated, info = agent.critic_update(_batch())
    assert finite_info(info)
    assert tree_leaves_allclose(actor_before, updated.network.params)
    assert tree_leaves_allclose(
        teacher_before, updated.teacher.network.params
    )
    assert tree_l2_distance(
        critic_before, updated.critic.params["modules_q"]
    ) > 0.0
    assert int(updated.critic_update_count) == 1


def test_actor_update_changes_actor_not_critic_or_teacher():
    agent = _create_agent()
    agent, _ = agent.critic_update(_batch())
    critic_before = copy.deepcopy(agent.critic.params)
    teacher_before = copy.deepcopy(agent.teacher.network.params)
    actor_before = copy.deepcopy(agent.network.params["modules_actor"])
    updated, info = agent.actor_update(_batch())
    assert finite_info(info)
    assert tree_l2_distance(
        actor_before, updated.network.params["modules_actor"]
    ) > 0.0
    assert tree_leaves_allclose(critic_before, updated.critic.params)
    assert tree_leaves_allclose(
        teacher_before, updated.teacher.network.params
    )
    assert float(info["actor_entropy_per_register"]) >= 0.0
    assert float(info["actor_reference_kl"]) >= -1e-5
    # Advantage normalization keeps the SAC loss O(1), not O(|v_min|).
    assert abs(float(info["actor_sac_loss"])) < 20.0
    assert abs(float(info["actor_advantage_mean"])) < 1e-5


def test_actor_loss_not_dominated_by_raw_q_scale():
    """KL/entropy must remain visible against large negative C51 values."""
    agent = _create_agent(v_min=-200.0, v_max=0.0, num_atoms=21, offline_kl_coef=1.0)
    agent = agent.with_offline_reference()
    batch = _batch()
    # Warm a critic so Q is nonzero, then check loss scale.
    for _ in range(5):
        agent, _ = agent.critic_update(batch)
    _, info = agent.actor_update(batch)
    assert finite_info(info)
    assert abs(float(info["actor_sac_loss"])) < 20.0
    # Without normalization, loss ≈ -Q ≈ 100+; with it, KL can move the loss.
    assert float(info["actor_reference_kl"]) >= -1e-5


def test_st_forward_codes_are_exact_codebook_entries():
    agent = _create_agent()
    observations = jnp.zeros((3, OBS_DIM), dtype=jnp.float32)
    sample = agent._sample_ar_relaxed(
        observations,
        jax.random.PRNGKey(9),
        actor_name="actor",
        actor_params=agent.network.params,
        straight_through=True,
    )
    hard_codes = agent.codebook[sample["tokens"]]
    np.testing.assert_allclose(
        np.asarray(sample["codes"]),
        np.asarray(hard_codes),
        atol=1e-6,
    )


def test_checkpoint_roundtrip_preserves_phase_state():
    agent = _create_agent()
    agent, _ = agent.critic_update(_batch())
    agent = agent.record_online_env_step(7).with_offline_reference()
    with tempfile.TemporaryDirectory() as tmp:
        save_agent(agent, tmp, 1)
        template = _create_agent()
        restored = restore_agent(template, tmp, 1)
    assert int(restored.critic_update_count) == 1
    assert int(restored.online_env_step) == 7
    assert tree_leaves_allclose(agent.critic.params, restored.critic.params)
    assert tree_leaves_allclose(
        agent.reference_actor_params, restored.reference_actor_params
    )
