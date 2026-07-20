"""Focused unit tests for DiscreteDiffusionQdflDistillAgent."""

from __future__ import annotations

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

from agents.discrete_diffusion_qdfl_distill import (  # noqa: E402
    DiscreteDiffusionQdflDistillAgent,
    get_config,
    qdfl_dd_mixture_replace_probability,
    qdfl_dd_path_step_logprob,
)
from utils.flax_utils import restore_agent, save_agent  # noqa: E402

TOK_PATH = ROOT / "exp/ogbench_oattok/humanoidmaze-large_h1_d21.pkl"
OBS_DIM = 69
PRIM_DIM = 21
H = 1


def _tiny_config(**overrides):
    cfg = dict(get_config())
    cfg.update(
        {
            "h": H,
            "batch_size": 4,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (32, 32),
            "value_hidden_dims": (32, 32),
            "guidance_hidden_dims": (32, 32),
            "student_actor_hidden_dims": (32, 32),
            "tokenizer_path": str(TOK_PATH),
            "freeze_teacher": False,
            "distill_coef": 1.0,
            "dataset_bc_coef": 0.1,
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


@pytest.fixture(scope="module")
def agent():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")
    obs, actions = _ex_obs_actions()
    return DiscreteDiffusionQdflDistillAgent.create(
        0, obs, actions, _tiny_config()
    )


def test_random_init_teacher_and_student_differ(agent):
    t_leaves = jax.tree_util.tree_leaves(agent.teacher.network.params)
    s_leaves = jax.tree_util.tree_leaves(agent.network.params)
    assert t_leaves and s_leaves
    # Different architectures; at least student actor exists and is finite.
    flat_s = np.concatenate([np.asarray(x).ravel() for x in s_leaves])
    assert np.isfinite(flat_s).all()
    assert float(np.abs(flat_s).sum()) > 0.0


def test_update_finite_and_info_keys(agent):
    batch = _synthetic_batch(agent)
    new_agent, info = agent.update(batch)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert np.isfinite(float(np.asarray(info["distill_ce"])))
    assert "teacher_total_loss" in info
    assert float(np.asarray(info["freeze_teacher"])) == 0.0
    # Student and teacher steps advanced.
    assert int(new_agent.network.step) == int(agent.network.step) + 1
    assert int(new_agent.teacher.network.step) == int(
        agent.teacher.network.step
    ) + 1


def test_freeze_teacher_skips_teacher_update(agent):
    obs, actions = _ex_obs_actions()
    frozen = DiscreteDiffusionQdflDistillAgent.create(
        1, obs, actions, _tiny_config(freeze_teacher=True)
    )
    # Copy teacher params from the joint agent to have a warm tree, then freeze.
    frozen = frozen.replace(teacher=agent.teacher)
    before = jax.tree_util.tree_map(
        lambda x: np.array(x), frozen.teacher.network.params
    )
    teacher_step_before = int(frozen.teacher.network.step)
    batch = _synthetic_batch(frozen)
    after_agent, info = frozen.update(batch)
    assert float(np.asarray(info["freeze_teacher"])) == 1.0
    assert "teacher_total_loss" not in info
    assert int(after_agent.teacher.network.step) == teacher_step_before
    after = jax.tree_util.tree_map(
        lambda x: np.array(x), after_agent.teacher.network.params
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after)
    ):
        np.testing.assert_array_equal(a, b)
    # Student still updates.
    assert int(after_agent.network.step) == int(frozen.network.step) + 1


def test_checkpoint_roundtrip_teacher_and_student(agent):
    batch = _synthetic_batch(agent)
    agent2, _ = agent.update(batch)
    with tempfile.TemporaryDirectory() as tmp:
        save_agent(agent2, tmp, epoch=7)
        obs, actions = _ex_obs_actions()
        template = DiscreteDiffusionQdflDistillAgent.create(
            0, obs, actions, _tiny_config()
        )
        restored = restore_agent(template, tmp, 7)
    # Student params match.
    for a, b in zip(
        jax.tree_util.tree_leaves(agent2.network.params),
        jax.tree_util.tree_leaves(restored.network.params),
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)
    # Teacher params match.
    for a, b in zip(
        jax.tree_util.tree_leaves(agent2.teacher.network.params),
        jax.tree_util.tree_leaves(restored.teacher.network.params),
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=0)
    assert int(restored.network.step) == int(agent2.network.step)


def test_no_student_grad_into_teacher_or_tokenizer(agent):
    batch = _synthetic_batch(agent)
    teacher_tokens = agent.sample_teacher_tokens(
        batch["observations"][0], jax.random.PRNGKey(9)
    )

    def loss_from_student(params):
        loss, _ = agent.student_total_loss(
            batch, teacher_tokens, params, rng=jax.random.PRNGKey(3)
        )
        return loss

    grads = jax.grad(loss_from_student)(agent.network.params)
    flat_g = np.concatenate(
        [np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(grads)]
    )
    assert np.isfinite(flat_g).all()
    assert float(np.abs(flat_g).sum()) > 0.0

    # Student-only update must leave teacher params and tokenizer untouched.
    before_teacher = jax.tree_util.tree_map(
        lambda x: np.array(x), agent.teacher.network.params
    )
    before_tok = jax.tree_util.tree_map(
        lambda x: np.array(x), agent.teacher.tokenizer_params
    )
    after_agent, _ = agent._student_update(batch, teacher_tokens)
    after_teacher = jax.tree_util.tree_map(
        lambda x: np.array(x), after_agent.teacher.network.params
    )
    after_tok = jax.tree_util.tree_map(
        lambda x: np.array(x), after_agent.teacher.tokenizer_params
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(before_teacher),
        jax.tree_util.tree_leaves(after_teacher),
    ):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(
        jax.tree_util.tree_leaves(before_tok),
        jax.tree_util.tree_leaves(after_tok),
    ):
        np.testing.assert_array_equal(a, b)


def test_seeded_sample_actions_deterministic(agent):
    obs = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    seed = jax.random.PRNGKey(123)
    a1 = agent.sample_actions(obs, seed=seed, temperature=1.0)
    a2 = agent.sample_actions(obs, seed=seed, temperature=1.0)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    a3 = agent.sample_actions(obs, seed=jax.random.PRNGKey(999), temperature=1.0)
    assert not np.array_equal(np.asarray(a1), np.asarray(a3))
    assert a1.shape == (H, PRIM_DIM)


def test_sample_tokens_with_logprob_identity(agent):
    obs = jax.random.normal(jax.random.PRNGKey(0), (2, OBS_DIM))
    seed = jax.random.PRNGKey(42)
    out = agent.sample_tokens_with_logprob(obs, seed=seed, temperature=1.0)
    assert out["tokens"].shape == (2, int(agent.config["num_registers"]))
    n = int(agent.config["flow_steps"])
    assert out["token_trajectory"].shape[0] == n + 1
    assert out["replace_masks"].shape == (n, 2, int(agent.config["num_registers"]))
    assert out["logprob"].shape == (2,)
    assert np.isfinite(np.asarray(out["logprob"])).all()

    rescored = agent.rescore_trajectory_logprob(
        obs,
        out["token_trajectory"],
        replace_masks=out["replace_masks"],
        temperature=1.0,
        use_path_masks=True,
    )
    np.testing.assert_allclose(
        np.asarray(out["logprob"]),
        np.asarray(rescored["logprob"]),
        rtol=1e-5,
        atol=1e-5,
    )
    # Same seed → same tokens and logprob.
    out2 = agent.sample_tokens_with_logprob(obs, seed=seed, temperature=1.0)
    np.testing.assert_array_equal(
        np.asarray(out["tokens"]), np.asarray(out2["tokens"])
    )
    np.testing.assert_allclose(
        np.asarray(out["logprob"]),
        np.asarray(out2["logprob"]),
        rtol=0,
        atol=0,
    )


def test_force_replace_final_step_logprob():
    n = 10
    dt = 1.0 / n
    t = jnp.full((2, 1), (n - 1) / n, dtype=jnp.float32)
    tokens = jnp.zeros((2, 4), dtype=jnp.int32)
    next_tokens = jnp.ones((2, 4), dtype=jnp.int32)
    logits = jnp.full((2, 4, 5), -1e9, dtype=jnp.float32)
    logits = logits.at[..., 1].set(5.0)
    replace = jnp.ones((2, 4), dtype=bool)
    lp = qdfl_dd_path_step_logprob(
        tokens, next_tokens, logits, replace, t, dt, force_replace=True
    )
    # Pure categorical under force_replace.
    log_cat = jax.nn.log_softmax(logits, axis=-1)[..., 1]
    expected = log_cat.sum(axis=-1)
    np.testing.assert_allclose(
        np.asarray(lp), np.asarray(expected), rtol=1e-5, atol=1e-5
    )


def test_keep_replace_path_logprob_matches_manual():
    t = 0.5
    dt = 0.1
    p = float(qdfl_dd_mixture_replace_probability(t, dt))
    tokens = jnp.zeros((1, 2), dtype=jnp.int32)
    next_tokens = jnp.array([[0, 1]], dtype=jnp.int32)
    logits = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    logits = logits.at[..., 0].set(2.0)
    logits = logits.at[..., 1].set(1.0)
    replace = jnp.array([[False, True]])
    lp = qdfl_dd_path_step_logprob(
        tokens, next_tokens, logits, replace, t, dt, force_replace=False
    )
    log_cat = jax.nn.log_softmax(logits[0, 1])
    expected = np.log(1.0 - p) + float(log_cat[1]) + np.log(p)
    assert float(lp[0]) == pytest.approx(expected, abs=1e-5)


def test_sample_actions_clipped(agent):
    obs = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    actions = agent.sample_actions(
        obs, seed=jax.random.PRNGKey(0), temperature=1.0
    )
    assert np.asarray(actions).min() >= -1.0 - 1e-5
    assert np.asarray(actions).max() <= 1.0 + 1e-5
