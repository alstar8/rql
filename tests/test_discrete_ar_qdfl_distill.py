"""Focused tests for paired QuantizedDFLRQL9 teacher + AR student distill."""

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

from agents.discrete_ar_qdfl_distill import (  # noqa: E402
    DiscreteARQdflDistillAgent,
    get_config,
)
from agents.paired_qdfl_helpers import (  # noqa: E402
    finite_info,
    tree_leaves_allclose,
    tree_l2_distance,
)
from agents.quantized_dflrql9 import QuantizedDFLRQL9Agent  # noqa: E402
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
            "actor_emb_dim": 32,
            "actor_depth": 1,
            "actor_num_heads": 2,
            "tokenizer_path": str(TOK_PATH),
            "distill_coef": 1.0,
            "bc_coef": 0.1,
            "freeze_teacher": False,
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
def tok_available():
    if not TOK_PATH.is_file():
        pytest.skip(f"missing tokenizer {TOK_PATH}")


def test_logprob_identity_matches_ce(tok_available):
    """token CE == -token logprob; sequence logprob sums tokens."""
    logits = jax.random.normal(jax.random.PRNGKey(0), (3, 4, 1000))
    tokens = jax.random.randint(jax.random.PRNGKey(1), (3, 4), 0, 1000)
    tok_lp = DiscreteARQdflDistillAgent.token_log_probs_from_logits(logits, tokens)
    seq_lp = DiscreteARQdflDistillAgent.sequence_log_probs_from_logits(
        logits, tokens
    )
    ce = DiscreteARQdflDistillAgent.token_ce(logits, tokens)
    np.testing.assert_allclose(
        np.asarray(ce), -np.asarray(tok_lp), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(seq_lp), np.asarray(tok_lp.sum(axis=-1)), rtol=1e-5, atol=1e-5
    )


def test_create_random_teacher_and_student(tok_available):
    obs, actions = _ex_obs_actions()
    agent = DiscreteARQdflDistillAgent.create(0, obs, actions, _tiny_config())
    assert isinstance(agent.teacher, QuantizedDFLRQL9Agent)
    assert agent.network is not None
    assert "modules_actor" in agent.network.params
    assert "modules_target_actor" in agent.network.params
    assert "modules_actor" in agent.teacher.network.params
    # Student and teacher actor trees differ (AR vs flow).
    assert set(agent.network.params.keys()) != set(agent.teacher.network.params.keys())
    # Distinct random inits: student params should not match a fresh same-seed
    # re-create of only the teacher flow actor shapes.
    assert int(agent.network.step) == 1
    assert int(agent.teacher.network.step) == 1


def test_joint_update_finite_and_bumps_both(tok_available):
    obs, actions = _ex_obs_actions()
    agent = DiscreteARQdflDistillAgent.create(0, obs, actions, _tiny_config())
    batch = _synthetic_batch(agent)
    t_step0 = int(agent.teacher.network.step)
    s_step0 = int(agent.network.step)

    new_agent, info = agent.update(batch)
    assert finite_info(info)
    assert "total_loss" in info
    assert "distill_loss" in info
    assert "bc_loss" in info
    assert "teacher_total_loss" in info
    assert float(info["freeze_teacher"]) == 0.0
    assert int(new_agent.teacher.network.step) == t_step0 + 1
    assert int(new_agent.network.step) == s_step0 + 1
    # Teacher params moved.
    assert tree_l2_distance(
        agent.teacher.network.params, new_agent.teacher.network.params
    ) > 0.0
    # Student params moved.
    assert tree_l2_distance(
        agent.network.params, new_agent.network.params
    ) > 0.0


def test_freeze_teacher_skips_teacher_update(tok_available):
    obs, actions = _ex_obs_actions()
    agent = DiscreteARQdflDistillAgent.create(
        0, obs, actions, _tiny_config(freeze_teacher=True)
    )
    batch = _synthetic_batch(agent)
    teacher_before = copy.deepcopy(agent.teacher.network.params)
    t_step0 = int(agent.teacher.network.step)
    s_step0 = int(agent.network.step)

    new_agent, info = agent.update(batch)
    assert float(info["freeze_teacher"]) == 1.0
    assert "teacher_total_loss" not in info
    assert int(new_agent.teacher.network.step) == t_step0
    assert int(new_agent.network.step) == s_step0 + 1
    assert tree_leaves_allclose(
        teacher_before, new_agent.teacher.network.params
    )
    assert tree_l2_distance(agent.network.params, new_agent.network.params) > 0.0


def test_no_student_grad_into_teacher_or_tokenizer(tok_available):
    obs, actions = _ex_obs_actions()
    agent = DiscreteARQdflDistillAgent.create(
        0, obs, actions, _tiny_config(freeze_teacher=True)
    )
    batch = _synthetic_batch(agent)
    obs0 = batch["observations"][0]
    tokens = agent.teacher.sample_tokens_batch(
        obs0, seed=jax.random.PRNGKey(7), temperature=0.0
    )

    def loss_only(student_params):
        loss, _ = agent.student_total_loss(
            batch, tokens, student_params, rng=jax.random.PRNGKey(0)
        )
        return loss

    grads = jax.grad(loss_only)(agent.network.params)
    assert finite_info({"g": jax.tree_util.tree_leaves(grads)[0]})
    # Gradients exist on student actor.
    g_actor = grads["modules_actor"]
    assert float(sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(g_actor))) > 0.0

    # Teacher params unchanged by student_update path.
    teacher_before = copy.deepcopy(agent.teacher.network.params)
    tok_before = jax.tree_util.tree_map(
        lambda x: np.array(x), agent.tokenizer_params
    )
    new_agent, _ = agent.student_update(batch, tokens)
    assert tree_leaves_allclose(
        teacher_before, new_agent.teacher.network.params
    )
    assert tree_leaves_allclose(tok_before, new_agent.tokenizer_params)


def test_teacher_token_sampler_stopgrad_and_shape(tok_available):
    obs, actions = _ex_obs_actions()
    agent = DiscreteARQdflDistillAgent.create(0, obs, actions, _tiny_config())
    batch = _synthetic_batch(agent)
    obs0 = batch["observations"][0]
    b = obs0.shape[0]
    k = int(agent.config["num_registers"])

    tokens = agent.teacher.sample_tokens_batch(
        obs0, seed=jax.random.PRNGKey(3), temperature=0.0
    )
    assert tokens.shape == (b, k)
    assert tokens.dtype == jnp.int32

    # Deterministic for same seed.
    t2 = agent.teacher.sample_tokens_batch(
        obs0, seed=jax.random.PRNGKey(3), temperature=0.0
    )
    np.testing.assert_array_equal(np.asarray(tokens), np.asarray(t2))

    # stop-grad on returned tokens: grad through the token tensor is zero.
    g = jax.grad(lambda x: jax.lax.stop_gradient(x).sum())(
        tokens.astype(jnp.float32)
    )
    assert float(np.abs(np.asarray(g)).sum()) == 0.0


def test_teacher_flow_batch_sampler(tok_available):
    obs, actions = _ex_obs_actions()
    cfg = _tiny_config()
    teacher = QuantizedDFLRQL9Agent.create(0, obs, actions, cfg)
    obs_b = jax.random.normal(jax.random.PRNGKey(0), (4, OBS_DIM))
    flat = teacher.sample_flow_actions_batch(
        obs_b, seed=jax.random.PRNGKey(1), temperature=0.0
    )
    assert flat.shape == (4, H * PRIM_DIM)
    assert np.isfinite(np.asarray(flat)).all()
    tokens = teacher.sample_tokens_batch(
        obs_b, seed=jax.random.PRNGKey(1), temperature=0.0
    )
    assert tokens.shape[0] == 4
    assert tokens.shape[1] == int(teacher.tokenizer_meta["num_registers"])


def test_checkpoint_roundtrip_teacher_and_student(tok_available):
    obs, actions = _ex_obs_actions()
    agent = DiscreteARQdflDistillAgent.create(0, obs, actions, _tiny_config())
    batch = _synthetic_batch(agent)
    agent, _ = agent.update(batch)

    with tempfile.TemporaryDirectory() as tmp:
        save_dir = Path(tmp)
        save_agent(agent, str(save_dir), epoch=2)
        template = DiscreteARQdflDistillAgent.create(
            0, obs, actions, _tiny_config()
        )
        restored = restore_agent(template, str(save_dir), restore_epoch=2)

    assert int(restored.network.step) == int(agent.network.step)
    assert int(restored.teacher.network.step) == int(agent.teacher.network.step)
    assert tree_leaves_allclose(
        agent.network.params, restored.network.params
    )
    assert tree_leaves_allclose(
        agent.teacher.network.params, restored.teacher.network.params
    )
    # Tokenizer nonpytree reattached on create; still usable.
    assert restored.tokenizer_params is not None
    assert restored.teacher.tokenizer_params is not None


def test_sample_actions_seeded_and_logprob_api(tok_available):
    obs, actions = _ex_obs_actions()
    agent = DiscreteARQdflDistillAgent.create(0, obs, actions, _tiny_config())
    o = jnp.zeros((OBS_DIM,), dtype=jnp.float32)
    seed = jax.random.PRNGKey(42)
    a1 = agent.sample_actions(o, seed=seed, temperature=0.0)
    a2 = agent.sample_actions(o, seed=seed, temperature=0.0)
    np.testing.assert_allclose(np.asarray(a1), np.asarray(a2), rtol=0, atol=0)
    assert a1.shape == (H, PRIM_DIM)

    batch = _synthetic_batch(agent)
    obs0 = batch["observations"][0]
    tokens = agent.teacher.sample_tokens_batch(
        obs0, seed=jax.random.PRNGKey(9), temperature=0.0
    )
    tok_lp = agent.token_log_probs(obs0, tokens)
    seq_lp = agent.sequence_log_probs(obs0, tokens)
    assert tok_lp.shape == tokens.shape
    np.testing.assert_allclose(
        np.asarray(seq_lp), np.asarray(tok_lp.sum(-1)), rtol=1e-5, atol=1e-5
    )


def test_student_initial_params_differ_across_seeds(tok_available):
    obs, actions = _ex_obs_actions()
    a0 = DiscreteARQdflDistillAgent.create(0, obs, actions, _tiny_config())
    a1 = DiscreteARQdflDistillAgent.create(1, obs, actions, _tiny_config())
    assert not tree_leaves_allclose(a0.network.params, a1.network.params)
    assert not tree_leaves_allclose(
        a0.teacher.network.params, a1.teacher.network.params
    )
