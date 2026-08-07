"""Tests for isolated, shape-checked BC-backbone restoration."""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dflrql10 import (  # noqa: E402
    DFLRQL10Agent,
    get_config as get_dflrql10_config,
)
from agents.dflrql11 import (  # noqa: E402
    DFLRQL11Agent,
    get_config as get_dflrql11_config,
)
from utils.flax_utils import restore_agent_backbone, save_agent  # noqa: E402

OBS_DIM = 7
ACTION_DIM = 3
BATCH_SIZE = 4


def _source_agent():
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
        }
    )
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    source = DFLRQL10Agent.create(0, observations, actions, config)
    params = source.network.params.copy()
    params["modules_actor"] = jax.tree_util.tree_map(
        lambda value: value + 0.25,
        params["modules_actor"],
    )
    params["modules_target_actor"] = jax.tree_util.tree_map(
        lambda value: value - 0.5,
        params["modules_target_actor"],
    )
    params["modules_value"] = jax.tree_util.tree_map(
        lambda value: value + 0.75,
        params["modules_value"],
    )
    params["modules_target_value"] = jax.tree_util.tree_map(
        lambda value: value - 0.125,
        params["modules_target_value"],
    )
    return source.replace(network=source.network.replace(params=params))


def _target_agent(h=1):
    config = dict(get_dflrql11_config())
    config.update(
        {
            "h": h,
            "batch_size": BATCH_SIZE,
            "ensemble_ct": 2,
            "flow_steps": 2,
            "actor_hidden_dims": (16, 16),
            "value_hidden_dims": (16, 16),
            "refiner_hidden_dims": (16, 16),
        }
    )
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    return DFLRQL11Agent.create(1, observations, actions, config)


def _tree_delta_l1(left, right) -> float:
    deltas = jax.tree_util.tree_map(lambda a, b: a - b, left, right)
    return sum(
        float(np.abs(np.asarray(leaf)).sum())
        for leaf in jax.tree_util.tree_leaves(deltas)
    )


def _write_source(source, tmp_path):
    save_agent(source, str(tmp_path), 7)
    return tmp_path / "params_7.pkl"


def test_actor_only_restore_isolates_new_rl_modules_and_state(tmp_path):
    source = _source_agent()
    target = _target_agent()
    checkpoint = _write_source(source, tmp_path)

    restored = restore_agent_backbone(target, str(checkpoint))

    assert restored.network.step == 1
    assert _tree_delta_l1(restored.rng, target.rng) == 0.0
    assert (
        _tree_delta_l1(
            restored.network.params["modules_actor"],
            source.network.params["modules_actor"],
        )
        == 0.0
    )
    assert (
        _tree_delta_l1(
            restored.network.params["modules_target_actor"],
            source.network.params["modules_target_actor"],
        )
        == 0.0
    )
    for module_name in ("refiner", "target_refiner", "critic", "target_critic"):
        assert (
            _tree_delta_l1(
                restored.network.params[f"modules_{module_name}"],
                target.network.params[f"modules_{module_name}"],
            )
            == 0.0
        )
    assert _tree_delta_l1(restored.network.opt_state, target.network.opt_state) == 0.0


def test_optional_endpoint_critic_restore_maps_legacy_value_modules(tmp_path):
    source = _source_agent()
    target = _target_agent()
    checkpoint = _write_source(source, tmp_path)

    restored = restore_agent_backbone(
        target,
        str(checkpoint),
        restore_critic=True,
    )

    assert (
        _tree_delta_l1(
            restored.network.params["modules_critic"],
            source.network.params["modules_value"],
        )
        == 0.0
    )
    assert (
        _tree_delta_l1(
            restored.network.params["modules_target_critic"],
            source.network.params["modules_target_value"],
        )
        == 0.0
    )


def test_shape_mismatch_fails_before_partial_copy(tmp_path):
    checkpoint = _write_source(_source_agent(), tmp_path)
    mismatched_target = _target_agent(h=2)

    with pytest.raises(ValueError, match="parameter shape mismatch"):
        restore_agent_backbone(mismatched_target, str(checkpoint))
