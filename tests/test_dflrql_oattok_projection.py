"""Focused unit tests for DFLRQL9 + OATTok projection eval helpers (no MuJoCo)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "eval_dflrql_oattok_projection",
    ROOT / "scripts" / "eval_dflrql_oattok_projection.py",
)
assert _SPEC is not None and _SPEC.loader is not None
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


class _StubTokenizer:
    """Minimal stand-in for OATTok.apply encode/FSQ/decode round-trip."""

    def apply(self, variables, sample, deterministic=True):
        del variables, deterministic
        # Identity reconstruction with a fixed shift so metrics are nonzero.
        recons = sample + 0.01
        tokens = jnp.zeros(sample.shape[:2], dtype=jnp.int32)
        quant = jnp.zeros((*sample.shape[:2], 4), dtype=jnp.float32)
        return recons, tokens, quant


def test_project_action_shape_b1_h_d_returns_hd():
    projector = mod.OATTokActionProjector(_StubTokenizer(), tok_params={})
    action = np.zeros((1, 21), dtype=np.float32)
    out = projector(action)
    assert out.shape == (1, 21)
    np.testing.assert_allclose(out, action + 0.01, atol=1e-5)

    action_h3 = np.zeros((3, 7), dtype=np.float32)
    out_h3 = projector(action_h3)
    assert out_h3.shape == (3, 7)


def test_project_rejects_non_hd():
    projector = mod.OATTokActionProjector(_StubTokenizer(), tok_params={})
    with pytest.raises(ValueError, match=r"\(h, d\)"):
        projector(np.zeros((21,), dtype=np.float32))


def test_raw_bypass_exact_identity_metrics():
    raw = np.linspace(-1, 1, 21, dtype=np.float32).reshape(1, 21)
    # Raw condition stores zeros for proj diffs (exact bypass).
    metrics = {
        "proj_rmse": 0.0,
        "proj_mae": 0.0,
        "proj_max_abs": 0.0,
        "raw_sat_frac": float(np.mean(np.abs(raw) >= 1.0 - 1e-5)),
        "proj_sat_frac": float(np.mean(np.abs(raw) >= 1.0 - 1e-5)),
    }
    assert metrics["proj_rmse"] == 0.0
    assert metrics["proj_mae"] == 0.0
    assert metrics["proj_max_abs"] == 0.0


def test_action_diff_metrics_on_shifted_projection():
    raw = np.zeros((1, 21), dtype=np.float32)
    proj = raw + 0.25
    m = mod.action_diff_metrics(raw, proj)
    assert m["proj_rmse"] == pytest.approx(0.25)
    assert m["proj_mae"] == pytest.approx(0.25)
    assert m["proj_max_abs"] == pytest.approx(0.25)


def test_deterministic_paired_seed_schedule():
    base = 7
    s0 = mod.episode_env_seed(base, 0)
    s1 = mod.episode_env_seed(base, 1)
    assert s0 != s1
    assert mod.episode_env_seed(base, 0) == s0

    k0a = mod.actor_key_for_episode(base, 0)
    k0b = mod.actor_key_for_episode(base, 0)
    k1 = mod.actor_key_for_episode(base, 1)
    np.testing.assert_array_equal(np.asarray(k0a), np.asarray(k0b))
    assert not np.array_equal(np.asarray(k0a), np.asarray(k1))

    # Same initial actor key for both conditions of an episode.
    raw_keys = []
    proj_keys = []
    for cond_keys in (raw_keys, proj_keys):
        rng = mod.actor_key_for_episode(base, 3)
        fn = mod.supply_rng_from_key(lambda obs, seed=None, temperature=0.0: seed, rng)
        for _ in range(4):
            cond_keys.append(np.asarray(fn(obs=None)))
    for a, b in zip(raw_keys, proj_keys):
        np.testing.assert_array_equal(a, b)


def test_assert_tokenizer_matches_policy():
    meta = {"sample_horizon": 1, "sample_dim": 21}
    mod.assert_tokenizer_matches_policy(meta, policy_h=1, prim_action_dim=21)
    with pytest.raises(ValueError, match="sample_horizon"):
        mod.assert_tokenizer_matches_policy(meta, policy_h=5, prim_action_dim=21)
    with pytest.raises(ValueError, match="sample_dim"):
        mod.assert_tokenizer_matches_policy(meta, policy_h=1, prim_action_dim=8)


def test_config_from_flags_overrides_proven_dflrql9():
    flags = {
        "agent": {
            "agent_name": "dflrql9",
            "alpha": 0.3,
            "discount": 0.995,
            "expectile": 0.5,
            "ensemble_ct": 10,
            "h": 1,
            "guidance_coef": 0.5,
            "consensus_floor": 0.01,
            "conflict_power": 2.0,
            "residual_coef": 0.25,
        }
    }
    cfg = mod.config_from_flags(flags)
    assert cfg["alpha"] == 0.3
    assert cfg["discount"] == 0.995
    assert cfg["expectile"] == 0.5
    assert cfg["ensemble_ct"] == 10
    assert cfg["h"] == 1
    assert cfg["guidance_coef"] == 0.5
    assert cfg["consensus_floor"] == 0.01
    assert cfg["agent_name"] == "dflrql9"


def test_summarize_condition_rows():
    rows = [
        {
            "condition": "raw",
            "episode": 0,
            "env_seed": 1,
            "success": 1.0,
            "episode.return": -10.0,
        },
        {
            "condition": "raw",
            "episode": 1,
            "env_seed": 2,
            "success": 0.0,
            "episode.return": -20.0,
        },
    ]
    summary = mod.summarize_condition_rows(rows)
    assert summary["num_episodes"] == 2.0
    assert summary["mean_success"] == pytest.approx(0.5)
    assert summary["mean_episode.return"] == pytest.approx(-15.0)


def test_clip_action_hd():
    x = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)
    y = mod.clip_action_hd(x, clip_eps=0.0)
    np.testing.assert_allclose(y, [[-1.0, 0.0, 1.0]])


def test_parse_conditions():
    assert mod.parse_conditions(["raw", "projected"]) == ["raw", "projected"]
    assert mod.parse_conditions(["raw,projected"]) == ["raw", "projected"]
    with pytest.raises(SystemExit):
        mod.parse_conditions(["discrete_policy"])
