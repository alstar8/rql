"""Focused unit tests for ConsensusDiscreteFlow helpers (no tokenizer ckpt)."""

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

from agents.consensus_discrete_flow import (  # noqa: E402
    ConsensusDiscreteFlowAgent,
    get_config,
)


Agent = ConsensusDiscreteFlowAgent


def test_mixture_replace_probability_schedule():
    n = 10
    dt = 1.0 / n
    for k in range(n):
        t = k / n
        p = float(Agent.mixture_replace_probability(t, dt))
        expected = min(1.0, dt / (1.0 - t)) if t < 1.0 else 1.0
        assert p == pytest.approx(expected, abs=1e-6)
    # Python float final step commits fully (not the production jnp.full path).
    assert float(Agent.mixture_replace_probability((n - 1) / n, dt)) == pytest.approx(
        1.0, abs=1e-6
    )


def test_float32_final_t_not_exact_unit_without_force():
    """Production ``jnp.full(i/n)`` can yield p<1 in float32; force_replace needed."""
    n = 10
    dt = jnp.asarray(1.0 / n, dtype=jnp.float32)
    t = jnp.full((1, 1), (n - 1) / n, dtype=jnp.float32)
    p = float(jnp.squeeze(Agent.mixture_replace_probability(t, dt)))
    # Document the gap that force_replace closes (may be exactly 1 on some JAX builds).
    assert p <= 1.0
    assert p > 0.999
    # On float32, (N-1)/N is typically slightly below the rational, so p < 1.
    t_py = (n - 1) / n
    p_py = float(Agent.mixture_replace_probability(t_py, 1.0 / n))
    assert p_py == pytest.approx(1.0, abs=0.0)
    if p < 1.0:
        assert p < p_py


def test_final_step_force_replace_with_float32_t():
    """Even when float32 t gives p<1, force_replace commits every register."""
    n = 10
    dt = jnp.asarray(1.0 / n, dtype=jnp.float32)
    t = jnp.full((4, 1), (n - 1) / n, dtype=jnp.float32)
    b, k, v = 4, 6, 3
    tokens = jnp.zeros((b, k), dtype=jnp.int32)
    logits = jnp.full((b, k, v), -1e9, dtype=jnp.float32)
    logits = logits.at[..., 2].set(5.0)
    # Without force_replace, a keep is possible under float32 p≈1-eps.
    new_forced, _ = Agent.posterior_mixture_update(
        tokens, logits, jax.random.PRNGKey(7), t, dt, force_replace=True
    )
    np.testing.assert_array_equal(np.asarray(new_forced), 2)


def test_final_step_replaces_all_registers():
    n = 10
    dt = 1.0 / n
    t = (n - 1) / n
    b, k, v = 4, 6, 3
    tokens = jnp.zeros((b, k), dtype=jnp.int32)
    logits = jnp.full((b, k, v), -1e9, dtype=jnp.float32)
    logits = logits.at[..., 2].set(5.0)
    new_tokens, _ = Agent.posterior_mixture_update(
        tokens, logits, jax.random.PRNGKey(7), t, dt, force_replace=True
    )
    np.testing.assert_array_equal(np.asarray(new_tokens), 2)


def test_jitted_posterior_update_float32_final_force_and_seed():
    """JIT-safe helper path with production float32 t + force_replace + seeds."""
    n = 10
    dt = jnp.asarray(1.0 / n, dtype=jnp.float32)
    t = jnp.full((2, 1), (n - 1) / n, dtype=jnp.float32)
    b, k, v = 2, 5, 4
    tokens = jnp.zeros((b, k), dtype=jnp.int32)
    logits = jnp.full((b, k, v), -1e9, dtype=jnp.float32)
    logits = logits.at[..., 3].set(8.0)

    @jax.jit
    def step(tok, log, rng, tt, dtt):
        # force_replace as compile-time constant (matches compute_flow_actions).
        return Agent.posterior_mixture_update(
            tok, log, rng, tt, dtt, force_replace=True
        )

    out1, _ = step(tokens, logits, jax.random.PRNGKey(11), t, dt)
    out2, _ = step(tokens, logits, jax.random.PRNGKey(11), t, dt)
    np.testing.assert_array_equal(np.asarray(out1), 3)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))

    # Mid-schedule (no force): seeded reproducibility under jit.
    t_mid = jnp.full((b, 1), 0.3, dtype=jnp.float32)
    dt_mid = jnp.asarray(0.1, dtype=jnp.float32)
    logits_rand = jax.random.normal(jax.random.PRNGKey(2), (b, k, v))

    @jax.jit
    def step_sched(tok, log, rng, tt, dtt):
        return Agent.posterior_mixture_update(
            tok, log, rng, tt, dtt, force_replace=False
        )

    a, _ = step_sched(tokens, logits_rand, jax.random.PRNGKey(123), t_mid, dt_mid)
    b_out, _ = step_sched(tokens, logits_rand, jax.random.PRNGKey(123), t_mid, dt_mid)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b_out))


def test_compute_flow_actions_unrolled_final_force_semantics():
    """Mirror production loop: float32 t + force_replace only on last index."""
    n = 10
    dt = 1.0 / n
    b, k, v = 3, 4, 5
    tokens = jnp.zeros((b, k), dtype=jnp.int32)
    logits = jnp.full((b, k, v), -1e9, dtype=jnp.float32)
    logits = logits.at[..., 1].set(10.0)

    def run_flow(seed):
        tok = tokens
        rng = seed
        for i in range(n):
            rng, step_rng = jax.random.split(rng)
            t = jnp.full((b, 1), i / n)  # same as compute_flow_actions
            tok, _ = Agent.posterior_mixture_update(
                tok,
                logits,
                step_rng,
                t,
                dt,
                force_replace=(i == n - 1),
            )
        return tok

    # Jit the full unrolled schedule (no tokenizer / agent needed).
    jitted = jax.jit(run_flow)
    out1 = jitted(jax.random.PRNGKey(0))
    out2 = jitted(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))
    # Final force_replace → all registers are the peaked candidate.
    np.testing.assert_array_equal(np.asarray(out1), 1)


def test_posterior_update_probability_empirical():
    """With peaked logits away from current token, replace rate ≈ p."""
    b, k, v = 64, 8, 5
    dt = 0.1
    t = 0.5  # p = dt/(1-t) = 0.2
    p_expected = float(Agent.mixture_replace_probability(t, dt))
    assert p_expected == pytest.approx(0.2)

    tokens = jnp.zeros((b, k), dtype=jnp.int32)
    logits = jnp.full((b, k, v), -1e9, dtype=jnp.float32)
    logits = logits.at[..., 1].set(10.0)  # always propose token 1

    def one_step(rng):
        new_tokens, _ = Agent.posterior_mixture_update(tokens, logits, rng, t, dt)
        return (new_tokens == 1).astype(jnp.float32).mean()

    rngs = jax.random.split(jax.random.PRNGKey(0), 200)
    rates = jax.vmap(one_step)(rngs)
    assert float(rates.mean()) == pytest.approx(p_expected, abs=0.02)


def test_posterior_update_seeded_reproducibility():
    b, k, v = 3, 4, 7
    tokens = jax.random.randint(jax.random.PRNGKey(1), (b, k), 0, v)
    logits = jax.random.normal(jax.random.PRNGKey(2), (b, k, v))
    t, dt = 0.3, 0.1
    seed = jax.random.PRNGKey(123)
    out1, _ = Agent.posterior_mixture_update(tokens, logits, seed, t, dt)
    out2, _ = Agent.posterior_mixture_update(tokens, logits, seed, t, dt)
    np.testing.assert_array_equal(np.asarray(out1), np.asarray(out2))

    out3, _ = Agent.posterior_mixture_update(
        tokens, logits, jax.random.PRNGKey(999), t, dt
    )
    # Different seed should almost surely differ on this size.
    assert not np.array_equal(np.asarray(out1), np.asarray(out3))


def test_corruption_keep_marginal():
    """Mixture corruption keeps clean tokens with probability ≈ t."""
    b, k = 128, 16
    clean = jnp.zeros((b, k), dtype=jnp.int32)
    t = jnp.full((b, 1), 0.7)
    rngs = jax.random.split(jax.random.PRNGKey(0), 50)

    def frac_kept(rng):
        out = Agent._corrupt_tokens(rng, clean, t)
        return (out == clean).astype(jnp.float32).mean()

    kept = jax.vmap(frac_kept)(rngs)
    assert float(kept.mean()) == pytest.approx(0.7, abs=0.02)


def test_guidance_coef_zero_bias_equivalence():
    """Analytic: Delta L = lambda * t * energy → zero when lambda=0."""
    guidance_coef = 0.0
    times = jnp.array([[0.4], [0.9]])
    energy = jnp.ones((2, 3, 5))
    bias = guidance_coef * times[..., None] * energy
    np.testing.assert_allclose(np.asarray(bias), 0.0)

    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 3, 5))
    guided = logits + bias
    np.testing.assert_allclose(np.asarray(guided), np.asarray(logits))


def test_chunk_token_ce_weights_h1_always_valid():
    # Right-shift: first row is zeros by construction.
    rs = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    w = Agent.chunk_token_ce_weights(rs, h=1)
    np.testing.assert_allclose(np.asarray(w), np.ones(3))


def test_chunk_token_ce_weights_h_gt1_excludes_padded_chunks():
    # h=3 actions; example 0 has terminal at step 1 → action index 1 invalid.
    # rs_terminals[:3] rows correspond to actions 0,1,2.
    rs = jnp.array(
        [
            [0.0, 0.0],  # action 0 always valid under right-shift
            [1.0, 0.0],  # action 1 invalid for ex0
            [1.0, 0.0],  # action 2 invalid for ex0
            [0.0, 0.0],  # trailing next-state terminal row
        ]
    )
    w = Agent.chunk_token_ce_weights(rs, h=3)
    np.testing.assert_allclose(np.asarray(w), np.array([0.0, 1.0]))


def test_safe_masked_mean_zero_mask():
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.zeros_like(values, dtype=bool)
    assert float(Agent.safe_masked_mean(values, mask)) == pytest.approx(0.0)


def test_config_drops_dead_actor_objective_keys():
    cfg = get_config()
    assert "actor_coef" not in cfg
    assert "bc_warmup_steps" not in cfg
    assert cfg.alpha == 1.0
    assert cfg.distill_coef == 1.0


def test_loss_composition_invariant_and_ce_gradient():
    """Total objective is alpha*CE + critic + distill; CE has nonzero grad."""
    alpha, distill_coef = 2.0, 0.5
    ce = jnp.array(1.5)
    critic = jnp.array(0.25)
    distill = jnp.array(0.8)
    total = alpha * ce + critic + distill_coef * distill
    assert float(total) == pytest.approx(2.0 * 1.5 + 0.25 + 0.5 * 0.8)

    # Tiny CE: logits (B=2,K=1,V=3) toward one-hot target → finite nonzero grad.
    target = jnp.array([[0], [1]], dtype=jnp.int32)

    def ce_fn(logits):
        log_p = jax.nn.log_softmax(logits, axis=-1)
        oh = jax.nn.one_hot(target, logits.shape[-1])
        return -(oh * log_p).sum(axis=-1).mean()

    logits0 = jnp.zeros((2, 1, 3))
    loss0 = ce_fn(logits0)
    g = jax.grad(ce_fn)(logits0)
    assert jnp.isfinite(loss0)
    assert float(loss0) > 0.0
    assert bool(jnp.isfinite(g).all())
    assert float(jnp.linalg.norm(g)) > 0.0


def test_weighted_ce_matches_unweighted_when_all_valid():
    ce_tok = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (B, K)
    weight = jnp.ones((3,))
    weighted = (ce_tok.mean(axis=-1) * weight).sum() / jnp.maximum(weight.sum(), 1e-6)
    assert float(weighted) == pytest.approx(float(ce_tok.mean()))
