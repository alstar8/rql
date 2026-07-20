"""Focused unit tests for DiscreteCoordMaskIQL helpers (no tokenizer ckpt)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.discrete_coord_mask_iql import (  # noqa: E402
    CoordMaskActor,
    DiscreteCoordMaskIQLAgent,
    MAX_FSQ_CLASSES,
    get_config,
)
from agents.oattok_jax import (  # noqa: E402
    CODEBOOK_SIZE,
    FSQ_DIM,
    FSQ_LEVELS,
    OATTok,
    coord_indices_to_token_ids,
    fsq_basis,
    fsq_class_valid_mask,
    indices_to_codes,
    token_ids_to_coord_indices,
)
from utils.flax_utils import ModuleDict, TrainState  # noqa: E402
from utils.networks import Value  # noqa: E402


Agent = DiscreteCoordMaskIQLAgent


def test_coord_unpack_pack_roundtrip_all_ids():
    ids = jnp.arange(CODEBOOK_SIZE, dtype=jnp.int32)
    coords = token_ids_to_coord_indices(ids, FSQ_LEVELS)
    assert coords.shape == (CODEBOOK_SIZE, FSQ_DIM)
    basis = np.asarray(fsq_basis(FSQ_LEVELS))
    np.testing.assert_array_equal(basis, np.array([1, 8, 40, 200]))
    for axi, level in enumerate(FSQ_LEVELS):
        assert int(coords[:, axi].min()) >= 0
        assert int(coords[:, axi].max()) < level
    packed = coord_indices_to_token_ids(coords, FSQ_LEVELS)
    np.testing.assert_array_equal(np.asarray(packed), np.asarray(ids))


def test_class_valid_mask_and_invalid_logit_exclusion():
    valid = fsq_class_valid_mask(FSQ_LEVELS, MAX_FSQ_CLASSES)
    assert valid.shape == (FSQ_DIM, MAX_FSQ_CLASSES)
    np.testing.assert_array_equal(np.asarray(valid[0]), [True] * 8)
    np.testing.assert_array_equal(
        np.asarray(valid[1]), [True, True, True, True, True, False, False, False]
    )
    logits = jnp.zeros((2, 3, FSQ_DIM, MAX_FSQ_CLASSES))
    logits = logits.at[..., 7].set(100.0)  # invalid for axes 1..3
    safe = Agent.apply_class_valid_logits(logits, valid)
    # Axis 0: class 7 is valid → stays large.
    assert float(safe[0, 0, 0, 7]) == pytest.approx(100.0)
    # Axis 1: class 7 invalid → masked to -1e9.
    assert float(safe[0, 0, 1, 7]) < -1e8
    probs = jax.nn.softmax(safe, axis=-1)
    # Invalid slots must have ~0 probability.
    assert float(probs[0, 0, 1, 7]) == pytest.approx(0.0, abs=1e-6)
    assert float(probs[0, 0, 1, :5].sum()) == pytest.approx(1.0, abs=1e-5)


def test_corruption_guarantees_mask_and_no_identity_copy():
    b, k, q = 64, 4, 4
    # Distinct nonzero coords so zeroing is detectable.
    clean = (jnp.arange(b * k * q).reshape(b, k, q) % 5) + 1
    t = jnp.full((b, 1), 0.999)  # almost never mask → force path dominates
    rngs = jax.random.split(jax.random.PRNGKey(0), 20)

    def one(rng):
        _, mask, inp = Agent.corrupt_coords(rng, clean, t)
        any_masked = mask.any(axis=(1, 2)).astype(jnp.float32).mean()
        masked_zero = ((inp == 0) | (~mask)).astype(jnp.float32).mean()
        differ = (mask & (inp != clean.astype(inp.dtype))).any().astype(jnp.float32)
        return mask.mean(), any_masked, masked_zero, differ

    fracs, any_m, mz, differ = jax.vmap(one)(rngs)
    assert float(any_m.min()) == pytest.approx(1.0)
    assert float(mz.min()) == pytest.approx(1.0)
    assert float(differ.min()) == pytest.approx(1.0)
    assert float(fracs.mean()) > 0.0


def test_masked_only_ce_behavior():
    valid = fsq_class_valid_mask(FSQ_LEVELS, MAX_FSQ_CLASSES)
    b, k = 2, 2
    # Perfect logits on axis targets; garbage elsewhere.
    targets = jnp.zeros((b, k, FSQ_DIM), dtype=jnp.int32)
    targets = targets.at[..., 0].set(3)
    targets = targets.at[..., 1].set(2)
    logits = jnp.full((b, k, FSQ_DIM, MAX_FSQ_CLASSES), -5.0)
    logits = logits.at[..., 0, 3].set(10.0)
    logits = logits.at[..., 1, 2].set(10.0)
    logits = logits.at[..., 2, 0].set(10.0)
    logits = logits.at[..., 3, 0].set(10.0)
    mask = jnp.zeros((b, k, FSQ_DIM), dtype=bool)
    mask = mask.at[:, :, 0].set(True)  # only first axis masked
    ce = Agent.hard_coord_ce(logits, targets, valid, mask)
    # Unmasked sites contribute 0 by construction.
    np.testing.assert_allclose(np.asarray(ce[:, :, 1:]), 0.0)
    assert float(ce[:, :, 0].mean()) < 0.1


def test_h_step_td_helper_and_chunk_validity():
    h = 1
    rewards = jnp.array([[1.0, 2.0], [0.0, 0.0]])
    terminals = jnp.array([[0.0, 0.0], [0.0, 1.0]])
    masks = jnp.array([[1.0, 1.0], [1.0, 0.0]])
    discount = 0.995
    # Production: discount ** [0..h-1, inf] → last entry is 0, not inf.
    discount_mul = discount ** jnp.array([0.0, jnp.inf])
    next_v = jnp.array([10.0, 20.0])
    target, valids, rs = Agent.h_step_td_target(
        rewards, terminals, masks, discount, discount_mul, h, next_v
    )
    # Right-shift: first terminal row zeros.
    np.testing.assert_allclose(np.asarray(rs[0]), 0.0)
    expected0 = 1.0 + (discount**h) * 10.0 * float(masks[-2, 0])
    assert float(target[0]) == pytest.approx(expected0, abs=1e-5)

    # h=1 chunk weights always 1.
    w1 = Agent.chunk_coord_ce_weights(rs, h=1)
    np.testing.assert_allclose(np.asarray(w1), 1.0)

    rs3 = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )
    w3 = Agent.chunk_coord_ce_weights(rs3, h=3)
    np.testing.assert_allclose(np.asarray(w3), np.array([0.0, 1.0]))


def test_expectile_sign_and_awr_warmup_ramp_clip_ess():
    # Positive residual (target > pred) weighted by expectile.
    pos = Agent.expectile_loss(jnp.array(2.0), 0.7)
    neg = Agent.expectile_loss(jnp.array(-2.0), 0.7)
    assert float(pos) == pytest.approx(0.7 * 4.0)
    assert float(neg) == pytest.approx(0.3 * 4.0)

    warmup, ramp = 50_000, 50_000
    assert float(Agent.awr_ramp_fraction(0, warmup, ramp)) == pytest.approx(0.0)
    assert float(Agent.awr_ramp_fraction(warmup, warmup, ramp)) == pytest.approx(0.0)
    assert float(Agent.awr_ramp_fraction(warmup + ramp // 2, warmup, ramp)) == pytest.approx(
        0.5, abs=1e-5
    )
    assert float(Agent.awr_ramp_fraction(warmup + ramp, warmup, ramp)) == pytest.approx(1.0)

    adv = jnp.array([0.0, 1000.0, -1000.0])
    w0 = Agent.awr_example_weights(adv, temperature=40.0, max_weight=100.0, ramp_frac=0.0)
    np.testing.assert_allclose(np.asarray(w0), 1.0)
    w1 = Agent.awr_example_weights(adv, temperature=40.0, max_weight=100.0, ramp_frac=1.0)
    assert float(w1[0]) == pytest.approx(1.0)
    assert float(w1[1]) == pytest.approx(100.0)  # clipped
    assert float(w1[2]) < 1.0
    ess = float(Agent.awr_ess(w1))
    assert ess > 0.0
    assert ess <= float(len(w1)) + 1e-6


def test_maskgit_schedule_monotonic_and_final_unmasked():
    m, n = 64, 16
    counts = Agent.maskgit_remaining_counts(m, n)
    assert len(counts) == n
    assert counts[-1] == 0
    # Exact cosine+progress schedule for the production (M=64, N=16) case:
    # step 0 must unmask (≥1), remaining monotone, final forced to 0.
    assert counts == (
        63,
        62,
        61,
        60,
        57,
        54,
        50,
        46,
        41,
        36,
        31,
        25,
        19,
        13,
        7,
        0,
    )
    for a, b in zip(counts, counts[1:]):
        assert b <= a
    # Every non-final step makes progress from the previous remaining count.
    prev = m
    for rem in counts[:-1]:
        assert rem < prev
        prev = rem
    # Degenerate: single step is just the final flush.
    assert Agent.maskgit_remaining_counts(64, 1) == (0,)


def test_unmask_topk_forces_progress():
    mask = jnp.ones((2, 8), dtype=bool)
    conf = jnp.linspace(0.1, 0.9, 8)[None, :].repeat(2, axis=0)
    new_mask = Agent.unmask_topk_by_confidence(mask, conf, 3)
    assert int(new_mask.sum()) == 2 * 5
    # Highest conf sites unmasked.
    assert bool((~new_mask[0, -3:]).all())


def test_deterministic_vs_stochastic_seed_behavior():
    # Argmax path identical across seeds for fixed logits.
    logits = jnp.zeros((1, 2, 4, 8))
    logits = logits.at[..., 0].set(5.0)
    valid = fsq_class_valid_mask(FSQ_LEVELS, MAX_FSQ_CLASSES)
    safe = Agent.apply_class_valid_logits(logits, valid)
    c0, conf0, _ = Agent._sample_or_argmax(None, safe, jax.random.PRNGKey(1), 0.0)
    c1, conf1, _ = Agent._sample_or_argmax(None, safe, jax.random.PRNGKey(99), 0.0)
    np.testing.assert_array_equal(np.asarray(c0), np.asarray(c1))
    np.testing.assert_allclose(np.asarray(conf0), np.asarray(conf1))

    # Stochastic path: same seed reproduces; different seed differs.
    soft = jax.random.normal(jax.random.PRNGKey(0), (4, 3, 4, 8))
    soft = Agent.apply_class_valid_logits(soft, valid)
    a, _, _ = Agent._sample_or_argmax(None, soft, jax.random.PRNGKey(7), 1.0)
    b, _, _ = Agent._sample_or_argmax(None, soft, jax.random.PRNGKey(7), 1.0)
    c, _, _ = Agent._sample_or_argmax(None, soft, jax.random.PRNGKey(8), 1.0)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert not np.array_equal(np.asarray(a), np.asarray(c))


def test_stochastic_sample_actions_seed_contract():
    """temperature>0: supplied seed reproduces; seed=None repeats (no rng writeback)."""
    agent = _make_tiny_agent_for_update()
    obs = jnp.zeros((8,), dtype=jnp.float32)
    rng_before = np.asarray(agent.rng)

    seed = jax.random.PRNGKey(123)
    a1 = agent.sample_actions(obs, seed=seed, temperature=1.0)
    a2 = agent.sample_actions(obs, seed=seed, temperature=1.0)
    a3 = agent.sample_actions(obs, seed=jax.random.PRNGKey(456), temperature=1.0)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    assert not np.array_equal(np.asarray(a1), np.asarray(a3))

    # Fallback seed=None uses self.rng but does not mutate it → repeats.
    b1 = agent.sample_actions(obs, seed=None, temperature=1.0)
    b2 = agent.sample_actions(obs, seed=None, temperature=1.0)
    np.testing.assert_array_equal(np.asarray(b1), np.asarray(b2))
    np.testing.assert_array_equal(np.asarray(agent.rng), rng_before)

def test_post_update_target_formula():
    d = 0.25
    new_online = {"w": jnp.array(10.0)}
    old_target = {"w": jnp.array(1.0)}
    old_online = {"w": jnp.array(0.0)}

    class _Net:
        def __init__(self, params):
            self.params = params

    new_network = _Net(
        {
            "modules_q": new_online,
            "modules_target_q": dict(old_target),
        }
    )
    Agent.target_update(None, new_network, "q", d=d)
    expected = float(new_online["w"] * d + old_target["w"] * (1 - d))
    buggy = float(old_online["w"] * d + old_target["w"] * (1 - d))
    got = float(new_network.params["modules_target_q"]["w"])
    assert got == pytest.approx(expected)
    assert got != pytest.approx(buggy)


def test_actor_loss_changes_with_advantage_weights():
    """Higher AWR weight on high-CE example increases weighted actor loss."""
    ce_ex = jnp.array([1.0, 10.0])
    w_uniform = jnp.ones(2)
    w_adv = jnp.array([1.0, 5.0])
    loss_u = (ce_ex * w_uniform).sum() / w_uniform.sum()
    loss_a = (ce_ex * w_adv).sum() / w_adv.sum()
    assert float(loss_a) > float(loss_u)

    # Q/V vs actor separation: actor CE depends on weights; Q MSE does not.
    q_err = jnp.array([0.5, -0.5])
    q_loss = jnp.square(q_err).mean()
    q_loss2 = jnp.square(q_err).mean()  # independent of awr weights
    assert float(q_loss) == pytest.approx(float(q_loss2))
    assert float(loss_a) != pytest.approx(float(loss_u))


def test_config_defaults():
    cfg = get_config()
    assert cfg.agent_name == "discrete_coord_mask_iql"
    assert cfg.expectile == 0.7
    assert cfg.discount == 0.995
    assert cfg.awr_temperature == 40.0
    assert cfg.max_weight == 100.0
    assert cfg.bc_warmup_steps == 50_000
    assert cfg.awr_ramp_steps == 50_000
    assert cfg.maskgit_steps == 16
    assert cfg.ensemble_ct == 2
    assert cfg.alpha == 1.0


def test_safe_masked_mean_zero_mask():
    values = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    mask = jnp.zeros_like(values, dtype=bool)
    assert float(Agent.safe_masked_mean(values, mask)) == pytest.approx(0.0)


class _StubOATTok:
    """Flax-apply stand-in: encode/decode without a real tokenizer checkpoint."""

    def __init__(self, num_registers: int, sample_horizon: int, sample_dim: int):
        self.num_registers = num_registers
        self.sample_horizon = sample_horizon
        self.sample_dim = sample_dim

    def apply(self, variables, x, *, method=None, deterministic=True):
        del variables, deterministic
        if method is OATTok.encode:
            b = x.shape[0]
            tokens = jnp.zeros((b, self.num_registers), dtype=jnp.int32)
            codes = indices_to_codes(tokens, FSQ_LEVELS)
            return codes, tokens
        if method is OATTok.decode:
            # Deterministic decode that depends on codes so MaskGIT samples
            # with different seeds are distinguishable in action space.
            b = x.shape[0]
            flat = x.reshape(b, -1).astype(jnp.float32)
            # Tile/trim to (B, H, D).
            need = self.sample_horizon * self.sample_dim
            if flat.shape[-1] < need:
                reps = int(np.ceil(need / flat.shape[-1]))
                flat = jnp.tile(flat, (1, reps))
            out = flat[:, :need].reshape(b, self.sample_horizon, self.sample_dim)
            return jnp.tanh(out)
        raise ValueError(f"Unexpected stub method: {method}")


def _make_tiny_agent_for_update(
    *,
    batch_size: int = 4,
    obs_dim: int = 8,
    act_dim: int = 3,
    h: int = 1,
    num_registers: int = 2,
):
    """Tiny DiscreteCoordMaskIQL agent for exercising stock total_loss/update."""
    cfg = get_config()
    cfg = dict(cfg)
    cfg.update(
        dict(
            h=h,
            batch_size=batch_size,
            num_registers=num_registers,
            actor_hidden_dims=(32, 32),
            value_hidden_dims=(32, 32),
            ensemble_ct=2,
            v_ensemble_ct=1,
            maskgit_steps=2,
            bc_warmup_steps=0,
            awr_ramp_steps=1,
            discount=0.99,
        )
    )
    action_dim = act_dim * h
    cfg["action_dim"] = action_dim
    cfg["prim_action_dim"] = act_dim
    cfg["discount_mul"] = jnp.array(
        cfg["discount"] ** jnp.array(list(range(h)) + [jnp.inf])
    )

    ex_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_flat_actions = jnp.zeros((1, action_dim), dtype=jnp.float32)
    ex_times = jnp.zeros((1, 1), dtype=jnp.float32)
    ex_coords = jnp.zeros((1, num_registers, FSQ_DIM), dtype=jnp.float32)
    ex_mask = jnp.ones_like(ex_coords)
    flat_c = ex_coords.reshape(1, -1)
    flat_m = ex_mask.reshape(1, -1)
    ex_actor_in = jnp.concatenate([ex_obs, flat_c, flat_m, ex_times], axis=-1)
    ex_q_in = jnp.concatenate([ex_obs, ex_flat_actions], axis=-1)

    q_def = Value(
        hidden_dims=cfg["value_hidden_dims"],
        layer_norm=cfg["layer_norm"],
        num_ensembles=cfg["ensemble_ct"],
    )
    v_def = Value(
        hidden_dims=cfg["value_hidden_dims"],
        layer_norm=cfg["layer_norm"],
        num_ensembles=cfg["v_ensemble_ct"],
    )
    actor_def = CoordMaskActor(
        hidden_dims=cfg["actor_hidden_dims"],
        num_registers=num_registers,
        layer_norm=cfg["layer_norm"],
    )
    network_info = dict(
        q=(q_def, (ex_q_in,)),
        target_q=(copy.deepcopy(q_def), (ex_q_in,)),
        v=(v_def, (ex_obs,)),
        target_v=(copy.deepcopy(v_def), (ex_obs,)),
        actor=(actor_def, (ex_actor_in,)),
        target_actor=(copy.deepcopy(actor_def), (ex_actor_in,)),
    )
    networks = {name: spec[0] for name, spec in network_info.items()}
    network_args = {name: spec[1] for name, spec in network_info.items()}
    network_def = ModuleDict(networks)
    network_tx = optax.adam(learning_rate=cfg["lr"])
    network_params = network_def.init(jax.random.PRNGKey(0), **network_args)["params"]
    network = TrainState.create(network_def, network_params, tx=network_tx)
    params = network.params
    params["modules_target_q"] = params["modules_q"]
    params["modules_target_v"] = params["modules_v"]
    params["modules_target_actor"] = params["modules_actor"]

    return Agent(
        rng=jax.random.PRNGKey(1),
        network=network,
        tokenizer_def=_StubOATTok(num_registers, h, act_dim),
        tokenizer_params={},
        class_valid=fsq_class_valid_mask(FSQ_LEVELS, MAX_FSQ_CLASSES),
        config=flax.core.FrozenDict(**cfg),
    )


def _synthetic_batch(agent, *, batch_size=None, obs_dim=8, act_dim=3):
    h = int(agent.config["h"])
    b = int(batch_size if batch_size is not None else agent.config["batch_size"])
    # Trajectories are (H+1, B, ...) for obs/masks/terminals/rewards; actions (H, B, D).
    return {
        "observations": jnp.zeros((h + 1, b, obs_dim), dtype=jnp.float32),
        "actions": jnp.zeros((h, b, act_dim), dtype=jnp.float32),
        "rewards": jnp.zeros((h + 1, b), dtype=jnp.float32),
        "terminals": jnp.zeros((h + 1, b), dtype=jnp.float32),
        "masks": jnp.ones((h + 1, b), dtype=jnp.float32),
    }


def test_update_total_loss_no_q_axis_shadow():
    """Regression: FSQ axis count must not be overwritten by Q outputs in total_loss.

    Stock update() JITs total_loss; range(Q_array) previously raised TypeError.
    """
    agent = _make_tiny_agent_for_update()
    batch = _synthetic_batch(agent)
    new_agent, info = agent.update(batch)
    loss = float(np.asarray(info["total_loss"]))
    assert np.isfinite(loss)
    assert "axis0_acc_masked" in info
    assert "axis3_acc_masked" in info
    assert float(np.asarray(info["q_mean"])) == pytest.approx(
        float(np.asarray(info["q"]))
    )
    # Second update confirms post-update targets + re-JIT stay healthy.
    _, info2 = new_agent.update(batch)
    assert np.isfinite(float(np.asarray(info2["total_loss"])))
