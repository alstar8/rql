"""Focused unit tests for DiscreteARIQL (v6) helpers and actor contracts."""

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

from agents.discrete_ar_iql import (  # noqa: E402
    BOS_ID,
    CausalARActor,
    DEFAULT_EMPIRICAL_REGISTER_CE_WEIGHTS_K16,
    DiscreteARIQLAgent,
    get_config,
)
from agents.oattok_jax import (  # noqa: E402
    CODEBOOK_SIZE,
    FSQ_LEVELS,
    OATTok,
    build_codebook,
    indices_to_codes,
)
from utils.flax_utils import ModuleDict, TrainState  # noqa: E402
from utils.networks import Value  # noqa: E402

Agent = DiscreteARIQLAgent


def test_teacher_inputs_bos_alignment_no_target_leakage():
    tokens = jnp.array([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=jnp.int32)
    inp = Agent.make_teacher_inputs(tokens, bos_id=BOS_ID)
    assert inp.shape == tokens.shape
    np.testing.assert_array_equal(np.asarray(inp[:, 0]), BOS_ID)
    np.testing.assert_array_equal(np.asarray(inp[:, 1:]), np.asarray(tokens[:, :-1]))
    # Position i must not contain target z_i (except coincidental equal values).
    for i in range(tokens.shape[-1]):
        # Input at i is BOS or z_{i-1}, never z_i by construction of the shift.
        if i == 0:
            assert int(inp[0, i]) == BOS_ID
        else:
            assert int(inp[0, i]) == int(tokens[0, i - 1])


def test_causal_mask_future_input_invariance():
    """Output at position i is invariant to changes in future input tokens."""
    k, emb, depth, heads = 4, 32, 2, 2
    actor = CausalARActor(
        emb_dim=emb, depth=depth, num_heads=heads, num_registers=k, dropout=0.0
    )
    b, obs_dim = 3, 8
    obs = jax.random.normal(jax.random.PRNGKey(0), (b, obs_dim))
    base_inp = jnp.full((b, k), BOS_ID, dtype=jnp.int32)
    base_inp = base_inp.at[:, 1].set(7)
    base_inp = base_inp.at[:, 2].set(11)
    base_inp = base_inp.at[:, 3].set(13)

    variables = actor.init(jax.random.PRNGKey(1), obs, base_inp)
    logits0 = actor.apply(variables, obs, base_inp)

    # Perturb only future inputs relative to position i=1 (index 1).
    pert = base_inp.at[:, 2:].set(999)  # 999 is a valid embed id (BOS=1000)
    logits1 = actor.apply(variables, obs, pert)
    # Positions 0 and 1 must be unchanged; later positions may change.
    np.testing.assert_allclose(
        np.asarray(logits0[:, :2]), np.asarray(logits1[:, :2]), atol=1e-5
    )
    # Sanity: something after position 1 can differ.
    assert not np.allclose(
        np.asarray(logits0[:, 2:]), np.asarray(logits1[:, 2:]), atol=1e-5
    )


def test_actor_output_shapes_vocab():
    k = 4
    actor = CausalARActor(
        emb_dim=32, depth=1, num_heads=2, num_registers=k, dropout=0.0
    )
    obs = jnp.zeros((2, 6), dtype=jnp.float32)
    inp = jnp.full((2, k), BOS_ID, dtype=jnp.int32)
    variables = actor.init(jax.random.PRNGKey(0), obs, inp)
    logits = actor.apply(variables, obs, inp)
    assert logits.shape == (2, k, CODEBOOK_SIZE)


def test_ce_weighting_and_prefix_metrics():
    logits = jnp.zeros((2, 4, CODEBOOK_SIZE))
    # Make argmax = targets for example 0 fully; example 1 only first 2.
    targets = jnp.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
    logits = logits.at[0, 0, 1].set(10.0)
    logits = logits.at[0, 1, 2].set(10.0)
    logits = logits.at[0, 2, 3].set(10.0)
    logits = logits.at[0, 3, 4].set(10.0)
    logits = logits.at[1, 0, 5].set(10.0)
    logits = logits.at[1, 1, 6].set(10.0)
    # Wrong on last two for example 1.
    logits = logits.at[1, 2, 0].set(10.0)
    logits = logits.at[1, 3, 0].set(10.0)

    pred = jnp.argmax(logits, axis=-1)
    prefix = Agent.prefix_exact_rates(pred, targets, lengths=(1, 2, 4))
    assert float(prefix[1]) == pytest.approx(1.0)
    assert float(prefix[2]) == pytest.approx(1.0)
    assert float(prefix[4]) == pytest.approx(0.5)
    assert float(prefix["seq_exact"]) == pytest.approx(0.5)

    ce = Agent.token_ce(logits, targets)
    # Perfect example has near-zero CE; imperfect has higher mean CE.
    assert float(ce[0].mean()) < float(ce[1].mean())

    # Weighted CE: heavier weight on high-CE example increases loss.
    ce_ex = ce.mean(axis=-1)
    w_u = jnp.ones(2)
    w_a = jnp.array([1.0, 5.0])
    loss_u = (ce_ex * w_u).sum() / w_u.sum()
    loss_a = (ce_ex * w_a).sum() / w_a.sum()
    assert float(loss_a) > float(loss_u)


def test_iql_awr_helpers():
    pos = Agent.expectile_loss(jnp.array(2.0), 0.9)
    neg = Agent.expectile_loss(jnp.array(-2.0), 0.9)
    assert float(pos) == pytest.approx(0.9 * 4.0)
    assert float(neg) == pytest.approx(0.1 * 4.0)

    warmup, ramp = 50_000, 50_000
    assert float(Agent.awr_ramp_fraction(0, warmup, ramp)) == pytest.approx(0.0)
    assert float(Agent.awr_ramp_fraction(warmup + ramp // 2, warmup, ramp)) == pytest.approx(
        0.5, abs=1e-5
    )
    assert float(Agent.awr_ramp_fraction(warmup + ramp, warmup, ramp)) == pytest.approx(1.0)

    adv = jnp.array([0.0, 1000.0, -1000.0])
    w0 = Agent.awr_example_weights(adv, temperature=5.0, max_weight=100.0, ramp_frac=0.0)
    np.testing.assert_allclose(np.asarray(w0), 1.0)
    w1 = Agent.awr_example_weights(adv, temperature=5.0, max_weight=100.0, ramp_frac=1.0)
    assert float(w1[1]) == pytest.approx(100.0)
    assert float(Agent.awr_ess(w1)) > 0.0


def test_h_step_masks_and_post_update_target():
    h = 1
    rewards = jnp.array([[1.0, 2.0], [0.0, 0.0]])
    terminals = jnp.array([[0.0, 0.0], [0.0, 1.0]])
    masks = jnp.array([[1.0, 1.0], [1.0, 0.0]])
    discount = 0.995
    discount_mul = discount ** jnp.array([0.0, jnp.inf])
    next_v = jnp.array([10.0, 20.0])
    target, valids, rs = Agent.h_step_td_target(
        rewards, terminals, masks, discount, discount_mul, h, next_v
    )
    np.testing.assert_allclose(np.asarray(rs[0]), 0.0)
    expected0 = 1.0 + (discount**h) * 10.0 * float(masks[-2, 0])
    assert float(target[0]) == pytest.approx(expected0, abs=1e-5)
    w1 = Agent.chunk_ce_weights(rs, h=1)
    np.testing.assert_allclose(np.asarray(w1), 1.0)

    rs3 = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ]
    )
    w3 = Agent.chunk_ce_weights(rs3, h=3)
    np.testing.assert_allclose(np.asarray(w3), np.array([0.0, 1.0]))

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


def test_config_defaults():
    cfg = get_config()
    assert cfg.agent_name == "discrete_ar_iql"
    assert cfg.expectile == 0.9
    assert cfg.awr_temperature == 5.0
    assert cfg.max_weight == 100.0
    assert cfg.bc_warmup_steps == 50_000
    assert cfg.awr_ramp_steps == 50_000
    assert cfg.eval_sampling_temperature == 1.0
    assert cfg.actor_emb_dim == 256
    assert cfg.actor_depth == 4
    assert cfg.actor_num_heads == 4
    assert cfg.actor_dropout == 0.0
    assert cfg.ensemble_ct == 2
    assert cfg.discount == 0.995
    assert cfg.advantage_source == "iql"
    assert cfg.use_mc_returns is False
    assert cfg.mc_expectile == 0.7
    assert cfg.mc_actor_warmup_steps == 0
    assert cfg.use_trajectory_success is False
    assert cfg.success_weight == 20.0
    assert cfg.success_actor_warmup_steps == 0
    assert cfg.ss_start_steps == 50_000
    assert cfg.ss_ramp_steps == 50_000
    assert cfg.ss_loss_coef == 0.5
    assert cfg.ss_prefix_prob_min == 0.1
    assert cfg.ss_prefix_prob_max == 0.35
    assert cfg.ss_pred_mode == "argmax"
    assert cfg.use_register_weights is True
    assert len(cfg.register_ce_weights) == 16
    assert cfg.q_actor_coef == 0.0
    assert cfg.q_actor_warmup_steps == 50_000
    assert cfg.q_actor_ramp_steps == 50_000
    assert cfg.q_actor_prefix_mode == "teacher_forced"
    assert cfg.st_temperature == 1.0


def test_validate_create_config_emb_heads_and_dropout():
    cfg = dict(get_config())
    Agent.validate_create_config(cfg)  # defaults ok

    bad_heads = dict(cfg)
    bad_heads["actor_emb_dim"] = 32
    bad_heads["actor_num_heads"] = 3
    with pytest.raises(ValueError, match="divisible"):
        Agent.validate_create_config(bad_heads)

    bad_drop = dict(cfg)
    bad_drop["actor_dropout"] = 0.1
    with pytest.raises(ValueError, match="actor_dropout"):
        Agent.validate_create_config(bad_drop)

    bad_zero_heads = dict(cfg)
    bad_zero_heads["actor_num_heads"] = 0
    with pytest.raises(ValueError, match="actor_num_heads"):
        Agent.validate_create_config(bad_zero_heads)

    bad_src = dict(cfg)
    bad_src["advantage_source"] = "td_lambda"
    with pytest.raises(ValueError, match="advantage_source"):
        Agent.validate_create_config(bad_src)

    mc_no_flag = dict(cfg)
    mc_no_flag["advantage_source"] = "mc_return"
    mc_no_flag["use_mc_returns"] = False
    with pytest.raises(ValueError, match="use_mc_returns"):
        Agent.validate_create_config(mc_no_flag)

    mc_ok = dict(cfg)
    mc_ok["advantage_source"] = "mc_return"
    mc_ok["use_mc_returns"] = True
    Agent.validate_create_config(mc_ok)

    traj_no_flag = dict(cfg)
    traj_no_flag["advantage_source"] = "trajectory_success"
    traj_no_flag["use_trajectory_success"] = False
    with pytest.raises(ValueError, match="use_trajectory_success"):
        Agent.validate_create_config(traj_no_flag)

    traj_bad_w = dict(cfg)
    traj_bad_w["advantage_source"] = "trajectory_success"
    traj_bad_w["use_trajectory_success"] = True
    traj_bad_w["success_weight"] = 0.0
    with pytest.raises(ValueError, match="success_weight"):
        Agent.validate_create_config(traj_bad_w)

    traj_bad_warmup = dict(cfg)
    traj_bad_warmup["advantage_source"] = "trajectory_success"
    traj_bad_warmup["use_trajectory_success"] = True
    traj_bad_warmup["success_actor_warmup_steps"] = -1
    with pytest.raises(ValueError, match="success_actor_warmup_steps"):
        Agent.validate_create_config(traj_bad_warmup)

    traj_ok = dict(cfg)
    traj_ok["advantage_source"] = "trajectory_success"
    traj_ok["use_trajectory_success"] = True
    traj_ok["success_weight"] = 20.0
    Agent.validate_create_config(traj_ok)

    bad_prefix = dict(cfg)
    bad_prefix["q_actor_prefix_mode"] = "freerun"
    with pytest.raises(ValueError, match="q_actor_prefix_mode"):
        Agent.validate_create_config(bad_prefix)

    bad_st = dict(cfg)
    bad_st["st_temperature"] = 0.0
    with pytest.raises(ValueError, match="st_temperature"):
        Agent.validate_create_config(bad_st)

    bad_qcoef = dict(cfg)
    bad_qcoef["q_actor_coef"] = -1.0
    with pytest.raises(ValueError, match="q_actor_coef"):
        Agent.validate_create_config(bad_qcoef)

    q_ok = dict(cfg)
    q_ok["q_actor_coef"] = 1.0
    q_ok["q_actor_prefix_mode"] = "self_conditioned"
    q_ok["st_temperature"] = 0.5
    Agent.validate_create_config(q_ok)


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
            # Distinct tokens so AR diversity is visible after decode.
            tokens = (
                jnp.arange(b * self.num_registers, dtype=jnp.int32).reshape(
                    b, self.num_registers
                )
                % CODEBOOK_SIZE
            )
            codes = indices_to_codes(tokens, FSQ_LEVELS)
            return codes, tokens
        if method is OATTok.decode:
            b = x.shape[0]
            flat = x.reshape(b, -1).astype(jnp.float32)
            need = self.sample_horizon * self.sample_dim
            if flat.shape[-1] < need:
                reps = int(np.ceil(need / flat.shape[-1]))
                flat = jnp.tile(flat, (1, reps))
            out = flat[:, :need].reshape(b, self.sample_horizon, self.sample_dim)
            return jnp.tanh(out)
        raise ValueError(f"Unexpected stub method: {method}")


def _make_tiny_agent(
    *,
    batch_size: int = 4,
    obs_dim: int = 8,
    act_dim: int = 3,
    h: int = 1,
    num_registers: int = 4,
    eval_sampling_temperature: float = 1.0,
    ss_start_steps: int = 10**9,
    ss_ramp_steps: int = 50_000,
    ss_loss_coef: float = 0.5,
    ss_prefix_prob_min: float = 0.1,
    ss_prefix_prob_max: float = 0.35,
    ss_pred_mode: str = "argmax",
    use_register_weights: bool = False,
    register_ce_weights=None,
    network_step: int = 0,
    advantage_source: str = "iql",
    use_mc_returns: bool = False,
    mc_expectile: float = 0.7,
    mc_actor_warmup_steps: int = 0,
    use_trajectory_success: bool = False,
    success_weight: float = 20.0,
    success_actor_warmup_steps: int = 0,
    q_actor_coef: float = 0.0,
    q_actor_warmup_steps: int = 50_000,
    q_actor_ramp_steps: int = 50_000,
    q_actor_prefix_mode: str = "teacher_forced",
    st_temperature: float = 1.0,
    rho: float = 0.0,
):
    cfg = dict(get_config())
    if register_ce_weights is None:
        register_ce_weights = DEFAULT_EMPIRICAL_REGISTER_CE_WEIGHTS_K16
    cfg.update(
        dict(
            h=h,
            batch_size=batch_size,
            num_registers=num_registers,
            actor_emb_dim=32,
            actor_depth=1,
            actor_num_heads=2,
            value_hidden_dims=(32, 32),
            ensemble_ct=2,
            v_ensemble_ct=1,
            bc_warmup_steps=0,
            awr_ramp_steps=1,
            discount=0.99,
            eval_sampling_temperature=eval_sampling_temperature,
            ss_start_steps=ss_start_steps,
            ss_ramp_steps=ss_ramp_steps,
            ss_loss_coef=ss_loss_coef,
            ss_prefix_prob_min=ss_prefix_prob_min,
            ss_prefix_prob_max=ss_prefix_prob_max,
            ss_pred_mode=ss_pred_mode,
            use_register_weights=use_register_weights,
            register_ce_weights=tuple(register_ce_weights),
            advantage_source=advantage_source,
            use_mc_returns=use_mc_returns,
            mc_expectile=mc_expectile,
            mc_actor_warmup_steps=mc_actor_warmup_steps,
            use_trajectory_success=use_trajectory_success,
            success_weight=success_weight,
            success_actor_warmup_steps=success_actor_warmup_steps,
            q_actor_coef=q_actor_coef,
            q_actor_warmup_steps=q_actor_warmup_steps,
            q_actor_ramp_steps=q_actor_ramp_steps,
            q_actor_prefix_mode=q_actor_prefix_mode,
            st_temperature=st_temperature,
            rho=rho,
        )
    )
    cfg = Agent.finalize_register_weight_config(cfg, num_registers)
    Agent.validate_create_config(cfg)
    action_dim = act_dim * h
    cfg["action_dim"] = action_dim
    cfg["prim_action_dim"] = act_dim
    cfg["discount_mul"] = jnp.array(
        cfg["discount"] ** jnp.array(list(range(h)) + [jnp.inf])
    )

    ex_obs = jnp.zeros((1, obs_dim), dtype=jnp.float32)
    ex_flat_actions = jnp.zeros((1, action_dim), dtype=jnp.float32)
    ex_token_inputs = jnp.full((1, num_registers), BOS_ID, dtype=jnp.int32)
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
    actor_def = CausalARActor(
        emb_dim=cfg["actor_emb_dim"],
        depth=cfg["actor_depth"],
        num_heads=cfg["actor_num_heads"],
        num_registers=num_registers,
        dropout=0.0,
    )
    network_info = dict(
        q=(q_def, (ex_q_in,)),
        target_q=(copy.deepcopy(q_def), (ex_q_in,)),
        v=(v_def, (ex_obs,)),
        target_v=(copy.deepcopy(v_def), (ex_obs,)),
        actor=(actor_def, (ex_obs, ex_token_inputs)),
        target_actor=(copy.deepcopy(actor_def), (ex_obs, ex_token_inputs)),
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
    if network_step:
        network = network.replace(step=network_step)

    return Agent(
        rng=jax.random.PRNGKey(1),
        network=network,
        tokenizer_def=_StubOATTok(num_registers, h, act_dim),
        tokenizer_params={},
        codebook=build_codebook(FSQ_LEVELS),
        config=flax.core.FrozenDict(**cfg),
    )


def _synthetic_batch(
    agent, *, batch_size=None, obs_dim=8, act_dim=3, with_mc=False, with_traj_success=False
):
    h = int(agent.config["h"])
    b = int(batch_size if batch_size is not None else agent.config["batch_size"])
    batch = {
        "observations": jnp.zeros((h + 1, b, obs_dim), dtype=jnp.float32),
        "actions": jnp.zeros((h, b, act_dim), dtype=jnp.float32),
        "rewards": jnp.zeros((h + 1, b), dtype=jnp.float32),
        "terminals": jnp.zeros((h + 1, b), dtype=jnp.float32),
        "masks": jnp.ones((h + 1, b), dtype=jnp.float32),
    }
    if with_mc:
        # Distinct per-example MC returns (large negative offset like maze RTG).
        g0 = jnp.linspace(-200.0, -100.0, b, dtype=jnp.float32)
        batch["mc_returns"] = jnp.stack([g0, g0], axis=0)[: h + 1]
    if with_traj_success:
        # ~7.7% success: first ceil(0.077*B) examples flagged.
        n_succ = max(1, int(np.ceil(0.077 * b)))
        flag0 = jnp.zeros((b,), dtype=jnp.float32).at[:n_succ].set(1.0)
        batch["trajectory_success"] = jnp.stack([flag0, flag0], axis=0)[: h + 1]
    return batch


def test_ar_inference_fills_k_seed_repro_diversity_argmax():
    agent = _make_tiny_agent(eval_sampling_temperature=1.0)
    obs = jnp.zeros((8,), dtype=jnp.float32)

    # temperature=0 → target actor + eval_sampling_temperature=1 → stochastic.
    seed = jax.random.PRNGKey(123)
    a1 = agent.sample_actions(obs, seed=seed, temperature=0.0)
    a2 = agent.sample_actions(obs, seed=seed, temperature=0.0)
    a3 = agent.sample_actions(obs, seed=jax.random.PRNGKey(456), temperature=0.0)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))
    assert not np.array_equal(np.asarray(a1), np.asarray(a3))
    assert a1.shape == (1, 3)  # h=1, act_dim=3

    # Argmax path via eval_sampling_temperature=0.
    agent_det = _make_tiny_agent(eval_sampling_temperature=0.0)
    d1 = agent_det.sample_actions(obs, seed=jax.random.PRNGKey(1), temperature=0.0)
    d2 = agent_det.sample_actions(obs, seed=jax.random.PRNGKey(99), temperature=0.0)
    np.testing.assert_array_equal(np.asarray(d1), np.asarray(d2))

    # External temperature>0 uses online actor + that temp (still seed-repro).
    b1 = agent.sample_actions(obs, seed=seed, temperature=1.0)
    b2 = agent.sample_actions(obs, seed=seed, temperature=1.0)
    np.testing.assert_array_equal(np.asarray(b1), np.asarray(b2))

    # Direct AR fill returns clipped flat actions of length h*Da.
    flat = agent.compute_ar_actions(obs[None, :], seed=seed, temperature=0.0)
    assert flat.shape == (1, 3)


def test_seed_none_does_not_mutate_rng():
    agent = _make_tiny_agent()
    obs = jnp.zeros((8,), dtype=jnp.float32)
    rng_before = np.asarray(agent.rng)
    x1 = agent.sample_actions(obs, seed=None, temperature=1.0)
    x2 = agent.sample_actions(obs, seed=None, temperature=1.0)
    np.testing.assert_array_equal(np.asarray(x1), np.asarray(x2))
    np.testing.assert_array_equal(np.asarray(agent.rng), rng_before)


def test_update_stock_regression_stub_tokenizer():
    agent = _make_tiny_agent()
    batch = _synthetic_batch(agent)
    new_agent, info = agent.update(batch)
    loss = float(np.asarray(info["total_loss"]))
    assert np.isfinite(loss)
    assert "token_acc" in info
    assert "seq_exact" in info
    assert "prefix_exact_1" in info
    assert "reg0_ce" in info
    assert "awr_ess" in info
    assert "decode_rmse" in info
    assert "ss_coef" in info
    assert "ss_prefix_prob" in info
    assert "ss_replaced_frac" in info
    assert "ss_token_acc" in info
    assert "tf_actor_loss" in info
    assert "ss_actor_loss" in info
    assert "register_weight_min" in info
    assert "register_weight_max" in info
    assert float(np.asarray(info["ss_coef"])) == pytest.approx(0.0)
    assert float(np.asarray(info["q_mean"])) == pytest.approx(
        float(np.asarray(info["q"]))
    )
    _, info2 = new_agent.update(batch)
    assert np.isfinite(float(np.asarray(info2["total_loss"])))


def test_ss_shifted_replacement_alignment_never_bos_or_future():
    tokens = jnp.array([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=jnp.int32)
    teacher = Agent.make_teacher_inputs(tokens, bos_id=BOS_ID)
    pred = jnp.array([[90, 91, 92, 93], [80, 81, 82, 83]], dtype=jnp.int32)
    # Replace all non-BOS positions.
    replace = jnp.ones_like(teacher, dtype=bool)
    mixed = Agent.make_scheduled_sampling_inputs(teacher, pred, replace)
    np.testing.assert_array_equal(np.asarray(mixed[:, 0]), BOS_ID)
    # Position i>0 must equal pred[:, i-1], never pred[:, i] (future) or tokens[:, i].
    np.testing.assert_array_equal(np.asarray(mixed[:, 1:]), np.asarray(pred[:, :-1]))
    for i in range(1, tokens.shape[-1]):
        assert int(mixed[0, i]) == int(pred[0, i - 1])
        assert int(mixed[0, i]) != int(pred[0, i]) or i == tokens.shape[-1] - 1
    # Partial mask: only position 2 replaced.
    replace2 = jnp.zeros_like(teacher, dtype=bool).at[:, 2].set(True)
    mixed2 = Agent.make_scheduled_sampling_inputs(teacher, pred, replace2)
    np.testing.assert_array_equal(np.asarray(mixed2[:, 0]), BOS_ID)
    np.testing.assert_array_equal(np.asarray(mixed2[:, 1]), np.asarray(teacher[:, 1]))
    np.testing.assert_array_equal(np.asarray(mixed2[:, 2]), np.asarray(pred[:, 1]))
    np.testing.assert_array_equal(np.asarray(mixed2[:, 3]), np.asarray(teacher[:, 3]))


def test_ss_schedule_endpoints():
    start, ramp = 50_000, 50_000
    max_coef, p_min, p_max = 0.5, 0.1, 0.35
    c0, p0 = Agent.ss_schedule(0, start, ramp, max_coef, p_min, p_max)
    assert float(c0) == pytest.approx(0.0)
    assert float(p0) == pytest.approx(0.0)
    c_start, p_start = Agent.ss_schedule(start, start, ramp, max_coef, p_min, p_max)
    assert float(c_start) == pytest.approx(0.0)
    assert float(p_start) == pytest.approx(0.0)
    mid = start + ramp // 2
    c_mid, p_mid = Agent.ss_schedule(mid, start, ramp, max_coef, p_min, p_max)
    assert float(c_mid) == pytest.approx(0.25, abs=1e-5)
    assert float(p_mid) == pytest.approx(0.1 + 0.5 * 0.25, abs=1e-5)
    c_end, p_end = Agent.ss_schedule(
        start + ramp, start, ramp, max_coef, p_min, p_max
    )
    assert float(c_end) == pytest.approx(0.5)
    assert float(p_end) == pytest.approx(0.35)
    # Resumed 200k sits at ceiling.
    c200, p200 = Agent.ss_schedule(200_000, start, ramp, max_coef, p_min, p_max)
    assert float(c200) == pytest.approx(0.5)
    assert float(p200) == pytest.approx(0.35)


def test_register_weights_normalization_and_length_fallback():
    w = Agent.resolve_register_ce_weights(
        16, True, DEFAULT_EMPIRICAL_REGISTER_CE_WEIGHTS_K16
    )
    assert w.shape == (16,)
    assert float(w.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(w.min()) > 0.0
    # Wrong length → uniform (never silent misuse).
    w4 = Agent.resolve_register_ce_weights(
        4, True, DEFAULT_EMPIRICAL_REGISTER_CE_WEIGHTS_K16
    )
    np.testing.assert_allclose(np.asarray(w4), 1.0)
    w_off = Agent.resolve_register_ce_weights(16, False, DEFAULT_EMPIRICAL_REGISTER_CE_WEIGHTS_K16)
    np.testing.assert_allclose(np.asarray(w_off), 1.0)
    cfg = dict(get_config())
    cfg["num_registers"] = 4
    cfg["use_register_weights"] = True
    cfg["register_ce_weights"] = DEFAULT_EMPIRICAL_REGISTER_CE_WEIGHTS_K16
    cfg = Agent.finalize_register_weight_config(cfg, 4)
    assert cfg["use_register_weights"] is False


def test_lambda0_equivalence_no_nan_contamination():
    """λ=0 must match TF-only actor loss and stay finite (skip pass B)."""
    agent = _make_tiny_agent(
        ss_start_steps=10**9,
        ss_loss_coef=0.5,
        use_register_weights=False,
    )
    batch = _synthetic_batch(agent)
    rng = jax.random.PRNGKey(0)
    loss, info = agent.total_loss(batch, agent.network.params, rng=rng)
    assert np.isfinite(float(np.asarray(loss)))
    assert float(np.asarray(info["ss_coef"])) == pytest.approx(0.0)
    assert float(np.asarray(info["ss_actor_loss"])) == pytest.approx(0.0)
    assert float(np.asarray(info["actor_loss"])) == pytest.approx(
        float(np.asarray(info["tf_actor_loss"])), abs=1e-5
    )
    # Explicit max coef 0 also TF-only even past start.
    agent2 = _make_tiny_agent(
        ss_start_steps=0,
        ss_ramp_steps=1,
        ss_loss_coef=0.0,
        network_step=100,
        use_register_weights=False,
    )
    loss2, info2 = agent2.total_loss(batch, agent2.network.params, rng=rng)
    assert np.isfinite(float(np.asarray(loss2)))
    assert float(np.asarray(info2["ss_coef"])) == pytest.approx(0.0)
    assert float(np.asarray(info2["actor_loss"])) == pytest.approx(
        float(np.asarray(info2["tf_actor_loss"])), abs=1e-5
    )


def test_noisy_prefix_ss_changes_loss_and_gradients():
    """Active SS with p>0 must change actor loss and actor grads vs TF-only."""
    k = 4
    # Distinct per-register weights so weighting is exercised when enabled.
    reg_w = (0.5, 1.0, 1.5, 1.0)
    agent_tf = _make_tiny_agent(
        num_registers=k,
        ss_start_steps=10**9,
        use_register_weights=True,
        register_ce_weights=reg_w,
    )
    agent_ss = _make_tiny_agent(
        num_registers=k,
        ss_start_steps=0,
        ss_ramp_steps=1,
        ss_loss_coef=1.0,
        ss_prefix_prob_min=1.0,
        ss_prefix_prob_max=1.0,
        network_step=10,
        use_register_weights=True,
        register_ce_weights=reg_w,
        ss_pred_mode="argmax",
    )
    # Share params so the only difference is SS path.
    agent_ss = agent_ss.replace(network=agent_tf.network.replace(step=10))
    batch = _synthetic_batch(agent_tf)
    rng = jax.random.PRNGKey(7)

    loss_tf, info_tf = agent_tf.total_loss(batch, agent_tf.network.params, rng=rng)
    loss_ss, info_ss = agent_ss.total_loss(batch, agent_ss.network.params, rng=rng)
    assert float(np.asarray(info_ss["ss_coef"])) == pytest.approx(1.0)
    assert float(np.asarray(info_ss["ss_prefix_prob"])) == pytest.approx(1.0)
    assert float(np.asarray(info_ss["ss_replaced_frac"])) > 0.0
    # Full replace of previous tokens → SS loss generally differs from TF.
    assert float(np.asarray(info_ss["ss_actor_loss"])) != pytest.approx(
        float(np.asarray(info_tf["tf_actor_loss"])), abs=1e-6
    ) or float(np.asarray(info_ss["actor_loss"])) != pytest.approx(
        float(np.asarray(info_tf["actor_loss"])), abs=1e-6
    )

    def _actor_loss_only(agent, params, rng_key):
        _, info = agent.total_loss(batch, params, rng=rng_key)
        return info["actor_loss"]

    g_tf = jax.grad(lambda p: _actor_loss_only(agent_tf, p, rng))(
        agent_tf.network.params
    )
    g_ss = jax.grad(lambda p: _actor_loss_only(agent_ss, p, rng))(
        agent_ss.network.params
    )
    leaf_tf = jax.tree_util.tree_leaves(g_tf["modules_actor"])
    leaf_ss = jax.tree_util.tree_leaves(g_ss["modules_actor"])
    diff = sum(float(np.sum(np.abs(np.asarray(a) - np.asarray(b)))) for a, b in zip(leaf_tf, leaf_ss))
    assert diff > 1e-8
    assert np.isfinite(float(np.asarray(loss_tf)))
    assert np.isfinite(float(np.asarray(loss_ss)))


def test_checkpoint_param_tree_unchanged_by_ss_config():
    """SS is loss-only: actor param tree matches regardless of SS schedule."""
    a0 = _make_tiny_agent(ss_start_steps=10**9)
    a1 = _make_tiny_agent(ss_start_steps=0, network_step=100_000)
    flat0, _ = jax.tree_util.tree_flatten_with_path(a0.network.params)
    flat1, _ = jax.tree_util.tree_flatten_with_path(a1.network.params)
    paths0 = [tuple(map(str, p)) for p, _ in flat0]
    paths1 = [tuple(map(str, p)) for p, _ in flat1]
    assert paths0 == paths1


def test_registry_import():
    from agents import agents as agent_registry

    assert "discrete_ar_iql" in agent_registry
    assert agent_registry["discrete_ar_iql"] is DiscreteARIQLAgent


def test_diagnose_teacher_vs_freerun_keys_and_finite():
    agent = _make_tiny_agent(eval_sampling_temperature=0.0)
    batch = _synthetic_batch(agent)
    info = agent.diagnose_teacher_vs_freerun(
        batch["observations"][0],
        batch["actions"],
        seed=jax.random.PRNGKey(0),
        temperature=0.0,
        force_argmax=True,
    )
    for key in (
        "tf_token_acc",
        "fr_token_acc",
        "tf_seq_exact",
        "fr_seq_exact",
        "tf_prefix_exact_1",
        "fr_prefix_exact_1",
        "tf_reg0_acc",
        "fr_reg0_acc",
        "fr_action_rmse_gt",
        "fr_action_corr_gt",
        "fr_action_rmse_clean",
        "fr_action_corr_clean",
        "decode_rmse",
    ):
        assert key in info
        assert np.isfinite(float(np.asarray(info[key])))
    # Argmax freerun is deterministic for fixed params + force_argmax.
    info2 = agent.diagnose_teacher_vs_freerun(
        batch["observations"][0],
        batch["actions"],
        seed=jax.random.PRNGKey(99),
        temperature=0.0,
        force_argmax=True,
    )
    np.testing.assert_allclose(
        float(np.asarray(info["fr_token_acc"])),
        float(np.asarray(info2["fr_token_acc"])),
    )


def test_plot_build_series_dari_wiring():
    sys.path.insert(0, str(ROOT / "scripts"))
    from plot_dflrql_vs_baseline import (  # noqa: E402
        CORE_SCRATCH_GROUPS,
        DEFAULT_AR_QDFL_PHASE1_GROUP,
        DEFAULT_CDF_SCRATCH_GROUP,
        DEFAULT_DARI_RUN_GROUP,
        DEFAULT_DARI_SCRATCH_GROUP,
        DEFAULT_DCMI_RUN_GROUP,
        DEFAULT_DD_QDFL_PHASE1_GROUP,
        DEFAULT_DFLRQL9_SCRATCH_GROUP,
        DEFAULT_PLOT_MAX_STEP,
        DEFAULT_QUANTIZED_DFLRQL9_SCRATCH_GROUP,
        DEFAULT_RQL_QFLOW_PHASE1_GROUP,
        DEFAULT_RQL_QFLOW_PHASE2_GROUP,
        ONLINE_START_STEP,
        PURE_QFLOW_ONLINE_START_STEP,
        PURE_QFLOW_PLACEHOLDER,
        QFLOW_RQL_WARMSTART_V2_PLACEHOLDER,
        RQL_QFLOW_ONLINE_PLACEHOLDER,
        build_series,
    )

    # Default core: RQL reference + six from-scratch 2M offline actors only.
    core = build_series(
        "clf-group",
        all_methods=False,
        dcmi_run_group=DEFAULT_DCMI_RUN_GROUP,
        dari_run_group="humanoidmaze-large-dari-v6-2m",
    )
    core_labels = [s[0] for s in core]
    core_groups = [s[1] for s in core]
    assert core_labels == [
        "RQL baseline",
        "DFL-RQL v9",
        "Quantized DFL-RQL v9",
        "DARI (AR OAT)",
        "CDF (discrete FM)",
        "AR QDFL student",
        "DD QDFL student",
    ]
    assert core_groups == [
        "humanoidmaze-large-rql-tuned-2m",
        DEFAULT_DFLRQL9_SCRATCH_GROUP,
        DEFAULT_QUANTIZED_DFLRQL9_SCRATCH_GROUP,
        DEFAULT_DARI_SCRATCH_GROUP,
        DEFAULT_CDF_SCRATCH_GROUP,
        DEFAULT_AR_QDFL_PHASE1_GROUP,
        DEFAULT_DD_QDFL_PHASE1_GROUP,
    ]
    assert core_groups[1:7] == list(CORE_SCRATCH_GROUPS)
    # Direct scratch groups — not piecewise placeholders.
    assert "__quantized_v9__" not in core_groups
    assert "__ar_qdfl_distill__" not in core_groups
    assert "__dd_qdfl_distill__" not in core_groups
    assert RQL_QFLOW_ONLINE_PLACEHOLDER not in core_groups
    assert QFLOW_RQL_WARMSTART_V2_PLACEHOLDER not in core_groups
    assert PURE_QFLOW_PLACEHOLDER not in core_groups
    assert "DARI v6" not in core_labels
    assert "DCMI v5" not in core_labels
    assert DEFAULT_DARI_RUN_GROUP == "humanoidmaze-large-dari-v6-2m"
    assert DEFAULT_DFLRQL9_SCRATCH_GROUP == "humanoidmaze-large-dflrql9-scratch-2m"
    assert (
        DEFAULT_QUANTIZED_DFLRQL9_SCRATCH_GROUP
        == "humanoidmaze-large-quantized-dflrql9-scratch-2m"
    )
    assert DEFAULT_DARI_SCRATCH_GROUP == "humanoidmaze-large-dari-scratch-2m"
    assert DEFAULT_CDF_SCRATCH_GROUP == "humanoidmaze-large-cdf-scratch-2m"
    assert (
        DEFAULT_AR_QDFL_PHASE1_GROUP
        == "humanoidmaze-large-discrete-ar-qdfl-distill-2m"
    )
    assert (
        DEFAULT_DD_QDFL_PHASE1_GROUP
        == "humanoidmaze-large-discrete-diffusion-qdfl-distill-2m"
    )
    assert DEFAULT_RQL_QFLOW_PHASE1_GROUP == "humanoidmaze-large-rql-qflow-ready-2m"
    assert DEFAULT_RQL_QFLOW_PHASE2_GROUP == "humanoidmaze-large-rql-qflow-online-4m"
    assert ONLINE_START_STEP == 2_000_000
    assert PURE_QFLOW_ONLINE_START_STEP == 1_000_000
    assert DEFAULT_PLOT_MAX_STEP == 2_000_000

    # --with-extras / with_extras=True wires historical DARI v6 (and DCMI).
    extras = build_series(
        "clf-group",
        all_methods=False,
        dcmi_run_group=DEFAULT_DCMI_RUN_GROUP,
        dari_run_group="humanoidmaze-large-dari-v6-2m",
        with_extras=True,
    )
    extras_labels = [s[0] for s in extras]
    extras_groups = {s[0]: s[1] for s in extras}
    assert "DARI v6" in extras_labels
    assert "DCMI v5" in extras_labels
    assert extras_groups["DARI v6"] == "humanoidmaze-large-dari-v6-2m"
    assert extras_groups["DCMI v5"] == DEFAULT_DCMI_RUN_GROUP
    # Scratch DARI remains in core prefix; historical DARI v6 is separate.
    assert extras_groups["DARI (AR OAT)"] == DEFAULT_DARI_SCRATCH_GROUP
    assert extras_groups["DARI v6"] != extras_groups["DCMI v5"]
    assert extras_groups["DARI v6"] != extras_groups["DARI (AR OAT)"]
    assert "RQL→Q-Flow online" not in extras_groups
    assert "Pure Q-Flow" not in extras_groups

    # --all-methods keeps piecewise Quantized / QDFL / Q-Flow placeholders.
    all_m = build_series("clf-group", all_methods=True)
    all_groups = {s[0]: s[1] for s in all_m}
    assert all_groups["Quantized DFL-RQL v9 (400k restore)"] == "__quantized_v9__"
    assert all_groups["AR QDFL student (phase1+2)"] == "__ar_qdfl_distill__"
    assert all_groups["DD QDFL student (phase1+2)"] == "__dd_qdfl_distill__"
    assert "RQL→Q-Flow online (phase1+2)" not in all_groups
    assert (
        all_groups["Q-Flow RQL warmstart (phase1+2)"]
        == QFLOW_RQL_WARMSTART_V2_PLACEHOLDER
    )
    assert all_groups["Pure Q-Flow (phase1+2)"] == PURE_QFLOW_PLACEHOLDER

    custom = build_series(
        "clf-group",
        all_methods=False,
        dari_run_group="my-dari-smoke",
        with_extras=True,
    )
    assert dict(zip([s[0] for s in custom], [s[1] for s in custom]))[
        "DARI v6"
    ] == "my-dari-smoke"


def test_stable_awr_shift_invariant_normalize_clip_ess():
    """Stable weights ignore global offset; mean-1, clip, ESS finite."""
    adv = jnp.array([-180.0, -160.0, -200.0, -170.0], dtype=jnp.float32)
    shifted = adv - 500.0
    w0 = Agent.stable_awr_example_weights(
        adv, temperature=5.0, max_weight=100.0, ramp_frac=1.0
    )
    w1 = Agent.stable_awr_example_weights(
        shifted, temperature=5.0, max_weight=100.0, ramp_frac=1.0
    )
    np.testing.assert_allclose(np.asarray(w0), np.asarray(w1), atol=1e-5)
    assert float(w0.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(w0.max()) <= 100.0 + 1e-5
    # Naive exp(A/T) underflows to ~0 for huge negative A.
    naive = jnp.exp(adv / 5.0)
    assert float(naive.max()) < 1e-10
    assert float(w0.max()) > 0.1
    ess = float(Agent.awr_ess(w0))
    assert ess > 1.0
    # Clip path: extreme relative advantage hits max_weight then renorms.
    extreme = jnp.array([0.0, 1000.0], dtype=jnp.float32)
    w_clip = Agent.stable_awr_example_weights(
        extreme, temperature=1.0, max_weight=10.0, ramp_frac=1.0
    )
    assert float(w_clip.mean()) == pytest.approx(1.0, abs=1e-5)
    assert float(w_clip.max()) <= 10.0 + 1e-4


def test_mc_mode_v_target_advantage_and_missing_field():
    agent = _make_tiny_agent(
        advantage_source="mc_return",
        use_mc_returns=True,
        mc_expectile=0.7,
        mc_actor_warmup_steps=0,
        ss_loss_coef=0.0,
    )
    batch = _synthetic_batch(agent, with_mc=True)
    rng = jax.random.PRNGKey(0)
    loss, info = agent.total_loss(batch, agent.network.params, rng=rng)
    assert np.isfinite(float(np.asarray(loss)))
    assert float(np.asarray(info["advantage_source"])) == pytest.approx(1.0)
    assert "mc_return_mean" in info
    assert "mc_v_error" in info
    assert float(np.asarray(info["mc_return_mean"])) < 0.0
    # A = G - V; with V≈0 at init, adv ≈ G (large negative).
    assert float(np.asarray(info["adv_mean"])) < -50.0
    # Stabilized AWR weights stay O(1), not underflow.
    assert float(np.asarray(info["awr_weight_mean"])) == pytest.approx(1.0, abs=1e-3)
    assert float(np.asarray(info["awr_ess"])) > 0.0

    batch_missing = _synthetic_batch(agent, with_mc=False)
    with pytest.raises(KeyError, match="mc_returns"):
        agent.total_loss(batch_missing, agent.network.params, rng=rng)


def test_iql_mode_unchanged_with_optional_mc_field():
    """IQL path ignores mc_returns and keeps advantage_source indicator 0."""
    agent = _make_tiny_agent(advantage_source="iql", use_mc_returns=False)
    batch = _synthetic_batch(agent, with_mc=True)
    rng = jax.random.PRNGKey(1)
    loss, info = agent.total_loss(batch, agent.network.params, rng=rng)
    assert np.isfinite(float(np.asarray(loss)))
    assert float(np.asarray(info["advantage_source"])) == pytest.approx(0.0)
    assert "mc_return_mean" not in info


def test_mc_update_stock_with_mc_batch():
    agent = _make_tiny_agent(
        advantage_source="mc_return",
        use_mc_returns=True,
        ss_loss_coef=0.0,
        use_register_weights=False,
    )
    batch = _synthetic_batch(agent, with_mc=True)
    new_agent, info = agent.update(batch)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert float(np.asarray(info["advantage_source"])) == pytest.approx(1.0)
    assert "mc_v_error" in info
    _, info2 = new_agent.update(batch)
    assert np.isfinite(float(np.asarray(info2["total_loss"])))


def test_mc_checkpoint_param_tree_matches_iql():
    """MC advantage_source is loss-only: param tree identical to IQL agent."""
    a_iql = _make_tiny_agent(advantage_source="iql")
    a_mc = _make_tiny_agent(advantage_source="mc_return", use_mc_returns=True)
    flat0, _ = jax.tree_util.tree_flatten_with_path(a_iql.network.params)
    flat1, _ = jax.tree_util.tree_flatten_with_path(a_mc.network.params)
    paths0 = [tuple(map(str, p)) for p, _ in flat0]
    paths1 = [tuple(map(str, p)) for p, _ in flat1]
    assert paths0 == paths1


def test_trajectory_success_weight_formula_ramp_and_share():
    """w=(1-r)*1 + r*(1+(W-1)*f); share ≈ 62.5% at W=20, 7.7% success."""
    flags = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0], dtype=jnp.float32)  # 1/13 ≈ 7.7%
    # At ramp=0: uniform 1.
    w0 = Agent.trajectory_success_example_weights(flags, 20.0, ramp_frac=0.0)
    np.testing.assert_allclose(np.asarray(w0), np.ones(13), atol=1e-6)
    # At ramp=1: success=20, fail=1.
    w1 = Agent.trajectory_success_example_weights(flags, 20.0, ramp_frac=1.0)
    assert float(w1[0]) == pytest.approx(20.0)
    assert float(w1[1]) == pytest.approx(1.0)
    share = float((w1 * flags).sum() / w1.sum())
    # 20 / (20 + 12*1) = 20/32 = 0.625
    assert share == pytest.approx(0.625, abs=1e-5)
    # Mid-ramp blend.
    w_mid = Agent.trajectory_success_example_weights(flags, 20.0, ramp_frac=0.5)
    assert float(w_mid[0]) == pytest.approx(10.5)  # 0.5*1 + 0.5*20
    assert float(w_mid[1]) == pytest.approx(1.0)


def test_trajectory_success_mode_logs_and_missing_field():
    agent = _make_tiny_agent(
        advantage_source="trajectory_success",
        use_trajectory_success=True,
        success_weight=20.0,
        success_actor_warmup_steps=0,
        ss_loss_coef=0.0,
        network_step=10**6,  # ramp fully on (warmup=0, ramp_steps=1)
    )
    batch = _synthetic_batch(agent, with_traj_success=True)
    rng = jax.random.PRNGKey(0)
    loss, info = agent.total_loss(batch, agent.network.params, rng=rng)
    assert np.isfinite(float(np.asarray(loss)))
    assert float(np.asarray(info["advantage_source"])) == pytest.approx(2.0)
    assert "traj_success_frac" in info
    assert "traj_success_weight_mean" in info
    assert "traj_fail_weight_mean" in info
    assert "traj_weighted_success_share" in info
    assert float(np.asarray(info["traj_success_weight_mean"])) == pytest.approx(
        20.0, abs=1e-3
    )
    assert float(np.asarray(info["traj_fail_weight_mean"])) == pytest.approx(
        1.0, abs=1e-3
    )
    # No MC logs; V still trains (v_loss finite, no mc_return_*).
    assert "mc_return_mean" not in info
    assert np.isfinite(float(np.asarray(info["v_loss"])))

    batch_missing = _synthetic_batch(agent, with_traj_success=False)
    with pytest.raises(KeyError, match="trajectory_success"):
        agent.total_loss(batch_missing, agent.network.params, rng=rng)


def test_trajectory_success_update_and_param_tree():
    agent = _make_tiny_agent(
        advantage_source="trajectory_success",
        use_trajectory_success=True,
        ss_loss_coef=0.0,
        use_register_weights=False,
        network_step=10**6,
    )
    batch = _synthetic_batch(agent, with_traj_success=True)
    new_agent, info = agent.update(batch)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert float(np.asarray(info["advantage_source"])) == pytest.approx(2.0)
    _, info2 = new_agent.update(batch)
    assert np.isfinite(float(np.asarray(info2["total_loss"])))

    a_iql = _make_tiny_agent(advantage_source="iql")
    a_ts = _make_tiny_agent(
        advantage_source="trajectory_success", use_trajectory_success=True
    )
    flat0, _ = jax.tree_util.tree_flatten_with_path(a_iql.network.params)
    flat1, _ = jax.tree_util.tree_flatten_with_path(a_ts.network.params)
    paths0 = [tuple(map(str, p)) for p, _ in flat0]
    paths1 = [tuple(map(str, p)) for p, _ in flat1]
    assert paths0 == paths1


def test_iql_and_mc_modes_unchanged_with_traj_field():
    """IQL/MC ignore trajectory_success; indicators stay 0/1."""
    agent_iql = _make_tiny_agent(advantage_source="iql")
    batch = _synthetic_batch(agent_iql, with_mc=True, with_traj_success=True)
    rng = jax.random.PRNGKey(2)
    _, info_iql = agent_iql.total_loss(batch, agent_iql.network.params, rng=rng)
    assert float(np.asarray(info_iql["advantage_source"])) == pytest.approx(0.0)
    assert "traj_success_frac" not in info_iql
    assert "mc_return_mean" not in info_iql

    agent_mc = _make_tiny_agent(
        advantage_source="mc_return", use_mc_returns=True, ss_loss_coef=0.0
    )
    _, info_mc = agent_mc.total_loss(batch, agent_mc.network.params, rng=rng)
    assert float(np.asarray(info_mc["advantage_source"])) == pytest.approx(1.0)
    assert "mc_return_mean" in info_mc
    assert "traj_success_frac" not in info_mc


# ---- v8 direct action-level critic gradients ----


def test_fsq_id_code_mapping_and_codebook_shape():
    codebook = build_codebook(FSQ_LEVELS)
    assert codebook.shape == (CODEBOOK_SIZE, 4)
    # Exact mapping for known ids (FSQ levels 8,5,5,5).
    np.testing.assert_allclose(
        np.asarray(indices_to_codes(jnp.array(0), FSQ_LEVELS)),
        [-1.0, -1.0, -1.0, -1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(indices_to_codes(jnp.array(1), FSQ_LEVELS)),
        [-0.75, -1.0, -1.0, -1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(indices_to_codes(jnp.array(8), FSQ_LEVELS)),
        [-1.0, -0.5, -1.0, -1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(indices_to_codes(jnp.array(999), FSQ_LEVELS)),
        [0.75, 1.0, 1.0, 1.0],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(codebook[0]),
        np.asarray(indices_to_codes(jnp.array(0), FSQ_LEVELS)),
    )
    np.testing.assert_allclose(
        np.asarray(codebook[999]),
        np.asarray(indices_to_codes(jnp.array(999), FSQ_LEVELS)),
    )


def test_st_forward_equals_hard_and_temperature_shift():
    codebook = build_codebook(FSQ_LEVELS)
    b, k, v = 2, 3, CODEBOOK_SIZE
    logits = jax.random.normal(jax.random.PRNGKey(0), (b, k, v))
    codes, probs = Agent.straight_through_fsq_codes(logits, codebook, temperature=1.0)
    hard = indices_to_codes(jnp.argmax(logits, axis=-1), FSQ_LEVELS)
    np.testing.assert_allclose(np.asarray(codes), np.asarray(hard), atol=1e-5)
    assert probs.shape == (b, k, v)
    # Lower temperature sharpens softmax (lower entropy).
    _, p_hot = Agent.straight_through_fsq_codes(logits, codebook, temperature=0.1)
    _, p_cold = Agent.straight_through_fsq_codes(logits, codebook, temperature=10.0)
    ent = lambda p: float((-p * jnp.log(jnp.maximum(p, 1e-8))).sum(axis=-1).mean())
    assert ent(p_hot) < ent(p_cold)


def test_st_nonzero_gradient_to_logits():
    codebook = build_codebook(FSQ_LEVELS)
    logits = jax.random.normal(jax.random.PRNGKey(1), (2, 4, CODEBOOK_SIZE))

    def loss_fn(lg):
        codes, _ = Agent.straight_through_fsq_codes(lg, codebook, temperature=1.0)
        return jnp.square(codes).sum()

    g = jax.grad(loss_fn)(logits)
    assert float(jnp.abs(g).sum()) > 1e-8
    assert np.isfinite(float(jnp.abs(g).sum()))


def test_q_actor_schedule_and_nan_skip():
    warmup, ramp, coef = 50_000, 50_000, 1.0
    assert float(Agent.q_actor_weight_from_step(0, warmup, ramp, coef)) == pytest.approx(
        0.0
    )
    assert float(
        Agent.q_actor_weight_from_step(warmup, warmup, ramp, coef)
    ) == pytest.approx(0.0)
    mid = warmup + ramp // 2
    assert float(
        Agent.q_actor_weight_from_step(mid, warmup, ramp, coef)
    ) == pytest.approx(0.5, abs=1e-5)
    assert float(
        Agent.q_actor_weight_from_step(warmup + ramp, warmup, ramp, coef)
    ) == pytest.approx(1.0)
    # coef=0 always zero weight (skip branch).
    assert float(
        Agent.q_actor_weight_from_step(10**9, warmup, ramp, 0.0)
    ) == pytest.approx(0.0)

    # Warmup agent: weight 0, finite loss, q_actor_loss logged as 0.
    agent = _make_tiny_agent(
        q_actor_coef=1.0,
        q_actor_warmup_steps=10**9,
        q_actor_ramp_steps=1,
        ss_loss_coef=0.0,
    )
    batch = _synthetic_batch(agent)
    loss, info = agent.total_loss(batch, agent.network.params, rng=jax.random.PRNGKey(0))
    assert np.isfinite(float(np.asarray(loss)))
    assert float(np.asarray(info["q_actor_weight"])) == pytest.approx(0.0)
    assert float(np.asarray(info["q_actor_loss"])) == pytest.approx(0.0)


def test_q_actor_coef0_legacy_invariant_and_param_tree():
    """q_actor_coef=0 must match CE-only loss and leave param tree unchanged."""
    agent0 = _make_tiny_agent(q_actor_coef=0.0, ss_loss_coef=0.0)
    agent1 = _make_tiny_agent(
        q_actor_coef=1.0,
        q_actor_warmup_steps=0,
        q_actor_ramp_steps=1,
        network_step=10,
        ss_loss_coef=0.0,
    )
    # Param trees identical (Q-actor is loss-only; codebook is nonpytree).
    flat0, _ = jax.tree_util.tree_flatten_with_path(agent0.network.params)
    flat1, _ = jax.tree_util.tree_flatten_with_path(agent1.network.params)
    paths0 = [tuple(map(str, p)) for p, _ in flat0]
    paths1 = [tuple(map(str, p)) for p, _ in flat1]
    assert paths0 == paths1

    batch = _synthetic_batch(agent0)
    rng = jax.random.PRNGKey(3)
    loss0, info0 = agent0.total_loss(batch, agent0.network.params, rng=rng)
    # Same params on agent1 but coef active — different total when weight>0.
    agent1 = agent1.replace(network=agent0.network.replace(step=10))
    loss1, info1 = agent1.total_loss(batch, agent1.network.params, rng=rng)
    assert float(np.asarray(info0["q_actor_weight"])) == pytest.approx(0.0)
    assert float(np.asarray(info1["q_actor_weight"])) == pytest.approx(1.0)
    # Legacy CE path preserved under coef=0.
    assert float(np.asarray(info0["actor_loss"])) == pytest.approx(
        float(np.asarray(info0["tf_actor_loss"])), abs=1e-5
    )
    assert np.isfinite(float(np.asarray(loss0)))
    assert np.isfinite(float(np.asarray(loss1)))
    # Active Q term should change total vs CE-only (same shared params).
    assert float(np.asarray(loss1)) != pytest.approx(float(np.asarray(loss0)), abs=1e-6)


def test_q_actor_target_q_frozen_actor_grad_isolated():
    """Isolated q_actor term: nonzero actor grad, zero q/v/target_q grads."""
    agent = _make_tiny_agent(
        q_actor_coef=1.0,
        q_actor_warmup_steps=0,
        q_actor_ramp_steps=1,
        network_step=10,
        ss_loss_coef=0.0,
        use_register_weights=False,
        rho=0.25,
    )
    batch = _synthetic_batch(agent)
    rng = jax.random.PRNGKey(5)

    def q_actor_only(params):
        _, info = agent.total_loss(batch, params, rng=rng)
        # Isolate the Q-actor objective (effective weight already applied in total,
        # but we differentiate the raw term so grads are well-defined).
        return info["q_actor_loss"]

    grads = jax.grad(q_actor_only)(agent.network.params)

    def _leaf_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        if not leaves:
            return 0.0
        return float(jnp.sqrt(sum(jnp.square(x).sum() for x in leaves)))

    assert _leaf_norm(grads["modules_actor"]) > 1e-8
    assert _leaf_norm(grads["modules_q"]) == pytest.approx(0.0, abs=1e-8)
    assert _leaf_norm(grads["modules_v"]) == pytest.approx(0.0, abs=1e-8)
    assert _leaf_norm(grads["modules_target_q"]) == pytest.approx(0.0, abs=1e-8)
    assert _leaf_norm(grads["modules_target_v"]) == pytest.approx(0.0, abs=1e-8)
    assert _leaf_norm(grads["modules_target_actor"]) == pytest.approx(0.0, abs=1e-8)


def test_st_decode_finite_gradient_stub():
    """Stub decoder depends on codes → finite ST→decode gradient."""
    agent = _make_tiny_agent(
        q_actor_coef=1.0,
        q_actor_warmup_steps=0,
        q_actor_ramp_steps=1,
        network_step=5,
        ss_loss_coef=0.0,
    )
    batch = _synthetic_batch(agent)
    rng = jax.random.PRNGKey(9)

    def loss_fn(params):
        loss, _ = agent.total_loss(batch, params, rng=rng)
        return loss

    grads = jax.grad(loss_fn)(agent.network.params)
    actor_norm = float(
        jnp.sqrt(
            sum(
                jnp.square(x).sum()
                for x in jax.tree_util.tree_leaves(grads["modules_actor"])
            )
        )
    )
    assert np.isfinite(actor_norm)
    assert actor_norm > 0.0


def test_self_conditioned_input_construction_and_extra_logits():
    tokens = jnp.array([[10, 20, 30, 40], [1, 2, 3, 4]], dtype=jnp.int32)
    inp = Agent.make_self_conditioned_inputs(tokens, bos_id=BOS_ID)
    np.testing.assert_array_equal(np.asarray(inp[:, 0]), BOS_ID)
    np.testing.assert_array_equal(np.asarray(inp[:, 1:]), np.asarray(tokens[:, :-1]))

    agent_tf = _make_tiny_agent(
        q_actor_coef=1.0,
        q_actor_warmup_steps=0,
        q_actor_ramp_steps=1,
        network_step=10,
        q_actor_prefix_mode="teacher_forced",
        ss_loss_coef=0.0,
    )
    agent_sc = _make_tiny_agent(
        q_actor_coef=1.0,
        q_actor_warmup_steps=0,
        q_actor_ramp_steps=1,
        network_step=10,
        q_actor_prefix_mode="self_conditioned",
        ss_loss_coef=0.0,
    )
    agent_sc = agent_sc.replace(network=agent_tf.network)
    batch = _synthetic_batch(agent_tf)
    rng = jax.random.PRNGKey(11)
    _, info_tf = agent_tf.total_loss(batch, agent_tf.network.params, rng=rng)
    _, info_sc = agent_sc.total_loss(batch, agent_sc.network.params, rng=rng)
    # Both log q-branch metrics; self-conditioned uses predicted prefix.
    assert "q_token_acc" in info_tf and "q_token_acc" in info_sc
    assert "q_prefix_exact_1" in info_sc
    # Losses/metrics can differ when pass-A argmax ≠ clean tokens.
    assert np.isfinite(float(np.asarray(info_tf["q_actor_loss"])))
    assert np.isfinite(float(np.asarray(info_sc["q_actor_loss"])))


def test_q_actor_real_data_compatible_shapes():
    """ST codes (B,K,4) decode to (B,h*Da); Q input matches critic."""
    h, act_dim, k, b = 1, 3, 4, 4
    agent = _make_tiny_agent(
        batch_size=b,
        h=h,
        act_dim=act_dim,
        num_registers=k,
        q_actor_coef=1.0,
        q_actor_warmup_steps=0,
        q_actor_ramp_steps=1,
        network_step=10,
        ss_loss_coef=0.0,
    )
    logits = jax.random.normal(jax.random.PRNGKey(0), (b, k, CODEBOOK_SIZE))
    codes, _ = Agent.straight_through_fsq_codes(
        logits, agent.codebook, temperature=1.0
    )
    assert codes.shape == (b, k, 4)
    decoded = agent._decode_codes(codes)
    assert decoded.shape == (b, h * act_dim)
    clipped = Agent.differentiable_action_clip(decoded)
    assert clipped.shape == decoded.shape
    batch = _synthetic_batch(agent)
    loss, info = agent.total_loss(batch, agent.network.params, rng=jax.random.PRNGKey(0))
    assert np.isfinite(float(np.asarray(loss)))
    assert float(np.asarray(info["q_actor_weight"])) == pytest.approx(1.0)
    for key in (
        "q_actor_loss",
        "q_policy_mean",
        "q_policy_min",
        "q_policy_max",
        "q_policy_rmse",
        "q_action_sat_frac",
        "st_entropy_mean",
        "q_token_acc",
        "q_seq_exact",
    ):
        assert key in info
        assert np.isfinite(float(np.asarray(info[key])))


def test_q_actor_update_stock():
    agent = _make_tiny_agent(
        q_actor_coef=1.0,
        q_actor_warmup_steps=0,
        q_actor_ramp_steps=1,
        network_step=10,
        ss_loss_coef=0.0,
        use_register_weights=False,
    )
    batch = _synthetic_batch(agent)
    new_agent, info = agent.update(batch)
    assert np.isfinite(float(np.asarray(info["total_loss"])))
    assert float(np.asarray(info["q_actor_weight"])) == pytest.approx(1.0)
    _, info2 = new_agent.update(batch)
    assert np.isfinite(float(np.asarray(info2["total_loss"])))
