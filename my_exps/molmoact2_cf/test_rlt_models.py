"""Unit tests for RLT-Consensus CF modules (CPU)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_replay import ChunkReplay, ChunkTransition, TokenReplay
from rlt_models import (
    ACTION_DIM,
    CHUNK_SIZE,
    FEATURE_DIM,
    STATE_DIM,
    Z_DIM,
    CFGRL_O_POS,
    CFGRL_O_UNCOND,
    CFGradientGuide,
    ChunkGaussianActor,
    MolmoAct2RLTCF,
    RLTokenAutoencoder,
    bootstrap_scale,
    chunk_return,
    common_scale_normalize,
    lower_tail_mean,
    normalized_grad_target,
)
from train_rlt import (
    actor_step,
    build_rlt_optimizers,
    cfgrl_condition_and_target,
    critic_td_step,
    guide_step,
    predicted_lcb_advantage,
    stochastic_target_critic_gradient,
    token_step,
)


def test_rl_token_recon_and_mask():
    ae = RLTokenAutoencoder(token_dim=FEATURE_DIM, z_dim=Z_DIM, d_model=128, n_layers=1, n_heads=4)
    b, s = 2, 17
    tokens = torch.randn(b, s, FEATURE_DIM)
    mask = torch.ones(b, s)
    mask[:, -3:] = 0
    z = ae.encode(tokens, mask)
    assert z.shape == (b, Z_DIM)
    loss, info = ae.reconstruction_loss(tokens, mask)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert info["z_rl"].shape == (b, Z_DIM)


def test_stop_grad_actor_no_token_grad():
    model = MolmoAct2RLTCF(token_layers=1, token_d_model=128, use_cf_guide=True)
    tokens = torch.randn(4, 12, FEATURE_DIM, requires_grad=False)
    proprio = torch.randn(4, 8)
    ref = torch.randn(4, CHUNK_SIZE, ACTION_DIM)
    state = model.encode_state(tokens, proprio, detach_token=True)
    act, info = model.actor_chunk(state, model.normalize_action(ref), deterministic=True)
    loss = act.mean()
    loss.backward()
    for p in model.token_ae.parameters():
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0
    assert info["actor_delta"].shape == ref.shape


def test_chunk_replay_partial_terminal():
    buf = ChunkReplay(chunk_size=CHUNK_SIZE, max_transitions=100)
    C, A = CHUNK_SIZE, ACTION_DIM
    zs = [np.random.randn(Z_DIM).astype(np.float32) for _ in range(2)]
    props = [np.random.randn(8).astype(np.float32) for _ in range(2)]
    refs = [np.random.randn(C, A).astype(np.float32) for _ in range(2)]
    exs = [r.copy() for r in refs]
    rewards = [np.zeros(C, dtype=np.float32) for _ in range(2)]
    rewards[-1][-1] = 1.0
    masks = [np.ones(C, dtype=np.float32), np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32)]
    n = buf.add_episode_chunks(zs, props, refs, exs, rewards, masks, success=True, gamma=0.99)
    assert n == 2
    assert buf.rows[-1].terminal is True
    batch = buf.sample(2)
    assert batch["executed_actions"].shape == (2, C, A)
    y = chunk_return(batch["rewards"], 0.99, batch["action_mask"])
    boot = bootstrap_scale(0.99, C, batch["action_mask"], batch["terminal"])
    assert y.shape == (2,)
    assert torch.all(boot[batch["terminal"] > 0.5] == 0)


def test_td_actor_guide_steps_finite():
    torch.manual_seed(0)
    model = MolmoAct2RLTCF(token_layers=1, token_d_model=128, use_cf_guide=True, tune_token_online=True)
    model.set_norm_stats(torch.zeros(8), torch.ones(8), torch.zeros(8), torch.ones(8))
    opts = build_rlt_optimizers(model)
    buf = ChunkReplay()
    for i in range(16):
        succ = i % 2 == 0
        C = CHUNK_SIZE
        rewards = np.zeros(C, dtype=np.float32)
        if succ:
            rewards[-1] = 1.0
        buf.add(
            ChunkTransition(
                z=np.random.randn(Z_DIM).astype(np.float32),
                proprio=np.random.randn(8).astype(np.float32),
                reference_actions=np.random.randn(C, 8).astype(np.float32) * 0.01,
                executed_actions=np.random.randn(C, 8).astype(np.float32) * 0.01,
                rewards=rewards,
                action_mask=np.ones(C, dtype=np.float32),
                next_z=np.random.randn(Z_DIM).astype(np.float32),
                next_proprio=np.random.randn(8).astype(np.float32),
                next_reference_actions=np.random.randn(C, 8).astype(np.float32) * 0.01,
                terminal=True,
                mc_return=float(succ),
                success=float(succ),
                episode_id=i,
                start_step=0,
            )
        )
    batch = buf.sample(8)
    cinfo = critic_td_step(model, opts["critic"], batch)
    ainfo = actor_step(model, opts["actor"], opts["alpha"], batch)
    ginfo = guide_step(model, opts["guide"], batch)
    assert all(np.isfinite(v) for v in cinfo.values())
    assert all(np.isfinite(v) for v in ainfo.values())
    assert all(np.isfinite(v) for v in ginfo.values())
    adv = predicted_lcb_advantage(model, batch)
    assert np.isfinite(adv)


def test_residual_guide_is_additive():
    """Deploy must be ã+Δπ+Δg, not replace the actor with ref+Δg (v8 bug)."""
    torch.manual_seed(0)
    model = MolmoAct2RLTCF(token_layers=1, token_d_model=128, use_cf_guide=True)
    model.set_norm_stats(torch.zeros(8), torch.ones(8), torch.zeros(8), torch.ones(8))
    # Break zero-init so the actor residual is nonzero.
    with torch.no_grad():
        for p in model.actor.parameters():
            p.add_(0.05 * torch.randn_like(p))
    state = torch.randn(4, STATE_DIM)
    ref = torch.zeros(4, CHUNK_SIZE, ACTION_DIM)
    act_a, info_a = model.actor_chunk(state, ref, deterministic=True, apply_guide=False)
    act_g, info_g = model.actor_chunk(state, ref, deterministic=True, apply_guide=True)
    assert "guide_delta" in info_g
    expected = info_a["actor_mean"] + info_g["guide_delta"]
    assert torch.allclose(act_g, expected, atol=1e-5)
    assert float(info_a["actor_delta"].abs().mean()) > 1e-4
    # Must not collapse to reference + guide only.
    ref_plus_g = ref + info_g["guide_delta"]
    assert not torch.allclose(act_g, ref_plus_g, atol=1e-5)


def test_normalized_grad_and_guide():
    grads = [torch.randn(3, CHUNK_SIZE, ACTION_DIM) for _ in range(2)]
    tgt = normalized_grad_target(grads)
    assert tgt.shape == (3, CHUNK_SIZE, ACTION_DIM)
    g = CFGradientGuide()
    s = torch.randn(3, STATE_DIM)
    ref = torch.zeros(3, CHUNK_SIZE, ACTION_DIM)
    guided, delta = g.guide(s, ref, actor_delta=torch.ones_like(ref) * 0.01)
    assert guided.shape == ref.shape
    assert float(delta.abs().max()) <= 0.05 + 1e-5


def test_common_scale_preserves_magnitude_and_opposed_members():
    grad = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    target = common_scale_normalize(grad, floor_coef=0.1, eps=0.0)
    mean_norm = torch.tensor(5.5)
    expected = torch.tensor(
        [1.0 / (1.0 + 0.1 * mean_norm), 10.0 / (10.0 + 0.1 * mean_norm)]
    )
    assert torch.allclose(target.norm(dim=-1), expected)
    assert target[1].norm() > target[0].norm()

    class OpposedTarget:
        def q_chunk(
            self,
            _state,
            actions,
            *,
            target=False,
            t=None,
        ):
            del target, t
            score = actions.flatten(1).sum(dim=-1)
            return torch.stack([score, -score], dim=0)

    actions = torch.zeros(2, 1, 2)
    selected = stochastic_target_critic_gradient(
        OpposedTarget(),
        torch.zeros(2, 1),
        actions,
        member=torch.tensor([0, 1]),
    )
    assert selected[0].norm() > 0.0
    assert selected[1].norm() > 0.0
    assert torch.allclose(selected[0], -selected[1])


def test_token_replay_and_checkpoint():
    tok = TokenReplay(max_seq=32)
    for _ in range(5):
        s = np.random.randint(8, 20)
        tok.add(np.random.randn(s, FEATURE_DIM).astype(np.float16), np.ones(s, dtype=np.uint8))
    batch = tok.sample(2)
    assert batch["tokens"].shape[0] == 2
    model = MolmoAct2RLTCF(token_layers=1, token_d_model=128)
    opts = build_rlt_optimizers(model)
    info = token_step(model, opts["token"], batch)
    assert np.isfinite(info["token_recon_loss"])
    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "rlt.pt")
        model.save(path, meta={"test": True})
        m2 = MolmoAct2RLTCF.load(path)
        assert m2.schema_version == model.schema_version
        assert m2.loaded_meta == {"test": True}
        tpath = str(Path(td) / "tok.npz")
        tok.save_npz(tpath)
        tok2 = TokenReplay.load_npz(tpath)
        assert len(tok2) == len(tok)


def test_lower_tail_mean_is_robust_to_one_dead_head():
    values = torch.tensor(
        [
            [0.0, 0.0],
            [0.4, 0.5],
            [0.5, 0.6],
            [0.6, 0.7],
            [0.7, 0.8],
            [0.8, 0.9],
            [0.9, 0.9],
            [0.9, 0.9],
            [0.9, 0.9],
            [0.9, 0.9],
        ]
    )
    aggregate = lower_tail_mean(values, fraction=0.25, min_heads=2, dim=0)
    assert torch.allclose(aggregate, torch.tensor([0.3, 11.0 / 30.0]))
    assert torch.all(aggregate > values.min(dim=0).values)


def test_q_tail_config_roundtrip():
    model = MolmoAct2RLTCF(
        token_layers=1,
        token_d_model=64,
        n_critics=4,
        q_tail_fraction=0.5,
        q_tail_min_heads=3,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tail.pt"
        model.save(str(path))
        loaded = MolmoAct2RLTCF.load(str(path))
    assert loaded.q_tail_fraction == 0.5
    assert loaded.q_tail_min_heads == 3


def test_explicit_critic_head_recovery_resets_target_copy():
    model = MolmoAct2RLTCF(
        token_layers=1,
        token_d_model=64,
        n_critics=3,
    )
    before = [
        parameter.detach().clone()
        for parameter in model.critic.critics[1].parameters()
    ]
    reset_parameters = model.reinitialize_critic_heads([1], seed=917)
    after = list(model.critic.critics[1].parameters())
    target = list(model.target_critic.critics[1].parameters())
    assert reset_parameters
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, after)
    )
    assert all(
        torch.equal(online.detach(), target_parameter.detach())
        for online, target_parameter in zip(after, target)
    )


def test_reference_dropout_changes_input_path():
    actor = ChunkGaussianActor(residual=True)
    s = torch.randn(4, STATE_DIM)
    ref = torch.randn(4, CHUNK_SIZE, ACTION_DIM)
    present = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mean_on, _ = actor(s, ref, present)
    mean_off, _ = actor(s, torch.zeros_like(ref), present)
    # When present=0, network sees zeros; residual still adds to original ref in forward
    # (residual uses detached ref). With present=0, ref_in is zero so raw differs.
    mean_drop, _ = actor(s, ref, present)
    assert mean_drop.shape == ref.shape
    assert not torch.allclose(mean_on, mean_off)


def test_cfgrl_labels_map_failures_to_uncond():
    torch.manual_seed(0)
    a_data = torch.ones(4, CHUNK_SIZE, ACTION_DIM)
    a_ref = torch.zeros(4, CHUNK_SIZE, ACTION_DIM)
    success = torch.tensor([1.0, 0.0, 1.0, 0.0])
    o, a_star, info = cfgrl_condition_and_target(
        a_data,
        a_ref,
        success=success,
        cond_dropout=0.0,
        use_advantage_labels=False,
    )
    assert info["cfgrl_pos_frac"] == 0.5
    assert float((o == CFGRL_O_POS).float().mean()) == 0.5
    assert float((o == CFGRL_O_UNCOND).float().mean()) == 0.5
    assert torch.allclose(a_star[0], a_data[0])
    assert torch.allclose(a_star[1], a_ref[1])
    o_drop, a_star_drop, _ = cfgrl_condition_and_target(
        a_data,
        a_ref,
        success=torch.ones(4),
        cond_dropout=1.0,
        use_advantage_labels=False,
    )
    assert torch.equal(o_drop, torch.full((4,), CFGRL_O_UNCOND, dtype=torch.long))
    assert torch.allclose(a_star_drop, a_ref)


if __name__ == "__main__":
    tests = [
        test_rl_token_recon_and_mask,
        test_stop_grad_actor_no_token_grad,
        test_chunk_replay_partial_terminal,
        test_td_actor_guide_steps_finite,
        test_normalized_grad_and_guide,
        test_common_scale_preserves_magnitude_and_opposed_members,
        test_token_replay_and_checkpoint,
        test_lower_tail_mean_is_robust_to_one_dead_head,
        test_q_tail_config_roundtrip,
        test_explicit_critic_head_recovery_resets_target_copy,
        test_reference_dropout_changes_input_path,
        test_cfgrl_labels_map_failures_to_uncond,
    ]
    for fn in tests:
        fn()
        print("OK", fn.__name__)
    print(f"passed {len(tests)} tests")
