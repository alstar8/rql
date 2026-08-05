"""Unit tests for MolmoAct2 CF modules (no GPU required)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import FEATURE_DIM, EndpointG, EnsembleCQL, MolmoAct2CF, STATE_DIM, Z_DIM
from train_full import OnlineReplay
from train_offline import build_optimizers, critic_is_healthy, critic_step, refiner_step


def test_endpoint_g_near_zero_init():
    g = EndpointG(STATE_DIM, 8, max_delta=0.05)
    s = torch.randn(4, STATE_DIM)
    a = torch.randn(4, 8)
    delta = g(s, a)
    assert delta.shape == a.shape
    assert float(delta.abs().max()) < 1e-5


def test_refine_raw_with_features():
    m = MolmoAct2CF(use_vla_features=True)
    m.set_norm_stats(
        torch.zeros(8),
        torch.ones(8),
        torch.zeros(8),
        torch.ones(8),
    )
    s = torch.randn(2, 8)
    h = torch.randn(2, FEATURE_DIM)
    a = torch.randn(2, 8)
    refined, delta = m.refine_raw(s, a, features=h, delta_clip=0.05)
    assert refined.shape == a.shape
    assert torch.allclose(refined, a + delta)
    x = m.encode_state(h, s)
    assert x.shape == (2, STATE_DIM)


def test_cql_penalty_shape():
    critic = EnsembleCQL(STATE_DIM, 8, n_critics=2)
    s = torch.randn(16, STATE_DIM)
    a = torch.randn(16, 8)
    loss, info = critic.cql_penalty(s, a, n_actions=4, coef=1.0, action_radius=0.05, far_scale=1.0)
    assert loss.ndim == 0
    assert float(loss) >= 0.0
    assert "cql_loss" in info
    q = critic(s, a)
    assert bool(torch.all((q >= 0.0) & (q <= 1.0)))


def test_cql_far_samples_outside_residual_ball():
    torch.manual_seed(0)
    critic = EnsembleCQL(STATE_DIM, 8, n_critics=2, bounded=True)
    s = torch.zeros(8, STATE_DIM)
    a = torch.zeros(8, 8)
    # Monkeypatch q_mean to capture sampled actions.
    captured: list[torch.Tensor] = []
    orig = critic.q_mean

    def _capture(state, actions):
        if actions.shape[0] == 8 * 4:  # far batch flattened
            captured.append(actions.detach().clone())
        return orig(state, actions)

    critic.q_mean = _capture  # type: ignore[method-assign]
    critic.cql_penalty(s, a, n_actions=4, action_radius=0.05, far_scale=1.0)
    assert captured, "expected far-action forward"
    deltas = captured[0].reshape(4, 8, 8) - a.unsqueeze(0)
    norms = deltas.norm(dim=-1)
    assert float(norms.min()) >= 0.05


def test_projector_dim():
    m = MolmoAct2CF(use_vla_features=True)
    assert m.projector is not None
    assert m.z_dim == Z_DIM
    h = torch.randn(3, FEATURE_DIM)
    z = m.projector(h)
    assert z.shape == (3, Z_DIM)


def test_optimizers_have_disjoint_projector_ownership():
    model = MolmoAct2CF(use_vla_features=True)
    opt_q, opt_g, _ = build_optimizers(model, lr_q=1e-3, lr_g=1e-3, lr_alpha=1e-4)
    q_ids = {id(p) for group in opt_q.param_groups for p in group["params"]}
    g_ids = {id(p) for group in opt_g.param_groups for p in group["params"]}
    projector_ids = {id(p) for p in model.projector.parameters()}
    assert projector_ids <= q_ids
    assert projector_ids.isdisjoint(g_ids)
    assert q_ids.isdisjoint(g_ids)


def test_online_replay_separates_deployed_and_base_actions():
    model = MolmoAct2CF(use_vla_features=False)
    replay = OnlineReplay(model, seed=0)
    proprio = [np.zeros(8, dtype=np.float32) for _ in range(2)]
    base = [np.zeros(8, dtype=np.float32) for _ in range(2)]
    deployed = [np.full(8, 0.25, dtype=np.float32) for _ in range(2)]
    features = [np.zeros(FEATURE_DIM, dtype=np.float32) for _ in range(2)]
    replay.add_episode(proprio, base, deployed, success=True, gamma=0.99, features_raw=features)
    batch = replay.sample_batch(8)
    assert torch.allclose(batch["base_actions"], torch.zeros_like(batch["base_actions"]))
    assert torch.allclose(batch["actions"], torch.full_like(batch["actions"], 0.25))


def test_replay_save_fits_matched_norm_stats(tmp_path: Path | None = None):
    model = MolmoAct2CF(use_vla_features=True)
    replay = OnlineReplay(model, seed=0)
    proprio = [
        np.zeros(8, dtype=np.float32),
        np.ones(8, dtype=np.float32),
    ]
    base = [
        np.zeros(8, dtype=np.float32),
        np.full(8, 2.0, dtype=np.float32),
    ]
    features = [
        np.zeros(FEATURE_DIM, dtype=np.float32),
        np.ones(FEATURE_DIM, dtype=np.float32),
    ]
    replay.add_episode(proprio, base, base, success=True, gamma=0.99, features_raw=features)
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "replay.npz"
            _assert_replay_save(replay, path)
    else:
        _assert_replay_save(replay, tmp_path / "replay.npz")


def _assert_replay_save(replay: OnlineReplay, path: Path) -> None:
    replay.save_npz(path, fit_norm_stats=True)
    with np.load(path) as arrays:
        assert arrays["features"].shape == (2, FEATURE_DIM)
        assert np.allclose(arrays["action_mean"], 1.0)
        assert np.allclose(arrays["actions"], np.asarray([[-1.0] * 8, [1.0] * 8]))
        assert np.allclose(arrays["base_actions"], arrays["actions"])


def test_refiner_step_does_not_update_critic_or_projector():
    torch.manual_seed(0)
    model = MolmoAct2CF(use_vla_features=True)
    opt_q, opt_g, opt_alpha = build_optimizers(
        model,
        lr_q=1e-3,
        lr_g=1e-3,
        lr_alpha=1e-4,
    )
    del opt_q
    batch = {
        "features": torch.randn(16, FEATURE_DIM),
        "proprio": torch.randn(16, 8),
        "actions": torch.randn(16, 8),
        "base_actions": torch.randn(16, 8),
    }
    critic_before = [p.detach().clone() for p in model.critic.parameters()]
    projector_before = [p.detach().clone() for p in model.projector.parameters()]
    refiner_step(
        model,
        batch,
        opt_g,
        opt_alpha,
        target_divergence=1e-3,
    )
    assert all(torch.equal(before, after) for before, after in zip(critic_before, model.critic.parameters()))
    assert all(
        torch.equal(before, after)
        for before, after in zip(projector_before, model.projector.parameters())
    )


def test_bounded_critic_training_stays_finite():
    torch.manual_seed(0)
    model = MolmoAct2CF(use_vla_features=False)
    opt_q, _, _ = build_optimizers(model, lr_q=1e-3, lr_g=1e-3, lr_alpha=1e-4)
    states = torch.randn(128, 8)
    actions = torch.randn(128, 8)
    returns = (actions[:, 0] > 0).float()
    batch = {
        "states": states,
        "actions": actions,
        "returns": returns,
    }
    first = critic_step(
        model,
        batch,
        opt_q,
        cql_coef=0.1,
        cql_n_actions=8,
    )
    info = first
    for _ in range(99):
        info = critic_step(
            model,
            batch,
            opt_q,
            cql_coef=0.1,
            cql_n_actions=8,
        )
    assert np.isfinite(list(info.values())).all()
    assert 0.0 <= info["q_min"] <= info["q_max"] <= 1.0
    assert info["cql_loss"] >= 0.0
    assert info["mc_loss"] < first["mc_loss"]


def test_checkpoint_roundtrip_preserves_bounded_critic(tmp_path: Path | None = None):
    model = MolmoAct2CF(use_vla_features=False, bounded_critic=True)
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "model.pt"
            model.save(str(path))
            loaded = MolmoAct2CF.load(str(path))
    else:
        path = tmp_path / "model.pt"
        model.save(str(path))
        loaded = MolmoAct2CF.load(str(path))
    assert loaded.bounded_critic


def test_critic_health_gate_rejects_exploding_metrics():
    assert critic_is_healthy(
        {
            "critic_loss": 0.1,
            "mc_loss": 0.05,
            "q_min": 0.0,
            "q_max": 0.9,
            "cql_loss": 0.01,
        },
        max_mc_loss=0.2,
    )
    assert not critic_is_healthy(
        {
            "critic_loss": 1e4,
            "mc_loss": 5.0,
            "q_min": -20.0,
            "q_max": 12.0,
            "cql_loss": -30000.0,
        },
        max_mc_loss=0.2,
    )


def test_refine_delta_clip_matches_max_delta():
    m = MolmoAct2CF(use_vla_features=False)
    # Force large raw logits so the tanh/max_delta path saturates.
    with torch.no_grad():
        for p in m.refiner.parameters():
            p.fill_(3.0)
    s = torch.zeros(4, 8)
    a = torch.zeros(4, 8)
    refined, delta = m.refiner.refine(s, a, delta_clip=m.refiner.max_delta)
    assert float(delta.abs().max()) <= m.refiner.max_delta + 1e-6
    assert torch.allclose(refined, a + delta)


if __name__ == "__main__":
    test_endpoint_g_near_zero_init()
    test_refine_raw_with_features()
    test_cql_penalty_shape()
    test_cql_far_samples_outside_residual_ball()
    test_projector_dim()
    test_optimizers_have_disjoint_projector_ownership()
    test_online_replay_separates_deployed_and_base_actions()
    test_replay_save_fits_matched_norm_stats()
    test_refiner_step_does_not_update_critic_or_projector()
    test_bounded_critic_training_stays_finite()
    test_checkpoint_roundtrip_preserves_bounded_critic()
    test_critic_health_gate_rejects_exploding_metrics()
    test_refine_delta_clip_matches_max_delta()
    print("ok")
