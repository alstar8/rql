"""Focused CPU regressions for V14 critic, endpoint, and gate contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_rlt_online as online
from chunk_replay import ChunkReplay, ChunkTransition
from rlt_models import ACTION_DIM, CHUNK_SIZE, Z_DIM, MolmoAct2RLTCF
from train_rlt import (
    _per_head_rank_loss,
    critic_td_step,
    flow_actor_step,
    guide_step,
)


def _model(*, flow: bool, guide: bool = True) -> MolmoAct2RLTCF:
    return MolmoAct2RLTCF(
        hidden=32,
        n_critics=4,
        token_d_model=32,
        token_layers=1,
        token_heads=4,
        cf_mode="flow" if flow else "residual",
        flow_steps=3,
        use_cf_guide=guide,
        tune_token_online=False,
    )


def _batch(batch_size: int = 4) -> dict[str, torch.Tensor]:
    reference = torch.randn(batch_size, CHUNK_SIZE, ACTION_DIM) * 0.02
    return {
        "z": torch.randn(batch_size, Z_DIM),
        "proprio": torch.randn(batch_size, ACTION_DIM),
        "reference_actions": reference,
        "executed_actions": reference + 0.01,
        "rewards": torch.zeros(batch_size, CHUNK_SIZE),
        "action_mask": torch.ones(batch_size, CHUNK_SIZE),
        "next_z": torch.randn(batch_size, Z_DIM),
        "next_proprio": torch.randn(batch_size, ACTION_DIM),
        "next_reference_actions": reference + 0.005,
        "terminal": torch.zeros(batch_size),
        "mc_return": torch.tensor([0.0, 1.0, 0.0, 1.0])[:batch_size],
        "success": torch.tensor([0.0, 1.0, 0.0, 1.0])[:batch_size],
    }


def _transition(episode_id: int, success: bool) -> ChunkTransition:
    reference = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    return ChunkTransition(
        z=np.full(Z_DIM, episode_id, dtype=np.float32),
        proprio=np.zeros(ACTION_DIM, dtype=np.float32),
        reference_actions=reference,
        executed_actions=reference,
        rewards=np.zeros(CHUNK_SIZE, dtype=np.float32),
        action_mask=np.ones(CHUNK_SIZE, dtype=np.float32),
        next_z=np.zeros(Z_DIM, dtype=np.float32),
        next_proprio=np.zeros(ACTION_DIM, dtype=np.float32),
        next_reference_actions=reference,
        terminal=True,
        mc_return=float(success),
        success=float(success),
        episode_id=episode_id,
        start_step=0,
    )


def test_per_head_rank_objective_backpropagates_to_every_member() -> None:
    positive = torch.zeros(4, 3, requires_grad=True)
    negative = torch.full((4, 3), 0.2, requires_grad=True)
    loss = _per_head_rank_loss(
        positive,
        negative,
        torch.ones(3),
        margin=0.05,
    )
    loss.backward()
    assert positive.grad is not None
    assert negative.grad is not None
    assert torch.all(positive.grad.abs().sum(dim=1) > 0)
    assert torch.all(negative.grad.abs().sum(dim=1) > 0)


def test_flow_actor_optimizes_through_every_deployment_step() -> None:
    torch.manual_seed(3)
    model = _model(flow=True, guide=True)
    batch = _batch()
    original_velocity = model.flow_velocity
    outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def recording_velocity(state, actions, timestep, reference):
        output = original_velocity(state, actions, timestep, reference)
        output.retain_grad()
        outputs.append((timestep.detach().clone(), output))
        return output

    model.flow_velocity = recording_velocity
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-4)
    alpha_optimizer = torch.optim.Adam(model.log_alpha.parameters(), lr=1e-4)
    info = flow_actor_step(
        model,
        actor_optimizer,
        alpha_optimizer,
        batch,
        endpoint_aux_coef=1.0,
        endpoint_aux_steps=1,
    )

    endpoint_outputs = outputs[-model.flow_steps :]
    assert [float(t.mean()) for t, _ in endpoint_outputs] == pytest.approx(
        [0.0, 1.0 / 3.0, 2.0 / 3.0]
    )
    assert all(output.grad is not None for _, output in endpoint_outputs)
    assert info["endpoint_steps"] == pytest.approx(model.flow_steps)
    assert info["endpoint_t"] == pytest.approx(1.0)
    assert info["actor_adv"] == pytest.approx(info["actor_endpoint_adv"])
    assert np.isfinite(info["residual_mse"])


@pytest.mark.parametrize("use_guide", [False, True])
def test_critic_target_guide_use_is_explicit(use_guide: bool) -> None:
    model = _model(flow=False, guide=True)
    batch = _batch()
    calls: list[bool] = []
    original_actor_chunk = model.actor_chunk

    def recording_actor_chunk(*args, **kwargs):
        calls.append(bool(kwargs.get("apply_guide", False)))
        return original_actor_chunk(*args, **kwargs)

    model.actor_chunk = recording_actor_chunk
    optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-4)
    critic_td_step(
        model,
        optimizer,
        batch,
        cql_coef=0.0,
        rank_coef=0.0,
        far_rank_coef=0.0,
        shuffle_rank_coef=0.0,
        target_noise=0.0,
        critic_target_use_guide=use_guide,
    )
    assert calls
    assert calls[0] is use_guide


def test_actor_can_pass_while_guide_remains_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay = ChunkReplay(max_transitions=8, seed=4)
    replay.add(_transition(0, False))
    replay.add(_transition(1, True))
    model = _model(flow=False, guide=True)
    args = SimpleNamespace(
        actor_mode="rlt",
        g_start_episodes=0,
        min_replay_chunks=2,
        batch_size=4,
        ae_batch_size=4,
        ae_min_success_episodes=1,
        gate_sensitivity_noise=0.08,
        g_min_advantage=0.003,
        g_min_guide_advantage=0.003,
        g_min_action_sensitivity=0.003,
        use_cf_guide=True,
        deploy_policy="gated",
    )
    monkeypatch.setattr(
        online,
        "critic_health_metrics",
        lambda *_args, **_kwargs: {"healthy": 1.0},
    )
    monkeypatch.setattr(
        online,
        "_split_gate_metrics",
        lambda *_args, **_kwargs: {
            "actor_lcb": 0.01,
            "actor_advantage": 0.01,
            "actor_sensitivity": 0.01,
            "guide_lcb": -0.01,
            "guide_advantage": -0.01,
            "guide_sensitivity": 0.01,
        },
    )

    status = online._gate_status(
        args,
        model,
        replay,
        valid_episodes=2,
        device=torch.device("cpu"),
    )
    assert status.actor_ready
    assert status.deploy_actor
    assert not status.guide_ready
    assert not status.deploy_guide
    assert status.actor_block_reason == ""
    assert status.guide_block_reason == "guide_lcb_below_threshold"


def test_dead_critic_gradient_suspends_guide_update() -> None:
    model = _model(flow=False, guide=True)
    for head in model.target_critic.critics:
        for parameter in head.parameters():
            parameter.data.zero_()
    optimizer = torch.optim.Adam(model.guide.parameters(), lr=1e-4)
    info = guide_step(model, optimizer, _batch())
    assert info["guide_update_skipped"] == 1.0
    assert info["guide_skip_tiny_critic_gradient"] == 1.0
    assert info["critic_gradient_raw_norm_min"] == pytest.approx(0.0)


def test_ae_initialization_resets_legacy_critic_into_native_coordinates() -> None:
    model = _model(flow=True, guide=True)
    with torch.no_grad():
        model.action_mean.fill_(0.5)
        model.action_std.fill_(0.2)
    before = next(model.critic.parameters()).detach().clone()
    args = SimpleNamespace(seed=9, eval_only=False)

    reset = online._configure_ae_native_rlt_coordinates(args, model, {})

    assert reset
    assert torch.equal(model.action_mean, torch.zeros_like(model.action_mean))
    assert torch.equal(model.action_std, torch.ones_like(model.action_std))
    assert not torch.equal(before, next(model.critic.parameters()))
    for online_head, target_head in zip(
        model.critic.critics,
        model.target_critic.critics,
    ):
        for online_parameter, target_parameter in zip(
            online_head.parameters(),
            target_head.parameters(),
        ):
            assert torch.equal(online_parameter, target_parameter)


def test_ae_eval_rejects_legacy_coordinate_checkpoint() -> None:
    model = _model(flow=True, guide=False)
    args = SimpleNamespace(seed=9, eval_only=True)
    with pytest.raises(RuntimeError, match="V14 native-coordinate checkpoint"):
        online._configure_ae_native_rlt_coordinates(args, model, {})
