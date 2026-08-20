"""CPU tests for the critic-free V20 challenger path."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_replay import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    Z_DIM,
    ChunkReplay,
    ReplaySource,
)
from rlt_models import MolmoAct2RLTCF  # noqa: E402
from v20_training import (  # noqa: E402
    ChallengerTrainingConfig,
    _diagnose_candidate,
    build_v20_optimizers,
    optimizer_parameter_ids,
    target_positive_fraction,
    train_challenger,
)


def _replay() -> ChunkReplay:
    replay = ChunkReplay(
        max_transitions=100,
        pos_frac=0.5,
        benchmark_pose_cycle=24,
        seed=11,
    )
    for episode_id in range(16):
        success = episode_id < 8
        reference = np.full(
            (CHUNK_SIZE, ACTION_DIM),
            episode_id / 100.0,
            dtype=np.float32,
        )
        replay.add_episode_chunks(
            [np.full((Z_DIM,), episode_id / 10.0, dtype=np.float32)],
            [np.zeros((ACTION_DIM,), dtype=np.float32)],
            [reference],
            [reference + (0.01 if success else 0.0)],
            [np.full((CHUNK_SIZE,), float(success), dtype=np.float32)],
            [np.ones((CHUNK_SIZE,), dtype=np.float32)],
            success=success,
            gamma=0.99,
            episode_id=episode_id,
            trajectory_uid=episode_id,
            pose_idx=0 if success else episode_id % 2,
            source_policy=ReplaySource.OFFLINE_REFERENCE,
            worker_id=0,
            round_id=-1,
            policy_version=0,
        )
    return replay


def _small_model() -> MolmoAct2RLTCF:
    model = MolmoAct2RLTCF(
        token_layers=1,
        token_d_model=64,
        n_critics=2,
        cf_mode="flow",
        flow_steps=2,
        use_cf_guide=False,
        use_cfgrl=True,
        cfgrl_o_dim=8,
        hidden=32,
        n_hidden_actor=2,
        n_hidden_critic=2,
        z_expand_dim=Z_DIM + 8,
        layernorm_heads=True,
    )
    model.set_norm_stats(
        torch.zeros(ACTION_DIM),
        torch.ones(ACTION_DIM),
        torch.zeros(ACTION_DIM),
        torch.ones(ACTION_DIM),
    )
    return model


def _copies(parameters: object) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in parameters  # type: ignore[union-attr]
    ]


def test_v20_optimizer_paths_are_disjoint() -> None:
    model = _small_model()
    optimizers = build_v20_optimizers(model, actor_lr=1e-4)
    actor_ids = optimizer_parameter_ids(optimizers["actor"])
    critic_ids = optimizer_parameter_ids(optimizers["critic"])
    adapter_ids = {
        id(parameter) for parameter in model.rlt_adapter_parameters()
    }
    assert not actor_ids & critic_ids
    assert adapter_ids <= actor_ids
    assert not adapter_ids & critic_ids


def test_critic_free_training_updates_actor_adapter_only() -> None:
    model = _small_model()
    replay = _replay()
    critic_before = _copies(model.critic.parameters())
    token_before = _copies(model.token_ae.parameters())
    actor_before = _copies(model.actor.parameters())
    adapter_before = _copies(model.rlt_adapter_parameters())
    result = train_challenger(
        model,
        replay,
        config=ChallengerTrainingConfig(
            actor_steps=2,
            batch_size=8,
            actor_lr=1e-3,
            cond_dropout=0.1,
            target_pose_idx=0,
            temporal_bins=2,
            diagnostic_batches=1,
            max_normalized_action=100.0,
        ),
        device="cpu",
        seed=123,
    )
    assert result.critic_updates == 0
    assert result.token_updates == 0
    assert result.target_positive_fraction == 0.5
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(critic_before, model.critic.parameters())
    )
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(token_before, model.token_ae.parameters())
    )
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(actor_before, model.actor.parameters())
    )
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(
            adapter_before,
            model.rlt_adapter_parameters(),
        )
    )


def test_target_positive_fraction_schedule() -> None:
    assert target_positive_fraction(0) == 0.5
    assert target_positive_fraction(8) == 0.5
    assert target_positive_fraction(15) == 0.5
    assert target_positive_fraction(16) == 0.75
    assert target_positive_fraction(31) == 0.75
    assert target_positive_fraction(32) == 0.9


def test_diagnose_candidate_gates_deployed_cond_head() -> None:
    model = _small_model()
    replay = _replay()
    strict = ChallengerTrainingConfig(
        batch_size=8,
        diagnostic_batches=1,
        cond_ref_mse_max=1e-9,
        max_normalized_action=100.0,
    )
    diagnostics = _diagnose_candidate(
        model,
        replay,
        config=strict,
        target_fraction=0.5,
        device="cpu",
        seed=7,
    )
    assert diagnostics["cond_ref_mse"] > 1e-9
    assert diagnostics["cond_ok"] is False
    generous = ChallengerTrainingConfig(
        batch_size=8,
        diagnostic_batches=1,
        cond_ref_mse_max=1e6,
        max_normalized_action=100.0,
    )
    diagnostics = _diagnose_candidate(
        model,
        replay,
        config=generous,
        target_fraction=0.5,
        device="cpu",
        seed=7,
    )
    assert diagnostics["cond_ok"] is True
