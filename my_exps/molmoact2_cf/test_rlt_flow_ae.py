"""Focused CPU tests for V13 ConsensusFlow and AE correctness."""

from __future__ import annotations

import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_replay import ChunkReplay, ImageChunkReplay
from molmo_ae_backend import MolmoAEBackend
from rlt_models import (
    ACTION_DIM,
    CF_MODE_FLOW,
    CHUNK_SIZE,
    STATE_DIM,
    Z_DIM,
    CFGradientGuide,
    MolmoAct2RLTCF,
)
from train_rlt import (
    _batch_state,
    _flow_reverse_state,
    _index_batch_rows,
    ae_flow_critic_td_step,
    flow_gate_metrics,
    guide_step,
    stochastic_target_critic_gradient,
)
from train_rlt_online import (
    RLTFeatureError,
    RLTOnlinePolicy,
    _gate_status,
    _parse_snapshot_episodes,
    _resolve_resume_artifacts,
    _save_eval_snapshot,
    parse_args,
)


def _small_model(*, flow: bool, guide: bool = True) -> MolmoAct2RLTCF:
    return MolmoAct2RLTCF(
        hidden=32,
        n_critics=2,
        token_d_model=32,
        token_layers=1,
        token_heads=4,
        cf_mode=CF_MODE_FLOW if flow else "residual",
        flow_steps=2,
        use_cf_guide=guide,
        tune_token_online=False,
    )


def _image_episode() -> ImageChunkReplay:
    replay = ImageChunkReplay(max_transitions=8, pos_frac=0.0, seed=0)
    zs = [
        np.full(Z_DIM, 1.0, dtype=np.float32),
        np.full(Z_DIM, 2.0, dtype=np.float32),
    ]
    proprios = [
        np.full(ACTION_DIM, 3.0, dtype=np.float32),
        np.full(ACTION_DIM, 4.0, dtype=np.float32),
    ]
    references = [
        np.full((CHUNK_SIZE, ACTION_DIM), 0.1, dtype=np.float32),
        np.full((CHUNK_SIZE, ACTION_DIM), 0.2, dtype=np.float32),
    ]
    executed = [row + 0.01 for row in references]
    rewards = [
        np.arange(CHUNK_SIZE, dtype=np.float32) / 100.0,
        np.full(CHUNK_SIZE, 0.5, dtype=np.float32),
    ]
    masks = [
        np.ones(CHUNK_SIZE, dtype=np.float32),
        np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    ]
    external = [
        np.full((3, 4, 3), 11, dtype=np.uint8),
        np.full((3, 4, 3), 22, dtype=np.uint8),
    ]
    wrist = [
        np.full((2, 3, 3), 33, dtype=np.uint8),
        np.full((2, 3, 3), 44, dtype=np.uint8),
    ]
    replay.add_episode(
        zs=zs,
        proprios=proprios,
        references=references,
        executed=executed,
        rewards=rewards,
        masks=masks,
        external_cams=external,
        wrist_cams=wrist,
        instructions=["first", "second"],
        success=True,
        gamma=0.9,
        episode_id=7,
    )
    return replay


class FakeAEBackend:
    def __init__(self) -> None:
        self.context_calls: list[int] = []
        self.velocity_calls = 0

    def eval(self) -> None:
        return None

    def train(self, _mode: bool = True) -> None:
        return None

    def action_contract(self):
        return {
            "action_horizon": 15,
            "action_dim": ACTION_DIM,
            "max_action_dim": 32,
        }

    def normalize_actions(self, actions):
        return torch.as_tensor(actions, dtype=torch.float32)

    def encode_context(
        self,
        external,
        _wrist,
        _instruction,
        _state,
        *,
        action_horizon,
    ):
        assert action_horizon == 15
        marker = int(np.asarray(external)[0, 0, 0])
        self.context_calls.append(marker)
        return SimpleNamespace(
            marker=marker,
            action_horizon=15,
            action_dim=ACTION_DIM,
            max_action_dim=32,
        ), {}

    def velocity(self, context, x_t, t):
        self.velocity_calls += 1
        return (
            torch.zeros_like(x_t)
            + (float(context.marker) / 1000.0)
            + 0.0 * t.view(-1, 1, 1)
        )


def test_raw_w_distill_and_deploy_bound():
    torch.manual_seed(4)
    model = _small_model(flow=False, guide=True)
    guide = model.guide
    assert isinstance(guide, CFGradientGuide)
    last_linear = [module for module in guide.net.modules() if isinstance(module, nn.Linear)][-1]
    with torch.no_grad():
        last_linear.bias.fill_(2.0)

    batch = {
        "z": torch.randn(3, Z_DIM),
        "proprio": torch.randn(3, ACTION_DIM),
        "reference_actions": torch.randn(3, CHUNK_SIZE, ACTION_DIM) * 0.01,
    }
    with torch.no_grad():
        state = _batch_state(model, batch, detach_token=True)
        reference = model.normalize_action(batch["reference_actions"])
        actor, _ = model.actor_chunk(state, reference, deterministic=True)
        raw_w = guide.raw_w(state, reference)
        bounded = guide(state, reference)
    assert raw_w.abs().max() > 1.0
    assert bounded.abs().max() <= guide.max_delta + 1e-6

    torch.manual_seed(17)
    target = stochastic_target_critic_gradient(model, state, actor)
    expected = torch.nn.functional.mse_loss(raw_w, target)
    optimizer = torch.optim.SGD(guide.parameters(), lr=0.0)
    torch.manual_seed(17)
    info = guide_step(
        model,
        optimizer,
        batch,
        beta=0.0,
        target_delta_frac=999.0,
    )
    assert info["guide_distill"] == pytest.approx(float(expected), rel=1e-5)
    assert info["guide_loss"] == pytest.approx(float(expected), rel=1e-5)
    assert info["w_norm"] == pytest.approx(float(raw_w.norm(dim=-1).mean()), rel=1e-5)
    assert info["target_norm"] == pytest.approx(float(target.norm(dim=-1).mean()), rel=1e-5)


def test_reverse_state_uses_injected_flow_provider():
    model = _small_model(flow=True, guide=False)
    state = torch.randn(4, STATE_DIM)
    actions = torch.randn(4, CHUNK_SIZE, ACTION_DIM)
    reference = torch.zeros_like(actions)
    calls = 0

    def forbidden(*_args, **_kwargs):
        raise AssertionError("lightweight flow_velocity must not be called")

    def provider(_state, x_t, _t, _reference):
        nonlocal calls
        calls += 1
        return torch.ones_like(x_t) * 0.25

    model.flow_velocity = forbidden
    x_t, t = _flow_reverse_state(
        model,
        state,
        actions,
        reference,
        apply_guide=False,
        velocity_provider=provider,
    )
    assert calls == model.flow_steps
    assert x_t.shape == actions.shape
    assert t.shape == (actions.shape[0], 1)


def test_flow_gate_pairs_actor_and_guide_source_noise():
    model = _small_model(flow=True, guide=True)
    batch = {
        "z": torch.randn(3, Z_DIM),
        "proprio": torch.randn(3, ACTION_DIM),
        "reference_actions": torch.randn(3, CHUNK_SIZE, ACTION_DIM) * 0.01,
        "executed_actions": torch.randn(3, CHUNK_SIZE, ACTION_DIM) * 0.01,
    }
    source_noises: list[torch.Tensor] = []
    original = model.flow_sample

    def recording_flow_sample(state, reference, **kwargs):
        source_noises.append(kwargs["x0"].detach().clone())
        return original(state, reference, **kwargs)

    model.flow_sample = recording_flow_sample
    metrics = flow_gate_metrics(model, batch)
    assert len(source_noises) == 2
    assert torch.equal(source_noises[0], source_noises[1])
    assert all(np.isfinite(value) for value in metrics.values())


def test_image_replay_has_true_next_reward_and_terminal_fields():
    replay = _image_episode()
    first, last = replay.rows
    assert np.array_equal(first.next_z, last.z)
    assert np.array_equal(first.next_external_cam, last.external_cam)
    assert first.next_instruction == "second"
    assert np.array_equal(first.rewards, np.arange(CHUNK_SIZE, dtype=np.float32) / 100.0)
    assert first.terminal is False
    assert last.terminal is True
    assert np.count_nonzero(last.next_external_cam) == 0
    assert last.next_instruction == ""
    assert first.mc_return == pytest.approx(0.9 ** 9)
    assert last.mc_return == pytest.approx(0.9)
    batch = replay.sample(2)
    for key in ("rewards", "terminal", "mc_return", "next_z", "next_proprio"):
        assert key in batch
    assert "next_external_cam" in batch
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "image_replay.npz"
        replay.save_npz(str(path))
        restored = ImageChunkReplay.load_npz(str(path), max_transitions=8)
    assert len(restored) == len(replay)
    assert restored.n_episodes == replay.n_episodes
    assert np.array_equal(
        restored.rows[0].next_external_cam,
        replay.rows[0].next_external_cam,
    )
    assert restored.rows[0].instruction == replay.rows[0].instruction


def test_ae_critic_routes_all_flow_through_backend():
    torch.manual_seed(0)
    model = _small_model(flow=True, guide=False)
    backend = FakeAEBackend()
    replay = _image_episode()
    batch = replay.sample(2)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("model.flow_velocity must not run in AE critic mode")

    model.flow_velocity = forbidden
    optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-4)
    info = ae_flow_critic_td_step(
        model,
        backend,
        optimizer,
        batch,
        cql_coef=0.0,
        cql_n_actions=1,
        rank_coef=0.0,
        far_rank_coef=0.0,
        shuffle_rank_coef=0.0,
        target_noise=0.0,
    )
    assert backend.velocity_calls > 0
    assert all(np.isfinite(value) for value in info.values())
    # Terminal next observations are placeholders and must not be encoded.
    assert 0 not in backend.context_calls


class FakePolicyBackend:
    def __init__(self) -> None:
        self.reference_calls = 0
        self.actor_calls: list[bool] = []

    def predict_reference(self, *_args, **_kwargs):
        self.reference_calls += 1
        return {
            "actions": np.ones((CHUNK_SIZE, ACTION_DIM), dtype=np.float32),
            "z_rl": np.ones(Z_DIM, dtype=np.float32),
        }

    def predict(self, *_args, apply_guide=False, rlt_state=None, **_kwargs):
        self.actor_calls.append(bool(apply_guide))
        if apply_guide:
            assert rlt_state is not None
        value = 3.0 if apply_guide else 2.0
        return {"actions": np.full((CHUNK_SIZE, ACTION_DIM), value, dtype=np.float32)}


def _bare_policy(model: MolmoAct2RLTCF, backend: FakePolicyBackend) -> RLTOnlinePolicy:
    policy = object.__new__(RLTOnlinePolicy)
    policy.rlt_model = model
    policy.rlt_device = torch.device("cpu")
    policy.use_cf_guide = True
    policy.actor_mode = "rlt"
    policy.prefer_server_z = True
    policy.retain_tokens = False
    policy.explore_residual_std = 0.0
    policy.explore_deploy_std = 0.0
    policy.explore_warmup_mult = 1.0
    policy.ae_backend = backend
    policy.deploy_actor = False
    policy.deploy_guide = False
    policy.chunk_size = CHUNK_SIZE
    policy._rng = np.random.default_rng(0)
    policy._clear_episode()
    return policy


def test_frozen_reference_gate_switches_ae_deployment():
    model = _small_model(flow=True, guide=True)
    model.v_source = "molmo_ae"
    backend = FakePolicyBackend()
    policy = _bare_policy(model, backend)
    model_input = {
        "external_cam": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_cam": np.zeros((2, 2, 3), dtype=np.uint8),
        "instruction": "task",
        "state": np.zeros(ACTION_DIM, dtype=np.float32),
    }

    policy._start_chunk(model_input)
    assert np.allclose(policy.ep_references[-1], 1.0)
    assert np.allclose(policy.actions_buffer, 1.0)
    assert backend.actor_calls == []

    policy._clear_episode()
    policy.deploy_actor = True
    policy.deploy_guide = False
    policy._start_chunk(model_input)
    assert np.allclose(policy.ep_references[-1], 1.0)
    assert np.allclose(policy.actions_buffer, 2.0)

    policy._clear_episode()
    policy.deploy_guide = True
    policy._start_chunk(model_input)
    assert np.allclose(policy.ep_references[-1], 1.0)
    assert np.allclose(policy.actions_buffer, 3.0)
    assert backend.reference_calls == 3
    assert backend.actor_calls == [False, True]


def test_ae_trainable_roundtrip_and_adapter_disable():
    backend = object.__new__(MolmoAEBackend)
    backend.invalidate_modulation_cache = lambda: None
    backend.model = nn.Linear(3, 2)
    original = backend.model.weight.detach().clone()
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "ae.pt"
        backend.save_trainable(path, meta={"env_steps": 5})
        with torch.no_grad():
            backend.model.weight.add_(10.0)
        backend.load_trainable(path)
        bad_path = Path(temp_dir) / "bad_ae.pt"
        torch.save({"ae_trainable": {"wrong": torch.zeros(1)}}, bad_path)
        with pytest.raises(RuntimeError, match="checkpoint mismatch"):
            backend.load_trainable(bad_path)
    assert torch.allclose(backend.model.weight, original)

    class AdapterModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.disabled = False

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

    adapter_model = AdapterModel()
    backend.model = adapter_model
    assert not adapter_model.disabled
    with backend.adapter_disabled():
        assert adapter_model.disabled
    assert not adapter_model.disabled

    backend.model = nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="stable frozen reference"):
        with backend.adapter_disabled():
            pass


def test_guided_ae_requires_model_guide_and_state():
    backend = object.__new__(MolmoAEBackend)
    backend.num_steps = 1
    backend._lock = threading.Lock()
    backend.rlt = None
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    state = np.zeros(ACTION_DIM, dtype=np.float32)
    with pytest.raises(RuntimeError, match="RLT model"):
        backend.predict(image, image, "task", state, apply_guide=True)

    backend.rlt = _small_model(flow=True, guide=True)
    with pytest.raises(RuntimeError, match="encoded RLT state"):
        backend.predict(image, image, "task", state, apply_guide=True)


def test_safe_resume_requires_model_replay_and_ae_together():
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)
        replay_path = out_dir / "chunk_replay.npz"
        args = SimpleNamespace(
            no_resume=False,
            out_dir=str(out_dir),
            replay_out=str(replay_path),
            ae_trainable=True,
            eval_only=False,
        )
        (out_dir / "rlt_cf_latest.pt").write_bytes(b"checkpoint")
        replay_path.write_bytes(b"replay")
        with pytest.raises(RuntimeError, match="partial AE resume"):
            _resolve_resume_artifacts(args)
        ae_path = out_dir / "molmo_ae_lora_latest.pt"
        ae_path.write_bytes(b"adapter")
        with pytest.raises(RuntimeError, match="partial AE resume"):
            _resolve_resume_artifacts(args)
        ae_replay_path = out_dir / "ae_image_replay.npz"
        ae_replay_path.write_bytes(b"image replay")
        artifacts = _resolve_resume_artifacts(args)
        assert artifacts.checkpoint == out_dir / "rlt_cf_latest.pt"
        assert artifacts.replay == replay_path
        assert artifacts.ae_trainable == ae_path
        assert artifacts.ae_replay == ae_replay_path


def test_eval_infers_ae_checkpoint_beside_rlt_checkpoint():
    with tempfile.TemporaryDirectory() as temp_dir:
        train_dir = Path(temp_dir) / "train"
        eval_dir = Path(temp_dir) / "eval"
        train_dir.mkdir()
        rlt_path = train_dir / "rlt_cf_latest.pt"
        ae_path = train_dir / "molmo_ae_lora_latest.pt"
        rlt_path.write_bytes(b"rlt")
        ae_path.write_bytes(b"ae")
        args = SimpleNamespace(
            no_resume=False,
            out_dir=str(eval_dir),
            replay_out="",
            rlt_ckpt=str(rlt_path),
            ae_trainable_ckpt="",
            ae_trainable=True,
            eval_only=True,
        )
        artifacts = _resolve_resume_artifacts(args)
        assert artifacts.checkpoint is None
        assert artifacts.replay is None
        assert artifacts.ae_trainable == ae_path
        assert artifacts.ae_replay is None


def test_eval_snapshot_is_immutable_and_keeps_gate_meta():
    model = _small_model(flow=False, guide=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        out_dir = Path(temp_dir)
        snapshot = _save_eval_snapshot(
            model,
            None,
            out_dir,
            valid_episodes=100,
            env_steps=1234,
            meta={"gate_deploy_actor": True, "gate_deploy_guide": True},
        )
        checkpoint = snapshot / "rlt_cf.pt"
        first_mtime = checkpoint.stat().st_mtime_ns
        second = _save_eval_snapshot(
            model,
            None,
            out_dir,
            valid_episodes=100,
            env_steps=9999,
            meta={"gate_deploy_actor": False},
        )
        second_mtime = checkpoint.stat().st_mtime_ns
        restored = MolmoAct2RLTCF.load(str(checkpoint))
    assert second == snapshot
    assert second_mtime == first_mtime
    assert restored.loaded_meta["valid_episodes"] == 100
    assert restored.loaded_meta["gate_deploy_actor"] is True
    assert _parse_snapshot_episodes("0, 100,100,400") == {0, 100, 400}


def _gate_args(policy: str) -> SimpleNamespace:
    return SimpleNamespace(
        actor_mode="rlt",
        deploy_policy=policy,
        g_start_episodes=10,
        min_replay_chunks=2,
        batch_size=2,
        ae_batch_size=2,
        gate_sensitivity_noise=0.1,
        g_min_advantage=0.1,
        g_min_action_sensitivity=0.1,
        g_min_guide_advantage=0.0,
        use_cf_guide=True,
    )


def test_deploy_policy_preserves_would_enable():
    model = _small_model(flow=False, guide=True)
    replay = ChunkReplay()
    gated = _gate_status(_gate_args("gated"), model, replay, 0, torch.device("cpu"))
    actor = _gate_status(_gate_args("actor"), model, replay, 0, torch.device("cpu"))
    guided = _gate_status(
        _gate_args("actor_guide"),
        model,
        replay,
        0,
        torch.device("cpu"),
    )
    checkpoint = _gate_status(
        _gate_args("checkpoint_gate"),
        model,
        replay,
        0,
        torch.device("cpu"),
        checkpoint_meta={"gate_deploy_actor": True, "gate_deploy_guide": False},
    )
    reference = _gate_status(
        _gate_args("reference"),
        model,
        replay,
        0,
        torch.device("cpu"),
    )
    assert not gated.would_enable and not gated.deploy_actor
    assert not actor.would_enable and actor.deploy_actor and not actor.deploy_guide
    assert not guided.would_enable and guided.deploy_actor and guided.deploy_guide
    assert not checkpoint.would_enable and checkpoint.deploy_actor
    assert not reference.would_enable and not reference.deploy_actor


def test_strict_reward_cache_errors():
    policy = object.__new__(RLTOnlinePolicy)
    policy.fatal_error = None
    policy.task = SimpleNamespace()
    with pytest.raises(RLTFeatureError, match="no reward_cache"):
        policy._task_rewards(1, False)

    policy.fatal_error = None
    policy.task = SimpleNamespace(reward_cache=[0.0, 0.1])
    with pytest.raises(RLTFeatureError, match="short"):
        policy._task_rewards(2, False)

    policy.fatal_error = None
    policy.task = SimpleNamespace(reward_cache=[0.0, np.nan])
    with pytest.raises(RLTFeatureError, match="non-finite"):
        policy._task_rewards(1, False)

    policy.fatal_error = None
    policy.task = SimpleNamespace(reward_cache=[0.0, 0.25, 0.5])
    assert np.allclose(policy._task_rewards(2, False), [0.25, 0.5])

    policy.fatal_error = None
    policy.task = SimpleNamespace(reward_cache=[0.0, 0.0])
    with pytest.raises(RLTFeatureError, match="no positive"):
        policy._task_rewards(1, True)


def test_zero_rl_feature_is_rejected():
    model = _small_model(flow=False, guide=False)
    policy = object.__new__(RLTOnlinePolicy)
    policy.rlt_model = model
    policy.prefer_server_z = True
    with pytest.raises(RLTFeatureError, match="all zero"):
        policy._response_z(
            {"z_rl": np.zeros(Z_DIM, dtype=np.float32)},
            None,
            None,
        )


def test_index_batch_rows_handles_numpy_and_tensors() -> None:
    batch_size = 4
    row_index = torch.tensor([0, 2])
    batch = {
        "t": torch.arange(4),
        "n": np.arange(4),
        "l": ["a", "b", "c", "d"],
        "scalar": 7,
    }
    sub = _index_batch_rows(batch, row_index, batch_size)
    assert list(sub["t"].tolist()) == [0, 2]
    assert list(sub["n"]) == [0, 2]
    assert sub["l"] == ["a", "c"]
    assert sub["scalar"] == 7


def test_eval_only_configuration_disables_training_and_exploration(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_rlt_online.py", "--eval_only", "--target_env_steps", "1"],
    )
    args = parse_args()
    assert args.eval_only
    assert args.updates_per_episode == 0
    assert args.tune_token_online is False
    assert args.explore_residual_std == 0.0
    assert args.explore_deploy_std == 0.0
