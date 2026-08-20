"""Determinism contracts used by V20 paired evaluation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlt_models import ACTION_DIM, CHUNK_SIZE, Z_DIM, MolmoAct2RLTCF  # noqa: E402
from train_rlt_online import RLTOnlinePolicy, select_benchmark_episode_idx  # noqa: E402
from v20_runner import _normalize_trajectory  # noqa: E402


class _Response:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class _Session:
    def __init__(self, *, echo_seed: bool) -> None:
        self.echo_seed = echo_seed
        self.last_payload: dict[str, Any] | None = None

    def post(
        self,
        _url: str,
        *,
        json: dict[str, Any],
        timeout: float,
    ) -> _Response:
        del timeout
        self.last_payload = json
        body: dict[str, Any] = {
            "actions": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
        }
        if self.echo_seed:
            body["source_seed"] = int(json["source_seed"])
        return _Response(body)


def test_select_benchmark_episode_idx_pins_and_wraps() -> None:
    assert (
        select_benchmark_episode_idx(
            n_bench=24,
            cycle=82,
            shard_start=0,
            shard_size=125,
            pinned_idx=0,
        )
        == 0
    )
    assert (
        select_benchmark_episode_idx(
            n_bench=24,
            cycle=24,
            shard_start=0,
            shard_size=125,
        )
        == 0
    )
    assert (
        select_benchmark_episode_idx(
            n_bench=24,
            cycle=23,
            shard_start=0,
            shard_size=125,
        )
        == 23
    )
    with pytest.raises(ValueError, match="out of range"):
        select_benchmark_episode_idx(
            n_bench=24,
            cycle=0,
            shard_start=0,
            shard_size=1,
            pinned_idx=24,
        )


def test_flow_noise_seed_is_independent_of_global_rng() -> None:
    torch.manual_seed(1)
    model = MolmoAct2RLTCF(
        token_layers=1,
        token_d_model=64,
        n_critics=2,
        cf_mode="flow",
        use_cf_guide=False,
        use_cfgrl=True,
        hidden=32,
        n_hidden_actor=2,
        n_hidden_critic=2,
    )
    model.eval()
    state = torch.randn(2, Z_DIM + ACTION_DIM)
    reference = torch.zeros(2, CHUNK_SIZE, ACTION_DIM)
    first, _ = model.flow_sample(state, reference, flow_noise_seed=917)
    _ = torch.randn(1000)
    second, _ = model.flow_sample(state, reference, flow_noise_seed=917)
    other, _ = model.flow_sample(state, reference, flow_noise_seed=918)
    assert torch.equal(first, second)
    assert not torch.equal(first, other)
    with pytest.raises(ValueError, match="either x0 or flow_noise_seed"):
        model.flow_sample(
            state,
            reference,
            x0=torch.zeros_like(reference),
            flow_noise_seed=917,
        )


def test_http_source_seed_is_sent_and_must_be_echoed() -> None:
    model_input = {
        "external_cam": np.zeros((2, 2, 3), dtype=np.uint8),
        "wrist_cam": np.zeros((2, 2, 3), dtype=np.uint8),
        "instruction": "pick up the kettle",
        "state": np.zeros(ACTION_DIM, dtype=np.float32),
        "timestamp": 0.0,
    }
    policy = object.__new__(RLTOnlinePolicy)
    policy.ae_backend = None
    policy.url = "http://example.invalid/act"
    policy.request_timeout_sec = 1.0
    policy.deterministic_rollout_seeds = True
    policy.session = _Session(echo_seed=True)
    body = policy._post_act(model_input, source_seed=12345)
    assert body["source_seed"] == 12345
    assert policy.session.last_payload is not None
    assert policy.session.last_payload["source_seed"] == 12345

    policy.session = _Session(echo_seed=False)
    with pytest.raises(RuntimeError, match="did not preserve"):
        policy._post_act(model_input, source_seed=12345)


def test_normalize_trajectory_maps_pop_episode_keys() -> None:
    reference = [np.ones((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)]
    executed = [np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)]
    masks = [np.ones((CHUNK_SIZE,), dtype=np.float32)]
    real = {
        "zs": [np.zeros((Z_DIM,), dtype=np.float32)],
        "proprios": [np.zeros((ACTION_DIM,), dtype=np.float32)],
        "references": reference,
        "executed": executed,
        "rewards": [np.zeros((CHUNK_SIZE,), dtype=np.float32)],
        "masks": masks,
        "n_steps": CHUNK_SIZE,
    }
    normalized = _normalize_trajectory(real)
    assert normalized["reference_actions"] is reference
    assert normalized["executed_actions"] is executed
    assert normalized["action_masks"] is masks

    canonical = {
        "reference_actions": reference,
        "executed_actions": executed,
        "action_masks": masks,
    }
    assert _normalize_trajectory(canonical)["reference_actions"] is reference

    with pytest.raises(KeyError, match="reference_actions"):
        _normalize_trajectory({"executed": executed, "masks": masks})
