"""Unit tests for VLA chunk prefetch overlap and obs-mismatch fallback."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import pytest


def _obs(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "external_cam": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
        "wrist_cam": rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8),
        "state": rng.standard_normal(7).astype(np.float32),
        "instruction": "pick up the kettle",
        "timestamp": time.time(),
    }


def _make_policy(*, require_match: bool = True, k: int = 2):
    os.environ["RLT_VLA_PREFETCH"] = "1"
    os.environ["RLT_VLA_PREFETCH_K"] = str(k)
    os.environ["RLT_VLA_PREFETCH_REQUIRE_OBS_MATCH"] = "1" if require_match else "0"
    # Import after env gates so _init_vla_prefetch sees them if constructed normally.
    from train_rlt_online import RLTOnlinePolicy

    policy = object.__new__(RLTOnlinePolicy)
    policy.request_timeout_sec = 5.0
    policy.ae_backend = None
    policy.chunk_size = 8
    policy.actions_buffer = [np.zeros(7, dtype=np.float32) for _ in range(8)]
    policy.current_buffer_index = 0
    policy._init_vla_prefetch()
    assert policy.vla_prefetch_enabled
    assert policy.vla_prefetch_require_obs_match is require_match

    call_times: list[float] = []
    call_obs: list[str] = []

    def fake_post_act(model_input, source_seed=None):
        call_times.append(time.perf_counter())
        call_obs.append(policy._obs_fingerprint(model_input))
        time.sleep(0.08)
        return {
            "actions": np.zeros((8, 7), dtype=np.float32),
            "source_seed": source_seed,
        }

    policy._post_act = fake_post_act  # type: ignore[method-assign]
    return policy, call_times, call_obs


def test_prefetch_overlaps_before_chunk_boundary() -> None:
    policy, call_times, _ = _make_policy(require_match=True, k=2)
    obs = _obs(0)
    policy.current_buffer_index = 6
    t0 = time.perf_counter()
    policy._maybe_start_vla_prefetch(obs)
    # Simulate remaining MuJoCo work while /act runs in the background.
    time.sleep(0.09)
    response = policy._consume_vla_prefetch(obs, source_seed=None)
    elapsed = time.perf_counter() - t0
    assert response is not None
    assert policy.vla_prefetch_hit == 1
    assert policy.vla_prefetch_miss == 0
    assert len(call_times) == 1
    # Wall should be ~MuJoCo sleep, not sleep + full /act (overlap).
    assert elapsed < 0.14
    assert policy.vla_prefetch_wait_ms < 40.0


def test_prefetch_miss_on_obs_mismatch_falls_back() -> None:
    policy, call_times, _ = _make_policy(require_match=True, k=2)
    obs_a = _obs(1)
    obs_b = _obs(2)
    policy.current_buffer_index = 6
    policy._maybe_start_vla_prefetch(obs_a)
    time.sleep(0.09)
    response = policy._consume_vla_prefetch(obs_b, source_seed=None)
    assert response is None
    assert policy.vla_prefetch_hit == 0
    assert policy.vla_prefetch_miss == 1
    assert policy.vla_prefetch_discarded >= 1
    # Caller would sync; ensure a sync call still works.
    synced = policy._post_act(obs_b)
    assert "actions" in synced
    assert len(call_times) == 2


def test_prefetch_accepts_without_obs_match_when_disabled() -> None:
    policy, call_times, _ = _make_policy(require_match=False, k=2)
    obs_a = _obs(3)
    obs_b = _obs(4)
    policy.current_buffer_index = 6
    policy._maybe_start_vla_prefetch(obs_a)
    time.sleep(0.09)
    response = policy._consume_vla_prefetch(obs_b, source_seed=None)
    assert response is not None
    assert policy.vla_prefetch_hit == 1
    assert policy.vla_prefetch_miss == 0
    assert len(call_times) == 1


def test_prefetch_hit_rate_over_chunks_without_obs_match() -> None:
    """V16 launch default: late-chunk prefetch consumed as next chunk."""
    policy, call_times, _ = _make_policy(require_match=False, k=2)
    hits = 0
    misses = 0
    chunks = 12
    for chunk_idx in range(chunks):
        obs = _obs(100 + chunk_idx)
        if chunk_idx == 0:
            # First chunk has no prefetch.
            response = policy._consume_vla_prefetch(obs, source_seed=None)
            assert response is None
            misses += 1
            policy._post_act(obs)
        else:
            response = policy._consume_vla_prefetch(obs, source_seed=None)
            if response is None:
                misses += 1
                policy._post_act(obs)
            else:
                hits += 1
        policy.actions_buffer = [np.zeros(7, dtype=np.float32) for _ in range(8)]
        policy.current_buffer_index = 0
        # Drain chunk and trigger prefetch near the end.
        for step in range(8):
            policy.current_buffer_index = step + 1
            policy._maybe_start_vla_prefetch(_obs(100 + chunk_idx))
            if step >= 5:
                time.sleep(0.01)
        time.sleep(0.09)
    hit_rate = hits / max(hits + misses, 1)
    assert hit_rate >= 0.8
    assert len(call_times) >= chunks
