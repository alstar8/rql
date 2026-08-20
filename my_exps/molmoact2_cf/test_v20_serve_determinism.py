"""Server-side VLA RNG isolation contract for V20."""

from __future__ import annotations

import sys
import threading
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rlt_models import ACTION_DIM, CHUNK_SIZE  # noqa: E402
from serve import CFPolicy  # noqa: E402


class _SeededServerModel:
    model = None

    def predict_action(self, **_kwargs: Any) -> SimpleNamespace:
        offset = float(np.random.random()) + random.random()
        return SimpleNamespace(
            actions=torch.randn(1, CHUNK_SIZE, ACTION_DIM) + offset
        )


def test_server_source_seed_forks_and_restores_torch_rng() -> None:
    policy = object.__new__(CFPolicy)
    policy.model = _SeededServerModel()
    policy.processor = object()
    policy.device = "cpu"
    policy.return_features = False
    policy.feature_mode = "mean_pool"
    policy.rlt = None
    policy.enable_g = False
    policy.cf = None
    policy._lock = threading.Lock()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    state = np.zeros(ACTION_DIM, dtype=np.float32)

    torch.manual_seed(77)
    expected_after = torch.randn(5)
    np.random.seed(77)
    expected_numpy_after = np.random.random(5)
    random.seed(77)
    expected_python_after = [random.random() for _ in range(5)]
    torch.manual_seed(77)
    np.random.seed(77)
    random.seed(77)
    first, _ = policy.predict(
        image,
        image,
        "pick up the kettle",
        state,
        source_seed=999,
    )
    actual_after = torch.randn(5)
    actual_numpy_after = np.random.random(5)
    actual_python_after = [random.random() for _ in range(5)]
    second, _ = policy.predict(
        image,
        image,
        "pick up the kettle",
        state,
        source_seed=999,
    )
    other, _ = policy.predict(
        image,
        image,
        "pick up the kettle",
        state,
        source_seed=1000,
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert torch.equal(actual_after, expected_after)
    assert np.array_equal(actual_numpy_after, expected_numpy_after)
    assert actual_python_after == expected_python_after
