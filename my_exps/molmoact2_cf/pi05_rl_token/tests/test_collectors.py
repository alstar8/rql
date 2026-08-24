"""Online collectors share one GPU with a compiled VLA server.

The crash on the first /act was torch.compile CUDA-graph capture
(CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED). Collectors still need
the GPU visible: NVIDIA EGL follows CUDA_VISIBLE_DEVICES, and hiding the card
made MuJoCo fall through to the macOS CGL renderer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05.collectors import collector_env  # noqa: E402
from pi05.config import EvalConfig, RLConfig  # noqa: E402
from pi05.eval import policy_server_env, prepare_environment  # noqa: E402


def test_collector_env_marks_the_process_and_keeps_cuda_visible(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    env = collector_env(tmp_path)
    assert env["RLT_COLLECTOR"] == "1"
    assert env["MUJOCO_GL"] == "egl"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PI05_VLA_LOCK_DIR"] == str(tmp_path / "vla_lock")


def test_prepare_environment_keeps_cuda_for_collectors_egl(tmp_path, monkeypatch):
    monkeypatch.setenv("RLT_COLLECTOR", "1")
    cfg = RLConfig(scene="desk_mug", out_dir=tmp_path, gpu=1, egl_device=1, tag="c")
    prepare_environment(cfg)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["MUJOCO_EGL_DEVICE_ID"] == "1"
    assert os.environ["MUJOCO_GL"] == "egl"


def test_policy_server_disables_inductor_cudagraphs():
    cfg = EvalConfig(scene="desk_mug", gpu=1, port=8540)
    env = policy_server_env(cfg)
    assert env["TORCHINDUCTOR_CUDAGRAPHS"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"


def test_disable_policy_compile_replaces_torch_compile():
    import torch

    from pi05.eval import disable_policy_compile

    original = torch.compile
    try:
        disable_policy_compile()
        sentinel = object()
        assert torch.compile(sentinel) is sentinel
    finally:
        torch.compile = original
