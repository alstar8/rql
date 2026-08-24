"""Client side of the frozen VLA: one HTTP call gives both halves of the state.

`serve_vla.py` returns the reference action chunk and the final-layer token
sequence from the same forward pass. The token sequence goes through the frozen
phase-1 encoder to become z_rl; the chunk becomes the actor's reference.
"""

from __future__ import annotations

import time

import json_numpy
import numpy as np
import requests
import torch

from .config import ACTION_DIM, VLA_TOKEN_DIM
from .token_ae import RLTokenAE

json_numpy.patch()


class VlaClient:
    """POST /act. Raises on anything unexpected -- a silent fallback here would
    train the actor on a state that does not match the reference."""

    def __init__(self, host: str = "localhost", port: int = 8000, timeout_sec: float = 120.0) -> None:
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}/act"
        self.timeout = timeout_sec
        self.session = requests.Session()
        self.calls = 0
        self.seconds = 0.0

    def health(self) -> dict:
        response = self.session.get(self.url, timeout=10)
        response.raise_for_status()
        return response.json()

    def wait_until_ready(self, wait_sec: float = 1800.0) -> dict:
        deadline = time.time() + wait_sec
        last: Exception | None = None
        while time.time() < deadline:
            try:
                return self.health()
            except Exception as error:  # noqa: BLE001
                last = error
                time.sleep(5.0)
        raise RuntimeError(f"VLA server at {self.url} never came up: {last}")

    def act(self, external_cam, wrist_cam, instruction: str, state: np.ndarray) -> dict:
        payload = {
            "external_cam": np.asarray(external_cam, dtype=np.uint8),
            "wrist_cam": np.asarray(wrist_cam, dtype=np.uint8),
            "instruction": instruction,
            "state": np.asarray(state, dtype=np.float32),
        }
        t0 = time.perf_counter()
        response = self.session.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        out = response.json()
        self.calls += 1
        self.seconds += time.perf_counter() - t0

        if "error" in out:
            raise RuntimeError(f"VLA server error: {out['error']}")
        actions = np.asarray(out["actions"], dtype=np.float32)
        tokens = np.asarray(out["token_features"], dtype=np.float32)
        mask = np.asarray(out["token_attention_mask"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise RuntimeError(f"expected (chunk,{ACTION_DIM}) actions, got {actions.shape}")
        if tokens.ndim != 2 or tokens.shape[1] != VLA_TOKEN_DIM:
            raise RuntimeError(f"expected (S,{VLA_TOKEN_DIM}) token features, got {tokens.shape}")
        return {"actions": actions, "tokens": tokens, "mask": mask}


def token_z_dim(checkpoint: str) -> int:
    """The RL token width, without building the model or touching a GPU."""
    return int(torch.load(checkpoint, map_location="meta", weights_only=False)["config"]["z_dim"])


class TokenEncoder:
    """The frozen phase-1 encoder: VLA token sequence -> z_rl."""

    def __init__(self, checkpoint: str, device: str = "cuda:0") -> None:
        self.device = torch.device(device)
        model = RLTokenAE.load(checkpoint, map_location=self.device)
        self.encoder = model.encoder.to(self.device).eval().requires_grad_(False)
        self.z_dim = model.z_dim

    @torch.no_grad()
    def encode(self, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # `build()` may replace `self.encoder` with a shared AE module that was
        # constructed on CPU. Move it here rather than crashing on the first matmul.
        if next(self.encoder.parameters()).device != self.device:
            self.encoder.to(self.device)
        t = torch.as_tensor(tokens, dtype=torch.float32, device=self.device).unsqueeze(0)
        m = torch.as_tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.encoder(t, m).squeeze(0).cpu().numpy()
