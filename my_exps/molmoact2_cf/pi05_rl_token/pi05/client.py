"""Talking to the frozen pi0.5 server.

One server, one contract, used by both evaluation and RL training -- so a result can
never come from a different serving path than the run it is compared against.

    POST /act
        {"external_cam", "wrist_cam", "instruction", "state", "want_tokens"}
     -> {"actions": (VLA_CHUNK, 8), "converts_delta_to_absolute": false,
         "token_features": (968, 2048), "token_attention_mask": (968,)}   # if asked

The server is `scripts/serve_pi05_http.py`. It returns actions exactly as the model
produces them -- joint DELTAS. Nothing on the wire is ever an absolute joint target; the
conversion is `pi05/model.py`'s job and happens on this side, once. The
``converts_delta_to_absolute`` field exists so a client can refuse to run against a
server that converts, which would add the arm state twice and fly the arm.

Tokens are ~7.9 MB per call, so they are only requested when something needs them: RL
training and RL evaluation ask, a plain VLA benchmark does not.

This deliberately does not import `rlt`. The rlt client reads its token width from
$RLT_VLA_TOKEN_DIM, defaulting to MolmoAct2's 2560, and a plain evaluation that forgot to
export 2048 would fail in a confusing place.
"""

from __future__ import annotations

import fcntl
import os
import time
from pathlib import Path

import json_numpy
import numpy as np
import requests

from .config import ACTION_DIM, VLA_CHUNK, VLA_TOKEN_DIM

json_numpy.patch()


class Pi05Client:
    """HTTP client for the frozen pi0.5 server.

    Raises on anything unexpected. A silent fallback here would let a run proceed against
    a state that does not match the actions, which is invisible until the success rate
    comes back wrong hours later.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8080,
        timeout_sec: float = 600.0,
        *,
        require_deltas: bool = True,
    ) -> None:
        # 127.0.0.1 rather than "localhost": the name resolves to ::1 first and this
        # container has no IPv6, so connecting fails with EAFNOSUPPORT (errno 97).
        self.host = "127.0.0.1" if host == "localhost" else host
        self.port = port
        self.url = f"http://{self.host}:{port}/act"
        self.timeout = timeout_sec
        self.require_deltas = require_deltas
        self.session = requests.Session()
        self.calls = 0
        self.seconds = 0.0
        lock_dir = Path(os.environ.get("PI05_VLA_LOCK_DIR", "/tmp/pi05_vla_locks"))
        lock_dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = lock_dir / f"port_{self.port}.lock"

    # --- lifecycle ------------------------------------------------------------------

    def health(self) -> dict:
        response = self.session.get(self.url, timeout=10)
        response.raise_for_status()
        return response.json()

    def wait_until_ready(self, wait_sec: float = 1800.0) -> dict:
        """Block until the server answers. Loading the checkpoint takes minutes."""
        deadline = time.time() + wait_sec
        last: Exception | None = None
        while time.time() < deadline:
            try:
                info = self.health()
            except Exception as error:  # noqa: BLE001
                last = error
                time.sleep(5.0)
                continue
            self._check_contract(info)
            return info
        raise RuntimeError(f"pi0.5 server at {self.url} never came up: {last}")

    def _check_contract(self, info: dict) -> None:
        """Refuse a server that converts deltas itself; we would convert twice."""
        if not self.require_deltas:
            return
        converts = info.get("converts_delta_to_absolute")
        if converts is None:
            raise RuntimeError(
                f"the server at {self.url} does not report converts_delta_to_absolute, so "
                "it is an older build whose action space cannot be verified. Restart it "
                "from scripts/serve_pi05_http.py."
            )
        if converts:
            raise RuntimeError(
                f"the server at {self.url} converts deltas to absolute joint targets, but "
                "this client converts too, which would add the arm state twice and drive "
                "the arm into a folded pose. Start the server without --convert."
            )

    # --- inference ------------------------------------------------------------------

    def act(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        want_tokens: bool = False,
    ) -> dict:
        """One inference. ``state`` is the 8-vector [7 arm joints, gripper 0..1]."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"state must be ({ACTION_DIM},), got {state.shape}")

        payload = {
            "external_cam": np.asarray(external_cam, dtype=np.uint8),
            "wrist_cam": np.asarray(wrist_cam, dtype=np.uint8),
            "instruction": str(instruction),
            "state": state,
            "want_tokens": bool(want_tokens),
        }
        started = time.perf_counter()
        with open(self._lock_path, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            response = self.session.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        out = response.json()
        self.calls += 1
        self.seconds += time.perf_counter() - started

        if "error" in out:
            raise RuntimeError(f"pi0.5 server error: {out['error']}")

        actions = np.asarray(out["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise RuntimeError(f"expected (chunk, {ACTION_DIM}) actions, got {actions.shape}")
        if actions.shape[0] != VLA_CHUNK:
            raise RuntimeError(
                f"expected {VLA_CHUNK} actions per call, got {actions.shape[0]}; the "
                "checkpoint's action horizon is not what config.VLA_CHUNK says"
            )
        result = {"actions": actions}

        if want_tokens:
            tokens = np.asarray(out["token_features"], dtype=np.float32)
            if tokens.ndim != 2 or tokens.shape[1] != VLA_TOKEN_DIM:
                raise RuntimeError(
                    f"expected (S, {VLA_TOKEN_DIM}) token features, got {tokens.shape}; a "
                    "phase-1 encoder is built for one fixed width"
                )
            result["tokens"] = tokens
            result["mask"] = np.asarray(out["token_attention_mask"], dtype=np.float32)
        return result

    def reload(self, checkpoint: str) -> dict:
        """Ask the server to copy the learning expert's weights into the frozen expert."""
        url = f"http://{self.host}:{self.port}/reload"
        response = self.session.post(
            url, json={"checkpoint": str(checkpoint)}, timeout=self.timeout
        )
        response.raise_for_status()
        out = response.json()
        if "error" in out:
            raise RuntimeError(f"pi0.5 reload error: {out['error']}")
        return out

    @property
    def ms_per_call(self) -> float:
        return 1000.0 * self.seconds / max(self.calls, 1)
