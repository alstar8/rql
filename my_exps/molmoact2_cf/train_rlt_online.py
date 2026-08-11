"""Online RLT + ConsensusFlow training for MolmoAct2 on MolmoSpaces.

The frozen MolmoAct2 server supplies an action horizon and either raw VLM
tokens or an ``z_rl`` embedding.  The local RLT stack acts once per
non-overlapping eight-step chunk and trains from chunk-aligned replay after
each valid environment episode.
"""

from __future__ import annotations

import argparse
import fcntl
import gc
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import json_numpy
import numpy as np
import requests
import torch

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from chunk_replay import ChunkReplay, ImageChunkReplay, TokenReplay  # noqa: E402
from molmo_ae_backend import MolmoAEBackend  # noqa: E402
from rlt_models import (  # noqa: E402
    ACTION_DIM,
    CF_MODE_FLOW,
    CF_MODE_RESIDUAL,
    CHUNK_SIZE,
    MolmoAct2RLTCF,
    Z_DIM,
)
from train_100m import _bench_size  # noqa: E402
from train_full import _default_bench  # noqa: E402
from train_rlt import (  # noqa: E402
    action_sensitivity,
    ae_flow_actor_step,
    ae_flow_critic_td_step,
    ae_flow_gate_metrics,
    ae_flow_guide_step,
    actor_step,
    build_rlt_optimizers,
    critic_is_healthy,
    critic_td_step,
    flow_actor_step,
    flow_critic_td_step,
    flow_gate_metrics,
    flow_guide_step,
    guide_step,
    predicted_deploy_advantage,
    predicted_guide_advantage,
    predicted_lcb_advantage,
    token_step,
)
from molmo_spaces.evaluation.configs.evaluation_configs import (  # noqa: E402
    MolmoAct2PolicyEvalConfig,
)
from molmo_spaces.evaluation.eval_main import run_evaluation  # noqa: E402
import molmo_spaces.data_generation.pipeline as molmo_pipeline  # noqa: E402
from molmo_spaces.policy.learned_policy.molmoact2_policy import (  # noqa: E402
    MolmoAct2_Policy,
)
from molmo_spaces.renderer import opengl_context as egl_ctx  # noqa: E402
from molmo_spaces.renderer import opengl_rendering as ogl  # noqa: E402


def _patch_molmo_single_worker_mp() -> None:
    """Stop MolmoSpaces from starting a CUDA-unsafe forkserver on every episode.

    ``ParallelRolloutRunner`` always allocates ``mp_context.Value`` / ``Lock`` /
    ``Event`` even when ``num_workers=1``.  With CUDA available that context is
    ``forkserver``, which races with EGL/CUDA across 3 trainers/GPU and ends in
    SIGABRT.  Our trainers always use ``num_workers=1``, so thread-local stand-ins
    are sufficient and avoid the resource_tracker / forkserver entirely.
    """
    class _LocalValue:
        def __init__(self, _typecode: str, value: int) -> None:
            self.value = int(value)

    class _LocalMPContext:
        Lock = staticmethod(threading.Lock)
        Event = staticmethod(threading.Event)

        @staticmethod
        def Value(typecode: str, value: int) -> _LocalValue:
            return _LocalValue(typecode, value)

        @staticmethod
        def Process(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "molmoact2_cf expects num_workers=1; refusing to spawn forkserver workers"
            )

    molmo_pipeline.mp_context = _LocalMPContext()  # type: ignore[assignment]


_patch_molmo_single_worker_mp()

log = logging.getLogger("molmoact2_cf.train_rlt_online")

# Keep scratch / locks / rollouts under the workspace (no system /tmp).
_B1K_ROOT = Path("/workspace-SR008.nfs2/users/staroverov/B1K")
_B1K_TMP = Path(os.environ.get("B1K_TMP", str(_B1K_ROOT / "tmp")))
_DEFAULT_EGL_LOCK_DIR = str(_B1K_TMP / "rlt_egl_locks")
_DEFAULT_TMP_ROLLOUT_DIR = str(_B1K_TMP / "molmoact2_rlt_rollouts")
_IO_RETRY_ATTEMPTS = max(1, int(os.environ.get("RLT_IO_RETRY_ATTEMPTS", "5")))
_IO_RETRY_BASE_SEC = float(os.environ.get("RLT_IO_RETRY_BASE_SEC", "1.0"))


def _egl_lock_dir() -> Path:
    return Path(os.environ.get("RLT_EGL_LOCK_DIR", _DEFAULT_EGL_LOCK_DIR))


def _io_retry(
    label: str,
    fn: Any,
    *,
    attempts: int | None = None,
    base_sec: float | None = None,
) -> Any:
    """Retry NFS-ish I/O failures (ENOSPC / short writes) with exponential backoff."""
    n_attempts = _IO_RETRY_ATTEMPTS if attempts is None else max(1, int(attempts))
    delay = _IO_RETRY_BASE_SEC if base_sec is None else max(0.05, float(base_sec))
    last_error: BaseException | None = None
    for attempt in range(1, n_attempts + 1):
        try:
            return fn()
        except (OSError, RuntimeError) as error:
            last_error = error
            if attempt >= n_attempts:
                break
            log.warning(
                "%s failed (attempt %d/%d): %s; retrying in %.1fs",
                label,
                attempt,
                n_attempts,
                error,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, 30.0)
    assert last_error is not None
    raise last_error


def _cleanup_path(path: Path) -> None:
    try:
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def _patch_renderer_device_id() -> None:
    """Pin MjOpenGLRenderer to the physical ``MUJOCO_EGL_DEVICE_ID``.

    Without this, MolmoSpaces sets ``device_id=0`` from remapped CUDA and can
    attach every trainer's EGL context to the wrong GPU under contention.
    """
    original_init = ogl.MjOpenGLRenderer.__init__

    def _init(self: Any, *args: Any, device_id: int | None = None, **kwargs: Any) -> None:
        raw = os.environ.get("MUJOCO_EGL_DEVICE_ID", "").strip()
        if raw:
            device_id = int(raw.split(",")[0].strip())
        original_init(self, *args, device_id=device_id, **kwargs)

    ogl.MjOpenGLRenderer.__init__ = _init  # type: ignore[method-assign]


_patch_renderer_device_id()


def _patch_egl_context_serialize() -> None:
    """Serialize EGL display/context construction across all trainer processes.

    Per-GPU rollout locks are not enough during a multi-GPU start storm: many
    processes call ``eglInitialize`` / ``eglCreateContext`` at once and the
    NVIDIA EGL stack occasionally SIGABRTs with no Python traceback.  A short
    global flock around context creation keeps device-parallel rendering while
    making init single-flight.
    """
    original_init = egl_ctx.EGLGLContext.__init__
    lock_dir = _egl_lock_dir()
    lock_path = lock_dir / "egl_init.lock"

    def _init(self: Any, *args: Any, **kwargs: Any) -> None:
        lock_dir.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                original_init(self, *args, **kwargs)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    egl_ctx.EGLGLContext.__init__ = _init  # type: ignore[method-assign]


_patch_egl_context_serialize()

json_numpy.patch()

_STOP_REQUESTED = False


class RLTFeatureError(RuntimeError):
    """The expert server did not return an RLT-compatible representation."""


@dataclass(frozen=True)
class GateStatus:
    """Production gate diagnostics separated from deployment overrides."""

    would_enable: bool
    deploy_actor: bool
    deploy_guide: bool
    paired_lcb: float
    q_min_advantage: float
    guide_advantage: float
    critic_health: bool
    sensitivity: float


@dataclass(frozen=True)
class ResumeArtifacts:
    checkpoint: Path | None
    replay: Path | None
    ae_trainable: Path | None
    ae_replay: Path | None


def _parse_snapshot_episodes(raw: str) -> set[int]:
    episodes: set[int] = set()
    for token in str(raw or "").split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError("snapshot episodes must be non-negative")
        episodes.add(value)
    return episodes


def _request_stop(signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    log.warning("Received signal %d; stopping after the current episode", signum)


def _egl_device_id() -> str:
    """Physical GPU id used for MuJoCo EGL (ignores CUDA remapping)."""
    for key in ("MUJOCO_EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES"):
        raw = os.environ.get(key, "").strip()
        if raw:
            return raw.split(",")[0].strip()
    return "0"


@contextmanager
def _egl_concurrency_slot() -> Iterator[int]:
    """Limit how many MuJoCo EGL rollouts run at once across the machine."""
    lock_dir = _egl_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    max_concurrent = max(1, int(os.environ.get("RLT_EGL_MAX_CONCURRENT", "4")))
    handles: list[Any] = []
    slot = -1
    waited_rounds = 0
    try:
        while not _STOP_REQUESTED:
            for idx in range(max_concurrent):
                path = lock_dir / f"slot_{idx}.lock"
                handle = open(path, "a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                handle.seek(0)
                handle.truncate()
                handle.write(f"pid={os.getpid()} slot={idx}\n")
                handle.flush()
                handles.append(handle)
                slot = idx
                log.info(
                    "Acquired EGL concurrency slot %d/%d",
                    idx + 1,
                    max_concurrent,
                )
                yield idx
                return
            waited_rounds += 1
            # Heartbeat so trainer_watchdog does not treat slot waits as hung.
            if waited_rounds == 1 or waited_rounds % 20 == 0:
                log.info(
                    "Waiting for EGL concurrency slot (%d/%d busy, waited ~%.0fs)",
                    max_concurrent,
                    max_concurrent,
                    waited_rounds * 0.5,
                )
            time.sleep(0.5)
        raise RuntimeError("Stop requested while waiting for an EGL concurrency slot")
    finally:
        for handle in handles:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        if slot >= 0:
            log.info("Released EGL concurrency slot %d", slot)


@contextmanager
def _egl_gpu_lock() -> Iterator[None]:
    """Bound concurrent classic/EGL MuJoCo rollouts per physical GPU.

    Concurrent MjOpenGLRenderer contexts on one device can SIGABRT, so we keep
    a small per-GPU slot pool (``RLT_EGL_PER_GPU``, default 3) rather than a
    single exclusive lock.  With 4 trainers/GPU the old exclusive lock left
    75% of workers idle for tens of minutes.

    Only the MuJoCo rollout should hold this lock; CUDA critic/actor updates
    run *after* release so training does not extend the EGL queue.  A short
    cooldown after unlock lets GL teardown finish.  Take a per-GPU slot
    *before* the global concurrency slot so one busy GPU cannot consume every
    machine-wide slot while blocked.
    """
    device = _egl_device_id()
    lock_dir = _egl_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    per_gpu = max(1, int(os.environ.get("RLT_EGL_PER_GPU", "3")))
    cooldown = float(os.environ.get("RLT_EGL_COOLDOWN_SEC", "0.5"))
    handles: list[Any] = []
    slot = -1
    waited_rounds = 0
    try:
        log.info(
            "Waiting for EGL lock on GPU %s (%d slots under %s)",
            device,
            per_gpu,
            lock_dir,
        )
        while not _STOP_REQUESTED:
            for idx in range(per_gpu):
                path = lock_dir / f"gpu_{device}_s{idx}.lock"
                handle = open(path, "a+", encoding="utf-8")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                handle.seek(0)
                handle.truncate()
                handle.write(f"pid={os.getpid()} device={device} slot={idx}\n")
                handle.flush()
                handles.append(handle)
                slot = idx
                log.info(
                    "Acquired EGL lock on GPU %s slot %d/%d",
                    device,
                    idx + 1,
                    per_gpu,
                )
                with _egl_concurrency_slot():
                    yield
                return
            waited_rounds += 1
            if waited_rounds == 1 or waited_rounds % 20 == 0:
                log.info(
                    "Still waiting for EGL lock on GPU %s (waited ~%.0fs, per_gpu=%d)",
                    device,
                    waited_rounds * 0.5,
                    per_gpu,
                )
            time.sleep(0.5)
        raise RuntimeError(
            f"Stop requested while waiting for EGL lock on GPU {device}"
        )
    finally:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if cooldown > 0.0:
                time.sleep(cooldown)
        finally:
            for handle in handles:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                finally:
                    handle.close()
            if slot >= 0:
                log.info("Released EGL lock on GPU %s slot %d", device, slot)

class RLTOnlinePolicy(MolmoAct2_Policy):
    """MolmoSpaces policy that applies one local RLT decision per action chunk."""

    def __init__(
        self,
        exp_config: Any,
        model: MolmoAct2RLTCF,
        device: torch.device,
        *,
        use_cf_guide: bool,
        actor_mode: str,
        prefer_server_z: bool = True,
        retain_tokens: bool = False,
        request_timeout_sec: float = 120.0,
        explore_residual_std: float = 0.02,
        explore_deploy_std: float | None = None,
        explore_warmup_mult: float = 1.0,
        ae_backend: Any | None = None,
    ) -> None:
        self.rlt_model = model
        self.rlt_device = device
        self.use_cf_guide = bool(use_cf_guide)
        self.actor_mode = str(actor_mode)
        self.prefer_server_z = bool(prefer_server_z)
        self.retain_tokens = bool(retain_tokens)
        self.request_timeout_sec = float(request_timeout_sec)
        self.explore_residual_std = float(explore_residual_std)
        self.explore_deploy_std = float(
            explore_residual_std if explore_deploy_std is None else explore_deploy_std
        )
        self.explore_warmup_mult = float(explore_warmup_mult)
        self.ae_backend = ae_backend
        self.deploy_actor = False
        self.deploy_guide = False
        self.fatal_error: RLTFeatureError | None = None
        self._rng = np.random.default_rng(0)
        self._clear_episode()
        super().__init__(exp_config)
        self.chunk_size = int(model.chunk_size)
        if self.chunk_size != CHUNK_SIZE:
            raise ValueError(
                f"RLT policy requires chunk_size={CHUNK_SIZE}, checkpoint has {self.chunk_size}"
            )

    @property
    def enable_rlt(self) -> bool:
        """Deprecated compatibility view of deploy_actor."""
        return self.deploy_actor

    @enable_rlt.setter
    def enable_rlt(self, enabled: bool) -> None:
        self.deploy_actor = bool(enabled)
        self.deploy_guide = bool(enabled) and self.use_cf_guide

    def _clear_episode(self) -> None:
        self.ep_zs: list[np.ndarray] = []
        self.ep_proprios: list[np.ndarray] = []
        self.ep_references: list[np.ndarray] = []
        self.ep_executed: list[np.ndarray] = []
        self.ep_action_counts: list[int] = []
        self.ep_tokens: list[np.ndarray | None] = []
        self.ep_token_masks: list[np.ndarray | None] = []
        self.ep_z_sources: list[str] = []
        self.ep_external_cams: list[np.ndarray] = []
        self.ep_wrist_cams: list[np.ndarray] = []
        self.ep_instructions: list[str] = []

    def reset(self) -> None:
        super().reset()
        self.fatal_error = None
        self._clear_episode()

    def prepare_model(self, model_name: str | None = None) -> None:
        if self.ae_backend is not None:
            log.info("In-process Molmo AE backend; skipping HTTP health check")
            return
        super().prepare_model(model_name)

    def _post_act(self, model_input: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "external_cam": np.asarray(model_input["external_cam"], dtype=np.uint8),
            "wrist_cam": np.asarray(model_input["wrist_cam"], dtype=np.uint8),
            "instruction": model_input["instruction"],
            "state": np.asarray(model_input["state"], dtype=np.float32),
            "timestamp": model_input.get("timestamp", time.time()),
        }
        if self.ae_backend is not None:
            # Replay/reference actions must remain the frozen base-AE output.
            out = self.ae_backend.predict_reference(
                payload["external_cam"],
                payload["wrist_cam"],
                str(payload["instruction"]),
                payload["state"],
            )
            body = {"actions": out["actions"]}
            for key in (
                "features",
                "token_features",
                "token_attention_mask",
                "z_rl",
            ):
                if key in out:
                    body[key] = out[key]
            return body
        response = self.session.post(self.url, json=payload, timeout=self.request_timeout_sec)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(f"MolmoAct2 /act returned {type(body).__name__}, expected object")
        return body

    def _response_tokens(
        self,
        response: dict[str, Any],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if "token_features" not in response:
            return None, None
        tokens = np.asarray(response["token_features"], dtype=np.float32)
        if tokens.ndim != 2 or tokens.shape[1] != self.rlt_model.feature_dim:
            raise RLTFeatureError(
                "Invalid token_features shape "
                f"{tokens.shape}; expected (S, {self.rlt_model.feature_dim})"
            )
        max_tokens = self.rlt_model.token_ae.encoder.max_len - 1
        tokens = tokens[:max_tokens]
        raw_mask = response.get("token_attention_mask")
        if raw_mask is None:
            mask = np.ones(tokens.shape[0], dtype=np.float32)
        else:
            mask = np.asarray(raw_mask, dtype=np.float32).reshape(-1)[: tokens.shape[0]]
            if mask.shape != (tokens.shape[0],):
                raise RLTFeatureError(
                    f"Invalid token_attention_mask shape {mask.shape} for {tokens.shape[0]} tokens"
                )
        if not np.isfinite(tokens).all() or not np.isfinite(mask).all():
            raise RLTFeatureError("MolmoAct2 server returned non-finite token features")
        if tokens.shape[0] == 0 or float(mask.sum()) <= 0.0:
            raise RLTFeatureError("MolmoAct2 server returned no valid token features")
        if float(np.linalg.norm(tokens.astype(np.float32))) <= 1e-8:
            raise RLTFeatureError("MolmoAct2 server returned an all-zero token tensor")
        return tokens, mask

    def _response_z(
        self,
        response: dict[str, Any],
        tokens: np.ndarray | None,
        token_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, str]:
        if self.prefer_server_z and "z_rl" in response:
            z = np.asarray(response["z_rl"], dtype=np.float32).reshape(-1)
            source = "z_rl"
        elif tokens is not None and token_mask is not None:
            tok = torch.from_numpy(tokens).unsqueeze(0).to(self.rlt_device)
            mask = torch.from_numpy(token_mask).unsqueeze(0).to(self.rlt_device)
            with torch.inference_mode():
                normalized = self.rlt_model.normalize_features(tok)
                z_tensor = self.rlt_model.token_ae.encoder(normalized, mask)
            z = z_tensor[0].float().cpu().numpy()
            source = "token_features"
        elif "features" in response:
            raise RLTFeatureError(
                "MolmoAct2 server returned only mean-pooled `features`, but the RLT model "
                "has no mean-pool projection. Restart the server with "
                "`--feature_mode tokens` or `--feature_mode rl_token`."
            )
        else:
            raise RLTFeatureError(
                "MolmoAct2 server returned neither `z_rl` nor `token_features`. "
                "Use `--feature_mode tokens` or `--feature_mode rl_token`."
            )
        if z.shape != (Z_DIM,) or not np.isfinite(z).all():
            raise RLTFeatureError(
                f"Invalid RLT state shape/values: shape={z.shape}, expected ({Z_DIM},)"
            )
        if float(np.linalg.norm(z)) <= 1e-8:
            raise RLTFeatureError("RLT state is all zero; refusing a mock/fallback feature")
        return z.astype(np.float32, copy=False), source

    def _start_chunk(self, model_input: dict[str, Any]) -> None:
        response = self._post_act(model_input)
        actions = np.asarray(response.get("actions"), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"Invalid MolmoAct2 action horizon shape {actions.shape}")
        if actions.shape[0] < self.chunk_size:
            raise ValueError(
                f"MolmoAct2 returned only {actions.shape[0]} actions; "
                f"RLT requires at least {self.chunk_size}"
            )
        if not np.isfinite(actions).all():
            raise FloatingPointError("MolmoAct2 returned non-finite actions")

        reference = actions[: self.chunk_size].copy()
        proprio = np.asarray(model_input["state"], dtype=np.float32).reshape(-1)
        if proprio.shape != (self.rlt_model.proprio_dim,):
            raise ValueError(
                f"Invalid proprio shape {proprio.shape}; "
                f"expected ({self.rlt_model.proprio_dim},)"
            )
        tokens, token_mask = self._response_tokens(response)
        z, z_source = self._response_z(response, tokens, token_mask)

        deployed = reference.copy()
        use_molmo_ae = (
            self.ae_backend is not None
            and getattr(self.rlt_model, "v_source", "rlt") == "molmo_ae"
        )
        if self.deploy_actor and self.actor_mode == "rlt":
            z_t = torch.from_numpy(z).unsqueeze(0).to(self.rlt_device)
            proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.rlt_device)
            with torch.inference_mode():
                state = self.rlt_model.encode_state_from_z(z_t, proprio_t)
                if use_molmo_ae:
                    # Gate on: trainable AE, with guide only when requested.
                    predicted = self.ae_backend.predict(
                        np.asarray(model_input["external_cam"], dtype=np.uint8),
                        np.asarray(model_input["wrist_cam"], dtype=np.uint8),
                        str(model_input["instruction"]),
                        proprio,
                        apply_guide=self.deploy_guide,
                        rlt_state=state if self.deploy_guide else None,
                    )
                    deployed = np.asarray(predicted["actions"], dtype=np.float32)[
                        : self.chunk_size
                    ].copy()
                else:
                    reference_t = torch.from_numpy(reference).unsqueeze(0).to(
                        self.rlt_device
                    )
                    reference_n = self.rlt_model.normalize_action(reference_t)
                    deployed_n, _ = self.rlt_model.actor_chunk(
                        state,
                        reference_n,
                        deterministic=True,
                        apply_guide=self.deploy_guide,
                    )
                    deployed = self.rlt_model.denormalize_action(deployed_n)[
                        0
                    ].cpu().numpy()
            if deployed.shape != reference.shape or not np.isfinite(deployed).all():
                raise FloatingPointError(
                    f"RLT/AE actor returned invalid chunk shape={deployed.shape}"
                )
        # Exploration in *normalized* action space. v12: smaller default and no
        # gate-off boost (v11's 0.05×1.5 drove residual_rms~0.07–0.15 and SR tax).
        explore_std = float(self.explore_residual_std)
        if self.deploy_actor:
            explore_std = float(self.explore_deploy_std)
        elif self.explore_warmup_mult != 1.0:
            explore_std = explore_std * float(self.explore_warmup_mult)
        if (
            explore_std > 0.0
            and self.actor_mode == "rlt"
            and self.rlt_model is not None
        ):
            a_t = torch.from_numpy(np.asarray(deployed, dtype=np.float32)).unsqueeze(0)
            a_t = a_t.to(self.rlt_device)
            with torch.inference_mode():
                a_n = self.rlt_model.normalize_action(a_t)
                noise = torch.randn_like(a_n) * explore_std
                deployed = (
                    self.rlt_model.denormalize_action(a_n + noise)[0]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

        self.actions_buffer = [row.copy() for row in deployed]
        self.current_buffer_index = 0
        self.ep_zs.append(z.copy())
        self.ep_proprios.append(proprio.copy())
        self.ep_references.append(reference.copy())
        self.ep_executed.append(np.asarray(deployed, dtype=np.float32).copy())
        self.ep_action_counts.append(0)
        self.ep_tokens.append(
            tokens.astype(np.float16)
            if self.retain_tokens and tokens is not None
            else None
        )
        self.ep_token_masks.append(
            token_mask.astype(np.uint8)
            if self.retain_tokens and token_mask is not None
            else None
        )
        self.ep_z_sources.append(z_source)
        self.ep_external_cams.append(
            np.asarray(model_input["external_cam"], dtype=np.uint8).copy()
        )
        self.ep_wrist_cams.append(
            np.asarray(model_input["wrist_cam"], dtype=np.uint8).copy()
        )
        self.ep_instructions.append(str(model_input.get("instruction", "")))

    def inference_model(self, model_input: dict[str, Any]) -> np.ndarray:
        if self.actions_buffer is None or self.current_buffer_index >= min(
            self.chunk_size, len(self.actions_buffer or [])
        ):
            try:
                self._start_chunk(model_input)
            except RLTFeatureError as error:
                self.fatal_error = error
                raise
            except Exception as error:
                if self.deploy_guide:
                    fatal = RLTFeatureError(
                        f"Guided policy failed without a valid fallback: {error}"
                    )
                    self.fatal_error = fatal
                    raise fatal from error
                raise
        if self.actions_buffer is None:
            raise RuntimeError("RLT action buffer was not initialized")
        output = np.asarray(
            self.actions_buffer[self.current_buffer_index], dtype=np.float32
        ).copy()
        self.current_buffer_index += 1
        self.ep_action_counts[-1] += 1
        return output

    def _task_rewards(self, n_steps: int, success: bool) -> np.ndarray:
        if n_steps == 0:
            return np.zeros(0, dtype=np.float32)
        reward_cache = getattr(self.task, "reward_cache", None)
        if reward_cache is None:
            error = RLTFeatureError("Active task has no reward_cache")
            self.fatal_error = error
            raise error
        cached = list(reward_cache)
        # reward_cache[0] belongs to task.reset; subsequent entries match actions.
        if len(cached) < n_steps + 1:
            error = RLTFeatureError(
                f"Reward cache is short: have {max(len(cached) - 1, 0)} "
                f"action rewards for {n_steps} steps"
            )
            self.fatal_error = error
            raise error
        values: list[float] = []
        for step, reward in enumerate(cached[1 : n_steps + 1]):
            array = np.asarray(reward, dtype=np.float32).reshape(-1)
            if array.size == 0 or not np.isfinite(array[0]):
                error = RLTFeatureError(
                    f"Reward cache entry {step} is empty or non-finite"
                )
                self.fatal_error = error
                raise error
            values.append(float(array[0]))
        result = np.asarray(values, dtype=np.float32)
        if result.shape != (n_steps,) or not np.isfinite(result).all():
            error = RLTFeatureError("Reward cache produced invalid active-path rewards")
            self.fatal_error = error
            raise error
        if success and float(result.max(initial=0.0)) <= 0.0:
            error = RLTFeatureError(
                "Successful rollout has no positive environment reward"
            )
            self.fatal_error = error
            raise error
        return result

    def pop_episode(self, success: bool) -> dict[str, Any]:
        n_steps = int(sum(self.ep_action_counts))
        step_rewards = self._task_rewards(n_steps, success)
        rewards: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        offset = 0
        residual_rows: list[np.ndarray] = []
        for reference, executed, count in zip(
            self.ep_references, self.ep_executed, self.ep_action_counts
        ):
            mask = np.zeros(self.chunk_size, dtype=np.float32)
            mask[:count] = 1.0
            chunk_rewards = np.zeros(self.chunk_size, dtype=np.float32)
            chunk_rewards[:count] = step_rewards[offset : offset + count]
            offset += count
            rewards.append(chunk_rewards)
            masks.append(mask)
            if count:
                residual_rows.append(executed[:count] - reference[:count])

        token_batches = [
            (tokens, mask)
            for tokens, mask in zip(self.ep_tokens, self.ep_token_masks)
            if tokens is not None and mask is not None
        ]
        residual_rms = (
            float(np.sqrt(np.mean(np.square(np.concatenate(residual_rows, axis=0)))))
            if residual_rows
            else 0.0
        )
        trajectory = {
            "zs": self.ep_zs,
            "proprios": self.ep_proprios,
            "references": self.ep_references,
            "executed": self.ep_executed,
            "rewards": rewards,
            "masks": masks,
            "token_batches": token_batches,
            "z_sources": list(self.ep_z_sources),
            "n_steps": n_steps,
            "residual_rms": residual_rms,
            "external_cams": list(self.ep_external_cams),
            "wrist_cams": list(self.ep_wrist_cams),
            "instructions": list(self.ep_instructions),
        }
        self._clear_episode()
        return trajectory


def _server_health(host: str, port: int, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        response = requests.get(f"http://{host}:{port}/act", timeout=timeout)
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, dict) and body.get("status") == "ok" else None
    except Exception:  # noqa: BLE001
        return None


def _wait_for_server(host: str, port: int, max_wait_sec: float) -> dict[str, Any]:
    start = time.time()
    while time.time() - start < max_wait_sec and not _STOP_REQUESTED:
        health = _server_health(host, port)
        if health is not None:
            return health
        time.sleep(5.0)
    raise RuntimeError(f"MolmoAct2 server {host}:{port} was not ready within {max_wait_sec}s")


def _validate_server_features(health: dict[str, Any]) -> None:
    feature_mode = health.get("feature_mode")
    if feature_mode == "mean_pool":
        raise RLTFeatureError(
            "The MolmoAct2 server is in mean_pool mode. RLT requires "
            "`--feature_mode tokens` or `--feature_mode rl_token`."
        )
    if health.get("return_features") is False:
        raise RLTFeatureError("The MolmoAct2 server was started with --no_features")


def _resolve_resume_artifacts(args: argparse.Namespace) -> ResumeArtifacts:
    """Resolve an all-or-none online training resume bundle."""
    ae_mode = bool(getattr(args, "ae_trainable", False))
    eval_only = bool(getattr(args, "eval_only", False))
    explicit_ae_raw = str(getattr(args, "ae_trainable_ckpt", "") or "")
    explicit_ae = Path(explicit_ae_raw) if explicit_ae_raw else None
    if explicit_ae is not None and not explicit_ae.is_file():
        raise FileNotFoundError(f"AE trainable checkpoint not found: {explicit_ae}")
    if args.no_resume:
        return ResumeArtifacts(None, None, explicit_ae if ae_mode else None, None)
    latest = Path(args.out_dir) / "rlt_cf_latest.pt"
    replay = Path(args.replay_out) if str(getattr(args, "replay_out", "")) else None
    ae_path = Path(args.out_dir) / "molmo_ae_lora_latest.pt"
    ae_replay_raw = str(getattr(args, "ae_image_replay_out", "") or "")
    ae_replay_path = (
        Path(ae_replay_raw)
        if ae_replay_raw
        else Path(args.out_dir) / "ae_image_replay.npz"
    )
    metrics_path = Path(args.out_dir) / "metrics.jsonl"
    if not latest.is_file():
        if (
            ae_mode
            and eval_only
            and explicit_ae is None
            and str(getattr(args, "rlt_ckpt", ""))
        ):
            inferred_ae = Path(args.rlt_ckpt).parent / "molmo_ae_lora_latest.pt"
            if inferred_ae.is_file():
                explicit_ae = inferred_ae
        if ae_mode and ae_path.is_file():
            raise RuntimeError(
                f"Orphan AE resume artifact without RLT checkpoint: {ae_path}"
            )
        if not eval_only and metrics_path.is_file():
            raise RuntimeError(
                f"Orphan training metrics without RLT checkpoint: {metrics_path}"
            )
        return ResumeArtifacts(None, None, explicit_ae if ae_mode else None, None)
    if not eval_only and (replay is None or not replay.is_file()):
        raise RuntimeError(
            "Refusing partial resume: rlt_cf_latest.pt exists but chunk replay "
            f"is missing ({replay})"
        )
    if ae_mode and explicit_ae is None and not ae_path.is_file():
        raise RuntimeError(
            "Refusing partial AE resume: rlt_cf_latest.pt exists but "
            f"{ae_path.name} is missing"
        )
    if ae_mode and not eval_only and not ae_replay_path.is_file():
        raise RuntimeError(
            "Refusing partial AE resume: rlt_cf_latest.pt exists but "
            f"{ae_replay_path.name} is missing"
        )
    return ResumeArtifacts(
        latest,
        replay if replay is not None and replay.is_file() else None,
        (explicit_ae or ae_path) if ae_mode else None,
        ae_replay_path if ae_mode and not eval_only else None,
    )


def _resolve_resume_checkpoint(args: argparse.Namespace) -> Path | None:
    """Compatibility helper returning the checkpoint from the safe bundle."""
    return _resolve_resume_artifacts(args).checkpoint


def _load_metrics_resume(out_dir: Path) -> dict[str, Any] | None:
    metrics_path = out_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        return None
    last: dict[str, Any] | None = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _load_model(
    args: argparse.Namespace,
    device: torch.device,
    *,
    resume_checkpoint: Path | None = None,
) -> MolmoAct2RLTCF:
    cf_mode = str(getattr(args, "cf_mode", CF_MODE_RESIDUAL)).lower()
    resume_ckpt = (
        _resolve_resume_checkpoint(args)
        if resume_checkpoint is None
        else resume_checkpoint
    )
    if resume_ckpt is not None:
        model = MolmoAct2RLTCF.load(str(resume_ckpt), map_location=device).to(device)
        log.info("Resumed RLT checkpoint %s (cf_mode=%s)", resume_ckpt, model.cf_mode)
    elif args.rlt_ckpt:
        checkpoint = Path(args.rlt_ckpt)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RLT checkpoint not found: {checkpoint}")
        # Residual ckpt → flow: rebuild time-dependent heads, keep token AE.
        if cf_mode == CF_MODE_FLOW:
            peek = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
            src_mode = str(peek.get("cf_mode", CF_MODE_RESIDUAL))
            if src_mode != CF_MODE_FLOW:
                model = MolmoAct2RLTCF.from_token_ckpt_as_flow(
                    str(checkpoint),
                    map_location=device,
                    use_cf_guide=args.use_cf_guide,
                    n_critics=args.n_critics,
                    flow_steps=int(getattr(args, "flow_steps", 10)),
                    guidance_coef=float(getattr(args, "guidance_coef", 0.5)),
                ).to(device)
                log.info(
                    "Upgraded residual/token ckpt %s → flow CF (fresh time critic/actor/guide)",
                    checkpoint,
                )
            else:
                model = MolmoAct2RLTCF.load(str(checkpoint), map_location=device).to(device)
                log.info("Loaded flow CF checkpoint %s", checkpoint)
        else:
            model = MolmoAct2RLTCF.load(str(checkpoint), map_location=device).to(device)
            log.info("Loaded RLT checkpoint %s", checkpoint)
    else:
        model = MolmoAct2RLTCF(
            use_cf_guide=args.use_cf_guide,
            tune_token_online=args.tune_token_online,
            n_critics=args.n_critics,
            cf_mode=cf_mode,
            flow_steps=int(getattr(args, "flow_steps", 10)),
            guidance_coef=float(getattr(args, "guidance_coef", 0.5)),
        ).to(device)
        log.info("Initialized a fresh RLT model cf_mode=%s n_critics=%d", cf_mode, args.n_critics)
    if cf_mode == CF_MODE_FLOW and not model.is_flow:
        raise ValueError(f"--cf_mode=flow but loaded model is {model.cf_mode}")
    if cf_mode == CF_MODE_RESIDUAL and model.is_flow:
        raise ValueError("--cf_mode=residual but loaded model is flow")
    if int(getattr(model, "n_critics", 0) or 0) != int(args.n_critics):
        raise ValueError(
            f"Checkpoint n_critics={getattr(model, 'n_critics', '?')} "
            f"does not match --n_critics={args.n_critics}"
        )
    if model.chunk_size != CHUNK_SIZE or model.action_dim != ACTION_DIM:
        raise ValueError(
            f"Incompatible checkpoint chunk/action shape "
            f"({model.chunk_size}, {model.action_dim}); expected ({CHUNK_SIZE}, {ACTION_DIM})"
        )
    if not model.bounded_critic:
        raise ValueError("Online RLT training requires a bounded critic checkpoint")
    if args.use_cf_guide and model.guide is None:
        raise ValueError(
            "--use_cf_guide was requested, but the checkpoint has no guide module"
        )
    if not args.use_cf_guide:
        model.guide = None
        model.use_cf_guide = False
    if args.tune_token_online:
        model.unfreeze_token_encoder()
    else:
        model.freeze_token_encoder()
    return model


def _build_eval_policy(
    args: argparse.Namespace,
    model: MolmoAct2RLTCF,
    device: torch.device,
    *,
    prefer_server_z: bool = True,
    retain_tokens: bool | None = None,
    ae_backend: Any | None = None,
) -> tuple[RLTOnlinePolicy, MolmoAct2PolicyEvalConfig]:
    eval_only = bool(getattr(args, "eval_only", False))
    exp_config = MolmoAct2PolicyEvalConfig()
    exp_config.policy_config.remote_config = {
        "host": args.server_host,
        "port": int(args.server_port),
    }
    if hasattr(exp_config, "filter_for_successful_trajectories"):
        exp_config.filter_for_successful_trajectories = False
    if hasattr(exp_config, "datagen_profiler"):
        exp_config.datagen_profiler = False
    policy = RLTOnlinePolicy(
        exp_config,
        model,
        device,
        use_cf_guide=args.use_cf_guide,
        actor_mode=args.actor_mode,
        prefer_server_z=prefer_server_z,
        retain_tokens=(
            bool(args.tune_token_online) and not eval_only
            if retain_tokens is None
            else bool(retain_tokens) and not eval_only
        ),
        request_timeout_sec=args.server_request_timeout_sec,
        explore_residual_std=(
            0.0
            if eval_only
            else float(getattr(args, "explore_residual_std", 0.02))
        ),
        explore_deploy_std=float(
            0.0
            if eval_only
            else getattr(
                args,
                "explore_deploy_std",
                getattr(args, "explore_residual_std", 0.02),
            )
        ),
        explore_warmup_mult=(
            1.0
            if eval_only
            else float(getattr(args, "explore_warmup_mult", 1.0))
        ),
        ae_backend=ae_backend,
    )
    policy._rng = np.random.default_rng(int(getattr(args, "seed", 0)))
    return policy, exp_config


def _gate_status(
    args: argparse.Namespace,
    model: MolmoAct2RLTCF,
    replay: ChunkReplay,
    valid_episodes: int,
    device: torch.device,
    *,
    ae_backend: Any | None = None,
    image_replay: ImageChunkReplay | None = None,
    checkpoint_meta: dict[str, Any] | None = None,
) -> GateStatus:
    paired_lcb = 0.0
    q_min_advantage = 0.0
    guide_advantage = 0.0
    sensitivity = 0.0
    healthy = False
    ae_mode = ae_backend is not None
    gate_replay: Any = image_replay if ae_mode else replay
    enough_data = (
        args.actor_mode == "rlt"
        and valid_episodes >= args.g_start_episodes
        and gate_replay is not None
        and len(gate_replay) >= args.min_replay_chunks
        and gate_replay.has_both_outcomes()
    )
    if enough_data:
        batch_size = (
            int(getattr(args, "ae_batch_size", 2))
            if ae_mode
            else int(args.batch_size)
        )
        batch = gate_replay.sample(batch_size, device=device)
        healthy = critic_is_healthy(model, batch)
        if healthy and ae_mode:
            ae_metrics = ae_flow_gate_metrics(
                model,
                ae_backend,
                batch,
                sensitivity_noise=float(args.gate_sensitivity_noise),
            )
            paired_lcb = ae_metrics["paired_lcb"]
            q_min_advantage = ae_metrics["q_min_advantage"]
            guide_advantage = ae_metrics["guide_advantage"]
            sensitivity = ae_metrics["sensitivity"]
        elif healthy and model.is_flow:
            flow_metrics = flow_gate_metrics(
                model,
                batch,
                sensitivity_noise=float(args.gate_sensitivity_noise),
            )
            paired_lcb = flow_metrics["paired_lcb"]
            q_min_advantage = flow_metrics["q_min_advantage"]
            guide_advantage = flow_metrics["guide_advantage"]
            sensitivity = flow_metrics["sensitivity"]
        elif healthy:
            paired_lcb = predicted_lcb_advantage(model, batch)
            q_min_advantage = predicted_deploy_advantage(model, batch)
            guide_advantage = predicted_guide_advantage(model, batch)
            sensitivity = action_sensitivity(
                model,
                batch,
                noise=float(args.gate_sensitivity_noise),
            )

    would_enable = (
        enough_data
        and healthy
        and np.isfinite(paired_lcb)
        and paired_lcb >= float(args.g_min_advantage)
        and np.isfinite(sensitivity)
        and sensitivity >= float(args.g_min_action_sensitivity)
    )
    min_guide_adv = float(getattr(args, "g_min_guide_advantage", 0.0) or 0.0)
    if (
        would_enable
        and args.use_cf_guide
        and model.guide is not None
        and min_guide_adv > 0.0
    ):
        would_enable = (
            np.isfinite(guide_advantage)
            and guide_advantage >= min_guide_adv
        )

    deploy_policy = str(getattr(args, "deploy_policy", "gated"))
    deploy_actor = False
    deploy_guide = False
    if deploy_policy == "gated":
        deploy_actor = bool(would_enable)
        deploy_guide = deploy_actor and args.use_cf_guide and model.guide is not None
    elif deploy_policy == "checkpoint_gate":
        source = checkpoint_meta or {}
        deploy_actor = bool(
            source.get(
                "gate_deploy_actor",
                source.get("deploy_actor", source.get("g_enabled", False)),
            )
        )
        deploy_guide = bool(
            source.get(
                "gate_deploy_guide",
                deploy_actor and args.use_cf_guide,
            )
        )
    elif deploy_policy == "reference":
        deploy_actor = False
        deploy_guide = False
    elif deploy_policy == "actor":
        deploy_actor = True
        deploy_guide = False
    elif deploy_policy == "actor_guide":
        deploy_actor = True
        deploy_guide = True
    else:
        raise ValueError(f"Unknown deploy policy: {deploy_policy}")

    if args.actor_mode != "rlt":
        deploy_actor = False
        deploy_guide = False
    if deploy_guide and (not args.use_cf_guide or model.guide is None):
        raise RuntimeError(
            f"deploy_policy={deploy_policy} requested a guide, but no guide is loaded"
        )
    if deploy_guide and not deploy_actor:
        raise RuntimeError("Guide deployment requires actor deployment")
    return GateStatus(
        would_enable=bool(would_enable),
        deploy_actor=bool(deploy_actor),
        deploy_guide=bool(deploy_guide),
        paired_lcb=float(paired_lcb),
        q_min_advantage=float(q_min_advantage),
        guide_advantage=float(guide_advantage),
        critic_health=bool(healthy),
        sensitivity=float(sensitivity),
    )


def _train_after_episode(
    args: argparse.Namespace,
    model: MolmoAct2RLTCF,
    optimizers: dict[str, torch.optim.Optimizer],
    replay: ChunkReplay,
    token_replay: TokenReplay,
    device: torch.device,
    *,
    ae_backend: Any | None = None,
    image_replay: ImageChunkReplay | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    q_info: dict[str, float] = {}
    actor_info: dict[str, float] = {}
    guide_info: dict[str, float] = {}
    token_info: dict[str, float] = {}
    if len(replay) < args.min_replay_chunks:
        return q_info, actor_info, guide_info, token_info

    ae_mode = bool(getattr(args, "ae_trainable", False)) and ae_backend is not None
    ae_batch_size = int(getattr(args, "ae_batch_size", 2))
    if ae_mode and (
        image_replay is None
        or len(image_replay) < max(1, ae_batch_size)
    ):
        return q_info, actor_info, guide_info, token_info

    for _ in range(args.updates_per_episode):
        # RLT uses two critic updates for every actor update.
        for _critic_update in range(2):
            critic_kwargs = {
                "gamma": args.gamma,
                "mc_coef": args.mc_coef,
                "cql_coef": args.cql_coef,
                "cql_n_actions": args.cql_n_actions,
                "cql_action_radius": args.cql_action_radius,
                "ref_dropout": args.ref_dropout,
                "rank_coef": args.rank_coef,
                "rank_margin": args.rank_margin,
                "rank_noise": args.rank_noise,
                "far_rank_coef": args.far_rank_coef,
                "far_rank_noise": args.far_rank_noise,
                "shuffle_rank_coef": args.shuffle_rank_coef,
                "target_noise": args.target_noise,
            }
            if ae_mode:
                batch = image_replay.sample(ae_batch_size, device=device)
                q_info = ae_flow_critic_td_step(
                    model,
                    ae_backend,
                    optimizers["critic"],
                    batch,
                    **critic_kwargs,
                )
            else:
                batch = replay.sample(args.batch_size, device=device)
                critic_fn = flow_critic_td_step if model.is_flow else critic_td_step
                q_info = critic_fn(
                    model,
                    optimizers["critic"],
                    batch,
                    **critic_kwargs,
                )
        if args.actor_mode == "rlt":
            if (
                ae_mode
                and image_replay is not None
                and len(image_replay) >= max(1, ae_batch_size)
            ):
                actor_batch = image_replay.sample(ae_batch_size, device=device)
                actor_info = ae_flow_actor_step(
                    model,
                    ae_backend,
                    optimizers["actor"],
                    optimizers["alpha"],
                    actor_batch,
                    beta=args.actor_beta,
                    target_divergence=args.target_divergence,
                    bc_ref_coef=float(getattr(args, "bc_ref_coef", 1.0)),
                )
                if args.use_cf_guide:
                    guide_info = ae_flow_guide_step(
                        model,
                        ae_backend,
                        optimizers["guide"],
                        actor_batch,
                        beta=args.guide_beta,
                    )
            else:
                actor_batch = replay.sample(args.batch_size, device=device)
                actor_fn = flow_actor_step if model.is_flow else actor_step
                actor_info = actor_fn(
                    model,
                    optimizers["actor"],
                    optimizers["alpha"],
                    actor_batch,
                    beta=args.actor_beta,
                    target_divergence=args.target_divergence,
                    ref_dropout=args.ref_dropout,
                    **(
                        {"bc_ref_coef": float(getattr(args, "bc_ref_coef", 1.0))}
                        if model.is_flow
                        else {}
                    ),
                )
                if args.use_cf_guide:
                    guide_fn = flow_guide_step if model.is_flow else guide_step
                    guide_kwargs = {"beta": args.guide_beta}
                    guide_info = guide_fn(
                        model,
                        optimizers["guide"],
                        actor_batch,
                        **guide_kwargs,
                    )
        if args.tune_token_online and len(token_replay) > 0:
            token_batch = token_replay.sample(args.token_batch_size, device=device)
            token_info = token_step(model, optimizers["token"], token_batch)
    return q_info, actor_info, guide_info, token_info


def _atomic_model_save(
    model: MolmoAct2RLTCF,
    path: Path,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")

    def _write() -> None:
        _cleanup_path(temporary)
        model.save(str(temporary), meta=meta)
        # Ensure bytes hit stable storage before the atomic rename.
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    try:
        _io_retry(f"checkpoint save {path}", _write)
    except Exception:
        _cleanup_path(temporary)
        raise


def _save_checkpoint(
    model: MolmoAct2RLTCF,
    out_dir: Path,
    env_steps: int,
    meta: dict[str, Any],
) -> Path:
    latest = out_dir / "rlt_cf_latest.pt"
    _atomic_model_save(model, latest, {**meta, "env_steps": int(env_steps)})

    def _write_pointer() -> None:
        (out_dir / "LATEST_CKPT.txt").write_text(f"{latest.name}\n", encoding="utf-8")

    _io_retry(f"LATEST_CKPT.txt in {out_dir}", _write_pointer)
    return latest


def _save_chunk_replay(replay: ChunkReplay, replay_out: str) -> None:
    if not replay_out or len(replay) == 0:
        return
    path = Path(replay_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")

    def _write() -> None:
        _cleanup_path(temporary)
        replay.save_npz(str(temporary))
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    try:
        _io_retry(f"chunk replay save {path}", _write)
    except Exception:
        _cleanup_path(temporary)
        raise


def _save_ae_image_replay(
    replay: ImageChunkReplay,
    replay_out: str,
) -> None:
    if not replay_out or len(replay) == 0:
        return
    path = Path(replay_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")

    def _write() -> None:
        _cleanup_path(temporary)
        replay.save_npz(str(temporary))
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    try:
        _io_retry(f"AE image replay save {path}", _write)
    except Exception:
        _cleanup_path(temporary)
        raise


def _save_ae_trainable(
    ae_backend: MolmoAEBackend,
    out_dir: Path,
    env_steps: int,
) -> None:
    path = out_dir / "molmo_ae_lora_latest.pt"
    temporary = path.with_name(f".{path.name}.tmp")

    def _write() -> None:
        _cleanup_path(temporary)
        ae_backend.save_trainable(
            temporary,
            meta={"env_steps": int(env_steps)},
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    try:
        _io_retry(f"AE trainable save {path}", _write)
    except Exception:
        _cleanup_path(temporary)
        raise


def _save_eval_snapshot(
    model: MolmoAct2RLTCF,
    ae_backend: MolmoAEBackend | None,
    out_dir: Path,
    valid_episodes: int,
    env_steps: int,
    meta: dict[str, Any],
) -> Path:
    """Save an immutable evaluation bundle at a requested episode milestone."""
    snapshot_dir = out_dir / "snapshots" / f"ep_{int(valid_episodes):06d}"
    required = [snapshot_dir / "rlt_cf.pt", snapshot_dir / "snapshot.json"]
    if ae_backend is not None:
        required.append(snapshot_dir / "molmo_ae_lora.pt")
    if snapshot_dir.exists():
        if all(path.is_file() for path in required):
            return snapshot_dir
        raise RuntimeError(f"Refusing partial evaluation snapshot: {snapshot_dir}")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_meta = {
        **meta,
        "snapshot": True,
        "valid_episodes": int(valid_episodes),
        "env_steps": int(env_steps),
    }
    rlt_path = snapshot_dir / "rlt_cf.pt"
    _atomic_model_save(model, rlt_path, snapshot_meta)
    if ae_backend is not None:
        ae_path = snapshot_dir / "molmo_ae_lora.pt"
        temporary = ae_path.with_name(f".{ae_path.name}.tmp")
        _cleanup_path(temporary)
        ae_backend.save_trainable(
            temporary,
            meta={
                "valid_episodes": int(valid_episodes),
                "env_steps": int(env_steps),
            },
        )
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, ae_path)
    manifest_path = snapshot_dir / "snapshot.json"
    temporary_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest.write_text(
        json.dumps(snapshot_meta, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return snapshot_dir


def _append_metrics_row(metrics_path: Path, row: dict[str, Any]) -> None:
    def _write() -> None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    _io_retry(f"metrics append {metrics_path}", _write)


def _persist_training_state(
    *,
    model: MolmoAct2RLTCF,
    out_dir: Path,
    metrics_path: Path,
    env_steps: int,
    row: dict[str, Any],
    replay: ChunkReplay,
    replay_out: str,
    required: bool,
    ae_backend: MolmoAEBackend | None = None,
    ae_image_replay: ImageChunkReplay | None = None,
    ae_image_replay_out: str = "",
) -> Path | None:
    """Checkpoint first, then metrics — so counters never outrun weights.

    On transient NFS failures, log and continue unless ``required`` (final save).
    Always runs outside the EGL GPU lock.
    """
    checkpoint: Path | None = None
    try:
        checkpoint = _save_checkpoint(model, out_dir, env_steps, row)
    except Exception as error:  # noqa: BLE001
        log.error(
            "Checkpoint save failed after retries (steps=%d): %s",
            env_steps,
            error,
        )
        if required:
            raise
        return None

    try:
        _save_chunk_replay(replay, replay_out)
    except Exception as error:  # noqa: BLE001
        log.warning(
            "Chunk replay save failed after retries (steps=%d): %s",
            env_steps,
            error,
        )
        if required:
            raise

    if ae_backend is not None:
        if ae_image_replay is None or not ae_image_replay_out:
            raise RuntimeError(
                "AE checkpoint persistence requires its paired image replay"
            )
        try:
            _save_ae_image_replay(ae_image_replay, ae_image_replay_out)
        except Exception as error:  # noqa: BLE001
            log.error(
                "AE image replay save failed after retries (steps=%d): %s",
                env_steps,
                error,
            )
            if required:
                raise
            return None
        try:
            _save_ae_trainable(ae_backend, out_dir, env_steps)
        except Exception as error:  # noqa: BLE001
            log.error(
                "AE trainable save failed after retries (steps=%d): %s",
                env_steps,
                error,
            )
            if required:
                raise
            return None

    try:
        _append_metrics_row(metrics_path, row)
    except Exception as error:  # noqa: BLE001
        log.error(
            "Metrics append failed after retries (steps=%d): %s",
            env_steps,
            error,
        )
        if required:
            raise
        # Keep weights on disk, but signal caller to retry metrics next cadence
        # so resume counters do not lag the checkpoint forever unnoticed.
        return None

    return checkpoint


def _metrics_row(
    args: argparse.Namespace,
    *,
    env_steps: int,
    valid_episodes: int,
    skipped_episodes: int,
    successes: int,
    recent: deque[float],
    start_time: float,
    gate_status: GateStatus,
    replay: ChunkReplay,
    q_info: dict[str, float],
    actor_info: dict[str, float],
    guide_info: dict[str, float],
    token_info: dict[str, float],
) -> dict[str, Any]:
    elapsed = max(time.time() - start_time, 1e-6)
    return {
        "env_steps": int(env_steps),
        "target_env_steps": int(args.target_env_steps),
        "valid_episodes": int(valid_episodes),
        "skipped_episodes": int(skipped_episodes),
        "cumulative_success_rate": successes / max(valid_episodes, 1),
        "window_success_rate": float(np.mean(recent)) if recent else 0.0,
        "g_enabled": bool(gate_status.deploy_actor),
        "g_predicted_advantage": float(gate_status.paired_lcb),
        "gate_would_enable": bool(gate_status.would_enable),
        "gate_deploy_actor": bool(gate_status.deploy_actor),
        "gate_deploy_guide": bool(gate_status.deploy_guide),
        "gate_paired_lcb": float(gate_status.paired_lcb),
        "gate_q_min_advantage": float(gate_status.q_min_advantage),
        "gate_guide_advantage": float(gate_status.guide_advantage),
        "gate_critic_health": bool(gate_status.critic_health),
        "gate_sensitivity": float(gate_status.sensitivity),
        "critic_healthy": bool(gate_status.critic_health),
        "action_sensitivity": float(gate_status.sensitivity),
        "q_td_loss": float(q_info.get("q_td_loss", 0.0)),
        "q_rank_loss": float(q_info.get("q_rank_loss", 0.0)),
        "q_rank_gap": float(q_info.get("q_rank_gap", 0.0)),
        "q_mean": float(q_info.get("q_mean", 0.0)),
        "q_std": float(q_info.get("q_std", 0.0)),
        "actor_adv": float(actor_info.get("actor_adv", 0.0)),
        "ae_grad_norm": float(actor_info.get("ae_grad_norm", 0.0)),
        "guide_adv": float(guide_info.get("guide_adv", 0.0)),
        "guide_w_norm": float(guide_info.get("w_norm", 0.0)),
        "guide_target_norm": float(guide_info.get("target_norm", 0.0)),
        "token_recon_loss": float(token_info.get("token_recon_loss", 0.0)),
        "config_name": args.config_name,
        "v_source": str(getattr(args, "v_source", "rlt")),
        "steps_per_sec": env_steps / elapsed,
        "chunk_transitions": len(replay),
        "server_port": int(args.server_port),
        "elapsed_sec": elapsed,
    }


def train_rlt_online(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.assets_dir:
        os.environ["MLSPACES_ASSETS_DIR"] = args.assets_dir
    eval_only = bool(getattr(args, "eval_only", False))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not str(getattr(args, "replay_out", "")):
        args.replay_out = str(out_dir / "chunk_replay.npz")
    if (
        bool(getattr(args, "ae_trainable", False))
        and not str(getattr(args, "ae_image_replay_out", ""))
    ):
        args.ae_image_replay_out = str(out_dir / "ae_image_replay.npz")
    metrics_path = out_dir / "metrics.jsonl"
    validation_results_path = out_dir / "validation_results.jsonl"
    validation_summary_path = out_dir / "validation_summary.json"
    tmp_rollouts = (
        Path(args.tmp_rollout_dir)
        / f"rlt_port{args.server_port}_pid{os.getpid()}"
    )
    tmp_rollouts.mkdir(parents=True, exist_ok=True)

    resume_artifacts = _resolve_resume_artifacts(args)
    model = _load_model(
        args,
        device,
        resume_checkpoint=resume_artifacts.checkpoint,
    )
    ae_backend = None
    image_replay: ImageChunkReplay | None = None
    if bool(getattr(args, "ae_trainable", False)):
        if str(args.cf_mode) != CF_MODE_FLOW:
            raise ValueError("--ae_trainable requires --cf_mode flow")
        if not bool(getattr(args, "ae_lora", True)):
            raise ValueError(
                "--ae_trainable requires AE LoRA so adapter-disabled "
                "predictions remain a frozen reference"
            )
        # Plan: RL token frozen; Molmo AE is V.
        args.tune_token_online = False
        model.freeze_token_encoder()
        model.v_source = "molmo_ae"
        args.v_source = "molmo_ae"
        ae_backend = MolmoAEBackend(
            device=device,
            dtype=torch.bfloat16,
            enable_lora=bool(getattr(args, "ae_lora", True)),
            lora_rank=int(getattr(args, "ae_lora_rank", 16)),
            lora_alpha=int(getattr(args, "ae_lora_alpha", 32)),
            num_steps=int(args.flow_steps),
            rlt=model,
            feature_mode="tokens",
        )
        ae_backend.rlt = model
        if resume_artifacts.ae_trainable is not None:
            ae_meta = ae_backend.load_trainable(resume_artifacts.ae_trainable)
            model_steps = (getattr(model, "loaded_meta", {}) or {}).get("env_steps")
            ae_steps = ae_meta.get("env_steps")
            if (
                model_steps is not None
                and ae_steps is not None
                and int(model_steps) != int(ae_steps)
            ):
                raise RuntimeError(
                    "Refusing mismatched AE/RLT resume artifacts: "
                    f"RLT env_steps={model_steps}, AE env_steps={ae_steps}"
                )
            log.info(
                "Resumed AE trainable weights %s",
                resume_artifacts.ae_trainable,
            )
        elif (
            eval_only
            and int((getattr(model, "loaded_meta", {}) or {}).get("env_steps", 0) or 0) > 0
        ):
            raise RuntimeError(
                "Evaluation of an online AE checkpoint requires its paired "
                "molmo_ae_lora_latest.pt (use --ae_trainable_ckpt)"
            )
        # Freeze unused RLT FlowVelocityActor; AE LoRA is the trainable V.
        for p in model.actor.parameters():
            p.requires_grad_(False)
        if resume_artifacts.ae_replay is not None:
            image_replay = ImageChunkReplay.load_npz(
                str(resume_artifacts.ae_replay),
                max_transitions=int(
                    getattr(args, "ae_image_replay_capacity", 512)
                ),
                pos_frac=args.pos_frac,
                seed=args.seed,
            )
            log.info(
                "Resumed AE image replay %s (%d transitions)",
                resume_artifacts.ae_replay,
                len(image_replay),
            )
        else:
            image_replay = ImageChunkReplay(
                max_transitions=int(getattr(args, "ae_image_replay_capacity", 512)),
                chunk_size=CHUNK_SIZE,
                action_dim=ACTION_DIM,
                z_dim=Z_DIM,
                pos_frac=args.pos_frac,
                seed=args.seed,
            )
        log.info(
            "V11_1 AE-as-V enabled: trainable_ae=%d params, token frozen, v_source=molmo_ae",
            sum(p.numel() for p in ae_backend.trainable_parameters()),
        )

    optimizers: dict[str, torch.optim.Optimizer] = {}
    if not eval_only:
        optimizers = build_rlt_optimizers(
            model,
            lr_token=args.lr_token,
            lr_critic=args.lr_critic,
            lr_actor=args.lr_actor,
            lr_guide=args.lr_guide,
            lr_alpha=args.lr_alpha,
        )
        if ae_backend is not None:
            optimizers["actor"] = torch.optim.Adam(
                ae_backend.trainable_parameters(),
                lr=float(getattr(args, "lr_ae", args.lr_actor)),
            )

    replay = ChunkReplay(
        max_transitions=args.replay_capacity,
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
        z_dim=Z_DIM,
        pos_frac=args.pos_frac,
        seed=args.seed,
    )
    if args.replay_out and Path(args.replay_out).is_file() and not args.no_resume:
        try:
            replay = ChunkReplay.load_npz(
                args.replay_out,
                max_transitions=args.replay_capacity,
                pos_frac=args.pos_frac,
                seed=args.seed,
            )
            log.info("Resumed chunk replay %s (%d transitions)", args.replay_out, len(replay))
        except Exception as error:  # noqa: BLE001
            if resume_artifacts.checkpoint is not None:
                raise RuntimeError(
                    f"Failed to load replay paired with resumed model: {args.replay_out}"
                ) from error
            log.warning("Failed to resume chunk replay %s: %s", args.replay_out, error)
    token_replay = TokenReplay(
        max_seq=args.token_max_seq,
        token_dim=model.feature_dim,
    )

    if ae_backend is None:
        health = _wait_for_server(
            args.server_host, args.server_port, args.server_wait_sec
        )
        _validate_server_features(health)
        feature_mode_log = health.get("feature_mode", "unknown")
    else:
        log.info("In-process AE backend active; skipping HTTP MolmoAct2 server wait")
        feature_mode_log = "in_process_ae_tokens"
    policy, _exp_config = _build_eval_policy(
        args, model, device, ae_backend=ae_backend
    )
    bench = Path(args.benchmark_dir) if args.benchmark_dir else _default_bench()
    n_bench = _bench_size(bench)
    shard_size = int(args.shard_size) if args.shard_size > 0 else n_bench
    shard_start = int(args.start_episode) % n_bench

    log.info(
        "RLT online config=%s target_steps=%d server=%s:%d feature_mode=%s "
        "actor_mode=%s guide=%s tune_token=%s ae_trainable=%s v_source=%s shard=[%d,+%d)",
        args.config_name,
        args.target_env_steps,
        args.server_host,
        args.server_port,
        feature_mode_log,
        args.actor_mode,
        args.use_cf_guide,
        args.tune_token_online,
        bool(ae_backend is not None),
        getattr(model, "v_source", "rlt"),
        shard_start,
        shard_size,
    )

    env_steps = 0
    valid_episodes = 0
    skipped_episodes = 0
    successes = 0
    cycle = 0
    recent: deque[float] = deque(maxlen=args.window_episodes)
    start_time = time.time()
    last_q: dict[str, float] = {}
    last_actor: dict[str, float] = {}
    last_guide: dict[str, float] = {}
    last_token: dict[str, float] = {}
    last_gate = GateStatus(False, False, False, 0.0, 0.0, 0.0, False, 0.0)
    last_logged_episode = -1
    warned_missing_tokens = False

    resume_row = None if args.no_resume else _load_metrics_resume(out_dir)
    if (
        resume_row is None
        and resume_artifacts.checkpoint is not None
        and getattr(model, "loaded_meta", None)
    ):
        resume_row = dict(model.loaded_meta)
    checkpoint_gate_meta = dict(getattr(model, "loaded_meta", {}) or {})
    if resume_row is not None:
        checkpoint_gate_meta.update(resume_row)
    if resume_row is not None and not eval_only:
        env_steps = int(resume_row.get("env_steps", 0) or 0)
        valid_episodes = int(resume_row.get("valid_episodes", 0) or 0)
        skipped_episodes = int(resume_row.get("skipped_episodes", 0) or 0)
        rate = float(resume_row.get("cumulative_success_rate", 0.0) or 0.0)
        successes = int(round(rate * max(valid_episodes, 1))) if valid_episodes else 0
        cycle = valid_episodes + skipped_episodes
        last_logged_episode = valid_episodes
        log.info(
            "Resumed counters steps=%d eps=%d skipped=%d successes=%d",
            env_steps,
            valid_episodes,
            skipped_episodes,
            successes,
        )

    snapshot_episodes = _parse_snapshot_episodes(
        str(getattr(args, "snapshot_episodes", ""))
    )
    if not eval_only and valid_episodes in snapshot_episodes:
        startup_snapshot_meta = {
            **checkpoint_gate_meta,
            "config_name": str(args.config_name),
            "gate_would_enable": bool(
                checkpoint_gate_meta.get("gate_would_enable", False)
            ),
            "gate_deploy_actor": bool(
                checkpoint_gate_meta.get("gate_deploy_actor", False)
            ),
            "gate_deploy_guide": bool(
                checkpoint_gate_meta.get("gate_deploy_guide", False)
            ),
        }
        snapshot_dir = _save_eval_snapshot(
            model,
            ae_backend,
            out_dir,
            valid_episodes,
            env_steps,
            startup_snapshot_meta,
        )
        log.info("Evaluation snapshot ready: %s", snapshot_dir)

    while (
        not _STOP_REQUESTED
        and env_steps < args.target_env_steps
        and (
            args.max_valid_episodes <= 0
            or valid_episodes < args.max_valid_episodes
        )
    ):
        if ae_backend is None:
            health = _server_health(args.server_host, args.server_port)
            if health is None:
                log.warning("MolmoAct2 server is unavailable; waiting")
                health = _wait_for_server(
                    args.server_host, args.server_port, args.server_wait_sec
                )
                _validate_server_features(health)
                policy.prepare_model()

        gate = _gate_status(
            args,
            model,
            replay,
            valid_episodes,
            device,
            ae_backend=ae_backend,
            image_replay=image_replay,
            checkpoint_meta=checkpoint_gate_meta,
        )
        policy.deploy_actor = gate.deploy_actor
        policy.deploy_guide = gate.deploy_guide
        model.eval()

        episode_idx = shard_start + (cycle % shard_size)
        cycle += 1
        episode_dir = tmp_rollouts / f"ep_{cycle:08d}"
        shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_start = time.time()
        rollout_ok = True
        success = False
        trajectory: dict[str, Any] = {"n_steps": 0, "token_batches": [], "residual_rms": 0.0}
        n_steps = 0
        last_q, last_actor, last_guide, last_token = {}, {}, {}, {}
        try:
            # Serialize MuJoCo EGL only; CUDA train runs after unlock so it does
            # not extend the per-GPU EGL queue.
            with _egl_gpu_lock():
                results = run_evaluation(
                    eval_config_cls=MolmoAct2PolicyEvalConfig,
                    benchmark_dir=bench,
                    task_horizon_steps=args.horizon,
                    num_workers=1,
                    use_wandb=False,
                    preloaded_policy=policy,
                    episode_idx=episode_idx,
                    output_dir=episode_dir,
                )
                success = bool(results.success_count > 0)
                rollout_ok = bool(results.total_count > 0)
                trajectory = policy.pop_episode(success)
                if policy.fatal_error is not None:
                    raise policy.fatal_error

                n_steps = int(trajectory["n_steps"])
                if rollout_ok and n_steps > 0 and not eval_only:
                    replay.add_episode_chunks(
                        trajectory["zs"],
                        trajectory["proprios"],
                        trajectory["references"],
                        trajectory["executed"],
                        trajectory["rewards"],
                        trajectory["masks"],
                        success=success,
                        gamma=args.gamma,
                        episode_id=valid_episodes,
                    )
                    if image_replay is not None and trajectory.get("external_cams"):
                        image_replay.add_episode(
                            zs=trajectory["zs"],
                            proprios=trajectory["proprios"],
                            references=trajectory["references"],
                            executed=trajectory["executed"],
                            rewards=trajectory["rewards"],
                            masks=trajectory["masks"],
                            external_cams=trajectory["external_cams"],
                            wrist_cams=trajectory["wrist_cams"],
                            instructions=trajectory["instructions"],
                            success=success,
                            gamma=args.gamma,
                            episode_id=valid_episodes,
                        )
                    for tokens, mask in trajectory["token_batches"]:
                        token_replay.add(tokens, mask)
                    token_overflow = len(token_replay) - args.token_replay_capacity
                    if token_overflow > 0:
                        del token_replay.tokens[:token_overflow]
                        del token_replay.masks[:token_overflow]
            if rollout_ok and n_steps > 0 and not eval_only:
                last_q, last_actor, last_guide, last_token = _train_after_episode(
                    args,
                    model,
                    optimizers,
                    replay,
                    token_replay,
                    device,
                    ae_backend=ae_backend,
                    image_replay=image_replay,
                )
        except Exception as error:  # noqa: BLE001
            log.warning("Episode %d rollout failed: %s", episode_idx, error)
            success = False
            rollout_ok = False
            if int(trajectory.get("n_steps", 0) or 0) <= 0:
                trajectory = policy.pop_episode(success)
                n_steps = int(trajectory.get("n_steps", 0) or 0)
        shutil.rmtree(episode_dir, ignore_errors=True)
        if policy.fatal_error is not None:
            raise policy.fatal_error

        if not rollout_ok or n_steps <= 0:
            skipped_episodes += 1
            if eval_only:
                _append_metrics_row(
                    validation_results_path,
                    {
                        "episode_idx": int(episode_idx),
                        "valid": False,
                        "success": False,
                        "n_steps": int(n_steps),
                        "deploy_policy": str(args.deploy_policy),
                        "gate_would_enable": bool(gate.would_enable),
                        "gate_deploy_actor": bool(gate.deploy_actor),
                        "gate_deploy_guide": bool(gate.deploy_guide),
                    },
                )
            log.warning(
                "Skipping invalid episode idx=%d rollout_ok=%s steps=%d skipped=%d",
                episode_idx,
                rollout_ok,
                n_steps,
                skipped_episodes,
            )
            time.sleep(2.0)
            continue

        if (
            args.tune_token_online
            and not trajectory["token_batches"]
            and not warned_missing_tokens
        ):
            log.warning(
                "--tune_token_online is enabled, but the server response contains no "
                "token_features; token reconstruction updates are disabled"
            )
            warned_missing_tokens = True

        env_steps += n_steps
        valid_episodes += 1
        successes += int(success)
        recent.append(float(success))
        # Post-episode gate (metrics + next-episode deploy use the same check).
        last_gate = _gate_status(
            args,
            model,
            replay,
            valid_episodes,
            device,
            ae_backend=ae_backend,
            image_replay=image_replay,
            checkpoint_meta=checkpoint_gate_meta,
        )

        log.info(
            "steps=%d/%d eps=%d idx=%d success=%s ep_steps=%d gate=%s "
            "lcb=%.5f qmin=%.5f sens=%.5f q_td=%.5f rank=%.5f "
            "actor_adv=%.5f guide_adv=%.5f "
            "residual_rms=%.5f dt=%.1fs sr=%.3f",
            env_steps,
            args.target_env_steps,
            valid_episodes,
            episode_idx,
            success,
            n_steps,
            last_gate.deploy_actor,
            last_gate.paired_lcb,
            last_gate.q_min_advantage,
            last_gate.sensitivity,
            last_q.get("q_td_loss", 0.0),
            last_q.get("q_rank_loss", 0.0),
            last_actor.get("actor_adv", 0.0),
            last_guide.get("guide_adv", 0.0),
            trajectory["residual_rms"],
            time.time() - episode_start,
            successes / max(valid_episodes, 1),
        )

        if eval_only:
            validation_row = _metrics_row(
                args,
                env_steps=env_steps,
                valid_episodes=valid_episodes,
                skipped_episodes=skipped_episodes,
                successes=successes,
                recent=recent,
                start_time=start_time,
                gate_status=last_gate,
                replay=replay,
                q_info={},
                actor_info={},
                guide_info={},
                token_info={},
            )
            validation_row.update(
                {
                    "episode_idx": int(episode_idx),
                    "valid": True,
                    "success": bool(success),
                    "episode_steps": int(n_steps),
                    "deploy_policy": str(args.deploy_policy),
                    "eval_only": True,
                }
            )
            _append_metrics_row(validation_results_path, validation_row)
            continue

        should_log = (
            valid_episodes % args.log_every_episodes == 0
            or valid_episodes % args.ckpt_every_episodes == 0
            or env_steps >= args.target_env_steps
            or (
                args.max_valid_episodes > 0
                and valid_episodes >= args.max_valid_episodes
            )
            or _STOP_REQUESTED
        )
        if should_log:
            row = _metrics_row(
                args,
                env_steps=env_steps,
                valid_episodes=valid_episodes,
                skipped_episodes=skipped_episodes,
                successes=successes,
                recent=recent,
                start_time=start_time,
                gate_status=last_gate,
                replay=replay,
                q_info=last_q,
                actor_info=last_actor,
                guide_info=last_guide,
                token_info=last_token,
            )
            # Persist outside the EGL lock (already released) so NFS stalls
            # cannot block other trainers on this GPU.
            checkpoint = _persist_training_state(
                model=model,
                out_dir=out_dir,
                metrics_path=metrics_path,
                env_steps=env_steps,
                row=row,
                replay=replay,
                replay_out=args.replay_out,
                required=False,
                ae_backend=ae_backend,
                ae_image_replay=image_replay,
                ae_image_replay_out=str(
                    getattr(args, "ae_image_replay_out", "")
                ),
            )
            if checkpoint is None:
                log.warning(
                    "Skipping metrics bump this cycle; will retry next ckpt cadence "
                    "(steps=%d eps=%d)",
                    env_steps,
                    valid_episodes,
                )
                continue
            checkpoint_gate_meta.clear()
            checkpoint_gate_meta.update(row)
            model.loaded_meta = dict(row)
            if valid_episodes in snapshot_episodes:
                snapshot_dir = _save_eval_snapshot(
                    model,
                    ae_backend,
                    out_dir,
                    valid_episodes,
                    env_steps,
                    row,
                )
                log.info("Evaluation snapshot ready: %s", snapshot_dir)
            last_logged_episode = valid_episodes
            log.info(
                "METRICS config=%s steps=%d sr=%.3f window_sr=%.3f "
                "lcb=%.5f sps=%.2f checkpoint=%s",
                args.config_name,
                env_steps,
                row["cumulative_success_rate"],
                row["window_success_rate"],
                row["g_predicted_advantage"],
                row["steps_per_sec"],
                checkpoint.name,
            )

    final_row = _metrics_row(
        args,
        env_steps=env_steps,
        valid_episodes=valid_episodes,
        skipped_episodes=skipped_episodes,
        successes=successes,
        recent=recent,
        start_time=start_time,
        gate_status=last_gate,
        replay=replay,
        q_info=last_q,
        actor_info=last_actor,
        guide_info=last_guide,
        token_info=last_token,
    )
    if eval_only:
        summary = {
            **final_row,
            "eval_only": True,
            "deploy_policy": str(args.deploy_policy),
            "stopped_by_signal": bool(_STOP_REQUESTED),
            "validation_results_path": str(validation_results_path),
        }
        _io_retry(
            f"validation summary in {out_dir}",
            lambda: validation_summary_path.write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            ),
        )
        shutil.rmtree(tmp_rollouts, ignore_errors=True)
        log.info("Validation done: %s", json.dumps(summary))
        return

    checkpoint = _persist_training_state(
        model=model,
        out_dir=out_dir,
        metrics_path=metrics_path,
        env_steps=env_steps,
        row=final_row,
        replay=replay,
        replay_out=args.replay_out,
        required=False,
        ae_backend=ae_backend,
        ae_image_replay=image_replay,
        ae_image_replay_out=str(getattr(args, "ae_image_replay_out", "")),
    )
    if checkpoint is None:
        checkpoint = out_dir / "rlt_cf_latest.pt"
        log.error(
            "Final checkpoint/metrics persist failed; last good ckpt may be stale: %s",
            checkpoint,
        )
    summary = {
        **final_row,
        "stopped_by_signal": bool(_STOP_REQUESTED),
        "checkpoint": str(checkpoint),
        "metrics_path": str(metrics_path),
    }
    try:
        _io_retry(
            f"summary.json in {out_dir}",
            lambda: (out_dir / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            ),
        )
    except Exception as error:  # noqa: BLE001
        log.error("Failed to write summary.json: %s", error)
    shutil.rmtree(tmp_rollouts, ignore_errors=True)
    log.info("Done: %s", json.dumps(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server_host", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--server_wait_sec", type=float, default=1800.0)
    parser.add_argument("--server_request_timeout_sec", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--out_dir", type=str, default=str(_HERE / "runs/rlt_cf_online"))
    parser.add_argument("--rlt_ckpt", type=str, default="")
    parser.add_argument(
        "--n_critics",
        type=int,
        default=10,
        help="Critic ensemble size (ConsensusFlow default K=10)",
    )
    parser.add_argument("--config_name", type=str, default="rlt_cf")
    parser.add_argument("--benchmark_dir", type=str, default="")
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=os.path.expanduser("~/.cache/molmospaces/assets"),
    )
    parser.add_argument(
        "--tmp_rollout_dir",
        type=str,
        default=_DEFAULT_TMP_ROLLOUT_DIR,
    )
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=125)
    parser.add_argument("--target_env_steps", type=int, default=12_500_000)
    parser.add_argument("--max_valid_episodes", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--log_every_episodes", type=int, default=100)
    parser.add_argument(
        "--ckpt_every_episodes",
        type=int,
        default=10,
        help="Checkpoint/metrics cadence for crash recovery (watchdog resumes).",
    )
    parser.add_argument(
        "--snapshot_episodes",
        type=str,
        default="",
        help="Comma-separated valid-episode milestones for immutable eval bundles.",
    )
    parser.add_argument("--window_episodes", type=int, default=100)
    parser.add_argument("--replay_out", type=str, default="")
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore out_dir/rlt_cf_latest.pt and chunk_replay.npz on start.",
    )
    parser.add_argument("--replay_capacity", type=int, default=50_000)
    parser.add_argument("--pos_frac", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--min_replay_chunks", type=int, default=8)
    parser.add_argument("--updates_per_episode", type=int, default=5)
    parser.add_argument(
        "--actor_mode",
        choices=["rlt", "vla_only"],
        default="rlt",
        help="vla_only always executes the frozen MolmoAct2 reference chunk",
    )
    parser.add_argument(
        "--cf_mode",
        choices=[CF_MODE_RESIDUAL, CF_MODE_FLOW],
        default=CF_MODE_RESIDUAL,
        help="residual = one-shot CF; flow = paper ConsensusFlow denoising ODE",
    )
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--guidance_coef", type=float, default=0.5)

    guide_group = parser.add_mutually_exclusive_group()
    guide_group.add_argument(
        "--use_cf_guide",
        dest="use_cf_guide",
        action="store_true",
    )
    guide_group.add_argument(
        "--no_cf_guide",
        dest="use_cf_guide",
        action="store_false",
    )
    parser.set_defaults(use_cf_guide=True)

    token_group = parser.add_mutually_exclusive_group()
    token_group.add_argument(
        "--tune_token_online",
        dest="tune_token_online",
        action="store_true",
    )
    token_group.add_argument(
        "--freeze_token",
        dest="tune_token_online",
        action="store_false",
    )
    parser.set_defaults(tune_token_online=False)

    parser.add_argument("--g_start_episodes", type=int, default=40)
    parser.add_argument(
        "--deploy_policy",
        choices=["gated", "checkpoint_gate", "reference", "actor", "actor_guide"],
        default="gated",
        help=(
            "Deployment selection: production gate, saved gate decision, frozen "
            "reference, trainable actor, or trainable actor plus guide."
        ),
    )
    parser.add_argument(
        "--force_deploy_rlt",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help=(
            "Run validation rollouts without exploration, replay/training updates, "
            "or training-artifact writes."
        ),
    )
    parser.add_argument(
        "--ae_trainable",
        action="store_true",
        help=(
            "V11_1: load MolmoAct2 Action Expert in-process as trainable V "
            "(AE-only LoRA); RLT guide is G; forces --freeze_token and flow mode."
        ),
    )
    parser.add_argument(
        "--ae_lora",
        dest="ae_lora",
        action="store_true",
        default=True,
        help="Use AE-only LoRA (default on with --ae_trainable)",
    )
    parser.add_argument(
        "--no_ae_lora",
        dest="ae_lora",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--ae_lora_rank", type=int, default=16)
    parser.add_argument("--ae_lora_alpha", type=int, default=32)
    parser.add_argument(
        "--ae_trainable_ckpt",
        type=str,
        default="",
        help=(
            "Explicit AE LoRA checkpoint for eval-only runs. If omitted, "
            "infer molmo_ae_lora_latest.pt beside --rlt_ckpt."
        ),
    )
    parser.add_argument("--ae_batch_size", type=int, default=2)
    parser.add_argument("--ae_image_replay_capacity", type=int, default=512)
    parser.add_argument(
        "--ae_image_replay_out",
        type=str,
        default="",
        help="Persisted AE image replay path (defaults under out_dir).",
    )
    parser.add_argument("--lr_ae", type=float, default=1e-4)
    parser.add_argument("--g_min_advantage", type=float, default=0.003)
    parser.add_argument(
        "--g_min_guide_advantage",
        type=float,
        default=0.0,
        help=(
            "When >0 and CF guide is on, also require guide-vs-actor q_min "
            "advantage before deploy (residual CF safety)."
        ),
    )
    parser.add_argument(
        "--g_min_action_sensitivity",
        type=float,
        default=0.003,
        help="Refuse deploy unless mean |Q(a)-Q(a+ε)| clears this floor",
    )
    parser.add_argument("--gate_sensitivity_noise", type=float, default=0.08)
    parser.add_argument(
        "--explore_residual_std",
        type=float,
        default=0.02,
        help="Normalized-action exploration while the RLT gate is closed (v12: 0.02)",
    )
    parser.add_argument(
        "--explore_deploy_std",
        type=float,
        default=0.02,
        help="Normalized-action exploration once the RLT/CF gate is open",
    )
    parser.add_argument(
        "--explore_warmup_mult",
        type=float,
        default=1.0,
        help="Multiplier on explore_residual_std while gate is closed (v12: 1.0, was 1.5)",
    )
    parser.add_argument(
        "--bc_ref_coef",
        type=float,
        default=1.0,
        help=(
            "Flow/AE actor: weight on BC toward VLA reference vs executed "
            "(1.0 = reference only; v12 default)."
        ),
    )
    parser.add_argument("--rank_coef", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--rank_noise", type=float, default=0.08)
    parser.add_argument("--far_rank_coef", type=float, default=0.5)
    parser.add_argument("--far_rank_noise", type=float, default=0.35)
    parser.add_argument("--shuffle_rank_coef", type=float, default=0.5)
    parser.add_argument("--target_noise", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mc_coef", type=float, default=0.1)
    parser.add_argument("--cql_coef", type=float, default=0.1)
    parser.add_argument("--cql_n_actions", type=int, default=8)
    parser.add_argument("--cql_action_radius", type=float, default=0.2)
    parser.add_argument("--ref_dropout", type=float, default=0.5)
    parser.add_argument("--actor_beta", type=float, default=1.0)
    parser.add_argument("--guide_beta", type=float, default=0.05)
    parser.add_argument(
        "--guide_target_delta_frac",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--target_divergence", type=float, default=0.0025)
    parser.add_argument("--lr_token", type=float, default=1e-4)
    parser.add_argument("--lr_critic", type=float, default=3e-4)
    parser.add_argument("--lr_actor", type=float, default=1e-4)
    parser.add_argument("--lr_guide", type=float, default=1e-4)
    parser.add_argument("--lr_alpha", type=float, default=1e-4)
    parser.add_argument("--token_batch_size", type=int, default=2)
    parser.add_argument("--token_max_seq", type=int, default=512)
    parser.add_argument(
        "--token_replay_capacity",
        type=int,
        default=256,
        help="Maximum raw token sequences retained for online AE updates",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.ae_trainable:
        if not args.ae_lora:
            parser.error(
                "--ae_trainable requires LoRA so adapter-disabled reference "
                "predictions stay frozen"
            )
        args.tune_token_online = False
        args.cf_mode = CF_MODE_FLOW
        args.v_source = "molmo_ae"
    else:
        args.v_source = "rlt"

    if args.force_deploy_rlt:
        log.warning(
            "--force_deploy_rlt is deprecated; use --deploy_policy actor_guide"
        )
        if args.deploy_policy != "gated":
            parser.error(
                "--force_deploy_rlt cannot be combined with an explicit --deploy_policy"
            )
        args.deploy_policy = "actor_guide" if args.use_cf_guide else "actor"
    if args.deploy_policy == "actor_guide" and not args.use_cf_guide:
        parser.error("--deploy_policy actor_guide requires --use_cf_guide")
    if args.eval_only:
        args.explore_residual_std = 0.0
        args.explore_deploy_std = 0.0
        args.explore_warmup_mult = 1.0
        args.updates_per_episode = 0
        args.tune_token_online = False

    if args.target_env_steps <= 0:
        parser.error("--target_env_steps must be positive")
    if args.log_every_episodes <= 0:
        parser.error("--log_every_episodes must be positive")
    if args.ckpt_every_episodes <= 0:
        parser.error("--ckpt_every_episodes must be positive")
    try:
        _parse_snapshot_episodes(args.snapshot_episodes)
    except ValueError as error:
        parser.error(str(error))
    if args.updates_per_episode < 0:
        parser.error("--updates_per_episode cannot be negative")
    if args.batch_size <= 0 or args.token_batch_size <= 0:
        parser.error("batch sizes must be positive")
    if args.token_replay_capacity <= 0:
        parser.error("--token_replay_capacity must be positive")
    if args.min_replay_chunks <= 0:
        parser.error("--min_replay_chunks must be positive")
    if not 0.0 <= args.pos_frac <= 1.0:
        parser.error("--pos_frac must be in [0, 1]")
    return args


if __name__ == "__main__":
    train_rlt_online(parse_args())
