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
import hashlib
import json
import logging
import os
import random
import shutil
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
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
    ae_flow_actor_step,
    ae_flow_critic_td_step,
    ae_flow_gate_metrics,
    ae_flow_guide_step,
    actor_step,
    build_rlt_optimizers,
    critic_health_metrics,
    critic_td_step,
    endpoint_critic_mc_step,
    flow_actor_step,
    flow_critic_td_step,
    flow_guide_step,
    guide_step,
    token_step,
)
from v17_helpers import (  # noqa: E402
    ActorPhaseConfig,
    EmpiricalGateTracker,
    actor_phase_for_episode,
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
_V14_ONLINE_SCHEMA = 14
_V15_ONLINE_SCHEMA = 15
_V16_ONLINE_SCHEMA = 16
_AE_NATIVE_HORIZON = 15
_AE_PADDED_ACTION_DIM = 32
_RNG_ROLES = (
    "molmo_ae_source",
    "exploration",
    "update_sampling",
    "gate_probes",
    "actor_mixture",
)


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


def _stable_seed(
    root_seed: int,
    role: str,
    episode_id: int,
    decision_id: int,
    nonce: int = 0,
) -> int:
    """Derive a portable uint63 seed without Python's randomized hash()."""
    payload = (
        f"rlt-online-v14\0{int(root_seed)}\0{role}\0"
        f"{int(episode_id)}\0{int(decision_id)}\0{int(nonce)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False) & ((1 << 63) - 1)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _capture_process_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_torch_byte_rng_state(value: Any) -> torch.Tensor:
    """Coerce checkpoint RNG blobs to the CPU ByteTensor torch expects."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    else:
        tensor = torch.as_tensor(value, device="cpu", dtype=torch.uint8).contiguous()
    # torch.cuda Generator.set_state requires a true ByteTensor view.
    return tensor.byte()


def _restore_process_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(tuple(state["python"]))
    if "numpy" in state:
        numpy_state = state["numpy"]
        if isinstance(numpy_state, list):
            numpy_state = tuple(numpy_state)
        np.random.set_state(numpy_state)
    if "torch" in state:
        torch.set_rng_state(_as_torch_byte_rng_state(state["torch"]))
    if torch.cuda.is_available() and "torch_cuda" in state:
        saved_cuda = [
            _as_torch_byte_rng_state(item) for item in list(state["torch_cuda"])
        ]
        if len(saved_cuda) != torch.cuda.device_count():
            raise RuntimeError(
                "V14 resume CUDA RNG device count mismatch: "
                f"saved={len(saved_cuda)} current={torch.cuda.device_count()}"
            )
        try:
            torch.cuda.set_rng_state_all(saved_cuda)
        except TypeError as exc:
            # Some mid-run checkpoints carry CUDA RNG blobs that fail Generator
            # set_state under lazy CUDA init. Prefer continuing over crash-loop.
            log.warning(
                "Skipping CUDA RNG restore after TypeError (%s); "
                "CPU/policy RNG still restored",
                exc,
            )


def _validate_ae_contract(ae_backend: Any) -> dict[str, int]:
    contract = dict(ae_backend.action_contract())
    stats = ae_backend.model._get_robot_stats()
    expected = {
        "action_dim": ACTION_DIM,
        "max_action_dim": _AE_PADDED_ACTION_DIM,
        "action_horizon": _AE_NATIVE_HORIZON,
        "n_action_steps": _AE_NATIVE_HORIZON,
        "n_obs_steps": 1,
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if int(contract.get(key, -1)) != value
    }
    if getattr(stats, "norm_mode", None) != "q01_q99" or mismatches:
        raise RuntimeError(
            "V14 AE action contract mismatch: "
            f"norm_mode={getattr(stats, 'norm_mode', None)!r}, "
            f"contract={contract}, expected={expected}"
        )
    return contract


def _reset_module_parameters(
    module: torch.nn.Module,
    *,
    seed: int,
) -> None:
    parameter = next(module.parameters(), None)
    devices = (
        [parameter.device.index if parameter.device.index is not None else 0]
        if parameter is not None and parameter.is_cuda
        else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()


def _configure_ae_native_rlt_coordinates(
    args: argparse.Namespace,
    model: MolmoAct2RLTCF,
    checkpoint_meta: dict[str, Any],
) -> bool:
    """Put the AE critic and guide in Molmo-native normalized coordinates."""
    is_v14_checkpoint = (
        int(checkpoint_meta.get("online_schema_version", -1) or -1)
        == _V14_ONLINE_SCHEMA
    )
    identity_mean = torch.zeros_like(model.action_mean)
    identity_std = torch.ones_like(model.action_std)
    if is_v14_checkpoint:
        if not torch.equal(model.action_mean, identity_mean) or not torch.equal(
            model.action_std,
            identity_std,
        ):
            raise RuntimeError(
                "V14 AE checkpoint does not use identity action statistics in "
                "Molmo-native normalized coordinates"
            )
        return False
    if bool(getattr(args, "eval_only", False)):
        raise RuntimeError(
            "AE evaluation requires a V14 native-coordinate checkpoint; "
            "legacy AE forced deployment is invalid by construction"
        )

    with torch.no_grad():
        model.action_mean.copy_(identity_mean)
        model.action_std.copy_(identity_std)
    model.reinitialize_critic_heads(
        list(range(model.n_critics)),
        seed=_stable_seed(args.seed, "ae_native_critic_init", 0, 0),
    )
    if model.guide is not None:
        _reset_module_parameters(
            model.guide,
            seed=_stable_seed(args.seed, "ae_native_guide_init", 0, 0),
        )
    log.warning(
        "Initialized AE critic%s in identity Molmo-native coordinates; "
        "legacy raw-action critic weights were not reused",
        "/guide" if model.guide is not None else "",
    )
    return True


@dataclass(frozen=True)
class GateStatus:
    """Independent actor-vs-reference and guide-vs-actor gate diagnostics."""

    actor_ready: bool
    guide_ready: bool
    deploy_actor: bool
    deploy_guide: bool
    actor_lcb: float
    actor_advantage: float
    actor_sensitivity: float
    guide_lcb: float
    guide_advantage: float
    guide_sensitivity: float
    critic_health: bool
    critic_health_fields: dict[str, float]
    actor_block_reason: str = ""
    guide_block_reason: str = ""
    empirical_lcb: float = float("-inf")
    empirical_ready: bool = False
    empirical_block_reason: str = ""
    empirical_actor_episodes: float = 0.0
    empirical_ref_episodes: float = 0.0
    empirical_actor_sr: float = 0.0
    empirical_ref_sr: float = 0.0

    @property
    def would_enable(self) -> bool:
        """Backward-compatible actor-gate view."""
        return self.actor_ready

    @property
    def paired_lcb(self) -> float:
        return self.actor_lcb

    @property
    def q_min_advantage(self) -> float:
        return self.actor_advantage

    @property
    def sensitivity(self) -> float:
        return self.actor_sensitivity


@dataclass(frozen=True)
class UpdateStatus:
    skipped_reason: str
    stop_reason: str
    rounds: int
    critic_updates: int
    actor_updates: int
    guide_updates: int
    token_updates: int
    elapsed_sec: float


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


def _parse_critic_head_indices(value: str, n_critics: int) -> list[int]:
    indices: set[int] = set()
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if not 0 <= index < int(n_critics):
            raise ValueError(
                f"critic recovery index {index} outside [0, {n_critics})"
            )
        indices.add(index)
    return sorted(indices)


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
        root_seed: int = 0,
        actor_mixture_prob: float = 0.0,
        guide_on_reference: bool = False,
        residual_clip: float | None = None,
        always_collect_actor: bool = False,
        actor_bc_episodes: int = 50,
        always_collect_after_episodes: int | None = None,
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
        self.actor_mixture_prob = float(max(0.0, min(1.0, actor_mixture_prob)))
        self.guide_on_reference = bool(guide_on_reference)
        self.residual_clip = (
            None if residual_clip is None else float(residual_clip)
        )
        # V16 / RLT paper Alg.1: after BC warmup, always collect from π_θ.
        # V17: prefer episode-level mixture; delay always-collect via
        # always_collect_after_episodes (defaults to actor_bc_episodes).
        self.always_collect_actor = bool(always_collect_actor)
        self.actor_bc_episodes = int(max(0, actor_bc_episodes))
        self.always_collect_after_episodes = (
            self.actor_bc_episodes
            if always_collect_after_episodes is None
            else int(max(0, always_collect_after_episodes))
        )
        self.collect_episode_index = 0
        self.episode_used_actor = False
        self.episode_mixture_draws = 0
        self.episode_actor_chunks = 0
        self.episode_collect_policy = "reference"
        self.episode_mixture_use_actor: bool | None = None
        self.fatal_error: RLTFeatureError | None = None
        self.root_seed = int(root_seed)
        self.episode_counter = 0
        self.decision_counter = 0
        self.episode_decision_counter = 0
        self._role_counters = {role: 0 for role in _RNG_ROLES}
        self._rng_streams = {
            role: np.random.default_rng(
                _stable_seed(self.root_seed, role, 0, 0)
            )
            for role in _RNG_ROLES
        }
        # Deprecated test/extension compatibility; V14 code never consumes it.
        self._rng = self._rng_streams["exploration"]
        self._init_vla_prefetch()
        self._clear_episode()
        super().__init__(exp_config)
        self.chunk_size = int(model.chunk_size)
        if self.chunk_size != CHUNK_SIZE:
            raise ValueError(
                f"RLT policy requires chunk_size={CHUNK_SIZE}, checkpoint has {self.chunk_size}"
            )

    def _init_vla_prefetch(self) -> None:
        """Optional background /act to hide VLA latency behind late-chunk MuJoCo."""
        enabled_raw = os.environ.get("RLT_VLA_PREFETCH", "0").strip().lower()
        self.vla_prefetch_enabled = enabled_raw in {"1", "true", "yes", "on"}
        self.vla_prefetch_k = max(1, int(os.environ.get("RLT_VLA_PREFETCH_K", "2")))
        require_raw = os.environ.get(
            "RLT_VLA_PREFETCH_REQUIRE_OBS_MATCH", "1"
        ).strip().lower()
        self.vla_prefetch_require_obs_match = require_raw in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._prefetch_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="vla_prefetch")
            if self.vla_prefetch_enabled
            else None
        )
        self._prefetch_future: Future[dict[str, Any]] | None = None
        self._prefetch_fingerprint: str | None = None
        self._prefetch_source_seed: int | None = None
        self.vla_prefetch_hit = 0
        self.vla_prefetch_miss = 0
        self.vla_prefetch_wait_ms = 0.0
        self.vla_prefetch_discarded = 0

    def _reset_vla_prefetch_episode_stats(self) -> None:
        self.vla_prefetch_hit = 0
        self.vla_prefetch_miss = 0
        self.vla_prefetch_wait_ms = 0.0
        self.vla_prefetch_discarded = 0
        self._cancel_vla_prefetch(discard=True)

    def _cancel_vla_prefetch(self, *, discard: bool) -> None:
        future = self._prefetch_future
        self._prefetch_future = None
        self._prefetch_fingerprint = None
        self._prefetch_source_seed = None
        if future is None:
            return
        if discard and future.done():
            try:
                future.result(timeout=0)
            except Exception:  # noqa: BLE001
                pass
            self.vla_prefetch_discarded += 1

    def _obs_fingerprint(self, model_input: dict[str, Any]) -> str:
        external = np.ascontiguousarray(
            np.asarray(model_input["external_cam"], dtype=np.uint8)
        )
        wrist = np.ascontiguousarray(
            np.asarray(model_input["wrist_cam"], dtype=np.uint8)
        )
        state = np.ascontiguousarray(
            np.asarray(model_input["state"], dtype=np.float32)
        )
        digest = hashlib.sha1()
        digest.update(external.tobytes())
        digest.update(wrist.tobytes())
        digest.update(state.tobytes())
        digest.update(str(model_input.get("instruction", "")).encode("utf-8"))
        return digest.hexdigest()

    def _snapshot_model_input(self, model_input: dict[str, Any]) -> dict[str, Any]:
        return {
            "external_cam": np.asarray(model_input["external_cam"], dtype=np.uint8).copy(),
            "wrist_cam": np.asarray(model_input["wrist_cam"], dtype=np.uint8).copy(),
            "instruction": str(model_input.get("instruction", "")),
            "state": np.asarray(model_input["state"], dtype=np.float32).copy(),
            "timestamp": float(model_input.get("timestamp", time.time())),
        }

    def _maybe_start_vla_prefetch(
        self,
        model_input: dict[str, Any],
        *,
        source_seed: int | None = None,
    ) -> None:
        if not self.vla_prefetch_enabled or self._prefetch_executor is None:
            return
        # AE in-process path ties reference RNG to decision_id; skip speculative
        # calls rather than risk a seed mismatch under the backend lock.
        if self.ae_backend is not None:
            return
        if self.actions_buffer is None:
            return
        if self.current_buffer_index < max(0, self.chunk_size - self.vla_prefetch_k):
            return
        fingerprint = self._obs_fingerprint(model_input)
        if (
            self._prefetch_future is not None
            and self._prefetch_fingerprint == fingerprint
        ):
            return
        if self._prefetch_future is not None and not self._prefetch_future.done():
            # Keep the in-flight request so it can finish during MuJoCo; avoid
            # stacking calls behind serve.py's global lock.
            return
        if self._prefetch_future is not None and self._prefetch_future.done():
            self._cancel_vla_prefetch(discard=True)
        snapshot = self._snapshot_model_input(model_input)
        self._prefetch_fingerprint = fingerprint
        self._prefetch_source_seed = source_seed
        self._prefetch_future = self._prefetch_executor.submit(
            self._post_act,
            snapshot,
            source_seed=source_seed,
        )

    def _consume_vla_prefetch(
        self,
        model_input: dict[str, Any],
        *,
        source_seed: int | None,
    ) -> dict[str, Any] | None:
        if not self.vla_prefetch_enabled:
            return None
        future = self._prefetch_future
        if future is None:
            self.vla_prefetch_miss += 1
            return None
        fingerprint = self._obs_fingerprint(model_input)
        seed_ok = source_seed == self._prefetch_source_seed
        match_ok = (not self.vla_prefetch_require_obs_match) or (
            fingerprint == self._prefetch_fingerprint and seed_ok
        )
        if not match_ok:
            # If the stale request already finished, drop it without blocking
            # the correct sync /act. If it is still running, wait so we do not
            # contend on the server lock, then discard.
            wait_start = time.perf_counter()
            if not future.done():
                try:
                    future.result(timeout=self.request_timeout_sec)
                except Exception:  # noqa: BLE001
                    pass
                self.vla_prefetch_wait_ms += (
                    time.perf_counter() - wait_start
                ) * 1000.0
            self._cancel_vla_prefetch(discard=True)
            self.vla_prefetch_miss += 1
            return None
        wait_start = time.perf_counter()
        try:
            response = future.result(timeout=self.request_timeout_sec)
        except Exception:  # noqa: BLE001
            self._cancel_vla_prefetch(discard=False)
            self.vla_prefetch_miss += 1
            self.vla_prefetch_wait_ms += (time.perf_counter() - wait_start) * 1000.0
            return None
        self.vla_prefetch_wait_ms += (time.perf_counter() - wait_start) * 1000.0
        self._prefetch_future = None
        self._prefetch_fingerprint = None
        self._prefetch_source_seed = None
        self.vla_prefetch_hit += 1
        return response

    @property
    def enable_rlt(self) -> bool:
        """Deprecated compatibility view of deploy_actor."""
        return self.deploy_actor

    @enable_rlt.setter
    def enable_rlt(self, enabled: bool) -> None:
        self.deploy_actor = bool(enabled)
        self.deploy_guide = bool(enabled) and self.use_cf_guide

    def _clear_episode(self) -> None:
        self._cancel_vla_prefetch(discard=True)
        self.ep_zs: list[np.ndarray] = []
        self.ep_proprios: list[np.ndarray] = []
        self.ep_references: list[np.ndarray] = []
        self.ep_executed: list[np.ndarray] = []
        self.ep_actor_shadow: list[np.ndarray] = []
        self.ep_action_counts: list[int] = []
        self.ep_tokens: list[np.ndarray | None] = []
        self.ep_token_masks: list[np.ndarray | None] = []
        self.ep_z_sources: list[str] = []
        self.ep_external_cams: list[np.ndarray] = []
        self.ep_wrist_cams: list[np.ndarray] = []
        self.ep_instructions: list[str] = []
        self.ep_full_references: list[np.ndarray] = []
        self.ep_full_executed: list[np.ndarray] = []
        self.ep_sources_native: list[np.ndarray] = []
        self.ep_reference_raw: list[np.ndarray] = []
        self.ep_executed_raw: list[np.ndarray] = []
        self.ep_reference_raw_full: list[np.ndarray] = []
        self.ep_executed_raw_full: list[np.ndarray] = []
        self.episode_used_actor = False
        self.episode_mixture_draws = 0
        self.episode_actor_chunks = 0
        self.episode_collect_policy = "reference"
        self.episode_mixture_use_actor = None
        self.episode_residual_sq_sum = 0.0
        self.episode_residual_count = 0

    def reset(self) -> None:
        super().reset()
        self.fatal_error = None
        self._reset_vla_prefetch_episode_stats()
        self._clear_episode()
        self.episode_mixture_use_actor = None

    def _ensure_rng_streams(self) -> None:
        if not hasattr(self, "root_seed"):
            self.root_seed = 0
        if not hasattr(self, "episode_counter"):
            self.episode_counter = 0
        if not hasattr(self, "decision_counter"):
            self.decision_counter = 0
        if not hasattr(self, "episode_decision_counter"):
            self.episode_decision_counter = 0
        if not hasattr(self, "_role_counters"):
            self._role_counters = {role: 0 for role in _RNG_ROLES}
        if not hasattr(self, "_rng_streams"):
            self._rng_streams = {
                role: np.random.default_rng(
                    _stable_seed(self.root_seed, role, 0, 0)
                )
                for role in _RNG_ROLES
            }

    def begin_episode(self, episode_id: int) -> None:
        self._ensure_rng_streams()
        self.episode_counter = int(episode_id)
        self.episode_decision_counter = 0
        self.episode_collect_policy = "reference"
        # Draw mixture once per episode so empirical labels stay coherent
        # (V15 per-chunk mixture broke empirical_insufficient_episodes).
        self.episode_mixture_use_actor = None
        if (
            self.actor_mode == "rlt"
            and self.actor_mixture_prob > 0.0
            and not bool(getattr(self, "eval_force_reference", False))
            and not bool(getattr(self, "guide_on_reference", False))
        ):
            mixture_seed = self.next_seed("actor_mixture", episode_id=episode_id)
            mixture_rng = np.random.default_rng(mixture_seed)
            self.episode_mixture_use_actor = bool(
                mixture_rng.random() < float(self.actor_mixture_prob)
            )
            self.episode_mixture_draws = 1
            if self.episode_mixture_use_actor:
                self.episode_collect_policy = "mixture_actor"

    def next_seed(
        self,
        role: str,
        *,
        episode_id: int | None = None,
        decision_id: int | None = None,
    ) -> int:
        self._ensure_rng_streams()
        if role not in self._rng_streams:
            raise KeyError(f"Unknown RLTOnlinePolicy RNG role {role!r}")
        role_counter = int(self._role_counters[role])
        nonce = int(
            self._rng_streams[role].integers(
                0,
                np.iinfo(np.int64).max,
                dtype=np.int64,
            )
        )
        seed = _stable_seed(
            self.root_seed,
            role,
            self.episode_counter if episode_id is None else int(episode_id),
            role_counter if decision_id is None else int(decision_id),
            nonce,
        )
        self._role_counters[role] = role_counter + 1
        return seed

    def rng_state_dict(self) -> dict[str, Any]:
        self._ensure_rng_streams()
        return {
            "schema_version": _V14_ONLINE_SCHEMA,
            "root_seed": int(self.root_seed),
            "episode_counter": int(self.episode_counter),
            "decision_counter": int(self.decision_counter),
            "episode_decision_counter": int(self.episode_decision_counter),
            "role_counters": dict(self._role_counters),
            "streams": {
                role: _json_safe(generator.bit_generator.state)
                for role, generator in self._rng_streams.items()
            },
        }

    def load_rng_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != _V14_ONLINE_SCHEMA:
            raise RuntimeError("Cannot restore non-V14 online policy RNG state")
        saved_root = int(state["root_seed"])
        if saved_root != int(self.root_seed):
            raise RuntimeError(
                f"Resume seed mismatch: checkpoint={saved_root}, CLI={self.root_seed}"
            )
        self.episode_counter = int(state["episode_counter"])
        self.decision_counter = int(state["decision_counter"])
        self.episode_decision_counter = int(
            state.get("episode_decision_counter", 0)
        )
        role_counters = dict(state.get("role_counters") or {})
        stream_states = dict(state.get("streams") or {})
        if set(stream_states) != set(_RNG_ROLES):
            raise RuntimeError(
                "V14 policy RNG state is incomplete: "
                f"roles={sorted(stream_states)}"
            )
        self._role_counters = {
            role: int(role_counters.get(role, 0)) for role in _RNG_ROLES
        }
        self._rng_streams = {}
        for role in _RNG_ROLES:
            generator = np.random.default_rng()
            generator.bit_generator.state = stream_states[role]
            self._rng_streams[role] = generator
        self._rng = self._rng_streams["exploration"]

    def prepare_model(self, model_name: str | None = None) -> None:
        if self.ae_backend is not None:
            log.info("In-process Molmo AE backend; skipping HTTP health check")
            return
        super().prepare_model(model_name)

    def _post_act(
        self,
        model_input: dict[str, Any],
        *,
        source_seed: int | None = None,
    ) -> dict[str, Any]:
        payload = {
            "external_cam": np.asarray(model_input["external_cam"], dtype=np.uint8),
            "wrist_cam": np.asarray(model_input["wrist_cam"], dtype=np.uint8),
            "instruction": model_input["instruction"],
            "state": np.asarray(model_input["state"], dtype=np.float32),
            "timestamp": model_input.get("timestamp", time.time()),
        }
        if self.ae_backend is not None:
            if hasattr(self.ae_backend, "action_contract"):
                _validate_ae_contract(self.ae_backend)
            # Replay/reference actions must remain the frozen base-AE output.
            out = self.ae_backend.predict_reference(
                payload["external_cam"],
                payload["wrist_cam"],
                str(payload["instruction"]),
                payload["state"],
                source_seed=source_seed,
            )
            body = {"actions": out["actions"]}
            for key in (
                "features",
                "token_features",
                "token_attention_mask",
                "z_rl",
                "source_native",
                "source_seed",
                "actions_raw_full",
                "actions_native_full",
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
        self._ensure_rng_streams()
        decision_id = int(self.episode_decision_counter)
        use_molmo_ae = (
            self.ae_backend is not None
            and getattr(self.rlt_model, "v_source", "rlt") == "molmo_ae"
        )
        source_seed = None
        if self.ae_backend is not None:
            source_seed = self.next_seed(
                "molmo_ae_source",
                decision_id=decision_id,
            )
        self.decision_counter += 1
        self.episode_decision_counter += 1
        response = self._consume_vla_prefetch(
            model_input,
            source_seed=source_seed,
        )
        if response is None:
            response = self._post_act(model_input, source_seed=source_seed)
        actions_raw = np.asarray(response.get("actions"), dtype=np.float32)
        if actions_raw.ndim == 1:
            actions_raw = actions_raw.reshape(1, -1)
        if actions_raw.ndim != 2 or actions_raw.shape[1] != ACTION_DIM:
            raise ValueError(
                f"Invalid MolmoAct2 action horizon shape {actions_raw.shape}"
            )
        if actions_raw.shape[0] < self.chunk_size:
            raise ValueError(
                f"MolmoAct2 returned only {actions_raw.shape[0]} actions; "
                f"RLT requires at least {self.chunk_size}"
            )
        if not np.isfinite(actions_raw).all():
            raise FloatingPointError("MolmoAct2 returned non-finite actions")

        strict_ae_contract = (
            use_molmo_ae
            and self.ae_backend is not None
            and hasattr(self.ae_backend, "action_contract")
        )
        native_start = 0
        reference_raw = actions_raw[: self.chunk_size].copy()
        reference = reference_raw.copy()
        full_reference_raw = np.zeros(
            (_AE_NATIVE_HORIZON, ACTION_DIM),
            dtype=np.float32,
        )
        full_reference_raw[: self.chunk_size] = reference_raw
        full_reference_native = full_reference_raw.copy()
        source_native = np.zeros(
            (_AE_NATIVE_HORIZON, _AE_PADDED_ACTION_DIM),
            dtype=np.float32,
        )
        if use_molmo_ae:
            required = (
                "actions_raw_full",
                "actions_native_full",
                "source_native",
            )
            missing = [key for key in required if key not in response]
            if missing and strict_ae_contract:
                raise RuntimeError(
                    f"V14 AE reference prediction omitted {missing}"
                )
            if not missing:
                full_reference_raw = np.asarray(
                    response["actions_raw_full"],
                    dtype=np.float32,
                )
                full_reference_native = np.asarray(
                    response["actions_native_full"],
                    dtype=np.float32,
                )
                source_native = np.asarray(
                    response["source_native"],
                    dtype=np.float32,
                )
            expected_full = (_AE_NATIVE_HORIZON, ACTION_DIM)
            expected_source = (
                _AE_NATIVE_HORIZON,
                _AE_PADDED_ACTION_DIM,
            )
            if (
                full_reference_raw.shape != expected_full
                or full_reference_native.shape != expected_full
                or source_native.shape != expected_source
            ):
                raise RuntimeError(
                    "V14 AE prediction shapes must be raw/native "
                    f"{expected_full} and source {expected_source}; got "
                    f"{full_reference_raw.shape}, "
                    f"{full_reference_native.shape}, {source_native.shape}"
                )
            if not (
                np.isfinite(full_reference_raw).all()
                and np.isfinite(full_reference_native).all()
                and np.isfinite(source_native).all()
            ):
                raise FloatingPointError(
                    "V14 AE reference prediction contained non-finite coordinates"
                )
            if strict_ae_contract and int(response.get("source_seed", -1)) != int(
                source_seed
            ):
                raise RuntimeError("V14 AE reference did not preserve source_seed")
            if strict_ae_contract:
                padded_abs_max = float(
                    np.max(np.abs(source_native[:, ACTION_DIM:]))
                )
                if padded_abs_max > 1e-6:
                    raise RuntimeError(
                        "V14 AE source has non-zero padded coordinates: "
                        f"max_abs={padded_abs_max}"
                    )
                contract = _validate_ae_contract(self.ae_backend)
                native_start = max(0, int(contract["n_obs_steps"]) - 1)
            native_stop = native_start + self.chunk_size
            reference = full_reference_native[native_start:native_stop].copy()
            if reference.shape != (self.chunk_size, ACTION_DIM):
                raise RuntimeError(
                    f"Invalid native AE reference chunk shape {reference.shape}"
                )

        proprio = np.asarray(model_input["state"], dtype=np.float32).reshape(-1)
        if proprio.shape != (self.rlt_model.proprio_dim,):
            raise ValueError(
                f"Invalid proprio shape {proprio.shape}; "
                f"expected ({self.rlt_model.proprio_dim},)"
            )
        tokens, token_mask = self._response_tokens(response)
        z, z_source = self._response_z(response, tokens, token_mask)

        deployed_raw = reference_raw.copy()
        deployed = reference.copy()
        full_executed_raw = full_reference_raw.copy()
        full_executed_native = full_reference_native.copy()
        use_actor = bool(self.deploy_actor and self.actor_mode == "rlt")
        mixture_draw = False
        paper_collect = False
        always_after = int(
            getattr(
                self,
                "always_collect_after_episodes",
                getattr(self, "actor_bc_episodes", 50),
            )
        )
        if (
            not use_actor
            and bool(getattr(self, "always_collect_actor", False))
            and self.actor_mode == "rlt"
            and not bool(getattr(self, "guide_on_reference", False))
            and not bool(getattr(self, "eval_force_reference", False))
            and int(getattr(self, "collect_episode_index", 0)) >= always_after
        ):
            # RLT paper Alg.1: after delay, always execute π_θ(·|x, ã).
            use_actor = True
            paper_collect = True
            self.episode_collect_policy = "actor"
        elif (
            not use_actor
            and self.actor_mode == "rlt"
            and self.actor_mixture_prob > 0.0
            and not bool(getattr(self, "eval_force_reference", False))
            and not bool(getattr(self, "guide_on_reference", False))
        ):
            # Episode-level mixture drawn in begin_episode (not per-chunk).
            if self.episode_mixture_use_actor is None:
                mixture_seed = self.next_seed(
                    "actor_mixture",
                    episode_id=int(getattr(self, "episode_counter", 0)),
                )
                mixture_rng = np.random.default_rng(mixture_seed)
                self.episode_mixture_use_actor = bool(
                    mixture_rng.random() < float(self.actor_mixture_prob)
                )
                self.episode_mixture_draws = 1
            mixture_draw = bool(self.episode_mixture_use_actor)
            use_actor = mixture_draw
            if mixture_draw:
                self.episode_collect_policy = "mixture_actor"
            else:
                self.episode_collect_policy = "reference"
        use_guide_only = bool(
            self.deploy_guide
            and self.guide_on_reference
            and not use_actor
            and self.use_cf_guide
        )
        if use_actor:
            self.episode_used_actor = True
            self.episode_actor_chunks += 1
        if use_actor or use_guide_only:
            z_t = torch.from_numpy(z).unsqueeze(0).to(self.rlt_device)
            proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.rlt_device)
            with torch.inference_mode():
                state = self.rlt_model.encode_state_from_z(z_t, proprio_t)
                if use_molmo_ae and use_actor:
                    # Gate on: trainable AE, with guide only when requested.
                    predicted = self.ae_backend.predict(
                        np.asarray(model_input["external_cam"], dtype=np.uint8),
                        np.asarray(model_input["wrist_cam"], dtype=np.uint8),
                        str(model_input["instruction"]),
                        proprio,
                        apply_guide=self.deploy_guide,
                        rlt_state=state if self.deploy_guide else None,
                        source_seed=source_seed,
                        source_native=source_native,
                    )
                    candidate_required = (
                        "actions_raw_full",
                        "actions_native_full",
                        "source_native",
                        "source_seed",
                    )
                    candidate_missing = [
                        key for key in candidate_required if key not in predicted
                    ]
                    if candidate_missing and strict_ae_contract:
                        raise RuntimeError(
                            "V14 AE actor prediction omitted "
                            f"{candidate_missing}"
                        )
                    deployed_raw = np.asarray(
                        predicted["actions"],
                        dtype=np.float32,
                    )[
                        : self.chunk_size
                    ].copy()
                    full_executed_raw = np.asarray(
                        predicted.get("actions_raw_full", full_reference_raw),
                        dtype=np.float32,
                    )
                    full_executed_native = np.asarray(
                        predicted.get(
                            "actions_native_full",
                            full_reference_native,
                        ),
                        dtype=np.float32,
                    )
                    predicted_source = np.asarray(
                        predicted.get("source_native", source_native),
                        dtype=np.float32,
                    )
                    if strict_ae_contract and (
                        full_executed_raw.shape
                        != (_AE_NATIVE_HORIZON, ACTION_DIM)
                        or full_executed_native.shape
                        != (_AE_NATIVE_HORIZON, ACTION_DIM)
                        or predicted_source.shape
                        != (_AE_NATIVE_HORIZON, _AE_PADDED_ACTION_DIM)
                        or int(predicted.get("source_seed", -1))
                        != int(source_seed)
                        or not np.array_equal(predicted_source, source_native)
                    ):
                        raise RuntimeError(
                            "V14 AE actor did not preserve the paired source contract"
                        )
                    if not (
                        np.isfinite(full_executed_raw).all()
                        and np.isfinite(full_executed_native).all()
                    ):
                        raise FloatingPointError(
                            "V14 AE actor returned non-finite full actions"
                        )
                    deployed = full_executed_native[
                        native_start : native_start + self.chunk_size
                    ].copy()
                else:
                    reference_t = torch.from_numpy(reference).unsqueeze(0).to(
                        self.rlt_device
                    )
                    reference_n = self.rlt_model.normalize_action(reference_t)
                    if use_guide_only:
                        if self.rlt_model.guide is None:
                            raise RuntimeError(
                                "guide_on_reference deploy requested without a guide"
                            )
                        _guided, g_delta = self.rlt_model.guide.guide(
                            state,
                            reference_n,
                            actor_delta=torch.zeros_like(reference_n),
                        )
                        deployed_n = reference_n + g_delta
                    else:
                        deployed_n, _ = self.rlt_model.actor_chunk(
                            state,
                            reference_n,
                            # Paper collect uses stochastic π_θ; gated deploy stays deterministic.
                            deterministic=not paper_collect,
                            apply_guide=bool(
                                self.deploy_guide and not mixture_draw and not paper_collect
                            ),
                        )
                    if self.residual_clip is not None and float(self.residual_clip) > 0.0:
                        delta = (deployed_n - reference_n).clamp(
                            -float(self.residual_clip),
                            float(self.residual_clip),
                        )
                        deployed_n = reference_n + delta
                    deployed_raw = self.rlt_model.denormalize_action(deployed_n)[
                        0
                    ].cpu().numpy()
                    deployed = deployed_raw
            if (
                deployed.shape != reference.shape
                or deployed_raw.shape != reference_raw.shape
                or not np.isfinite(deployed).all()
                or not np.isfinite(deployed_raw).all()
            ):
                raise FloatingPointError(
                    f"RLT/AE actor returned invalid chunk shape={deployed.shape}"
                )
        # Always compute a shadow actor action for diagnostics when an RLT actor exists.
        shadow = reference.copy()
        if self.actor_mode == "rlt" and self.rlt_model is not None and not use_molmo_ae:
            try:
                z_t = torch.from_numpy(z).unsqueeze(0).to(self.rlt_device)
                proprio_t = torch.from_numpy(proprio).unsqueeze(0).to(self.rlt_device)
                reference_t = torch.from_numpy(reference).unsqueeze(0).to(self.rlt_device)
                with torch.inference_mode():
                    state = self.rlt_model.encode_state_from_z(z_t, proprio_t)
                    reference_n = self.rlt_model.normalize_action(reference_t)
                    shadow_n, _ = self.rlt_model.actor_chunk(
                        state,
                        reference_n,
                        deterministic=True,
                        apply_guide=False,
                    )
                    if self.residual_clip is not None and float(self.residual_clip) > 0.0:
                        delta = (shadow_n - reference_n).clamp(
                            -float(self.residual_clip),
                            float(self.residual_clip),
                        )
                        shadow_n = reference_n + delta
                    shadow = self.rlt_model.denormalize_action(shadow_n)[
                        0
                    ].cpu().numpy().astype(np.float32)
                    residual = shadow_n - reference_n
                    self.episode_residual_sq_sum += float(
                        residual.detach().float().pow(2).mean().cpu()
                    )
                    self.episode_residual_count += 1
            except Exception:  # noqa: BLE001
                shadow = reference.copy()
        # Exploration in *normalized* action space. v12: smaller default and no
        # gate-off boost (v11's 0.05×1.5 drove residual_rms~0.07–0.15 and SR tax).
        explore_std = float(self.explore_residual_std)
        if self.deploy_actor or use_actor:
            explore_std = float(self.explore_deploy_std)
        elif self.explore_warmup_mult != 1.0:
            explore_std = explore_std * float(self.explore_warmup_mult)
        # Mixture / paper-collect actor chunks rely on π_θ stochasticity, not additive noise.
        if mixture_draw or paper_collect:
            explore_std = 0.0
        if (
            explore_std > 0.0
            and self.actor_mode == "rlt"
            and self.rlt_model is not None
        ):
            exploration_seed = self.next_seed(
                "exploration",
                decision_id=decision_id,
            )
            exploration_rng = np.random.default_rng(exploration_seed)
            noise = exploration_rng.standard_normal(
                deployed.shape,
                dtype=np.float32,
            )
            if use_molmo_ae:
                deployed = (
                    deployed + noise * np.float32(explore_std)
                ).astype(np.float32)
                if hasattr(self.ae_backend, "unnormalize_actions"):
                    deployed_raw = np.asarray(
                        self.ae_backend.unnormalize_actions(
                            torch.from_numpy(deployed).to(self.rlt_device)
                        ),
                        dtype=np.float32,
                    )
                else:
                    deployed_raw = deployed.copy()
            else:
                a_t = torch.from_numpy(deployed).unsqueeze(0).to(self.rlt_device)
                with torch.inference_mode():
                    a_n = self.rlt_model.normalize_action(a_t)[0]
                    a_n = a_n + torch.from_numpy(noise).to(a_n) * explore_std
                    deployed_raw = (
                        self.rlt_model.denormalize_action(a_n)
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                deployed = deployed_raw

        if use_molmo_ae:
            native_stop = native_start + self.chunk_size
            full_executed_native = full_executed_native.copy()
            full_executed_raw = full_executed_raw.copy()
            full_executed_native[native_start:native_stop] = deployed
            full_executed_raw[native_start:native_stop] = deployed_raw
        self.actions_buffer = [row.copy() for row in deployed_raw]
        self.current_buffer_index = 0
        self.ep_zs.append(z.copy())
        self.ep_proprios.append(proprio.copy())
        self.ep_references.append(reference.copy())
        self.ep_executed.append(np.asarray(deployed, dtype=np.float32).copy())
        self.ep_actor_shadow.append(np.asarray(shadow, dtype=np.float32).copy())
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
        self.ep_full_references.append(full_reference_raw.copy())
        self.ep_full_executed.append(full_executed_raw.copy())
        self.ep_sources_native.append(source_native.copy())
        self.ep_reference_raw.append(reference_raw.copy())
        self.ep_executed_raw.append(deployed_raw.copy())
        self.ep_reference_raw_full.append(full_reference_raw.copy())
        self.ep_executed_raw_full.append(full_executed_raw.copy())

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
        # Hide the next /act behind the remaining MuJoCo steps of this chunk.
        self._maybe_start_vla_prefetch(model_input)
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
        source_values = (
            np.concatenate(
                [value.reshape(-1) for value in self.ep_sources_native],
                axis=0,
            )
            if self.ep_sources_native
            else np.zeros(0, dtype=np.float32)
        )
        source_padded = (
            np.concatenate(
                [
                    value[:, ACTION_DIM:].reshape(-1)
                    for value in self.ep_sources_native
                ],
                axis=0,
            )
            if self.ep_sources_native
            else np.zeros(0, dtype=np.float32)
        )
        raw_values = (
            np.concatenate(
                [value.reshape(-1) for value in self.ep_executed_raw],
                axis=0,
            )
            if self.ep_executed_raw
            else np.zeros(0, dtype=np.float32)
        )
        native_values = (
            np.concatenate(
                [value.reshape(-1) for value in self.ep_executed],
                axis=0,
            )
            if self.ep_executed
            else np.zeros(0, dtype=np.float32)
        )
        coordinate_diagnostics = {
            "ae_source_native_rms": (
                float(np.sqrt(np.mean(np.square(source_values))))
                if source_values.size
                else 0.0
            ),
            "ae_source_padded_abs_max": (
                float(np.max(np.abs(source_padded)))
                if source_padded.size
                else 0.0
            ),
            "ae_executed_native_rms": (
                float(np.sqrt(np.mean(np.square(native_values))))
                if native_values.size
                else 0.0
            ),
            "ae_executed_raw_rms": (
                float(np.sqrt(np.mean(np.square(raw_values))))
                if raw_values.size
                else 0.0
            ),
        }
        trajectory = {
            "zs": self.ep_zs,
            "proprios": self.ep_proprios,
            "references": self.ep_references,
            "executed": self.ep_executed,
            "actor_shadow": list(self.ep_actor_shadow),
            "rewards": rewards,
            "masks": masks,
            "token_batches": token_batches,
            "z_sources": list(self.ep_z_sources),
            "n_steps": n_steps,
            "residual_rms": residual_rms,
            "actor_ref_mse": (
                float(self.episode_residual_sq_sum / max(self.episode_residual_count, 1))
                if self.episode_residual_count
                else 0.0
            ),
            "episode_used_actor": bool(self.episode_used_actor),
            "episode_actor_chunks": int(self.episode_actor_chunks),
            "episode_mixture_draws": int(self.episode_mixture_draws),
            "episode_collect_policy": str(
                getattr(self, "episode_collect_policy", "reference")
            ),
            "vla_prefetch_hit": int(getattr(self, "vla_prefetch_hit", 0)),
            "vla_prefetch_miss": int(getattr(self, "vla_prefetch_miss", 0)),
            "vla_prefetch_wait_ms": float(getattr(self, "vla_prefetch_wait_ms", 0.0)),
            "vla_prefetch_discarded": int(getattr(self, "vla_prefetch_discarded", 0)),
            "external_cams": list(self.ep_external_cams),
            "wrist_cams": list(self.ep_wrist_cams),
            "instructions": list(self.ep_instructions),
            "full_references": list(self.ep_full_references),
            "full_executed": list(self.ep_full_executed),
            "sources_native": list(self.ep_sources_native),
            "full_reference_actions_raw": [
                value.copy() for value in self.ep_reference_raw_full
            ],
            "full_executed_actions_raw": [
                value.copy() for value in self.ep_executed_raw_full
            ],
            "coordinate_diagnostics": coordinate_diagnostics,
        }
        self.vla_prefetch_hit = 0
        self.vla_prefetch_miss = 0
        self.vla_prefetch_wait_ms = 0.0
        self.vla_prefetch_discarded = 0
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


def _require_v14_ae_metadata(
    metadata: dict[str, Any],
    *,
    label: str,
    allow_legacy: bool,
) -> None:
    version = int(metadata.get("online_schema_version", -1) or -1)
    if version == _V14_ONLINE_SCHEMA:
        return
    message = (
        f"{label} is legacy/non-V14 (online_schema_version={version}); "
        "native source/action replay cannot be proven correct"
    )
    if not allow_legacy:
        raise RuntimeError(
            message + ". Pass --allow_legacy_ae_resume only for explicit migration."
        )
    log.warning("%s; continuing because --allow_legacy_ae_resume was set", message)


def _require_matching_ae_contract(
    metadata: dict[str, Any],
    expected: dict[str, int],
    *,
    label: str,
    allow_legacy: bool,
) -> None:
    saved = dict(metadata.get("ae_action_contract") or {})
    keys = (
        "action_dim",
        "max_action_dim",
        "action_horizon",
        "n_action_steps",
        "n_obs_steps",
    )
    mismatches = {
        key: (saved.get(key), expected[key])
        for key in keys
        if int(saved.get(key, -1)) != int(expected[key])
    }
    if not mismatches:
        return
    message = f"{label} AE action contract mismatch: {mismatches}"
    if not allow_legacy:
        raise RuntimeError(
            message + ". Pass --allow_legacy_ae_resume only for explicit migration."
        )
    log.warning("%s; continuing because --allow_legacy_ae_resume was set", message)


def _validate_ae_replay_resume(path: Path, *, allow_legacy: bool) -> None:
    required = {
        "full_reference_actions",
        "full_executed_actions",
        "next_full_reference_actions",
        "source_native",
        "next_source_native",
        "rng_state_json",
    }
    with np.load(path, allow_pickle=False) as data:
        missing = sorted(required - set(data.files))
        horizon = (
            int(data["full_action_horizon"])
            if "full_action_horizon" in data
            else -1
        )
        padded_dim = (
            int(data["padded_action_dim"])
            if "padded_action_dim" in data
            else -1
        )
        full_shapes = {
            key: tuple(data[key].shape[1:])
            for key in (
                "full_reference_actions",
                "full_executed_actions",
                "next_full_reference_actions",
            )
            if key in data
        }
        source_shapes = {
            key: tuple(data[key].shape[1:])
            for key in ("source_native", "next_source_native")
            if key in data
        }
        source_padded_abs_max = max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(data[key], dtype=np.float32)[
                                ..., ACTION_DIM:
                            ]
                        ),
                        initial=0.0,
                    )
                )
                for key in ("source_native", "next_source_native")
                if key in data
            ),
            default=0.0,
        )
    expected_full_shape = (_AE_NATIVE_HORIZON, ACTION_DIM)
    expected_source_shape = (
        _AE_NATIVE_HORIZON,
        _AE_PADDED_ACTION_DIM,
    )
    shapes_match = all(
        shape == expected_full_shape for shape in full_shapes.values()
    ) and all(
        shape == expected_source_shape for shape in source_shapes.values()
    )
    if (
        not missing
        and horizon == _AE_NATIVE_HORIZON
        and padded_dim == _AE_PADDED_ACTION_DIM
        and shapes_match
        and source_padded_abs_max <= 1e-6
    ):
        return
    message = (
        f"AE replay {path} is legacy/non-V14: missing={missing}, "
        f"full_action_horizon={horizon}, padded_action_dim={padded_dim}, "
        f"full_shapes={full_shapes}, source_shapes={source_shapes}, "
        f"source_padded_abs_max={source_padded_abs_max}"
    )
    if not allow_legacy:
        raise RuntimeError(
            message + ". Pass --allow_legacy_ae_resume only for explicit migration."
        )
    log.warning("%s; continuing because --allow_legacy_ae_resume was set", message)


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
            (
                bool(args.tune_token_online)
                or bool(getattr(args, "export_offline_tokens", False))
                or bool(getattr(args, "retain_tokens", False))
            )
            and not eval_only
            if retain_tokens is None
            else bool(retain_tokens) and not eval_only
        ),
        request_timeout_sec=args.server_request_timeout_sec,
        explore_residual_std=(
            float(getattr(args, "eval_reference_noise_std"))
            if eval_only
            and getattr(args, "eval_reference_noise_std", None) is not None
            else (
                0.0
                if eval_only
                else float(getattr(args, "explore_residual_std", 0.02))
            )
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
        root_seed=int(getattr(args, "seed", 0)),
        actor_mixture_prob=(
            0.0
            if eval_only
            else float(getattr(args, "actor_mixture_prob", 0.0))
        ),
        guide_on_reference=bool(getattr(args, "guide_on_reference", False)),
        residual_clip=(
            None
            if getattr(args, "residual_clip", None) is None
            else float(args.residual_clip)
        ),
        always_collect_actor=(
            False
            if eval_only
            else bool(getattr(args, "always_collect_actor", False))
        ),
        actor_bc_episodes=int(getattr(args, "actor_bc_episodes", 50)),
        always_collect_after_episodes=(
            None
            if eval_only
            else (
                int(args.always_collect_after_episodes)
                if getattr(args, "always_collect_after_episodes", None) is not None
                else None
            )
        ),
    )
    return policy, exp_config


def _sample_with_policy_rng(
    policy: RLTOnlinePolicy | None,
    replay: Any,
    batch_size: int,
    device: torch.device,
    *,
    role: str,
    episode_id: int,
    decision_id: int,
    natural: bool = False,
    require_both_outcomes: bool = False,
) -> dict[str, Any]:
    if policy is not None:
        seed = policy.next_seed(
            role,
            episode_id=episode_id,
            decision_id=decision_id,
        )
        replay.rng = np.random.default_rng(seed)
    if natural:
        batch = replay.sample_natural(batch_size, device=device)
    else:
        batch = replay.sample(
            batch_size,
            device=device,
            require_both_outcomes=require_both_outcomes,
        )
    if policy is not None:
        policy._rng_streams[role] = replay.rng
    return batch


def _slice_batch(
    batch: dict[str, Any],
    start: int,
    stop: int,
) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[start:stop]
        elif isinstance(value, list):
            sliced[key] = value[start:stop]
        else:
            sliced[key] = value
    return sliced


def _mean_step_info(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = set().union(*(row.keys() for row in rows))
    return {
        key: float(
            np.mean(
                [
                    row[key]
                    for row in rows
                    if key in row and np.isfinite(row[key])
                ]
            )
        )
        for key in keys
        if any(key in row and np.isfinite(row[key]) for row in rows)
    }


def _sample_token_with_policy_rng(
    policy: RLTOnlinePolicy | None,
    replay: TokenReplay,
    batch_size: int,
    device: torch.device,
    *,
    episode_id: int,
    decision_id: int,
) -> dict[str, torch.Tensor]:
    if policy is None:
        return replay.sample(batch_size, device=device)
    seed = policy.next_seed(
        "update_sampling",
        episode_id=episode_id,
        decision_id=decision_id,
    )
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed & 0xFFFFFFFF)
        return replay.sample(batch_size, device=device)
    finally:
        np.random.set_state(numpy_state)


def _gate_action_metrics(
    model: MolmoAct2RLTCF,
    batch: dict[str, Any],
    actor: torch.Tensor,
    guided: torch.Tensor,
    *,
    sensitivity_noise: float,
) -> dict[str, float]:
    state = model.encode_state_from_z(batch["z"].detach(), batch["proprio"])
    reference = model.normalize_action(batch["reference_actions"])
    t = None
    if model.is_flow:
        t = torch.ones(
            reference.shape[0],
            1,
            device=reference.device,
            dtype=reference.dtype,
        )
    actor_qs = model.q_chunk(state, actor, t=t)
    reference_qs = model.q_chunk(state, reference, t=t)
    actor_pair = actor_qs - reference_qs
    actor_lcb = actor_pair.mean(dim=0) - actor_pair.std(
        dim=0,
        unbiased=False,
    )
    actor_advantage = (
        model.q_lower_tail_chunk(state, actor, t=t)
        - model.q_lower_tail_chunk(state, reference, t=t)
    )
    actor_perturbed = actor + float(sensitivity_noise) * torch.randn_like(actor)
    actor_sensitivity = (
        model.q_lower_tail_chunk(state, actor, t=t)
        - model.q_lower_tail_chunk(state, actor_perturbed, t=t)
    ).abs()

    guide_pair = model.q_chunk(state, guided, t=t) - actor_qs
    guide_lcb = guide_pair.mean(dim=0) - guide_pair.std(
        dim=0,
        unbiased=False,
    )
    guide_advantage = (
        model.q_lower_tail_chunk(state, guided, t=t)
        - model.q_lower_tail_chunk(state, actor, t=t)
    )
    guide_perturbed = guided + float(sensitivity_noise) * torch.randn_like(guided)
    guide_sensitivity = (
        model.q_lower_tail_chunk(state, guided, t=t)
        - model.q_lower_tail_chunk(state, guide_perturbed, t=t)
    ).abs()
    return {
        "actor_lcb": float(actor_lcb.mean().detach()),
        "actor_advantage": float(actor_advantage.mean().detach()),
        "actor_sensitivity": float(actor_sensitivity.mean().detach()),
        "guide_lcb": float(guide_lcb.mean().detach()),
        "guide_advantage": float(guide_advantage.mean().detach()),
        "guide_sensitivity": float(guide_sensitivity.mean().detach()),
    }


def _split_gate_metrics(
    model: MolmoAct2RLTCF,
    batch: dict[str, Any],
    *,
    sensitivity_noise: float,
    guide_on_reference: bool = False,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        state = model.encode_state_from_z(batch["z"].detach(), batch["proprio"])
        reference = model.normalize_action(batch["reference_actions"])
        if guide_on_reference:
            actor = reference
            if model.guide is None:
                guided = reference
            elif model.is_flow:
                source = torch.randn_like(reference)
                guided, _ = model.flow_sample(
                    state,
                    reference,
                    apply_guide=True,
                    x0=source,
                )
            else:
                _guided_ref, g_delta = model.guide.guide(
                    state,
                    reference,
                    actor_delta=torch.zeros_like(reference),
                )
                guided = reference + g_delta
            return _gate_action_metrics(
                model,
                batch,
                actor,
                guided,
                sensitivity_noise=sensitivity_noise,
            )
        if model.is_flow:
            source = torch.randn_like(reference)
            actor, _ = model.flow_sample(
                state,
                reference,
                apply_guide=False,
                x0=source,
            )
            guided = actor
            if model.guide is not None:
                guided, _ = model.flow_sample(
                    state,
                    reference,
                    apply_guide=True,
                    x0=source,
                )
        else:
            actor, _ = model.actor_chunk(
                state,
                reference,
                deterministic=True,
                apply_guide=False,
            )
            guided = actor
            if model.guide is not None:
                guided, _ = model.actor_chunk(
                    state,
                    reference,
                    deterministic=True,
                    apply_guide=True,
                )
        return _gate_action_metrics(
            model,
            batch,
            actor,
            guided,
            sensitivity_noise=sensitivity_noise,
        )


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
    policy: RLTOnlinePolicy | None = None,
    empirical_tracker: EmpiricalGateTracker | None = None,
) -> GateStatus:
    gate_metrics = {
        "actor_lcb": 0.0,
        "actor_advantage": 0.0,
        "actor_sensitivity": 0.0,
        "guide_lcb": 0.0,
        "guide_advantage": 0.0,
        "guide_sensitivity": 0.0,
    }
    health_fields: dict[str, float] = {}
    healthy = False
    ae_mode = ae_backend is not None
    guide_on_reference = bool(getattr(args, "guide_on_reference", False))
    gate_replay: Any = image_replay if ae_mode else replay
    min_success_episodes = (
        int(getattr(args, "ae_min_success_episodes", 3))
        if ae_mode
        else 1
    )
    enough_data = (
        (args.actor_mode == "rlt" or (guide_on_reference and args.use_cf_guide))
        and valid_episodes >= args.g_start_episodes
        and gate_replay is not None
        and len(gate_replay) >= args.min_replay_chunks
        and gate_replay.has_both_outcomes()
        and gate_replay.successful_episode_count() >= min_success_episodes
    )
    if enough_data:
        batch_size = (
            int(getattr(args, "ae_batch_size", 2))
            if ae_mode
            else int(args.batch_size)
        )
        if batch_size < 2:
            raise ValueError("Gate batch size must be at least 2 for both outcomes")
        batch = _sample_with_policy_rng(
            policy,
            gate_replay,
            batch_size,
            device,
            role="gate_probes",
            episode_id=valid_episodes,
            decision_id=0,
            natural=True,
        )
        health_batch = dict(batch)
        if "success" in batch:
            health_batch["mc_return"] = batch["success"]
        health_fields = critic_health_metrics(
            model,
            health_batch,
            sensitivity_noise=float(args.gate_sensitivity_noise),
        )
        healthy = bool(health_fields.get("healthy", 0.0))
        if healthy:
            devices = (
                [device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == "cuda"
                else []
            )
            with torch.random.fork_rng(devices=devices):
                probe_seed = (
                    policy.next_seed(
                        "gate_probes",
                        episode_id=valid_episodes,
                        decision_id=1,
                    )
                    if policy is not None
                    else _stable_seed(0, "gate_probes", valid_episodes, 1)
                )
                torch.manual_seed(probe_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed(probe_seed)
                if ae_mode:
                    ae_metrics = ae_flow_gate_metrics(
                        model,
                        ae_backend,
                        batch,
                        sensitivity_noise=float(args.gate_sensitivity_noise),
                    )
                    gate_metrics.update(
                        {
                            "actor_lcb": ae_metrics["actor_paired_lcb"],
                            "actor_advantage": ae_metrics["actor_advantage"],
                            "actor_sensitivity": ae_metrics["actor_sensitivity"],
                            "guide_lcb": ae_metrics["guide_paired_lcb"],
                            "guide_advantage": ae_metrics["guide_advantage"],
                            "guide_sensitivity": ae_metrics["guide_sensitivity"],
                        }
                    )
                else:
                    gate_metrics.update(
                        _split_gate_metrics(
                            model,
                            batch,
                            sensitivity_noise=float(args.gate_sensitivity_noise),
                            guide_on_reference=guide_on_reference,
                        )
                    )

    empirical_metrics = (
        empirical_tracker.metrics()
        if empirical_tracker is not None
        else {
            "empirical_lcb": -1.0,
            "empirical_ready": 0.0,
            "empirical_actor_episodes": 0.0,
            "empirical_ref_episodes": 0.0,
        }
    )
    require_empirical = bool(getattr(args, "require_empirical_gate", False))
    min_empirical_episodes = int(getattr(args, "empirical_min_episodes", 16))
    min_empirical_lcb = float(getattr(args, "g_min_empirical_advantage", 0.0))
    empirical_ready = bool(
        empirical_metrics.get("empirical_actor_episodes", 0.0) >= min_empirical_episodes
        and empirical_metrics.get("empirical_ref_episodes", 0.0) >= min_empirical_episodes
    )
    empirical_lcb = float(empirical_metrics.get("empirical_lcb", -1.0))
    empirical_pass = (not require_empirical) or (
        empirical_ready and empirical_lcb >= min_empirical_lcb
    )
    empirical_block_reason = ""
    if require_empirical and not empirical_ready:
        empirical_block_reason = "empirical_insufficient_episodes"
    elif require_empirical and empirical_lcb < min_empirical_lcb:
        empirical_block_reason = "empirical_lcb_below_threshold"

    actor_ready = (
        enough_data
        and healthy
        and args.actor_mode == "rlt"
        and (not guide_on_reference)
        and np.isfinite(gate_metrics["actor_lcb"])
        and gate_metrics["actor_lcb"] >= float(args.g_min_advantage)
        and np.isfinite(gate_metrics["actor_sensitivity"])
        and gate_metrics["actor_sensitivity"] >= float(args.g_min_action_sensitivity)
        and empirical_pass
    )
    min_guide_adv = float(getattr(args, "g_min_guide_advantage", 0.003))
    guide_ready = bool(
        enough_data
        and healthy
        and args.use_cf_guide
        and model.guide is not None
        and np.isfinite(gate_metrics["guide_lcb"])
        and gate_metrics["guide_lcb"] >= min_guide_adv
        and np.isfinite(gate_metrics["guide_sensitivity"])
        and gate_metrics["guide_sensitivity"] >= float(args.g_min_action_sensitivity)
        and (guide_on_reference or actor_ready)
        and (empirical_pass or guide_on_reference)
    )

    deploy_policy = str(getattr(args, "deploy_policy", "gated"))
    deploy_actor = False
    deploy_guide = False
    if deploy_policy == "gated":
        if guide_on_reference:
            deploy_actor = False
            deploy_guide = bool(guide_ready)
        else:
            deploy_actor = bool(actor_ready)
            deploy_guide = bool(actor_ready and guide_ready)
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
        if guide_on_reference:
            deploy_guide = bool(source.get("gate_deploy_guide", False))
            deploy_actor = False
    elif deploy_policy == "reference":
        deploy_actor = False
        deploy_guide = False
    elif deploy_policy == "actor":
        deploy_actor = True
        deploy_guide = False
    elif deploy_policy == "actor_guide":
        deploy_actor = not guide_on_reference
        deploy_guide = True
    else:
        raise ValueError(f"Unknown deploy policy: {deploy_policy}")

    if args.actor_mode != "rlt" and not guide_on_reference:
        deploy_actor = False
        deploy_guide = False
    if guide_on_reference:
        deploy_actor = False
    if deploy_guide and (not args.use_cf_guide or model.guide is None):
        raise RuntimeError(
            f"deploy_policy={deploy_policy} requested a guide, but no guide is loaded"
        )
    if deploy_guide and not deploy_actor and not guide_on_reference:
        raise RuntimeError("Guide deployment requires actor deployment")

    if guide_on_reference:
        actor_block_reason = "guide_on_reference_arm"
    elif args.actor_mode != "rlt":
        actor_block_reason = "actor_mode_is_reference"
    elif not enough_data:
        actor_block_reason = "natural_gate_replay_not_ready"
    elif not healthy:
        actor_block_reason = "critic_unhealthy"
    elif not np.isfinite(gate_metrics["actor_lcb"]):
        actor_block_reason = "actor_lcb_nonfinite"
    elif gate_metrics["actor_lcb"] < float(args.g_min_advantage):
        actor_block_reason = "actor_lcb_below_threshold"
    elif not np.isfinite(gate_metrics["actor_sensitivity"]):
        actor_block_reason = "actor_sensitivity_nonfinite"
    elif gate_metrics["actor_sensitivity"] < float(args.g_min_action_sensitivity):
        actor_block_reason = "actor_sensitivity_below_threshold"
    elif not empirical_pass:
        actor_block_reason = empirical_block_reason or "empirical_gate_blocked"
    else:
        actor_block_reason = ""

    if not args.use_cf_guide or model.guide is None:
        guide_block_reason = "guide_disabled"
    elif not guide_on_reference and not actor_ready:
        guide_block_reason = "actor_gate_closed"
    elif not healthy:
        guide_block_reason = "critic_unhealthy"
    elif not np.isfinite(gate_metrics["guide_lcb"]):
        guide_block_reason = "guide_lcb_nonfinite"
    elif gate_metrics["guide_lcb"] < min_guide_adv:
        guide_block_reason = "guide_lcb_below_threshold"
    elif not np.isfinite(gate_metrics["guide_sensitivity"]):
        guide_block_reason = "guide_sensitivity_nonfinite"
    elif gate_metrics["guide_sensitivity"] < float(args.g_min_action_sensitivity):
        guide_block_reason = "guide_sensitivity_below_threshold"
    else:
        guide_block_reason = ""

    return GateStatus(
        actor_ready=bool(actor_ready),
        guide_ready=bool(guide_ready),
        deploy_actor=bool(deploy_actor),
        deploy_guide=bool(deploy_guide),
        actor_lcb=float(gate_metrics["actor_lcb"]),
        actor_advantage=float(gate_metrics["actor_advantage"]),
        actor_sensitivity=float(gate_metrics["actor_sensitivity"]),
        guide_lcb=float(gate_metrics["guide_lcb"]),
        guide_advantage=float(gate_metrics["guide_advantage"]),
        guide_sensitivity=float(gate_metrics["guide_sensitivity"]),
        critic_health=bool(healthy),
        critic_health_fields=health_fields,
        actor_block_reason=actor_block_reason,
        guide_block_reason=guide_block_reason,
        empirical_lcb=float(empirical_lcb),
        empirical_ready=bool(empirical_ready),
        empirical_block_reason=empirical_block_reason,
        empirical_actor_episodes=float(
            empirical_metrics.get("empirical_actor_episodes", 0.0)
        ),
        empirical_ref_episodes=float(
            empirical_metrics.get("empirical_ref_episodes", 0.0)
        ),
        empirical_actor_sr=float(empirical_metrics.get("empirical_actor_sr", 0.0)),
        empirical_ref_sr=float(empirical_metrics.get("empirical_ref_sr", 0.0)),
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
    policy: RLTOnlinePolicy | None = None,
    valid_episodes: int = 0,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, float],
    dict[str, float],
    UpdateStatus,
]:
    update_start = time.monotonic()
    q_info: dict[str, float] = {}
    actor_info: dict[str, float] = {}
    guide_info: dict[str, float] = {}
    token_info: dict[str, float] = {}
    ae_mode = bool(getattr(args, "ae_trainable", False)) and ae_backend is not None
    ae_batch_size = int(getattr(args, "ae_batch_size", 2))
    active_replay: Any = image_replay if ae_mode else replay
    batch_size = ae_batch_size if ae_mode else int(args.batch_size)
    configured_max_updates = getattr(args, "max_updates_per_episode", None)
    if configured_max_updates is None:
        configured_max_updates = getattr(args, "updates_per_episode", 0)
    max_updates = int(configured_max_updates or 0)
    max_update_sec = float(
        getattr(args, "max_update_sec_per_episode", 300.0)
    )
    skip_reason = ""
    if max_updates <= 0:
        skip_reason = "updates_disabled"
    elif max_update_sec <= 0.0:
        skip_reason = "zero_time_budget"
    elif active_replay is None:
        skip_reason = "missing_active_replay"
    elif len(active_replay) < int(args.min_replay_chunks):
        skip_reason = "insufficient_replay_chunks"
    elif (
        ae_mode
        and len(replay) < int(args.min_replay_chunks)
    ):
        skip_reason = "insufficient_compact_replay_chunks"
    elif batch_size < 2 or (
        ae_mode and int(args.batch_size) < 2
    ):
        skip_reason = "batch_too_small_for_both_outcomes"
    elif not active_replay.has_both_outcomes():
        skip_reason = "replay_missing_both_outcomes"
    elif ae_mode and not replay.has_both_outcomes():
        skip_reason = "compact_replay_missing_both_outcomes"
    elif (
        ae_mode
        and active_replay.successful_episode_count()
        < int(getattr(args, "ae_min_success_episodes", 3))
    ):
        skip_reason = "insufficient_success_episodes"
    if skip_reason:
        positive, negative = (
            active_replay.outcome_counts()
            if active_replay is not None
            else (0, 0)
        )
        successful_episodes = (
            active_replay.successful_episode_count()
            if active_replay is not None
            else 0
        )
        log.info(
            "Skipping online updates: reason=%s rows=%d positive=%d "
            "negative=%d successful_episodes=%d",
            skip_reason,
            len(active_replay) if active_replay is not None else 0,
            positive,
            negative,
            successful_episodes,
        )
        status = UpdateStatus(
            skipped_reason=skip_reason,
            stop_reason=skip_reason,
            rounds=0,
            critic_updates=0,
            actor_updates=0,
            guide_updates=0,
            token_updates=0,
            elapsed_sec=time.monotonic() - update_start,
        )
        return q_info, actor_info, guide_info, token_info, status

    rounds = 0
    critic_updates = 0
    actor_updates = 0
    guide_updates = 0
    token_updates = 0
    sample_id = 0
    stop_reason = "max_updates"
    if policy is not None:
        update_torch_seed = policy.next_seed(
            "update_sampling",
            episode_id=valid_episodes,
            decision_id=-1,
        )
        torch.manual_seed(update_torch_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(update_torch_seed)

    def time_remaining() -> bool:
        return time.monotonic() - update_start < max_update_sec

    while rounds < max_updates and time_remaining():
        rounds += 1
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
            "critic_target_use_guide": bool(
                getattr(args, "critic_target_use_guide", False)
            ),
            "actor_cql_coef": float(getattr(args, "actor_cql_coef", 0.0)),
        }
        if ae_mode:
            # Cheap endpoint critic first, then actor/AE, then optional expensive
            # AE TD critic. The old 2:1 order let one AE critic step burn the
            # whole episode budget and starve LoRA updates.
            if not time_remaining():
                stop_reason = "time_budget"
                break
            batch = _sample_with_policy_rng(
                policy,
                replay,
                int(args.batch_size),
                device,
                role="update_sampling",
                episode_id=valid_episodes,
                decision_id=sample_id,
                require_both_outcomes=True,
            )
            sample_id += 1
            q_info = endpoint_critic_mc_step(
                model,
                optimizers["critic"],
                batch,
                rank_coef=args.rank_coef,
                rank_margin=args.rank_margin,
                rank_noise=args.rank_noise,
                far_rank_coef=args.far_rank_coef,
                far_rank_noise=args.far_rank_noise,
                shuffle_rank_coef=args.shuffle_rank_coef,
                actions_already_normalized=True,
            )
            critic_updates += 1
        else:
            # RLT uses two critic updates for every actor update.
            for _critic_update in range(2):
                if not time_remaining():
                    stop_reason = "time_budget"
                    break
                batch = _sample_with_policy_rng(
                    policy,
                    active_replay,
                    batch_size,
                    device,
                    role="update_sampling",
                    episode_id=valid_episodes,
                    decision_id=sample_id,
                    require_both_outcomes=True,
                )
                sample_id += 1
                critic_fn = flow_critic_td_step if model.is_flow else critic_td_step
                q_info = critic_fn(
                    model,
                    optimizers["critic"],
                    batch,
                    **critic_kwargs,
                )
                critic_updates += 1
            if not time_remaining():
                stop_reason = "time_budget"
                break
        guide_on_reference = bool(getattr(args, "guide_on_reference", False))
        if args.actor_mode == "rlt" or guide_on_reference:
            actor_batch = _sample_with_policy_rng(
                policy,
                active_replay,
                batch_size,
                device,
                role="update_sampling",
                episode_id=valid_episodes,
                decision_id=sample_id,
                natural=True,
            )
            sample_id += 1
            if ae_mode:
                actor_batch_size = int(actor_batch["z"].shape[0])
                microbatch_size = min(
                    int(getattr(args, "ae_microbatch_size", 4)),
                    actor_batch_size,
                )
                microbatch_ranges = [
                    (
                        start,
                        min(start + microbatch_size, actor_batch_size),
                    )
                    for start in range(0, actor_batch_size, microbatch_size)
                ]
                microbatch_infos = []
                for microbatch_index, (start, stop) in enumerate(
                    microbatch_ranges
                ):
                    microbatch_infos.append(
                        ae_flow_actor_step(
                            model,
                            ae_backend,
                            optimizers["actor"],
                            optimizers["alpha"],
                            _slice_batch(actor_batch, start, stop),
                            beta=args.actor_beta,
                            target_divergence=args.target_divergence,
                            bc_ref_coef=float(
                                getattr(args, "bc_ref_coef", 1.0)
                            ),
                            zero_grad=microbatch_index == 0,
                            optimizer_step=(
                                microbatch_index + 1
                                == len(microbatch_ranges)
                            ),
                            loss_scale=1.0 / len(microbatch_ranges),
                        )
                    )
                actor_info = _mean_step_info(microbatch_infos)
                actor_info["ae_microbatch_size"] = float(microbatch_size)
                actor_info["ae_accumulation_steps"] = float(
                    len(microbatch_ranges)
                )
                actor_updates += 1
                if args.use_cf_guide and time_remaining():
                    guide_health = critic_health_metrics(
                        model,
                        actor_batch,
                        sensitivity_noise=float(args.gate_sensitivity_noise),
                    )
                    if bool(guide_health.get("healthy", 0.0)):
                        guide_info = ae_flow_guide_step(
                            model,
                            ae_backend,
                            optimizers["guide"],
                            actor_batch,
                            beta=args.guide_beta,
                        )
                        if not bool(
                            guide_info.get("guide_update_skipped", 0.0)
                        ):
                            guide_updates += 1
                    else:
                        guide_info = {
                            "guide_loss": 0.0,
                            "guide_update_skipped": 1.0,
                            "guide_skip_unhealthy_critic": 1.0,
                            **{
                                f"guide_health_{key}": float(value)
                                for key, value in guide_health.items()
                            },
                        }
            else:
                phase = actor_phase_for_episode(
                    int(valid_episodes),
                    bc_episodes=int(getattr(args, "actor_bc_episodes", 50)),
                    q_ramp_episodes=int(getattr(args, "q_ramp_episodes", 0)),
                    residual_clip=float(getattr(args, "residual_clip", 0.02)),
                    advantage_clip=float(getattr(args, "advantage_clip", 0.05)),
                    endpoint_ref_mse_max=float(
                        getattr(args, "endpoint_ref_mse_max", 0.01)
                    ),
                    deploy_ref_dropout=float(
                        getattr(args, "train_ref_dropout", 0.0)
                    ),
                )
                train_actor = (
                    args.actor_mode == "rlt"
                    and not bool(getattr(args, "guide_on_reference", False))
                )
                if train_actor:
                    actor_fn = flow_actor_step if model.is_flow else actor_step
                    actor_kwargs = {
                        "beta": args.actor_beta,
                        "target_divergence": args.target_divergence,
                        "ref_dropout": phase.ref_dropout,
                        "q_coef": phase.q_coef,
                        "residual_clip": phase.residual_clip,
                        "advantage_clip": phase.advantage_clip,
                    }
                    if model.is_flow:
                        actor_kwargs["bc_ref_coef"] = float(
                            getattr(args, "bc_ref_coef", 1.0)
                        )
                        actor_kwargs["endpoint_ref_mse_max"] = (
                            phase.endpoint_ref_mse_max
                        )
                    actor_info = actor_fn(
                        model,
                        optimizers["actor"],
                        optimizers["alpha"],
                        actor_batch,
                        **actor_kwargs,
                    )
                    actor_info["actor_phase"] = phase.phase
                    actor_updates += 1
                if args.use_cf_guide and time_remaining():
                    guide_health = critic_health_metrics(
                        model,
                        actor_batch,
                        sensitivity_noise=float(args.gate_sensitivity_noise),
                    )
                    if bool(guide_health.get("healthy", 0.0)):
                        guide_fn = (
                            flow_guide_step if model.is_flow else guide_step
                        )
                        guide_info = guide_fn(
                            model,
                            optimizers["guide"],
                            actor_batch,
                            beta=args.guide_beta,
                            guide_on_reference=bool(
                                getattr(args, "guide_on_reference", False)
                            ),
                        )
                        if not bool(
                            guide_info.get("guide_update_skipped", 0.0)
                        ):
                            guide_updates += 1
                    else:
                        guide_info = {
                            "guide_loss": 0.0,
                            "guide_update_skipped": 1.0,
                            "guide_skip_unhealthy_critic": 1.0,
                            **{
                                f"guide_health_{key}": float(value)
                                for key, value in guide_health.items()
                            },
                        }
        if ae_mode and time_remaining():
            batch = _sample_with_policy_rng(
                policy,
                active_replay,
                batch_size,
                device,
                role="update_sampling",
                episode_id=valid_episodes,
                decision_id=sample_id,
                require_both_outcomes=True,
            )
            sample_id += 1
            q_info = ae_flow_critic_td_step(
                model,
                ae_backend,
                optimizers["critic"],
                batch,
                **critic_kwargs,
            )
            critic_updates += 1
        elif ae_mode and not time_remaining():
            stop_reason = "time_budget"
        if args.tune_token_online and len(token_replay) > 0:
            if not time_remaining():
                stop_reason = "time_budget"
                break
            token_batch = _sample_token_with_policy_rng(
                policy,
                token_replay,
                args.token_batch_size,
                device,
                episode_id=valid_episodes,
                decision_id=sample_id,
            )
            sample_id += 1
            token_info = token_step(model, optimizers["token"], token_batch)
            token_updates += 1
    if not time_remaining():
        stop_reason = "time_budget"
    status = UpdateStatus(
        skipped_reason="",
        stop_reason=stop_reason,
        rounds=rounds,
        critic_updates=critic_updates,
        actor_updates=actor_updates,
        guide_updates=guide_updates,
        token_updates=token_updates,
        elapsed_sec=time.monotonic() - update_start,
    )
    log.info(
        "Online update budget stopped: reason=%s rounds=%d critic=%d "
        "actor=%d guide=%d token=%d elapsed=%.2fs",
        status.stop_reason,
        status.rounds,
        status.critic_updates,
        status.actor_updates,
        status.guide_updates,
        status.token_updates,
        status.elapsed_sec,
    )
    return q_info, actor_info, guide_info, token_info, status


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


def _save_token_replay(replay: TokenReplay, replay_out: str) -> None:
    if not replay_out or len(replay) == 0:
        return
    path = Path(replay_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stage on local disk when possible — atomic NFS replace of multi-GB
    # object NPZs is what killed V17 collect at eps=150.
    staging_root = Path(
        os.environ.get("RLT_TOKEN_SAVE_STAGING", "/tmp/rlt_token_stage")
    )
    staging_root.mkdir(parents=True, exist_ok=True)
    temporary = staging_root / f"{path.stem}.{os.getpid()}.tmp.npz"
    nfs_tmp = path.with_name(f".{path.name}.tmp.npz")
    # Serialize token flushes across shards — 8× parallel compress+write
    # spikes RAM past 1TB and leaves 0-byte staging files.
    lock_path = staging_root / "token_flush.lock"

    def _write() -> None:
        _cleanup_path(temporary)
        _cleanup_path(nfs_tmp)
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                log.info(
                    "Token flush acquired lock n=%d -> %s",
                    len(replay),
                    temporary,
                )
                replay.save_npz(str(temporary))
                with open(temporary, "rb") as handle:
                    os.fsync(handle.fileno())
                shutil.copy2(temporary, nfs_tmp)
                with open(nfs_tmp, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(nfs_tmp, path)
                _cleanup_path(temporary)
                log.info("Token flush done -> %s (%.2f GB)", path, path.stat().st_size / 1e9)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    try:
        _io_retry(f"token replay save {path}", _write)
    except Exception:
        _cleanup_path(temporary)
        _cleanup_path(nfs_tmp)
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
    meta: dict[str, Any] | None = None,
) -> None:
    path = out_dir / "molmo_ae_lora_latest.pt"
    temporary = path.with_name(f".{path.name}.tmp")

    def _write() -> None:
        _cleanup_path(temporary)
        ae_backend.save_trainable(
            temporary,
            meta={
                **(meta or {}),
                "env_steps": int(env_steps),
                "online_schema_version": _V14_ONLINE_SCHEMA,
                "ae_action_contract": _validate_ae_contract(ae_backend),
            },
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
    if ae_backend is not None:
        snapshot_meta.update(
            {
                "online_schema_version": _V14_ONLINE_SCHEMA,
                "ae_action_contract": _validate_ae_contract(ae_backend),
            }
        )
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
                "online_schema_version": _V14_ONLINE_SCHEMA,
                "ae_action_contract": _validate_ae_contract(ae_backend),
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
    token_replay: TokenReplay | None = None,
    token_replay_out: str = "",
    chunk_token_replay: TokenReplay | None = None,
    chunk_token_replay_out: str = "",
    optimizers: dict[str, torch.optim.Optimizer] | None = None,
    policy: RLTOnlinePolicy | None = None,
    save_token_replays: bool = True,
) -> Path | None:
    """Checkpoint first, then metrics — so counters never outrun weights.

    On transient NFS failures, log and continue unless ``required`` (final save).
    Always runs outside the EGL GPU lock.

    ``save_token_replays`` gates multi-GB token NPZ writes. Offline collect can
    skip intermediate dumps (keep buffers in RAM) and flush once at the end.
    """
    checkpoint: Path | None = None
    checkpoint_meta = {
        **row,
        "online_schema_version": _V14_ONLINE_SCHEMA,
        "optimizer_states": {
            name: optimizer.state_dict()
            for name, optimizer in (optimizers or {}).items()
        },
        "process_rng_state": _capture_process_rng_state(),
        "policy_rng_state": (
            policy.rng_state_dict() if policy is not None else None
        ),
    }
    if ae_backend is not None:
        checkpoint_meta["ae_action_contract"] = _validate_ae_contract(
            ae_backend
        )
    try:
        checkpoint = _save_checkpoint(
            model,
            out_dir,
            env_steps,
            checkpoint_meta,
        )
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

    if save_token_replays and token_replay is not None and token_replay_out:
        # Shared buffer: only write once, then symlink the other path.
        same_buffer = (
            chunk_token_replay is token_replay
            and bool(chunk_token_replay_out)
            and str(chunk_token_replay_out) != str(token_replay_out)
        )
        if same_buffer:
            pass  # saved via chunk_token path below
        else:
            try:
                _save_token_replay(token_replay, token_replay_out)
            except Exception as error:  # noqa: BLE001
                log.warning(
                    "Token replay save failed after retries (steps=%d): %s",
                    env_steps,
                    error,
                )
                if required:
                    raise
    if (
        save_token_replays
        and chunk_token_replay is not None
        and chunk_token_replay_out
    ):
        try:
            _save_token_replay(chunk_token_replay, chunk_token_replay_out)
            if (
                token_replay is chunk_token_replay
                and token_replay_out
                and str(token_replay_out) != str(chunk_token_replay_out)
            ):
                token_path = Path(token_replay_out)
                chunk_path = Path(chunk_token_replay_out)
                if token_path.exists() or token_path.is_symlink():
                    token_path.unlink()
                token_path.symlink_to(chunk_path.resolve())
        except Exception as error:  # noqa: BLE001
            log.warning(
                "Chunk-token replay save failed after retries (steps=%d): %s",
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
            _save_ae_trainable(
                ae_backend,
                out_dir,
                env_steps,
                meta={
                    "valid_episodes": int(row.get("valid_episodes", 0)),
                    "policy_rng_state": (
                        policy.rng_state_dict()
                        if policy is not None
                        else None
                    ),
                },
            )
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
    image_replay: ImageChunkReplay | None = None,
    update_status: UpdateStatus | None = None,
    coordinate_diagnostics: dict[str, float] | None = None,
    policy: RLTOnlinePolicy | None = None,
    model: MolmoAct2RLTCF | None = None,
    episode_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed = max(time.time() - start_time, 1e-6)
    replay_positive, replay_negative = replay.outcome_counts()
    ae_positive, ae_negative = (
        image_replay.outcome_counts()
        if image_replay is not None
        else (0, 0)
    )
    update = update_status or UpdateStatus(
        skipped_reason="",
        stop_reason="not_run",
        rounds=0,
        critic_updates=0,
        actor_updates=0,
        guide_updates=0,
        token_updates=0,
        elapsed_sec=0.0,
    )
    row: dict[str, Any] = {
        "online_schema_version": _V16_ONLINE_SCHEMA,
        "env_steps": int(env_steps),
        "target_env_steps": int(args.target_env_steps),
        "valid_episodes": int(valid_episodes),
        "skipped_episodes": int(skipped_episodes),
        "cycle": int(valid_episodes + skipped_episodes),
        "successes": int(successes),
        "recent_outcomes": [float(value) for value in recent],
        "cumulative_success_rate": successes / max(valid_episodes, 1),
        "window_success_rate": float(np.mean(recent)) if recent else 0.0,
        "g_enabled": bool(gate_status.deploy_actor),
        "g_predicted_advantage": float(gate_status.paired_lcb),
        "gate_would_enable": bool(gate_status.would_enable),
        "gate_actor_ready": bool(gate_status.actor_ready),
        "gate_guide_ready": bool(gate_status.guide_ready),
        "gate_deploy_actor": bool(gate_status.deploy_actor),
        "gate_deploy_guide": bool(gate_status.deploy_guide),
        "gate_actor_lcb": float(gate_status.actor_lcb),
        "gate_actor_advantage": float(gate_status.actor_advantage),
        "gate_actor_sensitivity": float(gate_status.actor_sensitivity),
        "gate_guide_lcb": float(gate_status.guide_lcb),
        "gate_guide_advantage": float(gate_status.guide_advantage),
        "gate_guide_sensitivity": float(gate_status.guide_sensitivity),
        "gate_actor_block_reason": gate_status.actor_block_reason,
        "gate_guide_block_reason": gate_status.guide_block_reason,
        "gate_paired_lcb": float(gate_status.actor_lcb),
        "gate_q_min_advantage": float(gate_status.actor_advantage),
        "gate_critic_health": bool(gate_status.critic_health),
        "gate_sensitivity": float(gate_status.actor_sensitivity),
        "empirical_lcb": float(gate_status.empirical_lcb),
        "empirical_ready": bool(gate_status.empirical_ready),
        "empirical_block_reason": gate_status.empirical_block_reason,
        "empirical_actor_episodes": float(gate_status.empirical_actor_episodes),
        "empirical_ref_episodes": float(gate_status.empirical_ref_episodes),
        "empirical_actor_sr": float(gate_status.empirical_actor_sr),
        "empirical_ref_sr": float(gate_status.empirical_ref_sr),
        "critic_healthy": bool(gate_status.critic_health),
        "action_sensitivity": float(gate_status.actor_sensitivity),
        "q_td_loss": float(q_info.get("q_td_loss", 0.0)),
        "q_rank_loss": float(q_info.get("q_rank_loss", 0.0)),
        "q_rank_gap": float(q_info.get("q_rank_gap", 0.0)),
        "q_mean": float(q_info.get("q_mean", 0.0)),
        "q_std": float(q_info.get("q_std", 0.0)),
        "actor_adv": float(actor_info.get("actor_adv", 0.0)),
        "actor_ref_mse": float(actor_info.get("actor_ref_mse", 0.0)),
        "residual_mse": float(actor_info.get("residual_mse", 0.0)),
        "residual_rms": float(actor_info.get("residual_rms", 0.0)),
        "endpoint_ref_mse": float(actor_info.get("endpoint_ref_mse", 0.0)),
        "actor_q_coef": float(actor_info.get("actor_q_coef", 0.0)),
        "actor_phase": str(actor_info.get("actor_phase", "")),
        "actor_cql_loss": float(q_info.get("actor_cql_loss", 0.0)),
        "actor_mixture_prob": float(getattr(args, "actor_mixture_prob", 0.0)),
        "always_collect_actor": bool(getattr(args, "always_collect_actor", False)),
        "actor_beta": float(getattr(args, "actor_beta", 1.0)),
        "train_ref_dropout": float(getattr(args, "train_ref_dropout", 0.0)),
        "ae_grad_norm": float(actor_info.get("ae_grad_norm", 0.0)),
        "ae_context_k_proj_grad_norm": float(
            actor_info.get("ae_context_k_proj_grad_norm", 0.0)
        ),
        "ae_context_v_proj_grad_norm": float(
            actor_info.get("ae_context_v_proj_grad_norm", 0.0)
        ),
        "ae_context_proj_grad_norm": float(
            actor_info.get("ae_context_proj_grad_norm", 0.0)
        ),
        "ae_context_proj_grad_nonzero": float(
            actor_info.get("ae_context_proj_grad_nonzero", 0.0)
        ),
        "ae_optimizer_param_coverage": float(
            actor_info.get("ae_optimizer_param_coverage", 0.0)
        ),
        "guide_adv": float(guide_info.get("guide_adv", 0.0)),
        "guide_w_norm": float(guide_info.get("w_norm", 0.0)),
        "guide_target_norm": float(guide_info.get("target_norm", 0.0)),
        "token_recon_loss": float(token_info.get("token_recon_loss", 0.0)),
        "config_name": args.config_name,
        "v_source": str(getattr(args, "v_source", "rlt")),
        "critic_target_use_guide": bool(
            getattr(args, "critic_target_use_guide", False)
        ),
        "recovered_critic_heads": list(
            getattr(args, "recovered_critic_heads", [])
        ),
        "ae_native_coordinate_reset": bool(
            getattr(args, "ae_native_coordinate_reset", False)
        ),
        "ae_native_coordinate_initialized": bool(
            getattr(args, "ae_native_coordinate_initialized", False)
        ),
        "ae_native_action_coordinates": bool(image_replay is not None),
        "steps_per_sec": env_steps / elapsed,
        "chunk_transitions": len(replay),
        "replay_success_rows": int(replay_positive),
        "replay_failure_rows": int(replay_negative),
        "replay_success_episodes": int(replay.successful_episode_count()),
        "replay_storage_bytes": int(replay.storage_nbytes()),
        "ae_replay_transitions": (
            len(image_replay) if image_replay is not None else 0
        ),
        "ae_replay_success_rows": int(ae_positive),
        "ae_replay_failure_rows": int(ae_negative),
        "ae_replay_success_episodes": (
            int(image_replay.successful_episode_count())
            if image_replay is not None
            else 0
        ),
        "ae_replay_storage_bytes": (
            int(image_replay.storage_nbytes())
            if image_replay is not None
            else 0
        ),
        "update_skipped_reason": str(update.skipped_reason),
        "update_stop_reason": str(update.stop_reason),
        "update_rounds": int(update.rounds),
        "critic_updates": int(update.critic_updates),
        "actor_updates": int(update.actor_updates),
        "guide_updates": int(update.guide_updates),
        "token_updates": int(update.token_updates),
        "update_elapsed_sec": float(update.elapsed_sec),
        "policy_rng_state": (
            policy.rng_state_dict() if policy is not None else None
        ),
        "server_port": int(args.server_port),
        "elapsed_sec": elapsed,
    }
    for key, value in gate_status.critic_health_fields.items():
        row[f"critic_health_{key}"] = float(value)
    for source in (q_info, actor_info, guide_info):
        for key in (
            "compact_endpoint_update",
            "actor_local_adv",
            "actor_endpoint_adv",
            "actor_q_look",
            "actor_q_end",
            "residual_mse",
            "endpoint_steps",
            "endpoint_t",
            "ae_microbatch_size",
            "ae_accumulation_steps",
            "guide_update_skipped",
            "guide_skip_tiny_critic_gradient",
            "guide_skip_unhealthy_critic",
            "critic_gradient_raw_norm_mean",
            "critic_gradient_raw_norm_min",
            "critic_gradient_selected_norm_mean",
            "critic_gradient_selected_norm_min",
            "critic_gradient_nonzero_fraction",
            "critic_gradient_direction_agreement",
        ):
            if key in source:
                row[key] = float(source[key])
    row.update(coordinate_diagnostics or {})
    if episode_info:
        row["episode_used_actor"] = bool(episode_info.get("episode_used_actor", False))
        row["episode_actor_chunks"] = int(episode_info.get("episode_actor_chunks", 0))
        row["episode_mixture_draws"] = int(episode_info.get("episode_mixture_draws", 0))
        row["episode_collect_policy"] = str(
            episode_info.get("episode_collect_policy", "reference")
        )
        row["episode_residual_rms"] = float(episode_info.get("residual_rms", 0.0))
        row["episode_actor_ref_mse"] = float(episode_info.get("actor_ref_mse", 0.0))
        row["vla_prefetch_hit"] = int(episode_info.get("vla_prefetch_hit", 0))
        row["vla_prefetch_miss"] = int(episode_info.get("vla_prefetch_miss", 0))
        row["vla_prefetch_wait_ms"] = float(episode_info.get("vla_prefetch_wait_ms", 0.0))
        row["vla_prefetch_discarded"] = int(
            episode_info.get("vla_prefetch_discarded", 0)
        )
    if model is not None:
        row.update(
            {
                "critic_action_mean_abs_max": float(
                    model.action_mean.detach().abs().max().cpu()
                ),
                "critic_action_std_min": float(
                    model.action_std.detach().min().cpu()
                ),
                "critic_action_std_max": float(
                    model.action_std.detach().max().cpu()
                ),
            }
        )
    return row


def train_rlt_online(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.assets_dir:
        os.environ["MLSPACES_ASSETS_DIR"] = args.assets_dir
    eval_only = bool(getattr(args, "eval_only", False))
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    if not eval_only:
        out_dir.mkdir(parents=True, exist_ok=True)
    if not eval_only and not str(getattr(args, "replay_out", "")):
        args.replay_out = str(out_dir / "chunk_replay.npz")
    if (
        not eval_only
        and bool(getattr(args, "export_offline_tokens", False))
        and not str(getattr(args, "token_replay_out", ""))
        and not str(getattr(args, "chunk_token_replay_out", ""))
    ):
        # Only default token_replay_out when neither path is set. Collect that
        # only passes chunk_token_replay_out must not also auto-write token_replay.
        args.token_replay_out = str(out_dir / "token_replay.npz")
    if (
        not eval_only
        and bool(getattr(args, "export_offline_tokens", False))
        and not str(getattr(args, "chunk_token_replay_out", ""))
    ):
        args.chunk_token_replay_out = str(out_dir / "chunk_token_replay.npz")
    if (
        not eval_only
        and
        bool(getattr(args, "ae_trainable", False))
        and not str(getattr(args, "ae_image_replay_out", ""))
    ):
        args.ae_image_replay_out = str(out_dir / "ae_image_replay.npz")
    metrics_path = out_dir / "metrics.jsonl"
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
    loaded_checkpoint_meta = dict(getattr(model, "loaded_meta", {}) or {})
    ae_backend = None
    image_replay: ImageChunkReplay | None = None
    if bool(getattr(args, "ae_trainable", False)):
        allow_legacy_ae_resume = bool(
            getattr(args, "allow_legacy_ae_resume", False)
        )
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
        if resume_artifacts.checkpoint is not None:
            _require_v14_ae_metadata(
                dict(getattr(model, "loaded_meta", {}) or {}),
                label=f"RLT checkpoint {resume_artifacts.checkpoint}",
                allow_legacy=allow_legacy_ae_resume,
            )
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
        contract = _validate_ae_contract(ae_backend)
        if (
            eval_only
            and resume_artifacts.checkpoint is None
            and str(getattr(args, "rlt_ckpt", "") or "")
        ):
            _require_v14_ae_metadata(
                loaded_checkpoint_meta,
                label=f"RLT checkpoint {args.rlt_ckpt}",
                allow_legacy=allow_legacy_ae_resume,
            )
            _require_matching_ae_contract(
                loaded_checkpoint_meta,
                contract,
                label=f"RLT checkpoint {args.rlt_ckpt}",
                allow_legacy=allow_legacy_ae_resume,
            )
        if resume_artifacts.checkpoint is not None:
            _require_matching_ae_contract(
                loaded_checkpoint_meta,
                contract,
                label=f"RLT checkpoint {resume_artifacts.checkpoint}",
                allow_legacy=allow_legacy_ae_resume,
            )
        with torch.no_grad():
            model.action_mean.zero_()
            model.action_std.fill_(1.0)
        if not torch.equal(
            model.action_mean,
            torch.zeros_like(model.action_mean),
        ) or not torch.equal(
            model.action_std,
            torch.ones_like(model.action_std),
        ):
            raise RuntimeError("Failed to install identity AE critic coordinates")
        if resume_artifacts.ae_trainable is not None:
            ae_meta = ae_backend.load_trainable(resume_artifacts.ae_trainable)
            _require_v14_ae_metadata(
                ae_meta,
                label=f"AE checkpoint {resume_artifacts.ae_trainable}",
                allow_legacy=allow_legacy_ae_resume,
            )
            _require_matching_ae_contract(
                ae_meta,
                contract,
                label=f"AE checkpoint {resume_artifacts.ae_trainable}",
                allow_legacy=allow_legacy_ae_resume,
            )
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
            _validate_ae_replay_resume(
                resume_artifacts.ae_replay,
                allow_legacy=allow_legacy_ae_resume,
            )
            image_replay = ImageChunkReplay.load_npz(
                str(resume_artifacts.ae_replay),
                max_transitions=int(
                    getattr(args, "ae_image_replay_capacity", 512)
                ),
                pos_frac=args.pos_frac,
                seed=args.seed,
                benchmark_pose_cycle=int(args.benchmark_pose_cycle),
            )
            if int(image_replay.full_action_horizon) != _AE_NATIVE_HORIZON:
                raise RuntimeError(
                    "Loaded AE replay does not use the V14 15-step horizon"
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
                full_action_horizon=_AE_NATIVE_HORIZON,
                padded_action_dim=_AE_PADDED_ACTION_DIM,
                benchmark_pose_cycle=int(args.benchmark_pose_cycle),
            )
        log.info(
            "V14 AE-as-V enabled: trainable_ae=%d params, token frozen, "
            "v_source=molmo_ae contract=%s",
            sum(p.numel() for p in ae_backend.trainable_parameters()),
            contract,
        )
        args.ae_native_coordinate_reset = _configure_ae_native_rlt_coordinates(
            args,
            model,
            loaded_checkpoint_meta,
        )
        args.ae_native_coordinate_initialized = bool(
            args.ae_native_coordinate_reset
            or loaded_checkpoint_meta.get(
                "ae_native_coordinate_initialized",
                False,
            )
        )
    else:
        args.ae_native_coordinate_reset = False
        args.ae_native_coordinate_initialized = False

    optimizers: dict[str, torch.optim.Optimizer] = {}
    should_build_optimizers = (
        not eval_only
        and (
            int(
                getattr(
                    args,
                    "max_updates_per_episode",
                    getattr(args, "updates_per_episode", 0),
                )
                or 0
            )
            > 0
            or bool(args.tune_token_online)
        )
    )
    if should_build_optimizers:
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
        saved_optimizer_states = dict(
            (getattr(model, "loaded_meta", {}) or {}).get(
                "optimizer_states",
                {},
            )
        )
        if (
            ae_backend is not None
            and resume_artifacts.checkpoint is not None
            and int(loaded_checkpoint_meta.get("online_schema_version", -1) or -1)
            == _V14_ONLINE_SCHEMA
            and set(saved_optimizer_states) != set(optimizers)
        ):
            raise RuntimeError(
                "V14 checkpoint optimizer set does not match runtime: "
                f"saved={sorted(saved_optimizer_states)}, "
                f"runtime={sorted(optimizers)}"
            )
        for name, optimizer in optimizers.items():
            coordinate_reset_optimizer = bool(
                args.ae_native_coordinate_reset
                and name in {"critic", "guide"}
            )
            if name in saved_optimizer_states and not coordinate_reset_optimizer:
                optimizer.load_state_dict(saved_optimizer_states[name])
        if saved_optimizer_states:
            missing_optimizer_states = sorted(
                set(saved_optimizer_states) - set(optimizers)
            )
            if missing_optimizer_states:
                raise RuntimeError(
                    "Checkpoint optimizer state has unmatched optimizers: "
                    f"{missing_optimizer_states}"
                )
        recovered_critic_heads = _parse_critic_head_indices(
            getattr(args, "recover_critic_heads", ""),
            model.n_critics,
        )
        if recovered_critic_heads:
            reset_parameters = model.reinitialize_critic_heads(
                recovered_critic_heads,
                seed=_stable_seed(
                    args.seed,
                    "critic_recovery",
                    sum(
                        (position + 1) * (head + 1)
                        for position, head in enumerate(recovered_critic_heads)
                    ),
                    len(recovered_critic_heads),
                ),
            )
            critic_optimizer = optimizers["critic"]
            for parameter in reset_parameters:
                critic_optimizer.state.pop(parameter, None)
            log.warning(
                "Explicitly reinitialized critic heads %s and their target "
                "copies; cleared corresponding optimizer state",
                recovered_critic_heads,
            )
        args.recovered_critic_heads = recovered_critic_heads
    else:
        args.recovered_critic_heads = []

    replay = ChunkReplay(
        max_transitions=args.replay_capacity,
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
        z_dim=Z_DIM,
        pos_frac=args.pos_frac,
        seed=args.seed,
        benchmark_pose_cycle=int(args.benchmark_pose_cycle),
    )
    if args.replay_out and Path(args.replay_out).is_file() and not args.no_resume:
        try:
            replay = ChunkReplay.load_npz(
                args.replay_out,
                max_transitions=args.replay_capacity,
                pos_frac=args.pos_frac,
                seed=args.seed,
                benchmark_pose_cycle=int(args.benchmark_pose_cycle),
            )
            log.info("Resumed chunk replay %s (%d transitions)", args.replay_out, len(replay))
        except Exception as error:  # noqa: BLE001
            if resume_artifacts.checkpoint is not None:
                raise RuntimeError(
                    f"Failed to load replay paired with resumed model: {args.replay_out}"
                ) from error
            log.warning("Failed to resume chunk replay %s: %s", args.replay_out, error)
    if (
        not eval_only
        and resume_artifacts.checkpoint is not None
        and int(loaded_checkpoint_meta.get("online_schema_version", -1) or -1)
        == _V14_ONLINE_SCHEMA
    ):
        if "valid_episodes" not in loaded_checkpoint_meta:
            raise RuntimeError("V14 checkpoint is missing valid_episodes")
        expected_episodes = int(loaded_checkpoint_meta["valid_episodes"])
        replay_episode_count = int(getattr(replay, "n_episodes", -1))
        if replay_episode_count != expected_episodes:
            raise RuntimeError(
                "V14 checkpoint/chunk replay transaction mismatch: "
                f"checkpoint_episodes={expected_episodes}, "
                f"replay_episodes={replay_episode_count}"
            )
        if image_replay is not None:
            image_episode_count = int(
                getattr(image_replay, "n_episodes", -1)
            )
            if image_episode_count != expected_episodes:
                raise RuntimeError(
                    "V14 checkpoint/AE replay transaction mismatch: "
                    f"checkpoint_episodes={expected_episodes}, "
                    f"replay_episodes={image_episode_count}"
                )
    token_replay = TokenReplay(
        max_seq=args.token_max_seq,
        token_dim=model.feature_dim,
    )
    chunk_token_replay = TokenReplay(
        max_seq=args.token_max_seq,
        token_dim=model.feature_dim,
    )
    export_tokens = bool(getattr(args, "export_offline_tokens", False)) or bool(
        str(getattr(args, "token_replay_out", "") or "")
    ) or bool(str(getattr(args, "chunk_token_replay_out", "") or ""))
    if export_tokens and not eval_only:
        # One shared buffer: token_replay.npz and chunk_token_replay.npz are the
        # same sequences; duplicating them doubled RAM (~37GB/process) and made
        # the final flush kill the V17 collect.
        token_replay = chunk_token_replay
        # Keep all tokens for offline AE re-encode (no capacity trim).
        args.token_replay_capacity = max(
            int(getattr(args, "token_replay_capacity", 1)),
            int(getattr(args, "replay_capacity", 200000)),
            500000,
        )
        args.retain_tokens = True
        # Collect-only / pretrain export should keep reference policy.
        if int(getattr(args, "updates_per_episode", 1)) <= 0:
            args.actor_mixture_prob = 0.0
            args.always_collect_actor = False

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
    last_gate = GateStatus(
        actor_ready=False,
        guide_ready=False,
        deploy_actor=False,
        deploy_guide=False,
        actor_lcb=0.0,
        actor_advantage=0.0,
        actor_sensitivity=0.0,
        guide_lcb=0.0,
        guide_advantage=0.0,
        guide_sensitivity=0.0,
        critic_health=False,
        critic_health_fields={},
    )
    last_update = UpdateStatus(
        skipped_reason="",
        stop_reason="not_run",
        rounds=0,
        critic_updates=0,
        actor_updates=0,
        guide_updates=0,
        token_updates=0,
        elapsed_sec=0.0,
    )
    last_coordinate_diagnostics: dict[str, float] = {}
    last_logged_episode = -1
    last_logged_cycle = -1
    warned_missing_tokens = False
    validation_rows: list[dict[str, Any]] = []

    metrics_resume_row = (
        None
        if args.no_resume or eval_only
        else _load_metrics_resume(out_dir)
    )
    checkpoint_gate_meta = {
        key: value
        for key, value in loaded_checkpoint_meta.items()
        if key
        not in {
            "optimizer_states",
            "process_rng_state",
            "policy_rng_state",
        }
    }
    checkpoint_is_v14 = (
        resume_artifacts.checkpoint is not None
        and int(loaded_checkpoint_meta.get("online_schema_version", -1) or -1)
        == _V14_ONLINE_SCHEMA
    )
    if checkpoint_is_v14:
        # The checkpoint row, model/optimizer tensors, and RNG states are one
        # atomic file. Metrics are appended later and may lag after a crash.
        resume_row = dict(checkpoint_gate_meta)
        if metrics_resume_row is not None and (
            int(metrics_resume_row.get("env_steps", -1) or -1)
            != int(resume_row.get("env_steps", -1) or -1)
            or int(metrics_resume_row.get("cycle", -1) or -1)
            != int(resume_row.get("cycle", -1) or -1)
        ):
            log.warning(
                "Ignoring metrics row that does not match the V14 checkpoint: "
                "metrics_steps=%s checkpoint_steps=%s metrics_cycle=%s "
                "checkpoint_cycle=%s",
                metrics_resume_row.get("env_steps"),
                resume_row.get("env_steps"),
                metrics_resume_row.get("cycle"),
                resume_row.get("cycle"),
            )
    else:
        resume_row = metrics_resume_row
        if (
            resume_row is None
            and resume_artifacts.checkpoint is not None
            and checkpoint_gate_meta
        ):
            resume_row = dict(checkpoint_gate_meta)
        if resume_row is not None:
            checkpoint_gate_meta.update(resume_row)
    if resume_row is not None and not eval_only:
        env_steps = int(resume_row.get("env_steps", 0) or 0)
        valid_episodes = int(resume_row.get("valid_episodes", 0) or 0)
        skipped_episodes = int(resume_row.get("skipped_episodes", 0) or 0)
        rate = float(resume_row.get("cumulative_success_rate", 0.0) or 0.0)
        successes = int(
            resume_row.get(
                "successes",
                int(round(rate * max(valid_episodes, 1)))
                if valid_episodes
                else 0,
            )
            or 0
        )
        cycle = int(
            resume_row.get(
                "cycle",
                valid_episodes + skipped_episodes,
            )
            or 0
        )
        recent.extend(
            float(value)
            for value in list(resume_row.get("recent_outcomes") or [])[
                -args.window_episodes :
            ]
        )
        elapsed_before_resume = float(resume_row.get("elapsed_sec", 0.0) or 0.0)
        if elapsed_before_resume > 0.0:
            start_time = time.time() - elapsed_before_resume
        last_logged_episode = valid_episodes
        last_logged_cycle = cycle
        log.info(
            "Resumed counters steps=%d eps=%d skipped=%d successes=%d",
            env_steps,
            valid_episodes,
            skipped_episodes,
            successes,
        )
    if resume_artifacts.checkpoint is not None and not eval_only:
        policy_state = loaded_checkpoint_meta.get("policy_rng_state")
        if policy_state is not None:
            policy.load_rng_state_dict(dict(policy_state))
        elif int(loaded_checkpoint_meta.get("online_schema_version", -1) or -1) == _V14_ONLINE_SCHEMA:
            raise RuntimeError("V14 checkpoint is missing policy_rng_state")
        process_rng_state = loaded_checkpoint_meta.get("process_rng_state")
        if process_rng_state is not None:
            _restore_process_rng_state(dict(process_rng_state))
        elif int(loaded_checkpoint_meta.get("online_schema_version", -1) or -1) == _V14_ONLINE_SCHEMA:
            raise RuntimeError("V14 checkpoint is missing process_rng_state")

    snapshot_episodes = _parse_snapshot_episodes(
        str(getattr(args, "snapshot_episodes", ""))
    )
    if not eval_only and valid_episodes in snapshot_episodes:
        startup_snapshot_meta = {
            **checkpoint_gate_meta,
            "online_schema_version": _V14_ONLINE_SCHEMA,
            "config_name": str(args.config_name),
            "ae_native_action_coordinates": bool(ae_backend is not None),
            "ae_native_coordinate_reset": bool(
                getattr(args, "ae_native_coordinate_reset", False)
            ),
            "ae_native_coordinate_initialized": bool(
                getattr(args, "ae_native_coordinate_initialized", False)
            ),
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

    cycles_this_process = 0
    empirical_tracker = EmpiricalGateTracker()
    while (
        not _STOP_REQUESTED
        and env_steps < args.target_env_steps
        and (
            args.max_valid_episodes <= 0
            or valid_episodes < args.max_valid_episodes
        )
    ):
        cycles_this_process += 1
        if ae_backend is None:
            health = _server_health(args.server_host, args.server_port)
            if health is None:
                log.warning("MolmoAct2 server is unavailable; waiting")
                health = _wait_for_server(
                    args.server_host, args.server_port, args.server_wait_sec
                )
                _validate_server_features(health)
                policy.prepare_model()

        phase = actor_phase_for_episode(
            int(valid_episodes),
            bc_episodes=int(getattr(args, "actor_bc_episodes", 50)),
            q_ramp_episodes=int(getattr(args, "q_ramp_episodes", 0)),
            residual_clip=float(getattr(args, "residual_clip", 0.02)),
            advantage_clip=float(getattr(args, "advantage_clip", 0.05)),
            endpoint_ref_mse_max=float(
                getattr(args, "endpoint_ref_mse_max", 0.01)
            ),
            deploy_ref_dropout=float(getattr(args, "train_ref_dropout", 0.0)),
        )
        policy.residual_clip = phase.residual_clip
        policy.collect_episode_index = int(valid_episodes)
        policy.actor_bc_episodes = int(getattr(args, "actor_bc_episodes", 50))
        policy.always_collect_actor = bool(
            getattr(args, "always_collect_actor", False)
        )
        after = getattr(args, "always_collect_after_episodes", None)
        policy.always_collect_after_episodes = (
            int(getattr(args, "actor_bc_episodes", 50))
            if after is None
            else int(after)
        )

        gate = _gate_status(
            args,
            model,
            replay,
            valid_episodes,
            device,
            ae_backend=ae_backend,
            image_replay=image_replay,
            checkpoint_meta=checkpoint_gate_meta,
            policy=policy,
            empirical_tracker=empirical_tracker,
        )
        policy.deploy_actor = gate.deploy_actor
        policy.deploy_guide = gate.deploy_guide
        # Gate-aware collect mixture (V18): keep actor rare until deploy opens.
        base_mix = float(getattr(args, "actor_mixture_prob", 0.0))
        pre_mix = getattr(args, "actor_mixture_prob_pre_gate", None)
        post_mix = getattr(args, "actor_mixture_prob_post_gate", None)
        if pre_mix is None:
            pre_mix = base_mix
        if post_mix is None:
            post_mix = base_mix
        policy.actor_mixture_prob = float(
            post_mix if gate.deploy_actor else pre_mix
        )
        model.eval()

        episode_idx = shard_start + (cycle % shard_size)
        cycle += 1
        policy.begin_episode(cycle)
        episode_dir = tmp_rollouts / f"ep_{cycle:08d}"
        shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)
        episode_start = time.time()
        rollout_ok = True
        success = False
        trajectory: dict[str, Any] = {"n_steps": 0, "token_batches": [], "residual_rms": 0.0}
        n_steps = 0
        last_q, last_actor, last_guide, last_token = {}, {}, {}, {}
        last_update = UpdateStatus(
            skipped_reason="",
            stop_reason="not_run",
            rounds=0,
            critic_updates=0,
            actor_updates=0,
            guide_updates=0,
            token_updates=0,
            elapsed_sec=0.0,
        )
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
                            full_references=trajectory["full_references"],
                            full_executed=trajectory["full_executed"],
                            sources_native=trajectory["sources_native"],
                        )
                    for tokens, mask in trajectory["token_batches"]:
                        # Shared buffer when export_tokens (token_replay is
                        # chunk_token_replay); otherwise both may differ by trim.
                        chunk_token_replay.add(tokens, mask)
                        if token_replay is not chunk_token_replay:
                            token_replay.add(tokens, mask)
                    if not export_tokens:
                        token_overflow = len(token_replay) - args.token_replay_capacity
                        if token_overflow > 0:
                            del token_replay.tokens[:token_overflow]
                            del token_replay.masks[:token_overflow]
            if rollout_ok and n_steps > 0 and not eval_only:
                (
                    last_q,
                    last_actor,
                    last_guide,
                    last_token,
                    last_update,
                ) = _train_after_episode(
                    args,
                    model,
                    optimizers,
                    replay,
                    token_replay,
                    device,
                    ae_backend=ae_backend,
                    image_replay=image_replay,
                    policy=policy,
                    valid_episodes=valid_episodes,
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
                validation_rows.append(
                    {
                        "episode_idx": int(episode_idx),
                        "valid": False,
                        "success": False,
                        "n_steps": int(n_steps),
                        "deploy_policy": str(args.deploy_policy),
                        "gate_actor_ready": bool(gate.actor_ready),
                        "gate_guide_ready": bool(gate.guide_ready),
                        "gate_deploy_actor": bool(gate.deploy_actor),
                        "gate_deploy_guide": bool(gate.deploy_guide),
                        "gate_actor_block_reason": gate.actor_block_reason,
                        "gate_guide_block_reason": gate.guide_block_reason,
                    }
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
        if rollout_ok and n_steps > 0 and not eval_only:
            empirical_tracker.record(
                used_actor=bool(trajectory.get("episode_used_actor", False)),
                success=bool(success),
            )
        last_coordinate_diagnostics = dict(
            trajectory.get("coordinate_diagnostics") or {}
        )
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
            policy=policy,
            empirical_tracker=empirical_tracker,
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
                image_replay=image_replay,
                update_status=last_update,
                coordinate_diagnostics=last_coordinate_diagnostics,
                policy=policy,
                model=model,
                episode_info=trajectory,
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
            validation_rows.append(validation_row)
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
        token_every = int(
            getattr(args, "token_ckpt_every_episodes", 0) or 0
        )
        flush_tokens = bool(export_tokens) and (
            token_every > 0 and valid_episodes % token_every == 0
        )
        # Token export cadence must also open the persist path (otherwise
        # token_ckpt_every=25 never fires when ckpt_every=10).
        if flush_tokens:
            should_log = True
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
                image_replay=image_replay,
                update_status=last_update,
                coordinate_diagnostics=last_coordinate_diagnostics,
                policy=policy,
                model=model,
                episode_info=trajectory,
            )
            # Persist outside the EGL lock (already released) so NFS stalls
            # cannot block other trainers on this GPU.
            # Offline token export buffers grow to multi-GB; skipping intermediate
            # NPZ dumps avoids NFS stalls that freeze collectors for tens of minutes.
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
                token_replay=token_replay if export_tokens else None,
                token_replay_out=str(getattr(args, "token_replay_out", "")),
                chunk_token_replay=chunk_token_replay if export_tokens else None,
                chunk_token_replay_out=str(
                    getattr(args, "chunk_token_replay_out", "")
                ),
                optimizers=optimizers,
                policy=policy,
                save_token_replays=flush_tokens,
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
            last_logged_cycle = cycle
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

    if not eval_only and resume_row is not None and cycles_this_process == 0:
        shutil.rmtree(tmp_rollouts, ignore_errors=True)
        log.info(
            "Resume target already complete at steps=%d eps=%d; "
            "leaving checkpoint and metrics unchanged",
            env_steps,
            valid_episodes,
        )
        return

    already_persisted = bool(
        last_logged_episode == valid_episodes
        and last_logged_cycle == cycle
        and (out_dir / "rlt_cf_latest.pt").exists()
    )
    final_row = (
        dict(model.loaded_meta)
        if already_persisted
        else _metrics_row(
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
            image_replay=image_replay,
            update_status=last_update,
            coordinate_diagnostics=last_coordinate_diagnostics,
            policy=policy,
            model=model,
            episode_info=trajectory,
        )
    )
    if eval_only:
        summary = {
            **final_row,
            "eval_only": True,
            "deploy_policy": str(args.deploy_policy),
            "stopped_by_signal": bool(_STOP_REQUESTED),
            "validation_rows": validation_rows,
        }
        shutil.rmtree(tmp_rollouts, ignore_errors=True)
        try:
            _io_retry(
                f"validation_summary.json in {out_dir}",
                lambda: (out_dir / "validation_summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n",
                    encoding="utf-8",
                ),
            )
            def _write_results() -> None:
                with (out_dir / "validation_results.jsonl").open(
                    "w", encoding="utf-8"
                ) as stream:
                    for row in validation_rows:
                        stream.write(json.dumps(row) + "\n")

            _io_retry(
                f"validation_results.jsonl in {out_dir}",
                _write_results,
            )
        except Exception as error:  # noqa: BLE001
            log.error("Failed to write validation artifacts: %s", error)
            raise
        log.info("Validation done: %s", json.dumps(summary))
        return

    if already_persisted and not export_tokens:
        checkpoint = out_dir / "rlt_cf_latest.pt"
    else:
        checkpoint = _persist_training_state(
            model=model,
            out_dir=out_dir,
            metrics_path=metrics_path,
            env_steps=env_steps,
            row=final_row,
            replay=replay,
            replay_out=args.replay_out,
            required=bool(export_tokens),
            ae_backend=ae_backend,
            ae_image_replay=image_replay,
            ae_image_replay_out=str(
                getattr(args, "ae_image_replay_out", "")
            ),
            token_replay=token_replay if export_tokens else None,
            token_replay_out=str(getattr(args, "token_replay_out", "")),
            chunk_token_replay=chunk_token_replay if export_tokens else None,
            chunk_token_replay_out=str(
                getattr(args, "chunk_token_replay_out", "")
            ),
            optimizers=optimizers,
            policy=policy,
            save_token_replays=bool(export_tokens),
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
    parser.add_argument(
        "--recover_critic_heads",
        type=str,
        default="",
        help=(
            "Explicit comma-separated critic heads to reinitialize together "
            "with their target copies; recovery is persisted in run metadata."
        ),
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
    parser.add_argument(
        "--benchmark_pose_cycle",
        type=int,
        default=0,
        help=(
            "When positive, retain replay evenly across episode_id modulo "
            "this controlled benchmark pose cycle."
        ),
    )
    parser.add_argument("--pos_frac", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--min_replay_chunks", type=int, default=8)
    parser.add_argument(
        "--updates_per_episode",
        type=int,
        default=5,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max_updates_per_episode",
        type=int,
        default=None,
        help="Maximum online update rounds after one valid episode.",
    )
    parser.add_argument(
        "--max_update_sec_per_episode",
        type=float,
        default=300.0,
        help="Wall-clock cap for all updates after one valid episode.",
    )
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
    parser.add_argument(
        "--ae_microbatch_size",
        type=int,
        default=4,
        help="AE actor microbatch size used for full-endpoint accumulation.",
    )
    parser.add_argument(
        "--ae_min_success_episodes",
        type=int,
        default=3,
        help="Minimum distinct successful AE episodes before any RL update.",
    )
    parser.add_argument("--ae_image_replay_capacity", type=int, default=512)
    parser.add_argument(
        "--ae_image_replay_out",
        type=str,
        default="",
        help="Persisted AE image replay path (defaults under out_dir).",
    )
    parser.add_argument("--lr_ae", type=float, default=1e-4)
    parser.add_argument(
        "--allow_legacy_ae_resume",
        action="store_true",
        help=(
            "Explicitly allow pre-V14 AE checkpoints/replays. Unsafe by "
            "default because native source/action provenance is incomplete."
        ),
    )
    parser.add_argument("--g_min_advantage", type=float, default=0.003)
    parser.add_argument(
        "--g_min_guide_advantage",
        type=float,
        default=0.003,
        help=(
            "Require guide-vs-actor paired lower confidence bound before "
            "deploy; V14 keeps the 0.003 safety threshold."
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
        "--eval_reference_noise_std",
        type=float,
        default=None,
        help=(
            "Eval-only override: apply this normalized explore std on "
            "reference deploy (explore-tax control / reference_noise)."
        ),
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
    critic_target_group = parser.add_mutually_exclusive_group()
    critic_target_group.add_argument(
        "--critic_target_use_guide",
        dest="critic_target_use_guide",
        action="store_true",
        help="Bootstrap critic targets with actor plus guide.",
    )
    critic_target_group.add_argument(
        "--no_critic_target_use_guide",
        dest="critic_target_use_guide",
        action="store_false",
        help="Bootstrap critic targets with the actor only (V14 default).",
    )
    parser.set_defaults(critic_target_use_guide=False)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--mc_coef", type=float, default=0.1)
    parser.add_argument("--cql_coef", type=float, default=0.1)
    parser.add_argument("--cql_n_actions", type=int, default=8)
    parser.add_argument("--cql_action_radius", type=float, default=0.2)
    parser.add_argument("--ref_dropout", type=float, default=0.5)
    parser.add_argument(
        "--train_ref_dropout",
        type=float,
        default=0.0,
        help="Reference dropout used after BC warmup (V16 default 0.5 via harness).",
    )
    parser.add_argument("--actor_beta", type=float, default=1.0)
    parser.add_argument("--guide_beta", type=float, default=0.05)
    parser.add_argument(
        "--actor_mixture_prob",
        type=float,
        default=0.0,
        help="While Q-gate is closed, execute the learned actor with this probability.",
    )
    parser.add_argument(
        "--actor_mixture_prob_pre_gate",
        type=float,
        default=None,
        help="Override mixture while gate is closed (default: actor_mixture_prob).",
    )
    parser.add_argument(
        "--actor_mixture_prob_post_gate",
        type=float,
        default=None,
        help="Override mixture once gate deploys actor (default: actor_mixture_prob).",
    )
    parser.add_argument(
        "--always_collect_actor",
        action="store_true",
        help=(
            "V16/RLT paper Alg.1: after always_collect_after_episodes, always "
            "collect from π_θ regardless of the Q-gate (gate still controls guide)."
        ),
    )
    parser.add_argument(
        "--always_collect_after_episodes",
        type=int,
        default=None,
        help=(
            "Delay for --always_collect_actor (default: actor_bc_episodes). "
            "V17 sets this high so mixture can populate empirical gate first."
        ),
    )
    parser.add_argument(
        "--q_ramp_episodes",
        type=int,
        default=0,
        help="After BC warmup, linearly ramp actor q_coef 0→1 over this many episodes.",
    )
    parser.add_argument(
        "--export_offline_tokens",
        action="store_true",
        help="Persist token_replay + chunk_token_replay for offline kettle pretrain.",
    )
    parser.add_argument("--token_replay_out", type=str, default="")
    parser.add_argument("--chunk_token_replay_out", type=str, default="")
    parser.add_argument(
        "--token_ckpt_every_episodes",
        type=int,
        default=0,
        help=(
            "When exporting offline tokens, write multi-GB token NPZs every N "
            "episodes (0 = only at end). Intermediate dumps stall NFS collect."
        ),
    )
    parser.add_argument(
        "--require_empirical_gate",
        action="store_true",
        help="Require mixture empirical ΔSR LCB before opening the Q gate.",
    )
    parser.add_argument(
        "--g_min_empirical_advantage",
        type=float,
        default=0.0,
        help="Minimum empirical (actor-ref) Wilson LCB required by the empirical gate.",
    )
    parser.add_argument(
        "--empirical_min_episodes",
        type=int,
        default=16,
        help="Minimum actor and reference mixture episodes before empirical gate can pass.",
    )
    parser.add_argument(
        "--actor_bc_episodes",
        type=int,
        default=50,
        help="Episodes of BC-only actor warmup before clipped Q refinement.",
    )
    parser.add_argument(
        "--residual_clip",
        type=float,
        default=0.02,
        help="Hard per-element residual ball for actor deploy/train (normalized).",
    )
    parser.add_argument(
        "--advantage_clip",
        type=float,
        default=0.05,
        help="Clip actor advantage magnitude used in the loss.",
    )
    parser.add_argument(
        "--endpoint_ref_mse_max",
        type=float,
        default=0.01,
        help="Flow endpoint must be this close to the reference before Q term activates.",
    )
    parser.add_argument(
        "--actor_cql_coef",
        type=float,
        default=0.1,
        help="Extra CQL-style penalty on actor-proposed actions (V15 residual/flow).",
    )
    guide_ref_group = parser.add_mutually_exclusive_group()
    guide_ref_group.add_argument(
        "--guide_on_reference",
        dest="guide_on_reference",
        action="store_true",
        help="Train/deploy CF guide on frozen VLA (no learned actor).",
    )
    guide_ref_group.add_argument(
        "--no_guide_on_reference",
        dest="guide_on_reference",
        action="store_false",
        help="Compose the guide on top of the learned actor (default).",
    )
    parser.set_defaults(guide_on_reference=False)
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
    if args.max_updates_per_episode is None:
        args.max_updates_per_episode = int(args.updates_per_episode)

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
        eval_noise = getattr(args, "eval_reference_noise_std", None)
        if eval_noise is not None and float(eval_noise) > 0.0:
            args.explore_residual_std = float(eval_noise)
        else:
            args.explore_residual_std = 0.0
            args.eval_reference_noise_std = None
        args.explore_deploy_std = 0.0
        args.explore_warmup_mult = 1.0
        args.updates_per_episode = 0
        args.max_updates_per_episode = 0
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
    try:
        _parse_critic_head_indices(args.recover_critic_heads, args.n_critics)
    except ValueError as error:
        parser.error(str(error))
    if args.eval_only and str(args.recover_critic_heads).strip():
        parser.error("--recover_critic_heads is forbidden with --eval_only")
    if args.updates_per_episode < 0:
        parser.error("--updates_per_episode cannot be negative")
    if args.benchmark_pose_cycle < 0:
        parser.error("--benchmark_pose_cycle cannot be negative")
    if args.max_updates_per_episode < 0:
        parser.error("--max_updates_per_episode cannot be negative")
    if args.max_update_sec_per_episode < 0.0:
        parser.error("--max_update_sec_per_episode cannot be negative")
    if args.ae_min_success_episodes <= 0:
        parser.error("--ae_min_success_episodes must be positive")
    if args.ae_batch_size < 2:
        parser.error("--ae_batch_size must be at least 2")
    if args.ae_microbatch_size <= 0:
        parser.error("--ae_microbatch_size must be positive")
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
