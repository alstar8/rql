#!/usr/bin/env python3
"""NFS-safe coordination and promotion primitives for V20."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence


V20_SCHEMA_VERSION = 1
V20_MARKER = ".v20_monotone_incumbent"
CONFIRM_LOOKS = (128, 256, 512)
CONFIRM_ALPHAS = (0.002, 0.003, 0.005)


class StaleOperationError(RuntimeError):
    """Raised when a worker attempts to publish an obsolete operation."""


@dataclass(frozen=True)
class RoundState:
    schema_version: int
    generation: int
    round_id: int
    phase: str
    operation_id: str
    worker_count: int
    incumbent_version: int
    incumbent_checkpoint: str
    incumbent_mode: str
    rollout_seed_root: int
    candidate_checkpoint: str | None = None
    candidate_id: str | None = None
    candidate_mode: str = "actor"
    collect_wave: int = -1
    eval_stage: str | None = None
    pair_offset: int = 0
    pair_count: int = 0
    message: str = ""
    updated_at: float = 0.0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RoundState":
        return cls(
            schema_version=int(value["schema_version"]),
            generation=int(value["generation"]),
            round_id=int(value["round_id"]),
            phase=str(value["phase"]),
            operation_id=str(value["operation_id"]),
            worker_count=int(value["worker_count"]),
            incumbent_version=int(value["incumbent_version"]),
            incumbent_checkpoint=str(value["incumbent_checkpoint"]),
            incumbent_mode=str(value["incumbent_mode"]),
            rollout_seed_root=int(value["rollout_seed_root"]),
            candidate_checkpoint=(
                None
                if value.get("candidate_checkpoint") is None
                else str(value["candidate_checkpoint"])
            ),
            candidate_id=(
                None
                if value.get("candidate_id") is None
                else str(value["candidate_id"])
            ),
            candidate_mode=str(value.get("candidate_mode", "actor")),
            collect_wave=int(value.get("collect_wave", -1)),
            eval_stage=(
                None
                if value.get("eval_stage") is None
                else str(value["eval_stage"])
            ),
            pair_offset=int(value.get("pair_offset", 0)),
            pair_count=int(value.get("pair_count", 0)),
            message=str(value.get("message", "")),
            updated_at=float(value.get("updated_at", 0.0)),
        )


@dataclass(frozen=True)
class BarrierSnapshot:
    operation_id: str
    expected: int
    completed: int
    valid: int
    failures: tuple[dict[str, Any], ...]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class PairedCounts:
    pairs: int
    both_success: int
    candidate_only: int
    incumbent_only: int
    both_failure: int

    @property
    def discordant(self) -> int:
        return self.candidate_only + self.incumbent_only

    @property
    def incumbent_successes(self) -> int:
        return self.both_success + self.incumbent_only

    @property
    def candidate_successes(self) -> int:
        return self.both_success + self.candidate_only

    @property
    def gain(self) -> float:
        if self.pairs <= 0:
            return 0.0
        return (self.candidate_only - self.incumbent_only) / self.pairs


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    counts: PairedCounts
    alpha: float
    p_value: float
    point_gain: float
    gain_lower_bound: float
    clone_ok: bool
    cond_ok: bool
    action_bounds_ok: bool


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_bytes(
        Path(path),
        (
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )


def atomic_copy(source: Path, destination: Path) -> str:
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_handle, os.fdopen(
            descriptor,
            "wb",
        ) as output_handle:
            while True:
                block = input_handle.read(1024 * 1024)
                if not block:
                    break
                output_handle.write(block)
                digest.update(block)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return digest.hexdigest()


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def append_jsonl_locked(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    line = (
        json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def validate_run_dir(run_dir: Path) -> Path:
    run_dir = Path(run_dir).resolve()
    if not (run_dir / V20_MARKER).is_file():
        raise RuntimeError(f"{run_dir} is not an initialized V20 run")
    return run_dir


def initialize_run(
    run_dir: Path,
    *,
    worker_count: int,
    incumbent_checkpoint: Path,
    incumbent_mode: str,
    rollout_seed_root: int,
    config: dict[str, Any],
) -> RoundState:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(run_dir / ".initialize.lock"):
        marker = run_dir / V20_MARKER
        if not marker.exists():
            atomic_write_json(
                marker,
                {
                    "schema_version": V20_SCHEMA_VERSION,
                    "created_at": time.time(),
                },
            )
        for relative in (
            "checkpoints/archive",
            "checkpoints/candidates",
            "coordination/done",
            "coordination/claims",
            "coordination/heartbeats",
            "replay/journal",
            "replay/journal_index",
            "reports",
            "videos",
        ):
            (run_dir / relative).mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.json"
        normalized_config = {
            **config,
            "schema_version": V20_SCHEMA_VERSION,
            "worker_count": int(worker_count),
            "rollout_seed_root": int(rollout_seed_root),
        }
        if config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != normalized_config:
                raise RuntimeError(
                    f"resume config mismatch in {config_path}: "
                    f"{existing} != {normalized_config}"
                )
        else:
            atomic_write_json(config_path, normalized_config)
        state_path = run_dir / "round_state.json"
        if state_path.exists():
            return read_round_state(run_dir)
        state = RoundState(
            schema_version=V20_SCHEMA_VERSION,
            generation=0,
            round_id=0,
            phase="idle",
            operation_id="bootstrap",
            worker_count=int(worker_count),
            incumbent_version=0,
            incumbent_checkpoint=str(Path(incumbent_checkpoint).resolve()),
            incumbent_mode=str(incumbent_mode),
            rollout_seed_root=int(rollout_seed_root),
            updated_at=time.time(),
        )
        atomic_write_json(state_path, asdict(state))
        atomic_write_json(
            run_dir / "uid_state.json",
            {"schema_version": V20_SCHEMA_VERSION, "next_uid": 20_000_000},
        )
        return state


def read_round_state(run_dir: Path) -> RoundState:
    run_dir = validate_run_dir(run_dir)
    value = json.loads(
        (run_dir / "round_state.json").read_text(encoding="utf-8")
    )
    state = RoundState.from_dict(value)
    if state.schema_version != V20_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported V20 state schema {state.schema_version}"
        )
    return state


def publish_operation(
    run_dir: Path,
    *,
    phase: str,
    round_id: int,
    operation_id: str,
    incumbent_version: int | None = None,
    incumbent_checkpoint: Path | None = None,
    incumbent_mode: str | None = None,
    candidate_checkpoint: Path | None = None,
    candidate_id: str | None = None,
    candidate_mode: str = "actor",
    collect_wave: int = -1,
    eval_stage: str | None = None,
    pair_offset: int = 0,
    pair_count: int = 0,
    message: str = "",
) -> RoundState:
    run_dir = validate_run_dir(run_dir)
    with file_lock(run_dir / ".state.lock"):
        previous = read_round_state(run_dir)
        state = RoundState(
            schema_version=V20_SCHEMA_VERSION,
            generation=previous.generation + 1,
            round_id=int(round_id),
            phase=str(phase),
            operation_id=str(operation_id),
            worker_count=previous.worker_count,
            incumbent_version=(
                previous.incumbent_version
                if incumbent_version is None
                else int(incumbent_version)
            ),
            incumbent_checkpoint=(
                previous.incumbent_checkpoint
                if incumbent_checkpoint is None
                else str(Path(incumbent_checkpoint).resolve())
            ),
            incumbent_mode=(
                previous.incumbent_mode
                if incumbent_mode is None
                else str(incumbent_mode)
            ),
            rollout_seed_root=previous.rollout_seed_root,
            candidate_checkpoint=(
                None
                if candidate_checkpoint is None
                else str(Path(candidate_checkpoint).resolve())
            ),
            candidate_id=candidate_id,
            candidate_mode=str(candidate_mode),
            collect_wave=int(collect_wave),
            eval_stage=eval_stage,
            pair_offset=int(pair_offset),
            pair_count=int(pair_count),
            message=str(message),
            updated_at=time.time(),
        )
        atomic_write_json(run_dir / "round_state.json", asdict(state))
    return state


def publish_stop(run_dir: Path, message: str) -> RoundState:
    previous = read_round_state(run_dir)
    return publish_operation(
        run_dir,
        phase="stop",
        round_id=previous.round_id,
        operation_id=f"stop_g{previous.generation + 1:06d}",
        message=message,
    )


def derive_seed(root_seed: int, *components: Any) -> int:
    payload = "|".join(
        [str(int(root_seed)), *(str(component) for component in components)]
    ).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        "little",
    ) & ((1 << 63) - 1)


def cumulative_alpha_spent(
    confirmed_attempt: int,
    look_index: int,
    *,
    alphas: Sequence[float] = CONFIRM_ALPHAS,
    max_attempts: int = 5,
) -> float:
    confirmed_attempt = int(confirmed_attempt)
    look_index = int(look_index)
    if not 1 <= confirmed_attempt <= int(max_attempts):
        raise ValueError("confirmed_attempt is outside the allocated budget")
    if not 0 <= look_index < len(alphas):
        raise ValueError("look_index is outside the allocated budget")
    return float(
        (confirmed_attempt - 1) * math.fsum(alphas)
        + math.fsum(alphas[: look_index + 1])
    )


def pair_seed(state: RoundState, pair_id: int) -> int:
    return derive_seed(
        state.rollout_seed_root,
        "paired-eval",
        state.round_id,
        state.eval_stage,
        int(pair_id),
    )


def collection_seed(state: RoundState, worker_id: int) -> int:
    return derive_seed(
        state.rollout_seed_root,
        "collect",
        state.round_id,
        state.collect_wave,
        int(worker_id),
    )


def allocate_global_uid(run_dir: Path) -> int:
    run_dir = validate_run_dir(run_dir)
    path = run_dir / "uid_state.json"
    with file_lock(run_dir / ".uid.lock"):
        state = json.loads(path.read_text(encoding="utf-8"))
        uid = int(state["next_uid"])
        state["next_uid"] = uid + 1
        atomic_write_json(path, state)
    return uid


def ensure_next_uid(run_dir: Path, minimum_next_uid: int) -> None:
    run_dir = validate_run_dir(run_dir)
    path = run_dir / "uid_state.json"
    with file_lock(run_dir / ".uid.lock"):
        state = json.loads(path.read_text(encoding="utf-8"))
        state["next_uid"] = max(
            int(state["next_uid"]),
            int(minimum_next_uid),
        )
        atomic_write_json(path, state)


def worker_done_path(
    run_dir: Path,
    operation_id: str,
    worker_id: int,
) -> Path:
    return (
        Path(run_dir)
        / "coordination"
        / "done"
        / operation_id
        / f"worker_{int(worker_id):03d}.json"
    )


@contextmanager
def claim_worker_operation(
    run_dir: Path,
    operation_id: str,
    worker_id: int,
) -> Iterator[bool]:
    run_dir = validate_run_dir(run_dir)
    claim = (
        run_dir
        / "coordination"
        / "claims"
        / operation_id
        / f"worker_{int(worker_id):03d}.lock"
    )
    claim.parent.mkdir(parents=True, exist_ok=True)
    with claim.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield not worker_done_path(
                run_dir,
                operation_id,
                worker_id,
            ).exists()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_worker_done(
    run_dir: Path,
    state: RoundState,
    worker_id: int,
    *,
    valid: bool,
    payload: dict[str, Any],
) -> Path:
    run_dir = validate_run_dir(run_dir)
    current = read_round_state(run_dir)
    if current.operation_id != state.operation_id:
        raise StaleOperationError(
            f"operation advanced from {state.operation_id} "
            f"to {current.operation_id}"
        )
    path = worker_done_path(run_dir, state.operation_id, worker_id)
    atomic_write_json(
        path,
        {
            "schema_version": V20_SCHEMA_VERSION,
            "operation_id": state.operation_id,
            "generation": state.generation,
            "round_id": state.round_id,
            "phase": state.phase,
            "worker_id": int(worker_id),
            "valid": bool(valid),
            "completed_at": time.time(),
            **payload,
        },
    )
    return path


def write_worker_heartbeat(
    run_dir: Path,
    worker_id: int,
    *,
    operation_id: str,
    status: str,
) -> None:
    run_dir = validate_run_dir(run_dir)
    atomic_write_json(
        run_dir
        / "coordination"
        / "heartbeats"
        / f"worker_{int(worker_id):03d}.json",
        {
            "worker_id": int(worker_id),
            "operation_id": str(operation_id),
            "status": str(status),
            "timestamp": time.time(),
            "pid": os.getpid(),
        },
    )


def barrier_snapshot(
    run_dir: Path,
    operation_id: str,
    worker_count: int,
) -> BarrierSnapshot:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for worker_id in range(int(worker_count)):
        path = worker_done_path(run_dir, operation_id, worker_id)
        if not path.is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("operation_id") != operation_id:
            continue
        rows.append(row)
        if not bool(row.get("valid", False)):
            failures.append(row)
    return BarrierSnapshot(
        operation_id=operation_id,
        expected=int(worker_count),
        completed=len(rows),
        valid=len(rows) - len(failures),
        failures=tuple(failures),
        rows=tuple(rows),
    )


def wait_for_barrier(
    run_dir: Path,
    operation_id: str,
    worker_count: int,
    *,
    timeout_sec: float,
    poll_sec: float = 1.0,
) -> BarrierSnapshot:
    deadline = time.monotonic() + float(timeout_sec)
    while True:
        snapshot = barrier_snapshot(run_dir, operation_id, worker_count)
        if snapshot.completed >= int(worker_count):
            return snapshot
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"barrier {operation_id} timed out: "
                f"{snapshot.completed}/{worker_count}"
            )
        state = read_round_state(run_dir)
        if state.phase == "stop":
            raise RuntimeError(f"run stopped while waiting for {operation_id}")
        time.sleep(max(0.01, float(poll_sec)))


def register_journal_episode(
    run_dir: Path,
    entry: dict[str, Any],
) -> bool:
    run_dir = validate_run_dir(run_dir)
    uid = int(entry["trajectory_uid"])
    index_path = run_dir / "replay" / "journal_index" / f"{uid}.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            index_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError:
        existing = json.loads(index_path.read_text(encoding="utf-8"))
        if existing != entry:
            raise RuntimeError(
                f"trajectory_uid {uid} already has different journal metadata"
            )
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(entry, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    append_jsonl_locked(run_dir / "replay" / "journal.jsonl", entry)
    return True


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def paired_counts(rows: Sequence[dict[str, Any]]) -> PairedCounts:
    both_success = 0
    candidate_only = 0
    incumbent_only = 0
    both_failure = 0
    seen_pairs: set[int] = set()
    for row in rows:
        if not bool(row.get("valid", True)):
            raise ValueError("paired outcome contains an invalid row")
        pair_id = int(row["pair_id"])
        if pair_id in seen_pairs:
            raise ValueError(f"duplicate pair_id {pair_id}")
        seen_pairs.add(pair_id)
        incumbent = bool(row["incumbent_success"])
        candidate = bool(row["candidate_success"])
        if incumbent and candidate:
            both_success += 1
        elif candidate:
            candidate_only += 1
        elif incumbent:
            incumbent_only += 1
        else:
            both_failure += 1
    return PairedCounts(
        pairs=len(rows),
        both_success=both_success,
        candidate_only=candidate_only,
        incumbent_only=incumbent_only,
        both_failure=both_failure,
    )


def binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    successes = int(successes)
    trials = int(trials)
    probability = float(probability)
    if not 0 <= successes <= trials:
        raise ValueError("successes must lie in [0, trials]")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must lie in [0, 1]")
    if successes <= 0:
        return 1.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    return math.fsum(
        math.comb(trials, value)
        * (probability**value)
        * ((1.0 - probability) ** (trials - value))
        for value in range(successes, trials + 1)
    )


def exact_mcnemar_p_value(counts: PairedCounts) -> float:
    discordant = counts.discordant
    if discordant <= 0:
        return 1.0
    return binomial_upper_tail(
        counts.candidate_only,
        discordant,
        0.5,
    )


def clopper_pearson_lower(
    successes: int,
    trials: int,
    alpha: float,
) -> float:
    successes = int(successes)
    trials = int(trials)
    alpha = float(alpha)
    if trials <= 0 or successes <= 0:
        return 0.0
    if successes > trials:
        raise ValueError("successes cannot exceed trials")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    low = 0.0
    high = 1.0
    for _ in range(80):
        midpoint = 0.5 * (low + high)
        if binomial_upper_tail(successes, trials, midpoint) < alpha:
            low = midpoint
        else:
            high = midpoint
    return 0.5 * (low + high)


def paired_gain_lower_bound(
    counts: PairedCounts,
    alpha: float,
) -> float:
    """Conservative bound for q*(2*theta-1) on discordant outcomes."""

    if counts.pairs <= 0 or counts.discordant <= 0:
        return 0.0
    split_alpha = float(alpha) / 2.0
    discordance_lower = clopper_pearson_lower(
        counts.discordant,
        counts.pairs,
        split_alpha,
    )
    candidate_win_lower = clopper_pearson_lower(
        counts.candidate_only,
        counts.discordant,
        split_alpha,
    )
    win_margin_lower = 2.0 * candidate_win_lower - 1.0
    if win_margin_lower <= 0.0:
        # q <= 1, so the margin itself is a valid lower bound when negative.
        return win_margin_lower
    return discordance_lower * win_margin_lower


def decide_promotion(
    rows: Sequence[dict[str, Any]],
    *,
    alpha: float,
    min_gain: float = 0.03,
    clone_ok: bool,
    cond_ok: bool,
    action_bounds_ok: bool,
) -> PromotionDecision:
    counts = paired_counts(rows)
    p_value = exact_mcnemar_p_value(counts)
    lower_bound = paired_gain_lower_bound(counts, alpha)
    reasons: list[str] = []
    if not clone_ok:
        reasons.append("clone_check_failed")
    if not cond_ok:
        reasons.append("cond_drift_check_failed")
    if not action_bounds_ok:
        reasons.append("action_bounds_failed")
    if counts.gain < float(min_gain):
        reasons.append("gain_below_threshold")
    if p_value > float(alpha):
        reasons.append("paired_test_not_significant")
    if lower_bound <= 0.0:
        reasons.append("gain_lower_bound_not_positive")
    promote = not reasons
    return PromotionDecision(
        promote=promote,
        reason="promote" if promote else ",".join(reasons),
        counts=counts,
        alpha=float(alpha),
        p_value=p_value,
        point_gain=counts.gain,
        gain_lower_bound=lower_bound,
        clone_ok=bool(clone_ok),
        cond_ok=bool(cond_ok),
        action_bounds_ok=bool(action_bounds_ok),
    )


def decision_to_dict(decision: PromotionDecision) -> dict[str, Any]:
    value = asdict(decision)
    value["counts"] = asdict(decision.counts)
    return value


def operation_rows(snapshot: BarrierSnapshot) -> list[dict[str, Any]]:
    return [
        dict(row["result"])
        for row in sorted(
            snapshot.rows,
            key=lambda value: int(value["worker_id"]),
        )
        if bool(row.get("valid", False))
    ]


def clean_operation_artifacts(run_dir: Path, operation_id: str) -> None:
    """Delete only transient claim files after a completed operation.

    NFS can leave a directory non-empty for a beat after unlinking its
    children. Claim files are disposable once the barrier has completed, so
    leftover residue is logged by the caller rather than failing the round.
    """

    run_dir = validate_run_dir(run_dir)
    claim_dir = run_dir / "coordination" / "claims" / operation_id
    if not claim_dir.is_dir():
        return
    for attempt in range(8):
        try:
            shutil.rmtree(claim_dir)
            return
        except OSError:
            if not claim_dir.exists():
                return
            for child in list(claim_dir.glob("*")):
                try:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
                except OSError:
                    continue
            try:
                claim_dir.rmdir()
                return
            except OSError:
                time.sleep(0.05 * (attempt + 1))
    shutil.rmtree(claim_dir, ignore_errors=True)
