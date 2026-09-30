"""Record (observation, reference chunk, teacher chunk) triples for V25 distillation.

The iterated action-expert distillation needs, at every corrected decision, the exact
input the VLA saw and the chunk the deployed teacher `(expert + V + G)` committed. The
student action expert is then trained so that `student.sample_actions(obs) ≈ teacher`.

Recording is client-side, in `Pi05RLPolicy`: the teacher chunk is produced by the
corrector there, and only there do the observation and the corrected chunk coexist.
Each collector writes its own episode files (named with the pid), so concurrent workers
never collide -- the same trick as `token_recorder.py`.

gOn commits `TEACHER_CHUNK=8` steps per replan; the VLA emits `VLA_HORIZON=16`. Two
successive gOn chunks are stitched into one 16-step student target at the first
observation (closed-loop: the second 8 is gOn at the next decision). The last unpaired
chunk of an episode is padded with that plan's unused VLA tail so the prefix is not
dropped.

Each finished episode is one npz, tagged with success and env-step length so a round
can keep the best 500 trajectories (successes first, shorter successes before longer
ones, then longer failures).

    external_cam  (N, 224, 224, 3) uint8
    wrist_cam     (N, 224, 224, 3) uint8
    state         (N, 8)  float32   arm positions + gripper, as sent to the server
    instruction   (N,)    unicode
    reference     (N, 16, 8) float32  the expert's raw chunk at the first observation
    teacher       (N, 16, 8) float32  stitched deployed chunks (distillation target)
    executed      (N, 16, 8) float32  stitched exploring chunks the collector ran
    success       () int8
    steps         () int32

All chunks are in the server's native space: arm joint deltas, gripper absolute.

`teacher` and `executed` differ because collectors explore while the number being
preserved belongs to the greedy policy; `teacher` is the greedy chunk at the same state
and is what the student is trained on. `executed` is kept so the gap between the two is
measurable after the fact rather than assumed.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

TEACHER_CHUNK = 8
VLA_HORIZON = 16
ACTION_DIM = 8
DEFAULT_KEEP_BEST = 500
_EPISODE_NAME = re.compile(r"_ok([01])_n(\d+)\.npz$")


def _chunk8(action: np.ndarray) -> np.ndarray:
    rows = np.asarray(action, dtype=np.float32).reshape(-1, ACTION_DIM)
    if rows.shape[0] < TEACHER_CHUNK:
        raise ValueError(f"teacher/executed needs {TEACHER_CHUNK} steps, got {rows.shape}")
    return rows[:TEACHER_CHUNK].copy()


def stitch_teachers(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Two committed 8-step gOn chunks → one 16-step VLA-shaped target."""
    return np.concatenate([_chunk8(first), _chunk8(second)], axis=0)


def pad_with_reference_tail(head: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Last unpaired 8-step chunk of an episode: keep the prefix, VLA tail fills 8:16."""
    ref = np.asarray(reference, dtype=np.float32).reshape(-1, ACTION_DIM)
    if ref.shape[0] < VLA_HORIZON:
        raise ValueError(f"reference needs {VLA_HORIZON} steps, got {ref.shape}")
    return np.concatenate([_chunk8(head), ref[TEACHER_CHUNK:VLA_HORIZON]], axis=0)


def trajectory_rank(success: int, steps: int) -> tuple[int, int]:
    """Sort key, higher is better: successes first, then shorter successes, then longer failures."""
    if int(success):
        return (1, -int(steps))
    return (0, int(steps))


def parse_trajectory_meta(path: Path) -> tuple[int, int]:
    """(success, env steps) from the filename, or from npz fields on older shards."""
    match = _EPISODE_NAME.search(path.name)
    if match:
        return int(match.group(1)), int(match.group(2))
    with np.load(path) as bundle:
        success = int(np.asarray(bundle["success"]).reshape(-1)[0]) if "success" in bundle.files else 0
        if "steps" in bundle.files:
            steps = int(np.asarray(bundle["steps"]).reshape(-1)[0])
        else:
            steps = int(bundle["teacher"].shape[0])
    return success, steps


def keep_best_trajectories(distill_out, keep: int = DEFAULT_KEEP_BEST) -> dict:
    """Delete the worst episode shards so at most `keep` remain. keep<=0 deletes all."""
    directory = Path(distill_out)
    if not directory.exists():
        return {"kept": 0, "deleted": 0, "total": 0}
    paths = list(directory.glob("distill_*.npz"))
    total = len(paths)
    if keep <= 0:
        for path in paths:
            path.unlink(missing_ok=True)
        log.info("wiped %d distill shards in %s", total, directory)
        return {"kept": 0, "deleted": total, "total": total}
    if total <= keep:
        log.info("keep-best: %d/%d trajectories, nothing to drop", total, keep)
        return {"kept": total, "deleted": 0, "total": total}
    ranked: list[tuple[tuple[int, int], Path]] = []
    for path in paths:
        try:
            ranked.append((trajectory_rank(*parse_trajectory_meta(path)), path))
        except (OSError, ValueError, KeyError) as exc:
            log.warning("keep-best: dropping unreadable %s (%s)", path.name, exc)
            path.unlink(missing_ok=True)
            continue
    ranked.sort(key=lambda item: item[0], reverse=True)
    kept_paths = {path for _, path in ranked[:keep]}
    deleted = 0
    for _, path in ranked[keep:]:
        path.unlink(missing_ok=True)
        deleted += 1
    log.info(
        "keep-best: kept %d / %d trajectories (deleted %d) in %s",
        len(kept_paths), total, deleted, directory,
    )
    return {"kept": len(kept_paths), "deleted": deleted, "total": total}


class DistillShardWriter:
    """Buffers one episode and writes it as an npz when the episode finishes."""

    def __init__(self, record_dir, *, shard_size: int = 256, tag: str = "distill") -> None:
        self.dir = Path(record_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._shard_size = shard_size
        self._tag = tag
        self._external: list[np.ndarray] = []
        self._wrist: list[np.ndarray] = []
        self._state: list[np.ndarray] = []
        self._instruction: list[str] = []
        self._reference: list[np.ndarray] = []
        self._teacher: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []
        self._pending: dict | None = None
        self._episode_index = 0
        self._written = 0
        self._lock = threading.Lock()
        log.info("recording distillation episodes to %s (tag %s)", self.dir, tag)

    @property
    def written(self) -> int:
        return self._written

    def append(
        self,
        *,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        state: np.ndarray,
        instruction: str,
        reference: np.ndarray,
        teacher: np.ndarray,
        executed: np.ndarray | None = None,
    ) -> None:
        rec = {
            "external_cam": np.asarray(external_cam, dtype=np.uint8),
            "wrist_cam": np.asarray(wrist_cam, dtype=np.uint8),
            "state": np.asarray(state, dtype=np.float32).reshape(-1),
            "instruction": str(instruction),
            "reference": np.asarray(reference, dtype=np.float32),
            "teacher": np.asarray(teacher, dtype=np.float32),
            "executed": np.asarray(teacher if executed is None else executed, dtype=np.float32),
        }
        with self._lock:
            if self._pending is None:
                self._pending = rec
                return
            stitched = dict(self._pending)
            stitched["teacher"] = stitch_teachers(self._pending["teacher"], rec["teacher"])
            stitched["executed"] = stitch_teachers(self._pending["executed"], rec["executed"])
            self._push_locked(stitched)
            self._pending = rec

    def finish_episode(self, *, success: bool, steps: int) -> Path | None:
        """Close the pending stitch and write this episode's shard with SR/length tags."""
        with self._lock:
            self._close_pending_locked()
            return self._write_episode_locked(success=bool(success), steps=int(steps))

    def _close_pending_locked(self) -> None:
        if self._pending is None:
            return
        rec = self._pending
        rec["teacher"] = pad_with_reference_tail(rec["teacher"], rec["reference"])
        rec["executed"] = pad_with_reference_tail(rec["executed"], rec["reference"])
        self._push_locked(rec)
        self._pending = None

    def _push_locked(self, rec: dict) -> None:
        self._external.append(rec["external_cam"])
        self._wrist.append(rec["wrist_cam"])
        self._state.append(rec["state"])
        self._instruction.append(rec["instruction"])
        self._reference.append(rec["reference"])
        self._teacher.append(np.asarray(rec["teacher"], dtype=np.float32))
        self._executed.append(np.asarray(rec["executed"], dtype=np.float32))

    def _write_episode_locked(self, *, success: bool, steps: int) -> Path | None:
        if not self._teacher:
            return None
        ok = 1 if success else 0
        path = (
            self.dir
            / f"distill_{self._tag}_pid{os.getpid()}_e{self._episode_index:06d}_ok{ok}_n{int(steps)}.npz"
        )
        np.savez(
            path,
            external_cam=np.stack(self._external),
            wrist_cam=np.stack(self._wrist),
            state=np.stack(self._state),
            instruction=np.asarray(self._instruction),
            reference=np.stack(self._reference),
            teacher=np.stack(self._teacher),
            executed=np.stack(self._executed),
            success=np.int8(ok),
            steps=np.int32(steps),
        )
        n_decisions = len(self._teacher)
        self._written += n_decisions
        log.info(
            "wrote %s (%d decisions, success=%d steps=%d, %d total)",
            path.name, n_decisions, ok, steps, self._written,
        )
        self._external.clear()
        self._wrist.clear()
        self._state.clear()
        self._instruction.clear()
        self._reference.clear()
        self._teacher.clear()
        self._executed.clear()
        self._episode_index += 1
        return path

    def flush(self) -> int:
        """Dump an unfinished episode as a failure (SIGTERM / reset without pop_rollout)."""
        with self._lock:
            self._close_pending_locked()
            steps = 8 * len(self._teacher)
            self._write_episode_locked(success=False, steps=steps)
        return self._written
