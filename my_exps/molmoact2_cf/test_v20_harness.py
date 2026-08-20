"""Promotion, barrier, resume, and rollback tests for V20."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk_replay import (  # noqa: E402
    ACTION_DIM,
    CHUNK_SIZE,
    Z_DIM,
    ChunkReplay,
    ReplaySource,
)
from v20_harness import (  # noqa: E402
    StaleOperationError,
    barrier_snapshot,
    claim_worker_operation,
    clean_operation_artifacts,
    decide_promotion,
    initialize_run,
    publish_operation,
    write_worker_done,
)
from v20_runner import (  # noqa: E402
    _import_collect_snapshot,
    _pending_offline_round_ids,
)


def _run(tmp_path: Path, worker_count: int = 2) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    checkpoint = tmp_path / "incumbent.pt"
    checkpoint.write_bytes(b"incumbent-v0")
    initialize_run(
        run_dir,
        worker_count=worker_count,
        incumbent_checkpoint=checkpoint,
        incumbent_mode="reference",
        rollout_seed_root=123,
        config={"test": True},
    )
    return run_dir, checkpoint


def test_clean_operation_artifacts_tolerates_nonempty_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir, _checkpoint = _run(tmp_path)
    claim_dir = run_dir / "coordination" / "claims" / "op_nfs"
    claim_dir.mkdir(parents=True)
    leftover = claim_dir / "worker_000.lock"
    leftover.write_text("stale", encoding="utf-8")
    calls = {"n": 0}

    def flaky_rmtree(path: object, *args: object, **kwargs: object) -> None:
        del args
        calls["n"] += 1
        if calls["n"] == 1 and not kwargs.get("ignore_errors"):
            raise OSError(39, "Directory not empty")
        if Path(str(path)).exists():
            for child in list(Path(str(path)).glob("*")):
                if child.is_file():
                    child.unlink()
            Path(str(path)).rmdir()

    monkeypatch.setattr("v20_harness.shutil.rmtree", flaky_rmtree)
    clean_operation_artifacts(run_dir, "op_nfs")
    assert not leftover.exists()


def _episode_replay(uid: int, success: bool) -> ChunkReplay:
    replay = ChunkReplay(
        max_transitions=100,
        pos_frac=0.5,
        benchmark_pose_cycle=24,
    )
    actions = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    replay.add_episode_chunks(
        [np.zeros((Z_DIM,), dtype=np.float32)],
        [np.zeros((ACTION_DIM,), dtype=np.float32)],
        [actions],
        [actions],
        [np.full((CHUNK_SIZE,), float(success), dtype=np.float32)],
        [np.ones((CHUNK_SIZE,), dtype=np.float32)],
        success=success,
        gamma=0.99,
        episode_id=uid,
        trajectory_uid=uid,
        pose_idx=0,
        source_policy=ReplaySource.INCUMBENT,
        worker_id=0,
        round_id=0,
        policy_version=0,
    )
    return replay


def _paired_rows(
    candidate_only: int,
    incumbent_only: int,
    both_success: int = 20,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pair_id = 0
    for _ in range(candidate_only):
        rows.append(
            {
                "pair_id": pair_id,
                "candidate_success": True,
                "incumbent_success": False,
            }
        )
        pair_id += 1
    for _ in range(incumbent_only):
        rows.append(
            {
                "pair_id": pair_id,
                "candidate_success": False,
                "incumbent_success": True,
            }
        )
        pair_id += 1
    for _ in range(both_success):
        rows.append(
            {
                "pair_id": pair_id,
                "candidate_success": True,
                "incumbent_success": True,
            }
        )
        pair_id += 1
    return rows


def test_decide_promotion_gates_deployed_cond_head() -> None:
    rows = _paired_rows(candidate_only=12, incumbent_only=2)
    base = {"alpha": 0.05, "min_gain": 0.03, "action_bounds_ok": True}
    promoted = decide_promotion(rows, clone_ok=True, cond_ok=True, **base)
    assert promoted.promote
    blocked = decide_promotion(rows, clone_ok=True, cond_ok=False, **base)
    assert not blocked.promote
    assert "cond_drift_check_failed" in blocked.reason
    assert blocked.cond_ok is False


def test_decide_promotion_bonferroni_alpha_is_respected() -> None:
    # 10 vs 2 discordant pairs: McNemar p ~= 0.0193, which clears alpha=0.05
    # but not the Bonferroni-corrected alpha=0.05/8.
    rows = _paired_rows(candidate_only=10, incumbent_only=2)
    base = {"min_gain": 0.03, "clone_ok": True, "cond_ok": True,
            "action_bounds_ok": True}
    loose = decide_promotion(rows, alpha=0.05, **base)
    strict = decide_promotion(rows, alpha=0.05 / 8.0, **base)
    assert loose.p_value == strict.p_value
    assert loose.promote
    assert not strict.promote
    assert "paired_test_not_significant" in strict.reason


def test_operation_barrier_is_idempotent_and_rejects_stale_writes(
    tmp_path: Path,
) -> None:
    run_dir, _checkpoint = _run(tmp_path)
    state = publish_operation(
        run_dir,
        phase="collect",
        round_id=0,
        operation_id="r000_collect_w000",
        collect_wave=0,
    )
    with claim_worker_operation(run_dir, state.operation_id, 0) as should_run:
        assert should_run
        write_worker_done(
            run_dir,
            state,
            0,
            valid=True,
            payload={"result": {"success": False}},
        )
    with claim_worker_operation(run_dir, state.operation_id, 0) as should_run:
        assert not should_run
    write_worker_done(
        run_dir,
        state,
        1,
        valid=True,
        payload={"result": {"success": True}},
    )
    snapshot = barrier_snapshot(run_dir, state.operation_id, 2)
    assert snapshot.completed == 2
    assert snapshot.valid == 2

    publish_operation(
        run_dir,
        phase="train",
        round_id=0,
        operation_id="r000_train",
    )
    with pytest.raises(StaleOperationError):
        write_worker_done(
            run_dir,
            state,
            0,
            valid=True,
            payload={"result": {}},
        )


def test_pending_offline_round_ids_skips_finished_and_off_schedule(
    tmp_path: Path,
) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "round_000_final.json").write_text("{}", encoding="utf-8")
    (reports / "round_001_final.json").write_text("{}", encoding="utf-8")
    (reports / "round_002_final.json").write_text("{}", encoding="utf-8")
    (reports / "round_001_offline.json").write_text("{}", encoding="utf-8")
    assert _pending_offline_round_ids(tmp_path, rounds_per_offline=2) == []
    (reports / "round_001_offline.json").unlink()
    assert _pending_offline_round_ids(tmp_path, rounds_per_offline=2) == [1]

