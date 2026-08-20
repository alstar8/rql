"""CPU tests for V19 phase-SR schedule and aggregation."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import v19_harness


HERE = Path(__file__).resolve().parent


def test_phase_schedule_labels() -> None:
    assert v19_harness.phase_schedule(0) == ("0",)
    assert v19_harness.phase_schedule(2) == ("0", "1A", "1B", "2A", "2B")
    assert v19_harness.phase_schedule(6)[0] == "0"
    assert v19_harness.phase_schedule(6)[-1] == "6B"
    assert len(v19_harness.phase_schedule(6)) == 13
    assert v19_harness.phase_schedule(100)[-1] == "100B"
    assert len(v19_harness.phase_schedule(100)) == 201


def test_aggregate_phase_sr_pools_one_row_per_worker(tmp_path: Path) -> None:
    run_dir = tmp_path / "rlt_cf_v19_kettle"
    for shard in range(4):
        shard_dir = run_dir / "flow_cfgrl" / f"shard_{shard}"
        shard_dir.mkdir(parents=True)
        rows = [
            {"phase": "0", "success": shard < 2, "policy": "reference", "shard": shard},
            {"phase": "1A", "success": True, "policy": "actor", "shard": shard},
        ]
        (shard_dir / "phase_probe.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    table = v19_harness.aggregate_phase_sr(
        run_dir, n_shards=4, n_rounds=1, n_poses=4, variant="flow_cfgrl"
    )
    by_phase = {row["phase"]: row for row in table}
    assert by_phase["0"]["n"] == 4
    assert by_phase["0"]["successes"] == 2
    assert by_phase["0"]["sr"] == pytest.approx(0.5)
    assert by_phase["0"]["complete"] is True
    assert by_phase["1A"]["successes"] == 4
    assert by_phase["1A"]["sr"] == pytest.approx(1.0)
    assert by_phase["1B"]["n"] == 0
    assert by_phase["1B"]["complete"] is False
    text = v19_harness.format_phase_sr_table(table)
    assert "Phase 0" in text or "| 0 |" in text
    assert "50.0%" in text


def test_next_phase_label() -> None:
    assert v19_harness.next_phase_label("0", 12) == "1A"
    assert v19_harness.next_phase_label("4B", 12) == "5A"
    assert v19_harness.next_phase_label("5A", 12) == "5B"
    assert v19_harness.next_phase_label("12B", 12) is None


def test_phase_slot_and_probe_idx_are_stable() -> None:
    assert v19_harness.phase_slot("0") == 0
    assert v19_harness.phase_slot("1A") == 1
    assert v19_harness.phase_slot("1B") == 2
    assert v19_harness.phase_slot("2A") == 3
    n_poses = v19_harness.TRAIN_EPISODES
    n_shards = 32
    assert n_poses == 24
    assert v19_harness.phase_probes_per_worker(n_poses, 32) == 1
    assert v19_harness.phase_probes_per_worker(n_poses, 8) == 3
    assert v19_harness.phase_probe_episode_indices(0, n_poses, n_shards=8) == (0, 8, 16)
    assert v19_harness.phase_probe_episode_indices(7, n_poses, n_shards=8) == (7, 15, 23)
    covered = []
    for shard in range(8):
        covered.extend(v19_harness.phase_probe_episode_indices(shard, n_poses, n_shards=8))
    assert sorted(covered) == list(range(24))
    for phase in v19_harness.phase_schedule(6):
        counts = [0] * n_poses
        for shard in range(n_shards):
            idx = v19_harness.phase_probe_episode_idx(
                shard, phase, n_poses, n_shards=n_shards
            )
            assert idx == shard % n_poses
            counts[idx] += 1
        assert min(counts) == 1
        assert max(counts) == 2
        assert counts.count(2) == n_shards - n_poses
        assert [
            v19_harness.phase_probe_episode_idx(s, "0", n_poses, n_shards=n_shards)
            for s in range(n_shards)
        ] == [
            v19_harness.phase_probe_episode_idx(s, phase, n_poses, n_shards=n_shards)
            for s in range(n_shards)
        ]


def test_single_pose_cycle_all_workers_share_episode_zero() -> None:
    assert v19_harness.phase_probes_per_worker(1, 32) == 1
    for shard in range(32):
        assert v19_harness.phase_probe_episode_indices(shard, 1, n_shards=32) == (0,)


def test_cfgrl_collect_mixture_prob_compares_latest_actor_to_phase0(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "rlt_cf_v19_kettle"
    for shard in range(4):
        shard_dir = run_dir / "flow_cfgrl" / f"shard_{shard}"
        shard_dir.mkdir(parents=True)
        rows = [
            {"phase": "0", "success": shard == 0, "policy": "reference"},
            {"phase": "1A", "success": shard < 2, "policy": "actor"},
        ]
        (shard_dir / "phase_probe.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    assert (
        v19_harness.cfgrl_collect_mixture_prob(
            run_dir, n_shards=4, n_rounds=1, n_poses=4, pre=0.0, post=0.25
        )
        == 0.75
    )
    for shard in range(4):
        path = run_dir / "flow_cfgrl" / f"shard_{shard}" / "phase_probe.jsonl"
        rows = [
            {"phase": "0", "success": True, "policy": "reference"},
            {"phase": "1A", "success": False, "policy": "actor"},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert (
        v19_harness.cfgrl_collect_mixture_prob(
            run_dir, n_shards=4, n_rounds=1, n_poses=4, pre=0.0, post=0.25
        )
        == 0.0
    )
    for shard in range(4):
        path = run_dir / "flow_cfgrl" / f"shard_{shard}" / "phase_probe.jsonl"
        rows = [
            {"phase": "0", "success": shard < 2, "policy": "reference"},
            {"phase": "1A", "success": shard < 2, "policy": "actor"},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert (
        v19_harness.cfgrl_collect_mixture_prob(
            run_dir, n_shards=4, n_rounds=1, n_poses=4, pre=0.0, post=0.25
        )
        == 0.25
    )


def test_phase_barrier_status_and_wait(tmp_path: Path) -> None:
    run_dir = tmp_path / "rlt_cf_v19_kettle"
    for shard in range(3):
        shard_dir = run_dir / "flow_cfgrl" / f"shard_{shard}"
        shard_dir.mkdir(parents=True)
        if shard < 2:
            (shard_dir / "phase_probe.jsonl").write_text(
                json.dumps({"phase": "0", "success": False})
                + "\n"
                + json.dumps({"phase": "1A", "success": True})
                + "\n",
                encoding="utf-8",
            )
    n_ready, missing = v19_harness.phase_barrier_status(
        run_dir, "1A", n_shards=3, n_poses=3, variant="flow_cfgrl"
    )
    assert n_ready == 2
    assert missing == [2]
    assert v19_harness.last_recorded_phase(run_dir / "flow_cfgrl" / "shard_0") == "1A"
    assert v19_harness.last_recorded_phase(run_dir / "flow_cfgrl" / "shard_2") is None
    assert (
        v19_harness.wait_for_phase_barrier(
            run_dir,
            "1A",
            n_shards=3,
            n_poses=3,
            poll_sec=0.05,
            log_every_sec=0.05,
            should_stop=lambda: True,
        )
        is False
    )
    (run_dir / "flow_cfgrl" / "shard_2" / "phase_probe.jsonl").write_text(
        json.dumps({"phase": "1A", "success": False}) + "\n",
        encoding="utf-8",
    )
    assert (
        v19_harness.wait_for_phase_barrier(
            run_dir,
            "1A",
            n_shards=3,
            n_poses=3,
            poll_sec=0.05,
            should_stop=lambda: True,
        )
        is True
    )


def test_last_recorded_phase_uses_highest_slot(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shard_0"
    shard_dir.mkdir()
    rows = [
        {"phase": "0"},
        {"phase": "2A"},
        {"phase": "1B"},
    ]
    (shard_dir / "phase_probe.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    assert v19_harness.last_recorded_phase(shard_dir) == "2A"


def test_phase_probe_row_is_complete_skips_crash_junk() -> None:
    assert v19_harness.phase_probe_row_is_complete({"phase": "0", "success": True})
    assert v19_harness.phase_probe_row_is_complete(
        {"phase": "0", "success": False, "valid": True, "n_steps": 500}
    )
    assert not v19_harness.phase_probe_row_is_complete(
        {"phase": "0", "success": False, "valid": False, "n_steps": 0}
    )
    assert not v19_harness.phase_probe_row_is_complete(
        {"phase": "0", "success": False, "n_steps": 0}
    )
    assert v19_harness.is_egl_crash_returncode(-6)
    assert v19_harness.is_egl_crash_returncode(134)
    assert v19_harness.is_egl_crash_returncode(None)
    assert not v19_harness.is_egl_crash_returncode(0)
    assert not v19_harness.is_egl_crash_returncode(1)
    assert 480.0 <= v19_harness.isolated_rollout_timeout_sec(500) <= 900.0
    assert v19_harness.isolated_rollout_startup_sec() == 180.0
    assert v19_harness.isolated_rollout_attempts() >= 1
    assert v19_harness.isolated_rollouts_enabled() is True


def test_isolated_timeout_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RLT_ISOLATED_TIMEOUT_SEC", "120")
    monkeypatch.setenv("RLT_ISOLATED_STARTUP_SEC", "30")
    assert v19_harness.isolated_rollout_timeout_sec(500) == 120.0
    assert v19_harness.isolated_rollout_startup_sec() == 30.0


def test_launch_script_syntax_and_phase_probe() -> None:
    launch = HERE / "launch_v19_rlt_cfgrl.sh"
    subprocess.run(["bash", "-n", str(launch)], check=True)
    body = launch.read_text(encoding="utf-8")
    assert "V19_PHASE_ROUNDS" in body
    assert 'V19_PHASE_ROUNDS="${V19_PHASE_ROUNDS:-100}"' in body
    assert "V19_PHASE_BARRIER" in body
    assert "phase_barrier" in body
    assert "aggregate-phase-sr" in body
    assert "summary.json" in body
    assert "VAL_BENCHMARK" in body
    assert "cfgrl_phase_probe" not in body or "V19_PHASE_PROBE" in body
    assert 'INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}"' in body
    assert 'V19_POSE_CYCLE="${V19_POSE_CYCLE:-24}"' in body
    assert "--poses" in body
    assert 'RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-3}"' in body
    assert 'V19_HIDDEN="${V19_HIDDEN:-1024}"' in body
    assert 'V19_Z_EXPAND_DIM="${V19_Z_EXPAND_DIM:-512}"' in body
    assert 'V19_CFGRL_O_DIM="${V19_CFGRL_O_DIM:-128}"' in body
    assert 'V19_CFGRL_KQ="${V19_CFGRL_KQ:-4096}"' in body
    assert (
        'RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-$((NUM_GPUS * RLT_EGL_PER_GPU))}"'
        in body
    )
    assert "PYTHONFAULTHANDLER=1" in body
    assert "RLT_ISOLATED_ROLLOUT" in body
    assert "RLT_ISOLATED_GL" in body
    assert "RLT_ISOLATED_STARTUP_SEC" in body
    assert "RLT_ISOLATED_TIMEOUT_SEC" in body
    trainer = (HERE / "train_rlt_online.py").read_text(encoding="utf-8")
    assert "--isolated_rollout" in trainer
    assert "RLT_ISOLATED_CHILD" in trainer
    assert "start_new_session=True" in trainer
    stop = HERE / "stop_run.sh"
    subprocess.run(["bash", "-n", str(stop)], check=True)
    stop_body = stop.read_text(encoding="utf-8")
    assert "launch_v19_rlt_cfgrl.sh" in stop_body
