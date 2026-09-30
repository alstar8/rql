"""Parallel action-expert swap: what gets copied, when, and the learner protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi05.config import RLConfig  # noqa: E402
from pi05.expert_io import copy_expert_weights, is_expert_param, swap_due  # noqa: E402
from pi05.expert_swap import read_status, request_student_save, write_command  # noqa: E402


def test_expert_filter_keeps_the_head_and_drops_the_backbone():
    assert is_expert_param("time_mlp_in.weight")
    assert is_expert_param("action_in_proj.bias")
    assert is_expert_param("paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj.weight")
    assert not is_expert_param("paligemma_with_expert.paligemma.language_model.layers.0.mlp.down_proj.weight")
    assert not is_expert_param("paligemma_with_expert.paligemma.vision_tower.vision_model.embeddings.patch_embedding.weight")


def test_rollout_server_port_is_a_fixed_offset():
    from pi05.config import rollout_server_port

    assert rollout_server_port(9600) == 9605
    assert rollout_server_port(9610) == 9615


def test_rollout_checkpoint_requires_a_distill_dir():
    problem = RLConfig(scene="desk_mug", rollout_checkpoint="/tmp/student").validate()
    assert problem and "distill_out" in problem
    ok = RLConfig(
        scene="desk_mug",
        rollout_checkpoint="/tmp/student",
        distill_out="/tmp/distill",
    ).validate()
    assert ok == ""


def test_swap_due_is_every_n_and_never_episode_zero():
    assert not swap_due(0, 50)
    assert not swap_due(49, 50)
    assert swap_due(50, 50)
    assert swap_due(100, 50)
    assert not swap_due(51, 50)
    assert not swap_due(50, 0)
    assert not swap_due(50, -1)


def test_swap_every_requires_the_student_paths():
    problem = RLConfig(scene="desk_mug", distill_swap_every=50).validate()
    assert problem and "distill_out" in problem
    problem = RLConfig(
        scene="desk_mug",
        distill_swap_every=50,
        distill_out="/tmp/distill",
        distill_student_dir="/tmp/student",
        distill_control_dir="/tmp/control",
    ).validate()
    assert problem == ""


def test_request_save_waits_for_a_new_generation(tmp_path):
    write_command(tmp_path, "run")
    (tmp_path / "status.json").write_text('{"generation": 0, "saved": false}')

    def after_save_command():
        if (tmp_path / "command").read_text() == "save":
            (tmp_path / "status.json").write_text('{"generation": 1, "saved": true, "step": 12}')
            return True
        return True

    status = request_student_save(
        tmp_path,
        previous_generation=0,
        timeout_sec=2.0,
        poll_sec=0.05,
        is_alive=after_save_command,
    )
    assert status["generation"] == 1
    assert read_status(tmp_path)["saved"] is True
    assert (tmp_path / "command").read_text() == "run"


def test_a_dead_student_fails_the_swap_loudly(tmp_path):
    write_command(tmp_path, "run")
    with pytest.raises(RuntimeError, match="exited before saving"):
        request_student_save(
            tmp_path,
            previous_generation=0,
            timeout_sec=1.0,
            poll_sec=0.05,
            is_alive=lambda: False,
        )


def test_copy_expert_weights_writes_the_head_and_leaves_the_backbone(tmp_path):
    import torch
    from torch import nn

    import safetensors.torch

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_mlp_in = nn.Linear(2, 2, bias=False)
            self.paligemma_with_expert = nn.Linear(2, 2, bias=False)

    src, dst = Toy(), Toy()
    with torch.no_grad():
        src.time_mlp_in.weight.fill_(3.0)
        src.paligemma_with_expert.weight.fill_(7.0)
        dst.time_mlp_in.weight.zero_()
        dst.paligemma_with_expert.weight.fill_(1.0)
    safetensors.torch.save_model(src, str(tmp_path / "model.safetensors"))
    copied = copy_expert_weights(dst, tmp_path)
    assert copied == 1
    assert torch.allclose(dst.time_mlp_in.weight, src.time_mlp_in.weight)
    assert torch.allclose(dst.paligemma_with_expert.weight, torch.ones(2, 2))


def test_claim_below_does_not_advance_when_at_limit(tmp_path):
    from pi05.collectors import FileCounter

    counter = FileCounter(tmp_path / "c")
    assert counter.claim_below(1) == 0
    assert counter.claim_below(1) is None
    assert counter.claim_below(1) is None
    assert counter.claim_below(2) == 1


def test_stitched_teacher_is_two_gOn_chunks(tmp_path):
    from pi05.distill_recorder import DistillShardWriter

    writer = DistillShardWriter(tmp_path, shard_size=8, tag="t")
    cam = np.zeros((4, 4, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    ref = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    first = np.ones((8, 8), dtype=np.float32)
    second = np.full((8, 8), 2.0, dtype=np.float32)
    kwargs = dict(
        external_cam=cam, wrist_cam=cam, state=state, instruction="pick", reference=ref,
    )
    writer.append(**kwargs, teacher=first, executed=first)
    writer.append(**kwargs, teacher=second, executed=second)
    writer.flush()
    shard = next(tmp_path.glob("*.npz"))
    with np.load(shard) as z:
        teacher = z["teacher"]
        executed = z["executed"]
        assert teacher.shape == (2, 16, 8)
        assert np.allclose(teacher[0, :8], 1.0)
        assert np.allclose(teacher[0, 8:], 2.0)
        assert np.allclose(teacher[1, :8], 2.0)
        assert np.allclose(teacher[1, 8:], ref[8:])
        assert executed.shape == teacher.shape
        assert int(np.asarray(z["success"])) == 0
        assert int(np.asarray(z["steps"])) == 16


def test_make_target_keeps_the_committed_prefix_and_the_reference_tail():
    from scripts.train_expert_distill import make_target

    teacher = np.concatenate([np.ones((8, 8)), np.full((8, 8), 3.0)], axis=0)
    ref = np.zeros((16, 8), dtype=np.float32)
    ref[8:] = 4.0
    out = make_target(teacher, ref)
    assert out.shape == (16, 8)
    assert np.allclose(out[:8], 1.0)
    assert np.allclose(out[8:], 4.0)
    old_ref = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    old = make_target(np.ones((8, 8)), old_ref)
    assert old.shape == (16, 8)
    assert np.allclose(old[:8], 1.0)
    assert np.allclose(old[8:], old_ref[8:])


def test_prefix_flow_loss_ignores_the_open_loop_tail():
    import torch

    from scripts.train_expert_distill import prefix_flow_loss

    per_step = torch.zeros(2, 16, 32)
    per_step[:, 8:, :8] = 100.0
    assert float(prefix_flow_loss(per_step)) == 0.0
    per_step[0, 0, 0] = 4.0
    assert abs(float(prefix_flow_loss(per_step)) - 4.0 / (2 * 8 * 8)) < 1e-6


def test_split_train_holdout_is_disjoint_and_keeps_a_train_shard(tmp_path):
    from scripts.train_expert_distill import split_train_holdout

    paths = [tmp_path / f"distill_{i}.npz" for i in range(10)]
    train, hold = split_train_holdout(paths, 0.1, seed=0)
    assert len(hold) == 1
    assert len(train) == 9
    assert set(train).isdisjoint(hold)
    assert set(train) | set(hold) == set(paths)
    again, again_hold = split_train_holdout(paths, 0.1, seed=0)
    assert again == train and again_hold == hold
    all_train, empty = split_train_holdout(paths, 0.0, seed=0)
    assert all_train == paths and empty == []


def test_round_eval_requires_a_swap_interval():
    problem = RLConfig(
        scene="desk_mug",
        round_eval_per_task=2,
        distill_out="/tmp/d",
        distill_student_dir="/tmp/s",
        distill_control_dir="/tmp/c",
    ).validate()
    assert problem and "distill_swap_every" in problem
    ok = RLConfig(
        scene="desk_mug",
        distill_swap_every=50,
        round_eval_per_task=2,
        distill_out="/tmp/d",
        distill_student_dir="/tmp/s",
        distill_control_dir="/tmp/c",
    ).validate()
    assert ok == ""


def test_next_actor_limit_steps_by_round():
    from pi05.round_eval import next_actor_limit

    assert next_actor_limit(36, 0, 50, 1800) == 86
    assert next_actor_limit(36, 50, 50, 1800) == 136
    assert next_actor_limit(36, 1800, 50, 1800) == 1836


def test_apply_arm_toggles_actor_explore_and_recording():
    from types import SimpleNamespace

    from pi05.round_eval import apply_arm

    policy = SimpleNamespace(
        use_actor=True,
        record_distill=True,
        corrector=SimpleNamespace(explore=True),
    )
    apply_arm(policy, "base_off")
    assert policy.use_actor is False
    assert policy.record_distill is False
    apply_arm(policy, "gon")
    assert policy.use_actor is True
    assert policy.corrector.explore is False
    apply_arm(policy, "student_off")
    assert policy.use_actor is False
    apply_arm(policy, "run")
    assert policy.use_actor is True
    assert policy.record_distill is True
    assert policy.corrector.explore is True


def test_wipe_distill_shards_removes_only_npz(tmp_path):
    from pi05.round_eval import wipe_distill_shards

    (tmp_path / "distill_a.npz").write_bytes(b"x")
    (tmp_path / "distill_b.npz").write_bytes(b"y")
    (tmp_path / "keep.txt").write_text("ok")
    assert wipe_distill_shards(tmp_path) == 2
    assert not (tmp_path / "distill_a.npz").exists()
    assert (tmp_path / "keep.txt").exists()
    assert wipe_distill_shards(tmp_path) == 0


def test_keep_best_prefers_success_then_length(tmp_path):
    from pi05.distill_recorder import keep_best_trajectories, trajectory_rank

    def write(name: str) -> None:
        np.savez(tmp_path / name, teacher=np.zeros((1, 16, 8), dtype=np.float32))

    write("distill_x_ok1_n120.npz")
    write("distill_x_ok1_n400.npz")
    write("distill_x_ok0_n500.npz")
    write("distill_x_ok0_n40.npz")
    write("distill_x_ok1_n80.npz")
    assert trajectory_rank(1, 80) > trajectory_rank(1, 400)
    assert trajectory_rank(1, 400) > trajectory_rank(0, 500)
    kept = keep_best_trajectories(tmp_path, keep=3)
    assert kept["kept"] == 3
    assert kept["deleted"] == 2
    names = {p.name for p in tmp_path.glob("distill_*.npz")}
    assert names == {
        "distill_x_ok1_n80.npz",
        "distill_x_ok1_n120.npz",
        "distill_x_ok1_n400.npz",
    }


def test_student_beats_frozen_is_strict_and_complete():
    from pi05.round_eval import student_beats_frozen

    better = {"n": 36, "sr": 0.64, "incomplete": False}
    frozen = {"n": 36, "sr": 0.53, "incomplete": False}
    assert student_beats_frozen(better, frozen)
    assert not student_beats_frozen(frozen, frozen)
    assert not student_beats_frozen({"n": 36, "sr": 0.9, "incomplete": True}, frozen)
    assert not student_beats_frozen({"n": 0, "sr": 1.0, "incomplete": False}, frozen)


def test_finish_episode_tags_success_and_length(tmp_path):
    from pi05.distill_recorder import DistillShardWriter

    writer = DistillShardWriter(tmp_path, shard_size=8, tag="t")
    cam = np.zeros((4, 4, 3), dtype=np.uint8)
    state = np.zeros(8, dtype=np.float32)
    ref = np.arange(16 * 8, dtype=np.float32).reshape(16, 8)
    kwargs = dict(
        external_cam=cam, wrist_cam=cam, state=state, instruction="pick", reference=ref,
    )
    writer.append(**kwargs, teacher=np.ones((8, 8), dtype=np.float32))
    writer.append(**kwargs, teacher=np.full((8, 8), 2.0, dtype=np.float32))
    path = writer.finish_episode(success=True, steps=173)
    assert path is not None
    assert "_ok1_n173.npz" in path.name
    with np.load(path) as z:
        assert int(z["success"]) == 1
        assert int(z["steps"]) == 173
        assert z["teacher"].shape == (2, 16, 8)


def test_snapshot_checkpoint_and_frozen_path(tmp_path):
    from types import SimpleNamespace

    from pi05.expert_swap import frozen_checkpoint, snapshot_checkpoint

    src = tmp_path / "src"
    src.mkdir()
    (src / "model.safetensors").write_bytes(b"weights")
    (src / "config.json").write_text("{}")
    dest = tmp_path / "frozen"
    snapshot_checkpoint(src, dest)
    assert (dest / "model.safetensors").read_bytes() == b"weights"
    assert (dest / "config.json").read_text() == "{}"
    cfg = SimpleNamespace(
        distill_frozen_dir=str(dest),
        distill_student_dir=str(tmp_path / "student"),
        checkpoint=tmp_path / "base",
    )
    assert frozen_checkpoint(cfg) == str(dest)
    empty = SimpleNamespace(
        distill_frozen_dir=str(tmp_path / "missing"),
        distill_student_dir=str(tmp_path / "student"),
        checkpoint=tmp_path / "base",
    )
    assert frozen_checkpoint(empty) == str(tmp_path / "base")
    from pi05.collectors import FileCounter

    counter = FileCounter(tmp_path / "eval")
    counter.reset(0, seq=100)
    assert counter.claim_below(4, seq=100) == 0
    assert counter.claim_below(4, seq=99) is None
    assert counter.claim_below(4, seq=100) == 1
    counter.reset(0, seq=101)
    assert counter.claim_below(4, seq=100) is None
    assert counter.claim_below(4, seq=101) == 0


def test_copy_checkpoint_sidecars_skips_same_directory(tmp_path):
    from scripts.train_expert_distill import copy_checkpoint_sidecars

    src = tmp_path / "ckpt"
    src.mkdir()
    (src / "config.json").write_text('{"ok": true}')
    (src / "assets").mkdir()
    (src / "assets" / "x").write_text("a")
    copy_checkpoint_sidecars(str(src), str(src))
    dest = tmp_path / "out"
    dest.mkdir()
    copy_checkpoint_sidecars(str(src), str(dest))
    assert (dest / "config.json").read_text() == '{"ok": true}'
    assert (dest / "assets" / "x").read_text() == "a"


def test_run_arm_returns_partial_on_timeout(tmp_path):
    from pi05.round_eval import _run_arm, write_heartbeat

    n_collectors = 2
    for rank in range(n_collectors):
        write_heartbeat(tmp_path, rank, "hold")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    got = _run_arm(
        tmp_path,
        arm="student_off",
        n_eval=4,
        seq=7,
        n_collectors=n_collectors,
        inbox=inbox,
        timeout_sec=0.4,
    )
    assert got == []


def test_summarize_arm_marks_incomplete():
    from pi05.round_eval import summarize_arm

    summary = summarize_arm(
        [{"scene": "cup", "success": 1}, {"scene": "cup", "success": 0}],
        n_eval=4,
    )
    assert summary["n"] == 2
    assert summary["n_requested"] == 4
    assert summary["incomplete"] is True
    assert summary["sr"] == 0.5
    from pi05.round_eval import merge_arm_payloads

    got = merge_arm_payloads(
        [],
        [
            {"round_eval": True, "arm": "gon", "seq": 1, "episode": 0, "success": 1},
            {"round_eval": True, "arm": "gon", "seq": 1, "episode": 0, "success": 0},
            {"round_eval": True, "arm": "base_off", "seq": 1, "episode": 1, "success": 1},
            {"round_eval": True, "arm": "gon", "seq": 1, "episode": 2, "success": 1},
        ],
        arm="gon",
        seq=1,
    )
    assert [payload["episode"] for payload in got] == [0, 2]
    assert got[0]["success"] == 1

