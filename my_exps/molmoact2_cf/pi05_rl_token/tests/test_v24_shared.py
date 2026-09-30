"""V24 Stage A: the shared-policy multi-task arm.

Three things have to hold for one V/W/Q/AE to serve the whole Pick-18 pool:

  - W and Q read the VLA reference chunk (cf_ref_conditioned), so a shared
    critic and guidance can tell tasks apart; V always could.
  - The flag defaults off, so stored specialist checkpoints keep their shapes.
  - The run config can name a task cycle and several VLA servers, and per-task
    buffers merge into one replay without mixing incompatible encodings.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pi05.config import RLConfig
from rlt.config import OnlineConfig
from rlt.consensusflow import make_agent
from rlt.replay import ChunkReplay

STATE_DIM = 8
CHUNK = 4
CHUNK_DIM = CHUNK * 2


def make_cfg(**overrides) -> OnlineConfig:
    return OnlineConfig(
        episode_idx=128,
        device="cpu",
        batch_size=4,
        hidden_dim=16,
        algorithm="v22_24",
        cf_hidden_dim=16,
        cf_n_layers=2,
        cf_ensemble=3,
        cf_flow_steps=2,
        token_batch_size=2,
        z_dim=4,
        **overrides,
    )


def make_batch(n: int = 16) -> dict:
    rng = np.random.default_rng(0)
    state = rng.normal(size=(n, STATE_DIM)).astype(np.float32)
    reference = (0.2 * np.tanh(state[:, :1] * rng.normal(size=(1, CHUNK_DIM)))).astype(np.float32)
    return {
        "state": state,
        "action": reference.copy(),
        "reference": reference,
        "next_state": rng.normal(size=(n, STATE_DIM)).astype(np.float32),
        "next_reference": reference.copy(),
        "reward": rng.random(n).astype(np.float32),
        "done": np.zeros(n, np.float32),
    }


# --- cf_ref_conditioned ------------------------------------------------------


def test_ref_conditioning_defaults_off():
    agent = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    assert agent.critic.ref_dim == 0
    assert agent.guidance.ref_dim == 0


def test_ref_conditioned_shapes():
    agent = make_agent(make_cfg(cf_ref_conditioned=True), STATE_DIM, CHUNK_DIM, CHUNK)
    assert agent.critic.ref_dim == CHUNK_DIM
    assert agent.guidance.ref_dim == CHUNK_DIM
    stats = agent.critic_step(make_batch())
    assert "critic_loss" in stats
    stats = agent.actor_step(make_batch())
    for key in ("bc_loss", "actor_q", "distill_loss"):
        assert key in stats


def test_ref_conditioned_critic_answers_the_reference():
    """Q(s, x, t, ref) must change with ref only when ref is an input."""
    batch = make_batch()
    other_ref = np.zeros_like(batch["reference"])
    conditioned = make_agent(make_cfg(cf_ref_conditioned=True), STATE_DIM, CHUNK_DIM, CHUNK)
    plain = make_agent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)

    q_a = conditioned.q_values(batch["state"], batch["action"], batch["reference"])
    q_b = conditioned.q_values(batch["state"], batch["action"], other_ref)
    assert not np.allclose(q_a, q_b)

    q_a = plain.q_values(batch["state"], batch["action"], batch["reference"])
    q_b = plain.q_values(batch["state"], batch["action"], other_ref)
    assert np.allclose(q_a, q_b)


def test_ref_conditioned_guidance_is_zero_at_init():
    """The zero-init W head stays exact with the wider input: V alone at init."""
    torch.manual_seed(0)
    agent = make_agent(make_cfg(cf_ref_conditioned=True), STATE_DIM, CHUNK_DIM, CHUNK)
    batch = make_batch()
    state = torch.as_tensor(batch["state"])
    x = torch.as_tensor(batch["reference"])
    t = torch.rand(state.shape[0], 1)
    v = agent.actor(state, x, t, torch.as_tensor(batch["reference"]))
    g = agent.guidance_field(state, x, t, v, reference=torch.as_tensor(batch["reference"]))
    assert torch.equal(g, torch.zeros_like(g))


def test_ref_conditioned_save_load_roundtrip(tmp_path):
    torch.manual_seed(0)
    agent = make_agent(make_cfg(cf_ref_conditioned=True), STATE_DIM, CHUNK_DIM, CHUNK)
    agent.critic_step(make_batch())
    agent.actor_step(make_batch())
    ckpt = tmp_path / "shared.pt"
    agent.save(str(ckpt))

    fresh = make_agent(make_cfg(cf_ref_conditioned=True), STATE_DIM, CHUNK_DIM, CHUNK)
    fresh.load(str(ckpt))
    for p, q in zip(agent.critic.parameters(), fresh.critic.parameters()):
        assert torch.equal(p, q)
    for p, q in zip(agent.guidance.parameters(), fresh.guidance.parameters()):
        assert torch.equal(p, q)


# --- task pool / server endpoints --------------------------------------------


def test_task_list_defaults_to_the_single_scene():
    cfg = RLConfig(scene="desk_mug", horizon=500)
    assert cfg.task_list() == [("desk_mug", 500)]


def test_task_list_parses_the_cycle():
    cfg = RLConfig(scene="desk_mug", horizon=500, task_pool="desk_mug:500, kettle:400, remote")
    assert cfg.task_list() == [("desk_mug", 500), ("kettle", 400), ("remote", 500)]


def test_task_list_validates_scene_names():
    cfg = RLConfig(scene="desk_mug", task_pool="desk_mug,not_a_scene")
    with pytest.raises(KeyError):
        cfg.validate()


def test_server_endpoints_default_and_pairing():
    cfg = RLConfig(scene="desk_mug", gpu=2, port=8600)
    assert cfg.server_endpoints() == [(2, 8600)]

    cfg = RLConfig(scene="desk_mug", vla_ports="9600,9601", vla_gpus="0,1")
    assert cfg.server_endpoints() == [(0, 9600), (1, 9601)]

    cfg = RLConfig(scene="desk_mug", vla_ports="9600,9601")
    assert cfg.server_endpoints() == [(0, 9600), (0, 9601)]  # all on `gpu`

    cfg = RLConfig(scene="desk_mug", vla_ports="9600,9601", vla_gpus="0")
    assert cfg.validate() != ""  # mismatched pairing is a config error


# --- merge_replays -----------------------------------------------------------


def _load_merge_replays():
    path = Path(__file__).resolve().parent.parent / "scripts" / "merge_replays.py"
    spec = importlib.util.spec_from_file_location("merge_replays", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_replay(path: Path, rows: int, state_dim: int = STATE_DIM, chunk_dim: int = CHUNK_DIM) -> None:
    rng = np.random.default_rng(rows)
    replay = ChunkReplay(capacity=rows * 2, state_dim=state_dim, chunk_dim=chunk_dim, seed=0)
    for _ in range(rows):
        replay.add(
            {
                "state": rng.normal(size=state_dim).astype(np.float32),
                "action": rng.normal(size=chunk_dim).astype(np.float32),
                "reference": rng.normal(size=chunk_dim).astype(np.float32),
                "next_state": rng.normal(size=state_dim).astype(np.float32),
                "next_reference": rng.normal(size=chunk_dim).astype(np.float32),
                "reward": float(rng.random()),
                "done": False,
            }
        )
    replay.save(str(path))


def test_merge_replays_concatenates(tmp_path):
    _write_replay(tmp_path / "a.npz", 12)
    _write_replay(tmp_path / "b.npz", 7)
    out = tmp_path / "merged.npz"

    module = _load_merge_replays()
    argv = ["merge_replays", "--out", str(out), str(tmp_path / "a.npz"), str(tmp_path / "b.npz")]
    sys_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = sys_argv

    data = np.load(out)
    assert int(data["size"]) == 19
    for field in ChunkReplay.FIELDS:
        assert data[field].shape[0] == 19

    replay = ChunkReplay(capacity=32, state_dim=STATE_DIM, chunk_dim=CHUNK_DIM, seed=0)
    replay.load(str(out))
    assert len(replay) == 19


def test_merge_replays_refuses_mismatched_shapes(tmp_path):
    _write_replay(tmp_path / "a.npz", 12)
    _write_replay(tmp_path / "b.npz", 7, chunk_dim=CHUNK_DIM * 2)
    module = _load_merge_replays()
    argv = ["merge_replays", "--out", str(tmp_path / "m.npz"), str(tmp_path / "a.npz"), str(tmp_path / "b.npz")]
    sys_argv = sys.argv
    try:
        sys.argv = argv
        with pytest.raises(ValueError, match="cannot share a replay"):
            module.main()
    finally:
        sys.argv = sys_argv
