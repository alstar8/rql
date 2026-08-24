"""ConsensusFlow on the RL-Token mug stack: losses, freeze flags, AE recon."""

from __future__ import annotations

import numpy as np
import torch

from rlt.config import OnlineConfig
from rlt.consensusflow import ConsensusFlowAgent, make_agent
from rlt.networks import EnsembleCritic, FlowActor, Guidance
from rlt.token_ae import RLTokenAE, write_random_ae

STATE_DIM = 8
CHUNK = 4
CHUNK_DIM = CHUNK * 2
Z_DIM = 4


def make_cfg(**overrides) -> OnlineConfig:
    return OnlineConfig(
        episode_idx=128,
        device="cpu",
        batch_size=4,
        hidden_dim=16,
        algorithm="consensusflow",
        cf_hidden_dim=16,
        cf_n_layers=2,
        cf_ensemble=2,
        cf_flow_steps=2,
        token_batch_size=2,
        z_dim=Z_DIM,
        **overrides,
    )


def make_batch(n: int = 4, tokens: bool = False) -> dict:
    rng = np.random.default_rng(0)
    batch = {
        "state": rng.normal(size=(n, STATE_DIM)).astype(np.float32),
        "action": rng.normal(size=(n, CHUNK_DIM)).astype(np.float32),
        "reference": rng.normal(size=(n, CHUNK_DIM)).astype(np.float32),
        "next_state": rng.normal(size=(n, STATE_DIM)).astype(np.float32),
        "next_reference": rng.normal(size=(n, CHUNK_DIM)).astype(np.float32),
        "reward": rng.random(n).astype(np.float32),
        "done": np.zeros(n, np.float32),
    }
    if tokens:
        s, d = 6, 8
        batch["tokens"] = [rng.normal(size=(s, d)).astype(np.float16) for _ in range(n)]
        batch["mask"] = [np.ones(s, np.float16) for _ in range(n)]
        batch["next_tokens"] = [rng.normal(size=(s, d)).astype(np.float16) for _ in range(n)]
        batch["next_mask"] = [np.ones(s, np.float16) for _ in range(n)]
    return batch


def test_make_agent_picks_consensusflow():
    cfg = make_cfg()
    agent = make_agent(cfg, STATE_DIM, CHUNK_DIM, CHUNK)
    assert isinstance(agent, ConsensusFlowAgent)


def test_joint_update_runs_on_cpu():
    torch.manual_seed(0)
    agent = ConsensusFlowAgent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    stats = agent.update(make_batch())
    assert "critic_loss" in stats and "bc_loss" in stats and "distill_loss" in stats
    assert agent.critic_steps == 1
    assert agent.actor_steps == 1


def test_act_shape_matches_chunk():
    agent = ConsensusFlowAgent(make_cfg(), STATE_DIM, CHUNK_DIM, CHUNK)
    action = agent.act(np.zeros(STATE_DIM, np.float32), np.zeros(CHUNK_DIM, np.float32), explore=False)
    assert action.shape == (CHUNK_DIM,)


def test_freeze_token_stops_encoder_grads(tmp_path):
    ae = RLTokenAE(token_dim=8, z_dim=Z_DIM, d_model=16, n_heads=2, n_layers=1)
    cfg = make_cfg(train_token=True)
    agent = ConsensusFlowAgent(cfg, STATE_DIM, CHUNK_DIM, CHUNK, token_ae=ae)
    assert any(p.requires_grad for p in agent.token_ae.parameters())
    before = agent.token_ae.encoder.out_proj[-1].weight.detach().clone()
    agent.freeze_token()
    agent.update(make_batch(tokens=True))
    after = agent.token_ae.encoder.out_proj[-1].weight.detach()
    assert torch.equal(before, after)


def test_ae_finetune_changes_decoder(tmp_path):
    ae = RLTokenAE(token_dim=8, z_dim=Z_DIM, d_model=16, n_heads=2, n_layers=1)
    cfg = make_cfg(train_token=False, ae_finetune=True)
    agent = ConsensusFlowAgent(cfg, STATE_DIM, CHUNK_DIM, CHUNK, token_ae=ae)
    agent.freeze_token()
    agent.set_ae_finetune(True)
    before = agent.token_ae.decoder.out_proj.weight.detach().clone()
    agent.update(make_batch(tokens=True))
    after = agent.token_ae.decoder.out_proj.weight.detach()
    assert not torch.equal(before, after)


def test_write_random_ae(tmp_path):
    path = tmp_path / "ae.pt"
    write_random_ae(str(path), token_dim=8, z_dim=4, d_model=16, n_heads=2, n_layers=1)
    loaded = RLTokenAE.load(str(path), map_location="cpu")
    assert loaded.z_dim == 4


def test_flow_modules_exist():
    assert FlowActor(4, 3, 8, 2)(torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 1)).shape == (2, 3)
    q = EnsembleCritic(4, 3, 8, 2, ensemble=2)(torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 1))
    assert q.shape == (2, 2)
    w = Guidance(4, 3, 8, 2)(torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 1))
    assert w.shape == (2, 3)
