"""Every hyperparameter for RL Token, in one file.

Phase 1 is the encoder-decoder that exposes z_rl (TokenAEConfig) plus the
frozen-VLA server that produces the embeddings it reads (ServeConfig).
Phases 2-3 are the online actor-critic (OnlineConfig) and the benchmark
evaluation of a trained actor (EvalConfig).

Values marked (paper) come from arXiv 2604.23073; values marked (ours) are not
stated in the paper and were chosen by the project owner.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Fixed by the model / data / environment, not tunable.
# MolmoAct2-DROID text_config.hidden_size. pi0.5 emits 2048-wide tokens (gemma_2b), so
# a run against that backbone overrides this; the default keeps the MolmoAct2 path
# untouched rather than forcing every caller to thread a width through.
VLA_TOKEN_DIM = int(os.environ.get("RLT_VLA_TOKEN_DIM", "2560"))
ACTION_DIM = 8  # [q1..q7, gripper]
VLA_CHUNK = 15  # what the VLA emits per call
CHUNK = 8  # (ours) C: how many of those we execute; the eval client's default
PROPRIO_DIM = 16  # 7 arm positions + gripper + 7 arm velocities + gripper velocity
GRIPPER_SCALE = 0.824033  # full opening in joint units; normalizes gripper pos and vel


@dataclass
class TokenAEConfig:
    """Phase 1: RL token autoencoder (L_ro, paper Eq. 2)."""

    token_replay: str = ""  # npz path or glob of token shards
    token_cache: str = ""  # memmap dir from scripts/prepare_token_cache.py; wins if set
    out: str = "runs/rlt/token_ae.pt"

    z_dim: int = 256  # (ours) paper does not state the bottleneck width
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2

    steps: int = 8000  # (paper) 2000-10000 gradient steps
    batch_size: int = 8
    lr: float = 1e-4
    grad_clip: float = 1.0
    max_sequences: int = 6000  # cap on sequences held in RAM; 0 = all

    device: str = "cuda:0"
    seed: int = 0
    log_every: int = 100
    tb_dir: str = ""  # default: <out dir>/tb/<out stem>, so a sweep overlays

    def validate(self) -> str:
        if not self.token_replay and not self.token_cache:
            return "give either --token_replay (npz glob) or --token_cache (memmap dir)"
        if not self.out:
            return "--out is required"
        return ""


@dataclass
class ServeConfig:
    """Frozen-VLA inference server: returns an action chunk and the final-layer
    token sequence the RL token is read from."""

    repo_id: str = "allenai/MolmoAct2-DROID"
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    num_steps: int = 10  # flow-matching steps inside the action expert
    warmup: bool = True


@dataclass
class OnlineConfig:
    """Phases 2-3: online actor-critic on top of the frozen VLA (Algorithm 1)."""

    # --- what to run on ---
    episode_idx: int = -1  # single benchmark episode to train on; >= 128 (0-127 are the test set)
    episode_pool: str = ""  # several instead, e.g. "128-159"; visited round-robin
    benchmark_dir: str = ""  # empty: the default FrankaPickDroidMiniBench
    horizon: int = 500  # env steps per episode, as in the reference eval
    episodes: int = 300  # total rollouts, warmup included
    token_ae: str = "runs/ae_sweep/deep_decoder.pt"
    out_dir: str = "runs/online"
    resume: bool = False  # continue from agent.pt + buffer.npz in out_dir

    # --- rollout ---
    warmup_episodes: int = 40  # (ours) N_warm: episodes executed by the VLA before the actor takes over
    # (ours) The paper subsamples with a stride of 2, which stores overlapping
    # windows spliced from two committed chunks and pairs them with a reference
    # the robot never executed. At stride == C every stored row is one decision:
    # the action is the chunk that was chosen, the reference is what the VLA
    # proposed for that same state, and nothing is stitched.
    stride: int = CHUNK
    sigma: float = 0.02  # (ours) fixed exploration std of the Gaussian actor, N(mu, sigma^2 I) per chunk

    # --- learning ---
    gamma: float = 0.99  # (ours) per env step; the bootstrap carries gamma^C
    beta: float = 1.0  # (ours) pull towards the VLA reference chunk, paper Eq. 5
    utd: int = 5  # (paper) update-to-data ratio, counted per stored transition
    critic_updates_per_actor: int = 2  # (paper) two critic updates per actor update
    batch_size: int = 256  # (ours) TD3 default
    lr: float = 3e-4  # (ours) TD3 default
    tau: float = 0.005  # (ours) TD3 default, soft target update
    hidden_dim: int = 256  # (paper) two-layer MLP, hidden 256
    n_layers: int = 2  # (paper)
    reference_dropout: float = 0.5  # (paper) reference chunk masked out 50% of the time
    buffer_capacity: int = 400_000  # holds the whole run; nothing is ever evicted

    # --- stabilizers the paper does not spell out (see README) ---
    # Eq. 3 bootstraps through pi_theta, the online actor, which lets the actor
    # chase its own overestimates: measured on our own warmup data, Q reaches
    # 1e11 within 2700 iterations. Our reward is a single +1 on a terminal step,
    # so the true return never leaves [0, 1] and clipping the TD target to that
    # range removes nothing real. It is on by default because the unclipped
    # version reliably diverges here -- see scripts/replay_learner.py.
    clip_target: bool = True
    # Tried and rejected: a target actor alone does not prevent the divergence
    # (it arrives 900 iterations later). LayerNorm does, but changes the
    # architecture the paper specifies.
    target_actor: bool = False
    target_smoothing: float = 0.0  # extra noise on a' before the target Q, 0 = none
    critic_layer_norm: bool = False  # LayerNorm in the critic trunk (RLPD-style)

    # --- parallel collection (rlt.train_parallel; 1 is plain Algorithm 1) ---
    collectors: int = 1
    ports: str = ""  # one VLA server port per collector; default: --server_port for all
    egl_devices: str = "0,1,2,3"  # MUJOCO_EGL_DEVICE_ID per collector, not CUDA order
    encoder_devices: str = "cuda:0"  # where each collector runs the frozen token encoder

    # --- plumbing ---
    server_host: str = "localhost"
    server_port: int = 8000
    server_timeout_sec: float = 600.0
    device: str = "cuda:0"
    seed: int = 0
    save_every_episodes: int = 10
    eval_every_episodes: int = 0  # 0 = never; otherwise run --eval_rollouts greedy rollouts
    eval_rollouts: int = 20
    assets_dir: str = "/home/jovyan/users/staroverov/B1K/mlspaces/assets"
    cache_dir: str = ""  # default: the assets dir's sibling cache/
    tmp_rollout_dir: str = "/home/jovyan/users/staroverov/B1K/tmp/rlt_online"

    # --- V22 ConsensusFlow ---
    algorithm: str = "rl_token"  # "rl_token" | "consensusflow" | "flow_rlt"
    train_token: bool = False  # backprop RL through the token encoder
    ae_finetune: bool = False  # reconstruction loss on encoder+decoder
    store_decision_tokens: bool = False  # keep VLA tokens on replay rows
    token_batch_size: int = 8  # encoder/AE rows per update (full prefix is large)
    z_dim: int = 256
    cf_hidden_dim: int = 512
    cf_n_layers: int = 4
    cf_ensemble: int = 10
    cf_flow_steps: int = 10
    cf_alpha: float = 1.0
    cf_distill_coef: float = 1.0
    cf_guidance_coef: float = 0.5
    cf_expectile: float = 0.5
    cf_rho: float = 0.0
    cf_consensus_floor: float = 0.01
    cf_conflict_power: float = 2.0
    cf_residual_coef: float = 0.25
    cf_ema: float = 0.999
    ae_finetune_coef: float = 1.0
    update_every_steps: int = 0  # 0 = update on every absorbed row

    # --- corrected V22 (flow_rlt): flow actor + RL-Token critic ---
    rlt_critic: str = ""  # RLTokenAgent agent.pt -> frozen external critic; empty = learn one by TD
    flow_actor_coef: float = 1.0  # weight on -Q(s, a); Q is clipped to [0, 1]
    flow_bc_coef: float = 1.0  # weight on the flow-matching BC term
    # With a learned critic (rlt_critic empty): True freezes it after the offline
    # pretrain (comparison arm 2), False keeps TD running online (arm 3).
    flow_freeze_critic_online: bool = False
    # CF composition: the actor is a guidance field G added to the analytic base
    # velocity toward the reference (V = v_pi05_base + G), instead of the full
    # reference-conditioned corrector. Arms 4/5 of the comparison.
    flow_compose: bool = False

    def training_episodes(self) -> list[int]:
        """Which benchmark episodes the collectors cycle through."""
        from .cli import parse_episode_spec

        return parse_episode_spec(self.episode_pool) or [self.episode_idx]

    def validate(self) -> str:
        episodes = self.training_episodes()
        # The 0-127 rule protects the default benchmark's test split. A variant
        # benchmark from make_scene_variants.py has its own numbering and its own
        # held-out tail, so the rule would only misfire there.
        if not self.benchmark_dir and min(episodes) < 128:
            return "training episodes must be >= 128; episodes 0-127 are the held-out test set"
        if CHUNK % self.stride != 0:
            return f"--stride must divide C={CHUNK} so chunk boundaries stay decision points"
        if self.warmup_episodes >= self.episodes:
            return "--warmup_episodes must be smaller than --episodes"
        return ""


@dataclass
class EvalConfig:
    """Benchmark evaluation of the frozen VLA, or of a trained actor on top of it."""

    actor: str = ""  # checkpoint from train_online; empty = plain VLA baseline
    token_ae: str = "runs/ae_sweep/deep_decoder.pt"
    benchmark_dir: str = ""
    out_dir: str = "eval_output/rlt_eval"
    horizon: int = 500
    num_workers: int = 4
    max_episodes: int = 128  # the held-out test set
    episode_idx: int = -1  # >= 0: repeat this single episode --repeats times instead
    repeats: int = 1
    tag: str = ""  # suffix for the result file, so parallel shards do not collide
    video_dir: str = ""  # keep one mp4 per rollout here, named for its outcome
    camera: str = "exo_camera_1"  # which view to keep; the external one shows the whole reach
    chunk: int = CHUNK  # actions executed per plan; lower means replanning more often
    save_actions: bool = True  # write the executed action stream next to the results
    server_host: str = "localhost"
    server_port: int = 8000
    server_timeout_sec: float = 600.0
    device: str = "cuda:0"
    seed: int = 0
    assets_dir: str = "/home/jovyan/users/staroverov/B1K/mlspaces/assets"
    cache_dir: str = ""
    tmp_rollout_dir: str = "/home/jovyan/users/staroverov/B1K/tmp/rlt_eval"
