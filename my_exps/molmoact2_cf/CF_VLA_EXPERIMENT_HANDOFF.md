# CF_VLA / RLT-CF experiment handoff

Last updated: 2026-08-06 12:00 UTC.
Code root: `B1K_AIRI/submodules/rql/my_exps/molmoact2_cf/`  
Paper draft: `B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/ConsensusFlowMolmo/CF_VLA.tex`

Use this file to resume method + experiment work without re-deriving context.

---

## 1. Method (what we implement)

### 1.1 High-level recipe (RLT + residual CF, not full paper CF)

Frozen **MolmoAct2-DROID** serves reference action chunks + last-layer tokens over HTTP (`serve.py`). Lightweight modules train online:

1. **RL-token AE** — encode masked last-layer token sequence → \(z_{\mathrm{rl}}\) (recon warmup, optionally freeze).
2. **State** — \(x=(z_{\mathrm{rl}}, s^{p})\) with proprio.
3. **Chunk residual actor** — \(a = a_{\mathrm{VLA}} + \Delta_{\mathrm{actor}}\) (chunk size \(C{=}8\)).
4. **Ensemble critic** — \(K{=}10\) `EnsembleCQL` heads, **not** time-dependent: \(Q_k(s, a_{\mathrm{chunk}})\).
5. **Residual CF guide** — distill common-scale-normalized \(\nabla_a Q_k\) into bounded \(G_\phi\); one-shot residual, **no flow / denoising ODE steps**.

Deployed chunk (when gate on):

\[
a = a_{\mathrm{VLA}} + \Delta_{\mathrm{actor}}(+\,\Delta_{\mathrm{guide}})
\]

Contrast with paper ConsensusFlow (`ConsensusFlow.tex`): there \(Q_k(s,x,t)\) teaches a flow guidance field over many denoising steps. **V7 is the residual one-shot adaptation only.**

### 1.2 Gate (when RLT/CF actually runs)

`train_rlt_online.py` keeps `policy.enable_rlt=False` until:

- `valid_episodes >= g_start_episodes` (default **40**),
- enough replay + both success/fail outcomes,
- critic healthy,
- LCB advantage ≥ `g_min_advantage`,
- action sensitivity ≥ `g_min_action_sensitivity`.

Until then, executed actions are **VLA reference** (+ optional explore noise for RLT configs).  
Eval-only force path: `--force_deploy_rlt` (added for pretrain 10-ep eval).

### 1.3 Four online configs (V7 screen)

| Config | Actor | Guide | Token | Updates/ep |
|---|---|---|---|---|
| `mean_pool_baseline` | VLA only | off | frozen | 0 (eval-like) |
| `rlt_actor_no_guide` | RLT residual | off | frozen | 8 |
| `rlt_cf_frozen_token` | RLT residual | on | frozen | 8 |
| `rlt_cf_online_token` | RLT residual | on | online tune | 8 |

---

## 2. Code map

| Path | Role |
|---|---|
| `serve.py` | MolmoAct2 HTTP; `feature_mode` ∈ `{mean_pool,tokens,rl_token,both}` |
| `rlt_models.py` | RL-token AE, chunk actor, `EnsembleCQL` K=10, `CFGradientGuide` |
| `train_rlt.py` | `critic_td_step`, `actor_step`, `guide_step`, `token_step` |
| `train_rlt_online.py` | Online MolmoSpaces rollouts + gated deploy |
| `chunk_replay.py` | Chunk + Token replay; **load_npz must materialize arrays once** (NpzFile re-decompress bug fixed) |
| `encode_offline_demo_tokens.py` | Offline encode demos → token/chunk NPZs via `/act` |
| `droid_ipec_loader.py` | Public `IPEC-COMMUNITY/droid_lerobot` (AV1 via PyAV) |
| `warmup_rlt_token.py` | Token AE recon warmup |
| `warmup_rlt_critic.py` | Re-encode \(z\) from shards, offline critic TD; supports `--chunk_token_glob` / `--chunk_token_shards` |
| `launch_rlt_v7.sh` | 8 GPU online screen (prefer 3/GPU; EGL flock) |
| `launch_rlt_pretrain_demo1k.sh` | Offline 1k+1k encode → token → critic |
| `stop_run.sh` | Stop a run dir’s PIDs |

Python envs:

- Serve / MolmoAct2: `molmoact2/.venv`
- Train / MolmoSpaces / encode client: `molmospaces/.venv`

---

## 3. Experiment lineage (short)

1. **v3 mean-pool CF** (`molmoact2_cf_100m_vlacrit_v3`) — ~9.1M steps, SR≈0.32, **G never enabled** (adv ≪ gate).
2. **v4–v6 RLT screens** — weak action sensitivity → CF rarely on; ranking / explore fixes iterated.
3. **v7** (`runs/rlt_cf_v7_rank`) — normalized explore, local+far+shuffle ranking, K=10, EGL flock + forkserver fixes for multi-trainer/GPU.
4. **Offline demo pretrain** (`runs/rlt_pretrain_demo1k`) — 1k MolmoBot + 1k IPEC DROID → token AE + critic (actor/guide untrained).

Smoke 10-ep baseline (episodes 0–9, seed 0): VLA G=0 **SR=0.50**, G-on **SR=0.30** (`runs/molmoact2_cf_smoke_v2/`).

---

## 4. Offline pretrain (`rlt_pretrain_demo1k`) — DONE

### 4.1 Data

| Source | Path / repo | Episodes | Chunks (max 16/ep) |
|---|---|---|---|
| MolmoBot FrankaPick | `molmospaces/mbdata/FrankaPickOmniCamConfig/part0/train` | 1000 | ~9266 |
| DROID | `IPEC-COMMUNITY/droid_lerobot` (not gated `allenai/droid_lerobot`) | 1000 | ~15622 |
| **Merged** | | **2000** | **24888** |

Encode: frozen MolmoAct2, `--feature_mode tokens --disable_g`, 4 shards × GPUs 0–3.

### 4.2 Artifacts

```
runs/rlt_pretrain_demo1k/
  rlt_token_demo1k.pt              # token AE warmed (8k steps), recon ~4.7
  rlt_cf_pretrain_demo1k.pt        # token + critic (15k TD), actor/guide random
  chunk_replay_merged.npz
  chunk_replay_reencoded.npz       # z filled with warmed encoder
  chunk_token_replay_*_s*.npz    # per-shard tokens (prefer these over 45GB merged)
  token_replay_merged.npz          # ~45GB; avoid full load
  critic_warmup_metrics.jsonl
  critic_warmup_losses.png
  eval_10ep_seed0/rlt_cf_frozen_token/summary.json
```

Final critic metrics (step 15k): `q_std≈0.28`, `q_rank_gap≈0.37`, `action_sensitivity≈0.072`, healthy.

### 4.3 Zero-shot 10-ep eval (same bench as smoke)

- Protocol: `start_episode=0`, `shard_size=10`, `seed=0`, `updates_per_episode=0`, `--force_deploy_rlt`, `explore=0`.
- Config: `rlt_cf_frozen_token` with pretrained ckpt.
- **SR = 0.30** (3/10). Expected: actor + guide were **not** offline-trained; force-deploying them hurts vs pure VLA (0.50).

### 4.4 Pitfalls fixed during pretrain

- `ChunkReplay.load_npz`: calling `data["z"]` inside the row loop re-decompresses NPZ → multi-100GB RAM. **Materialize each array once** (fixed in `chunk_replay.py`).
- Merged `chunk_token_replay_merged.npz` (~45GB object arrays) OOMs if loaded whole; critic warmup streams **shard** NPZs.
- `episode_id` can collide across shards; re-encode uses contiguous episode runs.
- DROID AV1: use PyAV, not decord.
- Compressed object-array NPZ save/merge on NFS is very slow (hours).

---

## 5. Online V7 — current machine state

Run dir: `runs/rlt_cf_v7_rank/`  
Logs: `/tmp/rlt_cf_v7_logs/` (symlink `runs/rlt_cf_v7_rank/logs`).

As of 2026-08-06 ~08:35 UTC:

- `STATUS.json`: 32 trainers listed alive (0 dead).
- Trainers still load **old** warmup:  
  `runs/rlt_token_collect_v4/rlt_cf_warmup_k10.pt`  
  **not** `rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt`.
- Metrics files present but latest rows show **0 steps / 0 eps** on sampled shards → likely early restart or stuck waiting on servers (GPU mem ~2GB/GPU when checked).

**Next online experiment should relaunch with the demo1k pretrain ckpt:**

```bash
cd B1K_AIRI/submodules/rql/my_exps/molmoact2_cf
./stop_run.sh runs/rlt_cf_v7_rank   # if needed
RLT_CKPT=$PWD/runs/rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt \
  INSTANCES_PER_GPU=3 NUM_GPUS=8 \
  bash launch_rlt_v7.sh
```

Prefer **3/GPU** (not 4/GPU); keep EGL flock / forkserver fixes in `train_rlt_online.py`.

---

## 6. What is / isn’t trained offline

| Module | Offline demo1k | Needed for CF deploy |
|---|---|---|
| Token AE | yes (recon) | yes (freeze for frozen_token) |
| Critic ensemble K=10 | yes (TD+MC+CQL+rank) | yes (CF teacher) |
| CF guide \(G_\phi\) | **no** (random init in ckpt) | yes — distill online via `guide_step` |
| Residual actor | **no** | yes — online `actor_step` |
| VLA backbone | frozen always | frozen |

So pretrained critic **is** usable for residual CF online (same K=10 ensemble + `normalized_grad_target`). It is **not** paper flow-CF, and guide distillation still needs online interaction.

---

## 7. Suggested next steps

1. **Relaunch V7** with `RLT_CKPT=.../rlt_cf_pretrain_demo1k.pt`, confirm servers READY and metrics leave 0.
2. Watch early: `action_sensitivity`, `q_rank_gap`, `g_enabled`, cumSR vs `mean_pool_baseline`.
3. If guide stays ~0: lower gate / raise explore / check `guide_step` logs; pretrain should help critic/sensitivity first.
4. Optional: short offline **actor BC / advantage** warmup on demo chunks before online (not done).
5. Optional later: true flow-time CF (out of current V7 scope).
6. Disk: pretrain dir is huge (~100GB+ with token NPZs); delete shard duplicates after confirming ckpt if space-tight.

---

## 8. Quick commands

```bash
# Stop a run
./stop_run.sh runs/rlt_cf_v7_rank

# Offline pretrain (long; GPUs for encode)
ENCODE_SHARDS=4 NUM_EPISODES=1000 TOKEN_STEPS=8000 CRITIC_STEPS=15000 \
  bash launch_rlt_pretrain_demo1k.sh

# Critic-only resume (shards already encoded)
python -u warmup_rlt_critic.py \
  --rlt_ckpt runs/rlt_pretrain_demo1k/rlt_token_demo1k.pt \
  --chunk_replay runs/rlt_pretrain_demo1k/chunk_replay_merged.npz \
  --chunk_token_glob runs/rlt_pretrain_demo1k \
  --out_ckpt runs/rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt \
  --device cuda:0 --steps 15000

# 10-ep force-deploy eval (needs serve with --feature_mode rl_token --rlt_ckpt ...)
# see eval under runs/rlt_pretrain_demo1k/eval_10ep_seed0/
```

---

## 9. Related docs / chats

- Method card PDF: `.../ConsensusFlowMolmo/CF_VLA.pdf`
- Paper CF (full flow): `.../ConsensusFlow/ConsensusFlow.tex`
- Prior Cursor plan: RLT offline pretrain (~1k+1k), agent transcript `1397e3da-d53d-4903-acb6-dee9d6606ddc`

---

## 10. Dual v8 screen (2026-08-06) — residual vs full flow CF

Two matched online experiments on one 8-GPU node:

| Side | GPUs | Instances | Method | Ckpt | Run dir |
|---|---|---|---|---|---|
| Residual CF | 0–3 | 4/GPU (16 workers) | one-shot residual guide (V7) | `rlt_cf_pretrain_demo1k.pt` | `runs/rlt_cf_v8_residual/` |
| Full flow CF | 4–7 | 4/GPU (16 workers) | paper CF denoising ODE (`Q(s,x,t)`, N=10) | `rlt_cf_flow_pretrain_demo1k.pt` | `runs/rlt_cf_v8_flow/` |

### Code changes for full CF
- `rlt_models.py`: `EnsembleTimeCQL`, `FlowVelocityActor`, `FlowCFGuide`, `cf_mode=flow|residual`, Euler `flow_sample`
- `train_rlt.py`: `flow_critic_td_step` (reverse states), `flow_actor_step` (BC+lookahead), `flow_guide_step` (common-scale distill)
- `warmup_flow_critic.py` / `launch_flow_pretrain.sh`: offline time-critic + BC actor pretrain on demo1k reencoded chunks
- `launch_dual_cf_v8.sh` / `launch_rlt_v8_side.sh` / `launch_flow_side_when_ready.sh`

### Launch
```bash
# 1) Flow critic pretrain (GPU 7 while residual starts)
CUDA_VISIBLE_DEVICES=7 DEVICE=cuda:0 bash launch_flow_pretrain.sh

# 2) Residual online (GPUs 0-3)
RESIDUAL_ONLY=1 INSTANCES_PER_GPU=4 bash launch_dual_cf_v8.sh

# 3) Auto-start flow online when ckpt appears
bash launch_flow_side_when_ready.sh
# or manually:
FLOW_ONLY=1 FLOW_CKPT=$PWD/runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt \
  INSTANCES_PER_GPU=4 bash launch_dual_cf_v8.sh
```

Logs: `/tmp/rlt_cf_v8_residual_logs/`, `/tmp/rlt_cf_v8_flow_logs/`, `/tmp/rlt_flow_pretrain_logs/`.
Stop: `./stop_run.sh runs/rlt_cf_v8_residual` and `./stop_run.sh runs/rlt_cf_v8_flow`.

