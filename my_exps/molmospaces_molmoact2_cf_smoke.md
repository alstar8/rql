# MolmoAct2 + CF Offline Warmup Smoke

**Date:** 2026-07-31  
**Design:** [`molmoact2_cf_finetune.md`](molmoact2_cf_finetune.md)  
**Code:** [`molmoact2_cf/`](molmoact2_cf/)  
**Checkpoint:** `molmoact2_cf/runs/molmoact2_cf_smoke/molmoact2_cf.pt`  
**Warmup data:** MolmoBot `FrankaPickOmniCamConfig` part0 train shard → stratified buffer (pre- vs post-grasp frames)

## Setup

| Piece | Detail |
|---|---|
| Backbone | Frozen `allenai/MolmoAct2-DROID` |
| Features | Proprio \(s\in\mathbb{R}^8\) (joint+gripper), z-scored |
| Critic | Ensemble CQL, MC sparse returns (Phase 1: 2k steps) |
| Residual \(G\) | Endpoint MLP, `max_delta=0.05` tanh-capped (Phase 2: 1k steps) |
| AE BC | Skipped for smoke (`ae_bc_steps=0`) |
| Server | [`molmoact2_cf/serve.py`](molmoact2_cf/serve.py) on `:8000` |

## Pick-v1.1 first-10 smoke

Same bench / horizon as baseline ([`molmospaces_molmoact2_pick_baseline.md`](molmospaces_molmoact2_pick_baseline.md)).

| Policy | At-end SR | Oracle SR | Notes |
|---|---:|---:|---|
| MolmoAct2-DROID baseline (prior) | **6/10 (60%)** | 60% | Original host server |
| CF **\(G=0\)** (this run) | **4/10 (40%)** | 40% | Same frozen expert via CF serve |
| CF uncapped \(G\) (ablation) | **0/10 (0%)** | 0% | Residual RMS ~0.22 — harmful |
| CF **capped \(G\)** (`max_delta=0.05`) | **5/10 (50%)** | 50% | Residual RMS ~0.03 |

**Role-gap** \(\Delta=J(v,G)-J(v,0)=50\%-40\%=+10\%\) (n=10; positive but noisy).

## Offline training gates

| Gate | Result |
|---|---|
| Buffer | 8421 transitions, success_frac≈0.26 (step-level) |
| Critic AUROC (end Phase 1) | ~0.56 (weak; smoke-ok) |
| Predicted advantage Phase 2 | \(+9.8\) with residual RMS \(0.031\) |

## Reproduce

```bash
# Buffer + train
cd rql/my_exps/molmoact2_cf
python build_buffer.py --data_dir .../mbdata/FrankaPickOmniCamConfig/part0/train \
  --out runs/pick_buffer.npz --max_episodes 200
CUDA_VISIBLE_DEVICES=4 python train_offline.py --buffer runs/pick_buffer.npz \
  --out_dir runs/molmoact2_cf_smoke --phase1_steps 2000 --phase2_steps 1000 \
  --target_divergence 0.001 --ae_bc_steps 0

# Server (G on)
cd molmoact2
CUDA_VISIBLE_DEVICES=4 uv run python ../rql/my_exps/molmoact2_cf/serve.py \
  --cf_ckpt .../runs/molmoact2_cf_smoke/molmoact2_cf.pt --enable_g --action_clip 0.05

# Eval
cd molmospaces && source .venv/bin/activate
export MLSPACES_ASSETS_DIR=~/.cache/molmospaces/assets MUJOCO_GL=egl
python molmo_spaces/evaluation/eval_main.py \
  molmo_spaces.evaluation.configs.evaluation_configs:MolmoAct2PolicyEvalConfig \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231 \
  --task_horizon_steps 500 --max_episodes 10 --num_workers 1 --no_wandb \
  --output_dir eval_output/molmoact2_cf_g_capped_smoke
```

## Smoke verdict

**Pass (pipeline).** End-to-end offline CF warmup + CF serve + same-10 eval works.  
\(G=0\) is within n=10 noise of the prior 6/10 (4/10 here). Uncapped \(G\) destroys SR; tanh-capped \(G\) is flat-to-slightly-up vs \(G=0\) on this draw.

### Next (out of scope here)

- Phase 3 budgeted online on Pick-v1.1  
- Light AE BC with frozen VLM  
- Stronger critic (true fail eps / VLM features \(h\))  
- Larger n eval for SR confidence
