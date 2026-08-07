# MolmoAct2 Pick-v1.1 smoke (vs π₀.₅)

**Date:** 2026-07-31  
**Fork:** https://github.com/alstar8/molmoact2  
**Clone:** `B1K_AIRI/submodules/molmoact2`  
**Checkpoint:** `allenai/MolmoAct2-DROID`  
**Server:** `examples/droid/host_server_droid.py` on `:8000` (patched `inference_action_mode`)  
**Client:** `molmo_spaces/.../molmoact2_policy.py` + `MolmoAct2PolicyEvalConfig`

## Same protocol as π₀.₅ smoke

- Bench: MS Pick-v1.1 `FrankaPickDroidMiniBench` (first **10** / 1000 eps)
- Horizon: 500 steps, `policy_dt_ms=66`

## Results

| Policy | At-end SR | Oracle SR |
|---|---:|---:|
| π₀.₅ DROID (same 10 eps) | 0/10 (**0%**) | 0% |
| **MolmoAct2-DROID (same 10 eps)** | **6/10 (60%)** | **60%** |

Output: `molmospaces/eval_output/molmoact2_pick_v1_1_smoke/MolmoAct2PolicyEvalConfig/20260731_090116/`

Leaderboard combined MS (reference): MolmoAct2 ~22.5%, π₀.₅ ~14.6% oracle — so MolmoAct2 beating π₀.₅ here is expected; 60% on n=10 is high variance vs full 1000-ep bench.

## Reproduce

```bash
# Terminal 1 — server
cd submodules/molmoact2
CUDA_VISIBLE_DEVICES=5 HF_HOME=~/.cache/huggingface \
  uv run python examples/droid/host_server_droid.py --host 0.0.0.0 --port 8000 --dtype bfloat16

# Terminal 2 — eval
cd submodules/molmospaces && source .venv/bin/activate
export MLSPACES_ASSETS_DIR=~/.cache/molmospaces/assets MUJOCO_GL=egl
python molmo_spaces/evaluation/eval_main.py \
  molmo_spaces.evaluation.configs.evaluation_configs:MolmoAct2PolicyEvalConfig \
  --benchmark_dir $MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231 \
  --task_horizon_steps 500 --max_episodes 10 --num_workers 1 --no_wandb \
  --output_dir eval_output/molmoact2_pick_v1_1_smoke
```
