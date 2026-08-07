# MolmoSpaces π₀.₅ Pick baseline smoke eval

**Date:** 2026-07-31  
**Fork:** https://github.com/alstar8/molmospaces (upstream: allenai/molmospaces)  
**Local clone:** `B1K_AIRI/submodules/molmospaces`  
**OpenPI:** `B1K_AIRI/submodules/openpi`

## Task / protocol (default MS Pick-v1.1)

| Item | Value |
|---|---|
| Benchmark | **Pick-v1.1** (`FrankaPickDroidMiniBench`) |
| Spec | `assets/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231` |
| Episodes in full bench | **1000** |
| Eval config | `molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig` |
| Policy | remote OpenPI websocket `localhost:8080` |
| Horizon | 500 steps (`--task_horizon_steps 500`) |
| Control rate | `policy_dt_ms=66` (~15 Hz), `end_on_success=True` |
| Aggregation | `scripts/benchmarks/eval_to_csv.py … --success-condition both` |

Docs: `docs/evaluation_guide.md`, `docs/ms-bench.md`.

## Checkpoint

- Official GCS: `gs://openpi-assets/checkpoints/pi05_droid_jointpos` (~12 GB)
- Cached at: `~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos`
- Served with OpenPI config name `pi05_droid` (current openpi rename; symlink `pi05_droid → pi05_droid_jointpos`)

```bash
# Server
cd submodules/openpi
OPENPI_DATA_HOME=~/.cache/openpi CUDA_VISIBLE_DEVICES=0 \
  uv run scripts/serve_policy.py --port=8080 policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_droid
```

## Smoke result (first 10 / 1000 episodes)

| Metric | Value |
|---|---|
| Success (at-end) | **0 / 10 (0.0%)** |
| Oracle success | **0 / 10 (0.0%)** |
| Output | `molmospaces/eval_output/pi05_pick_v1_1_smoke/PiPolicyEvalConfig/20260731_082447` |
| CSV | `/tmp/pi05_pick_smoke.csv` |

Pipeline ran cleanly (policy connected, scenes loaded, videos + H5 saved). Zero successes on this 10-ep prefix is a **smoke measurement**, not the leaderboard score — full protocol is all 1000 episodes (~1–2 min/ep → many GPU-hours).

### Notes / caveats

1. Eval warned about **objaverse asset version** newer than pinned benchmark version; may affect reproducibility vs paper/leaderboard.
2. Current OpenPI uses config `pi05_droid`; MolmoSpaces docs still say `pi05_droid_jointpos` — weights from the jointpos GCS path were used.
3. For a full validation run, drop `--max_episodes` (or set large) and keep the same server + `PiPolicyEvalConfig`.

## Reproduce smoke

```bash
export MLSPACES_ASSETS_DIR=~/.cache/molmospaces/assets
export MUJOCO_GL=egl
cd B1K_AIRI/submodules/molmospaces && source .venv/bin/activate
BENCH=$MLSPACES_ASSETS_DIR/benchmarks/molmospaces-bench-v1/procthor-10k/FrankaPickDroidMiniBench/FrankaPickDroidMiniBench_json_benchmark_20251231
python molmo_spaces/evaluation/eval_main.py \
  molmo_spaces.evaluation.configs.evaluation_configs:PiPolicyEvalConfig \
  --benchmark_dir "$BENCH" \
  --checkpoint_path ~/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos \
  --task_horizon_steps 500 --max_episodes 10 --num_workers 1 --no_wandb \
  --output_dir eval_output/pi05_pick_v1_1_smoke
```
