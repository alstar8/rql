#!/usr/bin/env bash
# Restart v7 trainers only (servers kept). Applies forkserver/EGL fixes.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
RUN="${1:-$ROOT/runs/rlt_cf_v7_rank}"
LOCAL_LOG="${LOCAL_LOG_DIR:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_cf_v7_logs}"
PYTHON="$ROOT/../../../molmospaces/.venv/bin/python"
RLT_CKPT="${RLT_CKPT:-$ROOT/runs/rlt_token_collect_v4/rlt_cf_warmup_k10.pt}"
NUM_GPUS=8
INSTANCES_PER_GPU=3
NUM_CONFIGS=4
TOTAL_WORKERS=$((NUM_GPUS * INSTANCES_PER_GPU))
WORKERS_PER_CONFIG=$((TOTAL_WORKERS / NUM_CONFIGS))
BASE_PORT=8600
BENCH_N=1000
TARGET_ENV_STEPS=4166667
UPDATES_PER_EPISODE=8
LOG_EVERY_EPISODES=50
HORIZON=500
N_CRITICS=10
CONFIG_NAMES=(mean_pool_baseline rlt_actor_no_guide rlt_cf_frozen_token rlt_cf_online_token)

mkdir -p "$LOCAL_LOG" "$RUN/pids" /workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_egl_locks

# Kill only live python trainers (not this bash script).
while read -r pid; do
  [[ -n "$pid" ]] || continue
  kill -TERM "$pid" 2>/dev/null || true
done < <(pgrep -f "/train_rlt_online.py" || true)
sleep 4
while read -r pid; do
  [[ -n "$pid" ]] || continue
  kill -KILL "$pid" 2>/dev/null || true
done < <(pgrep -f "/train_rlt_online.py" || true)
sleep 1

rm -f /workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_egl_locks/*.lock
echo "[restart $(date -Is)] restarting ${TOTAL_WORKERS} trainers" | tee -a "$LOCAL_LOG/restart_trainers.log"

for ((worker = 0; worker < TOTAL_WORKERS; worker++)); do
  gpu=$((worker / INSTANCES_PER_GPU))
  port=$((BASE_PORT + worker))
  config_idx=$((worker % NUM_CONFIGS))
  config_worker=$((worker / NUM_CONFIGS))
  config_name="${CONFIG_NAMES[config_idx]}"
  shard_base=$((BENCH_N / WORKERS_PER_CONFIG))
  start_episode=$((config_worker * shard_base))
  shard_size=$shard_base
  if ((config_worker == WORKERS_PER_CONFIG - 1)); then
    shard_size=$((BENCH_N - start_episode))
  fi
  shard_out="$RUN/$config_name/shard_$config_worker"
  logfile="$LOCAL_LOG/train_${config_name}_s${config_worker}_gpu${gpu}.log"
  pidfile="$RUN/pids/train_${config_name}_s${config_worker}.pid"
  mkdir -p "$shard_out"
  if [[ -f "$logfile" ]]; then
    mv "$logfile" "${logfile}.pre_forkfix_$(date -u +%Y%m%dT%H%M%SZ)" || true
  fi

  extra=()
  case "$config_name" in
    mean_pool_baseline)
      extra=(--actor_mode vla_only --no_cf_guide --freeze_token --updates_per_episode 0)
      ;;
    rlt_actor_no_guide)
      extra=(--actor_mode rlt --no_cf_guide --freeze_token --updates_per_episode "$UPDATES_PER_EPISODE")
      ;;
    rlt_cf_frozen_token)
      extra=(--actor_mode rlt --use_cf_guide --freeze_token --updates_per_episode "$UPDATES_PER_EPISODE")
      ;;
    rlt_cf_online_token)
      extra=(--actor_mode rlt --use_cf_guide --tune_token_online --updates_per_episode "$UPDATES_PER_EPISODE")
      ;;
  esac

  echo "[restart] w=$worker $config_name s=$config_worker gpu=$gpu port=$port" \
    | tee -a "$LOCAL_LOG/restart_trainers.log"
  (
    exec setsid env \
      RLT_CF_V4_RUN_DIR="$RUN" \
      RLT_EGL_LOCK_DIR=/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_egl_locks \
      RLT_EGL_COOLDOWN_SEC=0.5 \
      MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}" \
      MUJOCO_GL=egl \
      PYOPENGL_PLATFORM=egl \
      MUJOCO_EGL_DEVICE_ID="$gpu" \
      CUDA_VISIBLE_DEVICES="$gpu" \
      "$PYTHON" "$ROOT/train_rlt_online.py" \
      --server_host localhost --server_port "$port" --device cuda:0 \
      --out_dir "$shard_out" --config_name "$config_name" \
      --target_env_steps "$TARGET_ENV_STEPS" \
      --start_episode "$start_episode" --shard_size "$shard_size" \
      --horizon "$HORIZON" --log_every_episodes "$LOG_EVERY_EPISODES" \
      --replay_out "$shard_out/chunk_replay.npz" --seed "$worker" \
      --n_critics "$N_CRITICS" \
      --explore_residual_std 0.05 --rank_coef 1.0 --rank_margin 0.05 \
      --rank_noise 0.08 --far_rank_coef 0.5 --far_rank_noise 0.35 \
      --shuffle_rank_coef 0.5 --target_noise 0.02 \
      --g_start_episodes 40 --g_min_advantage 0.005 \
      --g_min_action_sensitivity 0.003 --gate_sensitivity_noise 0.08 \
      --cql_n_actions 8 --rlt_ckpt "$RLT_CKPT" \
      "${extra[@]}"
  ) >>"$logfile" 2>&1 &
  pid=$!
  printf '%s trainer config=%s shard=%s gpu=%s port=%s\n' \
    "$pid" "$config_name" "$config_worker" "$gpu" "$port" >"$pidfile"
  sleep 6
done

echo "[restart] all trainers spawned" | tee -a "$LOCAL_LOG/restart_trainers.log"
sleep 25
alive=0
dead=0
for f in "$RUN"/pids/train_*.pid; do
  read -r p _ <"$f"
  if kill -0 "$p" 2>/dev/null; then
    alive=$((alive + 1))
  else
    dead=$((dead + 1))
    echo "DEAD $(cat "$f")"
  fi
done
echo "alive=$alive dead=$dead" | tee -a "$LOCAL_LOG/restart_trainers.log"
