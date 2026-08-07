#!/usr/bin/env bash
# Watchdog: restart only dead v7 trainers (servers untouched).
# Archives crashed logs as *.died_<UTC> and refreshes PID manifests.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
RUN="${1:-$ROOT/runs/rlt_cf_v7_rank}"
LOCAL_LOG="${LOCAL_LOG_DIR:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_cf_v7_logs}"
PYTHON="$ROOT/../../../molmospaces/.venv/bin/python"
RLT_CKPT="${RLT_CKPT:-$ROOT/runs/rlt_token_collect_v4/rlt_cf_warmup_k10.pt}"
POLL_SEC="${POLL_SEC:-60}"
NUM_GPUS="${NUM_GPUS:-8}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}"
NUM_CONFIGS=4
TOTAL_WORKERS=$((NUM_GPUS * INSTANCES_PER_GPU))
WORKERS_PER_CONFIG=$((TOTAL_WORKERS / NUM_CONFIGS))
BASE_PORT="${BASE_PORT:-8600}"
BENCH_N="${BENCH_N:-1000}"
TARGET_ENV_STEPS="${TARGET_ENV_STEPS:-3125000}"
UPDATES_PER_EPISODE="${UPDATES_PER_EPISODE:-8}"
LOG_EVERY_EPISODES="${LOG_EVERY_EPISODES:-50}"
CKPT_EVERY_EPISODES="${CKPT_EVERY_EPISODES:-10}"
HORIZON="${HORIZON:-500}"
N_CRITICS="${N_CRITICS:-10}"
CONFIG_NAMES=(mean_pool_baseline rlt_actor_no_guide rlt_cf_frozen_token rlt_cf_online_token)
if (( TOTAL_WORKERS % NUM_CONFIGS != 0 )); then
  echo "[watchdog] ${TOTAL_WORKERS} workers cannot be split across ${NUM_CONFIGS} configs" >&2
  exit 1
fi

mkdir -p "$LOCAL_LOG" "$RUN/pids" /workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_egl_locks
STATUS_JSON="$RUN/STATUS.json"
WATCH_LOG="$LOCAL_LOG/trainer_watchdog.log"

is_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

write_status() {
  local alive=0 dead=0
  local -a dead_names=()
  local f p name
  for f in "$RUN"/pids/train_*.pid; do
    [[ -f "$f" ]] || continue
    read -r p _ <"$f" || p=""
    name=$(basename "$f" .pid)
    if is_alive "$p"; then
      alive=$((alive + 1))
    else
      dead=$((dead + 1))
      dead_names+=("$name")
    fi
  done
  python3 - "$STATUS_JSON" "$alive" "$dead" "${dead_names[@]}" <<'PY'
import json, sys, time
path, alive, dead, *names = sys.argv[1:]
payload = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "alive_trainers": int(alive),
    "dead_trainers": int(dead),
    "dead_names": names,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
print(f"[status] alive={alive} dead={dead} names={names}")
PY
}

start_trainer() {
  local worker="$1"
  local gpu=$((worker / INSTANCES_PER_GPU))
  local port=$((BASE_PORT + worker))
  local config_idx=$((worker % NUM_CONFIGS))
  local config_worker=$((worker / NUM_CONFIGS))
  local config_name="${CONFIG_NAMES[config_idx]}"
  local shard_base=$((BENCH_N / WORKERS_PER_CONFIG))
  local start_episode=$((config_worker * shard_base))
  local shard_size=$shard_base
  if ((config_worker == WORKERS_PER_CONFIG - 1)); then
    shard_size=$((BENCH_N - start_episode))
  fi
  local shard_out="$RUN/$config_name/shard_$config_worker"
  local logfile="$LOCAL_LOG/train_${config_name}_s${config_worker}_gpu${gpu}.log"
  local pidfile="$RUN/pids/train_${config_name}_s${config_worker}.pid"
  mkdir -p "$shard_out"

  if [[ -f "$logfile" ]]; then
    mv "$logfile" "${logfile}.died_$(date -u +%Y%m%dT%H%M%SZ)" || true
  fi

  local -a extra=()
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

  echo "[watchdog $(date -Is)] RESTART w=$worker $config_name s=$config_worker gpu=$gpu port=$port" \
    | tee -a "$WATCH_LOG"
  (
    exec setsid env \
      RLT_CF_V4_RUN_DIR="$RUN" \
      RLT_EGL_LOCK_DIR=/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_egl_locks \
      RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-1.5}" \
      RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-4}" \
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
      --ckpt_every_episodes "$CKPT_EVERY_EPISODES" \
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
  local pid=$!
  printf '%s trainer config=%s shard=%s gpu=%s port=%s\n' \
    "$pid" "$config_name" "$config_worker" "$gpu" "$port" >"$pidfile"
}

echo "[watchdog $(date -Is)] starting poll=${POLL_SEC}s run=$RUN" | tee -a "$WATCH_LOG"

# One-shot pass if ONCE=1, else loop forever.
while true; do
  restarted=0
  for ((worker = 0; worker < TOTAL_WORKERS; worker++)); do
    config_idx=$((worker % NUM_CONFIGS))
    config_worker=$((worker / NUM_CONFIGS))
    config_name="${CONFIG_NAMES[config_idx]}"
    pidfile="$RUN/pids/train_${config_name}_s${config_worker}.pid"
    pid=""
    if [[ -f "$pidfile" ]]; then
      read -r pid _ <"$pidfile" || pid=""
    fi
    if is_alive "$pid"; then
      continue
    fi
    start_trainer "$worker"
    restarted=$((restarted + 1))
    # Stagger restarts so EGL init does not stampede again.
    sleep "${TRAINER_STAGGER_SEC:-8}"
  done
  write_status | tee -a "$WATCH_LOG"
  if [[ "${ONCE:-0}" == "1" ]]; then
    echo "[watchdog] ONCE=1 finished restarted=$restarted" | tee -a "$WATCH_LOG"
    exit 0
  fi
  sleep "$POLL_SEC"
done
