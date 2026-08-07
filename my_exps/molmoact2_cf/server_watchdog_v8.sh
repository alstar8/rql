#!/usr/bin/env bash
# Watchdog: keep MolmoAct2 CF VLA servers alive for one v8 side (residual or flow).
# Restarts unhealthy/dead servers; leaves trainers alone (trainer_watchdog_v8.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
RUN="${1:-$ROOT/runs/rlt_cf_v8_residual}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
LOCAL_LOG="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v8_residual_logs}"
mkdir -p "${B1K_TMP}" "${LOCAL_LOG}"
MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOACT2_PYTHON="${MOLMOACT2}/.venv/bin/python"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
POLL_SEC="${POLL_SEC:-45}"
NUM_GPUS="${NUM_GPUS:-4}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}"
TOTAL_WORKERS=$((NUM_GPUS * INSTANCES_PER_GPU))
BASE_PORT="${BASE_PORT:-8600}"
NUM_CONFIGS=4
CF_MODE="${CF_MODE:-residual}"
RLT_CKPT="${RLT_CKPT:-}"
DEFAULT_FEATURE_MODE="tokens"
if [[ -n "${RLT_CKPT}" ]]; then
  DEFAULT_FEATURE_MODE="rl_token"
fi
CONFIG_NAMES=(mean_pool_baseline rlt_actor_no_guide rlt_cf_frozen_token rlt_cf_online_token)

if [[ -n "${UV_BIN}" && -x "${UV_BIN}" ]]; then
  SERVE_CMD=("${UV_BIN}" run python)
else
  if [[ ! -x "${MOLMOACT2_PYTHON}" ]]; then
    echo "[server_watchdog] neither uv nor MolmoAct2 Python found (${MOLMOACT2_PYTHON})" >&2
    exit 1
  fi
  SERVE_CMD=("${MOLMOACT2_PYTHON}")
fi

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -ra GPU_ARR <<< "${GPU_IDS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_ARR=()
  for ((i = 0; i < NUM_GPUS; i++)); do GPU_ARR+=("$i"); done
fi
if ((${#GPU_ARR[@]} != NUM_GPUS)); then
  echo "[server_watchdog] GPU_IDS has ${#GPU_ARR[@]} entries but NUM_GPUS=${NUM_GPUS}" >&2
  exit 1
fi

if [[ "${RUN}" != /* ]]; then
  RUN="${ROOT}/${RUN}"
fi
mkdir -p "${LOCAL_LOG}" "${RUN}/pids"
WATCH_LOG="${LOCAL_LOG}/server_watchdog.log"
PIDFILE_SELF="${RUN}/pids/server_watchdog.pid"
printf '%s server_watchdog\n' "$$" > "${PIDFILE_SELF}"

server_healthy() {
  local port="$1"
  curl -sf --max-time 3 "http://127.0.0.1:${port}/act" 2>/dev/null | grep -q '"status":"ok"'
}

start_server() {
  local worker="$1"
  local gpu_idx=$((worker / INSTANCES_PER_GPU))
  local gpu="${GPU_ARR[$gpu_idx]}"
  local port=$((BASE_PORT + worker))
  local config_idx=$((worker % NUM_CONFIGS))
  local feature_mode="${DEFAULT_FEATURE_MODE}"
  if [[ "${CONFIG_NAMES[config_idx]}" == "rlt_cf_online_token" ]]; then
    feature_mode="tokens"
  fi
  local logfile="${LOCAL_LOG}/server_w${worker}_gpu${gpu}.log"
  local pidfile="${RUN}/pids/server_w${worker}.pid"

  if [[ -f "${pidfile}" ]]; then
    local old
    read -r old _ <"${pidfile}" || old=""
    if [[ "${old}" =~ ^[0-9]+$ ]]; then
      kill -TERM "${old}" 2>/dev/null || true
      sleep 1
      kill -KILL "${old}" 2>/dev/null || true
    fi
  fi
  # Best-effort clear of any orphan listener on this port for this serve.py.
  pkill -f "${ROOT}/serve.py .*--port ${port}" 2>/dev/null || true
  sleep 1

  if [[ -f "${logfile}" ]]; then
    mv "${logfile}" "${logfile}.died_$(date -u +%Y%m%dT%H%M%SZ)" || true
  fi

  local -a command=(
    "${SERVE_CMD[@]}" "${ROOT}/serve.py"
    --host 0.0.0.0
    --port "${port}"
    --device cuda:0
    --dtype bfloat16
    --disable_g
    --feature_mode "${feature_mode}"
  )
  if [[ -n "${RLT_CKPT}" && "${feature_mode}" == "rl_token" ]]; then
    command+=(--rlt_ckpt "${RLT_CKPT}")
  fi

  echo "[server_watchdog $(date -Is)] RESTART w=${worker} gpu=${gpu} port=${port} mode=${feature_mode}" \
    | tee -a "${WATCH_LOG}"
  (
    cd "${MOLMOACT2}"
    exec setsid env \
      RLT_CF_V4_RUN_DIR="${RUN}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
      "${command[@]}"
  ) >"${logfile}" 2>&1 &
  local pid=$!
  printf '%s server worker=%s gpu=%s port=%s\n' \
    "${pid}" "${worker}" "${gpu}" "${port}" >"${pidfile}"
}

echo "[server_watchdog $(date -Is)] starting poll=${POLL_SEC}s run=${RUN} gpus=${GPU_ARR[*]} ports=${BASE_PORT}-$((BASE_PORT + TOTAL_WORKERS - 1))" \
  | tee -a "${WATCH_LOG}"

while true; do
  restarted=0
  for ((worker = 0; worker < TOTAL_WORKERS; worker++)); do
    port=$((BASE_PORT + worker))
    if server_healthy "${port}"; then
      continue
    fi
    echo "[server_watchdog $(date -Is)] UNHEALTHY port=${port} w=${worker} — restarting" \
      | tee -a "${WATCH_LOG}"
    start_server "${worker}"
    restarted=$((restarted + 1))
    # Give the model time to load before checking the next unhealthy port.
    sleep "${SERVER_RESTART_SETTLE_SEC:-90}"
  done
  if [[ "${ONCE:-0}" == "1" ]]; then
    echo "[server_watchdog] ONCE=1 finished restarted=${restarted}" | tee -a "${WATCH_LOG}"
    exit 0
  fi
  sleep "${POLL_SEC}"
done
