#!/usr/bin/env bash
# Watchdog: keep MolmoAct2 CF expert servers alive for the 100M-step run.
#
# Args: LOG_DIR [NUM_GPUS=8] [BASE_PORT=8000] [POLL_SEC=30] [INSTANCES_PER_GPU=1]
# Server global index g = gpu * INSTANCES_PER_GPU + local → port BASE_PORT+g
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MOLMOACT2="${ROOT}/../../../molmoact2"
CKPT="${ROOT}/runs/molmoact2_cf_smoke/molmoact2_cf.pt"
LOG_DIR="${1:?log dir}"
NUM_GPUS="${2:-8}"
BASE_PORT="${3:-8000}"
POLL_SEC="${4:-30}"
INSTANCES_PER_GPU="${5:-1}"

TOTAL_INSTANCES=$((NUM_GPUS * INSTANCES_PER_GPU))
mkdir -p "${LOG_DIR}"

start_server() {
  local g="$1"
  local gpu=$((g / INSTANCES_PER_GPU))
  local port=$((BASE_PORT + g))
  local pidfile="${LOG_DIR}/server_g${g}.pid"
  local logfile="${LOG_DIR}/server_g${g}_gpu${gpu}.log"
  echo "[watchdog $(date -Is)] (re)start server inst=${g} gpu=${gpu} port=${port}"
  if [[ -f "${pidfile}" ]]; then
    local old
    old="$(cat "${pidfile}" 2>/dev/null || true)"
    if [[ -n "${old}" ]]; then
      kill "${old}" 2>/dev/null || true
      sleep 1
      kill -9 "${old}" 2>/dev/null || true
    fi
  fi
  pkill -f "molmoact2_cf/serve.py .*--port ${port}" 2>/dev/null || true
  sleep 1
  cd "${MOLMOACT2}"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
    nohup uv run python "${ROOT}/serve.py" \
      --host 0.0.0.0 --port "${port}" --dtype bfloat16 --device cuda:0 \
      --disable_g \
      >> "${logfile}" 2>&1 &
  echo $! > "${pidfile}"
}

for ((g=0; g<TOTAL_INSTANCES; g++)); do
  start_server "${g}"
  sleep 6
done

echo "[watchdog] initial ${TOTAL_INSTANCES} servers launched; polling every ${POLL_SEC}s"
while true; do
  for ((g=0; g<TOTAL_INSTANCES; g++)); do
    port=$((BASE_PORT + g))
    if ! curl -sf --max-time 3 "http://127.0.0.1:${port}/act" 2>/dev/null | grep -q '"status":"ok"'; then
      echo "[watchdog $(date -Is)] UNHEALTHY port=${port} inst=${g} — restarting"
      start_server "${g}"
      sleep 90
    fi
  done
  sleep "${POLL_SEC}"
done
