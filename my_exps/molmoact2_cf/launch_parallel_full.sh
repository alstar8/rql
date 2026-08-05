#!/usr/bin/env bash
# Launch multi-GPU × multi-instance MolmoAct2+CF online train.
#
# Default: 8 GPUs × 4 instances/GPU = 32 shards over MiniBench (1000 eps).
# Each instance: own MolmoAct2 server (port) + train_full shard on the same GPU.
# ~14GB VRAM × 4 ≈ 56GB per 80GB GPU — fits with headroom for CF/EGL.
#
# Usage:
#   bash launch_parallel_full.sh
#   bash launch_parallel_full.sh --num_gpus 8 --instances_per_gpu 4 --total_episodes 1000

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
CKPT="${ROOT}/runs/molmoact2_cf_smoke/molmoact2_cf.pt"
BUFFER="${ROOT}/runs/pick_buffer.npz"
OUT_ROOT="${ROOT}/runs/molmoact2_cf_full_8x4"
LOG_DIR="${OUT_ROOT}/logs"
BASE_PORT=8000
NUM_GPUS=8
INSTANCES_PER_GPU=4
TOTAL_EPS=1000
LOG_EVERY=10
UPDATES_PER_EP=50
HORIZON=500
STAGGER_SEC=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --instances_per_gpu) INSTANCES_PER_GPU="$2"; shift 2 ;;
    --total_episodes) TOTAL_EPS="$2"; shift 2 ;;
    --log_every) LOG_EVERY="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; LOG_DIR="${OUT_ROOT}/logs"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --stagger_sec) STAGGER_SEC="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TOTAL_INSTANCES=$((NUM_GPUS * INSTANCES_PER_GPU))
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "[parallel] stopping prior CF serve/train ..."
pkill -f 'molmoact2_cf/serve.py' 2>/dev/null || true
pkill -f 'molmoact2_cf/train_full.py' 2>/dev/null || true
sleep 4

EPS_BASE=$(( TOTAL_EPS / TOTAL_INSTANCES ))
echo "[parallel] GPUs=${NUM_GPUS} × ${INSTANCES_PER_GPU}/gpu = ${TOTAL_INSTANCES} instances"
echo "[parallel] total_eps=${TOTAL_EPS} ~${EPS_BASE}/instance out=${OUT_ROOT}"

# --- Start servers: global index g = gpu*K + k ---
for ((g=0; g<TOTAL_INSTANCES; g++)); do
  GPU=$((g / INSTANCES_PER_GPU))
  LOCAL=$((g % INSTANCES_PER_GPU))
  PORT=$((BASE_PORT + g))
  echo "[parallel] server inst=${g} gpu=${GPU} local=${LOCAL} port=${PORT}"
  (
    cd "${MOLMOACT2}"
    CUDA_VISIBLE_DEVICES="${GPU}" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
      uv run python "${ROOT}/serve.py" \
        --host 0.0.0.0 --port "${PORT}" --dtype bfloat16 --device cuda:0 \
        --cf_ckpt "${CKPT}" --disable_g \
        > "${LOG_DIR}/server_g${g}_gpu${GPU}.log" 2>&1
  ) &
  echo $! > "${LOG_DIR}/server_g${g}.pid"
  sleep "${STAGGER_SEC}"
done

echo "[parallel] waiting for ${TOTAL_INSTANCES} servers ..."
for ((g=0; g<TOTAL_INSTANCES; g++)); do
  PORT=$((BASE_PORT + g))
  for t in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:${PORT}/act" 2>/dev/null | grep -q '"status":"ok"'; then
      echo "[parallel] server inst=${g} port=${PORT} READY"
      break
    fi
    if (( t == 180 )); then
      echo "[parallel] ERROR: server inst=${g} port=${PORT} not ready"
      tail -50 "${LOG_DIR}/server_g${g}_gpu$((g / INSTANCES_PER_GPU)).log" || true
      exit 1
    fi
    sleep 5
  done
done

# --- Start train shards ---
START=0
for ((g=0; g<TOTAL_INSTANCES; g++)); do
  GPU=$((g / INSTANCES_PER_GPU))
  PORT=$((BASE_PORT + g))
  if (( g == TOTAL_INSTANCES - 1 )); then
    N=$(( TOTAL_EPS - START ))
  else
    N="${EPS_BASE}"
  fi
  SHARD_OUT="${OUT_ROOT}/shard_${g}"
  mkdir -p "${SHARD_OUT}"
  echo "[parallel] train inst=${g} gpu=${GPU} port=${PORT} episodes=[${START},${START}+${N})"
  (
    cd "${MOLMOSPACES}"
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}"
    export MUJOCO_GL=egl
    # Physical GPU index — EGL ignores CUDA_VISIBLE_DEVICES.
    export MUJOCO_EGL_DEVICE_ID="${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" \
      python "${ROOT}/train_full.py" \
        --cf_ckpt "${CKPT}" \
        --buffer "${BUFFER}" \
        --out_dir "${SHARD_OUT}" \
        --device cuda:0 \
        --server_host localhost \
        --server_port "${PORT}" \
        --start_episode "${START}" \
        --num_episodes "${N}" \
        --log_every_episodes "${LOG_EVERY}" \
        --updates_per_episode "${UPDATES_PER_EP}" \
        --horizon "${HORIZON}" \
        --seed "${g}" \
        > "${LOG_DIR}/train_g${g}_gpu${GPU}.log" 2>&1
  ) &
  echo $! > "${LOG_DIR}/train_g${g}.pid"
  START=$((START + N))
done

echo "[parallel] all ${TOTAL_INSTANCES} shards launched"
echo "  logs:    ${LOG_DIR}/train_g*_gpu*.log"
echo "  metrics: ${OUT_ROOT}/shard_*/metrics.jsonl"
echo "  watch:   grep -hE 'ep .*|METRICS' ${LOG_DIR}/train_g*_gpu*.log | tail"
echo "  gpus:    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv"
wait
echo "[parallel] all shards finished"
