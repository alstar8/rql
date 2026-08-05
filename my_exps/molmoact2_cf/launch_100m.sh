#!/usr/bin/env bash
# Launch 100M valid sim env-step MolmoAct2+CF experiment (VLA-feature critic).
#
# Default: 8 GPUs × 4 instances/GPU = 32 shards.
# Global target: 100_000_000 env steps → 3.125M steps / instance.
#
# Usage:
#   nohup bash launch_100m.sh --instances_per_gpu 4 \
#     --out_root .../runs/molmoact2_cf_100m_vlacrit \
#     --ckpt .../runs/molmoact2_cf_vlacrit_warmup_v2/molmoact2_cf.pt \
#     > /tmp/molmoact2_cf_100m_vlacrit_launch.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${MOLMOSPACES}/.venv/bin/python"
CKPT="${ROOT}/runs/molmoact2_cf_vlacrit_warmup_v3/molmoact2_cf.pt"
BUFFER="${ROOT}/runs/pick_buffer_vla_g0.npz"
OUT_ROOT="${ROOT}/runs/molmoact2_cf_100m_vlacrit_v3"
LOG_DIR="${OUT_ROOT}/logs"
BASE_PORT=8000
NUM_GPUS=8
INSTANCES_PER_GPU=4
TARGET_TOTAL=100000000
LOG_EVERY_STEPS=1000000
LOG_EVERY_EPS=100
BENCH_N=1000
STAGGER_SEC=2

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_gpus) NUM_GPUS="$2"; shift 2 ;;
    --instances_per_gpu) INSTANCES_PER_GPU="$2"; shift 2 ;;
    --target_total) TARGET_TOTAL="$2"; shift 2 ;;
    --out_root) OUT_ROOT="$2"; LOG_DIR="${OUT_ROOT}/logs"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --buffer) BUFFER="$2"; shift 2 ;;
    --stagger_sec) STAGGER_SEC="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ ! -f "${CKPT}" ]]; then
  echo "[100m] ERROR: validated bounded-critic checkpoint missing: ${CKPT}"
  echo "[100m] Run warmup_vlacrit.sh and fixed-seed G=0 vs G-on validation first."
  exit 1
fi
if [[ ! -f "${BUFFER}" ]]; then
  echo "[100m] ERROR: matched VLA replay missing: ${BUFFER}"
  exit 1
fi

TOTAL_INSTANCES=$((NUM_GPUS * INSTANCES_PER_GPU))
mkdir -p "${OUT_ROOT}" "${LOG_DIR}" /tmp/molmoact2_cf_rollouts

echo "[100m] stopping prior CF serve/train/watchdog ..."
pkill -9 -f 'molmoact2_cf/server_watchdog.sh' 2>/dev/null || true
pkill -9 -f 'molmoact2_cf/serve.py' 2>/dev/null || true
pkill -9 -f 'molmoact2_cf/train_full.py' 2>/dev/null || true
pkill -9 -f 'molmoact2_cf/train_100m.py' 2>/dev/null || true
sleep 8
if pgrep -f 'molmoact2_cf/(serve|train_100m|train_full)\.py' >/dev/null; then
  echo "[100m] WARN: leftover processes after pkill; forcing again"
  pkill -9 -f 'molmoact2_cf/serve.py' 2>/dev/null || true
  pkill -9 -f 'molmoact2_cf/train_100m.py' 2>/dev/null || true
  pkill -9 -f 'molmoact2_cf/train_full.py' 2>/dev/null || true
  sleep 3
fi

STEPS_PER=$(( TARGET_TOTAL / TOTAL_INSTANCES ))
SHARD_BASE=$(( BENCH_N / TOTAL_INSTANCES ))

echo "[100m] GPUs=${NUM_GPUS} × ${INSTANCES_PER_GPU}/gpu = ${TOTAL_INSTANCES} instances"
echo "[100m] target_total=${TARGET_TOTAL} steps/inst=${STEPS_PER} shard_base=${SHARD_BASE}"
echo "[100m] ckpt=${CKPT}"
echo "[100m] out=${OUT_ROOT}"
echo "[100m] python=${PYTHON}"

# Watchdog owns all servers (restarts on death).
nohup bash "${ROOT}/server_watchdog.sh" \
  "${LOG_DIR}" "${NUM_GPUS}" "${BASE_PORT}" 30 "${INSTANCES_PER_GPU}" \
  > "${LOG_DIR}/watchdog.log" 2>&1 &
echo $! > "${LOG_DIR}/watchdog.pid"
echo "[100m] watchdog pid=$(cat "${LOG_DIR}/watchdog.pid")"

echo "[100m] waiting for ${TOTAL_INSTANCES} servers ..."
for ((g=0; g<TOTAL_INSTANCES; g++)); do
  port=$((BASE_PORT + g))
  for t in $(seq 1 240); do
    if curl -sf --max-time 3 "http://127.0.0.1:${port}/act" 2>/dev/null | grep -q '"status":"ok"'; then
      echo "[100m] server inst=${g} port=${port} READY"
      break
    fi
    if (( t == 240 )); then
      gpu=$((g / INSTANCES_PER_GPU))
      echo "[100m] ERROR: server inst=${g} port=${port} not ready"
      tail -40 "${LOG_DIR}/server_g${g}_gpu${gpu}.log" || true
      exit 1
    fi
    sleep 5
  done
done

START=0
for ((g=0; g<TOTAL_INSTANCES; g++)); do
  GPU=$((g / INSTANCES_PER_GPU))
  LOCAL=$((g % INSTANCES_PER_GPU))
  port=$((BASE_PORT + g))
  if (( g == TOTAL_INSTANCES - 1 )); then
    N=$(( BENCH_N - START ))
  else
    N="${SHARD_BASE}"
  fi
  shard_out="${OUT_ROOT}/shard_${g}"
  mkdir -p "${shard_out}"
  echo "[100m] train inst=${g} gpu=${GPU} local=${LOCAL} port=${port} start_ep=${START} shard_size=${N} target_steps=${STEPS_PER}"
  TRAIN_CMD=(
    "${PYTHON}" "${ROOT}/train_100m.py"
    --buffer "${BUFFER}"
    --out_dir "${shard_out}"
    --device cuda:0
    --server_host localhost
    --server_port "${port}"
    --start_episode "${START}"
    --shard_size "${N}"
    --target_env_steps "${STEPS_PER}"
    --log_every_steps "${LOG_EVERY_STEPS}"
    --log_every_episodes "${LOG_EVERY_EPS}"
    --tmp_rollout_dir /tmp/molmoact2_cf_rollouts
    --updates_per_episode 5
    --g_start_episodes 20
    --g_min_advantage 0.005
    --policy_delay 2
    --explore_residual_std 0.02
    --cql_coef 0.1
    --cql_n_actions 8
    --cql_action_radius 0.05
    --cql_far_scale 1.0
    --target_divergence 0.0025
    --lr_alpha 0.0001
    --seed "${g}"
    --cf_ckpt "${CKPT}"
  )
  MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}" \
  MUJOCO_GL=egl \
  MUJOCO_EGL_DEVICE_ID="${GPU}" \
  CUDA_VISIBLE_DEVICES="${GPU}" \
    nohup "${TRAIN_CMD[@]}" \
      > "${LOG_DIR}/train_g${g}_gpu${GPU}.log" 2>&1 &
  echo $! > "${LOG_DIR}/train_g${g}.pid"
  START=$((START + N))
  sleep "${STAGGER_SEC}"
done

cat > "${OUT_ROOT}/README.md" <<EOF
# MolmoAct2 + CF 100M (VLA-feature critic) (${NUM_GPUS}x${INSTANCES_PER_GPU})

- Global target: **${TARGET_TOTAL}** valid sim env steps (${TOTAL_INSTANCES} x ${STEPS_PER})
- Critic/G state: x = concat(Projector(h_vla), proprio) (pragmatic RLT)
- Metrics every **${LOG_EVERY_STEPS}** steps -> shard_*/metrics.jsonl

Watch:
\`\`\`
grep -hE 'steps=|METRICS|critic_loss' ${LOG_DIR}/train_g*_gpu*.log | tail
\`\`\`
EOF

echo "[100m] all ${TOTAL_INSTANCES} train shards launched (nohup)"
echo "  logs: ${LOG_DIR}"
echo "  launcher exiting; trains+watchdog keep running"
exit 0
