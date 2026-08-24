#!/usr/bin/env bash
# Frozen-pi0.5 collect: 100 train trajectories per Pick object scene.
# Same recipe as kettle's jittered collect (tokens for a later AE, 100 traj cap).
# Mug and kettle already have this; this covers the other 16 categories.
#
#   GPUS="0 2 5 7" bash pipeline_pick_object_collect.sh
#   OBJECTS="spatula spoon" GPUS="0 2" bash pipeline_pick_object_collect.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick_objects}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick_object_collect_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 2 5 7}"
read -r -a OBJECTS <<< "${OBJECTS:-remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box}"

HORIZON=500
TARGET=100000
MAX_EPISODES="${MAX_EPISODES:-100}"

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[pick-collect $(date -u +%H:%M:%S)] $*"; }

start_collect() {
  local obj="$1" gpu="$2"
  local port=$((8700 + gpu))
  local tag="collect_${obj}"
  local dest="${RUN_DIR}/${obj}"
  mkdir -p "${dest}/tokens" "${dest}/collect" "${dest}/pids"
  if [[ -f "${dest}/collect/${tag}/collect.json" ]]; then
    log "skip ${obj}: collect.json already exists"
    return 0
  fi
  log "collect ${obj} gpu=${gpu} port=${port} max_episodes=${MAX_EPISODES}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_collect.py \
      --scene "${obj}" --gpu "${gpu}" --port "${port}" --target "${TARGET}" \
      --max-episodes "${MAX_EPISODES}" \
      --corpus "${dest}/tokens" --out "${dest}/collect" \
      --set "tag=${tag}" \
      --set "horizon=${HORIZON}" \
      --set "checkpoint=${CKPT}" \
      --set "save_video=False" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
  echo $! > "${dest}/pids/${tag}.pid"
}

# Fill free GPUs from the queue; when one collect exits, start the next.
queue=("${OBJECTS[@]}")
declare -A gpu_pid=()
for gpu in "${GPUS[@]}"; do
  gpu_pid["${gpu}"]=
done

still_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

still_any=1
while ((${#queue[@]})) || ((still_any)); do
  still_any=0
  for gpu in "${GPUS[@]}"; do
    pid="${gpu_pid[${gpu}]:-}"
    if still_running "${pid}"; then
      still_any=1
      continue
    fi
    gpu_pid["${gpu}"]=
    if ((${#queue[@]})); then
      obj="${queue[0]}"
      queue=("${queue[@]:1}")
      start_collect "${obj}" "${gpu}"
      gpu_pid["${gpu}"]="$(cat "${RUN_DIR}/pids/collect_${obj}.pid" 2>/dev/null || true)"
      if still_running "${gpu_pid[${gpu}]:-}"; then
        still_any=1
      fi
    fi
  done
  if ((still_any == 0)) && ((${#queue[@]} == 0)); then
    break
  fi
  sleep 30
done
log "all collects launched and finished"
