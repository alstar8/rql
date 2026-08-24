#!/usr/bin/env bash
# Launch remaining Pick-object collects packed 2-per-GPU.
# Existing remote/ladle/tissue/spoon jobs stay on GPUs 0/2/5/7; this fills
# the free GPUs and adds a second job on the occupied ones.
# Unique HTTP ports (88xx) so two pi0.5 servers can share a GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick_objects}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick_object_collect_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

HORIZON=500
TARGET=100000
MAX_EPISODES="${MAX_EPISODES:-100}"

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[pick-pack $(date -u +%H:%M:%S)] $*"; }

start_collect() {
  local obj="$1" gpu="$2" port="$3"
  local tag="collect_${obj}"
  local dest="${RUN_DIR}/${obj}"
  mkdir -p "${dest}/tokens" "${dest}/collect" "${dest}/pids"
  if [[ -f "${dest}/collect/${tag}/collect.json" ]]; then
    log "skip ${obj}: collect.json exists"
    return 0
  fi
  if [[ -f "${RUN_DIR}/pids/${tag}.pid" ]] && kill -0 "$(cat "${RUN_DIR}/pids/${tag}.pid")" 2>/dev/null; then
    log "skip ${obj}: already running pid=$(cat "${RUN_DIR}/pids/${tag}.pid")"
    return 0
  fi
  log "collect ${obj} gpu=${gpu} port=${port}"
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

# Remaining 12 objects, packed 2 per GPU (occupied GPUs get a second job).
#   GPU 1: spatula, pot
#   GPU 3: soap_dispenser, spray_bottle
#   GPU 4: cup, shaker
#   GPU 6: fork, bottle
#   GPU 0: fruit   (alongside remote)
#   GPU 2: bowl    (alongside ladle)
#   GPU 5: knife   (alongside tissue)
#   GPU 7: box     (alongside spoon)
start_collect spatula        1 8801
start_collect pot            1 8811
start_collect soap_dispenser 3 8803
start_collect spray_bottle   3 8813
start_collect cup            4 8804
start_collect shaker         4 8814
start_collect fork           6 8806
start_collect bottle         6 8816
start_collect fruit          0 8800
start_collect bowl           2 8802
start_collect knife          5 8805
start_collect box            7 8807

log "packed 12 remaining collects (2 jobs / GPU across 8 GPUs)"
ls "${RUN_DIR}/pids"/collect_*.pid | while read -r p; do
  pid=$(cat "$p")
  name=$(basename "$p" .pid)
  if kill -0 "$pid" 2>/dev/null; then
    echo "  ${name} pid=${pid} running"
  else
    echo "  ${name} pid=${pid} dead"
  fi
done
