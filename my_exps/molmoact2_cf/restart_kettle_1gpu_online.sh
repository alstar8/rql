#!/usr/bin/env bash
# Restart V21 kettle 1-GPU online RL after the spawn/SemLock collector crash.
# Keeps the finished AE + AC pretrain. Does not touch mug GPU 0.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/beta1_1gpu/kettle}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/beta1_1gpu_kettle_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
AE_PATH="${RUN_DIR}/ae/ae_kettle.pt"
AGENT="${RUN_DIR}/ac_pretrain/kettle_ac_pretrain/agent.pt"
BUFFER="${RUN_DIR}/ac_pretrain/kettle_ac_pretrain/buffer.npz"
GPU="${GPU:-1}"
PORT="${PORT:-8540}"
HORIZON=400
EPISODES=300
WARMUP=0
S0_TAG="kettle_ae-kettle_beta1_s0"

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/rl" "${RUN_DIR}/eval"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"
echo $$ > "${ROOT}/runs/beta1_1gpu/kettle_pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[kettle-1gpu $(date -u +%H:%M:%S)] $*"; }

wait_pidfile() {
  local pidfile="$1" name="$2"
  local pid
  pid="$(cat "${pidfile}")"
  log "wait ${name} pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
  wait "${pid}" 2>/dev/null || true
}

if [[ ! -f "${AGENT}" ]]; then
  log "ERROR: missing pretrained actor ${AGENT}"
  exit 1
fi

log "online RL gpu=${GPU} subprocess collectors (file IPC) ${S0_TAG}"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_train.py \
    --scene kettle --encoder kettle \
    --gpu "${GPU}" --port "${PORT}" \
    --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
    --tag "${S0_TAG}" --init-actor "${AGENT}" \
    --set "horizon=${HORIZON}" \
    --set "beta=1.0" \
    --set "seed=0" \
    --set "token_ae=${AE_PATH}" \
    --set "init_buffer=${BUFFER}" \
    --set "episode_pool=0-11" \
    --set "out_dir=${RUN_DIR}/rl" \
    --set "checkpoint=${CKPT}" \
    --set "probe_episodes=0" \
    > "${LOCAL_LOG}/${S0_TAG}.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/${S0_TAG}.pid"

wait_pidfile "${RUN_DIR}/pids/${S0_TAG}.pid" "${S0_TAG}"
summary="${RUN_DIR}/rl/${S0_TAG}/summary.json"
agent="${RUN_DIR}/rl/${S0_TAG}/agent.pt"
if [[ ! -f "${agent}" ]]; then
  log "ERROR: ${S0_TAG} missing agent"
  exit 1
fi
if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('actor_episodes') is not None else 1)" "${summary}"; then
  log "ERROR: ${S0_TAG} incomplete summary"
  exit 1
fi
log "eval actor ${S0_TAG} gpu=${GPU}"
setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_eval.py \
    --scene kettle --episodes 16 --gpu "${GPU}" --port "${PORT}" \
    --actor "${agent}" --token-ae "${AE_PATH}" \
    --tag "${S0_TAG}_actor" \
    --set "horizon=${HORIZON}" \
    --set "out_dir=${RUN_DIR}/eval" \
    --set "checkpoint=${CKPT}" \
    >> "${LOCAL_LOG}/${S0_TAG}_actor.log" 2>&1 < /dev/null
log "kettle 1gpu online+eval done"
