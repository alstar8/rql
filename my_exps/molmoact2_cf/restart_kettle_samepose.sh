#!/usr/bin/env bash
# Restart kettle RL + frozen eval on the HARD pose (train_k00) for both
# train and eval. Keeps the collected corpus and ae_kettle.pt. Does not touch mug.
#
#   bash restart_kettle_samepose.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/beta1_from_scratch/kettle}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/beta1_kettle_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
AE_PATH="${RUN_DIR}/ae/ae_kettle.pt"

HORIZON=400
BETA=1.0
EPISODES=300
WARMUP=40
GPU_RL0=4
GPU_RL1=5
GPU_EVAL=6
GPU_ACTOR1=7
PORT_EVAL=8542
PORT_RL0=8550
PORT_RL1=8551
PORT_ACTOR=8552

if [[ ! -f "${AE_PATH}" ]]; then
  echo "[kettle-samepose] missing AE ${AE_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/rl" "${RUN_DIR}/eval"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[kettle-samepose $(date -u +%H:%M:%S)] $*"; }

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

S0_TAG="kettle_ae-kettle_beta1_s0"
S1_TAG="kettle_ae-kettle_beta1_s1"
VLA_TAG="kettle_vla_baseline"

log "frozen VLA eval on hard pose gpu=${GPU_EVAL}"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_eval.py \
    --scene kettle --episodes 16 --gpu "${GPU_EVAL}" --port "${PORT_EVAL}" \
    --tag "${VLA_TAG}" \
    --set "horizon=${HORIZON}" \
    --set "out_dir=${RUN_DIR}/eval" \
    --set "checkpoint=${CKPT}" \
    > "${LOCAL_LOG}/${VLA_TAG}.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/${VLA_TAG}.pid"

start_rl() {
  local gpu="$1" port="$2" seed="$3" tag="$4"
  log "RL Token beta=${BETA} ${tag} gpu=${gpu} hard pose"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene kettle --encoder kettle \
      --gpu "${gpu}" --port "${port}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${tag}" \
      --set "horizon=${HORIZON}" \
      --set "beta=${BETA}" \
      --set "seed=${seed}" \
      --set "token_ae=${AE_PATH}" \
      --set "episode_pool=0-11" \
      --set "out_dir=${RUN_DIR}/rl" \
      --set "checkpoint=${CKPT}" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

sleep 20
start_rl "${GPU_RL0}" "${PORT_RL0}" 0 "${S0_TAG}"
sleep 20
start_rl "${GPU_RL1}" "${PORT_RL1}" 1 "${S1_TAG}"

eval_actor() {
  local tag="$1" gpu="$2" port="$3"
  local train_pidf="${RUN_DIR}/pids/${tag}.pid"
  local summary="${RUN_DIR}/rl/${tag}/summary.json"
  local agent="${RUN_DIR}/rl/${tag}/agent.pt"
  wait_pidfile "${train_pidf}" "${tag}"
  if [[ ! -f "${summary}" || ! -f "${agent}" ]]; then
    log "ERROR: ${tag} finished without agent/summary"
    return 1
  fi
  if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('actor_episodes') is not None else 1)" "${summary}"; then
    log "ERROR: ${tag} summary is incomplete (train was killed); not evaluating"
    return 1
  fi
  log "eval actor ${tag} gpu=${gpu}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene kettle --episodes 16 --gpu "${gpu}" --port "${port}" \
      --actor "${agent}" --token-ae "${AE_PATH}" \
      --tag "${tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${tag}_actor.log" 2>&1 < /dev/null
}

wait_pidfile "${RUN_DIR}/pids/${VLA_TAG}.pid" "vla_baseline"
eval_actor "${S0_TAG}" "${GPU_EVAL}" "${PORT_ACTOR}"
eval_actor "${S1_TAG}" "${GPU_ACTOR1}" "$((PORT_ACTOR + 1))"
log "kettle same-pose pipeline done"
