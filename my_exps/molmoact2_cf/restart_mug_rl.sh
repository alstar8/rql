#!/usr/bin/env bash
# Restart desk_mug RL Token beta=1 (keep collected corpus + AE). Mug benches
# already share one pose; this exists because an interrupted warmup must not
# be evaluated.
#
#   bash restart_mug_rl.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/beta1_from_scratch/desk_mug}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/beta1_desk_mug_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
AE_PATH="${RUN_DIR}/ae/ae_desk_mug.pt"

HORIZON=500
BETA=1.0
EPISODES=300
WARMUP=40
GPU_RL0=0
GPU_RL1=1
GPU_EVAL=2
GPU_ACTOR1=3
PORT_RL0=8510
PORT_RL1=8511
PORT_ACTOR=8512

if [[ ! -f "${AE_PATH}" ]]; then
  echo "[mug-rl] missing AE ${AE_PATH}" >&2
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

log() { echo "[mug-rl $(date -u +%H:%M:%S)] $*"; }

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

S0_TAG="desk_mug_ae-desk_mug_beta1_s0"
S1_TAG="desk_mug_ae-desk_mug_beta1_s1"

start_rl() {
  local gpu="$1" port="$2" seed="$3" tag="$4"
  log "RL Token beta=${BETA} ${tag} gpu=${gpu}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene desk_mug --encoder desk_mug \
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
      --scene desk_mug --episodes 16 --gpu "${gpu}" --port "${port}" \
      --actor "${agent}" --token-ae "${AE_PATH}" \
      --tag "${tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${tag}_actor.log" 2>&1 < /dev/null
}

eval_actor "${S0_TAG}" "${GPU_EVAL}" "${PORT_ACTOR}"
eval_actor "${S1_TAG}" "${GPU_ACTOR1}" "$((PORT_ACTOR + 1))"
log "mug RL pipeline done"
