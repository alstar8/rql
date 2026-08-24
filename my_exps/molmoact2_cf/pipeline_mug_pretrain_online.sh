#!/usr/bin/env bash
# Mug: keep existing AE. Pretrain actor-critic on 100 frozen-VLA train
# trajectories, then online RL from episode 0 (no 40-ep warmup).
#
#   bash pipeline_mug_pretrain_online.sh
# GPUs 0,1 RL; GPU 2 pretrain then actor eval; GPU 3 second actor eval.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
OLD_AE="${OLD_AE:-${ROOT}/runs/beta1_from_scratch/desk_mug/ae/ae_desk_mug.pt}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/beta1_jitter_ac/desk_mug}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/beta1_jitter_ac_desk_mug_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

HORIZON=500
BETA=1.0
EPISODES=300
WARMUP=0
PRETRAIN_EPS="${PRETRAIN_EPS:-100}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
AE_PATH="${OLD_AE}"
PRETRAIN_TAG="desk_mug_ac_pretrain"
S0_TAG="desk_mug_ae-desk_mug_beta1_s0"
S1_TAG="desk_mug_ae-desk_mug_beta1_s1"

if [[ ! -f "${AE_PATH}" ]]; then
  echo "[mug] missing AE ${AE_PATH}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/rl" "${RUN_DIR}/eval" "${RUN_DIR}/ac_pretrain"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[mug-ac $(date -u +%H:%M:%S)] $*"; }

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

log "pretrain AC gpu=2 episodes=${PRETRAIN_EPS} offline=${OFFLINE_STEPS}"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_pretrain_ac.py \
    --scene desk_mug --encoder desk_mug \
    --episodes "${PRETRAIN_EPS}" --offline-steps "${OFFLINE_STEPS}" \
    --gpu 2 --port 8512 --tag "${PRETRAIN_TAG}" \
    --set "horizon=${HORIZON}" \
    --set "beta=${BETA}" \
    --set "token_ae=${AE_PATH}" \
    --set "episode_pool=0-47" \
    --set "out_dir=${RUN_DIR}/ac_pretrain" \
    --set "checkpoint=${CKPT}" \
    > "${LOCAL_LOG}/${PRETRAIN_TAG}.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid"
wait_pidfile "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid" "ac_pretrain"

AGENT="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/agent.pt"
BUFFER="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/buffer.npz"
if [[ ! -f "${AGENT}" ]]; then
  log "ERROR: missing pretrained actor ${AGENT}"
  exit 1
fi

start_rl() {
  local gpu="$1" port="$2" seed="$3" tag="$4"
  log "online RL warmup=0 ${tag} gpu=${gpu}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene desk_mug --encoder desk_mug \
      --gpu "${gpu}" --port "${port}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${tag}" --init-actor "${AGENT}" \
      --set "horizon=${HORIZON}" \
      --set "beta=${BETA}" \
      --set "seed=${seed}" \
      --set "token_ae=${AE_PATH}" \
      --set "init_buffer=${BUFFER}" \
      --set "episode_pool=0-11" \
      --set "out_dir=${RUN_DIR}/rl" \
      --set "checkpoint=${CKPT}" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

start_rl 0 8510 0 "${S0_TAG}"
sleep 20
start_rl 1 8511 1 "${S1_TAG}"

eval_actor() {
  local tag="$1" gpu="$2" port="$3"
  wait_pidfile "${RUN_DIR}/pids/${tag}.pid" "${tag}"
  local summary="${RUN_DIR}/rl/${tag}/summary.json"
  local agent="${RUN_DIR}/rl/${tag}/agent.pt"
  if [[ ! -f "${agent}" ]]; then
    log "ERROR: ${tag} missing agent"
    return 1
  fi
  if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('actor_episodes') is not None else 1)" "${summary}"; then
    log "ERROR: ${tag} incomplete summary"
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

eval_actor "${S0_TAG}" 2 8512
eval_actor "${S1_TAG}" 3 8513
log "mug pretrain+online done"
