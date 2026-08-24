#!/usr/bin/env bash
# V22 mug GPU 6: ConsensusFlow + full AE finetune alongside the actor.
# Token/critic schedule matches GPU 5 (token RL offline only; critic both phases).
#
#   bash pipeline_mug_cf_ae_ft.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/v22_cf/mug_ae_ft}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/v22_cf_mug_ae_ft_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
GPU="${GPU:-6}"
PORT="${PORT:-8570}"

HORIZON=500
EPISODES=300
WARMUP=0
PRETRAIN_EPS="${PRETRAIN_EPS:-100}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
AE_PATH="${RUN_DIR}/ae/ae_desk_mug.pt"
TOKENS="${RUN_DIR}/tokens/desk_mug"
PRETRAIN_TAG="desk_mug_cf_ae_ft_pretrain"
S0_TAG="desk_mug_cf_ae_ft_s0"
S1_TAG="desk_mug_cf_ae_ft_s1"

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/ae" "${TOKENS}" \
  "${RUN_DIR}/rl" "${RUN_DIR}/eval" "${RUN_DIR}/ac_pretrain"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[mug-cf-aeft $(date -u +%H:%M:%S)] $*"; }

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

log "pretrain CF gpu=${GPU} (token+actor+critic + AE recon)"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_pretrain_ac.py \
    --scene desk_mug --encoder desk_mug \
    --episodes "${PRETRAIN_EPS}" --offline-steps "${OFFLINE_STEPS}" \
    --gpu "${GPU}" --port "${PORT}" --tag "${PRETRAIN_TAG}" \
    --set "algorithm=consensusflow" \
    --set "init_random_ae=true" \
    --set "token_ae=${AE_PATH}" \
    --set "horizon=${HORIZON}" \
    --set "episode_pool=0-47" \
    --set "out_dir=${RUN_DIR}/ac_pretrain" \
    --set "checkpoint=${CKPT}" \
    --set "train_token_offline=true" \
    --set "train_token_online=false" \
    --set "ae_finetune=true" \
    --set "store_decision_tokens=true" \
    --set "record_tokens=${TOKENS}" \
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
  local seed="$1" tag="$2"
  log "online CF gpu=${GPU} AE finetune alongside actor ${tag}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene desk_mug --encoder desk_mug \
      --gpu "${GPU}" --port "${PORT}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${tag}" --init-actor "${AGENT}" \
      --set "algorithm=consensusflow" \
      --set "init_random_ae=true" \
      --set "token_ae=${AE_PATH}" \
      --set "horizon=${HORIZON}" \
      --set "seed=${seed}" \
      --set "init_buffer=${BUFFER}" \
      --set "episode_pool=0-11" \
      --set "out_dir=${RUN_DIR}/rl" \
      --set "checkpoint=${CKPT}" \
      --set "train_token_offline=true" \
      --set "train_token_online=false" \
      --set "ae_finetune=true" \
      --set "store_decision_tokens=false" \
      --set "token_replay=${TOKENS}/*.npz" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

eval_actor() {
  local tag="$1"
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
  log "eval actor ${tag} gpu=${GPU}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene desk_mug --episodes 16 --gpu "${GPU}" --port "${PORT}" \
      --actor "${agent}" --token-ae "${AE_PATH}" \
      --tag "${tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${tag}_actor.log" 2>&1 < /dev/null
}

start_rl 0 "${S0_TAG}"
eval_actor "${S0_TAG}"
start_rl 1 "${S1_TAG}"
eval_actor "${S1_TAG}"
log "mug CF AE-finetune done"
