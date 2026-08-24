#!/usr/bin/env bash
# V22 kettle GPU 2: ConsensusFlow, no AE recon / no L_ro.
# Offline trajectories: V21 kettle GPU 1 AC-pretrain buffer (skip VLA collect).
# Encoder is the V21 kettle AE so stored z matches; CF trains actor/critic/guidance.
#
#   bash pipeline_kettle_cf_no_ae.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/v22_cf/kettle_no_ae}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/v22_cf_kettle_no_ae_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
GPU="${GPU:-2}"
PORT="${PORT:-8520}"

V21_KETTLE="${V21_KETTLE:-${ROOT}/runs/beta1_1gpu/kettle}"
SRC_AE="${SRC_AE:-${V21_KETTLE}/ae/ae_kettle.pt}"
VLA_TRAJ="${VLA_TRAJ:-${V21_KETTLE}/ac_pretrain/kettle_ac_pretrain/buffer.npz}"

HORIZON=400
EPISODES=300
WARMUP=0
PRETRAIN_EPS="${PRETRAIN_EPS:-100}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
AE_PATH="${RUN_DIR}/ae/ae_kettle.pt"
PRETRAIN_TAG="kettle_cf_no_ae_pretrain"
S0_TAG="kettle_cf_no_ae_s0"
S1_TAG="kettle_cf_no_ae_s1"

if [[ ! -f "${SRC_AE}" ]]; then
  echo "[kettle-cf-noae] missing V21 kettle AE ${SRC_AE}" >&2
  exit 1
fi
if [[ ! -f "${VLA_TRAJ}" ]]; then
  echo "[kettle-cf-noae] missing V21 kettle replay ${VLA_TRAJ}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/ae" "${RUN_DIR}/rl" \
  "${RUN_DIR}/eval" "${RUN_DIR}/ac_pretrain"
cp -n "${SRC_AE}" "${AE_PATH}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[kettle-cf-noae $(date -u +%H:%M:%S)] $*"; }

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

CF_SET=(
  --set "algorithm=consensusflow"
  --set "init_random_ae=false"
  --set "token_ae=${AE_PATH}"
  --set "horizon=${HORIZON}"
  --set "episode_pool=0-47"
  --set "out_dir=${RUN_DIR}/ac_pretrain"
  --set "checkpoint=${CKPT}"
  --set "train_token_offline=true"
  --set "train_token_online=false"
  --set "ae_finetune=false"
  --set "store_decision_tokens=false"
  --set "vla_traj=${VLA_TRAJ}"
)

if [[ "${CONTINUE_FROM:-}" != "eval_s0" ]]; then
  log "pretrain CF gpu=${GPU} replay=${VLA_TRAJ} (skip collect, no L_ro)"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_pretrain_ac.py \
      --scene kettle --encoder kettle \
      --episodes "${PRETRAIN_EPS}" --offline-steps "${OFFLINE_STEPS}" \
      --gpu "${GPU}" --port "${PORT}" --tag "${PRETRAIN_TAG}" \
      "${CF_SET[@]}" \
      > "${LOCAL_LOG}/${PRETRAIN_TAG}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid"
  wait_pidfile "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid" "ac_pretrain"
fi

AGENT="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/agent.pt"
BUFFER="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/buffer.npz"
if [[ ! -f "${AGENT}" ]]; then
  log "ERROR: missing pretrained actor ${AGENT}"
  exit 1
fi

start_rl() {
  local seed="$1" tag="$2"
  log "online CF gpu=${GPU} token frozen ${tag}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene kettle --encoder kettle \
      --gpu "${GPU}" --port "${PORT}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${tag}" --init-actor "${AGENT}" \
      --set "algorithm=consensusflow" \
      --set "init_random_ae=false" \
      --set "token_ae=${AE_PATH}" \
      --set "horizon=${HORIZON}" \
      --set "seed=${seed}" \
      --set "init_buffer=${BUFFER}" \
      --set "episode_pool=0-11" \
      --set "out_dir=${RUN_DIR}/rl" \
      --set "checkpoint=${CKPT}" \
      --set "train_token_offline=true" \
      --set "train_token_online=false" \
      --set "ae_finetune=false" \
      --set "store_decision_tokens=false" \
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
      --scene kettle --episodes 16 --gpu "${GPU}" --port "${PORT}" \
      --actor "${agent}" --token-ae "${AE_PATH}" \
      --tag "${tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${tag}_actor.log" 2>&1 < /dev/null
}

if [[ "${CONTINUE_FROM:-}" == "eval_s0" ]]; then
  AGENT="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/agent.pt"
  BUFFER="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/buffer.npz"
  if [[ ! -f "${AGENT}" ]]; then
    log "ERROR: missing pretrained actor ${AGENT}"
    exit 1
  fi
  eval_actor "${S0_TAG}"
  start_rl 1 "${S1_TAG}"
  eval_actor "${S1_TAG}"
  log "kettle CF no-AE done"
  exit 0
fi

start_rl 0 "${S0_TAG}"
eval_actor "${S0_TAG}"
start_rl 1 "${S1_TAG}"
eval_actor "${S1_TAG}"
log "kettle CF no-AE done"
