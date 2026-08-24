#!/usr/bin/env bash
# Corrected V22 (flow_rlt) mug test on GPU 7.
#
# Frozen: pi0.5, the RLT token encoder (V21 mug AE), and the RLT critic from the
# finished V21 jitter_ac mug s1 run. Trained: the flow actor only, with
# flow-matching BC to the VLA chunk plus the RLT Eq. 5 objective against the
# frozen critic (-Q + beta*||a - a_ref||^2, beta=100). The base AE finetunes
# inline (online) via reconstruction on the mug token corpus; the encoder that
# builds z_rl for the buffer stays the frozen on-disk copy.
#
# No new collect: the AC pretrain replays the V21 jitter_ac VLA buffer.
#
#   bash pipeline_mug_flow_rlt.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/v22_cf/mug_flow_rlt}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/v22_cf_mug_flow_rlt_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
GPU="${GPU:-7}"
PORT="${PORT:-8580}"

V21_AE="${ROOT}/runs/beta1_from_scratch/desk_mug/ae/ae_desk_mug.pt"
RLT_CRITIC="${ROOT}/runs/beta1_jitter_ac/desk_mug/rl/desk_mug_ae-desk_mug_beta1_s1/agent.pt"
V21_BUFFER="${ROOT}/runs/beta1_jitter_ac/desk_mug/ac_pretrain/desk_mug_ac_pretrain/buffer.npz"
TOKEN_CORPUS="${ROOT}/runs/beta1_1gpu/desk_mug/tokens/desk_mug"

HORIZON=500
EPISODES=300
WARMUP=0
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
AE_PATH="${RUN_DIR}/ae/ae_desk_mug.pt"
PRETRAIN_TAG="desk_mug_flow_rlt_pretrain"
S0_TAG="desk_mug_flow_rlt_s0"
S1_TAG="desk_mug_flow_rlt_s1"

for f in "${V21_AE}" "${RLT_CRITIC}" "${V21_BUFFER}"; do
  [[ -f "${f}" ]] || { echo "missing ${f}"; exit 1; }
done
mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/ae" \
  "${RUN_DIR}/rl" "${RUN_DIR}/eval" "${RUN_DIR}/ac_pretrain"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"
# The run reads its own copy so a later AE run cannot move the frozen encoder
# out from under the frozen critic.
cp -n "${V21_AE}" "${AE_PATH}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[mug-flow-rlt $(date -u +%H:%M:%S)] $*"; }

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

log "pretrain flow_rlt gpu=${GPU}: replay V21 buffer, flow BC + frozen-critic anchor"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_pretrain_ac.py \
    --scene desk_mug --encoder desk_mug \
    --episodes 100 --offline-steps "${OFFLINE_STEPS}" \
    --gpu "${GPU}" --port "${PORT}" --tag "${PRETRAIN_TAG}" \
    --set "algorithm=flow_rlt" \
    --set "token_ae=${AE_PATH}" \
    --set "rlt_critic=${RLT_CRITIC}" \
    --set "beta=100" \
    --set "flow_actor_coef=0" \
    --set "horizon=${HORIZON}" \
    --set "episode_pool=0-47" \
    --set "out_dir=${RUN_DIR}/ac_pretrain" \
    --set "checkpoint=${CKPT}" \
    --set "vla_traj=${V21_BUFFER}" \
    --set "train_token_offline=false" \
    --set "train_token_online=false" \
    --set "ae_finetune=false" \
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
  log "online flow_rlt gpu=${GPU}: actor only + base AE recon inline ${tag}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene desk_mug --encoder desk_mug \
      --gpu "${GPU}" --port "${PORT}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${tag}" --init-actor "${AGENT}" \
      --set "algorithm=flow_rlt" \
      --set "token_ae=${AE_PATH}" \
      --set "rlt_critic=${RLT_CRITIC}" \
      --set "beta=100" \
      --set "horizon=${HORIZON}" \
      --set "seed=${seed}" \
      --set "init_buffer=${BUFFER}" \
      --set "episode_pool=0-11" \
      --set "out_dir=${RUN_DIR}/rl" \
      --set "checkpoint=${CKPT}" \
      --set "train_token_offline=false" \
      --set "train_token_online=false" \
      --set "ae_finetune=true" \
      --set "store_decision_tokens=false" \
      --set "token_replay=${TOKEN_CORPUS}/*.npz" \
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
log "mug flow_rlt done"
