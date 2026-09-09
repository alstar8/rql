#!/usr/bin/env bash
# Kettle PPO: same AE + 100-traj VLA buffer as V21, on-policy PPO instead of Eq. 5.
#
#   bash pipeline_kettle_ppo.sh
#   GPU0=2 PORT0=9460 bash pipeline_kettle_ppo.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/ppo/kettle}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/ppo_kettle_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
GPU0="${GPU0:-0}"
PORT0="${PORT0:-9460}"

V21_KETTLE="${V21_KETTLE:-${ROOT}/runs/beta1_1gpu/kettle}"
SRC_AE="${SRC_AE:-${V21_KETTLE}/ae/ae_kettle.pt}"
VLA_TRAJ="${VLA_TRAJ:-${V21_KETTLE}/ac_pretrain/kettle_ac_pretrain/buffer.npz}"

HORIZON=400
EPISODES="${EPISODES:-300}"
WARMUP=0
PRETRAIN_EPS="${PRETRAIN_EPS:-100}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
PPO_CLIP="${PPO_CLIP:-0.2}"
PPO_LAMBDA="${PPO_LAMBDA:-0.95}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
PPO_HORIZON="${PPO_HORIZON:-256}"
PPO_MINIBATCH="${PPO_MINIBATCH:-64}"
AE_PATH="${RUN_DIR}/ae/ae_kettle.pt"
PRETRAIN_TAG="kettle_ppo_pretrain"
ON_TAG="${ON_TAG:-kettle_ppo_s0}"

if [[ ! -f "${SRC_AE}" ]]; then
  echo "[ppo] missing V21 kettle AE ${SRC_AE}" >&2
  exit 1
fi
if [[ ! -f "${VLA_TRAJ}" ]]; then
  echo "[ppo] missing V21 kettle replay ${VLA_TRAJ}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/ae" \
  "${RUN_DIR}/rl" "${RUN_DIR}/eval" "${RUN_DIR}/ac_pretrain"
cp -n "${SRC_AE}" "${AE_PATH}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[ppo-kettle $(date -u +%H:%M:%S)] $*"; }

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

AGENT="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/agent.pt"
BUFFER="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/buffer.npz"

if [[ "${SKIP_PRETRAIN:-0}" == "1" ]]; then
  if [[ ! -f "${AGENT}" ]]; then
    log "ERROR: SKIP_PRETRAIN=1 but missing ${AGENT}"
    exit 1
  fi
  log "skip pretrain, reuse ${AGENT}"
else
  log "pretrain ppo gpu=${GPU0} replay=${VLA_TRAJ} ppo_clip=${PPO_CLIP}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_pretrain_ac.py \
      --scene kettle --encoder kettle \
      --episodes "${PRETRAIN_EPS}" --offline-steps "${OFFLINE_STEPS}" \
      --gpu "${GPU0}" --port "${PORT0}" --tag "${PRETRAIN_TAG}" \
      --set "algorithm=ppo" \
      --set "init_random_ae=false" \
      --set "token_ae=${AE_PATH}" \
      --set "ppo_clip=${PPO_CLIP}" \
      --set "ppo_gae_lambda=${PPO_LAMBDA}" \
      --set "ppo_epochs=${PPO_EPOCHS}" \
      --set "ppo_horizon=${PPO_HORIZON}" \
      --set "ppo_minibatch=${PPO_MINIBATCH}" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "horizon=${HORIZON}" \
      --set "episode_pool=0-47" \
      --set "out_dir=${RUN_DIR}/ac_pretrain" \
      --set "checkpoint=${CKPT}" \
      --set "train_token_offline=false" \
      --set "train_token_online=false" \
      --set "ae_finetune=false" \
      --set "vla_traj=${VLA_TRAJ}" \
      > "${LOCAL_LOG}/${PRETRAIN_TAG}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid"
  wait_pidfile "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid" "ac_pretrain"
  if [[ ! -f "${AGENT}" ]]; then
    log "ERROR: missing pretrained agent ${AGENT}"
    exit 1
  fi
fi

ON_AGENT="${RUN_DIR}/rl/${ON_TAG}/agent.pt"
log "online ppo gpu=${GPU0} ${ON_TAG}"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_train.py \
    --scene kettle --encoder kettle \
    --gpu "${GPU0}" --port "${PORT0}" \
    --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
    --tag "${ON_TAG}" --init-actor "${AGENT}" \
    --set "algorithm=ppo" \
    --set "init_random_ae=false" \
    --set "token_ae=${AE_PATH}" \
    --set "ppo_clip=${PPO_CLIP}" \
    --set "ppo_gae_lambda=${PPO_LAMBDA}" \
    --set "ppo_epochs=${PPO_EPOCHS}" \
    --set "ppo_horizon=${PPO_HORIZON}" \
    --set "ppo_minibatch=${PPO_MINIBATCH}" \
    --set "gate_step=0" \
    --set "gate_frac=0" \
    --set "horizon=${HORIZON}" \
    --set "seed=0" \
    --set "init_buffer=${BUFFER}" \
    --set "episode_pool=0-11" \
    --set "out_dir=${RUN_DIR}/rl" \
    --set "checkpoint=${CKPT}" \
    --set "train_token_offline=false" \
    --set "train_token_online=false" \
    --set "ae_finetune=false" \
    > "${LOCAL_LOG}/${ON_TAG}.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/${ON_TAG}.pid"
wait_pidfile "${RUN_DIR}/pids/${ON_TAG}.pid" "online"
if [[ ! -f "${ON_AGENT}" ]]; then
  log "ERROR: missing online agent ${ON_AGENT}"
  exit 1
fi

log "eval ppo gpu=${GPU0}"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_eval.py \
    --scene kettle --episodes "${EVAL_EPISODES}" --gpu "${GPU0}" --port "${PORT0}" \
    --actor "${ON_AGENT}" --token-ae "${AE_PATH}" \
    --tag "${ON_TAG}_actor" \
    --set "horizon=${HORIZON}" \
    --set "gate_step=0" \
    --set "gate_frac=0" \
    --set "out_dir=${RUN_DIR}/eval" \
    --set "checkpoint=${CKPT}" \
    > "${LOCAL_LOG}/${ON_TAG}_actor.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/${ON_TAG}_actor.pid"
wait_pidfile "${RUN_DIR}/pids/${ON_TAG}_actor.pid" "eval"
log "kettle PPO done"
