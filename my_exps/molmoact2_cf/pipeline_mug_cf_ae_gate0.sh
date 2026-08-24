#!/usr/bin/env bash
# Mug cf_ae ablation: RL from env step 0 (no frozen-pi0.5 prefix).
#
# Same method as compare_mug/cf_ae (flow compose V=v_base+G, TD critic, AE finetune,
# beta=100, shared buffer + AE + pretrained actor) except the actor drives from the
# first env step. The previous cf_ae arm used the scene catalog gate (56).
#
#   GPU=0 PORT=8620 bash pipeline_mug_cf_ae_gate0.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/compare_mug_gate0}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/compare_mug_gate0_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

GPU="${GPU:-0}"
PORT="${PORT:-8620}"
HORIZON=500
EPISODES="${EPISODES:-300}"
WARMUP=0
BETA="${BETA:-100}"
# 0 = RL from the first env step (the new default). Set GATE_STEP=56 to recover the
# catalog prefix; any multiple of chunk_size 8 is a VLA prefix of that many steps.
GATE_STEP="${GATE_STEP:-0}"

V21_AE="${ROOT}/runs/beta1_from_scratch/desk_mug/ae/ae_desk_mug.pt"
PRETRAIN="${ROOT}/runs/compare_mug/ac_pretrain/mug_cf_ae_pretrain"
TOKEN_CORPUS="${ROOT}/runs/beta1_1gpu/desk_mug/tokens/desk_mug"
AE_PATH="${RUN_DIR}/ae/ae_desk_mug.pt"
TAG="mug_cf_ae_gate${GATE_STEP}_s0"

for f in "${V21_AE}" "${PRETRAIN}/agent.pt" "${PRETRAIN}/buffer.npz"; do
  [[ -f "${f}" ]] || { echo "missing ${f}"; exit 1; }
done
[[ -d "${TOKEN_CORPUS}" ]] || { echo "missing token corpus ${TOKEN_CORPUS}"; exit 1; }

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/ae" "${RUN_DIR}/rl" "${RUN_DIR}/eval"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"
cp -n "${V21_AE}" "${AE_PATH}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[cf-ae-gate0 $(date -u +%H:%M:%S)] $*"; }

wait_pidfile() {
  local pidfile="$1" name="$2" pid
  pid="$(cat "${pidfile}")"
  log "wait ${name} pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
  wait "${pid}" 2>/dev/null || true
}

log "online cf_ae gpu=${GPU} gate_step=${GATE_STEP} (RL from this env step; 0 = whole episode)"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_train.py \
    --scene desk_mug --encoder desk_mug \
    --gpu "${GPU}" --port "${PORT}" \
    --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
    --tag "${TAG}" --init-actor "${PRETRAIN}/agent.pt" \
    --set "algorithm=flow_rlt" \
    --set "token_ae=${AE_PATH}" \
    --set "beta=${BETA}" \
    --set "flow_actor_coef=1" \
    --set "flow_freeze_critic_online=false" \
    --set "flow_compose=true" \
    --set "gate_step=${GATE_STEP}" \
    --set "horizon=${HORIZON}" \
    --set "seed=0" \
    --set "init_buffer=${PRETRAIN}/buffer.npz" \
    --set "episode_pool=0-11" \
    --set "out_dir=${RUN_DIR}/rl" \
    --set "checkpoint=${CKPT}" \
    --set "train_token_offline=false" \
    --set "train_token_online=false" \
    --set "ae_finetune=true" \
    --set "store_decision_tokens=false" \
    --set "token_replay=${TOKEN_CORPUS}/*.npz" \
    > "${LOCAL_LOG}/${TAG}.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/${TAG}.pid"
wait_pidfile "${RUN_DIR}/pids/${TAG}.pid" "${TAG}"

AGENT="${RUN_DIR}/rl/${TAG}/agent.pt"
[[ -f "${AGENT}" ]] || { log "ERROR: missing online agent"; exit 1; }

log "eval ${TAG} gpu=${GPU} gate_step=${GATE_STEP}"
setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_eval.py \
    --scene desk_mug --episodes 16 --gpu "${GPU}" --port "${PORT}" \
    --actor "${AGENT}" --token-ae "${AE_PATH}" \
    --tag "${TAG}_actor" \
    --set "horizon=${HORIZON}" \
    --set "gate_step=${GATE_STEP}" \
    --set "out_dir=${RUN_DIR}/eval" \
    --set "checkpoint=${CKPT}" \
    >> "${LOCAL_LOG}/${TAG}_actor.log" 2>&1 < /dev/null
log "done"
