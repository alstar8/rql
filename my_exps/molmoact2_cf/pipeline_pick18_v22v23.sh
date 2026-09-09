#!/usr/bin/env bash
# Pick-18 paper method from V22_V23_METHODS.md (Pick instantiation).
#
# flow_rlt compose + live TD + AE finetune (cf_ae), β=100, gate_step=0.
# Reuses scene AEs and 100-traj VLA buffers from runs/pick18/. Fresh AC
# pretrain (8000) → 10 probe → 300 online → held-out eval48.
#
# All 18 tasks start together: 8 GPUs × 3 slots (6 idle workers).
#
#   GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=3 bash pipeline_pick18_v22v23.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SOURCE_DIR="${SOURCE_DIR:-${ROOT}/runs/pick18}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v22v23}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_v22v23_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
EPISODES="${EPISODES:-300}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
BETA="${BETA:-100}"
WARMUP=0

OBJECTS=(remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box)
read -r -a TASKS <<< "${TASKS:-desk_mug kettle ${OBJECTS[*]}}"

mkdir -p "${RUN_DIR}/pids" "${RUN_DIR}/queue" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
if [[ "${SKIP_ENQUEUE:-0}" != "1" ]]; then
  echo $$ > "${RUN_DIR}/pids/pipeline.pid"
else
  echo $$ > "${RUN_DIR}/pids/extra_workers.pid"
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[pick18-v22v23 $(date -u +%H:%M:%S)] $*"; }

task_meta() {
  case "$1" in
    desk_mug)
      SCENE=desk_mug HORIZON=500
      TOKENS="${ROOT}/runs/beta1_1gpu/desk_mug/tokens/desk_mug"
      ;;
    kettle)
      SCENE=kettle HORIZON=400
      TOKENS="${ROOT}/runs/beta1_jitter_ac/kettle/tokens/kettle"
      ;;
    *)
      SCENE="${1}" HORIZON=500
      TOKENS="${ROOT}/runs/pick_objects/${1}/tokens/${1}"
      ;;
  esac
}

seed_task() {
  local task="$1"
  task_meta "${task}"
  local src="${SOURCE_DIR}/${task}"
  local dest="${RUN_DIR}/${task}"
  local ae_src="${src}/ae/ae_${SCENE}.pt"
  local buf_src="${src}/vla_buffer.npz"
  if [[ ! -f "${ae_src}" ]]; then
    log "ERROR: ${task} missing source AE ${ae_src}"
    return 1
  fi
  if [[ ! -f "${buf_src}" ]]; then
    log "ERROR: ${task} missing VLA buffer ${buf_src}"
    return 1
  fi
  mkdir -p "${dest}/ae" "${dest}/ac_pretrain" "${dest}/rl" "${dest}/eval" "${dest}/pids"
  if [[ ! -e "${dest}/ae/ae_${SCENE}.pt" ]]; then
    cp -n "${ae_src}" "${dest}/ae/ae_${SCENE}.pt"
  fi
  if [[ ! -e "${dest}/vla_buffer.npz" ]]; then
    cp -n "${buf_src}" "${dest}/vla_buffer.npz"
  fi
}

run_method() {
  local task="$1" gpu="$2" slot="$3"
  task_meta "${task}"
  local port=$((9000 + gpu * 10 + slot))
  local dest="${RUN_DIR}/${task}"
  local ae="${dest}/ae/ae_${SCENE}.pt"
  local shared_buf="${dest}/vla_buffer.npz"
  local pre_tag="${task}_cf_ae_pretrain"
  local on_tag="${task}_cf_ae_s0"
  local pre_dir="${dest}/ac_pretrain"
  local on_dir="${dest}/rl"
  mkdir -p "${pre_dir}" "${on_dir}" "${dest}/eval" "${dest}/pids"

  if [[ -f "${dest}/eval/${on_tag}_actor/result.json" ]]; then
    log "${task}/cf_ae: already evaluated, skip"
    return 0
  fi

  seed_task "${task}" || return 1

  local agent="${pre_dir}/${pre_tag}/agent.pt"
  local buf="${pre_dir}/${pre_tag}/buffer.npz"
  local on_agent="${on_dir}/${on_tag}/agent.pt"

  if [[ -f "${agent}" ]]; then
    log "${task}/cf_ae: pretrained agent exists, skip pretrain"
  else
    log "${task}/cf_ae: pretrain gpu=${gpu} port=${port} beta=${BETA} compose=true"
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_pretrain_ac.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --episodes 100 --offline-steps "${OFFLINE_STEPS}" \
        --gpu "${gpu}" --port "${port}" --tag "${pre_tag}" \
        --set "algorithm=flow_rlt" \
        --set "token_ae=${ae}" \
        --set "beta=${BETA}" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "flow_actor_coef=0" \
        --set "flow_compose=true" \
        --set "flow_freeze_critic_online=false" \
        --set "horizon=${HORIZON}" \
        --set "episode_pool=0-47" \
        --set "out_dir=${pre_dir}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=false" \
        --set "vla_traj=${shared_buf}" \
        > "${LOCAL_LOG}/${pre_tag}.log" 2>&1
    if [[ ! -f "${agent}" ]]; then
      log "ERROR: ${task}/cf_ae missing pretrained agent"
      return 1
    fi
  fi

  if [[ -f "${on_agent}" ]]; then
    log "${task}/cf_ae: online agent exists, skip train"
  else
    log "${task}/cf_ae: online gpu=${gpu} port=${port}"
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --gpu "${gpu}" --port "${port}" \
        --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
        --tag "${on_tag}" --init-actor "${agent}" \
        --set "algorithm=flow_rlt" \
        --set "token_ae=${ae}" \
        --set "beta=${BETA}" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "flow_actor_coef=1" \
        --set "flow_compose=true" \
        --set "flow_freeze_critic_online=false" \
        --set "horizon=${HORIZON}" \
        --set "seed=0" \
        --set "init_buffer=${buf}" \
        --set "episode_pool=0-11" \
        --set "out_dir=${on_dir}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=true" \
        --set "store_decision_tokens=false" \
        --set "token_replay=${TOKENS}/*.npz" \
        > "${LOCAL_LOG}/${on_tag}.log" 2>&1
    if [[ ! -f "${on_agent}" ]]; then
      log "ERROR: ${task}/cf_ae missing online agent"
      return 1
    fi
  fi

  log "${task}/cf_ae: eval gpu=${gpu} port=${port}"
  cd "${CODE}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene "${SCENE}" --episodes 16 --gpu "${gpu}" --port "${port}" \
      --actor "${on_agent}" --token-ae "${ae}" \
      --tag "${on_tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "out_dir=${dest}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${on_tag}_actor.log" 2>&1
  log "${task}/cf_ae: done"
}

worker() {
  local gpu="$1" slot="$2"
  local q="${RUN_DIR}/queue/jobs.txt"
  local lock="${RUN_DIR}/queue/jobs.lock"
  while true; do
    local job=""
    exec 8>"${lock}"
    flock 8
    if [[ -s "${q}" ]]; then
      job="$(head -n 1 "${q}")"
      tail -n +2 "${q}" > "${q}.tmp"
      mv "${q}.tmp" "${q}"
    fi
    flock -u 8
    exec 8>&-
    if [[ -z "${job}" ]]; then
      return 0
    fi
    log "gpu=${gpu} slot=${slot} start ${job}"
    if ! run_method "${job}" "${gpu}" "${slot}"; then
      log "ERROR: ${job} failed on gpu=${gpu}"
    fi
  done
}

Q="${RUN_DIR}/queue/jobs.txt"
if [[ "${SKIP_ENQUEUE:-0}" != "1" ]]; then
  for task in "${TASKS[@]}"; do
    seed_task "${task}" || exit 1
  done
  : > "${Q}"
  for task in "${TASKS[@]}"; do
    echo "${task}"
  done >> "${Q}"
  log "queued $(grep -c . "${Q}") V22+V23 cf_ae jobs on ${#GPUS[@]} GPUs × ${SLOTS_PER_GPU} slots β=${BETA}"
else
  log "SKIP_ENQUEUE: extra workers GPUs=${GPUS[*]} slots=${SLOTS_PER_GPU} queue=$(grep -c . "${Q}" || echo 0)"
fi

cd "${CODE}"
pids=()
for gpu in "${GPUS[@]}"; do
  for ((slot = 0; slot < SLOTS_PER_GPU; slot++)); do
    worker "${gpu}" "${slot}" &
    pids+=($!)
  done
done
log "workers ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  wait "${p}" || fail=1
done
log "pick18 V22+V23 sweep finished (fail=${fail})"
exit "${fail}"
