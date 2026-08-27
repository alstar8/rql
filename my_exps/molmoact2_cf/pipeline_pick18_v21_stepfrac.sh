#!/usr/bin/env bash
# Pick-18 V21 β=1 with a ~10% frozen-VLA prefix (gate_frac=0.1).
#
# Reuses AEs and V21 pretrained actors from runs/pick18_beta1/. Only online +
# held-out eval run with the prefix: 10% of the horizon snapped down to a
# chunk boundary (48 of 500, 40 of 400 on kettle). Pretrain stays gate=0
# (actor off). Label: V21 β=1 Step=10%.
#
#   GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 bash pipeline_pick18_v21_stepfrac.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SOURCE_DIR="${SOURCE_DIR:-${ROOT}/runs/pick18_beta1}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_beta1_step10}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_beta1_step10_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
EPISODES="${EPISODES:-300}"
BETA="${BETA:-1}"
GATE_FRAC="${GATE_FRAC:-0.1}"
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

log() { echo "[pick18-step10 $(date -u +%H:%M:%S)] $*"; }

task_meta() {
  case "$1" in
    desk_mug) SCENE=desk_mug HORIZON=500 ;;
    kettle) SCENE=kettle HORIZON=400 ;;
    *) SCENE="${1}" HORIZON=500 ;;
  esac
}

seed_from_beta1() {
  local task="$1"
  task_meta "${task}"
  local src="${SOURCE_DIR}/${task}"
  local dest="${RUN_DIR}/${task}"
  local ae_src="${src}/ae/ae_${SCENE}.pt"
  local pre_src="${src}/ac_pretrain/${task}_v21_pretrain"
  if [[ ! -f "${ae_src}" ]]; then
    log "ERROR: ${task} missing source AE ${ae_src}"
    return 1
  fi
  if [[ ! -f "${pre_src}/agent.pt" || ! -f "${pre_src}/buffer.npz" ]]; then
    log "ERROR: ${task} missing V21 pretrain at ${pre_src}"
    return 1
  fi
  mkdir -p "${dest}/ae" "${dest}/ac_pretrain" "${dest}/rl" "${dest}/eval" "${dest}/pids"
  ln -sfn "${ae_src}" "${dest}/ae/ae_${SCENE}.pt"
  ln -sfn "${pre_src}" "${dest}/ac_pretrain/${task}_v21_pretrain"
}

run_method() {
  local task="$1" gpu="$2" slot="$3"
  task_meta "${task}"
  local port=$((9000 + gpu * 10 + slot))
  local dest="${RUN_DIR}/${task}"
  local ae="${dest}/ae/ae_${SCENE}.pt"
  local pre_tag="${task}_v21_pretrain"
  local on_tag="${task}_v21_step10"
  local pre_dir="${dest}/ac_pretrain"
  local on_dir="${dest}/rl"
  mkdir -p "${pre_dir}" "${on_dir}" "${dest}/eval" "${dest}/pids"

  if [[ -f "${dest}/eval/${on_tag}_actor/result.json" ]]; then
    log "${task}/v21: already evaluated, skip"
    return 0
  fi

  seed_from_beta1 "${task}" || return 1

  local agent="${pre_dir}/${pre_tag}/agent.pt"
  local buf="${pre_dir}/${pre_tag}/buffer.npz"
  local on_agent="${on_dir}/${on_tag}/agent.pt"

  if [[ -f "${on_agent}" ]]; then
    log "${task}/v21: online agent exists, skip train"
  else
    log "${task}/v21: online gpu=${gpu} port=${port} gate_frac=${GATE_FRAC} horizon=${HORIZON}"
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --gpu "${gpu}" --port "${port}" \
        --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
        --tag "${on_tag}" --init-actor "${agent}" \
        --set "algorithm=rl_token" \
        --set "token_ae=${ae}" \
        --set "beta=${BETA}" \
        --set "gate_frac=${GATE_FRAC}" \
        --set "horizon=${HORIZON}" \
        --set "seed=0" \
        --set "init_buffer=${buf}" \
        --set "episode_pool=0-11" \
        --set "out_dir=${on_dir}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=false" \
        > "${LOCAL_LOG}/${on_tag}.log" 2>&1
    if [[ ! -f "${on_agent}" ]]; then
      log "ERROR: ${task}/v21 missing online agent"
      return 1
    fi
  fi

  log "${task}/v21: eval gpu=${gpu} port=${port} gate_frac=${GATE_FRAC}"
  cd "${CODE}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene "${SCENE}" --episodes 16 --gpu "${gpu}" --port "${port}" \
      --actor "${on_agent}" --token-ae "${ae}" \
      --tag "${on_tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "gate_frac=${GATE_FRAC}" \
      --set "out_dir=${dest}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${on_tag}_actor.log" 2>&1
  log "${task}/v21: done"
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
  : > "${Q}"
  for task in "${TASKS[@]}"; do
    echo "${task}"
  done >> "${Q}"
  log "queued $(grep -c . "${Q}") V21 step=${GATE_FRAC} jobs on ${#GPUS[@]} GPUs × ${SLOTS_PER_GPU} slots"
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
log "pick18 V21 step=${GATE_FRAC} sweep finished (fail=${fail})"
exit "${fail}"
