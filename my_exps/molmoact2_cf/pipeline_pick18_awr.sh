#!/usr/bin/env bash
# Pick-18 AWR: Advantage-Weighted Regression on the V21 Gaussian actor.
#
# Same protocol as V21 / V22_24: per-task scene AE + 100-traj VLA buffer from
# runs/pick18/, 8000-step AC pretrain (AWR + TD, not unweighted BC) -> 10 probe
# -> 300 online -> held-out eval48 (64 rollouts). No guidance on/off pair.
#
#   GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 bash pipeline_pick18_awr.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SOURCE_DIR="${SOURCE_DIR:-${ROOT}/runs/pick18}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_awr}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_awr_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
# Extra workers on GPUs that already have jobs: SLOT_OFFSET=2 uses ports 94X2.
SLOT_OFFSET="${SLOT_OFFSET:-0}"
EPISODES="${EPISODES:-300}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
AWR_TEMP="${AWR_TEMP:-1}"
AWR_CLIP="${AWR_CLIP:-20}"
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

# Collectors import NLTK at startup. Parallel first-wave workers race
# nltk.download and corrupt wordnet2022.zip; pre-seed once before any job.
if [[ "${SKIP_NLTK_SEED:-0}" != "1" ]]; then
  "${SIM}" -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('wordnet2022', quiet=True)" >/dev/null 2>&1 || true
fi

log() { echo "[pick18-awr $(date -u +%H:%M:%S)] $*"; }

task_meta() {
  case "$1" in
    desk_mug) SCENE=desk_mug HORIZON=500 ;;
    kettle) SCENE=kettle HORIZON=400 ;;
    *) SCENE="${1}" HORIZON=500 ;;
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

run_task() {
  local task="$1" gpu="$2" slot="$3"
  task_meta "${task}"
  local port=$((9400 + gpu * 10 + slot))
  local dest="${RUN_DIR}/${task}"
  local ae="${dest}/ae/ae_${SCENE}.pt"
  local shared_buf="${dest}/vla_buffer.npz"
  local pre_tag="${task}_awr_pretrain"
  local on_tag="${task}_awr_s0"
  local pre_dir="${dest}/ac_pretrain"
  local on_dir="${dest}/rl"
  mkdir -p "${pre_dir}" "${on_dir}" "${dest}/eval" "${dest}/pids"

  if [[ -f "${dest}/eval/${on_tag}_actor/result.json" ]]; then
    log "${task}: already evaluated, skip"
    return 0
  fi

  seed_task "${task}" || return 1

  local agent="${pre_dir}/${pre_tag}/agent.pt"
  local buf="${pre_dir}/${pre_tag}/buffer.npz"
  local on_agent="${on_dir}/${on_tag}/agent.pt"

  if [[ -f "${agent}" ]]; then
    log "${task}: pretrained agent exists, skip pretrain"
  else
    log "${task}: pretrain gpu=${gpu} port=${port} awr_temp=${AWR_TEMP}"
    cd "${CODE}"
    setsid -w env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_pretrain_ac.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --episodes 100 --offline-steps "${OFFLINE_STEPS}" \
        --gpu "${gpu}" --port "${port}" --tag "${pre_tag}" \
        --set "algorithm=awr" \
        --set "init_random_ae=false" \
        --set "token_ae=${ae}" \
        --set "awr_temp=${AWR_TEMP}" \
        --set "awr_clip=${AWR_CLIP}" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
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
      log "ERROR: ${task} missing pretrained agent"
      return 1
    fi
  fi

  local progress_file="${on_dir}/${on_tag}/progress.json"
  local episodes_done=0
  if [[ -f "${progress_file}" ]]; then
    episodes_done="$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('episodes_done',0)))" "${progress_file}")"
  fi

  if [[ -f "${on_agent}" && "${episodes_done}" -ge "${EPISODES}" ]]; then
    log "${task}: online agent exists (${episodes_done} eps), skip train"
  else
    local resume_args=()
    if [[ -f "${on_agent}" && "${episodes_done}" -gt 0 ]]; then
      resume_args+=(--set "resume=true")
      log "${task}: resume online gpu=${gpu} port=${port} from ep ${episodes_done}/${EPISODES}"
    else
      log "${task}: online gpu=${gpu} port=${port} awr_temp=${AWR_TEMP}"
    fi
    cd "${CODE}"
    set +e
    setsid -w env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --gpu "${gpu}" --port "${port}" \
        --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
        --tag "${on_tag}" --init-actor "${agent}" \
        --set "algorithm=awr" \
        --set "init_random_ae=false" \
        --set "token_ae=${ae}" \
        --set "awr_temp=${AWR_TEMP}" \
        --set "awr_clip=${AWR_CLIP}" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "horizon=${HORIZON}" \
        --set "seed=0" \
        --set "init_buffer=${buf}" \
        --set "episode_pool=0-11" \
        --set "out_dir=${on_dir}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=false" \
        "${resume_args[@]}" \
        >> "${LOCAL_LOG}/${on_tag}.log" 2>&1
    train_rc=$?
    set -e
    if [[ -f "${progress_file}" ]]; then
      episodes_done="$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('episodes_done',0)))" "${progress_file}")"
    fi
    # `if ! run_task` disables set -e inside this function; agent.pt from a
    # previous incomplete run must not skip us into held-out eval.
    if [[ "${train_rc}" -ne 0 || ! -f "${on_agent}" || "${episodes_done}" -lt "${EPISODES}" ]]; then
      log "ERROR: ${task} online incomplete rc=${train_rc} eps=${episodes_done}/${EPISODES}"
      return 1
    fi
  fi

  log "${task}: eval gpu=${gpu} port=${port}"
  cd "${CODE}"
  set +e
  setsid -w env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene "${SCENE}" --episodes "${EVAL_EPISODES}" --gpu "${gpu}" --port "${port}" \
      --actor "${on_agent}" --token-ae "${ae}" \
      --tag "${on_tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "out_dir=${dest}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${on_tag}_actor.log" 2>&1
  eval_rc=$?
  set -e
  if [[ "${eval_rc}" -ne 0 || ! -f "${dest}/eval/${on_tag}_actor/result.json" ]]; then
    log "ERROR: ${task} eval failed rc=${eval_rc}"
    return 1
  fi
  log "${task}: done"
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
    if ! run_task "${job}" "${gpu}" "${slot}"; then
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
  log "queued $(grep -c . "${Q}") AWR jobs on ${#GPUS[@]} GPUs x ${SLOTS_PER_GPU} slots, awr_temp=${AWR_TEMP}"
else
  log "SKIP_ENQUEUE: extra workers GPUs=${GPUS[*]} slots=${SLOTS_PER_GPU} offset=${SLOT_OFFSET} queue=$(grep -c . "${Q}" || echo 0)"
fi

cd "${CODE}"
pids=()
for gpu in "${GPUS[@]}"; do
  for ((slot = SLOT_OFFSET; slot < SLOT_OFFSET + SLOTS_PER_GPU; slot++)); do
    worker "${gpu}" "${slot}" &
    pids+=($!)
  done
done
log "workers ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  wait "${p}" || fail=1
done
log "pick18 AWR sweep finished (fail=${fail})"
exit "${fail}"
