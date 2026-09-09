#!/usr/bin/env bash
# Pick-18 PPO: on-policy clipped surrogate on the V21 Gaussian actor.
#
# Same protocol as V21 / V22_24: per-task scene AE + 100-traj VLA buffer from
# runs/pick18/, 8000-step AC pretrain (BC + TD on V, not PPO) -> 10 probe
# -> 300 online PPO -> held-out eval48 (64 rollouts). No guidance on/off pair.
#
#   GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 bash pipeline_pick18_ppo.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SOURCE_DIR="${SOURCE_DIR:-${ROOT}/runs/pick18}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_ppo}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_ppo_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
# Extra workers on GPUs that already have jobs: SLOT_OFFSET=2 uses ports 94X2.
SLOT_OFFSET="${SLOT_OFFSET:-0}"
EPISODES="${EPISODES:-300}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
PPO_CLIP="${PPO_CLIP:-0.2}"
PPO_LAMBDA="${PPO_LAMBDA:-0.95}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
PPO_HORIZON="${PPO_HORIZON:-256}"
PPO_MINIBATCH="${PPO_MINIBATCH:-64}"
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

log() { echo "[pick18-ppo $(date -u +%H:%M:%S)] $*"; }

# Match `scripts/run_*.py --tag <tag> ` (trailing space) so eval's
# `--tag <tag>_actor` is not treated as the same job.
tag_alive() { pgrep -f -- "--tag ${1} " >/dev/null 2>&1; }

wait_tag() {
  local tag="$1"
  if tag_alive "${tag}"; then
    log "${tag}: already running, waiting"
    while tag_alive "${tag}"; do
      sleep 20
    done
    log "${tag}: previous process exited"
  fi
}

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
  local pre_tag="${task}_ppo_pretrain"
  local on_tag="${task}_ppo_s0"
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

  wait_tag "${pre_tag}"
  if [[ -f "${agent}" ]]; then
    log "${task}: pretrained agent exists, skip pretrain"
  else
    log "${task}: pretrain gpu=${gpu} port=${port} ppo_clip=${PPO_CLIP}"
    cd "${CODE}"
    setsid -w env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_pretrain_ac.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --episodes 100 --offline-steps "${OFFLINE_STEPS}" \
        --gpu "${gpu}" --port "${port}" --tag "${pre_tag}" \
        --set "algorithm=ppo" \
        --set "init_random_ae=false" \
        --set "token_ae=${ae}" \
        --set "ppo_clip=${PPO_CLIP}" \
        --set "ppo_gae_lambda=${PPO_LAMBDA}" \
        --set "ppo_epochs=${PPO_EPOCHS}" \
        --set "ppo_horizon=${PPO_HORIZON}" \
        --set "ppo_minibatch=${PPO_MINIBATCH}" \
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

  wait_tag "${on_tag}"
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
      log "${task}: online gpu=${gpu} port=${port} ppo_clip=${PPO_CLIP}"
    fi
    cd "${CODE}"
    setsid -w env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --gpu "${gpu}" --port "${port}" \
        --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
        --tag "${on_tag}" --init-actor "${agent}" \
        --set "algorithm=ppo" \
        --set "init_random_ae=false" \
        --set "token_ae=${ae}" \
        --set "ppo_clip=${PPO_CLIP}" \
        --set "ppo_gae_lambda=${PPO_LAMBDA}" \
        --set "ppo_epochs=${PPO_EPOCHS}" \
        --set "ppo_horizon=${PPO_HORIZON}" \
        --set "ppo_minibatch=${PPO_MINIBATCH}" \
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
    if [[ ! -f "${on_agent}" ]]; then
      log "ERROR: ${task} missing online agent"
      return 1
    fi
  fi

  wait_tag "${on_tag}_actor"
  log "${task}: eval gpu=${gpu} port=${port}"
  cd "${CODE}"
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
  log "queued $(grep -c . "${Q}") PPO jobs on ${#GPUS[@]} GPUs x ${SLOTS_PER_GPU} slots, ppo_clip=${PPO_CLIP}"
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
log "pick18 PPO sweep finished (fail=${fail})"
exit "${fail}"
