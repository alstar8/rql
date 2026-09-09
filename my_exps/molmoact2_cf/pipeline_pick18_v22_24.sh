#!/usr/bin/env bash
# Pick-18 V22_24: ConsensusFlow losses on the flow_rlt recipe, all 18 tasks.
#
# Per task: scene AE + 100-traj VLA buffer from runs/pick18/, fresh AC pretrain
# (8000, cf_actor_coef=0, beta=100) -> 10 probe -> 300 online -> paired held-out
# eval48 (guidance on vs off).
#
# Kettle ablation read (V22_24_METHODS.md): stage 0 (cf_actor_coef=0, distilled-G
# routing) is the arm that improved early online; stage 1 (lookahead on) with a
# weak anchor fell below V22_23 at the same count. So the sweep runs stage 0 by
# default; STAGES="0 1" adds the joint arm as a second job per task.
#
# All tasks start together: GPUS x SLOTS_PER_GPU workers pull from a queue.
# Re-running skips finished evals. Online runs with agent.pt but fewer than
# EPISODES stored episodes resume (`resume=true`) instead of restarting.
#
#   GPUS="0 1 2 3" SLOTS_PER_GPU=3 bash pipeline_pick18_v22_24.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SOURCE_DIR="${SOURCE_DIR:-${ROOT}/runs/pick18}"
#: If set, reuse that sweep's AC pretrain (agent.pt + buffer.npz) instead of
#: running 8000 offline steps again. Used by the β=1 online sweep, which starts
#: from the β=100 AC checkpoints in runs/pick18_v22_24/.
PRETRAIN_DIR="${PRETRAIN_DIR:-}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v22_24}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_v22_24_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
EPISODES="${EPISODES:-300}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
BETA_PRETRAIN="${BETA_PRETRAIN:-100}"
ONLINE_BETA="${ONLINE_BETA:-100}"
WARMUP=0
read -r -a STAGES <<< "${STAGES:-0}"

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

log() { echo "[pick18-v22_24 $(date -u +%H:%M:%S)] $*"; }

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
  if [[ -n "${PRETRAIN_DIR}" ]]; then
    local pre_src="${PRETRAIN_DIR}/${task}/ac_pretrain/${task}_v22_24_pretrain"
    local pre_dest="${dest}/ac_pretrain/${task}_v22_24_pretrain"
    if [[ -f "${pre_src}/agent.pt" && -f "${pre_src}/buffer.npz" && ! -e "${pre_dest}/agent.pt" ]]; then
      ln -sfn "${pre_src}" "${pre_dest}"
    fi
  fi
}

run_task() {
  local task="$1" stage="$2" gpu="$3" slot="$4"
  task_meta "${task}"
  local port=$((9200 + gpu * 10 + slot))
  local dest="${RUN_DIR}/${task}"
  local ae="${dest}/ae/ae_${SCENE}.pt"
  local shared_buf="${dest}/vla_buffer.npz"
  local pre_tag="${task}_v22_24_pretrain"
  local on_tag="${task}_v22_24_s${stage}"
  local pre_dir="${dest}/ac_pretrain"
  local on_dir="${dest}/rl"
  mkdir -p "${pre_dir}" "${on_dir}" "${dest}/eval" "${dest}/pids"

  if [[ -f "${dest}/eval/${on_tag}_gOn/result.json" && -f "${dest}/eval/${on_tag}_gOff/result.json" ]]; then
    log "${task}/s${stage}: already evaluated, skip"
    return 0
  fi

  seed_task "${task}" || return 1

  local agent="${pre_dir}/${pre_tag}/agent.pt"
  local buf="${pre_dir}/${pre_tag}/buffer.npz"
  local on_agent="${on_dir}/${on_tag}/agent.pt"

  # Shared AC pretrain: BC + tight anchor V, live reverse-TD critic, distilled G.
  # One per task; both stages start from this checkpoint.
  if [[ -f "${agent}" ]]; then
    log "${task}: pretrained agent exists, skip pretrain"
  else
    log "${task}: pretrain gpu=${gpu} port=${port} beta=${BETA_PRETRAIN} cf_actor_coef=0"
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_pretrain_ac.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --episodes 100 --offline-steps "${OFFLINE_STEPS}" \
        --gpu "${gpu}" --port "${port}" --tag "${pre_tag}" \
        --set "algorithm=v22_24" \
        --set "init_random_ae=false" \
        --set "token_ae=${ae}" \
        --set "beta=${BETA_PRETRAIN}" \
        --set "cf_actor_coef=0" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "horizon=${HORIZON}" \
        --set "episode_pool=0-47" \
        --set "out_dir=${pre_dir}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=true" \
        --set "store_decision_tokens=false" \
        --set "vla_traj=${shared_buf}" \
        --set "token_replay=${TOKENS}/*.npz" \
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
    log "${task}/s${stage}: online agent exists (${episodes_done} eps), skip train"
  else
    local resume_args=()
    if [[ -f "${on_agent}" && "${episodes_done}" -gt 0 ]]; then
      resume_args+=(--set "resume=true")
      log "${task}/s${stage}: resume online gpu=${gpu} port=${port} from ep ${episodes_done}/${EPISODES} cf_actor_coef=${stage} beta=${ONLINE_BETA}"
    else
      log "${task}/s${stage}: online gpu=${gpu} port=${port} cf_actor_coef=${stage} beta=${ONLINE_BETA}"
    fi
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --gpu "${gpu}" --port "${port}" \
        --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
        --tag "${on_tag}" --init-actor "${agent}" \
        --set "algorithm=v22_24" \
        --set "init_random_ae=false" \
        --set "token_ae=${ae}" \
        --set "beta=${ONLINE_BETA}" \
        --set "cf_actor_coef=${stage}" \
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
        --set "ae_finetune=true" \
        --set "store_decision_tokens=false" \
        --set "token_replay=${TOKENS}/*.npz" \
        "${resume_args[@]}" \
        >> "${LOCAL_LOG}/${on_tag}.log" 2>&1
    if [[ ! -f "${on_agent}" ]]; then
      log "ERROR: ${task}/s${stage} missing online agent"
      return 1
    fi
  fi

  # Paired eval on the same held-out bench: guidance on (trained lambda) vs off.
  for g in on off; do
    local gcoef="-1"
    [[ "${g}" == "off" ]] && gcoef="0"
    local etag="${on_tag}_g${g^}"
    if [[ -f "${dest}/eval/${etag}/result.json" ]]; then
      log "${task}/s${stage}: eval ${g} exists, skip"
      continue
    fi
    log "${task}/s${stage}: eval ${g} gpu=${gpu} port=${port} guidance_coef=${gcoef}"
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_eval.py \
        --scene "${SCENE}" --episodes "${EVAL_EPISODES}" --gpu "${gpu}" --port "${port}" \
        --actor "${on_agent}" --token-ae "${ae}" \
        --tag "${etag}" \
        --set "horizon=${HORIZON}" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "guidance_coef=${gcoef}" \
        --set "out_dir=${dest}/eval" \
        --set "checkpoint=${CKPT}" \
        >> "${LOCAL_LOG}/${etag}.log" 2>&1
  done
  log "${task}/s${stage}: done"
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
    local task="${job%:*}"
    local stage="${job##*:}"
    log "gpu=${gpu} slot=${slot} start ${task} stage ${stage}"
    if ! run_task "${task}" "${stage}" "${gpu}" "${slot}"; then
      log "ERROR: ${task} stage ${stage} failed on gpu=${gpu}"
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
    for stage in "${STAGES[@]}"; do
      echo "${task}:${stage}"
    done
  done >> "${Q}"
  log "queued $(grep -c . "${Q}") V22_24 jobs on ${#GPUS[@]} GPUs x ${SLOTS_PER_GPU} slots, stages=${STAGES[*]}, online_beta=${ONLINE_BETA}"
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
log "pick18 V22_24 sweep finished (fail=${fail})"
exit "${fail}"
