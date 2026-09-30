#!/usr/bin/env bash
# Stage B (V24): the lambda_pi=1 second arm on the shared 18-task policy.
#
# Stage A ran the stage-0 recipe (cf_actor_coef=0 online): V gets BC + anchor
# only, all value improvement arrives through the distilled G. Stage B turns
# the one-step guided lookahead back on (cf_actor_coef=1) from the same
# pretrain and the same merged buffer, then runs the same paired eval.
#
#   GPUS="0 1 2 3" bash pipeline_pick18_v24_stageB.sh
#   PHASES="eval" GPUS="0 1 2 3" bash pipeline_pick18_v24_stageB.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v24_shared}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_v24_shared_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
PHASES="${PHASES:-online eval}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"

ONLINE_BETA="${ONLINE_BETA:-1}"
EPISODES="${EPISODES:-1800}"
PROBE_EPISODES="${PROBE_EPISODES:-180}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
COLLECTORS_PER_GPU="${COLLECTORS_PER_GPU:-4}"
CF_ACTOR_COEF="${CF_ACTOR_COEF:-1}"

OBJECTS=(remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box)
read -r -a TASKS <<< "${TASKS:-desk_mug kettle ${OBJECTS[*]}}"

mkdir -p "${RUN_DIR}/pids" "${RUN_DIR}/queue" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline_stageB.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[pick18-v24-stageB $(date -u +%H:%M:%S)] $*"; }
has_phase() { [[ " ${PHASES} " == *" $1 "* ]]; }

task_meta() {
  case "$1" in
    desk_mug)
      SCENE=desk_mug HORIZON=500
      ;;
    kettle)
      SCENE=kettle HORIZON=400
      ;;
    *)
      SCENE="${1}" HORIZON=500
      ;;
  esac
}

task_pool_spec() {
  local parts=()
  local task
  for task in "${TASKS[@]}"; do
    task_meta "${task}"
    parts+=("${SCENE}:${HORIZON}")
  done
  local IFS=,
  echo "${parts[*]}"
}

AE="${RUN_DIR}/ae/ae_pick18_shared.pt"
AE_TOKENS="${RUN_DIR}/ae/tokens"
MERGED="${RUN_DIR}/merged_buffer.npz"
PRE_DIR="${RUN_DIR}/ac_pretrain"
PRE_TAG="shared_v24_pretrain"
ON_DIR="${RUN_DIR}/rl"
ON_TAG="${ON_TAG:-shared_v24_s1}"
EVAL_DIR="${RUN_DIR}/eval_stageB"

for f in "${AE}" "${MERGED}" "${PRE_DIR}/${PRE_TAG}/agent.pt" "${PRE_DIR}/${PRE_TAG}/buffer.npz"; do
  if [[ ! -f "${f}" ]]; then
    log "ERROR: Stage B needs ${f} from Stage A"
    exit 1
  fi
done

# --------------------------------------------------------------------------------------
# Online, lambda_pi=1 (cf_actor_coef=1), from the Stage A pretrain + merged buffer
# --------------------------------------------------------------------------------------
if has_phase online; then
  episodes_done=0
  if [[ -f "${ON_DIR}/${ON_TAG}/metrics.jsonl" ]]; then
    episodes_done="$(python3 -c "import pathlib,sys
n=0
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line.strip(): n+=1
print(n)" "${ON_DIR}/${ON_TAG}/metrics.jsonl")"
  fi
  if [[ -f "${ON_DIR}/${ON_TAG}/agent.pt" && "${episodes_done}" -ge "${EPISODES}" ]]; then
    log "online: already finished ${episodes_done} eps, skip"
  else
    vla_ports=()
    vla_gpus=()
    egl_devices=()
    base_port=9700
    for i in "${!GPUS[@]}"; do
      vla_ports+=($((base_port + i)))
      vla_gpus+=("${GPUS[$i]}")
      egl_devices+=("${GPUS[$i]}")
    done
    join_by() { local IFS="$1"; shift; echo "$*"; }
    n_collectors=$(( ${#GPUS[@]} * COLLECTORS_PER_GPU ))
    resume_args=()
    if [[ "${episodes_done}" -gt 0 ]]; then
      resume_args+=(--set "resume=true")
      log "online: resume from ${episodes_done}/${EPISODES} episodes, ${n_collectors} collectors on ${#GPUS[@]} GPUs, beta=${ONLINE_BETA} cf_actor_coef=${CF_ACTOR_COEF}"
    else
      log "online: ${EPISODES} episodes over ${#TASKS[@]} tasks, ${n_collectors} collectors on ${#GPUS[@]} GPUs, beta=${ONLINE_BETA} cf_actor_coef=${CF_ACTOR_COEF}"
    fi
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene desk_mug --encoder pick18_shared \
        --gpu "${GPUS[0]}" --port "${base_port}" \
        --episodes "${EPISODES}" --warmup-episodes 0 \
        --tag "${ON_TAG}" --init-actor "${PRE_DIR}/${PRE_TAG}/agent.pt" \
        --set "algorithm=v22_24" \
        --set "init_random_ae=false" \
        --set "token_ae=${AE}" \
        --set "beta=${ONLINE_BETA}" \
        --set "cf_actor_coef=${CF_ACTOR_COEF}" \
        --set "cf_ref_conditioned=true" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "seed=0" \
        --set "init_buffer=${PRE_DIR}/${PRE_TAG}/buffer.npz" \
        --set "task_pool=$(task_pool_spec)" \
        --set "episode_pool=0-11" \
        --set "probe_episodes=${PROBE_EPISODES}" \
        --set "collectors=${n_collectors}" \
        --set "egl_slots=${SLOTS_PER_GPU}" \
        --set "vla_ports=$(join_by , "${vla_ports[@]}")" \
        --set "vla_gpus=$(join_by , "${vla_gpus[@]}")" \
        --set "egl_devices=$(join_by , "${egl_devices[@]}")" \
        --set "buffer_capacity=400000" \
        --set "out_dir=${ON_DIR}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=true" \
        --set "store_decision_tokens=false" \
        --set "token_replay=${AE_TOKENS}/*/*.npz" \
        "${resume_args[@]}" \
        >> "${LOCAL_LOG}/${ON_TAG}.log" 2>&1
    if [[ ! -f "${ON_DIR}/${ON_TAG}/agent.pt" ]]; then
      log "ERROR: online produced no agent"
      exit 1
    fi
  fi
fi

# --------------------------------------------------------------------------------------
# Paired eval, same protocol as Stage A
# --------------------------------------------------------------------------------------
eval_task() {
  local task="$1" g="$2" gpu="$3" slot="$4"
  task_meta "${task}"
  local port=$((9800 + gpu * 10 + slot))
  local gcoef="-1"
  [[ "${g}" == "off" ]] && gcoef="0"
  local etag="${ON_TAG}_${task}_g${g^}"
  if [[ -f "${EVAL_DIR}/${etag}/result.json" ]]; then
    log "eval ${task} ${g}: exists, skip"
    return 0
  fi
  log "eval ${task} ${g}: gpu=${gpu} port=${port} guidance_coef=${gcoef}"
  cd "${CODE}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene "${SCENE}" --episodes "${EVAL_EPISODES}" --gpu "${gpu}" --port "${port}" \
      --actor "${ON_DIR}/${ON_TAG}/agent.pt" --token-ae "${AE}" \
      --tag "${etag}" \
      --set "horizon=${HORIZON}" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "guidance_coef=${gcoef}" \
      --set "out_dir=${EVAL_DIR}" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${etag}.log" 2>&1
}

worker() {
  local gpu="$1" slot="$2" phase="$3"
  local q="${RUN_DIR}/queue/${phase}_stageB.txt"
  local lock="${RUN_DIR}/queue/${phase}_stageB.lock"
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
    [[ -z "${job}" ]] && return 0
    log "gpu=${gpu} slot=${slot} ${phase}: ${job}"
    case "${phase}" in
      eval) eval_task ${job//:/ } "${gpu}" "${slot}" || log "ERROR: eval ${job} failed" ;;
    esac
  done
}

run_queue() {
  local phase="$1"
  local q="${RUN_DIR}/queue/${phase}_stageB.txt"
  [[ -s "${q}" ]] || { log "${phase}: empty queue, skip"; return 0; }
  log "${phase}: $(grep -c . "${q}") jobs on ${#GPUS[@]} GPUs x ${SLOTS_PER_GPU} slots"
  pids=()
  for gpu in "${GPUS[@]}"; do
    for ((slot = 0; slot < SLOTS_PER_GPU; slot++)); do
      worker "${gpu}" "${slot}" "${phase}" &
      pids+=($!)
    done
  done
  local fail=0
  for p in "${pids[@]}"; do
    wait "${p}" || fail=1
  done
  return "${fail}"
}

fail=0
if has_phase eval; then
  q="${RUN_DIR}/queue/eval_stageB.txt"
  : > "${q}"
  for task in "${TASKS[@]}"; do
    for g in on off; do
      echo "${task}:${g}" >> "${q}"
    done
  done
  run_queue eval || fail=1
fi

log "pick18 V24 Stage B finished (fail=${fail})"
exit "${fail}"
