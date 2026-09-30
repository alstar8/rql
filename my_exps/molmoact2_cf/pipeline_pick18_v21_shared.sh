#!/usr/bin/env bash
# Shared V21 (one-pass Gaussian RL Token) over the same Pick-18 pool as Stage A.
#
# Isolates actor architecture vs Shared gOn Stage A (shared_v24_s0):
#   same 18 tasks, same shared AE, same merged frozen-VLA buffer, same
#   30k-step AC pretrain budget, same 1800 online episodes (100/task),
#   then the official 1000-episode MolmoPick val.
# V21 differences: algorithm=rl_token (Gaussian π(a|s,ã) + twin Q), frozen AE
# (no ae_finetune), no flow guidance.
#
# Reuses Stage A artifacts (no re-collect):
#   runs/pick18_v24_shared/ae/ae_pick18_shared.pt
#   runs/pick18_v24_shared/merged_buffer.npz
#
#   GPUS="0 1 2 3" bash pipeline_pick18_v21_shared.sh
#   PHASES="pretrain" GPUS="1" bash pipeline_pick18_v21_shared.sh
#   PHASES="online val1000" GPUS="0 1 2 3" bash pipeline_pick18_v21_shared.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SRC_DIR="${SRC_DIR:-${ROOT}/runs/pick18_v24_shared}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v21_shared}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_v21_shared_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
PHASES="${PHASES:-pretrain online val1000}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"

OFFLINE_STEPS="${OFFLINE_STEPS:-30000}"
BETA_PRETRAIN="${BETA_PRETRAIN:-100}"
ONLINE_BETA="${ONLINE_BETA:-1}"
EPISODES="${EPISODES:-1800}"
PROBE_EPISODES="${PROBE_EPISODES:-180}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
COLLECTORS_PER_GPU="${COLLECTORS_PER_GPU:-4}"

OBJECTS=(remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box)
read -r -a TASKS <<< "${TASKS:-desk_mug kettle ${OBJECTS[*]}}"

mkdir -p "${RUN_DIR}/pids" "${RUN_DIR}/queue" "${RUN_DIR}/ae" "${RUN_DIR}/rl" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[pick18-v21-shared $(date -u +%H:%M:%S)] $*"; }
has_phase() { [[ " ${PHASES} " == *" $1 "* ]]; }

task_meta() {
  case "$1" in
    desk_mug) SCENE=desk_mug HORIZON=500 ;;
    kettle) SCENE=kettle HORIZON=400 ;;
    *) SCENE="${1}" HORIZON=500 ;;
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

SRC_AE="${SRC_DIR}/ae/ae_pick18_shared.pt"
SRC_MERGED="${SRC_DIR}/merged_buffer.npz"
AE="${RUN_DIR}/ae/ae_pick18_shared.pt"
MERGED="${RUN_DIR}/merged_buffer.npz"
PRE_DIR="${RUN_DIR}/ac_pretrain"
PRE_TAG="shared_v21_pretrain"
ON_DIR="${RUN_DIR}/rl"
ON_TAG="${ON_TAG:-shared_v21_s0}"
EVAL_DIR="${RUN_DIR}/eval"
VAL_DIR="${RUN_DIR}/eval_val1000"

if [[ ! -f "${SRC_AE}" ]]; then
  log "ERROR: Stage A AE missing: ${SRC_AE}"
  exit 1
fi
if [[ ! -f "${SRC_MERGED}" ]]; then
  log "ERROR: Stage A merged buffer missing: ${SRC_MERGED}"
  exit 1
fi
if [[ ! -e "${AE}" ]]; then
  ln -sfn "${SRC_AE}" "${AE}"
  log "ae: linked ${SRC_AE}"
fi
if [[ ! -e "${MERGED}" ]]; then
  ln -sfn "${SRC_MERGED}" "${MERGED}"
  log "buffer: linked ${SRC_MERGED}"
fi

# --------------------------------------------------------------------------------------
# Shared AC pretrain (replay-only): V21 Gaussian BC + TD on the Stage A merge
# --------------------------------------------------------------------------------------
if has_phase pretrain; then
  if [[ -f "${PRE_DIR}/${PRE_TAG}/agent.pt" ]]; then
    log "pretrain: agent exists, skip"
  else
    log "pretrain: ${OFFLINE_STEPS} offline steps on ${MERGED} (algorithm=rl_token beta=${BETA_PRETRAIN})"
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_pretrain_ac.py \
        --scene desk_mug --encoder pick18_shared \
        --episodes 1 --offline-steps "${OFFLINE_STEPS}" \
        --gpu "${GPUS[0]}" --port 9500 --tag "${PRE_TAG}" \
        --set "algorithm=rl_token" \
        --set "init_random_ae=false" \
        --set "token_ae=${AE}" \
        --set "beta=${BETA_PRETRAIN}" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "out_dir=${PRE_DIR}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=false" \
        --set "vla_traj=${MERGED}" \
        > "${LOCAL_LOG}/${PRE_TAG}.log" 2>&1
    if [[ ! -f "${PRE_DIR}/${PRE_TAG}/agent.pt" ]]; then
      log "ERROR: pretrain produced no agent (see ${LOCAL_LOG}/${PRE_TAG}.log)"
      exit 1
    fi
  fi
fi

# --------------------------------------------------------------------------------------
# Shared online: one learner, one VLA server per GPU, collectors cycle the 18 tasks
# --------------------------------------------------------------------------------------
if has_phase online; then
  if [[ ! -f "${PRE_DIR}/${PRE_TAG}/agent.pt" ]]; then
    log "ERROR: online needs ${PRE_DIR}/${PRE_TAG}/agent.pt"
    exit 1
  fi
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
    base_port=9400
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
      log "online: resume from ${episodes_done}/${EPISODES} episodes, ${n_collectors} collectors on ${#GPUS[@]} GPUs, beta=${ONLINE_BETA}"
    else
      log "online: ${EPISODES} episodes over ${#TASKS[@]} tasks, ${n_collectors} collectors on ${#GPUS[@]} GPUs, beta=${ONLINE_BETA} algorithm=rl_token"
    fi
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene desk_mug --encoder pick18_shared \
        --gpu "${GPUS[0]}" --port "${base_port}" \
        --episodes "${EPISODES}" --warmup-episodes 0 \
        --tag "${ON_TAG}" --init-actor "${PRE_DIR}/${PRE_TAG}/agent.pt" \
        --set "algorithm=rl_token" \
        --set "init_random_ae=false" \
        --set "token_ae=${AE}" \
        --set "beta=${ONLINE_BETA}" \
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
        --set "ae_finetune=false" \
        "${resume_args[@]}" \
        >> "${LOCAL_LOG}/${ON_TAG}.log" 2>&1
    if [[ ! -f "${ON_DIR}/${ON_TAG}/agent.pt" ]]; then
      log "ERROR: online produced no agent"
      exit 1
    fi
  fi
fi

# --------------------------------------------------------------------------------------
# Optional 18-task held-out eval (no gOn/gOff: V21 has no guidance scale)
# --------------------------------------------------------------------------------------
eval_task() {
  local task="$1" gpu="$2" slot="$3"
  task_meta "${task}"
  local port=$((9800 + gpu * 10 + slot))
  local etag="${ON_TAG}_${task}"
  if [[ -f "${EVAL_DIR}/${etag}/result.json" ]]; then
    log "eval ${task}: exists, skip"
    return 0
  fi
  log "eval ${task}: gpu=${gpu} port=${port}"
  cd "${CODE}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene "${SCENE}" --episodes "${EVAL_EPISODES}" --gpu "${gpu}" --port "${port}" \
      --actor "${ON_DIR}/${ON_TAG}/agent.pt" --token-ae "${AE}" \
      --tag "${etag}" \
      --set "horizon=${HORIZON}" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "out_dir=${EVAL_DIR}" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${etag}.log" 2>&1
}

worker() {
  local gpu="$1" slot="$2" phase="$3"
  local q="${RUN_DIR}/queue/${phase}.txt"
  local lock="${RUN_DIR}/queue/${phase}.lock"
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
    eval_task "${job}" "${gpu}" "${slot}" || log "ERROR: eval ${job} failed"
  done
}

run_queue() {
  local phase="$1"
  local q="${RUN_DIR}/queue/${phase}.txt"
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
  q="${RUN_DIR}/queue/eval.txt"
  : > "${q}"
  for task in "${TASKS[@]}"; do
    echo "${task}" >> "${q}"
  done
  run_queue eval || fail=1
fi

# --------------------------------------------------------------------------------------
# Official MolmoPick val-1000 (same shards as Stage A gOn)
# --------------------------------------------------------------------------------------
if has_phase val1000; then
  if [[ ! -f "${ON_DIR}/${ON_TAG}/agent.pt" ]]; then
    log "ERROR: val1000 needs ${ON_DIR}/${ON_TAG}/agent.pt"
    exit 1
  fi
  mkdir -p "${VAL_DIR}"
  if [[ ! -e "${VAL_DIR}/shards" && -d "${SRC_DIR}/eval_val1000/shards" ]]; then
    ln -sfn "${SRC_DIR}/eval_val1000/shards" "${VAL_DIR}/shards"
    log "val1000: reusing Stage A shards ${VAL_DIR}/shards"
  fi
  log "val1000: V21 actor on 1000 val episodes"
  cd "${ROOT}"
  if ! env RUN_DIR="${RUN_DIR}" EVAL_DIR="${VAL_DIR}" ACTOR="${ON_DIR}/${ON_TAG}/agent.pt" \
      AE="${AE}" LOCAL_LOG="${LOCAL_LOG}" PHASES="actor" GPUS="${GPUS[*]}" \
      bash pipeline_eval_val1000.sh; then
    log "ERROR: val1000 failed"
    fail=1
  fi
fi

log "pick18 V21 shared finished (fail=${fail})"
exit "${fail}"
