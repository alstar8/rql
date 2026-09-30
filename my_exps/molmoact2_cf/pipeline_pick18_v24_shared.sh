#!/usr/bin/env bash
# Stage A (V24): ONE shared v22_24 policy over the Pick-18 train pool.
#
# Answers the scaling question cheaply, on the 18 tasks that already have a
# finished specialist number (macro 92.8% gOn): does a single shared
# V/W/Q/AE match the per-task specialists when everything else is held fixed?
#
# Method differences from the specialist sweep (pipeline_pick18_v22_24.sh):
#   - one shared token AE over all 18 corpora (specialists used per-task AEs)
#   - W and Q read the VLA reference chunk (cf_ref_conditioned=true), so
#     guidance and value can tell tasks apart; V always could
#   - one learner, one replay: collectors cycle tasks round-robin
#     (episode N -> task_list[N % 18]) into a single shared buffer
#   - multi-GPU collection: one frozen-VLA server per GPU, collectors pinned
#     round-robin to (port, EGL device); learner trains on GPU 0
#
# Phases, each skippable when its output exists:
#   1. AE       shared token AE over the 18 token corpora (symlinked tree)
#   2. COLLECT  per task: 100 frozen-VLA episodes re-encoded by the SHARED AE
#               (runs/pick18 buffers live in per-task AE space -- not reusable)
#   3. MERGE    concatenate the 18 buffers into one replay
#   4. PRETRAIN replay-only AC pretrain on the merged buffer
#               (beta=100, cf_actor_coef=0 -- the recipe both specialist
#               stages started from)
#   5. ONLINE   one shared online run: beta=1, cf_actor_coef=0 (stage 0, the
#               arm that improved early online), 1800 episodes = 100/task
#   6. EVAL     per-task paired eval (guidance on vs off) with the shared
#               agent, same bench and episode count as the specialist sweep
#
#   GPUS="0 1 2 3 4 5 6 7" bash pipeline_pick18_v24_shared.sh
#   PHASES="eval" GPUS="0 1 2 3" bash pipeline_pick18_v24_shared.sh   # re-run one phase

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v24_shared}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_v24_shared_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
PHASES="${PHASES:-ae collect merge pretrain online eval}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"

# AE: 1000 sequences per task at ~4 MB each (float16) lands near 72 GB RAM.
AE_STEPS="${AE_STEPS:-8000}"
AE_MAX_SEQUENCES="${AE_MAX_SEQUENCES:-18000}"
COLLECT_EPISODES="${COLLECT_EPISODES:-100}"
OFFLINE_STEPS="${OFFLINE_STEPS:-30000}"
BETA_PRETRAIN="${BETA_PRETRAIN:-100}"
ONLINE_BETA="${ONLINE_BETA:-1}"
EPISODES="${EPISODES:-1800}"
PROBE_EPISODES="${PROBE_EPISODES:-180}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
COLLECTORS_PER_GPU="${COLLECTORS_PER_GPU:-4}"

OBJECTS=(remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box)
read -r -a TASKS <<< "${TASKS:-desk_mug kettle ${OBJECTS[*]}}"

mkdir -p "${RUN_DIR}/pids" "${RUN_DIR}/queue" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[pick18-v24-shared $(date -u +%H:%M:%S)] $*"; }
has_phase() { [[ " ${PHASES} " == *" $1 "* ]]; }

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

# Comma-separated scene:horizon cycle for the shared online run.
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

AE_DIR="${RUN_DIR}/ae"
AE="${AE_DIR}/ae_pick18_shared.pt"
AE_TOKENS="${AE_DIR}/tokens"
MERGED="${RUN_DIR}/merged_buffer.npz"
PRE_DIR="${RUN_DIR}/ac_pretrain"
PRE_TAG="shared_v24_pretrain"
ON_DIR="${RUN_DIR}/rl"
ON_TAG="${ON_TAG:-shared_v24_s0}"
EVAL_DIR="${RUN_DIR}/eval"

# --------------------------------------------------------------------------------------
# Phase 1: shared token AE
# --------------------------------------------------------------------------------------
if has_phase ae; then
  if [[ -f "${AE}" ]]; then
    log "ae: ${AE} exists, skip"
  else
    log "ae: linking ${#TASKS[@]} token corpora into ${AE_TOKENS}"
    mkdir -p "${AE_TOKENS}"
    for task in "${TASKS[@]}"; do
      task_meta "${task}"
      if [[ ! -d "${TOKENS}" ]]; then
        log "ERROR: ${task} missing token corpus ${TOKENS}"
        exit 1
      fi
      mkdir -p "${AE_TOKENS}/${SCENE}"
      ln -sfn "${TOKENS}"/*.npz "${AE_TOKENS}/${SCENE}/"
    done
    log "ae: training ${AE_STEPS} steps on ${AE_TOKENS}/*/*.npz (max ${AE_MAX_SEQUENCES} sequences)"
    cd "${CODE}"
    "${SIM}" -m rlt.train_token_ae \
      --token_replay "${AE_TOKENS}/*/*.npz" \
      --out "${AE}" \
      --steps "${AE_STEPS}" \
      --max_sequences "${AE_MAX_SEQUENCES}" \
      > "${LOCAL_LOG}/ae_pick18_shared.log" 2>&1
  fi
fi

# --------------------------------------------------------------------------------------
# Phase 2: re-collect per-task VLA buffers with the shared AE (queue over GPUs)
# --------------------------------------------------------------------------------------
collect_task() {
  local task="$1" gpu="$2" slot="$3"
  task_meta "${task}"
  local port=$((9400 + gpu * 10 + slot))
  local dest="${RUN_DIR}/collect/${task}"
  if [[ -f "${dest}/buffer.npz" ]]; then
    log "collect ${task}: buffer exists, skip"
    return 0
  fi
  log "collect ${task}: ${COLLECT_EPISODES} frozen episodes gpu=${gpu} port=${port}"
  cd "${CODE}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_pretrain_ac.py \
      --scene "${SCENE}" --encoder "${SCENE}" \
      --episodes "${COLLECT_EPISODES}" --offline-steps 0 \
      --gpu "${gpu}" --port "${port}" --tag "${task}" \
      --set "algorithm=v22_24" \
      --set "init_random_ae=false" \
      --set "token_ae=${AE}" \
      --set "cf_ref_conditioned=true" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "horizon=${HORIZON}" \
      --set "episode_pool=0-47" \
      --set "out_dir=${RUN_DIR}/collect" \
      --set "checkpoint=${CKPT}" \
      --set "store_decision_tokens=false" \
      > "${LOCAL_LOG}/collect_${task}.log" 2>&1
  if [[ ! -f "${dest}/buffer.npz" ]]; then
    log "ERROR: collect ${task} produced no buffer"
    return 1
  fi
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
    case "${phase}" in
      collect) collect_task "${job}" "${gpu}" "${slot}" || log "ERROR: collect ${job} failed" ;;
      eval)    eval_task ${job//:/ } "${gpu}" "${slot}" || log "ERROR: eval ${job} failed" ;;
    esac
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
if has_phase collect; then
  q="${RUN_DIR}/queue/collect.txt"
  : > "${q}"
  for task in "${TASKS[@]}"; do
    [[ -f "${RUN_DIR}/collect/${task}/buffer.npz" ]] || echo "${task}" >> "${q}"
  done
  run_queue collect || fail=1
fi

# --------------------------------------------------------------------------------------
# Phase 3: merge
# --------------------------------------------------------------------------------------
if has_phase merge; then
  if [[ -f "${MERGED}" ]]; then
    log "merge: ${MERGED} exists, skip"
  else
    inputs=()
    for task in "${TASKS[@]}"; do
      inputs+=("${RUN_DIR}/collect/${task}/buffer.npz")
    done
    log "merge: ${#inputs[@]} buffers -> ${MERGED}"
    cd "${CODE}"
    "${SIM}" scripts/merge_replays.py --out "${MERGED}" "${inputs[@]}" \
      > "${LOCAL_LOG}/merge.log" 2>&1
  fi
fi

# --------------------------------------------------------------------------------------
# Phase 4: shared AC pretrain (replay-only, no server)
# --------------------------------------------------------------------------------------
if has_phase pretrain; then
  if [[ -f "${PRE_DIR}/${PRE_TAG}/agent.pt" ]]; then
    log "pretrain: agent exists, skip"
  else
    log "pretrain: ${OFFLINE_STEPS} offline steps on ${MERGED} (beta=${BETA_PRETRAIN}, cf_actor_coef=0, cf_ref_conditioned=true)"
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_pretrain_ac.py \
        --scene desk_mug --encoder pick18_shared \
        --episodes 1 --offline-steps "${OFFLINE_STEPS}" \
        --gpu "${GPUS[0]}" --port 9500 --tag "${PRE_TAG}" \
        --set "algorithm=v22_24" \
        --set "init_random_ae=false" \
        --set "token_ae=${AE}" \
        --set "beta=${BETA_PRETRAIN}" \
        --set "cf_actor_coef=0" \
        --set "cf_ref_conditioned=true" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "out_dir=${PRE_DIR}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=true" \
        --set "store_decision_tokens=false" \
        --set "vla_traj=${MERGED}" \
        --set "token_replay=${AE_TOKENS}/*/*.npz" \
        > "${LOCAL_LOG}/${PRE_TAG}.log" 2>&1
    if [[ ! -f "${PRE_DIR}/${PRE_TAG}/agent.pt" ]]; then
      log "ERROR: pretrain produced no agent"
      exit 1
    fi
  fi
fi

# --------------------------------------------------------------------------------------
# Phase 5: shared online (one learner, one VLA server per GPU, collectors cycle tasks)
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
    base_port=9600
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
      log "online: resume from ${episodes_done}/${EPISODES} episodes, ${n_collectors} collectors on ${#GPUS[@]} GPUs, beta=${ONLINE_BETA} cf_actor_coef=0"
    else
      log "online: ${EPISODES} episodes over ${#TASKS[@]} tasks, ${n_collectors} collectors on ${#GPUS[@]} GPUs, beta=${ONLINE_BETA} cf_actor_coef=0"
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
        --set "cf_actor_coef=0" \
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
# Phase 6: per-task paired eval with the shared agent (queue over GPUs)
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

if has_phase eval; then
  q="${RUN_DIR}/queue/eval.txt"
  : > "${q}"
  for task in "${TASKS[@]}"; do
    for g in on off; do
      echo "${task}:${g}" >> "${q}"
    done
  done
  run_queue eval || fail=1
fi

log "pick18 V24 shared Stage A finished (fail=${fail})"
exit "${fail}"
