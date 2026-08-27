#!/usr/bin/env bash
# 18-task Pick sweep: per-task RL Token AE + V21 and cf_ae actors.
#
# For each task:
#   1. Train a scene-specific token AE on the frozen-pi0.5 collect tokens
#      (mug/kettle reuse the already-trained AEs).
#   2. Offline AC pretrain (8000 steps) then 300-ep online + held-out eval.
#      V21 = one-pass Gaussian (rl_token). cf_ae = V = v_pi05_base + G with
#      AE finetune. Same recipe as the mug comparison (beta=1, gate=0).
#
# Collect did not dump replay shards, so the first AC pretrain per task
# re-rolls 100 frozen-VLA train episodes (actor off) and writes buffer.npz.
# The other method loads that buffer (no second collect). Mug/kettle reuse
# their existing buffers.
#
#   GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 bash pipeline_pick18_v21_cfae.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
EPISODES="${EPISODES:-300}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
BETA="${BETA:-1}"
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

log() { echo "[pick18 $(date -u +%H:%M:%S)] $*"; }

task_meta() {
  # sets SCENE HORIZON TOKENS AE_SRC BUF_SRC (globals; used by callers)
  case "$1" in
    desk_mug)
      SCENE=desk_mug HORIZON=500
      TOKENS="${ROOT}/runs/beta1_1gpu/desk_mug/tokens/desk_mug"
      AE_SRC="${ROOT}/runs/beta1_from_scratch/desk_mug/ae/ae_desk_mug.pt"
      BUF_SRC="${ROOT}/runs/beta1_jitter_ac/desk_mug/ac_pretrain/desk_mug_ac_pretrain/buffer.npz"
      ;;
    kettle)
      SCENE=kettle HORIZON=400
      TOKENS="${ROOT}/runs/beta1_jitter_ac/kettle/tokens/kettle"
      AE_SRC="${ROOT}/runs/beta1_jitter_ac/kettle/ae/ae_kettle.pt"
      BUF_SRC="${ROOT}/runs/beta1_jitter_ac/kettle/ac_pretrain/kettle_ac_pretrain/buffer.npz"
      ;;
    *)
      SCENE="${task}" HORIZON=500
      TOKENS="${ROOT}/runs/pick_objects/${task}/tokens/${task}"
      AE_SRC="" BUF_SRC=""
      ;;
  esac
}

ensure_ae() {
  local task="$1" gpu="$2"
  task_meta "${task}"
  local dest="${RUN_DIR}/${task}/ae/ae_${SCENE}.pt"
  mkdir -p "${RUN_DIR}/${task}/ae"
  local lock="${RUN_DIR}/${task}/ae.lock"
  exec 9>"${lock}"
  flock 9
  if [[ -f "${dest}" ]]; then
    flock -u 9
    return 0
  fi
  if [[ -n "${AE_SRC}" && -f "${AE_SRC}" ]]; then
    log "${task}: reuse AE ${AE_SRC}"
    cp -n "${AE_SRC}" "${dest}"
    flock -u 9
    return 0
  fi
  shopt -s nullglob
  local shards=("${TOKENS}"/*.npz)
  shopt -u nullglob
  if (( ${#shards[@]} == 0 )); then
    log "ERROR: ${task} no token shards in ${TOKENS}"
    flock -u 9
    return 1
  fi
  log "${task}: train AE gpu=${gpu} shards=${#shards[@]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${SIM}" -m rlt.train_token_ae \
    --token_replay "${TOKENS}/*.npz" \
    --out "${dest}" \
    --device cuda:0 \
    --steps 8000 \
    --max_sequences 6000 \
    > "${LOCAL_LOG}/ae_${task}.log" 2>&1
  if [[ ! -f "${dest}" ]]; then
    log "ERROR: ${task} AE missing after train"
    flock -u 9
    return 1
  fi
  flock -u 9
}

wait_for_buffer() {
  local path="$1" task="$2"
  local n=0
  while [[ ! -f "${path}" ]]; do
    if (( n % 12 == 0 )); then
      log "${task}: waiting for shared VLA buffer"
    fi
    sleep 10
    n=$((n + 1))
    if (( n > 720 )); then
      log "ERROR: ${task} buffer never appeared (${path})"
      return 1
    fi
  done
}

run_method() {
  local task="$1" method="$2" gpu="$3" slot="$4"
  task_meta "${task}"
  local port=$((9000 + gpu * 10 + slot))
  local dest="${RUN_DIR}/${task}"
  local ae="${dest}/ae/ae_${SCENE}.pt"
  local shared_buf="${dest}/vla_buffer.npz"
  local pre_tag="${task}_${method}_pretrain"
  local on_tag="${task}_${method}_s0"
  local pre_dir="${dest}/ac_pretrain"
  local on_dir="${dest}/rl"
  mkdir -p "${pre_dir}" "${on_dir}" "${dest}/eval" "${dest}/pids"

  if [[ -f "${dest}/eval/${on_tag}_actor/result.json" ]]; then
    log "${task}/${method}: already evaluated, skip"
    return 0
  fi

  ensure_ae "${task}" "${gpu}" || return 1

  if [[ -n "${BUF_SRC}" && -f "${BUF_SRC}" && ! -f "${shared_buf}" ]]; then
    cp -n "${BUF_SRC}" "${shared_buf}" 2>/dev/null || true
  fi

  local algo compose ae_ft actor_coef_off actor_coef_on freeze
  case "${method}" in
    v21)
      algo=rl_token compose=false ae_ft=false
      actor_coef_off=0 actor_coef_on=1 freeze=false
      ;;
    cf_ae)
      algo=flow_rlt compose=true ae_ft=true
      actor_coef_off=0 actor_coef_on=1 freeze=false
      ;;
    *) log "unknown method ${method}"; return 1 ;;
  esac

  local agent="${pre_dir}/${pre_tag}/agent.pt"
  local buf="${pre_dir}/${pre_tag}/buffer.npz"
  local on_agent="${on_dir}/${on_tag}/agent.pt"

  # First method on a task without a buffer does the VLA collect; the other loads it.
  local vla_flag=()
  if [[ -f "${shared_buf}" ]]; then
    vla_flag+=(--set "vla_traj=${shared_buf}")
  elif [[ "${method}" == "cf_ae" ]]; then
    wait_for_buffer "${shared_buf}" "${task}" || return 1
    vla_flag+=(--set "vla_traj=${shared_buf}")
  fi

  if [[ -f "${agent}" ]]; then
    log "${task}/${method}: pretrained agent exists, skip pretrain"
  else
    log "${task}/${method}: pretrain gpu=${gpu} port=${port} compose=${compose} replay=${vla_flag[*]:-collect}"
    cd "${CODE}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_pretrain_ac.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --episodes 100 --offline-steps "${OFFLINE_STEPS}" \
        --gpu "${gpu}" --port "${port}" --tag "${pre_tag}" \
        --set "algorithm=${algo}" \
        --set "token_ae=${ae}" \
        --set "beta=${BETA}" \
        --set "gate_step=0" \
        --set "flow_actor_coef=${actor_coef_off}" \
        --set "flow_compose=${compose}" \
        --set "flow_freeze_critic_online=${freeze}" \
        --set "horizon=${HORIZON}" \
        --set "episode_pool=0-47" \
        --set "out_dir=${pre_dir}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        --set "ae_finetune=false" \
        "${vla_flag[@]}" \
        > "${LOCAL_LOG}/${pre_tag}.log" 2>&1
    if [[ ! -f "${agent}" ]]; then
      log "ERROR: ${task}/${method} missing pretrained agent"
      return 1
    fi
  fi
  if [[ -f "${buf}" && ! -f "${shared_buf}" ]]; then
    cp -n "${buf}" "${shared_buf}" 2>/dev/null || true
    log "${task}: published shared VLA buffer"
  fi

  local ae_online_sets=(--set "ae_finetune=false")
  if [[ "${ae_ft}" == "true" ]]; then
    ae_online_sets=(--set "ae_finetune=true" --set "store_decision_tokens=false" --set "token_replay=${TOKENS}/*.npz")
  fi

  if [[ -f "${on_agent}" ]]; then
    log "${task}/${method}: online agent exists, skip train"
  else
    log "${task}/${method}: online gpu=${gpu} port=${port}"
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --gpu "${gpu}" --port "${port}" \
        --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
        --tag "${on_tag}" --init-actor "${agent}" \
        --set "algorithm=${algo}" \
        --set "token_ae=${ae}" \
        --set "beta=${BETA}" \
        --set "gate_step=0" \
        --set "flow_actor_coef=${actor_coef_on}" \
        --set "flow_compose=${compose}" \
        --set "flow_freeze_critic_online=${freeze}" \
        --set "horizon=${HORIZON}" \
        --set "seed=0" \
        --set "init_buffer=${buf}" \
        --set "episode_pool=0-11" \
        --set "out_dir=${on_dir}" \
        --set "checkpoint=${CKPT}" \
        --set "train_token_offline=false" \
        --set "train_token_online=false" \
        "${ae_online_sets[@]}" \
        > "${LOCAL_LOG}/${on_tag}.log" 2>&1
    if [[ ! -f "${on_agent}" ]]; then
      log "ERROR: ${task}/${method} missing online agent"
      return 1
    fi
  fi

  log "${task}/${method}: eval gpu=${gpu} port=${port}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene "${SCENE}" --episodes 16 --gpu "${gpu}" --port "${port}" \
      --actor "${on_agent}" --token-ae "${ae}" \
      --tag "${on_tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "gate_step=0" \
      --set "out_dir=${dest}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${on_tag}_actor.log" 2>&1
  log "${task}/${method}: done"
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
    if [[ "${job}" == ae:* ]]; then
      ensure_ae "${job#ae:}" "${gpu}" || log "ERROR: ${job} failed"
    else
      local method="${job%%:*}"
      local task="${job#*:}"
      if ! run_method "${task}" "${method}" "${gpu}" "${slot}"; then
        log "ERROR: ${job} failed on gpu=${gpu}"
      fi
    fi
  done
}

# ---- queue: mug/kettle methods can start immediately; AEs then the 16×2 sweep ----
Q="${RUN_DIR}/queue/jobs.txt"
if [[ "${SKIP_ENQUEUE:-0}" != "1" ]]; then
  : > "${Q}"
  {
    for task in "${TASKS[@]}"; do
      if [[ "${task}" == desk_mug || "${task}" == kettle ]]; then
        echo "v21:${task}"
        echo "cf_ae:${task}"
      elif [[ "${SKIP_AE:-0}" != "1" ]]; then
        echo "ae:${task}"
      fi
    done
    for task in "${TASKS[@]}"; do
      if [[ "${task}" != desk_mug && "${task}" != kettle ]]; then
        echo "v21:${task}"
      fi
    done
    for task in "${TASKS[@]}"; do
      if [[ "${task}" != desk_mug && "${task}" != kettle ]]; then
        echo "cf_ae:${task}"
      fi
    done
  } >> "${Q}"
  log "queued $(grep -c . "${Q}") jobs on ${#GPUS[@]} GPUs × ${SLOTS_PER_GPU} slots"
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
log "pick18 sweep finished (fail=${fail})"
exit "${fail}"
