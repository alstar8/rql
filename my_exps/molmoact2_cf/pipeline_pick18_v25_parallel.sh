#!/usr/bin/env bash
# V25 parallel: two action experts on the Pick-18 pool, gated swap every 50 episodes.
#
# Frozen expert  — HTTP servers, the ã RLT/QUORUM conditions on.
# Learning expert — in-process pi0.5 action head, flow-matches deployed QUORUM chunks.
# Every 50 actor episodes: probe base G-off / gOn / student G-off (2×18). Copy the
# student into the frozen servers only if student G-off > frozen G-off.
# Distill buffer: closed-loop stitch of two gOn 8-step commits; keep the 500 best
# episode trajectories (success first, then shorter successes / longer failures).
# Eval is the student as a pure VLA on the official 1000-episode MolmoPick val.
#
# Reuses Stage A artifacts (no re-collect, no re-pretrain):
#   runs/pick18_v24_shared/ae/ae_pick18_shared.pt
#   runs/pick18_v24_shared/rl/shared_v24_s0/{agent.pt,buffer.npz}
#
# Writes to jovyan — the workspace NFS is full.
#
#   GPUS="0 1 2 3" bash pipeline_pick18_v25_parallel.sh
#   PHASES="online" GPUS="0 1 2 3" bash pipeline_pick18_v25_parallel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
SRC_DIR="${SRC_DIR:-${ROOT}/runs/pick18_v24_shared}"
JOVYAN_RUN="${JOVYAN_RUN:-/home/jovyan/users/staroverov/v25_keep500}"
RUN_DIR="${RUN_DIR:-${JOVYAN_RUN}}"
LOCAL_LOG="${LOCAL_LOG:-${JOVYAN_RUN}/logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
PHASES="${PHASES:-online val1000}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
COLLECTORS_PER_GPU="${COLLECTORS_PER_GPU:-4}"
VLA_SERVERS_PER_GPU="${VLA_SERVERS_PER_GPU:-1}"

EPISODES="${EPISODES:-1800}"
PROBE_EPISODES="${PROBE_EPISODES:-36}"
ONLINE_BETA="${ONLINE_BETA:-1}"
SWAP_EVERY="${SWAP_EVERY:-50}"
DISTILL_GPU="${DISTILL_GPU:-0}"
DISTILL_BATCH="${DISTILL_BATCH:-4}"
DISTILL_LR="${DISTILL_LR:-1e-5}"
ROUND_EVAL_PER_TASK="${ROUND_EVAL_PER_TASK:-2}"

OBJECTS=(remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box)
read -r -a TASKS <<< "${TASKS:-desk_mug kettle ${OBJECTS[*]}}"

ON_DIR="${RUN_DIR}/rl"
ON_TAG="${ON_TAG:-shared_v25_keep500}"
DISTILL_OUT="${RUN_DIR}/distill"
STUDENT_DIR="${RUN_DIR}/student_expert"
FROZEN_DIR="${RUN_DIR}/frozen_expert"
CONTROL_DIR="${RUN_DIR}/student_control"
VAL_DIR="${RUN_DIR}/eval_val1000"
KEEP_BEST="${KEEP_BEST:-500}"

mkdir -p "${RUN_DIR}/pids" "${ON_DIR}" "${DISTILL_OUT}" "${STUDENT_DIR}" "${FROZEN_DIR}" "${CONTROL_DIR}" "${LOCAL_LOG}"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[pick18-v25-parallel $(date -u +%H:%M:%S)] $*"; }
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

AE="${SRC_DIR}/ae/ae_pick18_shared.pt"
AE_TOKENS="${SRC_DIR}/ae/tokens"
STAGEA_AGENT="${SRC_DIR}/rl/shared_v24_s0/agent.pt"
STAGEA_BUFFER="${SRC_DIR}/rl/shared_v24_s0/buffer.npz"

for f in "${AE}" "${STAGEA_AGENT}" "${STAGEA_BUFFER}"; do
  [[ -f "${f}" ]] || { log "ERROR: Stage A artifact missing: ${f}"; exit 1; }
done
[[ -d "${AE_TOKENS}" ]] || { log "ERROR: Stage A token replay dir missing: ${AE_TOKENS}"; exit 1; }

fail=0

if has_phase online; then
  episodes_done=0
  if [[ -f "${ON_DIR}/${ON_TAG}/metrics.jsonl" ]]; then
    episodes_done="$(python3 -c "import json,pathlib,sys
n=0
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    row=json.loads(line)
    if 'success' in row:
        n+=1
print(n)" "${ON_DIR}/${ON_TAG}/metrics.jsonl")"
  fi
  if [[ -f "${ON_DIR}/${ON_TAG}/agent.pt" && "${episodes_done}" -ge "${EPISODES}" ]]; then
    log "online: already finished ${episodes_done} eps, skip"
  else
    vla_ports=(); vla_gpus=(); egl_devices=()
    local_base=9600
    local_i=0
    for local_i in "${!GPUS[@]}"; do
      local_s=0
      for ((local_s = 0; local_s < VLA_SERVERS_PER_GPU; local_s++)); do
        vla_ports+=($((local_base + local_i * 10 + local_s)))
        vla_gpus+=("${GPUS[$local_i]}")
        egl_devices+=("${GPUS[$local_i]}")
      done
    done
    join_by() { local IFS="$1"; shift; echo "$*"; }
    n_collectors=$(( ${#GPUS[@]} * COLLECTORS_PER_GPU ))
    resume_args=()
    if [[ "${episodes_done}" -gt 0 ]]; then
      resume_args+=(--set "resume=true")
      log "online: resume ${episodes_done}/${EPISODES}, swap every ${SWAP_EVERY}"
    else
      log "online: ${EPISODES} eps, ${n_collectors} collectors, ${#vla_ports[@]} frozen servers, student on GPU ${DISTILL_GPU}, swap every ${SWAP_EVERY}"
    fi
    cd "${CODE}"
    set +e
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene desk_mug --encoder pick18_shared \
        --gpu "${GPUS[0]}" --port "${local_base}" \
        --episodes "${EPISODES}" --warmup-episodes 0 \
        --tag "${ON_TAG}" --init-actor "${STAGEA_AGENT}" \
        --set "algorithm=v22_24" \
        --set "init_random_ae=false" \
        --set "token_ae=${AE}" \
        --set "beta=${ONLINE_BETA}" \
        --set "cf_actor_coef=0" \
        --set "cf_ref_conditioned=true" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "seed=0" \
        --set "init_buffer=${STAGEA_BUFFER}" \
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
        --set "distill_out=${DISTILL_OUT}" \
        --set "distill_swap_every=${SWAP_EVERY}" \
        --set "distill_student_dir=${STUDENT_DIR}" \
        --set "distill_control_dir=${CONTROL_DIR}" \
        --set "distill_gpu=${DISTILL_GPU}" \
        --set "distill_batch=${DISTILL_BATCH}" \
        --set "distill_lr=${DISTILL_LR}" \
        --set "distill_keep_best=${KEEP_BEST}" \
        --set "distill_frozen_dir=${FROZEN_DIR}" \
        --set "round_eval_per_task=${ROUND_EVAL_PER_TASK}" \
        "${resume_args[@]}" \
        >> "${LOCAL_LOG}/${ON_TAG}.log" 2>&1
    train_rc=$?
    set -e
    episodes_done=0
    if [[ -f "${ON_DIR}/${ON_TAG}/metrics.jsonl" ]]; then
      episodes_done="$(python3 -c "import json,pathlib,sys
n=0
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if not line.strip():
        continue
    row=json.loads(line)
    if 'success' in row:
        n+=1
print(n)" "${ON_DIR}/${ON_TAG}/metrics.jsonl")"
    fi
    if (( train_rc != 0 )) || [[ ! -f "${ON_DIR}/${ON_TAG}/agent.pt" ]] || (( episodes_done < EPISODES )); then
      log "ERROR: online stopped at ${episodes_done}/${EPISODES} rc=${train_rc} (see ${LOCAL_LOG}/${ON_TAG}.log)"
      fail=1
    fi
  fi
fi

if has_phase val1000 && (( fail == 0 )); then
  if [[ ! -f "${STUDENT_DIR}/model.safetensors" ]]; then
    log "ERROR: val1000 needs ${STUDENT_DIR}/model.safetensors"
    fail=1
  else
    mkdir -p "${VAL_DIR}"
    if [[ ! -e "${VAL_DIR}/shards" && -d "${SRC_DIR}/eval_val1000/shards" ]]; then
      ln -sfn "${SRC_DIR}/eval_val1000/shards" "${VAL_DIR}/shards"
      log "val1000: reusing Stage A shards"
    fi
    log "val1000: student expert as pure VLA"
    cd "${ROOT}"
    if ! env RUN_DIR="${RUN_DIR}" EVAL_DIR="${VAL_DIR}" CKPT="${STUDENT_DIR}" \
        LOCAL_LOG="${LOCAL_LOG}" PHASES="vla" GPUS="${GPUS[*]}" \
        bash pipeline_eval_val1000.sh; then
      log "ERROR: val1000 failed"
      fail=1
    fi
  fi
fi

log "pick18 V25 parallel finished (fail=${fail})"
exit "${fail}"
