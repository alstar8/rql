#!/usr/bin/env bash
# V25 fixed-anchor replication of sequential E2.
#
# The original pi0.5 action expert stays the QUORUM reference for all 600
# episodes. Nothing is copied into it. Collectors record greedy gOn chunks.
# The stitch tail stays in the npz for analysis; offline distillation trains
# only steps [0, 8). Shards are not ranked or deleted, so every task remains.
# After collection, one offline distill from the original checkpoint, then
# val-1000 of that student as a pure VLA.
#
#   GPUS="0 1 2 3" bash pipeline_pick18_v25_fixed600.sh
#   PHASES="distill val1000" bash pipeline_pick18_v25_fixed600.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
SRC_DIR="${SRC_DIR:-${ROOT}/runs/pick18_v24_shared}"
JOVYAN_RUN="${JOVYAN_RUN:-/home/jovyan/users/staroverov/v25_fixed600}"
RUN_DIR="${RUN_DIR:-${JOVYAN_RUN}}"
LOCAL_LOG="${LOCAL_LOG:-${JOVYAN_RUN}/logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
TORCH_PY="${TORCH_PY:-/home/jovyan/users/staroverov/pi05_molmospaces/venvs/pi05_torch/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
PHASES="${PHASES:-online distill val1000}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-3}"
COLLECTORS_PER_GPU="${COLLECTORS_PER_GPU:-4}"
VLA_SERVERS_PER_GPU="${VLA_SERVERS_PER_GPU:-1}"

EPISODES="${EPISODES:-600}"
ONLINE_BETA="${ONLINE_BETA:-1}"
DISTILL_STEPS="${DISTILL_STEPS:-12000}"
DISTILL_BATCH="${DISTILL_BATCH:-8}"
DISTILL_LR="${DISTILL_LR:-1e-5}"
SUPERVISE_STEPS="${SUPERVISE_STEPS:-8}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.1}"

OBJECTS=(remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box)
read -r -a TASKS <<< "${TASKS:-desk_mug kettle ${OBJECTS[*]}}"

ON_DIR="${RUN_DIR}/rl"
ON_TAG="${ON_TAG:-shared_v25_fixed600}"
DISTILL_OUT="${RUN_DIR}/distill"
STUDENT_DIR="${STUDENT_DIR:-${RUN_DIR}/student_expert}"
VAL_DIR="${VAL_DIR:-${RUN_DIR}/eval_val1000}"

mkdir -p "${RUN_DIR}/pids" "${ON_DIR}" "${DISTILL_OUT}" "${STUDENT_DIR}" "${LOCAL_LOG}"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[pick18-v25-fixed600 $(date -u +%H:%M:%S)] $*"; }
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

count_episodes() {
  python3 -c "import json,pathlib,sys
n=0
p=pathlib.Path(sys.argv[1])
if not p.exists():
    print(0); raise SystemExit
for line in p.read_text().splitlines():
    if not line.strip():
        continue
    row=json.loads(line)
    if 'success' in row:
        n+=1
print(n)" "$1"
}

AE="${SRC_DIR}/ae/ae_pick18_shared.pt"
AE_TOKENS="${SRC_DIR}/ae/tokens"
STAGEA_AGENT="${SRC_DIR}/rl/shared_v24_s0/agent.pt"
STAGEA_BUFFER="${SRC_DIR}/rl/shared_v24_s0/buffer.npz"
INIT_ACTOR="${INIT_ACTOR:-${STAGEA_AGENT}}"
INIT_BUFFER="${INIT_BUFFER:-${STAGEA_BUFFER}}"

for f in "${AE}" "${STAGEA_AGENT}" "${STAGEA_BUFFER}"; do
  [[ -f "${f}" ]] || { log "ERROR: Stage A artifact missing: ${f}"; exit 1; }
done
[[ -d "${AE_TOKENS}" ]] || { log "ERROR: Stage A token replay dir missing: ${AE_TOKENS}"; exit 1; }

fail=0

if has_phase online; then
  episodes_done=0
  if [[ -f "${ON_DIR}/${ON_TAG}/metrics.jsonl" ]]; then
    episodes_done="$(count_episodes "${ON_DIR}/${ON_TAG}/metrics.jsonl")"
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
      log "online: resume ${episodes_done}/${EPISODES} on frozen ${CKPT}"
    else
      log "online: ${EPISODES} eps, ${n_collectors} collectors, ${#vla_ports[@]} frozen servers, no expert copy"
    fi
    cd "${CODE}"
    set +e
    setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene desk_mug --encoder pick18_shared \
        --gpu "${GPUS[0]}" --port "${local_base}" \
        --episodes "${EPISODES}" --warmup-episodes 0 \
        --tag "${ON_TAG}" --init-actor "${INIT_ACTOR}" \
        --set "algorithm=v22_24" \
        --set "init_random_ae=false" \
        --set "token_ae=${AE}" \
        --set "beta=${ONLINE_BETA}" \
        --set "cf_actor_coef=0" \
        --set "cf_ref_conditioned=true" \
        --set "gate_step=0" \
        --set "gate_frac=0" \
        --set "seed=0" \
        --set "init_buffer=${INIT_BUFFER}" \
        --set "task_pool=$(task_pool_spec)" \
        --set "episode_pool=0-11" \
        --set "probe_episodes=0" \
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
        --set "distill_swap_every=0" \
        "${resume_args[@]}" \
        >> "${LOCAL_LOG}/${ON_TAG}.log" 2>&1
    train_rc=$?
    set -e
    episodes_done=0
    if [[ -f "${ON_DIR}/${ON_TAG}/metrics.jsonl" ]]; then
      episodes_done="$(count_episodes "${ON_DIR}/${ON_TAG}/metrics.jsonl")"
    fi
    if (( train_rc != 0 )) || [[ ! -f "${ON_DIR}/${ON_TAG}/agent.pt" ]] || (( episodes_done < EPISODES )); then
      log "ERROR: online stopped at ${episodes_done}/${EPISODES} rc=${train_rc} (see ${LOCAL_LOG}/${ON_TAG}.log)"
      fail=1
    else
      log "online: ${episodes_done} episodes recorded under ${DISTILL_OUT}"
    fi
  fi
fi

if has_phase distill && (( fail == 0 )); then
  if [[ -f "${STUDENT_DIR}/distill_summary.json" && -f "${STUDENT_DIR}/model.safetensors" ]]; then
    log "distill: ${STUDENT_DIR}/distill_summary.json exists, skip"
  else
    n_shards="$(find "${DISTILL_OUT}" -name 'distill_*.npz' | wc -l)"
    if (( n_shards < 18 )); then
      log "ERROR: distill dir has ${n_shards} shards, expected one episode file per task at least"
      fail=1
    else
      log "distill: ${n_shards} shards, supervise ${SUPERVISE_STEPS} steps, holdout ${HOLDOUT_FRAC}, budget ${DISTILL_STEPS} steps"
      cd "${CODE}"
      set +e
      env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" HF_HOME="${HF_HOME}" HF_HUB_OFFLINE=1 \
        CUDA_VISIBLE_DEVICES="${GPUS[0]}" TORCHINDUCTOR_CUDAGRAPHS=0 \
        "${TORCH_PY}" scripts/train_expert_distill.py \
        --data "${DISTILL_OUT}" \
        --init "${CKPT}" \
        --out "${STUDENT_DIR}" \
        --steps "${DISTILL_STEPS}" \
        --batch-size "${DISTILL_BATCH}" \
        --lr "${DISTILL_LR}" \
        --supervise-steps "${SUPERVISE_STEPS}" \
        --holdout-frac "${HOLDOUT_FRAC}" \
        --eval-every 1000 \
        --patience 3 \
        --min-steps 2000 \
        --self-check-batches 32 \
        >> "${LOCAL_LOG}/distill.log" 2>&1
      distill_rc=$?
      set -e
      if (( distill_rc != 0 )) || [[ ! -f "${STUDENT_DIR}/model.safetensors" ]]; then
        log "ERROR: distill failed rc=${distill_rc} (see ${LOCAL_LOG}/distill.log)"
        fail=1
      else
        log "distill: wrote ${STUDENT_DIR}"
      fi
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
    log "val1000: distilled expert as pure VLA"
    cd "${ROOT}"
    if ! env RUN_DIR="${RUN_DIR}" EVAL_DIR="${VAL_DIR}" CKPT="${STUDENT_DIR}" \
        LOCAL_LOG="${LOCAL_LOG}" PHASES="vla" GPUS="${GPUS[*]}" \
        bash pipeline_eval_val1000.sh; then
      log "ERROR: val1000 failed"
      fail=1
    fi
  fi
fi

log "pick18 V25 fixed600 finished (fail=${fail})"
exit "${fail}"
