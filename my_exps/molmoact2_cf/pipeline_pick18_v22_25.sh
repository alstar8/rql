#!/usr/bin/env bash
# Pick-18 V22_25: the fixed ConsensusFlow routing, all 18 tasks, as a BACKFILL
# scheduler on top of the running pick18_v22_24 sweep.
#
# V22_25 keeps the main idea (distil the Q ensemble into G, deploy V + G) and
# fixes what the v22_24 pick-18 loss analysis showed was broken:
#   1. policy bootstrap  -- TD target bootstraps Q_bar(s', a', t=1) from the EMA
#      actor's guided unroll instead of fresh noise, so Q keeps an action slope;
#   2. consensus distill -- all K heads' common-scale-normalized grads averaged
#      (head disagreement shrinks the target) instead of one random head/row;
#   3. dead-grad gate    -- distill skipped when batch-mean ||grad Q|| < 1e-3;
#   4. trust-gated conflict contraction -- kill = (1 - trust)^p, so only
#      ensemble-confident guidance may oppose the behavior velocity;
#   5. velocity-relative G -- ||G|| = lambda * t * ||v||, lambda = 0.25
#      (cf_guidance_coef), instead of ~2% unit-ball noise.
# Online runs the JOINT method: cf_actor_coef=1 with the strong anchor
# (beta=100, matching pretrain), so V absorbs the Q signal through the
# one-step guided lookahead while G keeps distilling.
#
# Scheduling contract: keep TARGET_JOBS learner processes (v22_24 + v22_25
# run_pretrain_ac/run_train/run_eval) alive at all times. Whenever a v22_24
# task finishes (or a slot is otherwise free), the next v22_25 task starts on
# the least-loaded GPU (cap MAX_PER_GPU). Per task: fresh AC pretrain
# (8000, cf_actor_coef=0, beta=100) on the same runs/pick18 AE + 100-traj VLA
# buffer -> 300 online (cf_actor_coef=1) -> paired held-out eval48
# (guidance on vs off).
#
#   bash pipeline_pick18_v22_25.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SOURCE_DIR="${SOURCE_DIR:-${ROOT}/runs/pick18}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v22_25}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_v22_25_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
TARGET_JOBS="${TARGET_JOBS:-18}"
MAX_PER_GPU="${MAX_PER_GPU:-3}"
POLL_SEC="${POLL_SEC:-120}"
EPISODES="${EPISODES:-300}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
BETA_PRETRAIN="${BETA_PRETRAIN:-100}"
ONLINE_BETA="${ONLINE_BETA:-100}"
GUIDANCE_COEF="${GUIDANCE_COEF:-0.25}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
WARMUP=0

OBJECTS=(remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box)
read -r -a TASKS <<< "${TASKS:-desk_mug kettle ${OBJECTS[*]}}"

mkdir -p "${RUN_DIR}/pids" "${RUN_DIR}/queue" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/scheduler.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

Q="${RUN_DIR}/queue/jobs.txt"
LOCK_DIR="${RUN_DIR}/pids"

log() { echo "[pick18-v22_25 $(date -u +%H:%M:%S)] $*"; }

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
}

task_done() {
  local task="$1"
  [[ -f "${RUN_DIR}/${task}/eval/${task}_v22_25_gOn/result.json" \
     && -f "${RUN_DIR}/${task}/eval/${task}_v22_25_gOff/result.json" ]]
}

# --- the per-task chain: pretrain -> online -> paired eval -------------------

run_task() {
  local task="$1" gpu="$2" slot="$3"
  task_meta "${task}"
  local port=$((9600 + gpu * 10 + slot))
  local dest="${RUN_DIR}/${task}"
  local ae="${dest}/ae/ae_${SCENE}.pt"
  local shared_buf="${dest}/vla_buffer.npz"
  local pre_tag="${task}_v22_25_pretrain"
  local on_tag="${task}_v22_25"
  local pre_dir="${dest}/ac_pretrain"
  local on_dir="${dest}/rl"
  mkdir -p "${pre_dir}" "${on_dir}" "${dest}/eval" "${dest}/pids"

  seed_task "${task}" || return 1

  local agent="${pre_dir}/${pre_tag}/agent.pt"
  local buf="${pre_dir}/${pre_tag}/buffer.npz"
  local on_agent="${on_dir}/${on_tag}/agent.pt"

  # Fresh AC pretrain with the fixed critic: BC + tight-anchor V, reverse-TD
  # critic with the policy bootstrap, consensus-distilled G. cf_actor_coef=0.
  if [[ -f "${agent}" ]]; then
    log "${task}: pretrained agent exists, skip pretrain"
  else
    log "${task}: pretrain gpu=${gpu} port=${port} beta=${BETA_PRETRAIN} cf_actor_coef=0 lambda=${GUIDANCE_COEF}"
    cd "${CODE}"
    env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_pretrain_ac.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --episodes 100 --offline-steps "${OFFLINE_STEPS}" \
        --gpu "${gpu}" --port "${port}" --tag "${pre_tag}" \
        --set "algorithm=v22_25" \
        --set "init_random_ae=false" \
        --set "token_ae=${ae}" \
        --set "beta=${BETA_PRETRAIN}" \
        --set "cf_actor_coef=0" \
        --set "cf_guidance_coef=${GUIDANCE_COEF}" \
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

  # Online: the joint method -- V gets BC + anchor(beta=100) + the one-step
  # guided lookahead (cf_actor_coef=1); G keeps distilling the live ensemble.
  if [[ -f "${on_agent}" ]]; then
    log "${task}: online agent exists, skip train"
  else
    log "${task}: online gpu=${gpu} port=${port} cf_actor_coef=1 beta=${ONLINE_BETA} lambda=${GUIDANCE_COEF}"
    cd "${CODE}"
    env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
      "${SIM}" scripts/run_train.py \
        --scene "${SCENE}" --encoder "${SCENE}" \
        --gpu "${gpu}" --port "${port}" \
        --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
        --tag "${on_tag}" --init-actor "${agent}" \
        --set "algorithm=v22_25" \
        --set "init_random_ae=false" \
        --set "token_ae=${ae}" \
        --set "beta=${ONLINE_BETA}" \
        --set "cf_actor_coef=1" \
        --set "cf_guidance_coef=${GUIDANCE_COEF}" \
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
        > "${LOCAL_LOG}/${on_tag}.log" 2>&1
    if [[ ! -f "${on_agent}" ]]; then
      log "ERROR: ${task} missing online agent"
      return 1
    fi
  fi

  # Paired eval on the same held-out bench: guidance on (trained lambda) vs off.
  for g in on off; do
    local gcoef="-1"
    [[ "${g}" == "off" ]] && gcoef="0"
    local etag="${on_tag}_g${g^}"
    if [[ -f "${dest}/eval/${etag}/result.json" ]]; then
      log "${task}: eval ${g} exists, skip"
      continue
    fi
    log "${task}: eval ${g} gpu=${gpu} port=${port} guidance_coef=${gcoef}"
    cd "${CODE}"
    env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
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
  if task_done "${task}"; then
    log "${task}: done"
    return 0
  fi
  log "ERROR: ${task} eval incomplete"
  return 1
}

# --- accounting ----------------------------------------------------------------

# Learner processes of either sweep, one per active job: "gpu script" lines.
active_jobs() {
  pgrep -af "scripts/run_(train|eval|pretrain_ac)\.py" \
    | grep -E "v22_24|v22_25" \
    | awk '{for(i=1;i<=NF;i++) if($i=="--gpu"){print $(i+1); break}}' || true
}

requeue() {
  local task="$1"
  exec 9>"${Q}.lock"
  flock 9
  echo "${task}" >> "${Q}"
  flock -u 9
  exec 9>&-
}

pop_task() {
  exec 8>"${Q}.lock"
  flock 8
  local job=""
  if [[ -s "${Q}" ]]; then
    job="$(head -n 1 "${Q}")"
    tail -n +2 "${Q}" > "${Q}.tmp"
    mv "${Q}.tmp" "${Q}"
  fi
  flock -u 8
  exec 8>&-
  echo "${job}"
}

# --- scheduler -------------------------------------------------------------------

if [[ ! -f "${Q}" ]]; then
  for task in "${TASKS[@]}"; do
    seed_task "${task}" || exit 1
  done
  : > "${Q}"
  for task in "${TASKS[@]}"; do
    echo "${task}"
  done >> "${Q}"
  log "queued $(grep -c . "${Q}") V22_25 tasks; target ${TARGET_JOBS} jobs on GPUs ${GPUS[*]} (cap ${MAX_PER_GPU}/gpu)"
else
  log "restart: queue has $(grep -c . "${Q}" || echo 0) tasks, resuming"
fi

while true; do
  # Per-GPU job counts from live learner processes of both sweeps.
  declare -A gpu_load=()
  for gpu in "${GPUS[@]}"; do gpu_load[${gpu}]=0; done
  while read -r gpu; do
    [[ -n "${gpu}" ]] && gpu_load[${gpu}]=$(( ${gpu_load[${gpu}]:-0} + 1 ))
  done < <(active_jobs)

  n_live=$(active_jobs | grep -c . || true)
  total=${n_live}
  # v22_25 chains hold a lockfile ("pid gpu") so the between-step gaps (pretrain
  # done, online not yet up) still count as an active job and reserve the GPU.
  for lock in "${LOCK_DIR}"/task_*.lock; do
    [[ -e "${lock}" ]] || continue
    read -r pid gpu < "${lock}"
    if kill -0 "${pid}" 2>/dev/null; then
      task="$(basename "${lock}" .lock)"
      task="${task#task_}"
      if ! pgrep -f "${task}_v22_25" > /dev/null 2>&1; then
        total=$((total + 1))
        gpu_load[${gpu}]=$(( ${gpu_load[${gpu}]:-0} + 1 ))
      fi
    else
      rm -f "${lock}"
    fi
  done

  launched=0
  while (( total < TARGET_JOBS )); do
    task="$(pop_task)"
    [[ -z "${task}" ]] && break
    if task_done "${task}"; then
      log "${task}: already evaluated, dropping from queue"
      continue
    fi
    if [[ -f "${LOCK_DIR}/task_${task}.lock" ]]; then
      continue  # chain already running (should not happen with a clean queue)
    fi
    # least-loaded GPU under the cap
    best=""
    for gpu in "${GPUS[@]}"; do
      (( ${gpu_load[${gpu}]} >= MAX_PER_GPU )) && continue
      if [[ -z "${best}" || ${gpu_load[${gpu}]} -lt ${gpu_load[${best}]} ]]; then
        best="${gpu}"
      fi
    done
    [[ -z "${best}" ]] && { requeue "${task}"; break; }
    slot="${gpu_load[${best}]}"
    gpu_load[${best}]=$((slot + 1))
    attempts=0
    [[ -f "${LOCK_DIR}/task_${task}.attempts" ]] && attempts="$(cat "${LOCK_DIR}/task_${task}.attempts")"
    echo $((attempts + 1)) > "${LOCK_DIR}/task_${task}.attempts"
    (
      ok=0
      run_task "${task}" "${best}" "${slot}" && ok=1
      rm -f "${LOCK_DIR}/task_${task}.lock"
      if (( ! ok )); then
        attempts="$(cat "${LOCK_DIR}/task_${task}.attempts" 2>/dev/null || echo 1)"
        if (( attempts < MAX_ATTEMPTS )) && ! task_done "${task}"; then
          log "${task}: attempt ${attempts} failed, requeueing"
          requeue "${task}"
        else
          log "ERROR: ${task} gave up after ${attempts} attempts"
        fi
      fi
    ) &
    echo $! "${best}" > "${LOCK_DIR}/task_${task}.lock"
    log "launch ${task} on gpu=${best} slot=${slot} (attempt $((attempts + 1)), active $((total + 1))/${TARGET_JOBS})"
    total=$((total + 1))
    launched=$((launched + 1))
    sleep 5  # let the process appear before the next accounting pass
  done

  if [[ ! -s "${Q}" ]]; then
    remaining=0
    for lock in "${LOCK_DIR}"/task_*.lock; do
      [[ -e "${lock}" ]] && remaining=$((remaining + 1))
    done
    if (( remaining == 0 )); then
      log "queue empty and no v22_25 chains active: sweep finished"
      break
    fi
  fi
  (( launched > 0 )) && log "active ${total}/${TARGET_JOBS} (v22_24+v22_25), queue $(grep -c . "${Q}" || echo 0)"
  sleep "${POLL_SEC}"
done

fail=0
for task in "${TASKS[@]}"; do
  task_done "${task}" || { log "MISSING: ${task}"; fail=1; }
done
log "pick18 V22_25 sweep finished (fail=${fail})"
exit "${fail}"
