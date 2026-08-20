#!/usr/bin/env bash
# V20: one pooled-data learner, 32 rollout-only workers, immutable incumbent.
#
# Fresh:
#   GPU_IDS=0,1,2,3,4,5,6,7 FRESH=1 bash launch_v20_rlt_cfgrl.sh
# Resume:
#   FRESH=0 bash launch_v20_rlt_cfgrl.sh

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v20_kettle}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v20_kettle_logs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
BENCHMARK_DIR="${BENCHMARK_DIR:-${BENCHMARK_ROOT}/train}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle_cfgrl_v20}"
OFFLINE_SHARD_ROOT="${OFFLINE_SHARD_ROOT:-${ROOT}/runs/rlt_pretrain_house0_kettle}"
FLOW_CKPT="${FLOW_CKPT:-${PRETRAIN_DIR}/rlt_cf_flow_pretrain_house0_kettle_cfgrl.pt}"
REENCODED_REPLAY="${REENCODED_REPLAY:-${PRETRAIN_DIR}/chunk_replay_reencoded.npz}"
V20_SEED_REPLAY="${V20_SEED_REPLAY:-${RUN_DIR}/seed_replay_provenance.npz}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v20_kettle}"

MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
SERVE_PYTHON="${SERVE_PYTHON:-${MOLMOACT2}/.venv/bin/python}"
SERVER_HELPER="${ROOT}/v19_harness.py"
RUNNER="${ROOT}/v20_runner.py"
MIGRATOR="${ROOT}/merge_chunk_replay_provenance.py"

FRESH="${FRESH:-1}"
BASE_HTTP_PORT="${BASE_HTTP_PORT:-8780}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}"
RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-3}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"
MONITOR_SEC="${MONITOR_SEC:-20}"
WORKER_STAGGER_SEC="${WORKER_STAGGER_SEC:-1}"

V20_MAX_ROUNDS="${V20_MAX_ROUNDS:-100}"
V20_BATCH_SIZE="${V20_BATCH_SIZE:-256}"
V20_COLLECT_WAVES_PER_ROUND="${V20_COLLECT_WAVES_PER_ROUND:-4}"
V20_ROUNDS_PER_OFFLINE="${V20_ROUNDS_PER_OFFLINE:-2}"
V20_AE_STEPS="${V20_AE_STEPS:-512}"
V20_AE_BATCH_SIZE="${V20_AE_BATCH_SIZE:-16}"
V20_ROLLOUT_SEED_ROOT="${V20_ROLLOUT_SEED_ROOT:-20260820}"
# Fixed-method knobs (I2/I3b/I5/I6/I7): CFGRL condition dropout, deployment
# guidance weight, per-wave gradient budget, promotion gate, LoRA accumulation.
V20_CFGRL_DROPOUT="${V20_CFGRL_DROPOUT:-0.1}"
V20_REF_DROPOUT="${V20_REF_DROPOUT:-0.5}"
V20_W_DEPLOY="${V20_W_DEPLOY:-1.0}"
V20_UPDATES_PER_WAVE="${V20_UPDATES_PER_WAVE:-32}"
V20_MAX_UPDATE_SEC_PER_WAVE="${V20_MAX_UPDATE_SEC_PER_WAVE:-120}"
V20_PROMOTION_ALPHA="${V20_PROMOTION_ALPHA:-0.05}"
V20_PROMOTION_MIN_GAIN="${V20_PROMOTION_MIN_GAIN:-0.03}"
V20_CLONE_MSE_MAX="${V20_CLONE_MSE_MAX:-0.02}"
V20_COND_REF_MSE_MAX="${V20_COND_REF_MSE_MAX:-0.5}"
V20_MAX_NORMALIZED_ACTION="${V20_MAX_NORMALIZED_ACTION:-12.0}"
V20_AE_ACCUMULATE="${V20_AE_ACCUMULATE:-1}"

if [[ "${RUN_DIR}" != /* ]]; then RUN_DIR="${ROOT}/${RUN_DIR}"; fi
if [[ "${LOCAL_LOG_DIR}" != /* ]]; then
  LOCAL_LOG_DIR="${ROOT}/${LOCAL_LOG_DIR}"
fi
RUN_DIR="${RUN_DIR%/}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR%/}"

if [[ "$(basename "${RUN_DIR}")" != "rlt_cf_v20_kettle" ]]; then
  echo "[v20] RUN_DIR basename must be rlt_cf_v20_kettle" >&2
  exit 1
fi
for required_file in \
  "${FLOW_CKPT}" \
  "${REENCODED_REPLAY}" \
  "${RUNNER}" \
  "${MIGRATOR}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[v20] missing required file ${required_file}" >&2
    exit 1
  fi
done
if [[ ! -d "${BENCHMARK_DIR}" || ! -d "${OFFLINE_SHARD_ROOT}" ]]; then
  echo "[v20] benchmark/offline shard directory missing" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" || ! -x "${SERVE_PYTHON}" ]]; then
  echo "[v20] Python environment missing" >&2
  exit 1
fi
if [[ "${FRESH}" == "1" && -f "${RUN_DIR}/.v20_monotone_incumbent" ]]; then
  echo "[v20] refusing to overwrite initialized run ${RUN_DIR}" >&2
  exit 1
fi
if [[ "${FRESH}" != "1" && ! -f "${RUN_DIR}/.v20_monotone_incumbent" ]]; then
  echo "[v20] resume requested but run marker is missing" >&2
  exit 1
fi

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
else
  mapfile -t GPU_ARRAY < <(
    nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}'
  )
fi
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS < 1 || NUM_GPUS > 8 )); then
  echo "[v20] need 1-8 GPUs, got ${NUM_GPUS}" >&2
  exit 1
fi
if (( INSTANCES_PER_GPU < 1 || RLT_EGL_PER_GPU < 1 )); then
  echo "[v20] worker/EGL counts must be positive" >&2
  exit 1
fi
WORKER_COUNT=$((NUM_GPUS * INSTANCES_PER_GPU))
LEARNER_GPU="${LEARNER_GPU:-${GPU_ARRAY[0]}}"

mkdir -p \
  "${RUN_DIR}/pids" \
  "${LOCAL_LOG_DIR}" \
  "${RLT_EGL_LOCK_DIR}"
ln -sfn "${LOCAL_LOG_DIR}" "${RUN_DIR}/logs"

if [[ "${FRESH}" == "1" && ! -f "${V20_SEED_REPLAY}" ]]; then
  echo "[v20] migrating provenance-correct seed replay"
  "${PYTHON}" "${MIGRATOR}" \
    --input-root "${OFFLINE_SHARD_ROOT}" \
    --output "${V20_SEED_REPLAY}" \
    --reencoded-replay "${REENCODED_REPLAY}" \
    --pose-cycle 1 \
    --target-pose-idx 0 \
    --expect-target-episodes -1 \
    --expect-target-successes -1 \
    --expect-target-positive-rows -1
fi
if [[ ! -f "${V20_SEED_REPLAY}" ]]; then
  echo "[v20] seed replay missing: ${V20_SEED_REPLAY}" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export RLT_VLA_PREFETCH=0
export RLT_EGL_LOCK_DIR
export RLT_EGL_PER_GPU
export RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-$((NUM_GPUS * RLT_EGL_PER_GPU))}"
export RLT_CF_V20_RUN_DIR="${RUN_DIR}"
export PYTHONUNBUFFERED=1

declare -a SERVER_PIDS=()
declare -a WORKER_PIDS=()
LEARNER_PID=""

cleanup() {
  local pid
  trap - EXIT INT TERM
  for pid in "${WORKER_PIDS[@]:-}" "${LEARNER_PID:-}" "${SERVER_PIDS[@]:-}"; do
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT INT TERM

AE_CKPT_FILE="${RUN_DIR}/checkpoints/ae_trainable_latest.pt"
AE_RELOAD_REQUEST="${RUN_DIR}/coordination/ae_reload_request.json"
AE_RELOAD_STAMP=""

server_command() {
  local -a command=(
    "${PYTHON}" "${SERVER_HELPER}" server-command
    --variant flow_cfgrl
    --root "${ROOT}"
    --checkpoint "${FLOW_CKPT}"
    --format nul
    --serve-prefix "${SERVE_PYTHON}"
  )
  mapfile -d '' -t GENERATED_COMMAND < <("${command[@]}")
  if [[ -f "${AE_CKPT_FILE}" ]]; then
    GENERATED_COMMAND+=(--ae_trainable_ckpt "${AE_CKPT_FILE}")
  fi
}

server_ready() {
  local port="$1" response
  response="$(curl -sf --max-time 3 "http://127.0.0.1:${port}/healthz" 2>/dev/null)" || return 1
  [[ "${response}" == *'"status":"ok"'* ]]
}

start_server() {
  local slot="$1" gpu="${GPU_ARRAY[$1]}" port=$((BASE_HTTP_PORT + slot))
  local logfile="${LOCAL_LOG_DIR}/server_gpu${gpu}_port${port}.log"
  server_command
  GENERATED_COMMAND+=(--port "${port}")
  (
    cd "${MOLMOACT2}"
    exec setsid env \
      RLT_CF_V20_RUN_DIR="${RUN_DIR}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME}" \
      "${GENERATED_COMMAND[@]}"
  ) >> "${logfile}" 2>&1 &
  SERVER_PIDS[$slot]="$!"
  printf '%s\n' "$!" > "${RUN_DIR}/pids/server_${slot}.pid"
}

wait_server() {
  local slot="$1" port=$((BASE_HTTP_PORT + slot)) attempt
  for ((attempt=1; attempt<=SERVER_WAIT_ATTEMPTS; attempt++)); do
    server_ready "${port}" && return 0
    sleep 5
  done
  echo "[v20] server ${slot} on port ${port} failed readiness" >&2
  return 1
}

start_worker() {
  local worker="$1"
  local slot=$((worker / INSTANCES_PER_GPU))
  local gpu="${GPU_ARRAY[$slot]}"
  local port=$((BASE_HTTP_PORT + slot))
  local logfile="${LOCAL_LOG_DIR}/worker_${worker}_gpu${gpu}.log"
  (
    exec setsid env \
      RLT_CF_V20_RUN_DIR="${RUN_DIR}" \
      CUDA_VISIBLE_DEVICES="" \
      MUJOCO_EGL_DEVICE_ID="${gpu}" \
      "${PYTHON}" "${RUNNER}" worker \
      --run_dir "${RUN_DIR}" \
      --worker_count "${WORKER_COUNT}" \
      --worker_id "${worker}" \
      --server_host 127.0.0.1 \
      --server_port "${port}" \
      --benchmark_dir "${BENCHMARK_DIR}" \
      --benchmark_episode_idx 0 \
      --target_pose_idx 0 \
      --val_benchmark_dir "${BENCHMARK_ROOT}/val" \
      --rollout_seed_root "${V20_ROLLOUT_SEED_ROOT}"
  ) >> "${logfile}" 2>&1 &
  WORKER_PIDS[$worker]="$!"
  printf '%s\n' "$!" > "${RUN_DIR}/pids/worker_${worker}.pid"
}

for ((slot=0; slot<NUM_GPUS; slot++)); do
  start_server "${slot}"
  sleep 2
done
for ((slot=0; slot<NUM_GPUS; slot++)); do
  wait_server "${slot}"
done

LEARNER_LOG="${LOCAL_LOG_DIR}/learner_gpu${LEARNER_GPU}.log"
LEARNER_ARGS=(
  "${PYTHON}" "${RUNNER}" learner
  --run_dir "${RUN_DIR}"
  --worker_count "${WORKER_COUNT}"
  --rollout_seed_root "${V20_ROLLOUT_SEED_ROOT}"
  --target_pose_idx 0
  --max_rounds "${V20_MAX_ROUNDS}"
  --collect_waves_per_round "${V20_COLLECT_WAVES_PER_ROUND}"
  --rounds_per_offline "${V20_ROUNDS_PER_OFFLINE}"
  --ae_steps "${V20_AE_STEPS}"
  --ae_batch_size "${V20_AE_BATCH_SIZE}"
  --batch_size "${V20_BATCH_SIZE}"
  --cfgrl_dropout "${V20_CFGRL_DROPOUT}"
  --ref_dropout "${V20_REF_DROPOUT}"
  --w_deploy "${V20_W_DEPLOY}"
  --updates_per_wave "${V20_UPDATES_PER_WAVE}"
  --max_update_sec_per_wave "${V20_MAX_UPDATE_SEC_PER_WAVE}"
  --promotion_alpha "${V20_PROMOTION_ALPHA}"
  --promotion_min_gain "${V20_PROMOTION_MIN_GAIN}"
  --clone_mse_max "${V20_CLONE_MSE_MAX}"
  --cond_ref_mse_max "${V20_COND_REF_MSE_MAX}"
  --max_normalized_action "${V20_MAX_NORMALIZED_ACTION}"
  --base_checkpoint "${FLOW_CKPT}"
  --seed_replay "${V20_SEED_REPLAY}"
  --device cuda
)
if [[ "${V20_AE_ACCUMULATE}" == "1" ]]; then
  LEARNER_ARGS+=(--ae_accumulate)
else
  LEARNER_ARGS+=(--no-ae_accumulate)
fi
(
  exec setsid env \
    RLT_CF_V20_RUN_DIR="${RUN_DIR}" \
    CUDA_VISIBLE_DEVICES="${LEARNER_GPU}" \
    "${LEARNER_ARGS[@]}"
) >> "${LEARNER_LOG}" 2>&1 &
LEARNER_PID="$!"
printf '%s\n' "${LEARNER_PID}" > "${RUN_DIR}/pids/learner.pid"

for ((attempt=1; attempt<=120; attempt++)); do
  [[ -f "${RUN_DIR}/.v20_monotone_incumbent" ]] && break
  kill -0 "${LEARNER_PID}" 2>/dev/null || {
    echo "[v20] learner exited during initialization" >&2
    exit 1
  }
  sleep 2
done
if [[ ! -f "${RUN_DIR}/.v20_monotone_incumbent" ]]; then
  echo "[v20] learner did not initialize run state" >&2
  exit 1
fi

for ((worker=0; worker<WORKER_COUNT; worker++)); do
  start_worker "${worker}"
  sleep "${WORKER_STAGGER_SEC}"
done

echo "[v20] running learner=${LEARNER_PID} workers=${WORKER_COUNT} GPUs=${NUM_GPUS}"
while kill -0 "${LEARNER_PID}" 2>/dev/null; do
  for ((worker=0; worker<WORKER_COUNT; worker++)); do
    if ! kill -0 "${WORKER_PIDS[$worker]}" 2>/dev/null; then
      echo "[v20] restarting worker ${worker}"
      start_worker "${worker}"
    fi
  done
  for ((slot=0; slot<NUM_GPUS; slot++)); do
    if ! kill -0 "${SERVER_PIDS[$slot]}" 2>/dev/null; then
      echo "[v20] restarting server slot ${slot}"
      start_server "${slot}"
      wait_server "${slot}"
    fi
  done
  # Reload servers with the updated AE after an offline phase.
  if [[ -f "${AE_RELOAD_REQUEST}" ]]; then
    stamp="$(stat -c %Y "${AE_RELOAD_REQUEST}" 2>/dev/null || echo '')"
    if [[ -n "${stamp}" && "${stamp}" != "${AE_RELOAD_STAMP}" ]]; then
      echo "[v20] AE reload requested; restarting servers with updated AE"
      for ((slot=0; slot<NUM_GPUS; slot++)); do
        if kill -0 "${SERVER_PIDS[$slot]}" 2>/dev/null; then
          kill -TERM "${SERVER_PIDS[$slot]}" 2>/dev/null || true
        fi
      done
      sleep 5
      for ((slot=0; slot<NUM_GPUS; slot++)); do
        start_server "${slot}"
      done
      for ((slot=0; slot<NUM_GPUS; slot++)); do
        wait_server "${slot}"
      done
      AE_RELOAD_STAMP="${stamp}"
    fi
  fi
  sleep "${MONITOR_SEC}"
done

set +e
wait "${LEARNER_PID}"
LEARNER_RC="$?"
set -e
echo "[v20] learner exited rc=${LEARNER_RC}"
exit "${LEARNER_RC}"
