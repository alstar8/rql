#!/usr/bin/env bash
# Controlled V14: eight fixed arms with strict provenance and ownership checks.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v14_controlled}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v14_controlled_logs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
RESIDUAL_CKPT="${RESIDUAL_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v14_controlled}"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts_v14_controlled}"
POLL_SEC="${POLL_SEC:-60}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"
SERVER_STAGGER_SEC="${SERVER_STAGGER_SEC:-3}"
TRAINER_STAGGER_SEC="${TRAINER_STAGGER_SEC:-8}"
FRESH="${FRESH:-0}"
V14_MODE="${V14_MODE:-full}"

case "${V14_MODE}" in
  full)
    V14_MAX_VALID_EPISODES="${V14_MAX_VALID_EPISODES:-400}"
    V14_TARGET_ENV_STEPS="${V14_TARGET_ENV_STEPS:-250000}"
    V14_SNAPSHOT_EPISODES="${V14_SNAPSHOT_EPISODES:-0,100,200,400}"
    if [[ "${V14_MAX_VALID_EPISODES}" != "400" ]]; then
      echo "[v14] full mode requires V14_MAX_VALID_EPISODES=400" >&2
      exit 1
    fi
    ;;
  smoke)
    V14_MAX_VALID_EPISODES="${V14_MAX_VALID_EPISODES:-2}"
    V14_TARGET_ENV_STEPS="${V14_TARGET_ENV_STEPS:-1000}"
    V14_SNAPSHOT_EPISODES="${V14_SNAPSHOT_EPISODES:-0,${V14_MAX_VALID_EPISODES}}"
    ;;
  *)
    echo "[v14] V14_MODE must be full or smoke, got ${V14_MODE}" >&2
    exit 1
    ;;
esac
V14_AE_BATCH_SIZE="${V14_AE_BATCH_SIZE:-16}"
V14_AE_MICROBATCH_SIZE="${V14_AE_MICROBATCH_SIZE:-4}"
V14_AE_MIN_SUCCESS_EPISODES="${V14_AE_MIN_SUCCESS_EPISODES:-3}"
V14_MAX_UPDATE_SEC_PER_EPISODE="${V14_MAX_UPDATE_SEC_PER_EPISODE:-30}"

MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
SERVE_PYTHON="${SERVE_PYTHON:-${MOLMOACT2}/.venv/bin/python}"
HELPER="${ROOT}/v14_harness.py"
TRAIN_SCRIPT="${ROOT}/train_rlt_online.py"

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT}/${RUN_DIR}"
fi
if [[ "${LOCAL_LOG_DIR}" != /* ]]; then
  LOCAL_LOG_DIR="${ROOT}/${LOCAL_LOG_DIR}"
fi
if [[ "${BENCHMARK_ROOT}" != /* ]]; then
  BENCHMARK_ROOT="${ROOT}/${BENCHMARK_ROOT}"
fi
RUN_DIR="${RUN_DIR%/}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR%/}"
BENCHMARK_ROOT="${BENCHMARK_ROOT%/}"
if [[ "$(basename "${RUN_DIR}")" != "rlt_cf_v14_controlled" ]]; then
  echo "[v14] RUN_DIR basename must be rlt_cf_v14_controlled: ${RUN_DIR}" >&2
  exit 1
fi
if [[ "$(basename "${LOCAL_LOG_DIR}")" != "rlt_cf_v14_controlled_logs" ]]; then
  echo "[v14] LOCAL_LOG_DIR basename must be rlt_cf_v14_controlled_logs: ${LOCAL_LOG_DIR}" >&2
  exit 1
fi
case "${RUN_DIR}" in
  *rlt_cf_v13_controlled*)
    echo "[v14] refusing V13 output path: ${RUN_DIR}" >&2
    exit 1
    ;;
esac

TRAIN_BENCHMARK="${BENCHMARK_ROOT}/train"
VAL_BENCHMARK="${BENCHMARK_ROOT}/val"
MANIFEST="${RUN_DIR}/MANIFEST.json"
SERVE_PREFIX=("${SERVE_PYTHON}")

export B1K_ROOT B1K_TMP RUN_DIR LOCAL_LOG_DIR BENCHMARK_ROOT
export RESIDUAL_CKPT FLOW_CKPT RLT_EGL_LOCK_DIR TMP_ROLLOUT_DIR
export RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-8}"
export RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-1}"
export RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}"
export RLT_IO_RETRY_ATTEMPTS="${RLT_IO_RETRY_ATTEMPTS:-5}"
export RLT_IO_RETRY_BASE_SEC="${RLT_IO_RETRY_BASE_SEC:-1.0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export FRESH POLL_SEC SERVER_WAIT_ATTEMPTS SERVER_STAGGER_SEC TRAINER_STAGGER_SEC
export V14_MODE V14_MAX_VALID_EPISODES V14_TARGET_ENV_STEPS
export V14_SNAPSHOT_EPISODES V14_AE_BATCH_SIZE V14_AE_MICROBATCH_SIZE
export V14_AE_MIN_SUCCESS_EPISODES
export V14_MAX_UPDATE_SEC_PER_EPISODE

if [[ ! "${FRESH}" =~ ^[01]$ ]]; then
  echo "[v14] FRESH must be 0 or 1, got ${FRESH}" >&2
  exit 1
fi
for positive_integer in \
  "${V14_MAX_VALID_EPISODES}" \
  "${V14_TARGET_ENV_STEPS}" \
  "${V14_AE_BATCH_SIZE}" \
  "${V14_AE_MICROBATCH_SIZE}" \
  "${V14_AE_MIN_SUCCESS_EPISODES}" \
  "${POLL_SEC}" \
  "${SERVER_WAIT_ATTEMPTS}"; do
  if [[ ! "${positive_integer}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[v14] expected a positive integer, got ${positive_integer}" >&2
    exit 1
  fi
done
if (( V14_AE_BATCH_SIZE < 2 )); then
  echo "[v14] V14_AE_BATCH_SIZE must be at least 2" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "[v14] MolmoSpaces Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${SERVE_PYTHON}" ]]; then
  echo "[v14] explicit MolmoAct2 server Python is not executable: ${SERVE_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${HELPER}" || ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[v14] helper or trainer source is missing" >&2
  exit 1
fi

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_ARRAY=(0 1 2 3 4 5 6 7)
fi
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "[v14] exactly eight physical GPU IDs are required; got ${#GPU_ARRAY[@]}" >&2
  exit 1
fi
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARRAY[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "[v14] physical GPU IDs must be numeric, got ${gpu}" >&2
    exit 1
  fi
  if [[ -n "${SEEN_GPUS[${gpu}]:-}" ]]; then
    echo "[v14] duplicate physical GPU ID: ${gpu}" >&2
    exit 1
  fi
  SEEN_GPUS["${gpu}"]=1
done
GPU_IDS_CSV="$(IFS=,; echo "${GPU_ARRAY[*]}")"
export GPU_IDS="${GPU_IDS_CSV}"

mapfile -t VARIANT_ROWS < <("${PYTHON}" "${HELPER}" variants --format tsv)
if (( ${#VARIANT_ROWS[@]} != 8 )); then
  echo "[v14] expected eight variant rows, got ${#VARIANT_ROWS[@]}" >&2
  exit 1
fi

pid_first_field() {
  local pidfile="$1"
  local pid=""
  [[ -f "${pidfile}" ]] || return 1
  IFS=$' \t\r\n' read -r pid _ < "${pidfile}"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || (( pid <= 1 )); then
    return 1
  fi
  printf '%s\n' "${pid}"
}

pid_belongs_to_run() {
  local pid="$1"
  local environ="/proc/${pid}/environ"
  [[ -r "${environ}" ]] || return 1
  local entry
  while IFS= read -r entry; do
    if [[ "${entry}" == "RLT_CF_V14_RUN_DIR=${RUN_DIR}" ]]; then
      return 0
    fi
  done < <(tr '\0' '\n' < "${environ}")
  return 1
}

pid_is_live_owned() {
  local pidfile="$1"
  local pid
  pid="$(pid_first_field "${pidfile}")" || return 1
  kill -0 "${pid}" 2>/dev/null && pid_belongs_to_run "${pid}"
}

clean_stale_pidfile() {
  local pidfile="$1"
  local pid
  if ! pid="$(pid_first_field "${pidfile}")"; then
    rm -f "${pidfile}"
    return 0
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    rm -f "${pidfile}"
    return 0
  fi
  if ! pid_belongs_to_run "${pid}"; then
    echo "[v14] refusing live unowned PID ${pid} in ${pidfile}" >&2
    return 1
  fi
  return 2
}

assert_no_live_v14_pid() {
  local pidfile
  shopt -s nullglob
  for pidfile in "${RUN_DIR}/pids"/*.pid; do
    if clean_stale_pidfile "${pidfile}"; then
      continue
    else
      local status=$?
      if (( status == 2 )); then
        echo "[v14] live V14 PID recorded in ${pidfile}; refusing duplicate launch" >&2
      fi
      return 1
    fi
  done
  shopt -u nullglob
}

checkpoint_for_kind() {
  local kind="$1"
  case "${kind}" in
    residual) printf '%s\n' "${RESIDUAL_CKPT}" ;;
    flow) printf '%s\n' "${FLOW_CKPT}" ;;
    *)
      echo "[v14] unknown checkpoint kind: ${kind}" >&2
      return 1
      ;;
  esac
}

trainer_command() {
  local variant="$1"
  local fresh_start="$2"
  local -a helper_args=(
    "${PYTHON}" "${HELPER}" train-command
    --variant "${variant}"
    --root "${ROOT}"
    --run-dir "${RUN_DIR}"
    --benchmark-train "${TRAIN_BENCHMARK}"
    --residual-checkpoint "${RESIDUAL_CKPT}"
    --flow-checkpoint "${FLOW_CKPT}"
    --python-executable "${PYTHON}"
    --tmp-rollout-dir "${TMP_ROLLOUT_DIR}"
    --format nul
  )
  if [[ "${fresh_start}" == "1" ]]; then
    helper_args+=(--fresh)
  fi
  mapfile -d '' -t GENERATED_COMMAND < <("${helper_args[@]}")
  if (( ${#GENERATED_COMMAND[@]} == 0 )); then
    echo "[v14] failed to generate trainer command for ${variant}" >&2
    return 1
  fi
}

server_command() {
  local variant="$1"
  local checkpoint="$2"
  local -a helper_args=(
    "${PYTHON}" "${HELPER}" server-command
    --variant "${variant}"
    --root "${ROOT}"
    --checkpoint "${checkpoint}"
    --format nul
  )
  local token
  for token in "${SERVE_PREFIX[@]}"; do
    helper_args+=(--serve-prefix "${token}")
  done
  mapfile -d '' -t GENERATED_COMMAND < <("${helper_args[@]}")
  if (( ${#GENERATED_COMMAND[@]} == 0 )); then
    echo "[v14] failed to generate server command for ${variant}" >&2
    return 1
  fi
}

server_ready() {
  local port="$1"
  local response
  if ! response="$(curl -sf --max-time 3 "http://127.0.0.1:${port}/healthz" 2>/dev/null)"; then
    return 1
  fi
  [[ "${response}" == *'"status":"ok"'* ]]
}

wait_for_server() {
  local variant="$1"
  local port="$2"
  local attempt
  for ((attempt=1; attempt<=SERVER_WAIT_ATTEMPTS; attempt++)); do
    if server_ready "${port}"; then
      return 0
    fi
    sleep 5
  done
  echo "[v14] server ${variant} on port ${port} did not become ready" >&2
  return 1
}

start_server() {
  local variant="$1"
  local gpu="$2"
  local port="$3"
  local checkpoint="$4"
  local pidfile="${RUN_DIR}/pids/server_${variant}.pid"
  local logfile="${LOCAL_LOG_DIR}/server_${variant}_gpu${gpu}.log"
  if pid_is_live_owned "${pidfile}"; then
    return 0
  fi
  local recorded_pid
  if recorded_pid="$(pid_first_field "${pidfile}")" \
    && kill -0 "${recorded_pid}" 2>/dev/null; then
    echo "[watchdog] refusing live unowned PID ${recorded_pid} in ${pidfile}" >&2
    return 1
  fi
  rm -f "${pidfile}"
  server_command "${variant}" "${checkpoint}"
  local -a command=("${GENERATED_COMMAND[@]}")
  echo "[watchdog $(date -Is)] starting server ${variant} gpu=${gpu} port=${port}"
  (
    cd "${MOLMOACT2}"
    exec setsid env \
      RLT_CF_V14_RUN_DIR="${RUN_DIR}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME}" \
      "${command[@]}"
  ) >> "${logfile}" 2>&1 &
  local pid=$!
  printf '%s server variant=%s gpu=%s port=%s\n' \
    "${pid}" "${variant}" "${gpu}" "${port}" > "${pidfile}"
}

start_trainer() {
  local variant="$1"
  local gpu="$2"
  local ae_mode="$3"
  local fresh_start="$4"
  local out_dir="${RUN_DIR}/${variant}"
  local pidfile="${RUN_DIR}/pids/train_${variant}.pid"
  local logfile="${LOCAL_LOG_DIR}/train_${variant}_gpu${gpu}.log"
  if pid_is_live_owned "${pidfile}"; then
    return 0
  fi
  local recorded_pid
  if recorded_pid="$(pid_first_field "${pidfile}")" \
    && kill -0 "${recorded_pid}" 2>/dev/null; then
    echo "[watchdog] refusing live unowned PID ${recorded_pid} in ${pidfile}" >&2
    return 1
  fi
  rm -f "${pidfile}"

  local -a state_args=(
    "${PYTHON}" "${HELPER}" resume-state
    --out-dir "${out_dir}"
  )
  if [[ "${ae_mode}" == "1" ]]; then
    state_args+=(--ae)
  fi
  local state=""
  if ! state="$("${state_args[@]}" 2>/dev/null)"; then
    if [[ "${state}" != "partial" ]]; then
      state="partial"
    fi
  fi
  if [[ "${fresh_start}" != "1" && "${state}" == "partial" ]]; then
    printf 'strict V14 resume blocked at %s for %s\n' \
      "$(date -Is)" "${out_dir}" > "${RUN_DIR}/pids/train_${variant}.resume_blocked"
    echo "[watchdog] strict resume rejected partial/legacy bundle for ${variant}" >&2
    return 2
  fi
  if [[ "${fresh_start}" != "1" && "${state}" == "empty" ]]; then
    echo "[watchdog] refusing a non-fresh start with no bundle for ${variant}" >&2
    return 2
  fi

  trainer_command "${variant}" "${fresh_start}"
  local -a command=("${GENERATED_COMMAND[@]}")
  local -a egl_env=()
  if [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    egl_env=(MUJOCO_EGL_DEVICE_ID="${gpu}")
  fi
  echo "[watchdog $(date -Is)] starting trainer ${variant} gpu=${gpu} fresh=${fresh_start}"
  (
    exec setsid env \
      RLT_CF_V14_RUN_DIR="${RUN_DIR}" \
      RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
      RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
      RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU}" \
      RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      "${egl_env[@]}" \
      HF_HOME="${HF_HOME}" \
      "${command[@]}"
  ) >> "${logfile}" 2>&1 &
  local pid=$!
  printf '%s train variant=%s gpu=%s fresh=%s\n' \
    "${pid}" "${variant}" "${gpu}" "${fresh_start}" > "${pidfile}"
}

run_watchdog() {
  trap 'echo "[watchdog $(date -Is)] stop requested"; exit 0' TERM INT
  export RLT_CF_V14_RUN_DIR="${RUN_DIR}"
  mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"
  printf '%s watchdog run=%s\n' "$$" "${RUN_DIR}" > "${RUN_DIR}/pids/watchdog.pid"
  "${PYTHON}" "${HELPER}" validate-manifest \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN_DIR}" >/dev/null
  "${PYTHON}" "${HELPER}" assert-gpu-ownership \
    --gpu-ids "${GPU_IDS_CSV}" \
    --run-dir "${RUN_DIR}" >/dev/null
  echo "[watchdog $(date -Is)] V14 supervisor started"

  local row variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
    if [[ -n "${port}" ]]; then
      start_server \
        "${variant}" \
        "${GPU_ARRAY[$gpu_index]}" \
        "${port}" \
        "$(checkpoint_for_kind "${checkpoint_kind}")"
      sleep "${SERVER_STAGGER_SEC}"
    fi
  done
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
    if [[ -n "${port}" ]]; then
      wait_for_server "${variant}" "${port}"
    fi
  done

  local initial_fresh=0
  if [[ "${FRESH}" == "1" && ! -f "${RUN_DIR}/.initial_launch_complete" ]]; then
    initial_fresh=1
  fi
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
    start_trainer "${variant}" "${GPU_ARRAY[$gpu_index]}" "${ae_mode}" "${initial_fresh}"
    sleep "${TRAINER_STAGGER_SEC}"
  done
  touch "${RUN_DIR}/.initial_launch_complete"
  echo "[watchdog $(date -Is)] initial V14 process set launched"

  while true; do
    for row in "${VARIANT_ROWS[@]}"; do
      IFS='|' read -r variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
      if [[ -n "${port}" ]] && ! pid_is_live_owned "${RUN_DIR}/pids/server_${variant}.pid"; then
        echo "[watchdog $(date -Is)] server exited: ${variant}" >&2
        start_server \
          "${variant}" \
          "${GPU_ARRAY[$gpu_index]}" \
          "${port}" \
          "$(checkpoint_for_kind "${checkpoint_kind}")"
        wait_for_server "${variant}" "${port}"
      fi
      if pid_is_live_owned "${RUN_DIR}/pids/train_${variant}.pid"; then
        continue
      fi
      rm -f "${RUN_DIR}/pids/train_${variant}.pid"
      if "${PYTHON}" "${HELPER}" training-complete \
        --out-dir "${RUN_DIR}/${variant}" \
        --expected-episodes "${V14_MAX_VALID_EPISODES}" >/dev/null 2>&1; then
        continue
      fi
      echo "[watchdog $(date -Is)] trainer exited before completion: ${variant}" >&2
      start_trainer "${variant}" "${GPU_ARRAY[$gpu_index]}" "${ae_mode}" "0"
      sleep "${TRAINER_STAGGER_SEC}"
    done
    sleep "${POLL_SEC}"
  done
}

if [[ "${1:-}" == "--watchdog" ]]; then
  run_watchdog
  exit 0
fi
if (( $# > 0 )); then
  echo "[v14] unknown argument: $1" >&2
  exit 2
fi

echo "[v14] validating required trainer CLI contract"
"${PYTHON}" "${HELPER}" validate-trainer-cli --train-script "${TRAIN_SCRIPT}"

for required in \
  "${TRAIN_BENCHMARK}/benchmark.json" \
  "${VAL_BENCHMARK}/benchmark.json" \
  "${BENCHMARK_ROOT}/manifest.json" \
  "${RESIDUAL_CKPT}" \
  "${FLOW_CKPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[v14] required artifact is missing: ${required}" >&2
    exit 1
  fi
done

echo "[v14] validating controlled train and validation benchmark"
"${PYTHON}" "${ROOT}/generate_controlled_benchmark.py" \
  --output-root "${BENCHMARK_ROOT}" \
  --validate-only

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"
assert_no_live_v14_pid

for port in 8700 8701 8702 8703 8704 8705; do
  if ! "${PYTHON}" "${HELPER}" port-free --port "${port}" >/dev/null; then
    echo "[v14] required HTTP port is owned by another process: ${port}" >&2
    exit 1
  fi
done
"${PYTHON}" "${HELPER}" assert-gpu-ownership \
  --gpu-ids "${GPU_IDS_CSV}" \
  --run-dir "${RUN_DIR}"

if [[ "${FRESH}" == "1" ]]; then
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant _ <<< "${row}"
    rm -rf "${RUN_DIR:?}/${variant}"
    rm -f "${RUN_DIR}/pids/train_${variant}.resume_blocked"
  done
  rm -rf "${RUN_DIR}/validation"
  rm -f \
    "${MANIFEST}" \
    "${RUN_DIR}/.eval_v14.lock" \
    "${RUN_DIR}/.initial_launch_complete"
  shopt -s nullglob
  rm -f "${LOCAL_LOG_DIR}"/*.log
  shopt -u nullglob
else
  if [[ ! -f "${MANIFEST}" ]]; then
    echo "[v14] FRESH=0 requires the existing V14 manifest: ${MANIFEST}" >&2
    exit 1
  fi
  "${PYTHON}" "${HELPER}" validate-manifest \
    --manifest "${MANIFEST}" \
    --run-dir "${RUN_DIR}"
  manifest_gpu_ids="$("${PYTHON}" "${HELPER}" manifest-gpu-ids --manifest "${MANIFEST}")"
  if [[ "${manifest_gpu_ids}" != "${GPU_IDS_CSV}" ]]; then
    echo "[v14] resume GPU mapping differs from the immutable manifest" >&2
    exit 1
  fi
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
    state_args=(
      "${PYTHON}" "${HELPER}" resume-state
      --out-dir "${RUN_DIR}/${variant}"
    )
    if [[ "${ae_mode}" == "1" ]]; then
      state_args+=(--ae)
    fi
    state=""
    if ! state="$("${state_args[@]}" 2>/dev/null)"; then
      if [[ "${state}" != "partial" ]]; then
        state="partial"
      fi
    fi
    if [[ "${state}" != "complete" ]]; then
      echo "[v14] FRESH=0 requires a strict V14 resume bundle for ${variant}; state=${state}" >&2
      exit 1
    fi
  done
fi

if [[ -e "${RUN_DIR}/logs" && ! -L "${RUN_DIR}/logs" ]]; then
  echo "[v14] ${RUN_DIR}/logs exists and is not a symlink" >&2
  exit 1
fi
ln -sfn "${LOCAL_LOG_DIR}" "${RUN_DIR}/logs"

if [[ "${FRESH}" == "1" ]]; then
  manifest_args=(
    "${PYTHON}" "${HELPER}" manifest
    --output "${MANIFEST}"
    --root "${ROOT}"
    --run-dir "${RUN_DIR}"
    --log-dir "${LOCAL_LOG_DIR}"
    --benchmark-root "${BENCHMARK_ROOT}"
    --residual-checkpoint "${RESIDUAL_CKPT}"
    --flow-checkpoint "${FLOW_CKPT}"
    --python-executable "${PYTHON}"
    --tmp-rollout-dir "${TMP_ROLLOUT_DIR}"
    --egl-lock-dir "${RLT_EGL_LOCK_DIR}"
    --gpu-ids "${GPU_IDS_CSV}"
  )
  for token in "${SERVE_PREFIX[@]}"; do
    manifest_args+=(--serve-prefix "${token}")
  done
  "${manifest_args[@]}"
fi
"${PYTHON}" "${HELPER}" validate-manifest \
  --manifest "${MANIFEST}" \
  --run-dir "${RUN_DIR}"

echo "[v14] launching detached watchdog"
nohup setsid env \
  RLT_CF_V14_RUN_DIR="${RUN_DIR}" \
  bash "${ROOT}/launch_v14_controlled.sh" --watchdog \
  >> "${LOCAL_LOG_DIR}/watchdog.log" 2>&1 < /dev/null &
watchdog_pid=$!
printf '%s watchdog run=%s\n' "${watchdog_pid}" "${RUN_DIR}" > "${RUN_DIR}/pids/watchdog.pid"
sleep 2
if ! kill -0 "${watchdog_pid}" 2>/dev/null; then
  echo "[v14] watchdog failed to start; inspect ${LOCAL_LOG_DIR}/watchdog.log" >&2
  exit 1
fi

echo "[v14] detached watchdog PID ${watchdog_pid}"
echo "[v14] mode: ${V14_MODE}"
echo "[v14] run: ${RUN_DIR}"
echo "[v14] logs: ${LOCAL_LOG_DIR}"
echo "[v14] stop: bash ${ROOT}/stop_run.sh ${RUN_DIR}"
