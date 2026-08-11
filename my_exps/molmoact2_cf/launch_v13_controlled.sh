#!/usr/bin/env bash
# Controlled V13: eight fixed arms, one trainer per GPU, detached watchdog.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v13_controlled}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v13_controlled_logs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
RESIDUAL_CKPT="${RESIDUAL_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v13_controlled}"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts_v13_controlled}"
POLL_SEC="${POLL_SEC:-60}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"
SERVER_STAGGER_SEC="${SERVER_STAGGER_SEC:-3}"
TRAINER_STAGGER_SEC="${TRAINER_STAGGER_SEC:-8}"
FRESH="${FRESH:-0}"

MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
MOLMOACT2_PYTHON="${MOLMOACT2_PYTHON:-${MOLMOACT2}/.venv/bin/python}"
HELPER="${ROOT}/v13_harness.py"

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
TRAIN_BENCHMARK="${BENCHMARK_ROOT}/train"
VAL_BENCHMARK="${BENCHMARK_ROOT}/val"

export B1K_ROOT B1K_TMP RUN_DIR LOCAL_LOG_DIR BENCHMARK_ROOT
export RESIDUAL_CKPT FLOW_CKPT RLT_EGL_LOCK_DIR TMP_ROLLOUT_DIR
export RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-8}"
export RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-1}"
export RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}"
export FRESH

if [[ ! "${FRESH}" =~ ^[01]$ ]]; then
  echo "[v13] FRESH must be 0 or 1, got ${FRESH}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "[v13] MolmoSpaces Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${HELPER}" ]]; then
  echo "[v13] missing helper: ${HELPER}" >&2
  exit 1
fi

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -n "${UV_BIN}" && -x "${UV_BIN}" ]]; then
  SERVE_PREFIX=("${UV_BIN}" run python)
elif [[ -x "${MOLMOACT2_PYTHON}" ]]; then
  SERVE_PREFIX=("${MOLMOACT2_PYTHON}")
else
  echo "[v13] neither uv nor MolmoAct2 Python is available" >&2
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
  echo "[v13] exactly eight physical GPU IDs are required; got ${#GPU_ARRAY[@]}" >&2
  exit 1
fi
GPU_IDS_CSV="$(IFS=,; echo "${GPU_ARRAY[*]}")"
export GPU_IDS="${GPU_IDS_CSV}"

mapfile -t VARIANT_ROWS < <("${PYTHON}" "${HELPER}" variants --format tsv)
if (( ${#VARIANT_ROWS[@]} != 8 )); then
  echo "[v13] expected eight variant rows, got ${#VARIANT_ROWS[@]}" >&2
  exit 1
fi

pid_first_field() {
  local pidfile="$1"
  local pid=""
  [[ -f "${pidfile}" ]] || return 1
  IFS=$' \t\r\n' read -r pid _ < "${pidfile}" || true
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
    if [[ "${entry}" == "RLT_CF_V4_RUN_DIR=${RUN_DIR}" ]]; then
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
    echo "[v13] refusing live PID collision in ${pidfile}: ${pid}" >&2
    return 1
  fi
  return 2
}

assert_no_live_v13_pid() {
  local pidfile
  shopt -s nullglob
  for pidfile in "${RUN_DIR}/pids"/*.pid; do
    if clean_stale_pidfile "${pidfile}"; then
      continue
    else
      local status=$?
      if (( status == 2 )); then
        echo "[v13] live V13 PID recorded in ${pidfile}; refusing duplicate launch" >&2
      fi
      return 1
    fi
  done
}

checkpoint_for_kind() {
  local kind="$1"
  case "${kind}" in
    residual) printf '%s\n' "${RESIDUAL_CKPT}" ;;
    flow) printf '%s\n' "${FLOW_CKPT}" ;;
    *)
      echo "[v13] unknown checkpoint kind: ${kind}" >&2
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
    echo "[v13] failed to generate trainer command for ${variant}" >&2
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
    echo "[v13] failed to generate server command for ${variant}" >&2
    return 1
  fi
}

server_ready() {
  local port="$1"
  local response
  response="$(curl -sf --max-time 3 "http://127.0.0.1:${port}/healthz" 2>/dev/null || true)"
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
  echo "[v13] server ${variant} on port ${port} did not become ready" >&2
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
  rm -f "${pidfile}"
  server_command "${variant}" "${checkpoint}"
  local -a command=("${GENERATED_COMMAND[@]}")
  echo "[watchdog $(date -Is)] starting server ${variant} gpu=${gpu} port=${port}"
  (
    cd "${MOLMOACT2}"
    exec setsid env \
      RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
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
  rm -f "${pidfile}"

  local state
  if ! state="$("${PYTHON}" "${HELPER}" resume-state --out-dir "${out_dir}" $(
    [[ "${ae_mode}" == "1" ]] && printf '%s' "--ae"
  ) 2>/dev/null)"; then
    state="${state:-partial}"
  fi
  if [[ "${fresh_start}" != "1" && "${state}" == "partial" ]]; then
    local blocked="${RUN_DIR}/pids/train_${variant}.resume_blocked"
    if [[ ! -f "${blocked}" ]]; then
      printf 'partial resume detected at %s for %s\n' "$(date -Is)" "${out_dir}" > "${blocked}"
      echo "[watchdog $(date -Is)] refusing partial resume for ${variant}: ${out_dir}"
    fi
    return 2
  fi

  trainer_command "${variant}" "${fresh_start}"
  local -a command=("${GENERATED_COMMAND[@]}")
  echo "[watchdog $(date -Is)] starting trainer ${variant} gpu=${gpu} fresh=${fresh_start}"
  local -a egl_env=()
  if [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    egl_env=(MUJOCO_EGL_DEVICE_ID="${gpu}")
  fi
  (
    exec setsid env \
      RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
      RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
      RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
      RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU}" \
      RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      "${egl_env[@]}" \
      HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
      "${command[@]}"
  ) >> "${logfile}" 2>&1 &
  local pid=$!
  printf '%s train variant=%s gpu=%s fresh=%s\n' \
    "${pid}" "${variant}" "${gpu}" "${fresh_start}" > "${pidfile}"
}

write_readme() {
  if [[ ! -f "${RUN_DIR}/README.md" ]]; then
    cat > "${RUN_DIR}/README.md" <<'EOF'
# V13 controlled house-0 kettle run

Eight GPUs run one trainer each. The first six arms use dedicated HTTP servers
on ports 8700-8705; the two Molmo AE LoRA arms run their model in-process.
Training cycles only controlled train indices 0-23 for 400 valid episodes.
Immutable bundles are written at episodes 0, 100, 200, and 400.

Initial launch:

```bash
FRESH=1 bash launch_v13_controlled.sh
```

Safe watchdog relaunch (all eight resume bundles must be complete):

```bash
FRESH=0 bash launch_v13_controlled.sh
```

Evaluation and reporting:

```bash
bash eval_v13_controlled.sh
python plot_v13_controlled.py
python snapshot_run_status.py --run-dir runs/rlt_cf_v13_controlled
bash status_v13_controlled.sh
```

Stop only this run with `bash stop_run.sh runs/rlt_cf_v13_controlled`.
EOF
  fi
  printf -- '- Launch request: %s (FRESH=%s, host=%s)\n' \
    "$(date -Is)" "${FRESH}" "$(hostname)" >> "${RUN_DIR}/README.md"
}

run_watchdog() {
  trap 'echo "[watchdog $(date -Is)] stop requested"; exit 0' TERM INT
  export RLT_CF_V4_RUN_DIR="${RUN_DIR}"
  mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"
  printf '%s watchdog run=%s\n' "$$" "${RUN_DIR}" > "${RUN_DIR}/pids/watchdog.pid"
  echo "[watchdog $(date -Is)] V13 supervisor started"

  local row variant gpu cf_mode actor_mode guide ae_mode checkpoint_kind updates port
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant gpu cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
    if [[ -n "${port}" ]]; then
      start_server \
        "${variant}" \
        "${GPU_ARRAY[$gpu]}" \
        "${port}" \
        "$(checkpoint_for_kind "${checkpoint_kind}")"
      sleep "${SERVER_STAGGER_SEC}"
    fi
  done
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant gpu cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
    if [[ -n "${port}" ]]; then
      wait_for_server "${variant}" "${port}"
    fi
  done

  local initial_fresh=0
  if [[ "${FRESH}" == "1" && ! -f "${RUN_DIR}/.initial_launch_complete" ]]; then
    initial_fresh=1
  fi
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant gpu cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
    start_trainer "${variant}" "${GPU_ARRAY[$gpu]}" "${ae_mode}" "${initial_fresh}" || true
    sleep "${TRAINER_STAGGER_SEC}"
  done
  touch "${RUN_DIR}/.initial_launch_complete"
  echo "[watchdog $(date -Is)] initial process set launched"

  while true; do
    for row in "${VARIANT_ROWS[@]}"; do
      IFS='|' read -r variant gpu cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
      if [[ -n "${port}" ]] && ! pid_is_live_owned "${RUN_DIR}/pids/server_${variant}.pid"; then
        start_server \
          "${variant}" \
          "${GPU_ARRAY[$gpu]}" \
          "${port}" \
          "$(checkpoint_for_kind "${checkpoint_kind}")"
        wait_for_server "${variant}" "${port}" || true
      fi
      if pid_is_live_owned "${RUN_DIR}/pids/train_${variant}.pid"; then
        continue
      fi
      rm -f "${RUN_DIR}/pids/train_${variant}.pid"
      if "${PYTHON}" "${HELPER}" training-complete \
        --out-dir "${RUN_DIR}/${variant}" \
        --expected-episodes 400 >/dev/null 2>&1; then
        continue
      fi
      start_trainer "${variant}" "${GPU_ARRAY[$gpu]}" "${ae_mode}" "0" || true
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
  echo "[v13] unknown argument: $1" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"
assert_no_live_v13_pid

for required in \
  "${TRAIN_BENCHMARK}/benchmark.json" \
  "${VAL_BENCHMARK}/benchmark.json" \
  "${RESIDUAL_CKPT}" \
  "${FLOW_CKPT}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[v13] required artifact is missing: ${required}" >&2
    exit 1
  fi
done

echo "[v13] validating controlled train and validation benchmark"
"${PYTHON}" "${ROOT}/generate_controlled_benchmark.py" \
  --output-root "${BENCHMARK_ROOT}" \
  --validate-only

if [[ "${FRESH}" == "1" ]]; then
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant _ <<< "${row}"
    rm -rf "${RUN_DIR:?}/${variant}"
    rm -f "${RUN_DIR}/pids/train_${variant}.resume_blocked"
  done
  rm -f "${RUN_DIR}/.initial_launch_complete"
else
  for row in "${VARIANT_ROWS[@]}"; do
    IFS='|' read -r variant gpu cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
    state_args=(
      "${PYTHON}" "${HELPER}" resume-state
      --out-dir "${RUN_DIR}/${variant}"
    )
    if [[ "${ae_mode}" == "1" ]]; then
      state_args+=(--ae)
    fi
    state="$("${state_args[@]}" 2>/dev/null || true)"
    if [[ "${state}" != "complete" ]]; then
      echo "[v13] FRESH=0 requires a complete resume bundle for ${variant}; state=${state:-unknown}" >&2
      echo "[v13] use FRESH=1 only for a deliberate new V13 run" >&2
      exit 1
    fi
  done
fi

if [[ -e "${RUN_DIR}/logs" && ! -L "${RUN_DIR}/logs" ]]; then
  echo "[v13] ${RUN_DIR}/logs exists and is not a symlink" >&2
  exit 1
fi
ln -sfn "${LOCAL_LOG_DIR}" "${RUN_DIR}/logs"

manifest_args=(
  "${PYTHON}" "${HELPER}" manifest
  --output "${RUN_DIR}/MANIFEST.json"
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
write_readme

echo "[v13] launching detached watchdog"
nohup setsid env \
  RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
  bash "${ROOT}/launch_v13_controlled.sh" --watchdog \
  >> "${LOCAL_LOG_DIR}/watchdog.log" 2>&1 < /dev/null &
watchdog_pid=$!
printf '%s watchdog run=%s\n' "${watchdog_pid}" "${RUN_DIR}" > "${RUN_DIR}/pids/watchdog.pid"
sleep 1
if ! kill -0 "${watchdog_pid}" 2>/dev/null; then
  echo "[v13] watchdog failed to start; inspect ${LOCAL_LOG_DIR}/watchdog.log" >&2
  exit 1
fi

echo "[v13] detached watchdog PID ${watchdog_pid}"
echo "[v13] run: ${RUN_DIR}"
echo "[v13] logs: ${LOCAL_LOG_DIR}"
echo "[v13] status: bash ${ROOT}/status_v13_controlled.sh"
echo "[v13] stop: bash ${ROOT}/stop_run.sh ${RUN_DIR}"
