#!/usr/bin/env bash
# V16 improved RLT+CF smoke/long matrix:
#   residual_rlt_cf → physical GPUs 0-3 (4 shards)
#   flow_rlt_cf     → physical GPUs 4-7 (4 shards)
# Uses z=256 / d=512 / 4-layer token AE checkpoints (non-overlapping chunks).

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v16_rlt_improved}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v16_rlt_improved_logs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${ROOT}/runs/rlt_pretrain_demo1k_z256_d512_l4}"
RESIDUAL_CKPT="${RESIDUAL_CKPT:-${PRETRAIN_DIR}/rlt_cf_pretrain_demo1k_z256_d512_l4.pt}"
FLOW_CKPT="${FLOW_CKPT:-${PRETRAIN_DIR}/rlt_cf_flow_pretrain_demo1k_z256_d512_l4.pt}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v16_rlt_improved}"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts_v16_rlt_improved}"

V16_MODE="${V16_MODE:-long}"
FRESH="${FRESH:-1}"
POLL_SEC="${POLL_SEC:-60}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"
SERVER_STAGGER_SEC="${SERVER_STAGGER_SEC:-3}"
TRAINER_STAGGER_SEC="${TRAINER_STAGGER_SEC:-8}"
BASE_HTTP_PORT="${BASE_HTTP_PORT:-8720}"
SHARDS_PER_ARM="${SHARDS_PER_ARM:-4}"

case "${V16_MODE}" in
  full)
    V16_MAX_VALID_EPISODES="${V16_MAX_VALID_EPISODES:-400}"
    V16_TARGET_ENV_STEPS="${V16_TARGET_ENV_STEPS:-250000}"
    V16_SNAPSHOT_EPISODES="${V16_SNAPSHOT_EPISODES:-0,100,200,400}"
    ;;
  long)
    V16_MAX_VALID_EPISODES="${V16_MAX_VALID_EPISODES:-1000}"
    V16_TARGET_ENV_STEPS="${V16_TARGET_ENV_STEPS:-600000}"
    V16_SNAPSHOT_EPISODES="${V16_SNAPSHOT_EPISODES:-0,100,200,400,700,1000}"
    ;;
  smoke)
    V16_MAX_VALID_EPISODES="${V16_MAX_VALID_EPISODES:-2}"
    V16_TARGET_ENV_STEPS="${V16_TARGET_ENV_STEPS:-1000}"
    V16_SNAPSHOT_EPISODES="${V16_SNAPSHOT_EPISODES:-0,${V16_MAX_VALID_EPISODES}}"
    ;;
  *)
    echo "[v16-improved] V16_MODE must be full, long, or smoke" >&2
    exit 1
    ;;
esac

V16_MAX_UPDATE_SEC_PER_EPISODE="${V16_MAX_UPDATE_SEC_PER_EPISODE:-60}"
V16_UPDATES_PER_EPISODE="${V16_UPDATES_PER_EPISODE:-128}"

MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
SERVE_PYTHON="${SERVE_PYTHON:-${MOLMOACT2}/.venv/bin/python}"
HELPER="${ROOT}/v16_harness.py"

if [[ "${RUN_DIR}" != /* ]]; then RUN_DIR="${ROOT}/${RUN_DIR}"; fi
if [[ "${LOCAL_LOG_DIR}" != /* ]]; then LOCAL_LOG_DIR="${ROOT}/${LOCAL_LOG_DIR}"; fi
RUN_DIR="${RUN_DIR%/}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR%/}"

if [[ "$(basename "${RUN_DIR}")" != "rlt_cf_v16_rlt_improved" ]]; then
  echo "[v16-improved] RUN_DIR basename must be rlt_cf_v16_rlt_improved" >&2
  exit 1
fi
if [[ ! -f "${RESIDUAL_CKPT}" || ! -f "${FLOW_CKPT}" ]]; then
  echo "[v16-improved] missing improved pretrain ckpts under ${PRETRAIN_DIR}" >&2
  echo "  run: bash pretrain_rlt_d512_l4.sh" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" || ! -x "${SERVE_PYTHON}" ]]; then
  echo "[v16-improved] python missing" >&2
  exit 1
fi

TRAIN_BENCHMARK="${BENCHMARK_ROOT}/train"
mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"
ln -sfn "${LOCAL_LOG_DIR}" "${RUN_DIR}/logs"

export B1K_ROOT B1K_TMP RUN_DIR LOCAL_LOG_DIR BENCHMARK_ROOT
export RESIDUAL_CKPT FLOW_CKPT RLT_EGL_LOCK_DIR TMP_ROLLOUT_DIR
export RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-16}"
export RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-2}"
export RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}"
export RLT_VLA_PREFETCH="${RLT_VLA_PREFETCH:-1}"
export RLT_VLA_PREFETCH_K="${RLT_VLA_PREFETCH_K:-2}"
export RLT_VLA_PREFETCH_REQUIRE_OBS_MATCH="${RLT_VLA_PREFETCH_REQUIRE_OBS_MATCH:-0}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export V16_MODE V16_MAX_VALID_EPISODES V16_TARGET_ENV_STEPS V16_SNAPSHOT_EPISODES
export V16_MAX_UPDATE_SEC_PER_EPISODE V16_UPDATES_PER_EPISODE
export RLT_CF_V16_RUN_DIR="${RUN_DIR}"

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
else
  GPU_ARRAY=(0 1 2 3 4 5 6 7)
fi
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "[v16-improved] need 8 GPUs, got ${#GPU_ARRAY[@]}" >&2
  exit 1
fi

# Arm → GPU list (4 shards each).
RESIDUAL_GPUS=("${GPU_ARRAY[@]:0:4}")
FLOW_GPUS=("${GPU_ARRAY[@]:4:4}")

pid_first_field() {
  local pidfile="$1" pid=""
  [[ -f "${pidfile}" ]] || return 1
  IFS=$' \t\r\n' read -r pid _ < "${pidfile}"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${pid}"
}

pid_belongs_to_run() {
  local pid="$1" environ="/proc/${pid}/environ" entry
  [[ -r "${environ}" ]] || return 1
  while IFS= read -r entry; do
    [[ "${entry}" == "RLT_CF_V16_RUN_DIR=${RUN_DIR}" ]] && return 0
  done < <(tr '\0' '\n' < "${environ}")
  return 1
}

pid_is_live_owned() {
  local pidfile="$1" pid
  pid="$(pid_first_field "${pidfile}")" || return 1
  kill -0 "${pid}" 2>/dev/null && pid_belongs_to_run "${pid}"
}

set_cli_arg() {
  local flag="$1" value="$2" idx
  for idx in "${!GENERATED_COMMAND[@]}"; do
    if [[ "${GENERATED_COMMAND[$idx]}" == "${flag}" ]]; then
      GENERATED_COMMAND[$((idx + 1))]="${value}"
      return 0
    fi
  done
  GENERATED_COMMAND+=("${flag}" "${value}")
}

trainer_command() {
  local variant="$1" fresh_start="$2"
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
  [[ "${fresh_start}" == "1" ]] && helper_args+=(--fresh)
  mapfile -d '' -t GENERATED_COMMAND < <("${helper_args[@]}")
  (( ${#GENERATED_COMMAND[@]} > 0 ))
}

server_command() {
  local variant="$1" checkpoint="$2"
  local -a helper_args=(
    "${PYTHON}" "${HELPER}" server-command
    --variant "${variant}"
    --root "${ROOT}"
    --checkpoint "${checkpoint}"
    --format nul
    --serve-prefix "${SERVE_PYTHON}"
  )
  mapfile -d '' -t GENERATED_COMMAND < <("${helper_args[@]}")
  (( ${#GENERATED_COMMAND[@]} > 0 ))
}

server_ready() {
  local port="$1" response
  response="$(curl -sf --max-time 3 "http://127.0.0.1:${port}/healthz" 2>/dev/null)" || return 1
  [[ "${response}" == *'"status":"ok"'* ]]
}

wait_for_server() {
  local variant="$1" port="$2" attempt
  for ((attempt=1; attempt<=SERVER_WAIT_ATTEMPTS; attempt++)); do
    server_ready "${port}" && return 0
    sleep 5
  done
  echo "[v16-improved] server ${variant} port ${port} not ready" >&2
  return 1
}

start_server() {
  local variant="$1" gpu="$2" port="$3" checkpoint="$4" shard="$5"
  local pidfile="${RUN_DIR}/pids/server_${variant}_s${shard}.pid"
  local logfile="${LOCAL_LOG_DIR}/server_${variant}_s${shard}_gpu${gpu}.log"
  pid_is_live_owned "${pidfile}" && return 0
  rm -f "${pidfile}"
  server_command "${variant}" "${checkpoint}"
  set_cli_arg --port "${port}"
  echo "[v16-improved $(date -Is)] server ${variant} s${shard} gpu=${gpu} port=${port}"
  (
    cd "${MOLMOACT2}"
    exec setsid env \
      RLT_CF_V16_RUN_DIR="${RUN_DIR}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME}" \
      "${GENERATED_COMMAND[@]}"
  ) >> "${logfile}" 2>&1 &
  printf '%s server variant=%s shard=%s gpu=%s port=%s\n' \
    "$!" "${variant}" "${shard}" "${gpu}" "${port}" > "${pidfile}"
}

start_trainer() {
  local variant="$1" gpu="$2" fresh_start="$3" shard="$4" port="$5"
  local out_dir="${RUN_DIR}/${variant}/shard_${shard}"
  local pidfile="${RUN_DIR}/pids/train_${variant}_s${shard}.pid"
  local logfile="${LOCAL_LOG_DIR}/train_${variant}_s${shard}_gpu${gpu}.log"
  mkdir -p "${out_dir}"
  pid_is_live_owned "${pidfile}" && return 0
  rm -f "${pidfile}"
  trainer_command "${variant}" "${fresh_start}"
  set_cli_arg --out_dir "${out_dir}"
  set_cli_arg --replay_out "${out_dir}/chunk_replay.npz"
  set_cli_arg --tmp_rollout_dir "${TMP_ROLLOUT_DIR}/${variant}_s${shard}"
  set_cli_arg --server_port "${port}"
  local base_seed=20260814
  local idx
  for idx in "${!GENERATED_COMMAND[@]}"; do
    if [[ "${GENERATED_COMMAND[$idx]}" == "--seed" ]]; then
      base_seed="${GENERATED_COMMAND[$((idx + 1))]}"
      break
    fi
  done
  set_cli_arg --seed "$((base_seed + shard * 17))"
  echo "[v16-improved $(date -Is)] train ${variant} s${shard} gpu=${gpu} port=${port} fresh=${fresh_start}"
  (
    exec setsid env \
      RLT_CF_V16_RUN_DIR="${RUN_DIR}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME}" \
      MUJOCO_GL="${MUJOCO_GL}" \
      PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}" \
      RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
      RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
      RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU}" \
      RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC}" \
      RLT_VLA_PREFETCH="${RLT_VLA_PREFETCH}" \
      RLT_VLA_PREFETCH_K="${RLT_VLA_PREFETCH_K}" \
      RLT_VLA_PREFETCH_REQUIRE_OBS_MATCH="${RLT_VLA_PREFETCH_REQUIRE_OBS_MATCH}" \
      "${GENERATED_COMMAND[@]}"
  ) >> "${logfile}" 2>&1 &
  printf '%s train variant=%s shard=%s gpu=%s fresh=%s\n' \
    "$!" "${variant}" "${shard}" "${gpu}" "${fresh_start}" > "${pidfile}"
}

launch_arm() {
  local variant="$1" checkpoint="$2"
  shift 2
  local -a gpus=("$@")
  local shard gpu port
  if (( ${#gpus[@]} != SHARDS_PER_ARM )); then
    echo "[v16-improved] expected ${SHARDS_PER_ARM} GPUs for ${variant}" >&2
    return 1
  fi
  if [[ "${FRESH}" == "1" ]]; then
    rm -rf "${RUN_DIR}/${variant}"
    mkdir -p "${RUN_DIR}/${variant}"
  fi
  for ((shard=0; shard<SHARDS_PER_ARM; shard++)); do
    gpu="${gpus[$shard]}"
    if [[ "${variant}" == "residual_rlt_cf" ]]; then
      port=$((BASE_HTTP_PORT + shard))
    else
      port=$((BASE_HTTP_PORT + SHARDS_PER_ARM + shard))
    fi
    start_server "${variant}" "${gpu}" "${port}" "${checkpoint}" "${shard}"
    sleep "${SERVER_STAGGER_SEC}"
  done
  for ((shard=0; shard<SHARDS_PER_ARM; shard++)); do
    gpu="${gpus[$shard]}"
    if [[ "${variant}" == "residual_rlt_cf" ]]; then
      port=$((BASE_HTTP_PORT + shard))
    else
      port=$((BASE_HTTP_PORT + SHARDS_PER_ARM + shard))
    fi
    wait_for_server "${variant}" "${port}"
  done
  for ((shard=0; shard<SHARDS_PER_ARM; shard++)); do
    gpu="${gpus[$shard]}"
    if [[ "${variant}" == "residual_rlt_cf" ]]; then
      port=$((BASE_HTTP_PORT + shard))
    else
      port=$((BASE_HTTP_PORT + SHARDS_PER_ARM + shard))
    fi
    start_trainer "${variant}" "${gpu}" "${FRESH}" "${shard}" "${port}"
    sleep "${TRAINER_STAGGER_SEC}"
  done
}

cat > "${RUN_DIR}/MANIFEST.json" <<EOF
{
  "schema_version": "v16-rlt-improved-1",
  "run_dir": "${RUN_DIR}",
  "mode": "${V16_MODE}",
  "arch": {"z_dim": 256, "token_d_model": 512, "token_layers": 4, "chunk_stride": "non_overlapping"},
  "residual_checkpoint": "${RESIDUAL_CKPT}",
  "flow_checkpoint": "${FLOW_CKPT}",
  "arms": {
    "residual_rlt_cf": {"gpus": [$(IFS=,; echo "${RESIDUAL_GPUS[*]}")], "shards": ${SHARDS_PER_ARM}},
    "flow_rlt_cf": {"gpus": [$(IFS=,; echo "${FLOW_GPUS[*]}")], "shards": ${SHARDS_PER_ARM}}
  },
  "ports": {"base": ${BASE_HTTP_PORT}, "count": $((SHARDS_PER_ARM * 2))},
  "launched_at": "$(date -Is)"
}
EOF

echo "[v16-improved $(date -Is)] launching residual_rlt_cf on GPUs ${RESIDUAL_GPUS[*]}"
launch_arm residual_rlt_cf "${RESIDUAL_CKPT}" "${RESIDUAL_GPUS[@]}"
echo "[v16-improved $(date -Is)] launching flow_rlt_cf on GPUs ${FLOW_GPUS[*]}"
launch_arm flow_rlt_cf "${FLOW_CKPT}" "${FLOW_GPUS[@]}"

touch "${RUN_DIR}/.initial_launch_complete"
echo "[v16-improved $(date -Is)] initial launch complete; polling every ${POLL_SEC}s"

# Lightweight watchdog: restart dead owned trainers/servers.
while true; do
  sleep "${POLL_SEC}"
  for variant in residual_rlt_cf flow_rlt_cf; do
    if [[ "${variant}" == "residual_rlt_cf" ]]; then
      arm_gpus=("${RESIDUAL_GPUS[@]}")
      ckpt="${RESIDUAL_CKPT}"
      port_base="${BASE_HTTP_PORT}"
    else
      arm_gpus=("${FLOW_GPUS[@]}")
      ckpt="${FLOW_CKPT}"
      port_base=$((BASE_HTTP_PORT + SHARDS_PER_ARM))
    fi
    for ((shard=0; shard<SHARDS_PER_ARM; shard++)); do
      gpu="${arm_gpus[$shard]}"
      port=$((port_base + shard))
      spidfile="${RUN_DIR}/pids/server_${variant}_s${shard}.pid"
      tpidfile="${RUN_DIR}/pids/train_${variant}_s${shard}.pid"
      if ! pid_is_live_owned "${spidfile}"; then
        echo "[watchdog $(date -Is)] restart server ${variant} s${shard}"
        start_server "${variant}" "${gpu}" "${port}" "${ckpt}" "${shard}"
        wait_for_server "${variant}" "${port}" || true
      fi
      if ! pid_is_live_owned "${tpidfile}"; then
        echo "[watchdog $(date -Is)] restart train ${variant} s${shard}"
        start_trainer "${variant}" "${gpu}" 0 "${shard}" "${port}" || true
      fi
    done
  done
done
