#!/usr/bin/env bash
# Launch one side of dual CF v8 (residual or flow). CUDA_VISIBLE_DEVICES selects GPUs.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
mkdir -p "${B1K_TMP}"

CF_MODE="${CF_MODE:-residual}"
RUN_DIR="${RLT_CF_V7_RUN_DIR:-${RUN_DIR:-runs/rlt_cf_v8_${CF_MODE}}}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v8_${CF_MODE}_logs}"
SCREEN_NAME="${SCREEN_NAME:-rlt_cf_v8_${CF_MODE}}"
NUM_GPUS="${NUM_GPUS:-4}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}"
NUM_CONFIGS=4
TOTAL_WORKERS=$((NUM_GPUS * INSTANCES_PER_GPU))
WORKERS_PER_CONFIG=$((TOTAL_WORKERS / NUM_CONFIGS))
BASE_PORT="${BASE_PORT:-8600}"
BENCH_N="${BENCH_N:-1000}"
TARGET_ENV_STEPS="${TARGET_ENV_STEPS:-4166667}"
UPDATES_PER_EPISODE="${UPDATES_PER_EPISODE:-8}"
LOG_EVERY_EPISODES="${LOG_EVERY_EPISODES:-50}"
HORIZON="${HORIZON:-500}"
N_CRITICS="${N_CRITICS:-10}"
SERVER_STAGGER_SEC="${SERVER_STAGGER_SEC:-3}"
# Stagger trainers to avoid concurrent NLTK/asset extraction races.
TRAINER_STAGGER_SEC="${TRAINER_STAGGER_SEC:-8}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"
RLT_CKPT="${RLT_CKPT:-}"
FLOW_STEPS="${FLOW_STEPS:-10}"
GUIDANCE_COEF="${GUIDANCE_COEF:-0.5}"
# v11 residual fixes: stronger guide distill, softer gate.
G_MIN_ADVANTAGE="${G_MIN_ADVANTAGE:-0.003}"
GUIDE_BETA="${GUIDE_BETA:-0.05}"
GUIDE_TARGET_DELTA_FRAC="${GUIDE_TARGET_DELTA_FRAC:-1.0}"
# v11 flow joint CF: trainable FlowVelocityActor v_θ + G_φ (paper-faithful).
JOINT_CF="${JOINT_CF:-0}"
# Isolate EGL locks per experiment side so dual runs do not starve each other.
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_${CF_MODE}}"
# Concurrent MuJoCo EGL rollouts per physical GPU (exclusive lock was 1 and
# queued 3/4 workers for tens of minutes).  Global cap = GPUs * per-GPU.
RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-3}"
RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-$(( NUM_GPUS * RLT_EGL_PER_GPU ))}"
RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts}"
mkdir -p "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"
MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${MOLMOSPACES}/.venv/bin/python"
MOLMOACT2_PYTHON="${MOLMOACT2}/.venv/bin/python"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
# Prefer uv when available; otherwise call MolmoAct2's venv python directly.
if [[ -n "${UV_BIN}" && -x "${UV_BIN}" ]]; then
  SERVE_CMD=("${UV_BIN}" run python)
else
  if [[ ! -x "${MOLMOACT2_PYTHON}" ]]; then
    echo "[launch] neither uv nor MolmoAct2 Python found (${MOLMOACT2_PYTHON})" >&2
    exit 1
  fi
  SERVE_CMD=("${MOLMOACT2_PYTHON}")
  UV_BIN=""
fi

if [[ "${CF_MODE}" != "residual" && "${CF_MODE}" != "flow" ]]; then
  echo "[launch] CF_MODE must be residual|flow, got ${CF_MODE}" >&2
  exit 1
fi

# Absolute physical GPU list. Prefer GPU_IDS; else snapshot CUDA_VISIBLE_DEVICES
# before per-worker override (nested remapping is unreliable with setsid/env).
if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -ra GPU_ARR <<< "${GPU_IDS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_ARR=($(seq 0 $((NUM_GPUS - 1))))
fi
if (( ${#GPU_ARR[@]} != NUM_GPUS )); then
  echo "[launch] GPU_ARR has ${#GPU_ARR[@]} entries but NUM_GPUS=${NUM_GPUS}: ${GPU_ARR[*]}" >&2
  exit 1
fi
echo "[launch] physical GPUs: ${GPU_ARR[*]}"

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT}/${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}" "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd -P)"
export RLT_CF_V7_RUN_DIR="${RUN_DIR}"
export RLT_CF_V4_RUN_DIR="${RUN_DIR}"

if [[ -e "${RUN_DIR}/logs" && ! -L "${RUN_DIR}/logs" ]]; then
  echo "[launch] ${RUN_DIR}/logs exists and is not a symlink; refusing to replace it" >&2
  exit 1
fi
ln -sfn "${LOCAL_LOG_DIR}" "${RUN_DIR}/logs"

if (( TOTAL_WORKERS % NUM_CONFIGS != 0 )); then
  echo "[launch] ${TOTAL_WORKERS} workers cannot be split across ${NUM_CONFIGS} configs" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "[launch] MolmoSpaces Python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ -n "${RLT_CKPT}" && ! -f "${RLT_CKPT}" ]]; then
  echo "[launch] RLT_CKPT does not exist: ${RLT_CKPT}" >&2
  exit 1
fi
if [[ -n "${RLT_CKPT}" ]]; then
  RLT_CKPT="$(cd "$(dirname "${RLT_CKPT}")" && pwd -P)/$(basename "${RLT_CKPT}")"
  export RLT_CKPT
fi

shopt -s nullglob
for pidfile in "${RUN_DIR}/pids"/*.pid; do
  read -r pid _rest < "${pidfile}" || pid=""
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[launch] active PID manifest ${pidfile} (pid ${pid}); run stop_run.sh first" >&2
    exit 1
  fi
  rm -f "${pidfile}"
done

if [[ "${RLT_V4_IN_SCREEN:-0}" != "1" && "${NO_SCREEN:-0}" != "1" ]]; then
  if ! command -v screen >/dev/null 2>&1; then
    echo "[launch] GNU screen is required (or set NO_SCREEN=1)" >&2
    exit 1
  fi
  if screen -ls 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "[launch] screen session ${SCREEN_NAME} already exists" >&2
    exit 1
  fi
  screen -dmS "${SCREEN_NAME}" \
    env RLT_V4_IN_SCREEN=1 RLT_CF_V7_RUN_DIR="${RUN_DIR}" RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
      CF_MODE="${CF_MODE}" RLT_CKPT="${RLT_CKPT}" NUM_GPUS="${NUM_GPUS}" \
      INSTANCES_PER_GPU="${INSTANCES_PER_GPU}" BASE_PORT="${BASE_PORT}" \
      LOCAL_LOG_DIR="${LOCAL_LOG_DIR}" FLOW_STEPS="${FLOW_STEPS}" \
      GUIDANCE_COEF="${GUIDANCE_COEF}" GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}" \
      RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
      RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU}" RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC}" \
      DETACH_AFTER_START="${DETACH_AFTER_START:-1}" \
      bash "${ROOT}/launch_rlt_v8_side.sh"
  echo "[launch] started screen session ${SCREEN_NAME}"
  echo "[launch] attach: screen -r ${SCREEN_NAME}"
  echo "[launch] logs: ${RUN_DIR}/logs"
  echo "[launch] stop: ${ROOT}/stop_run.sh ${RUN_DIR}"
  exit 0
fi

exec >> "${LOCAL_LOG_DIR}/launcher.log" 2>&1
echo "[launch $(date -Is)] starting CF_MODE=${CF_MODE} side"
echo "[launch] run_dir=${RUN_DIR}"
echo "[launch] workers=${TOTAL_WORKERS} (${NUM_GPUS} GPUs x ${INSTANCES_PER_GPU}/GPU)"
echo "[launch] workers_per_config=${WORKERS_PER_CONFIG} target_steps_per_worker=${TARGET_ENV_STEPS}"
echo "[launch] rlt_ckpt=${RLT_CKPT:-<fresh>}"

DEFAULT_FEATURE_MODE="tokens"
if [[ -n "${RLT_CKPT}" ]]; then
  DEFAULT_FEATURE_MODE="rl_token"
fi

CONFIG_NAMES=(
  "mean_pool_baseline"
  "rlt_actor_no_guide"
  "rlt_cf_frozen_token"
  "rlt_cf_online_token"
)

declare -a SERVER_PIDS=()
declare -a TRAINER_PIDS=()

start_server() {
  local worker="$1"
  local gpu_idx=$((worker / INSTANCES_PER_GPU))
  local gpu="${GPU_ARR[$gpu_idx]}"
  local port=$((BASE_PORT + worker))
  local config_idx=$((worker % NUM_CONFIGS))
  local feature_mode="${DEFAULT_FEATURE_MODE}"
  # Online-token workers must encode locally so their updated encoder affects
  # subsequent chunks; the other configs may use server-side frozen z_rl.
  if [[ "${CONFIG_NAMES[config_idx]}" == "rlt_cf_online_token" ]]; then
    feature_mode="tokens"
  fi
  local logfile="${LOCAL_LOG_DIR}/server_w${worker}_gpu${gpu}.log"
  local pidfile="${RUN_DIR}/pids/server_w${worker}.pid"
  local -a command=(
    "${SERVE_CMD[@]}" "${ROOT}/serve.py"
    --host 0.0.0.0
    --port "${port}"
    --device cuda:0
    --dtype bfloat16
    --disable_g
    --feature_mode "${feature_mode}"
  )
  if [[ -n "${RLT_CKPT}" && "${feature_mode}" == "rl_token" ]]; then
    command+=(--rlt_ckpt "${RLT_CKPT}")
  fi
  echo "[launch] server worker=${worker} gpu=${gpu} port=${port} mode=${feature_mode}"
  (
    cd "${MOLMOACT2}"
    exec setsid env \
      RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
      "${command[@]}"
  ) > "${logfile}" 2>&1 &
  local pid=$!
  printf '%s server worker=%s gpu=%s port=%s\n' \
    "${pid}" "${worker}" "${gpu}" "${port}" > "${pidfile}"
  SERVER_PIDS+=("${pid}")
}

for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
  start_server "${worker}"
  sleep "${SERVER_STAGGER_SEC}"
done

echo "[launch] waiting for all ${TOTAL_WORKERS} servers before starting trainers"
for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
  port=$((BASE_PORT + worker))
  ready=0
  for ((attempt=1; attempt<=SERVER_WAIT_ATTEMPTS; attempt++)); do
    if curl -sf --max-time 3 "http://127.0.0.1:${port}/act" \
      | grep -q '"status":"ok"'; then
      ready=1
      break
    fi
    sleep 5
  done
  if (( ready == 0 )); then
    gpu_idx=$((worker / INSTANCES_PER_GPU))
    gpu="${GPU_ARR[$gpu_idx]}"
    echo "[launch] ERROR server worker=${worker} gpu=${gpu} port=${port} not ready" >&2
    "${ROOT}/stop_run.sh" "${RUN_DIR}" || true
    exit 1
  fi
  echo "[launch] server worker=${worker} port=${port} READY"
done

start_trainer() {
  local worker="$1"
  local gpu_idx=$((worker / INSTANCES_PER_GPU))
  local gpu="${GPU_ARR[$gpu_idx]}"
  local port=$((BASE_PORT + worker))
  local config_idx=$((worker % NUM_CONFIGS))
  local config_worker=$((worker / NUM_CONFIGS))
  local config_name="${CONFIG_NAMES[config_idx]}"
  local shard_base=$((BENCH_N / WORKERS_PER_CONFIG))
  local start_episode=$((config_worker * shard_base))
  local shard_size="${shard_base}"
  if (( config_worker == WORKERS_PER_CONFIG - 1 )); then
    shard_size=$((BENCH_N - start_episode))
  fi
  local shard_out="${RUN_DIR}/${config_name}/shard_${config_worker}"
  local logfile="${LOCAL_LOG_DIR}/train_${config_name}_s${config_worker}_gpu${gpu}.log"
  local pidfile="${RUN_DIR}/pids/train_${config_name}_s${config_worker}.pid"
  mkdir -p "${shard_out}"

  local -a command=(
    "${PYTHON}" "${ROOT}/train_rlt_online.py"
    --server_host localhost
    --server_port "${port}"
    --device cuda:0
    --out_dir "${shard_out}"
    --config_name "${config_name}"
    --target_env_steps "${TARGET_ENV_STEPS}"
    --start_episode "${start_episode}"
    --shard_size "${shard_size}"
    --horizon "${HORIZON}"
    --log_every_episodes "${LOG_EVERY_EPISODES}"
    --ckpt_every_episodes "${CKPT_EVERY_EPISODES:-10}"
    --replay_out "${shard_out}/chunk_replay.npz"
    --seed "${worker}"
    --n_critics "${N_CRITICS}"
    --cf_mode "${CF_MODE}"
    --flow_steps "${FLOW_STEPS}"
    --guidance_coef "${GUIDANCE_COEF}"
    --explore_residual_std 0.05
    --rank_coef 1.0
    --rank_margin 0.05
    --rank_noise 0.08
    --far_rank_coef 0.5
    --far_rank_noise 0.35
    --shuffle_rank_coef 0.5
    --target_noise 0.02
    --g_start_episodes 40
    --g_min_advantage "${G_MIN_ADVANTAGE}"
    --g_min_action_sensitivity 0.003
    --gate_sensitivity_noise 0.08
    --guide_beta "${GUIDE_BETA}"
    --guide_target_delta_frac "${GUIDE_TARGET_DELTA_FRAC}"
    --cql_n_actions 8
    --tmp_rollout_dir "${TMP_ROLLOUT_DIR}"
  )
  if [[ -n "${RLT_CKPT}" ]]; then
    command+=(--rlt_ckpt "${RLT_CKPT}")
  fi
  if [[ "${JOINT_CF}" == "1" ]]; then
    command+=(--joint_cf)
  fi

  case "${config_name}" in
    mean_pool_baseline)
      command+=(
        --actor_mode vla_only
        --no_cf_guide
        --freeze_token
        --updates_per_episode 0
      )
      ;;
    rlt_actor_no_guide)
      command+=(
        --actor_mode rlt
        --no_cf_guide
        --freeze_token
        --updates_per_episode "${UPDATES_PER_EPISODE}"
      )
      ;;
    rlt_cf_frozen_token)
      command+=(
        --actor_mode rlt
        --use_cf_guide
        --freeze_token
        --updates_per_episode "${UPDATES_PER_EPISODE}"
      )
      ;;
    rlt_cf_online_token)
      command+=(
        --actor_mode rlt
        --use_cf_guide
        --tune_token_online
        --updates_per_episode "${UPDATES_PER_EPISODE}"
      )
      ;;
    *)
      echo "[launch] unknown config ${config_name}" >&2
      return 1
      ;;
  esac

  echo "[launch] trainer worker=${worker} config=${config_name} shard=${config_worker} " \
    "gpu=${gpu} port=${port} bench=[${start_episode},+$((shard_size))]"
  exec 9>> "${logfile}"
  setsid env \
    RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
    RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
    RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}" \
    RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
    RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-3}" \
    MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}" \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${gpu}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${command[@]}" >&9 2>&1 &
  local pid=$!
  exec 9>&-
  printf '%s trainer config=%s shard=%s gpu=%s port=%s\n' \
    "${pid}" "${config_name}" "${config_worker}" "${gpu}" "${port}" > "${pidfile}"
  TRAINER_PIDS+=("${pid}")
}

for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
  start_trainer "${worker}"
  sleep "${TRAINER_STAGGER_SEC}"
done

# Keep servers/trainers alive across crashes. Default DETACH_AFTER_START=1 so the
# launcher does NOT wait-for-trainers then stop_run (that killed residual servers).
start_watchdogs() {
  local gpu_ids_csv
  gpu_ids_csv=$(IFS=,; echo "${GPU_ARR[*]}")
  local common_env=(
    RLT_CF_V4_RUN_DIR="${RUN_DIR}"
    RLT_CF_V7_RUN_DIR="${RUN_DIR}"
    LOCAL_LOG_DIR="${LOCAL_LOG_DIR}"
    RLT_CKPT="${RLT_CKPT}"
    NUM_GPUS="${NUM_GPUS}"
    INSTANCES_PER_GPU="${INSTANCES_PER_GPU}"
    BASE_PORT="${BASE_PORT}"
    CF_MODE="${CF_MODE}"
    GPU_IDS="${gpu_ids_csv}"
    CUDA_VISIBLE_DEVICES="${gpu_ids_csv}"
    TARGET_ENV_STEPS="${TARGET_ENV_STEPS}"
    FLOW_STEPS="${FLOW_STEPS}"
    GUIDANCE_COEF="${GUIDANCE_COEF}"
    G_MIN_ADVANTAGE="${G_MIN_ADVANTAGE}"
    GUIDE_BETA="${GUIDE_BETA}"
    GUIDE_TARGET_DELTA_FRAC="${GUIDE_TARGET_DELTA_FRAC}"
    JOINT_CF="${JOINT_CF}"
    RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}"
    RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}"
    RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-3}"
    RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}"
    TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR}"
    HUNG_LOG_SEC="${HUNG_LOG_SEC:-1200}"
  )

  # Replace any prior watchdogs for this run dir.
  if [[ -f "${RUN_DIR}/pids/server_watchdog.pid" ]]; then
    read -r old_sw _ <"${RUN_DIR}/pids/server_watchdog.pid" || old_sw=""
    if [[ "${old_sw}" =~ ^[0-9]+$ ]]; then
      kill -TERM "${old_sw}" 2>/dev/null || true
    fi
  fi
  if [[ -f "${RUN_DIR}/pids/trainer_watchdog.pid" ]]; then
    read -r old_tw _ <"${RUN_DIR}/pids/trainer_watchdog.pid" || old_tw=""
    if [[ "${old_tw}" =~ ^[0-9]+$ ]]; then
      kill -TERM "${old_tw}" 2>/dev/null || true
    fi
  fi
  sleep 1

  setsid env "${common_env[@]}" \
    bash "${ROOT}/server_watchdog_v8.sh" "${RUN_DIR}" \
    >> "${LOCAL_LOG_DIR}/server_watchdog.out" 2>&1 &
  printf '%s server_watchdog\n' "$!" > "${RUN_DIR}/pids/server_watchdog.pid"
  echo "[launch] server_watchdog pid=$!"

  setsid env "${common_env[@]}" \
    bash "${ROOT}/trainer_watchdog_v8.sh" "${RUN_DIR}" \
    >> "${LOCAL_LOG_DIR}/trainer_watchdog.out" 2>&1 &
  printf '%s trainer_watchdog\n' "$!" > "${RUN_DIR}/pids/trainer_watchdog.pid"
  echo "[launch] trainer_watchdog pid=$!"
}

start_watchdogs

{
  echo "# MolmoAct2 RLT-CF v8 (${CF_MODE})"
  echo
  echo "- Workers: ${TOTAL_WORKERS} (${NUM_GPUS} GPUs x ${INSTANCES_PER_GPU}/GPU)"
  echo "- Configs: ${CONFIG_NAMES[*]}"
  echo "- Workers/config: ${WORKERS_PER_CONFIG}"
  echo "- Target/worker: ${TARGET_ENV_STEPS} valid env steps"
  echo "- Server feature mode: ${DEFAULT_FEATURE_MODE} (tokens for online-token workers)"
  echo "- RLT checkpoint: ${RLT_CKPT:-fresh initialization}"
  echo "- Critic ensemble size K: ${N_CRITICS}"
  echo "- Logs: ${LOCAL_LOG_DIR} (symlinked as logs/)"
  echo "- Watchdogs: server_watchdog_v8.sh + trainer_watchdog_v8.sh"
  echo
  echo "Stop only this run:"
  echo '```bash'
  echo "${ROOT}/stop_run.sh ${RUN_DIR}"
  echo '```'
} > "${RUN_DIR}/README.md"

echo "[launch] all trainers started; PID manifests are in ${RUN_DIR}/pids"
if [[ "${DETACH_AFTER_START:-1}" == "1" ]]; then
  echo "[launch $(date -Is)] DETACH_AFTER_START=1 — leaving servers/trainers/watchdogs running"
  exit 0
fi

echo "[launch] DETACH_AFTER_START=0 — waiting on trainers (legacy; will stop_run on exit)"
status=0
for pid in "${TRAINER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[launch] trainer pid ${pid} exited non-zero" >&2
    status=1
  fi
done

echo "[launch] trainers finished; stopping this run's servers"
"${ROOT}/stop_run.sh" "${RUN_DIR}" || status=1
echo "[launch $(date -Is)] finished status=${status}"
exit "${status}"
