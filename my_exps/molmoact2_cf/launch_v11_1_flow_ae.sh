#!/usr/bin/env bash
# V11_1: MolmoAct2 Action Expert as V, RLT CF guide as G.
# In-process AE (no serve.py), 1 worker/GPU, frozen RL token.
# Configs: ae_no_guide | ae_cf_frozen_token (baseline removed).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
mkdir -p "${B1K_TMP}"

CF_MODE=flow
RUN_DIR="${RUN_DIR:-runs/rlt_cf_v11_1_flow_ae}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v11_1_flow_ae_logs}"
SCREEN_NAME="${SCREEN_NAME:-rlt_cf_v11_1_flow_ae}"
NUM_GPUS="${NUM_GPUS:-8}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-1}"
NUM_CONFIGS=2
TOTAL_WORKERS=$((NUM_GPUS * INSTANCES_PER_GPU))
WORKERS_PER_CONFIG=$((TOTAL_WORKERS / NUM_CONFIGS))
BENCH_N="${BENCH_N:-1000}"
TARGET_ENV_STEPS="${TARGET_ENV_STEPS:-4166667}"
UPDATES_PER_EPISODE="${UPDATES_PER_EPISODE:-4}"
LOG_EVERY_EPISODES="${LOG_EVERY_EPISODES:-10}"
HORIZON="${HORIZON:-500}"
N_CRITICS="${N_CRITICS:-10}"
TRAINER_STAGGER_SEC="${TRAINER_STAGGER_SEC:-15}"
FLOW_STEPS="${FLOW_STEPS:-10}"
GUIDANCE_COEF="${GUIDANCE_COEF:-0.5}"
G_MIN_ADVANTAGE="${G_MIN_ADVANTAGE:-0.003}"
GUIDE_BETA="${GUIDE_BETA:-0.05}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-2}"
AE_LORA_RANK="${AE_LORA_RANK:-16}"

RLT_CKPT="${RLT_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v11_1_ae}"
RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-1}"
RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-$(( NUM_GPUS * RLT_EGL_PER_GPU ))}"
RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts_v11_1}"
mkdir -p "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"

MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${MOLMOSPACES}/.venv/bin/python"

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT}/${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}" "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd -P)"

if [[ -e "${RUN_DIR}/logs" && ! -L "${RUN_DIR}/logs" ]]; then
  echo "[launch] ${RUN_DIR}/logs exists and is not a symlink; refusing to replace it" >&2
  exit 1
fi
ln -sfn "${LOCAL_LOG_DIR}" "${RUN_DIR}/logs"

if (( TOTAL_WORKERS % NUM_CONFIGS != 0 )); then
  echo "[launch] ${TOTAL_WORKERS} workers cannot split across ${NUM_CONFIGS} configs" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "[launch] MolmoSpaces Python not found: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${RLT_CKPT}" ]]; then
  echo "[launch] RLT_CKPT missing: ${RLT_CKPT}" >&2
  exit 1
fi
RLT_CKPT="$(cd "$(dirname "${RLT_CKPT}")" && pwd -P)/$(basename "${RLT_CKPT}")"

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -ra GPU_ARR <<< "${GPU_IDS}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
else
  GPU_ARR=($(seq 0 $((NUM_GPUS - 1))))
fi
if (( ${#GPU_ARR[@]} != NUM_GPUS )); then
  echo "[launch] GPU_ARR has ${#GPU_ARR[@]} entries but NUM_GPUS=${NUM_GPUS}" >&2
  exit 1
fi

shopt -s nullglob
for pidfile in "${RUN_DIR}/pids"/*.pid; do
  read -r pid _rest < "${pidfile}" || pid=""
  if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[launch] active PID ${pidfile}; run stop_run.sh first" >&2
    exit 1
  fi
  rm -f "${pidfile}"
done

if [[ "${RLT_V11_1_IN_SCREEN:-0}" != "1" && "${NO_SCREEN:-0}" != "1" ]]; then
  if ! command -v screen >/dev/null 2>&1; then
    echo "[launch] GNU screen required (or NO_SCREEN=1)" >&2
    exit 1
  fi
  if screen -ls 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "[launch] screen ${SCREEN_NAME} already exists" >&2
    exit 1
  fi
  screen -dmS "${SCREEN_NAME}" \
    env RLT_V11_1_IN_SCREEN=1 RUN_DIR="${RUN_DIR}" LOCAL_LOG_DIR="${LOCAL_LOG_DIR}" \
      RLT_CKPT="${RLT_CKPT}" NUM_GPUS="${NUM_GPUS}" INSTANCES_PER_GPU="${INSTANCES_PER_GPU}" \
      GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}" \
      RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
      RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU}" \
      bash "${ROOT}/launch_v11_1_flow_ae.sh"
  echo "[launch] started screen ${SCREEN_NAME}"
  echo "[launch] logs: ${RUN_DIR}/logs"
  exit 0
fi

exec >> "${LOCAL_LOG_DIR}/launcher.log" 2>&1
echo "[launch $(date -Is)] V11_1 Molmo AE as V / RLT guide as G"
echo "[launch] run_dir=${RUN_DIR} workers=${TOTAL_WORKERS} (${NUM_GPUS}x${INSTANCES_PER_GPU})"
echo "[launch] rlt_ckpt=${RLT_CKPT}"

CONFIG_NAMES=(
  "ae_no_guide"
  "ae_cf_frozen_token"
)

declare -a TRAINER_PIDS=()

start_trainer() {
  local worker="$1"
  local gpu_idx=$((worker / INSTANCES_PER_GPU))
  local gpu="${GPU_ARR[$gpu_idx]}"
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
    --device cuda:0
    --out_dir "${shard_out}"
    --config_name "${config_name}"
    --target_env_steps "${TARGET_ENV_STEPS}"
    --start_episode "${start_episode}"
    --shard_size "${shard_size}"
    --horizon "${HORIZON}"
    --log_every_episodes "${LOG_EVERY_EPISODES}"
    --ckpt_every_episodes "${CKPT_EVERY_EPISODES:-5}"
    --replay_out "${shard_out}/chunk_replay.npz"
    --seed "${worker}"
    --n_critics "${N_CRITICS}"
    --cf_mode flow
    --flow_steps "${FLOW_STEPS}"
    --guidance_coef "${GUIDANCE_COEF}"
    --ae_trainable
    --ae_lora
    --ae_lora_rank "${AE_LORA_RANK}"
    --ae_batch_size "${AE_BATCH_SIZE}"
    --freeze_token
    --joint_cf
    --rlt_ckpt "${RLT_CKPT}"
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
    --cql_n_actions 8
    --tmp_rollout_dir "${TMP_ROLLOUT_DIR}"
    --server_port "$((9000 + worker))"
    --updates_per_episode "${UPDATES_PER_EPISODE}"
  )

  case "${config_name}" in
    ae_no_guide)
      command+=(--actor_mode rlt --no_cf_guide)
      ;;
    ae_cf_frozen_token)
      command+=(--actor_mode rlt --use_cf_guide)
      ;;
    *)
      echo "[launch] unknown config ${config_name}" >&2
      return 1
      ;;
  esac

  echo "[launch] trainer worker=${worker} gpu=${gpu} config=${config_name} shard=${config_worker}"
  (
    exec setsid env \
      RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
      RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
      RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
      RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU}" \
      RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
      "${command[@]}"
  ) > "${logfile}" 2>&1 &
  local pid=$!
  printf '%s train config=%s shard=%s gpu=%s\n' \
    "${pid}" "${config_name}" "${config_worker}" "${gpu}" > "${pidfile}"
  TRAINER_PIDS+=("${pid}")
}

for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
  start_trainer "${worker}"
  sleep "${TRAINER_STAGGER_SEC}"
done

echo "[launch] started ${#TRAINER_PIDS[@]} trainers (no HTTP serve — in-process AE)"
echo "[launch] stop: ${ROOT}/stop_run.sh ${RUN_DIR}"

# Lightweight watchdog: restart dead trainers.
WATCH_LOG="${LOCAL_LOG_DIR}/watchdog.log"
POLL_SEC="${POLL_SEC:-60}"
while true; do
  for ((worker=0; worker<TOTAL_WORKERS; worker++)); do
    config_idx=$((worker % NUM_CONFIGS))
    config_worker=$((worker / NUM_CONFIGS))
    config_name="${CONFIG_NAMES[config_idx]}"
    pidfile="${RUN_DIR}/pids/train_${config_name}_s${config_worker}.pid"
    if [[ -f "${pidfile}" ]]; then
      read -r pid _rest < "${pidfile}" || pid=""
      if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
        continue
      fi
    fi
    echo "[watchdog $(date -Is)] restarting worker=${worker} ${config_name}" | tee -a "${WATCH_LOG}"
    start_trainer "${worker}"
    sleep "${TRAINER_STAGGER_SEC}"
  done
  if [[ "${ONCE:-0}" == "1" ]]; then
    exit 0
  fi
  sleep "${POLL_SEC}"
done
