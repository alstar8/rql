#!/usr/bin/env bash
# V19 CFGRL extractor: 1 Molmo server / GPU, 4 trainers / GPU.
# 8-GPU host: ports 8760-8767, 32 trainers. 3 EGL contexts / GPU (1 of 4 waits).
# Wide RLT heads (hidden=1024, z_expand=512) and 100 CPI rounds.
# Train split only.
#   GPU_IDS=0,1,2,3,4,5,6,7 V19_MODE=long FRESH=1 bash launch_v19_rlt_cfgrl.sh

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v19_kettle}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v19_kettle_logs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
PRETRAIN_DIR="${PRETRAIN_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle_z256_d512_l4}"
FLOW_CKPT="${FLOW_CKPT:-${PRETRAIN_DIR}/rlt_cf_flow_pretrain_house0_kettle_z256_d512_l4.pt}"
RESIDUAL_CKPT="${RESIDUAL_CKPT:-${PRETRAIN_DIR}/rlt_cf_pretrain_house0_kettle_z256_d512_l4.pt}"
SEED_REPLAY="${V19_SEED_REPLAY:-${PRETRAIN_DIR}/chunk_replay_reencoded.npz}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v19_kettle}"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts_v19_kettle}"

V19_MODE="${V19_MODE:-long}"
FRESH="${FRESH:-1}"
POLL_SEC="${POLL_SEC:-60}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"
SERVER_STAGGER_SEC="${SERVER_STAGGER_SEC:-4}"
TRAINER_STAGGER_SEC="${TRAINER_STAGGER_SEC:-8}"
BASE_HTTP_PORT="${BASE_HTTP_PORT:-8760}"
# Isolated CUDA-free children own EGL, so parents can pack 4 trainers / GPU
# without stacking 4 CUDA+EGL contexts. 3 EGL slots / GPU; 1 of 4 waits.
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}"
VARIANT="flow_cfgrl"

case "${V19_MODE}" in
  full|long)
    V19_PHASE_ROUNDS="${V19_PHASE_ROUNDS:-100}"
    V19_PHASE_B_EPISODES="${V19_PHASE_B_EPISODES:-1}"
    V19_CFGRL_ROUND="${V19_CFGRL_ROUND:-${V19_PHASE_B_EPISODES}}"
    V19_LOG_EVERY_EPISODES="${V19_LOG_EVERY_EPISODES:-1}"
    V19_CFGRL_KQ="${V19_CFGRL_KQ:-4096}"
    V19_CFGRL_KPI="${V19_CFGRL_KPI:-2048}"
    V19_CFGRL_KQ_ONLINE="${V19_CFGRL_KQ_ONLINE:-2048}"
    V19_CFGRL_KPI_ONLINE="${V19_CFGRL_KPI_ONLINE:-1024}"
    ;;
  smoke)
    V19_PHASE_ROUNDS="${V19_PHASE_ROUNDS:-1}"
    V19_PHASE_B_EPISODES="${V19_PHASE_B_EPISODES:-1}"
    V19_TOTAL_VALID_EPISODES="${V19_TOTAL_VALID_EPISODES:-32}"
    V19_MAX_VALID_EPISODES="${V19_MAX_VALID_EPISODES:-1}"
    V19_TARGET_ENV_STEPS="${V19_TARGET_ENV_STEPS:-1000}"
    V19_SNAPSHOT_EPISODES="${V19_SNAPSHOT_EPISODES:-0,${V19_MAX_VALID_EPISODES}}"
    V19_CFGRL_KQ="${V19_CFGRL_KQ:-64}"
    V19_CFGRL_KPI="${V19_CFGRL_KPI:-32}"
    V19_CFGRL_KQ_ONLINE="${V19_CFGRL_KQ_ONLINE:-32}"
    V19_CFGRL_KPI_ONLINE="${V19_CFGRL_KPI_ONLINE:-16}"
    V19_CFGRL_ROUND="${V19_CFGRL_ROUND:-1}"
    V19_LOG_EVERY_EPISODES="${V19_LOG_EVERY_EPISODES:-1}"
    ;;
  *)
    echo "[v19] V19_MODE must be full, long, or smoke" >&2
    exit 1
    ;;
esac

V19_MAX_UPDATE_SEC_PER_EPISODE="${V19_MAX_UPDATE_SEC_PER_EPISODE:-45}"
V19_UPDATES_PER_EPISODE="${V19_UPDATES_PER_EPISODE:-16}"
V19_SEED_REPLAY="${SEED_REPLAY}"

MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
SERVE_PYTHON="${SERVE_PYTHON:-${MOLMOACT2}/.venv/bin/python}"
HELPER="${ROOT}/v19_harness.py"

if [[ "${RUN_DIR}" != /* ]]; then RUN_DIR="${ROOT}/${RUN_DIR}"; fi
if [[ "${LOCAL_LOG_DIR}" != /* ]]; then LOCAL_LOG_DIR="${ROOT}/${LOCAL_LOG_DIR}"; fi
RUN_DIR="${RUN_DIR%/}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR%/}"

if [[ "$(basename "${RUN_DIR}")" != "rlt_cf_v19_kettle" ]]; then
  echo "[v19] RUN_DIR basename must be rlt_cf_v19_kettle" >&2
  exit 1
fi
if [[ ! -f "${FLOW_CKPT}" ]]; then
  echo "[v19] missing flow pretrain ${FLOW_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${V19_SEED_REPLAY}" ]]; then
  echo "[v19] missing seed replay ${V19_SEED_REPLAY}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" || ! -x "${SERVE_PYTHON}" ]]; then
  echo "[v19] python missing" >&2
  exit 1
fi

TRAIN_BENCHMARK="${BENCHMARK_ROOT}/train"
VAL_BENCHMARK="${BENCHMARK_ROOT}/val"
if [[ ! -d "${TRAIN_BENCHMARK}" ]]; then
  echo "[v19] train benchmark missing: ${TRAIN_BENCHMARK}" >&2
  exit 1
fi
# Hard hold-out: never pass val into trainers.
if [[ "${TRAIN_BENCHMARK}" == *"/val" ]]; then
  echo "[v19] refusing val split" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"
ln -sfn "${LOCAL_LOG_DIR}" "${RUN_DIR}/logs"

export B1K_ROOT B1K_TMP RUN_DIR LOCAL_LOG_DIR BENCHMARK_ROOT
export FLOW_CKPT RESIDUAL_CKPT RLT_EGL_LOCK_DIR TMP_ROLLOUT_DIR
export RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-2.0}"
export RLT_VLA_PREFETCH="${RLT_VLA_PREFETCH:-0}"
export RLT_VLA_PREFETCH_K="${RLT_VLA_PREFETCH_K:-2}"
export RLT_VLA_PREFETCH_REQUIRE_OBS_MATCH="${RLT_VLA_PREFETCH_REQUIRE_OBS_MATCH:-0}"
if [[ -z "${HF_HOME:-}" && -d /workspace-SR008.nfs2/users/staroverov/.cache/huggingface ]]; then
  HF_HOME=/workspace-SR008.nfs2/users/staroverov/.cache/huggingface
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export V19_MODE
export V19_PHASE_ROUNDS="${V19_PHASE_ROUNDS:-100}"
export V19_PHASE_B_EPISODES="${V19_PHASE_B_EPISODES:-1}"
export V19_PHASE_PROBE="${V19_PHASE_PROBE:-1}"
export V19_PHASE_BARRIER="${V19_PHASE_BARRIER:-1}"
export V19_MAX_VALID_EPISODES="${V19_MAX_VALID_EPISODES:-$((V19_PHASE_ROUNDS * V19_PHASE_B_EPISODES))}"
export V19_TARGET_ENV_STEPS="${V19_TARGET_ENV_STEPS:-$((V19_MAX_VALID_EPISODES * 500 + 500))}"
export V19_SNAPSHOT_EPISODES="${V19_SNAPSHOT_EPISODES:-0,${V19_MAX_VALID_EPISODES}}"
export V19_MAX_UPDATE_SEC_PER_EPISODE V19_UPDATES_PER_EPISODE
export V19_LOG_EVERY_EPISODES="${V19_LOG_EVERY_EPISODES:-1}"
export V19_CFGRL_ROUND="${V19_CFGRL_ROUND:-${V19_PHASE_B_EPISODES}}"
export V19_SEED_REPLAY
export V19_CFGRL_KQ="${V19_CFGRL_KQ:-4096}"
export V19_CFGRL_KPI="${V19_CFGRL_KPI:-2048}"
export V19_CFGRL_KQ_ONLINE="${V19_CFGRL_KQ_ONLINE:-2048}"
export V19_CFGRL_KPI_ONLINE="${V19_CFGRL_KPI_ONLINE:-1024}"
export V19_CFGRL_O_DIM="${V19_CFGRL_O_DIM:-128}"
export V19_HIDDEN="${V19_HIDDEN:-1024}"
export V19_N_HIDDEN_ACTOR="${V19_N_HIDDEN_ACTOR:-5}"
export V19_N_HIDDEN_CRITIC="${V19_N_HIDDEN_CRITIC:-4}"
export V19_Z_EXPAND_DIM="${V19_Z_EXPAND_DIM:-512}"
export V19_LAYERNORM_HEADS="${V19_LAYERNORM_HEADS:-1}"
export V19_POSE_CYCLE="${V19_POSE_CYCLE:-24}"
if (( V19_POSE_CYCLE < 1 )); then
  echo "[v19] V19_POSE_CYCLE must be >= 1" >&2
  exit 1
fi
export RLT_CF_V19_RUN_DIR="${RUN_DIR}"
# Re-exec so RLT_CF_V19_RUN_DIR is in the *initial* /proc/pid/environ.
# stop_run.sh keys off that file, which does not see later bash exports.
if [[ "${RLT_CF_V19_REEXEC:-}" != "1" ]]; then
  exec env RLT_CF_V19_REEXEC=1 RLT_CF_V19_RUN_DIR="${RUN_DIR}" bash "$0" "$@"
fi

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
else
  mapfile -t GPU_ARRAY < <(nvidia-smi --query-gpu=index --format=csv,noheader | awk '{print $1}')
fi
NUM_GPUS="${#GPU_ARRAY[@]}"
if (( NUM_GPUS < 1 || NUM_GPUS > 8 )); then
  echo "[v19] need 1-8 GPUs, got ${NUM_GPUS}" >&2
  exit 1
fi
# 4 workers / GPU, 3 EGL contexts / GPU (1 worker waits). Isolated CUDA-free
# children own EGL. Machine-wide cap is per_gpu × n_gpus so all devices render.
RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-3}"
if (( RLT_EGL_PER_GPU < 1 )); then
  echo "[v19] RLT_EGL_PER_GPU must be >= 1" >&2
  exit 1
fi
if (( INSTANCES_PER_GPU < 1 )); then
  echo "[v19] INSTANCES_PER_GPU must be >= 1" >&2
  exit 1
fi
TOTAL_SHARDS=$((NUM_GPUS * INSTANCES_PER_GPU))
export RLT_EGL_PER_GPU
export RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-$((NUM_GPUS * RLT_EGL_PER_GPU))}"
export V19_N_SHARDS="${TOTAL_SHARDS}"
V19_TOTAL_VALID_EPISODES="${V19_TOTAL_VALID_EPISODES:-$((TOTAL_SHARDS * V19_PHASE_ROUNDS * V19_PHASE_B_EPISODES))}"
if (( V19_TOTAL_VALID_EPISODES < TOTAL_SHARDS )); then
  echo "[v19] V19_TOTAL_VALID_EPISODES (${V19_TOTAL_VALID_EPISODES}) < shards (${TOTAL_SHARDS})" >&2
  exit 1
fi

shard_quota() {
  local shard="$1"
  local base=$((V19_TOTAL_VALID_EPISODES / TOTAL_SHARDS))
  local rem=$((V19_TOTAL_VALID_EPISODES % TOTAL_SHARDS))
  if (( shard < rem )); then
    printf '%s\n' "$((base + 1))"
  else
    printf '%s\n' "${base}"
  fi
}

pid_first_field() {
  local pidfile="$1" pid=""
  [[ -f "${pidfile}" ]] || return 1
  IFS=$' \t\r\n' read -r pid _ < "${pidfile}"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${pid}"
}

pid_belongs_to_run() {
  local pid="$1" environ="/proc/${pid}/environ" cmdline="/proc/${pid}/cmdline" entry
  if [[ -r "${environ}" ]]; then
    while IFS= read -r entry; do
      [[ "${entry}" == "RLT_CF_V19_RUN_DIR=${RUN_DIR}" ]] && return 0
    done < <(tr '\0' '\n' < "${environ}")
  fi
  if [[ -r "${cmdline}" ]]; then
    entry="$(tr '\0' ' ' < "${cmdline}")"
    [[ "${entry}" == *"${RUN_DIR}"* ]] && return 0
  fi
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
  local fresh_start="$1"
  local -a helper_args=(
    "${PYTHON}" "${HELPER}" train-command
    --variant "${VARIANT}"
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
  local -a helper_args=(
    "${PYTHON}" "${HELPER}" server-command
    --variant "${VARIANT}"
    --root "${ROOT}"
    --checkpoint "${FLOW_CKPT}"
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
  local port="$1" attempt
  for ((attempt=1; attempt<=SERVER_WAIT_ATTEMPTS; attempt++)); do
    server_ready "${port}" && return 0
    sleep 5
  done
  echo "[v19] server port ${port} not ready" >&2
  return 1
}

start_server() {
  local gpu="$1" port="$2"
  local pidfile="${RUN_DIR}/pids/server_gpu${gpu}.pid"
  local logfile="${LOCAL_LOG_DIR}/server_gpu${gpu}_port${port}.log"
  pid_is_live_owned "${pidfile}" && return 0
  rm -f "${pidfile}"
  server_command
  set_cli_arg --port "${port}"
  echo "[v19 $(date -Is)] server gpu=${gpu} port=${port}"
  (
    cd "${MOLMOACT2}"
    exec setsid env \
      RLT_CF_V19_RUN_DIR="${RUN_DIR}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME}" \
      "${GENERATED_COMMAND[@]}"
  ) >> "${logfile}" 2>&1 &
  printf '%s server gpu=%s port=%s\n' "$!" "${gpu}" "${port}" > "${pidfile}"
}

start_trainer() {
  local gpu="$1" shard="$2" port="$3" fresh_start="$4"
  local out_dir="${RUN_DIR}/${VARIANT}/shard_${shard}"
  local pidfile="${RUN_DIR}/pids/train_s${shard}.pid"
  local logfile="${LOCAL_LOG_DIR}/train_s${shard}_gpu${gpu}.log"
  mkdir -p "${out_dir}"
  pid_is_live_owned "${pidfile}" && return 0
  rm -f "${pidfile}"
  trainer_command "${fresh_start}"
  set_cli_arg --out_dir "${out_dir}"
  set_cli_arg --replay_out "${out_dir}/chunk_replay.npz"
  set_cli_arg --tmp_rollout_dir "${TMP_ROLLOUT_DIR}/${VARIANT}_s${shard}"
  set_cli_arg --server_port "${port}"
  set_cli_arg --seed "$((20260817 + shard * 17))"
  set_cli_arg --start_episode "0"
  set_cli_arg --shard_size "${V19_POSE_CYCLE}"
  set_cli_arg --benchmark_pose_cycle "${V19_POSE_CYCLE}"
  set_cli_arg --cfgrl_n_shards "${TOTAL_SHARDS}"
  local quota
  quota="$(shard_quota "${shard}")"
  set_cli_arg --max_valid_episodes "${quota}"
  set_cli_arg --target_env_steps "$((quota * 500 + 500))"
  set_cli_arg --snapshot_episodes "0,${quota}"
  set_cli_arg --log_every_episodes "${V19_LOG_EVERY_EPISODES}"
  set_cli_arg --ckpt_every_episodes "${quota}"
  echo "[v19 $(date -Is)] train s${shard} gpu=${gpu} port=${port} fresh=${fresh_start} quota=${quota}"
  (
    exec setsid env \
      RLT_CF_V19_RUN_DIR="${RUN_DIR}" \
      V19_N_SHARDS="${TOTAL_SHARDS}" \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      MUJOCO_EGL_DEVICE_ID="${gpu}" \
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
      RLT_ISOLATED_ROLLOUT="${RLT_ISOLATED_ROLLOUT:-1}" \
      RLT_ISOLATED_ATTEMPTS="${RLT_ISOLATED_ATTEMPTS:-4}" \
      RLT_ISOLATED_TIMEOUT_SEC="${RLT_ISOLATED_TIMEOUT_SEC:-720}" \
      RLT_ISOLATED_STARTUP_SEC="${RLT_ISOLATED_STARTUP_SEC:-180}" \
      RLT_ISOLATED_GL="${RLT_ISOLATED_GL:-egl}" \
      V19_POSE_CYCLE="${V19_POSE_CYCLE}" \
      PYTHONFAULTHANDLER=1 \
      PYTHONUNBUFFERED=1 \
      "${GENERATED_COMMAND[@]}"
  ) >> "${logfile}" 2>&1 &
  printf '%s train shard=%s gpu=%s port=%s fresh=%s\n' \
    "$!" "${shard}" "${gpu}" "${port}" "${fresh_start}" > "${pidfile}"
}

if [[ "${FRESH}" == "1" ]]; then
  rm -rf "${RUN_DIR}/${VARIANT}"
  mkdir -p "${RUN_DIR}/${VARIANT}"
  rm -f "${RUN_DIR}/pids/"*.pid
  rm -f "${RUN_DIR}/PHASE_SR.md" "${RUN_DIR}/phase_sr.jsonl"
  # Truncate rather than unlink so an outer nohup >> launch.log keeps its path.
  : > "${LOCAL_LOG_DIR}/launch.log" 2>/dev/null || true
  rm -f "${LOCAL_LOG_DIR}"/train_s*.log "${LOCAL_LOG_DIR}"/server_gpu*.log
  # Drop stale EGL flocks from prior crashed trainers (dead PIDs leave files behind).
  rm -f "${RLT_EGL_LOCK_DIR}"/gpu_*.lock "${RLT_EGL_LOCK_DIR}"/slot_*.lock
fi

cat > "${RUN_DIR}/MANIFEST.json" <<EOF
{
  "schema_version": "v19-cfgrl-2",
  "run_dir": "${RUN_DIR}",
  "mode": "${V19_MODE}",
  "variant": "${VARIANT}",
  "packing": {"gpus": ${NUM_GPUS}, "instances_per_gpu": ${INSTANCES_PER_GPU}, "shards": ${TOTAL_SHARDS}, "servers": ${NUM_GPUS}, "egl_per_gpu": ${RLT_EGL_PER_GPU}, "egl_max_concurrent": ${RLT_EGL_MAX_CONCURRENT}},
  "phase_rounds": ${V19_PHASE_ROUNDS},
  "phase_b_episodes_per_shard": ${V19_PHASE_B_EPISODES},
  "phase_probe": true,
  "phase_probe_n": ${V19_POSE_CYCLE},
  "phase_barrier": true,
  "online_collect_eps_pooled": ${V19_TOTAL_VALID_EPISODES},
  "benchmark_train": "${TRAIN_BENCHMARK}",
  "benchmark_val_holdout": "${VAL_BENCHMARK}",
  "val_episodes_unused": 12,
  "pose_cycle": ${V19_POSE_CYCLE},
  "flow_checkpoint": "${FLOW_CKPT}",
  "seed_replay": "${V19_SEED_REPLAY}",
  "ports": {"base": ${BASE_HTTP_PORT}, "count": ${NUM_GPUS}},
  "note": "Phase SR covers ${V19_POSE_CYCLE} train pose(s); every worker uses start_episode=0. Barrier waits after each phase. Collect is ${V19_TOTAL_VALID_EPISODES} eps pooled from that pose set. Val split unused.",
  "launched_at": "$(date -Is)"
}
EOF

echo "[v19 $(date -Is)] starting ${NUM_GPUS} Molmo servers"
for ((g=0; g<NUM_GPUS; g++)); do
  gpu="${GPU_ARRAY[$g]}"
  port=$((BASE_HTTP_PORT + g))
  start_server "${gpu}" "${port}"
  sleep "${SERVER_STAGGER_SEC}"
done
for ((g=0; g<NUM_GPUS; g++)); do
  port=$((BASE_HTTP_PORT + g))
  wait_for_server "${port}"
done

echo "[v19 $(date -Is)] starting ${TOTAL_SHARDS} trainers (${INSTANCES_PER_GPU}/GPU), pooled collect=${V19_TOTAL_VALID_EPISODES}"
for ((shard=0; shard<TOTAL_SHARDS; shard++)); do
  g=$((shard / INSTANCES_PER_GPU))
  gpu="${GPU_ARRAY[$g]}"
  port=$((BASE_HTTP_PORT + g))
  start_trainer "${gpu}" "${shard}" "${port}" "${FRESH}"
  sleep "${TRAINER_STAGGER_SEC}"
done

touch "${RUN_DIR}/.initial_launch_complete"
printf '%s launch\n' "$$" > "${RUN_DIR}/pids/launch.pid"
echo "[v19 $(date -Is)] initial launch complete; polling every ${POLL_SEC}s"
echo "[v19] val holdout unused: ${VAL_BENCHMARK}"
echo "[v19] packing: ${NUM_GPUS} GPU x ${INSTANCES_PER_GPU} = ${TOTAL_SHARDS} shards, ports ${BASE_HTTP_PORT}-$((BASE_HTTP_PORT + NUM_GPUS - 1))"
echo "[v19] EGL: PER_GPU=${RLT_EGL_PER_GPU} MAX_CONCURRENT=${RLT_EGL_MAX_CONCURRENT} (both GPUs render; $((INSTANCES_PER_GPU - RLT_EGL_PER_GPU)) of ${INSTANCES_PER_GPU} workers wait per GPU)"
echo "[v19] phase SR: pose_cycle=${V19_POSE_CYCLE} (all workers start_episode=0) across ${V19_PHASE_ROUNDS} A/B rounds; barrier waits after each probe"

while true; do
  sleep "${POLL_SEC}"
  for ((g=0; g<NUM_GPUS; g++)); do
    gpu="${GPU_ARRAY[$g]}"
    port=$((BASE_HTTP_PORT + g))
    spidfile="${RUN_DIR}/pids/server_gpu${gpu}.pid"
    if ! pid_is_live_owned "${spidfile}"; then
      echo "[watchdog $(date -Is)] restart server gpu=${gpu} port=${port}"
      start_server "${gpu}" "${port}"
      wait_for_server "${port}" || true
    fi
  done
  done_shards=0
  for ((shard=0; shard<TOTAL_SHARDS; shard++)); do
    g=$((shard / INSTANCES_PER_GPU))
    gpu="${GPU_ARRAY[$g]}"
    port=$((BASE_HTTP_PORT + g))
    tpidfile="${RUN_DIR}/pids/train_s${shard}.pid"
    summary="${RUN_DIR}/${VARIANT}/shard_${shard}/summary.json"
    if [[ -f "${summary}" ]]; then
      done_shards=$((done_shards + 1))
      continue
    fi
    if ! pid_is_live_owned "${tpidfile}"; then
      echo "[watchdog $(date -Is)] restart train s${shard}"
      start_trainer "${gpu}" "${shard}" "${port}" 0 || true
      # Do not bring every dead shard back in the same poll — that re-creates
      # the dual-EGL start storm that SIGABRTs this host.
      sleep "${TRAINER_STAGGER_SEC}"
    fi
  done
  "${PYTHON}" "${HELPER}" aggregate-phase-sr \
    --run-dir "${RUN_DIR}" \
    --shards "${TOTAL_SHARDS}" \
    --rounds "${V19_PHASE_ROUNDS}" \
    --poses "${V19_POSE_CYCLE}" \
    --variant "${VARIANT}" >/dev/null || true
  if (( done_shards == TOTAL_SHARDS )); then
    echo "[v19 $(date -Is)] all ${TOTAL_SHARDS} shards complete"
    "${PYTHON}" "${HELPER}" aggregate-phase-sr \
      --run-dir "${RUN_DIR}" \
      --shards "${TOTAL_SHARDS}" \
      --rounds "${V19_PHASE_ROUNDS}" \
      --poses "${V19_POSE_CYCLE}" \
      --variant "${VARIANT}" || true
    break
  fi
done

echo "[v19] phase SR table: ${RUN_DIR}/PHASE_SR.md"
