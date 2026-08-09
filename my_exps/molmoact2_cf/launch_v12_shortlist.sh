#!/usr/bin/env bash
# V12 shortlist: 8 variants, 1 variant per GPU, N shards/GPU.
# Fresh from offline pretrain only (--no_resume). Includes v12 method fixes.
#
# GPU layout (default INSTANCES_PER_GPU=4 for HTTP; AE forced to 1/GPU):
#   0 residual mean_pool_baseline
#   1 residual rlt_actor_no_guide
#   2 residual rlt_cf_frozen_token
#   3 flow     mean_pool_baseline
#   4 flow     rlt_actor_no_guide  (--joint_cf)
#   5 flow     rlt_cf_frozen_token (--joint_cf)
#   6 flow AE  ae_no_guide         (in-process; 1 job/GPU)
#   7 flow AE  ae_cf_frozen_token  (in-process; 1 job/GPU)
#
# Override AE packing with AE_INSTANCES_PER_GPU (default 1 — Molmo AE ~15GB).

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
mkdir -p "${B1K_TMP}"

RUN_DIR="${RUN_DIR:-runs/rlt_cf_v12_shortlist}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_cf_v12_shortlist_logs}"
SCREEN_NAME="${SCREEN_NAME:-rlt_cf_v12_shortlist}"
NUM_GPUS="${NUM_GPUS:-8}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}"
AE_INSTANCES_PER_GPU="${AE_INSTANCES_PER_GPU:-4}"
BASE_PORT="${BASE_PORT:-8600}"
BENCH_N="${BENCH_N:-1000}"
TARGET_ENV_STEPS="${TARGET_ENV_STEPS:-4166667}"
UPDATES_PER_EPISODE="${UPDATES_PER_EPISODE:-8}"
AE_UPDATES_PER_EPISODE="${AE_UPDATES_PER_EPISODE:-4}"
LOG_EVERY_EPISODES="${LOG_EVERY_EPISODES:-50}"
HORIZON="${HORIZON:-500}"
N_CRITICS="${N_CRITICS:-10}"
SERVER_STAGGER_SEC="${SERVER_STAGGER_SEC:-3}"
TRAINER_STAGGER_SEC="${TRAINER_STAGGER_SEC:-8}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"
FLOW_STEPS="${FLOW_STEPS:-10}"
GUIDANCE_COEF="${GUIDANCE_COEF:-0.5}"
G_MIN_ADVANTAGE="${G_MIN_ADVANTAGE:-0.003}"
G_MIN_GUIDE_ADVANTAGE="${G_MIN_GUIDE_ADVANTAGE:-0.001}"
GUIDE_BETA="${GUIDE_BETA:-0.05}"
GUIDE_TARGET_DELTA_FRAC="${GUIDE_TARGET_DELTA_FRAC:-1.0}"
EXPLORE_STD="${EXPLORE_STD:-0.02}"
EXPLORE_DEPLOY_STD="${EXPLORE_DEPLOY_STD:-0.02}"
BC_REF_COEF="${BC_REF_COEF:-1.0}"
AE_BATCH_SIZE="${AE_BATCH_SIZE:-1}"
AE_LORA_RANK="${AE_LORA_RANK:-16}"
# AE 4×/GPU is tight on 80GB; set AE_INSTANCES_PER_GPU=1 if OOM.

RESIDUAL_CKPT="${RESIDUAL_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt}"

RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v12}"
RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-3}"
RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-$(( NUM_GPUS * RLT_EGL_PER_GPU ))}"
RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts_v12}"
mkdir -p "${LOCAL_LOG_DIR}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"

MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${MOLMOSPACES}/.venv/bin/python"
MOLMOACT2_PYTHON="${MOLMOACT2}/.venv/bin/python"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ -n "${UV_BIN}" && -x "${UV_BIN}" ]]; then
  SERVE_CMD=("${UV_BIN}" run python)
else
  if [[ ! -x "${MOLMOACT2_PYTHON}" ]]; then
    echo "[launch] neither uv nor MolmoAct2 Python found (${MOLMOACT2_PYTHON})" >&2
    exit 1
  fi
  SERVE_CMD=("${MOLMOACT2_PYTHON}")
fi

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

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT}/${RUN_DIR}"
fi
mkdir -p "${RUN_DIR}" "${RUN_DIR}/pids" "${LOCAL_LOG_DIR}"
RUN_DIR="$(cd "${RUN_DIR}" && pwd -P)"
export RLT_CF_V4_RUN_DIR="${RUN_DIR}"

if [[ -e "${RUN_DIR}/logs" && ! -L "${RUN_DIR}/logs" ]]; then
  echo "[launch] ${RUN_DIR}/logs exists and is not a symlink; refusing to replace it" >&2
  exit 1
fi
ln -sfn "${LOCAL_LOG_DIR}" "${RUN_DIR}/logs"

for ckpt in "${RESIDUAL_CKPT}" "${FLOW_CKPT}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "[launch] missing offline pretrain ckpt: ${ckpt}" >&2
    exit 1
  fi
done
RESIDUAL_CKPT="$(cd "$(dirname "${RESIDUAL_CKPT}")" && pwd -P)/$(basename "${RESIDUAL_CKPT}")"
FLOW_CKPT="$(cd "$(dirname "${FLOW_CKPT}")" && pwd -P)/$(basename "${FLOW_CKPT}")"
if [[ ! -x "${PYTHON}" ]]; then
  echo "[launch] MolmoSpaces Python not found: ${PYTHON}" >&2
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

# One variant per GPU. AE variants force AE_INSTANCES_PER_GPU (default 1).
# name|cf_mode|config|actor|guide|joint|ae|ckpt_kind
VARIANT_SPECS=(
  "res_baseline|residual|residual_mean_pool_baseline|vla_only|0|0|0|residual"
  "res_no_guide|residual|residual_rlt_actor_no_guide|rlt|0|0|0|residual"
  "res_cf_frozen|residual|residual_rlt_cf_frozen_token|rlt|1|0|0|residual"
  "flow_baseline|flow|flow_mean_pool_baseline|vla_only|0|0|0|flow"
  "flow_no_guide|flow|flow_rlt_actor_no_guide|rlt|0|1|0|flow"
  "flow_cf_frozen|flow|flow_rlt_cf_frozen_token|rlt|1|1|0|flow"
  "ae_no_guide|flow|ae_no_guide|rlt|0|1|1|flow"
  "ae_cf_frozen|flow|ae_cf_frozen_token|rlt|1|1|1|flow"
)

if (( ${#VARIANT_SPECS[@]} != NUM_GPUS )); then
  echo "[launch] need ${NUM_GPUS} variant specs, got ${#VARIANT_SPECS[@]}" >&2
  exit 1
fi

if [[ "${RLT_V12_IN_SCREEN:-0}" != "1" && "${NO_SCREEN:-0}" != "1" ]]; then
  if ! command -v screen >/dev/null 2>&1; then
    echo "[launch] GNU screen required (or NO_SCREEN=1)" >&2
    exit 1
  fi
  if screen -ls 2>/dev/null | grep -q "[.]${SCREEN_NAME}[[:space:]]"; then
    echo "[launch] screen ${SCREEN_NAME} already exists" >&2
    exit 1
  fi
  screen -dmS "${SCREEN_NAME}" \
    env RLT_V12_IN_SCREEN=1 RUN_DIR="${RUN_DIR}" LOCAL_LOG_DIR="${LOCAL_LOG_DIR}" \
      NUM_GPUS="${NUM_GPUS}" INSTANCES_PER_GPU="${INSTANCES_PER_GPU}" \
      AE_INSTANCES_PER_GPU="${AE_INSTANCES_PER_GPU}" \
      GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}" \
      RESIDUAL_CKPT="${RESIDUAL_CKPT}" FLOW_CKPT="${FLOW_CKPT}" \
      RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
      RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
      RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU}" \
      DETACH_AFTER_START="${DETACH_AFTER_START:-1}" \
      bash "${ROOT}/launch_v12_shortlist.sh"
  echo "[launch] started screen ${SCREEN_NAME}"
  echo "[launch] logs: ${RUN_DIR}/logs"
  echo "[launch] stop: ${ROOT}/stop_run.sh ${RUN_DIR}"
  exit 0
fi

exec >> "${LOCAL_LOG_DIR}/launcher.log" 2>&1
echo "[launch $(date -Is)] V12 shortlist 1-variant-per-GPU from offline pretrain"
echo "[launch] run_dir=${RUN_DIR}"
echo "[launch] residual_ckpt=${RESIDUAL_CKPT}"
echo "[launch] flow_ckpt=${FLOW_CKPT}"
echo "[launch] explore=${EXPLORE_STD}/${EXPLORE_DEPLOY_STD} bc_ref_coef=${BC_REF_COEF}"
echo "[launch] physical GPUs: ${GPU_ARR[*]}"

declare -a SERVER_PIDS=()
declare -a TRAINER_PIDS=()
declare -a SERVER_PORTS=()

# ---- HTTP servers: one per non-AE shard ----
server_slot=0
for ((gpu_i=0; gpu_i<NUM_GPUS; gpu_i++)); do
  IFS='|' read -r vname cf_mode config_name actor_mode use_guide joint_cf ae_mode ckpt_kind <<< "${VARIANT_SPECS[$gpu_i]}"
  if [[ "${ae_mode}" == "1" ]]; then
    continue
  fi
  gpu="${GPU_ARR[$gpu_i]}"
  n_inst="${INSTANCES_PER_GPU}"
  for ((s=0; s<n_inst; s++)); do
    port=$((BASE_PORT + server_slot))
    SERVER_PORTS+=("${port}")
    feature_mode="rl_token"
    if [[ "${actor_mode}" == "vla_only" ]]; then
      feature_mode="tokens"
    fi
    ckpt="${RESIDUAL_CKPT}"
    if [[ "${ckpt_kind}" == "flow" ]]; then
      ckpt="${FLOW_CKPT}"
    fi
    logfile="${LOCAL_LOG_DIR}/server_${vname}_s${s}_gpu${gpu}.log"
    pidfile="${RUN_DIR}/pids/server_${vname}_s${s}.pid"
    serve_cmd=(
      "${SERVE_CMD[@]}" "${ROOT}/serve.py"
      --host 0.0.0.0
      --port "${port}"
      --device cuda:0
      --dtype bfloat16
      --disable_g
      --feature_mode "${feature_mode}"
    )
    if [[ "${feature_mode}" == "rl_token" ]]; then
      serve_cmd+=(--rlt_ckpt "${ckpt}")
    fi
    echo "[launch] server ${vname} shard=${s} gpu=${gpu} port=${port} mode=${feature_mode}"
    (
      cd "${MOLMOACT2}"
      exec setsid env \
        RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
        "${serve_cmd[@]}"
    ) > "${logfile}" 2>&1 &
    pid=$!
    printf '%s server variant=%s shard=%s gpu=%s port=%s\n' \
      "${pid}" "${vname}" "${s}" "${gpu}" "${port}" > "${pidfile}"
    SERVER_PIDS+=("${pid}")
    server_slot=$((server_slot + 1))
    sleep "${SERVER_STAGGER_SEC}"
  done
done

echo "[launch] waiting for ${#SERVER_PIDS[@]} HTTP servers"
for port in "${SERVER_PORTS[@]}"; do
  ready=0
  for ((attempt=1; attempt<=SERVER_WAIT_ATTEMPTS; attempt++)); do
    if curl -sf --max-time 3 "http://127.0.0.1:${port}/healthz" 2>/dev/null \
      | grep -q '"status":"ok"'; then
      ready=1
      break
    fi
    if curl -sf --max-time 3 "http://127.0.0.1:${port}/act" 2>/dev/null \
      | grep -q '"status":"ok"'; then
      ready=1
      break
    fi
    sleep 5
  done
  if (( ready == 0 )); then
    echo "[launch] ERROR server port=${port} not ready" >&2
    "${ROOT}/stop_run.sh" "${RUN_DIR}" || true
    exit 1
  fi
  echo "[launch] server port=${port} READY"
done

start_trainer() {
  local gpu_i="$1"
  local shard="$2"
  local port_or_dummy="$3"
  IFS='|' read -r vname cf_mode config_name actor_mode use_guide joint_cf ae_mode ckpt_kind <<< "${VARIANT_SPECS[$gpu_i]}"
  local gpu="${GPU_ARR[$gpu_i]}"
  local n_inst="${INSTANCES_PER_GPU}"
  if [[ "${ae_mode}" == "1" ]]; then
    n_inst="${AE_INSTANCES_PER_GPU}"
  fi
  local shard_base=$((BENCH_N / n_inst))
  local start_episode=$((shard * shard_base))
  local shard_size="${shard_base}"
  if (( shard == n_inst - 1 )); then
    shard_size=$((BENCH_N - start_episode))
  fi
  local shard_out="${RUN_DIR}/${config_name}/shard_${shard}"
  local logfile="${LOCAL_LOG_DIR}/train_${config_name}_s${shard}_gpu${gpu}.log"
  local pidfile="${RUN_DIR}/pids/train_${config_name}_s${shard}.pid"
  mkdir -p "${shard_out}"
  # Fresh start: wipe any leftover online artifacts in this new run dir shard.
  rm -f "${shard_out}/metrics.jsonl" "${shard_out}/rlt_cf_latest.pt" \
    "${shard_out}/LATEST_CKPT.txt" "${shard_out}/chunk_replay.npz" \
    "${shard_out}/molmo_ae_lora_latest.pt" 2>/dev/null || true

  local ckpt="${RESIDUAL_CKPT}"
  if [[ "${ckpt_kind}" == "flow" ]]; then
    ckpt="${FLOW_CKPT}"
  fi
  local updates="${UPDATES_PER_EPISODE}"
  if [[ "${ae_mode}" == "1" ]]; then
    updates="${AE_UPDATES_PER_EPISODE}"
  fi
  if [[ "${actor_mode}" == "vla_only" ]]; then
    updates=0
  fi

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
    --ckpt_every_episodes "${CKPT_EVERY_EPISODES:-10}"
    --replay_out "${shard_out}/chunk_replay.npz"
    --seed "$((gpu_i * 10 + shard))"
    --n_critics "${N_CRITICS}"
    --cf_mode "${cf_mode}"
    --flow_steps "${FLOW_STEPS}"
    --guidance_coef "${GUIDANCE_COEF}"
    --explore_residual_std "${EXPLORE_STD}"
    --explore_deploy_std "${EXPLORE_DEPLOY_STD}"
    --explore_warmup_mult 1.0
    --bc_ref_coef "${BC_REF_COEF}"
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
    --rlt_ckpt "${ckpt}"
    --no_resume
    --updates_per_episode "${updates}"
    --freeze_token
  )

  if [[ "${ae_mode}" == "1" ]]; then
    command+=(
      --ae_trainable
      --ae_lora
      --ae_lora_rank "${AE_LORA_RANK}"
      --ae_batch_size "${AE_BATCH_SIZE}"
      --joint_cf
      --server_port "$((9000 + gpu_i * 10 + shard))"
    )
  else
    command+=(
      --server_host localhost
      --server_port "${port_or_dummy}"
    )
  fi

  if [[ "${joint_cf}" == "1" && "${ae_mode}" != "1" ]]; then
    command+=(--joint_cf)
  fi

  case "${actor_mode}" in
    vla_only)
      command+=(--actor_mode vla_only --no_cf_guide)
      ;;
    rlt)
      command+=(--actor_mode rlt)
      if [[ "${use_guide}" == "1" ]]; then
        command+=(--use_cf_guide --g_min_guide_advantage "${G_MIN_GUIDE_ADVANTAGE}")
      else
        command+=(--no_cf_guide)
      fi
      ;;
  esac

  echo "[launch] train ${config_name} shard=${shard} gpu=${gpu} port=${port_or_dummy} ae=${ae_mode}"
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
    "${pid}" "${config_name}" "${shard}" "${gpu}" > "${pidfile}"
  TRAINER_PIDS+=("${pid}")
}

# Map HTTP ports back in launch order for non-AE variants.
server_slot=0
for ((gpu_i=0; gpu_i<NUM_GPUS; gpu_i++)); do
  IFS='|' read -r vname cf_mode config_name actor_mode use_guide joint_cf ae_mode ckpt_kind <<< "${VARIANT_SPECS[$gpu_i]}"
  n_inst="${INSTANCES_PER_GPU}"
  if [[ "${ae_mode}" == "1" ]]; then
    n_inst="${AE_INSTANCES_PER_GPU}"
  fi
  for ((s=0; s<n_inst; s++)); do
    if [[ "${ae_mode}" == "1" ]]; then
      start_trainer "${gpu_i}" "${s}" "0"
    else
      port="${SERVER_PORTS[$server_slot]}"
      start_trainer "${gpu_i}" "${s}" "${port}"
      server_slot=$((server_slot + 1))
    fi
    sleep "${TRAINER_STAGGER_SEC}"
  done
done

echo "[launch] started ${#SERVER_PIDS[@]} servers + ${#TRAINER_PIDS[@]} trainers"
echo "[launch] stop: ${ROOT}/stop_run.sh ${RUN_DIR}"
cat > "${RUN_DIR}/README.md" <<'EOF'
# V12 shortlist (fresh from offline pretrain)

- Fixes: advantage gate for joint/AE, explore=0.02, BC toward reference, flow guide magnitude distill, guide-adv gate
- Packing: 1 variant / GPU; HTTP 4/GPU; AE 4/GPU (reduce AE_INSTANCES_PER_GPU if OOM)
- All trainers use --no_resume (offline pretrain ckpt only)

| GPU | Variant |
| --- | --- |
| 0 | residual_mean_pool_baseline |
| 1 | residual_rlt_actor_no_guide |
| 2 | residual_rlt_cf_frozen_token |
| 3 | flow_mean_pool_baseline |
| 4 | flow_rlt_actor_no_guide + joint_cf |
| 5 | flow_rlt_cf_frozen_token + joint_cf |
| 6 | ae_no_guide |
| 7 | ae_cf_frozen_token |
EOF
# stamp start time without nested heredoc expansion issues
echo "- Started: $(date -Is)" >> "${RUN_DIR}/README.md"
echo "- Residual ckpt: ${RESIDUAL_CKPT}" >> "${RUN_DIR}/README.md"
echo "- Flow ckpt: ${FLOW_CKPT}" >> "${RUN_DIR}/README.md"
echo "- Packing: HTTP ${INSTANCES_PER_GPU}/GPU; AE ${AE_INSTANCES_PER_GPU}/GPU" >> "${RUN_DIR}/README.md"

if [[ "${DETACH_AFTER_START:-1}" == "1" ]]; then
  echo "[launch] detach after start"
  exit 0
fi

# Optional watchdog for trainer restarts.
WATCH_LOG="${LOCAL_LOG_DIR}/watchdog.log"
POLL_SEC="${POLL_SEC:-60}"
while true; do
  server_slot=0
  for ((gpu_i=0; gpu_i<NUM_GPUS; gpu_i++)); do
    IFS='|' read -r vname cf_mode config_name actor_mode use_guide joint_cf ae_mode ckpt_kind <<< "${VARIANT_SPECS[$gpu_i]}"
    n_inst="${INSTANCES_PER_GPU}"
    if [[ "${ae_mode}" == "1" ]]; then
      n_inst="${AE_INSTANCES_PER_GPU}"
    fi
    for ((s=0; s<n_inst; s++)); do
      pidfile="${RUN_DIR}/pids/train_${config_name}_s${s}.pid"
      alive=0
      if [[ -f "${pidfile}" ]]; then
        read -r pid _rest < "${pidfile}" || pid=""
        if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
          alive=1
        fi
      fi
      if (( alive == 0 )); then
        echo "[watchdog $(date -Is)] restarting ${config_name} shard=${s}" | tee -a "${WATCH_LOG}"
        if [[ "${ae_mode}" == "1" ]]; then
          start_trainer "${gpu_i}" "${s}" "0"
        else
          # Recompute port index for this HTTP shard.
          port_idx=0
          for ((g2=0; g2<gpu_i; g2++)); do
            IFS='|' read -r _a _b _c _d _e _f ae2 _g <<< "${VARIANT_SPECS[$g2]}"
            if [[ "${ae2}" != "1" ]]; then
              port_idx=$((port_idx + INSTANCES_PER_GPU))
            fi
          done
          port_idx=$((port_idx + s))
          start_trainer "${gpu_i}" "${s}" "${SERVER_PORTS[$port_idx]}"
        fi
        sleep "${TRAINER_STAGGER_SEC}"
      fi
    done
  done
  sleep "${POLL_SEC}"
done
