#!/usr/bin/env bash
# Dual experiment launch (v8):
#   GPUs 0-3: residual one-shot CF (demo1k residual critic)
#   GPUs 4-7: full flow-time ConsensusFlow (requires FLOW_CKPT)
#
# Usage:
#   FLOW_CKPT=$PWD/runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt \
#     bash launch_dual_cf_v8.sh
#
# Optional: RESIDUAL_ONLY=1 or FLOW_ONLY=1 to launch one side.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
mkdir -p "${B1K_TMP}"

RESIDUAL_RUN_DIR="${RESIDUAL_RUN_DIR:-runs/rlt_cf_v8_residual}"
FLOW_RUN_DIR="${FLOW_RUN_DIR:-runs/rlt_cf_v8_flow}"
RESIDUAL_CKPT="${RESIDUAL_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt}"
FLOW_CKPT="${FLOW_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt}"
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}"
RESIDUAL_ONLY="${RESIDUAL_ONLY:-0}"
FLOW_ONLY="${FLOW_ONLY:-0}"
BASE_PORT_RESIDUAL="${BASE_PORT_RESIDUAL:-8600}"
BASE_PORT_FLOW="${BASE_PORT_FLOW:-8700}"
# Prefer screen when available; otherwise NO_SCREEN background launch.
if [[ -z "${NO_SCREEN:-}" ]]; then
  if command -v screen >/dev/null 2>&1; then
    NO_SCREEN=0
  else
    NO_SCREEN=1
  fi
fi
export NO_SCREEN

if [[ "${FLOW_ONLY}" != "1" && ! -f "${RESIDUAL_CKPT}" ]]; then
  echo "[dual] missing RESIDUAL_CKPT=${RESIDUAL_CKPT}" >&2
  exit 1
fi
if [[ "${RESIDUAL_ONLY}" != "1" && ! -f "${FLOW_CKPT}" ]]; then
  echo "[dual] missing FLOW_CKPT=${FLOW_CKPT}" >&2
  echo "[dual] run: bash launch_flow_pretrain.sh first" >&2
  exit 1
fi

launch_side() {
  local name="$1"
  local gpus="$2"
  local run_dir="$3"
  local ckpt="$4"
  local base_port="$5"
  local cf_mode="$6"
  local screen_name="$7"
  local log_dir="$8"
  local joint_cf="${9:-0}"
  local num_gpus
  num_gpus="$(awk -F',' '{print NF}' <<<"${gpus}")"
  # Prefer ~3 concurrent rollouts per GPU; allow override via RLT_EGL_*.
  local per_gpu="${RLT_EGL_PER_GPU:-3}"
  local side_concurrent="${RLT_EGL_MAX_CONCURRENT:-$(( num_gpus * per_gpu ))}"
  local egl_lock_dir="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_${cf_mode}}"
  mkdir -p "${log_dir}" "${egl_lock_dir}"

  echo "[dual] launching ${name}: GPUs=${gpus} run=${run_dir} mode=${cf_mode} ckpt=${ckpt} JOINT_CF=${joint_cf} NO_SCREEN=${NO_SCREEN} egl_concurrent=${side_concurrent} per_gpu=${per_gpu}"
  if [[ "${NO_SCREEN}" == "1" ]]; then
    nohup env \
      NO_SCREEN=1 \
      CUDA_VISIBLE_DEVICES="${gpus}" \
      GPU_IDS="${gpus}" \
      RLT_CF_V7_RUN_DIR="${run_dir}" \
      RUN_DIR="${run_dir}" \
      RLT_CKPT="${ckpt}" \
      NUM_GPUS="${num_gpus}" \
      INSTANCES_PER_GPU="${INSTANCES_PER_GPU}" \
      BASE_PORT="${base_port}" \
      SCREEN_NAME="${screen_name}" \
      LOCAL_LOG_DIR="${log_dir}" \
      CF_MODE="${cf_mode}" \
      JOINT_CF="${joint_cf}" \
      G_MIN_ADVANTAGE="${G_MIN_ADVANTAGE:-0.003}" \
      GUIDE_BETA="${GUIDE_BETA:-0.05}" \
      GUIDE_TARGET_DELTA_FRAC="${GUIDE_TARGET_DELTA_FRAC:-1.0}" \
      B1K_ROOT="${B1K_ROOT}" \
      B1K_TMP="${B1K_TMP}" \
      RLT_EGL_LOCK_DIR="${egl_lock_dir}" \
      RLT_EGL_MAX_CONCURRENT="${side_concurrent}" \
      RLT_EGL_PER_GPU="${per_gpu}" \
      RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}" \
      TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts}" \
      bash "${ROOT}/launch_rlt_v8_side.sh" \
      > "${log_dir}/nohup_launcher.out" 2>&1 &
    echo "[dual] ${name} launcher pid=$!  logs=${log_dir}/launcher.log"
  else
    CUDA_VISIBLE_DEVICES="${gpus}" \
    GPU_IDS="${gpus}" \
    RLT_CF_V7_RUN_DIR="${run_dir}" \
    RUN_DIR="${run_dir}" \
    RLT_CKPT="${ckpt}" \
    NUM_GPUS="${num_gpus}" \
    INSTANCES_PER_GPU="${INSTANCES_PER_GPU}" \
    BASE_PORT="${base_port}" \
    SCREEN_NAME="${screen_name}" \
    LOCAL_LOG_DIR="${log_dir}" \
    CF_MODE="${cf_mode}" \
    JOINT_CF="${joint_cf}" \
    G_MIN_ADVANTAGE="${G_MIN_ADVANTAGE:-0.003}" \
    GUIDE_BETA="${GUIDE_BETA:-0.05}" \
    GUIDE_TARGET_DELTA_FRAC="${GUIDE_TARGET_DELTA_FRAC:-1.0}" \
    B1K_ROOT="${B1K_ROOT}" \
    B1K_TMP="${B1K_TMP}" \
    RLT_EGL_LOCK_DIR="${egl_lock_dir}" \
    RLT_EGL_MAX_CONCURRENT="${side_concurrent}" \
    RLT_EGL_PER_GPU="${per_gpu}" \
    RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}" \
    TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts}" \
    bash "${ROOT}/launch_rlt_v8_side.sh"
  fi
}

if [[ "${FLOW_ONLY}" != "1" ]]; then
  launch_side residual "0,1,2,3" "${RESIDUAL_RUN_DIR}" "${RESIDUAL_CKPT}" \
    "${BASE_PORT_RESIDUAL}" residual "$(basename "${RESIDUAL_RUN_DIR}")" \
    "${B1K_TMP}/$(basename "${RESIDUAL_RUN_DIR}")_logs" \
    0
fi
if [[ "${RESIDUAL_ONLY}" != "1" ]]; then
  # Stagger so both sides don't hammer NFS/assets at once.
  sleep 15
  # Flow side: joint trainable v_θ (FlowVelocityActor) + steering G_φ.
  launch_side flow "4,5,6,7" "${FLOW_RUN_DIR}" "${FLOW_CKPT}" \
    "${BASE_PORT_FLOW}" flow "$(basename "${FLOW_RUN_DIR}")" \
    "${B1K_TMP}/$(basename "${FLOW_RUN_DIR}")_logs" \
    "${JOINT_CF:-1}"
fi

echo "[dual] both sides requested"
echo "[dual] residual: stop: ${ROOT}/stop_run.sh ${RESIDUAL_RUN_DIR}"
echo "[dual] flow:     stop: ${ROOT}/stop_run.sh ${FLOW_RUN_DIR}"
echo "[dual] logs/tmp: ${B1K_TMP}"