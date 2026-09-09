#!/usr/bin/env bash
# Wait for the live continue_v22_24.sh (leftover evals + kettle β=1 ablation),
# then run Pick-18 V22_24 stage 0 with online β=1 on all 18 tasks.
#
# Reuses AE + AC pretrain (β=100, λ_π=0) from runs/pick18_v22_24/.
# Flock so a restarted continue.sh cannot start a second copy on the same GPU.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v22_24_beta1}"
WAIT_PID="${WAIT_PID:-}"
GPU="${GPU:-0}"

mkdir -p "${RUN_DIR}/pids" "${RUN_DIR}/queue" "${B1K_TMP}/pick18_v22_24_beta1_logs"
echo $$ > "${RUN_DIR}/pids/waiter.pid"
log() { echo "[v22_24-beta1-queue $(date -u +'%F %H:%M:%S')] $*"; }

if [[ -z "${WAIT_PID}" && -f "${ROOT}/runs/v22_24/kettle/pids/continue.pid" ]]; then
  WAIT_PID="$(cat "${ROOT}/runs/v22_24/kettle/pids/continue.pid")"
fi

if [[ -n "${WAIT_PID}" ]] && kill -0 "${WAIT_PID}" 2>/dev/null; then
  log "waiting for continue pid ${WAIT_PID} (leftover evals + kettle β=1 ablation)"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
  log "continue ${WAIT_PID} exited"
else
  log "no live continue pid, starting Pick-18 β=1 now"
fi

exec 9>"${RUN_DIR}/pids/sweep.lock"
if ! flock -n 9; then
  log "another Pick-18 β=1 sweep holds the lock, skip"
  exit 0
fi

log "start Pick-18 V22_24 β=1 on gpu=${GPU}"
GPUS="${GPU}" SLOTS_PER_GPU=1 \
  RUN_DIR="${RUN_DIR}" \
  LOCAL_LOG="${B1K_TMP}/pick18_v22_24_beta1_logs" \
  SOURCE_DIR="${ROOT}/runs/pick18_v22_24" \
  PRETRAIN_DIR="${ROOT}/runs/pick18_v22_24" \
  ONLINE_BETA=1 BETA_PRETRAIN=100 STAGES="0" \
  bash "${ROOT}/pipeline_pick18_v22_24.sh"
log "Pick-18 V22_24 β=1 finished"
