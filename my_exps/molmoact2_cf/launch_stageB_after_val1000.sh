#!/usr/bin/env bash
# Wait for the val1000 eval pipeline to finish, then launch Stage B (lambda_pi=1).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
VAL_PID_FILE="${ROOT}/runs/pick18_v24_shared/eval_val1000/pids/pipeline.pid"
LOG="${B1K_TMP:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp}/pick18_v24_shared_logs/stageB_launcher.log"
mkdir -p "$(dirname "${LOG}")"

echo "[stageB-launcher $(date -u +%H:%M:%S)] waiting for val1000 pipeline (pid file ${VAL_PID_FILE})" >> "${LOG}"
while true; do
  if [[ -f "${VAL_PID_FILE}" ]]; then
    pid="$(cat "${VAL_PID_FILE}")"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      sleep 60
      continue
    fi
  fi
  if pgrep -f "pipeline_eval_val1000.sh" > /dev/null 2>&1; then
    sleep 60
    continue
  fi
  break
done

echo "[stageB-launcher $(date -u +%H:%M:%S)] val1000 done; launching Stage B" >> "${LOG}"
cd "${ROOT}"
setsid env GPUS="${GPUS:-0 1 2 3}" PHASES="${PHASES:-online eval}" PYTHONUNBUFFERED=1 \
  bash pipeline_pick18_v24_stageB.sh >> "${LOG}" 2>&1
echo "[stageB-launcher $(date -u +%H:%M:%S)] Stage B exited with $?" >> "${LOG}"
