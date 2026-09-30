#!/usr/bin/env bash
# Wait for Stage B paired eval to free the GPUs, and for V21 pretrain to
# finish, then run shared V21 online + official val-1000.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
STAGEB_PID_FILE="${ROOT}/runs/pick18_v24_shared/pids/pipeline_stageB.pid"
STAGEB_EVAL="${ROOT}/runs/pick18_v24_shared/eval_stageB"
PRETRAIN="${ROOT}/runs/pick18_v21_shared/ac_pretrain/shared_v21_pretrain/agent.pt"
LOG="${B1K_TMP:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp}/pick18_v21_shared_logs/v21_launcher.log"
mkdir -p "$(dirname "${LOG}")"

echo "[v21-launcher $(date -u +%H:%M:%S)] waiting for Stage B eval + V21 pretrain" >> "${LOG}"

stageb_busy() {
  local n
  n="$(ls "${STAGEB_EVAL}"/*/result.json 2>/dev/null | wc -l)"
  if (( n < 36 )); then
    return 0
  fi
  if pgrep -f "scripts/run_eval.py" > /dev/null 2>&1; then
    return 0
  fi
  return 1
}

while true; do
  busy=0
  if stageb_busy; then
    busy=1
  fi
  if [[ ! -f "${PRETRAIN}" ]]; then
    busy=1
  fi
  if (( busy == 0 )); then
    break
  fi
  sleep 60
done

echo "[v21-launcher $(date -u +%H:%M:%S)] Stage B eval done and pretrain present; launching V21 online + val1000" >> "${LOG}"
cd "${ROOT}"
setsid env GPUS="${GPUS:-0 1 2 3}" PHASES="${PHASES:-online val1000}" PYTHONUNBUFFERED=1 \
  bash pipeline_pick18_v21_shared.sh >> "${LOG}" 2>&1
echo "[v21-launcher $(date -u +%H:%M:%S)] V21 pipeline exited with $?" >> "${LOG}"
