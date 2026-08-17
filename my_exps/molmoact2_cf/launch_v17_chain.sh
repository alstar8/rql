#!/usr/bin/env bash
# Stop V16 improved (if live) → collect kettle offline → pretrain → launch V17.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
CHAIN_LOG="${CHAIN_LOG:-${B1K_TMP}/rlt_v17_chain.log}"
IMPROVED_DIR="${ROOT}/runs/rlt_cf_v16_rlt_improved"
SKIP_STOP="${SKIP_STOP:-0}"
SKIP_COLLECT="${SKIP_COLLECT:-0}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
SKIP_LAUNCH="${SKIP_LAUNCH:-0}"
EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-150}"
V17_MODE="${V17_MODE:-long}"
FRESH="${FRESH:-1}"

mkdir -p "$(dirname "${CHAIN_LOG}")"
exec > >(tee -a "${CHAIN_LOG}") 2>&1
echo "[v17-chain $(date -Is)] start"

stop_improved() {
  if [[ "${SKIP_STOP}" == "1" ]]; then
    echo "[v17-chain] skip stop"
    return 0
  fi
  echo "[v17-chain] stopping V16 improved processes"
  if [[ -d "${IMPROVED_DIR}/pids" ]]; then
    for pf in "${IMPROVED_DIR}/pids"/*.pid; do
      [[ -f "${pf}" ]] || continue
      pid="$(awk '{print $1}' "${pf}")"
      if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "  TERM ${pid} ($(basename "${pf}"))"
        kill -TERM "${pid}" 2>/dev/null || true
      fi
    done
  fi
  # Also stop any leftover improved rollout parents by env marker.
  for pid in $(pgrep -f 'rlt_cf_v16_rlt_improved' || true); do
    echo "  TERM leftover ${pid}"
    kill -TERM "${pid}" 2>/dev/null || true
  done
  sleep 8
  for pf in "${IMPROVED_DIR}/pids"/*.pid; do
    [[ -f "${pf}" ]] || continue
    pid="$(awk '{print $1}' "${pf}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "  KILL ${pid}"
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
  sleep 3
  echo "[v17-chain] stop done; GPU summary:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv || true
}

if [[ "${SKIP_COLLECT}" != "1" ]]; then
  stop_improved
  EPISODES_PER_SHARD="${EPISODES_PER_SHARD}" bash "${ROOT}/collect_house0_kettle_offline.sh"
else
  echo "[v17-chain] skip collect"
fi

if [[ "${SKIP_PRETRAIN}" != "1" ]]; then
  # Pretrain uses one GPU; free others are fine.
  DEVICE="${DEVICE:-cuda:0}" bash "${ROOT}/pretrain_rlt_house0_kettle_d512_l4.sh"
else
  echo "[v17-chain] skip pretrain"
fi

if [[ "${SKIP_LAUNCH}" != "1" ]]; then
  echo "[v17-chain] launching V17 online matrix"
  V17_MODE="${V17_MODE}" FRESH="${FRESH}" bash "${ROOT}/launch_v17_rlt_cf.sh"
else
  echo "[v17-chain] skip launch"
fi

echo "[v17-chain $(date -Is)] finished launcher handoff"
