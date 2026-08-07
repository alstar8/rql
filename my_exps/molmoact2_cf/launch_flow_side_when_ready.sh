#!/usr/bin/env bash
# Wait for flow critic pretrain ckpt, then launch flow CF on GPUs 4-7.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

FLOW_CKPT="${FLOW_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt}"
PRETRAIN_LOG="${PRETRAIN_LOG:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_flow_pretrain_logs/pretrain.log}"
POLL_SEC="${POLL_SEC:-30}"

echo "[wait-flow] waiting for ${FLOW_CKPT}"
while [[ ! -f "${FLOW_CKPT}" ]]; do
  if [[ -f "${PRETRAIN_LOG}" ]]; then
    tail -n 1 "${PRETRAIN_LOG}" || true
  fi
  sleep "${POLL_SEC}"
done
echo "[wait-flow] found ${FLOW_CKPT}; launching flow side"

FLOW_ONLY=1 \
INSTANCES_PER_GPU="${INSTANCES_PER_GPU:-4}" \
FLOW_CKPT="${FLOW_CKPT}" \
bash "${ROOT}/launch_dual_cf_v8.sh"
