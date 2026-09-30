#!/usr/bin/env bash
# Two follow-ups, in order, on all four GPUs:
#   1. val-1000 of the keep500 frozen expert (last accepted copy, round 34)
#   2. fixed-anchor 600-episode collection, offline 8-step distill, val-1000
#
#   bash pipeline_v25_followup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
GPUS="${GPUS:-0 1 2 3}"
KEEP500="${KEEP500:-/home/jovyan/users/staroverov/v25_keep500}"
SRC_DIR="${SRC_DIR:-${ROOT}/runs/pick18_v24_shared}"
LOG_DIR="${LOG_DIR:-/home/jovyan/users/staroverov/v25_followup_logs}"
mkdir -p "${LOG_DIR}"

log() { echo "[v25-followup $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

fail=0

log "salvage val: frozen_expert -> ${KEEP500}/eval_frozen_val1000"
mkdir -p "${KEEP500}/eval_frozen_val1000"
if [[ ! -e "${KEEP500}/eval_frozen_val1000/shards" && -d "${SRC_DIR}/eval_val1000/shards" ]]; then
  ln -sfn "${SRC_DIR}/eval_val1000/shards" "${KEEP500}/eval_frozen_val1000/shards"
fi
if [[ -f "${KEEP500}/eval_frozen_val1000/vla_summary.json" ]]; then
  log "salvage val: summary exists, skip"
else
  if ! env EVAL_DIR="${KEEP500}/eval_frozen_val1000" \
      CKPT="${KEEP500}/frozen_expert" \
      LOCAL_LOG="${LOG_DIR}" \
      PHASES="vla" GPUS="${GPUS}" \
      bash "${ROOT}/pipeline_eval_val1000.sh" \
      >> "${LOG_DIR}/salvage_val.log" 2>&1; then
    log "ERROR: salvage val failed (see ${LOG_DIR}/salvage_val.log)"
    fail=1
  else
    log "salvage val finished"
  fi
fi

log "fixed600: collect, distill, val"
if ! env GPUS="${GPUS}" bash "${ROOT}/pipeline_pick18_v25_fixed600.sh" \
    >> "${LOG_DIR}/fixed600.log" 2>&1; then
  log "ERROR: fixed600 failed (see ${LOG_DIR}/fixed600.log)"
  fail=1
else
  log "fixed600 finished"
fi

log "follow-up finished (fail=${fail})"
exit "${fail}"
