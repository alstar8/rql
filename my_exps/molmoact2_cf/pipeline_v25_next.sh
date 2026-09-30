#!/usr/bin/env bash
# Wait for the in-flight 12k-step prefix distill val, then launch one follow-up.
#
# Prefix-only supervision (8 steps, early-stopped) already scored 54.2% val,
# below frozen pi0.5 at 63.5%. The 12k-step rerun of that same loss is on the GPUs.
#   * If that val reaches the E1 bar (66.9%), collect one more 600 on the new
#     expert and distill again. That is the E2 iteration, which is the only
#     swap that has raised val so far.
#   * Otherwise distill the same 600 shards with all 16 steps supervised.
#     Steps [0, 8) are the greedy teacher; steps [8, 16) are the frozen
#     expert's own tail, so the expert cannot drift while fitting G.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
FIXED="${ROOT}/pipeline_pick18_v25_fixed600.sh"
S12K_SUMMARY="${S12K_SUMMARY:-/home/jovyan/users/staroverov/v25_fixed600/eval_val1000_s12k/vla_summary.json}"
S12K_CKPT="${S12K_CKPT:-/home/jovyan/users/staroverov/v25_fixed600/student_expert_s12k}"
FIXED_RL="${FIXED_RL:-/home/jovyan/users/staroverov/v25_fixed600/rl/shared_v25_fixed600}"
BAR="${BAR:-0.669}"
GPUS="${GPUS:-0 1 2 3}"

log() { echo "[v25-next $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

gpus_busy() {
  local n
  n="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000 { c++ } END { print c+0 }')"
  [[ "${n}" != "0" ]]
}

log "waiting for s12k val (${S12K_SUMMARY})"
while true; do
  if [[ -f "${S12K_SUMMARY}" ]] && ! pgrep -f "${S12K_CKPT}" >/dev/null && ! gpus_busy; then
    break
  fi
  if [[ ! -f "${S12K_SUMMARY}" ]] && ! pgrep -f "${S12K_CKPT}" >/dev/null && ! gpus_busy; then
    log "s12k val is not running and left no summary; treating the bar as missed"
    break
  fi
  sleep 30
done

sr="0"
if [[ -f "${S12K_SUMMARY}" ]]; then
  sr="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['success_rate'])" "${S12K_SUMMARY}")"
  log "s12k val success_rate=${sr} bar=${BAR}"
fi

clears="$(python3 -c "import sys; print(1 if float(sys.argv[1]) >= float(sys.argv[2]) else 0)" "${sr}" "${BAR}")"
if [[ "${clears}" == "1" ]]; then
  log "bar cleared: one iteration on the s12k expert, fresh 600, prefix loss, no copy"
  exec env GPUS="${GPUS}" \
    RUN_DIR="/home/jovyan/users/staroverov/v25_iter2" \
    ON_TAG="shared_v25_iter2" \
    CKPT="${S12K_CKPT}" \
    INIT_ACTOR="${FIXED_RL}/agent.pt" \
    INIT_BUFFER="${FIXED_RL}/buffer.npz" \
    PHASES="online distill val1000" \
    SUPERVISE_STEPS=8 \
    HOLDOUT_FRAC=0 \
    bash "${FIXED}"
fi

log "bar missed: tail-anchored distill of the same 600 shards (supervise 16, 12000 steps)"
exec env GPUS="${GPUS}" \
  PHASES="distill val1000" \
  SUPERVISE_STEPS=16 \
  HOLDOUT_FRAC=0 \
  STUDENT_DIR="/home/jovyan/users/staroverov/v25_fixed600/student_expert_anchor16" \
  VAL_DIR="/home/jovyan/users/staroverov/v25_fixed600/eval_val1000_anchor16" \
  LOCAL_LOG="/home/jovyan/users/staroverov/v25_fixed600/logs_anchor16" \
  bash "${FIXED}"
