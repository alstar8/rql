#!/usr/bin/env bash
# Held-out evals for the stopped kettle ablation β=100 arms (S0 ep 31, S1 ep 37).
# Sequential on GPU 0. Skip any result.json that already exists.
#
#   bash eval_kettle_beta100.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
KETTLE="${ROOT}/runs/v22_24/kettle"
AE="${KETTLE}/ae/ae_kettle.pt"
LOG="${B1K_TMP}/v22_24_beta100_eval_logs"
GPU="${GPU:-0}"
PORT="${PORT:-8560}"

mkdir -p "${LOG}" "${KETTLE}/eval" "${KETTLE}/pids"
echo $$ > "${KETTLE}/pids/eval_beta100.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[v22_24-b100-eval $(date -u +'%F %T')] $*"; }

run_eval() {
  local coef="$1" tag="$2" actor="$3"
  local dest="${KETTLE}/eval/${tag}/result.json"
  if [[ -f "${dest}" ]]; then
    log "${tag}: exists, skip"
    return 0
  fi
  if [[ ! -f "${actor}" ]]; then
    log "ERROR: missing actor ${actor}"
    return 1
  fi
  log "${tag}: gpu=${GPU} port=${PORT} guidance_coef=${coef}"
  "${SIM}" scripts/run_eval.py \
    --scene kettle --episodes 16 --gpu "${GPU}" --port "${PORT}" \
    --actor "${actor}" --token-ae "${AE}" \
    --tag "${tag}" \
    --set "horizon=400" \
    --set "gate_step=0" \
    --set "gate_frac=0" \
    --set "guidance_coef=${coef}" \
    --set "out_dir=${KETTLE}/eval" \
    --set "checkpoint=${CKPT}" \
    >> "${LOG}/${tag}.log" 2>&1
  if [[ ! -f "${dest}" ]]; then
    log "ERROR: ${tag} missing result.json"
    return 1
  fi
  log "${tag}: done $(python3 -c "import json; d=json.load(open('${dest}')); print(f\"{d['successes']}/{d['episodes_run']}={d['success_rate']}\")")"
}

log "kettle ablation β=100 held-out evals (stopped checkpoints)"
run_eval -1 kettle_v22_24_s0_gOn  "${KETTLE}/rl/kettle_v22_24_s0/agent.pt"
run_eval  0 kettle_v22_24_s0_gOff "${KETTLE}/rl/kettle_v22_24_s0/agent.pt"
run_eval -1 kettle_v22_24_s1_gOn  "${KETTLE}/rl/kettle_v22_24_s1/agent.pt"
run_eval  0 kettle_v22_24_s1_gOff "${KETTLE}/rl/kettle_v22_24_s1/agent.pt"
log "all four evals finished"
