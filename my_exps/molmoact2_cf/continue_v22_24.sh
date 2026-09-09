#!/usr/bin/env bash
# Sequential continue of V22_24 on the current 1xH100 node.
#
# 1. Pick-18 stage-0 leftover held-out evals: bottle gOff, knife gOff.
# 2. Kettle ablation β=1 stage 0 resume to 300 + paired eval.
# 3. Kettle ablation β=1 stage 1 resume to 300 + paired eval.
# 4. Pick-18 V22_24 online β=1, all 18 tasks, stage 0 (reuses β=100 AC).
# β=100 kettle arms stay stopped (V22_24_METHODS.md).
#
# A live continue started before step 4 existed is followed by
# queue_pick18_v22_24_beta1.sh (waits on this pid, then runs the same sweep).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
GPU="${GPU:-0}"
PORT="${PORT:-8500}"
LOG="${B1K_TMP}/v22_24_continue_logs"
KETTLE="${ROOT}/runs/v22_24/kettle"
AE="${KETTLE}/ae/ae_kettle.pt"
PRE="${KETTLE}/ac_pretrain/kettle_v22_24_pretrain"
TOKENS="${ROOT}/runs/beta1_jitter_ac/kettle/tokens/kettle/*.npz"

mkdir -p "${LOG}" "${KETTLE}/pids" "${KETTLE}/eval"
ln -sfn "${LOG}" "${ROOT}/runs/v22_24/continue_logs"
echo $$ > "${KETTLE}/pids/continue.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[v22_24-continue $(date -u +'%F %H:%M:%S')] $*"; }

finish_pick18_goff() {
  log "Pick-18 leftover gOff evals (bottle, knife) on gpu=${GPU} slot=1"
  GPUS="${GPU}" SLOTS_PER_GPU=1 TASKS="bottle knife" \
    bash "${ROOT}/pipeline_pick18_v22_24.sh"
}

resume_kettle() {
  local coef="$1" tag="$2"
  local dest="${KETTLE}/rl/${tag}"
  local agent="${dest}/agent.pt"
  local progress="${dest}/progress.json"
  local done=0
  if [[ -f "${progress}" ]]; then
    done="$(python3 -c "import json,sys; print(int(json.load(open(sys.argv[1])).get('episodes_done',0)))" "${progress}")"
  fi
  if [[ -f "${agent}" && "${done}" -ge 300 ]]; then
    log "kettle ${tag}: already ${done} eps, skip train"
    return 0
  fi
  log "kettle ${tag}: resume gpu=${GPU} from progress ${done}/300 cf_actor_coef=${coef}"
  cd "${CODE}"
  "${SIM}" scripts/run_train.py \
    --scene kettle --encoder kettle \
    --gpu "${GPU}" --port "${PORT}" \
    --episodes 300 --warmup-episodes 0 \
    --tag "${tag}" --init-actor "${PRE}/agent.pt" \
    --set "algorithm=v22_24" \
    --set "init_random_ae=false" \
    --set "token_ae=${AE}" \
    --set "beta=1" \
    --set "cf_actor_coef=${coef}" \
    --set "horizon=400" \
    --set "seed=0" \
    --set "init_buffer=${PRE}/buffer.npz" \
    --set "episode_pool=0-11" \
    --set "out_dir=${KETTLE}/rl" \
    --set "checkpoint=${CKPT}" \
    --set "train_token_offline=false" \
    --set "train_token_online=false" \
    --set "ae_finetune=true" \
    --set "store_decision_tokens=false" \
    --set "token_replay=${TOKENS}" \
    --set "resume=true" \
    >> "${LOG}/${tag}.log" 2>&1
  if [[ ! -f "${agent}" ]]; then
    log "ERROR: ${tag} missing agent after resume"
    return 1
  fi
}

eval_kettle_pair() {
  local tag="$1"
  local agent="${KETTLE}/rl/${tag}/agent.pt"
  if [[ ! -f "${agent}" ]]; then
    log "ERROR: ${tag} missing agent, skip eval"
    return 1
  fi
  cd "${CODE}"
  for g in on off; do
    local gcoef="-1"
    local etag="${tag}_gOn"
    if [[ "${g}" == "off" ]]; then
      gcoef="0"
      etag="${tag}_gOff"
    fi
    if [[ -f "${KETTLE}/eval/${etag}/result.json" ]]; then
      log "kettle ${etag}: exists, skip"
      continue
    fi
    log "kettle eval ${etag} guidance_coef=${gcoef}"
    "${SIM}" scripts/run_eval.py \
      --scene kettle --episodes 16 --gpu "${GPU}" --port "${PORT}" \
      --actor "${agent}" --token-ae "${AE}" \
      --tag "${etag}" \
      --set "horizon=400" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "guidance_coef=${gcoef}" \
      --set "out_dir=${KETTLE}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOG}/${etag}.log" 2>&1
  done
}

pick18_beta1() {
  local dest="${ROOT}/runs/pick18_v22_24_beta1"
  mkdir -p "${dest}/pids"
  exec 9>"${dest}/pids/sweep.lock"
  if ! flock -n 9; then
    log "Pick-18 β=1 sweep already running, skip"
    return 0
  fi
  log "Pick-18 V22_24 online β=1, 18 tasks, stage 0, gpu=${GPU}"
  GPUS="${GPU}" SLOTS_PER_GPU=1 \
    RUN_DIR="${dest}" \
    LOCAL_LOG="${B1K_TMP}/pick18_v22_24_beta1_logs" \
    SOURCE_DIR="${ROOT}/runs/pick18_v22_24" \
    PRETRAIN_DIR="${ROOT}/runs/pick18_v22_24" \
    ONLINE_BETA=1 BETA_PRETRAIN=100 STAGES="0" \
    bash "${ROOT}/pipeline_pick18_v22_24.sh"
}

log "start on gpu=${GPU} (1xH100)"
finish_pick18_goff
resume_kettle 0 kettle_v22_24_s0_beta1
eval_kettle_pair kettle_v22_24_s0_beta1
resume_kettle 1 kettle_v22_24_s1_beta1
eval_kettle_pair kettle_v22_24_s1_beta1
pick18_beta1
log "all remaining V22_24 work finished"
