#!/usr/bin/env bash
# V22_24 on 8xH100: β=100 leftover evals + kettle β=1 ablation + Pick-18 β=1,
# all at once.
#
#   GPUs 0-5  (3 slots = 18 workers): Pick-18 stage 0 online β=1
#   GPU 6:    bottle gOff (β=100) + kettle S0 β=1 resume
#   GPU 7:    knife gOff (β=100)  + kettle S1 β=1 resume
#
#   bash launch_v22_24_8gpu.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
LOG="${B1K_TMP}/v22_24_8gpu_logs"
KETTLE="${ROOT}/runs/v22_24/kettle"
AE="${KETTLE}/ae/ae_kettle.pt"
PRE="${KETTLE}/ac_pretrain/kettle_v22_24_pretrain"
TOKENS="${ROOT}/runs/beta1_jitter_ac/kettle/tokens/kettle/"'*.npz'

mkdir -p "${LOG}" "${KETTLE}/pids" "${KETTLE}/eval"
echo $$ > "${KETTLE}/pids/launch_8gpu.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[v22_24-8gpu $(date -u +'%F %H:%M:%S')] $*"; }

resume_kettle() {
  local gpu="$1" port="$2" coef="$3" tag="$4"
  local dest="${KETTLE}/rl/${tag}"
  local agent="${dest}/agent.pt"
  mkdir -p "${dest}"
  log "kettle ${tag}: resume gpu=${gpu} port=${port} cf_actor_coef=${coef}"
  "${SIM}" scripts/run_train.py \
    --scene kettle --encoder kettle \
    --gpu "${gpu}" --port "${port}" \
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
    log "ERROR: ${tag} missing agent"
    return 1
  fi
  for g in on off; do
    local gcoef="-1" etag="${tag}_gOn"
    if [[ "${g}" == "off" ]]; then
      gcoef="0"
      etag="${tag}_gOff"
    fi
    if [[ -f "${KETTLE}/eval/${etag}/result.json" ]]; then
      log "kettle ${etag}: exists, skip"
      continue
    fi
    log "kettle eval ${etag} gpu=${gpu} guidance_coef=${gcoef}"
    "${SIM}" scripts/run_eval.py \
      --scene kettle --episodes 16 --gpu "${gpu}" --port "${port}" \
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
  log "kettle ${tag}: done"
}

log "8xH100: leftover β=100 evals + kettle β=1 + Pick-18 β=1"

mkdir -p "${ROOT}/runs/pick18_v22_24_beta1/pids" "${ROOT}/runs/pick18_v22_24/pids"

# --- Pick-18 β=1, 18 tasks on GPUs 0-5 ---
log "Pick-18 β=1 GPUs 0-5 x 3 slots"
setsid env \
  GPUS="0 1 2 3 4 5" SLOTS_PER_GPU=3 \
  RUN_DIR="${ROOT}/runs/pick18_v22_24_beta1" \
  LOCAL_LOG="${B1K_TMP}/pick18_v22_24_beta1_logs" \
  SOURCE_DIR="${ROOT}/runs/pick18_v22_24" \
  PRETRAIN_DIR="${ROOT}/runs/pick18_v22_24" \
  ONLINE_BETA=1 BETA_PRETRAIN=100 STAGES="0" \
  bash "${ROOT}/pipeline_pick18_v22_24.sh" \
  > "${LOG}/pick18_beta1.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/pick18_v22_24_beta1/pids/launch.pid"

# --- leftover β=100 gOff evals on GPU 6 (bottle) and 7 (knife) ---
log "Pick-18 β=100 leftover gOff GPUs 6 7"
setsid env \
  GPUS="6 7" SLOTS_PER_GPU=1 TASKS="bottle knife" \
  RUN_DIR="${ROOT}/runs/pick18_v22_24" \
  LOCAL_LOG="${B1K_TMP}/pick18_v22_24_logs" \
  SOURCE_DIR="${ROOT}/runs/pick18_v22_24" \
  ONLINE_BETA=100 STAGES="0" \
  bash "${ROOT}/pipeline_pick18_v22_24.sh" \
  > "${LOG}/pick18_beta100_leftover.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/pick18_v22_24/pids/leftover.pid"

# --- kettle ablation β=1 on GPU 6 / 7 (ports 8560 / 8540, not 92xx) ---
resume_kettle 6 8560 0 kettle_v22_24_s0_beta1 &
echo $! > "${KETTLE}/pids/kettle_v22_24_s0_beta1.pid"
resume_kettle 7 8540 1 kettle_v22_24_s1_beta1 &
echo $! > "${KETTLE}/pids/kettle_v22_24_s1_beta1.pid"

log "launched all groups: β=1 pick18, leftover gOff, kettle s0/s1"
wait || true
log "all 8gpu groups finished"
