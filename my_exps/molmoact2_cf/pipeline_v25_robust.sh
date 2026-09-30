#!/usr/bin/env bash
# Anchored-endpoint distillation. Four students, one per GPU, then val-1000 each.
#
# The 74.4% endpoint run trained only at flow time <= 0.25. Eval integrates from
# t=1, so that cutoff never supervises the trajectory the policy actually samples.
# These arms keep the clean-action term (the part that can see a 0.002 shift) and
# put probability mass back on the Beta(1.5, 1) schedule used at inference.
# DAgger arms use the same 8-step target; the earlier DAgger number supervised all
# 16 steps on the uniform schedule.
#
#   bash pipeline_v25_robust.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
TORCH_PY="${TORCH_PY:-/home/jovyan/users/staroverov/pi05_molmospaces/venvs/pi05_torch/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
FIXED600="${FIXED600:-/home/jovyan/users/staroverov/v25_fixed600/distill}"
DAGGER="${DAGGER:-/home/jovyan/users/staroverov/v25_dagger/distill_combined}"
SRC_SHARDS="${SRC_SHARDS:-${ROOT}/runs/pick18_v24_shared/eval_val1000/shards}"
RUN_DIR="${RUN_DIR:-/home/jovyan/users/staroverov/v25_robust}"
HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

STEPS="${STEPS:-12000}"
BATCH="${BATCH:-8}"
LR="${LR:-1e-5}"

mkdir -p "${RUN_DIR}/logs" "${RUN_DIR}/pids"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

log() { echo "[robust $(date -u +%H:%M:%S)] $*"; }

launch_one() {
  local gpu="$1" name="$2" data="$3" time_max="$4" low_frac="$5"
  local out="${RUN_DIR}/${name}"
  local logf="${RUN_DIR}/logs/${name}.log"
  mkdir -p "${out}"
  if [[ -f "${out}/distill_summary.json" && -f "${out}/model.safetensors" ]]; then
    log "skip train ${name}: already finished"
    return 0
  fi
  log "train ${name} on GPU ${gpu} (time-max ${time_max}, low-frac ${low_frac})"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" HF_HOME="${HF_HOME}" HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="${gpu}" TORCHINDUCTOR_CUDAGRAPHS=0 \
    "${TORCH_PY}" "${CODE}/scripts/train_expert_distill.py" \
      --data "${data}" \
      --init "${CKPT}" \
      --out "${out}" \
      --steps "${STEPS}" \
      --batch-size "${BATCH}" \
      --lr "${LR}" \
      --supervise-steps 8 \
      --endpoint-coef 1 \
      --time-max "${time_max}" \
      --time-low-frac "${low_frac}" \
      --holdout-frac 0 \
      --save-every 12000 \
      --self-check-batches 32 \
    > "${logf}" 2>&1 &
  echo $! > "${RUN_DIR}/pids/${name}.pid"
  log "  pid $(cat "${RUN_DIR}/pids/${name}.pid") -> ${logf}"
}

# name | gpu | data | time-max | low-frac
launch_one 0 fulltime_fixed "${FIXED600}" 1.0 1.0
launch_one 1 mix_fixed     "${FIXED600}" 0.25 0.5
launch_one 2 low_dagger    "${DAGGER}"   0.25 1.0
launch_one 3 full_dagger   "${DAGGER}"   1.0 1.0

fail=0
for name in fulltime_fixed mix_fixed low_dagger full_dagger; do
  pid="$(cat "${RUN_DIR}/pids/${name}.pid")"
  if wait "${pid}"; then
    log "train ${name} done"
  else
    log "ERROR: train ${name} failed (see ${RUN_DIR}/logs/${name}.log)"
    fail=1
  fi
done

for name in fulltime_fixed mix_fixed low_dagger full_dagger; do
  student="${RUN_DIR}/${name}"
  val="${RUN_DIR}/eval_${name}"
  if [[ ! -f "${student}/model.safetensors" ]]; then
    log "skip val ${name}: no checkpoint"
    fail=1
    continue
  fi
  if [[ -f "${val}/vla_summary.json" ]]; then
    log "skip val ${name}: summary exists"
    continue
  fi
  mkdir -p "${val}"
  if [[ ! -e "${val}/shards" && -d "${SRC_SHARDS}" ]]; then
    ln -sfn "${SRC_SHARDS}" "${val}/shards"
  fi
  log "val1000 ${name}"
  if ! env RUN_DIR="${RUN_DIR}" EVAL_DIR="${val}" CKPT="${student}" \
      LOCAL_LOG="${RUN_DIR}/logs" PHASES="vla" GPUS="0 1 2 3" \
      bash "${ROOT}/pipeline_eval_val1000.sh" \
      >> "${RUN_DIR}/logs/val_${name}.log" 2>&1; then
    log "ERROR: val ${name} failed"
    fail=1
  else
    log "val ${name} done"
  fi
done

log "robust finished (fail=${fail})"
exit "${fail}"
