#!/usr/bin/env bash
# QUORUM on the 1,051 local Franka Pick demonstrations, starting from the
# pretrained pi0.5 checkpoint (no new VLA epoch).
#
#   1. Encode the demos (frozen pi0.5 reference + shared Pick-18 AE).
#   2. One epoch of offline v22_24 (V+G), beta 100, guidance loss on, lookahead off.
#   3. Eval that QUORUM on the official 1000-episode val.
#   4. Lowtime distill of the greedy chunks into a pi0.5 action expert.
#   5. Eval that student as a pure VLA.
#   6. Online QUORUM: all 48 train layouts of the 18 Pick tasks, 10 episodes each
#      (8640). There are not 1000 unique layouts (48 x 18 = 864).
#   7. Eval that online QUORUM on the same 1000-episode val.
#
#   GPUS="0 1 2 3" bash pipeline_pick1051_quorum.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
RUN="${RUN:-/home/jovyan/users/staroverov/v25_pick1051}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
AE="${AE:-${ROOT}/runs/pick18_v24_shared/ae/ae_pick18_shared.pt}"
AE_TOKENS="${AE_TOKENS:-${ROOT}/runs/pick18_v24_shared/ae/tokens}"
TORCH="${TORCH:-/home/jovyan/users/staroverov/pi05_molmospaces/venvs/pi05_torch/bin/python}"
SIM="${SIM:-/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
OPENPI="${OPENPI:-/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/openpi}"
HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
mkdir -p "${RUN}/logs"
log() { echo "[pick1051 $(date -u +%H:%M:%S)] $*"; }

export PYTHONUNBUFFERED=1
export HF_HOME
export HF_HUB_OFFLINE=1
export RLT_VLA_TOKEN_DIM=2048
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -f "${RUN}/buffer.npz" ]]; then
  log "encode 1051 demos on GPU ${GPUS[0]}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    PYTHONPATH="${CODE}:${OPENPI}/src:${OPENPI}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TORCH}" "${CODE}/scripts/encode_pick_demos.py" --out "${RUN}" \
    > "${RUN}/logs/encode.log" 2>&1
fi

STEPS="$(python3 -c "
import math, numpy as np
n=int(np.load('${RUN}/buffer.npz')['size'])
print(max(1, math.ceil(n/256)))
")"
log "offline epoch: ${STEPS} steps (one pass, batch 256)"
PRE="${RUN}/offline/pick1051_offline"
if [[ ! -f "${PRE}/agent.pt" ]]; then
  cd "${CODE}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    "${SIM}" scripts/run_pretrain_ac.py \
      --scene desk_mug --encoder pick18_shared \
      --episodes 2 --offline-steps "${STEPS}" \
      --gpu 0 --port 9700 --tag pick1051_offline \
      --set "algorithm=v22_24" \
      --set "init_random_ae=false" \
      --set "token_ae=${AE}" \
      --set "beta=100" \
      --set "cf_actor_coef=0" \
      --set "cf_ref_conditioned=true" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "out_dir=${RUN}/offline" \
      --set "checkpoint=${CKPT}" \
      --set "train_token_offline=false" \
      --set "train_token_online=false" \
      --set "ae_finetune=false" \
      --set "store_decision_tokens=false" \
      --set "vla_traj=${RUN}/buffer.npz" \
      > "${RUN}/logs/offline.log" 2>&1
fi

if [[ ! -f "${RUN}/eval_offline/vla_summary.json" && ! -f "${RUN}/eval_offline/gon_summary.json" ]]; then
  log "eval offline QUORUM on val-1000"
  GPUS="${GPUS[*]}" PHASES=gon \
    RUN_DIR="${RUN}" EVAL_DIR="${RUN}/eval_offline" CKPT="${CKPT}" \
    ACTOR="${PRE}/agent.pt" AE="${AE}" \
    LOCAL_LOG="${RUN}/logs" \
    bash "${ROOT}/pipeline_eval_val1000.sh" \
    > "${RUN}/logs/eval_offline.log" 2>&1
fi

if [[ ! -f "${RUN}/student/model.safetensors" ]]; then
  if [[ ! -f "${RUN}/distill/distill_obs_00000.npz" ]]; then
    log "label demo states with greedy QUORUM"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
      "${TORCH}" "${CODE}/scripts/label_quorum_demos.py" \
        --buffer "${RUN}/buffer.npz" --obs "${RUN}/obs" \
        --agent "${PRE}/agent.pt" --token-ae "${AE}" \
        --out "${RUN}/distill" \
        > "${RUN}/logs/label.log" 2>&1
  fi
  log "lowtime distill"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    PYTHONPATH="${CODE}:${OPENPI}/src:${OPENPI}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${TORCH}" "${CODE}/scripts/train_expert_distill.py" \
      --data "${RUN}/distill" --init "${CKPT}" --out "${RUN}/student" \
      --steps 12000 --batch-size 8 --lr 1e-5 \
      --supervise-steps 8 --endpoint-coef 0 --time-max 0.25 \
      --holdout-frac 0 --save-every 12000 --self-check-batches 32 \
      > "${RUN}/logs/lowtime.log" 2>&1
fi

if [[ ! -f "${RUN}/eval_lowtime/vla_summary.json" ]]; then
  log "eval distilled pi0.5 on val-1000"
  GPUS="${GPUS[*]}" PHASES=vla \
    RUN_DIR="${RUN}" EVAL_DIR="${RUN}/eval_lowtime" CKPT="${RUN}/student" \
    LOCAL_LOG="${RUN}/logs" \
    bash "${ROOT}/pipeline_eval_val1000.sh" \
    > "${RUN}/logs/eval_lowtime.log" 2>&1
fi

# 18 tasks x 48 layouts x 10 episodes. The layouts are the existing jittered
# train benches, not a fresh draw of 1000 unique houses.
ONLINE_EPS="${ONLINE_EPS:-8640}"
ON="${RUN}/online/pick1051_online"
if [[ ! -f "${ON}/summary.json" ]]; then
  log "online QUORUM, ${ONLINE_EPS} episodes"
  cd "${CODE}"
  GPUS_CSV="$(IFS=,; echo "${GPUS[*]}")"
  ports=()
  for i in "${!GPUS[@]}"; do ports+=($((9800 + i))); done
  PORTS_CSV="$(IFS=,; echo "${ports[*]}")"
  "${SIM}" scripts/run_train.py \
    --scene desk_mug --encoder pick18_shared \
    --gpu "${GPUS[0]}" --port "${ports[0]}" \
    --episodes "${ONLINE_EPS}" --warmup-episodes 0 \
    --tag pick1051_online --init-actor "${PRE}/agent.pt" \
    --set "algorithm=v22_24" \
    --set "init_random_ae=false" \
    --set "token_ae=${AE}" \
    --set "beta=1" \
    --set "cf_actor_coef=0" \
    --set "cf_ref_conditioned=true" \
    --set "gate_step=0" \
    --set "gate_frac=0" \
    --set "init_buffer=${PRE}/buffer.npz" \
    --set "task_pool=desk_mug:500,kettle:400,remote:500,ladle:500,tissue:500,spoon:500,spatula:500,pot:500,soap_dispenser:500,spray_bottle:500,cup:500,shaker:500,fork:500,bottle:500,fruit:500,bowl:500,knife:500,box:500" \
    --set "episode_pool=0-47" \
    --set "collectors=$(( ${#GPUS[@]} * 4 ))" \
    --set "egl_slots=3" \
    --set "vla_ports=${PORTS_CSV}" \
    --set "vla_gpus=${GPUS_CSV}" \
    --set "egl_devices=${GPUS_CSV}" \
    --set "buffer_capacity=400000" \
    --set "out_dir=${RUN}/online" \
    --set "checkpoint=${CKPT}" \
    --set "train_token_offline=false" \
    --set "train_token_online=false" \
    --set "ae_finetune=true" \
    --set "store_decision_tokens=false" \
    --set "token_replay=${AE_TOKENS}/*/*.npz" \
    > "${RUN}/logs/online.log" 2>&1
fi

if [[ ! -f "${RUN}/eval_online/gon_summary.json" && ! -f "${RUN}/eval_online/vla_summary.json" ]]; then
  log "eval online QUORUM on val-1000"
  GPUS="${GPUS[*]}" PHASES=gon \
    RUN_DIR="${RUN}" EVAL_DIR="${RUN}/eval_online" CKPT="${CKPT}" \
    ACTOR="${ON}/agent.pt" AE="${AE}" \
    LOCAL_LOG="${RUN}/logs" \
    bash "${ROOT}/pipeline_eval_val1000.sh" \
    > "${RUN}/logs/eval_online.log" 2>&1
fi

log "pick1051 pipeline finished"
