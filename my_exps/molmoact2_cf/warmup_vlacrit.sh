#!/usr/bin/env bash
# Stable VLA warmup: collect G=0 Pick episodes (optional), then train offline.
# Requires a live MolmoAct2 CF server with features (disable_g) only if collecting.
#
# Usage:
#   bash warmup_vlacrit.sh [port=8000] [gpu=0] [num_eps=20]
#   FORCE_COLLECT=1 bash warmup_vlacrit.sh   # rebuild pick_buffer_vla_g0.npz

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${MOLMOSPACES}/.venv/bin/python"
PORT="${1:-8000}"
GPU="${2:-0}"
NUM_EPS="${3:-20}"
COLLECT_OUT="${ROOT}/runs/molmoact2_cf_vlacrit_collect"
OUT="${ROOT}/runs/molmoact2_cf_vlacrit_warmup_v3"
BUFFER="${ROOT}/runs/pick_buffer_vla_g0.npz"

mkdir -p "${COLLECT_OUT}" "${OUT}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="${GPU}"

if [[ ! -f "${BUFFER}" || "${FORCE_COLLECT:-0}" == "1" ]]; then
  echo "[warmup] collecting ${NUM_EPS} matched G=0 VLA episodes → ${BUFFER}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${ROOT}/train_100m.py" \
    --buffer "" \
    --out_dir "${COLLECT_OUT}" \
    --device cuda:0 \
    --server_host localhost \
    --server_port "${PORT}" \
    --start_episode 0 \
    --shard_size "${NUM_EPS}" \
    --target_env_steps $((NUM_EPS * 500)) \
    --max_valid_episodes "${NUM_EPS}" \
    --log_every_steps 100000 \
    --log_every_episodes "${NUM_EPS}" \
    --updates_per_episode 0 \
    --online_frac 1.0 \
    --disable_g \
    --replay_out "${BUFFER}" \
    --fit_replay_norm_stats \
    --tmp_rollout_dir /workspace-SR008.nfs2/users/staroverov/B1K/tmp/molmoact2_cf_rollouts \
    --seed 0 \
    2>&1 | tee "${COLLECT_OUT}/collect.log"
else
  echo "[warmup] reusing existing buffer ${BUFFER}"
fi

echo "[warmup] fitting bounded critic, then capped residual → ${OUT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" "${ROOT}/train_offline.py" \
  --buffer "${BUFFER}" \
  --out_dir "${OUT}" \
  --device cuda:0 \
  --phase1_steps 3000 \
  --phase2_steps 2000 \
  --cql_coef 0.1 \
  --cql_n_actions 8 \
  --cql_action_radius 0.05 \
  --cql_far_scale 1.0 \
  --target_divergence 0.0025 \
  --initial_alpha 0.3 \
  --lr_alpha 0.0001 \
  --log_every 100 \
  2>&1 | tee "${OUT}/warmup.log"

echo "[warmup] saved ${OUT}/molmoact2_cf.pt"
