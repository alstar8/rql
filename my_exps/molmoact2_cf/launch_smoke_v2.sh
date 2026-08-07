#!/usr/bin/env bash
# Fixed-seed G=0 vs G-on smoke + short online train gate for the rewritten stack.
#
# Prerequisites: 8× CF serve.py on :8000–:8007 with --disable_g (features on).
# Uses the matched VLA buffer + bounded warmup_v2 ckpt.
#
# Usage:
#   nohup bash launch_smoke_v2.sh > /workspace-SR008.nfs2/users/staroverov/B1K/tmp/molmoact2_cf_smoke_v2.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${MOLMOSPACES}/.venv/bin/python"
CKPT="${ROOT}/runs/molmoact2_cf_vlacrit_warmup_v2/molmoact2_cf.pt"
BUFFER="${ROOT}/runs/pick_buffer_vla_g0.npz"
OUT_ROOT="${ROOT}/runs/molmoact2_cf_smoke_v2"
N_EVAL="${1:-10}"
N_ONLINE="${2:-40}"

if [[ ! -f "${CKPT}" ]]; then
  echo "[smoke_v2] ERROR: missing ${CKPT}; run warmup_vlacrit.sh first"
  exit 1
fi
if [[ ! -f "${BUFFER}" ]]; then
  echo "[smoke_v2] ERROR: missing ${BUFFER}"
  exit 1
fi
if ! curl -sf --max-time 3 "http://127.0.0.1:8000/act" | grep -q '"status":"ok"'; then
  echo "[smoke_v2] ERROR: server :8000 not ready"
  exit 1
fi

mkdir -p "${OUT_ROOT}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$HOME/.cache/molmospaces/assets}"
export MUJOCO_GL=egl

echo "[smoke_v2] G=0 fixed ${N_EVAL} eps on gpu0/:8000"
MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON}" "${ROOT}/train_100m.py" \
  --buffer "${BUFFER}" \
  --cf_ckpt "${CKPT}" \
  --out_dir "${OUT_ROOT}/eval_g0" \
  --device cuda:0 \
  --server_host localhost \
  --server_port 8000 \
  --start_episode 0 \
  --shard_size "${N_EVAL}" \
  --max_valid_episodes "${N_EVAL}" \
  --target_env_steps $((N_EVAL * 500)) \
  --updates_per_episode 0 \
  --disable_g \
  --log_every_episodes "${N_EVAL}" \
  --tmp_rollout_dir /workspace-SR008.nfs2/users/staroverov/B1K/tmp/molmoact2_cf_rollouts \
  --seed 0 \
  2>&1 | tee "${OUT_ROOT}/eval_g0.log"

echo "[smoke_v2] G-on fixed ${N_EVAL} eps on gpu1/:8001"
MUJOCO_EGL_DEVICE_ID=1 CUDA_VISIBLE_DEVICES=1 \
  "${PYTHON}" "${ROOT}/train_100m.py" \
  --buffer "${BUFFER}" \
  --cf_ckpt "${CKPT}" \
  --out_dir "${OUT_ROOT}/eval_g_on" \
  --device cuda:0 \
  --server_host localhost \
  --server_port 8001 \
  --start_episode 0 \
  --shard_size "${N_EVAL}" \
  --max_valid_episodes "${N_EVAL}" \
  --target_env_steps $((N_EVAL * 500)) \
  --updates_per_episode 0 \
  --force_g \
  --log_every_episodes "${N_EVAL}" \
  --tmp_rollout_dir /workspace-SR008.nfs2/users/staroverov/B1K/tmp/molmoact2_cf_rollouts \
  --seed 0 \
  2>&1 | tee "${OUT_ROOT}/eval_g_on.log"

echo "[smoke_v2] short online (${N_ONLINE} eps) on gpu2/:8002"
MUJOCO_EGL_DEVICE_ID=2 CUDA_VISIBLE_DEVICES=2 \
  "${PYTHON}" "${ROOT}/train_100m.py" \
  --buffer "${BUFFER}" \
  --cf_ckpt "${CKPT}" \
  --out_dir "${OUT_ROOT}/online_short" \
  --device cuda:0 \
  --server_host localhost \
  --server_port 8002 \
  --start_episode 10 \
  --shard_size "${N_ONLINE}" \
  --max_valid_episodes "${N_ONLINE}" \
  --target_env_steps $((N_ONLINE * 500)) \
  --updates_per_episode 5 \
  --g_start_episodes 10 \
  --policy_delay 2 \
  --cql_coef 0.1 \
  --cql_n_actions 8 \
  --cql_action_radius 0.05 \
  --lr_alpha 0.0001 \
  --online_frac 0.5 \
  --log_every_episodes 5 \
  --tmp_rollout_dir /workspace-SR008.nfs2/users/staroverov/B1K/tmp/molmoact2_cf_rollouts \
  --seed 2 \
  2>&1 | tee "${OUT_ROOT}/online_short.log"

echo "[smoke_v2] done → ${OUT_ROOT}"
for f in eval_g0 eval_g_on online_short; do
  if [[ -f "${OUT_ROOT}/${f}/summary.json" ]]; then
    echo "  ${f}: $(cat "${OUT_ROOT}/${f}/summary.json")"
  fi
done
