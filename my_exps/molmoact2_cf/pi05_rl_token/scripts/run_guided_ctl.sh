#!/bin/bash
# Guided evaluation (use_vlm=1) of the GRPO adapter on top of frozen pi0, run twice:
# once with the gripper channel corrected and once without, so the two arms differ
# in that number and nothing else. Mirrors notebooks/run_eval_bench.sh; the only
# substitution is the server entry point.
#
#   bash run_guided.sh fix   5571 <save_dir>
#   bash run_guided.sh nofix 5573 <save_dir>

set -uo pipefail

source /home/jovyan/users/staroverov/env.sh

REPO=/home/jovyan/users/staroverov/GuideVLA
RL=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token

MODE="${1:-fix}"
PORT="${2:-5571}"
SAVE_DIR="${3:-$RL/runs/guided/$MODE}"

SERVER_CONDA_ENV=/home/jovyan/users/staroverov/envs/archer
CLIENT_CONDA_ENV=/home/jovyan/users/staroverov/envs/trl_new
SERVER_SCRIPT="$RL/scripts/bench_server_fixed.py"
CLIENT_SCRIPT="$REPO/notebooks/benchs/bench_client.py"

SERVER_GPU=6
CLIENT_GPU=7
NUM_ENVS=64
OBJ_SET="test"
ENV_ID="PutOnColorInSceneMulti-v1"
VLM_PATH="$REPO/grpo/experiments/pi0/multi_thinking/best_mean_step_880"
MODEL="Qwen/Qwen3.5-9B"
VLA="pi0"
USE_VLM=1
TEMPERATURE=0.6
SEEDS=(0 1 2)

FIX_ARG=""
[ "$MODE" = "nofix" ] && FIX_ARG="--no-fix"

mkdir -p "$SAVE_DIR"
cd "$REPO"

for seed in "${SEEDS[@]}"; do
    echo "=============================================="
    echo "[guided/$MODE] seed=${seed} env=${ENV_ID} port=${PORT}"
    echo "=============================================="

    CUDA_VISIBLE_DEVICES="${SERVER_GPU}" conda run --no-capture-output -p "${SERVER_CONDA_ENV}" \
        python "${SERVER_SCRIPT}" ${FIX_ARG} \
        --seed "${seed}" \
        --scenes "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --obj_set "${OBJ_SET}" \
        --use_vlm "${USE_VLM}" \
        --ckpt_path "${VLM_PATH}" \
        --output_dir "${SAVE_DIR}" \
        --vla "${VLA}" \
        --port "${PORT}" \
        --model "${MODEL}" &
    SERVER_PID=$!

    CUDA_VISIBLE_DEVICES="${CLIENT_GPU}" conda run --no-capture-output -p "${CLIENT_CONDA_ENV}" \
        python "${CLIENT_SCRIPT}" \
        --seed "${seed}" \
        --scenes "${ENV_ID}" \
        --num_envs "${NUM_ENVS}" \
        --obj_set "${OBJ_SET}" \
        --use_vlm "${USE_VLM}" \
        --ckpt_path "${VLM_PATH}" \
        --output_dir "${SAVE_DIR}" \
        --port "${PORT}" \
        --temperature "${TEMPERATURE}" \
        --model "${MODEL}" &
    CLIENT_PID=$!

    wait ${SERVER_PID}
    wait ${CLIENT_PID}
    echo "[guided/$MODE] seed=${seed} finished"
    sleep 2
done

touch "$SAVE_DIR/.DONE"
echo "[guided/$MODE] all seeds done"
