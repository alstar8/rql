#!/bin/bash
# Evaluate the eight trained actors under the benchmark's success criterion, with video.
#
# This is not the number the training logs report. Training counts an episode successful
# if success ever occurred (rollout.py: success_count > 0), while the benchmark counts the
# flag at the final step -- lift-then-drop passes the first and fails the second. The
# training numbers are comparable to each other; only these are comparable to the 28% and
# 58% baselines.
#
# Videos are kept so the behaviour can be watched, not just counted.
set -u

R=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
P=/home/jovyan/users/staroverov/pi05_molmospaces
V=$P/venvs/pi05_torch
MS=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces
C=$P/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999
AE=$P/pi05_runs/token_ae
M=$P/pi05_runs/rlt_matrix
OUT=$P/pi05_runs/rlt_eval

EPISODES=${EPISODES:-24}

mkdir -p "$OUT"
cd "$MS" || exit 1

gpu=0
for scene in house10 house21; do
    for enc in house2 house10 house21 combined; do
        port=$((8700 + gpu))
        name="${scene}_ae-${enc}"
        setsid nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu \
            HF_HOME=/home/jovyan/users/staroverov/.cache/huggingface \
            "$V/bin/python" "$R/scripts/serve_pi05_http.py" \
            --checkpoint "$C" --port "$port" \
            > "$OUT/${name}.server.log" 2>&1 < /dev/null &
        gpu=$((gpu + 1))
    done
done

echo "waiting for servers..."
for _ in $(seq 1 60); do
    ready=$(grep -l "serving on http" "$OUT"/*.server.log 2>/dev/null | wc -l)
    [ "$ready" -ge 8 ] && break
    sleep 20
done
echo "servers ready: $(grep -l 'serving on http' "$OUT"/*.server.log 2>/dev/null | wc -l)/8"

gpu=0
for scene in house10 house21; do
    for enc in house2 house10 house21 combined; do
        port=$((8700 + gpu))
        name="${scene}_ae-${enc}"
        setsid nohup env PYTHONUNBUFFERED=1 RLT_VLA_TOKEN_DIM=2048 \
            RLT_SERVER_HOST=127.0.0.1 RLT_SERVER_PORT="$port" \
            RLT_TOKEN_AE="$AE/ae_${enc}.pt" \
            RLT_ACTOR="$M/${name}/agent.pt" \
            RLT_DEVICE=cuda:0 \
            CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
            MLSPACES_ASSETS_DIR=/home/jovyan/users/staroverov/B1K/mlspaces/assets \
            MLSPACES_CACHE_DIR=/home/jovyan/users/staroverov/B1K/mlspaces/cache \
            PYTHONPATH="$R" \
            "$MS/.venv/bin/python" molmo_spaces/evaluation/eval_main.py \
            rlt.eval_config:RLTokenEvalConfig \
            --benchmark_dir "$P/bench_collect_${scene}" \
            --task_horizon_steps 500 --max_episodes "$EPISODES" \
            --num_workers 1 --no_wandb \
            --output_dir "$OUT/$name" \
            > "$OUT/${name}.eval.log" 2>&1 < /dev/null &
        gpu=$((gpu + 1))
    done
done

echo "eight evaluations launched at $(date -u)"
