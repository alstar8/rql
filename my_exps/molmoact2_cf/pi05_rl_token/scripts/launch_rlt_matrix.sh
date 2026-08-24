#!/bin/bash
# Phase 2-3: eight online RL runs -- two scenes crossed with four phase-1 encoders.
#
# The comparison the ablation is for: does an encoder trained on this scene beat one
# trained elsewhere, or the combined one? That question only has an answer per task, so
# each cell gets its own run.
#
# One frozen-VLA server per run rather than a shared one: pi0.5 inference blocks the
# server's event loop, so concurrent clients would serialise behind each other and, as
# measured earlier, time out during the handshake.
set -u

R=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
P=/home/jovyan/users/staroverov/pi05_molmospaces
V=$P/venvs/pi05_torch
MS=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces
C=$P/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999
AE=$P/pi05_runs/token_ae
OUT=$P/pi05_runs/rlt_matrix

EPISODES=${EPISODES:-300}
WARMUP=${WARMUP:-40}

mkdir -p "$OUT"
cd "$R" || exit 1

gpu=0
for scene in house10 house21; do
    for enc in house2 house10 house21 combined; do
        port=$((8600 + gpu))
        name="${scene}_ae-${enc}"
        echo "GPU $gpu  port $port  $name"

        # Frozen VLA for this cell.
        setsid nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu \
            HF_HOME=/home/jovyan/users/staroverov/.cache/huggingface \
            "$V/bin/python" scripts/serve_pi05_http.py \
            --checkpoint "$C" --port "$port" \
            > "$OUT/${name}.server.log" 2>&1 < /dev/null &

        gpu=$((gpu + 1))
    done
done

echo "waiting for the eight servers to load..."
for _ in $(seq 1 60); do
    ready=$(grep -l "serving on http" "$OUT"/*.server.log 2>/dev/null | wc -l)
    [ "$ready" -ge 8 ] && break
    sleep 20
done
echo "servers ready: $(grep -l 'serving on http' "$OUT"/*.server.log 2>/dev/null | wc -l)/8"

gpu=0
for scene in house10 house21; do
    for enc in house2 house10 house21 combined; do
        port=$((8600 + gpu))
        name="${scene}_ae-${enc}"
        setsid nohup env PYTHONUNBUFFERED=1 RLT_VLA_TOKEN_DIM=2048 \
            CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
            "$MS/.venv/bin/python" -m rlt.train_online \
            --server_host 127.0.0.1 --server_port "$port" \
            --benchmark_dir "$P/bench_collect_${scene}" \
            --episode_pool "0-11" \
            --token_ae "$AE/ae_${enc}.pt" \
            --out_dir "$OUT/$name" \
            --episodes "$EPISODES" --warmup_episodes "$WARMUP" \
            --horizon 500 --device cuda:0 \
            > "$OUT/${name}.train.log" 2>&1 < /dev/null &
        gpu=$((gpu + 1))
    done
done

echo "eight runs launched at $(date -u)"
