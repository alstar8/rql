#!/bin/bash
# After the evaluations finish: the zero-baseline scene, plus a seed repeat of house21.
#
# house2 on GPUs 0-3 answers the question it was kept for -- can the method move at all
# from a scene the policy never solves. Sparse reward means the critic may have nothing
# to learn from; that is the point of running it.
#
# house21 repeated with a different seed on GPUs 4-7 is the control the first matrix
# lacked. There, the best cell used an encoder trained on a *different* scene, which is
# more likely sampling noise than a real effect; a second seed says which.
set -u

R=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
P=/home/jovyan/users/staroverov/pi05_molmospaces
V=$P/venvs/pi05_torch
MS=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces
C=$P/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999
AE=$P/pi05_runs/token_ae
OUT=$P/pi05_runs/rlt_matrix2

EPISODES=${EPISODES:-300}
WARMUP=${WARMUP:-40}

mkdir -p "$OUT"
cd "$R" || exit 1

echo "waiting for the evaluations to finish..."
while pgrep -f "eval_main.py" > /dev/null 2>&1; do sleep 60; done
echo "evaluations done at $(date -u); freeing their servers"
pkill -f "serve_pi05_http.py" 2>/dev/null
sleep 10

# cell definitions: scene, encoder, seed
cells=(
  "house2 house2 0" "house2 house10 0" "house2 house21 0" "house2 combined 0"
  "house21 house2 1" "house21 house10 1" "house21 house21 1" "house21 combined 1"
)

gpu=0
for cell in "${cells[@]}"; do
    set -- $cell
    port=$((8800 + gpu))
    setsid nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$gpu \
        HF_HOME=/home/jovyan/users/staroverov/.cache/huggingface \
        "$V/bin/python" scripts/serve_pi05_http.py \
        --checkpoint "$C" --port "$port" \
        > "$OUT/$1_ae-$2_s$3.server.log" 2>&1 < /dev/null &
    gpu=$((gpu + 1))
done

echo "waiting for servers..."
for _ in $(seq 1 60); do
    ready=$(grep -l "serving on http" "$OUT"/*.server.log 2>/dev/null | wc -l)
    [ "$ready" -ge 8 ] && break
    sleep 20
done
echo "servers ready: $(grep -l 'serving on http' "$OUT"/*.server.log 2>/dev/null | wc -l)/8"

gpu=0
for cell in "${cells[@]}"; do
    set -- $cell
    scene=$1; enc=$2; seed=$3
    port=$((8800 + gpu))
    name="${scene}_ae-${enc}_s${seed}"
    setsid nohup env PYTHONUNBUFFERED=1 RLT_VLA_TOKEN_DIM=2048 \
        CUDA_VISIBLE_DEVICES=$gpu MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$gpu \
        "$MS/.venv/bin/python" -m rlt.train_online \
        --server_host 127.0.0.1 --server_port "$port" \
        --benchmark_dir "$P/bench_collect_${scene}" \
        --episode_pool "0-11" \
        --token_ae "$AE/ae_${enc}.pt" \
        --out_dir "$OUT/$name" \
        --episodes "$EPISODES" --warmup_episodes "$WARMUP" \
        --horizon 500 --device cuda:0 --seed "$seed" \
        > "$OUT/${name}.train.log" 2>&1 < /dev/null &
    gpu=$((gpu + 1))
done

echo "eight runs launched at $(date -u)"
