#!/bin/bash
# Does executing more of the predicted chunk change the success rate?
#
# The model was trained on deltas labelled against each step's own state, so applying
# delta[k] to the state from step t is wrong by the distance travelled since -- measured
# at 0.027 rad per step. That argued for re-planning every step, which is how every
# benchmark number so far was produced.
#
# But the RL harness executed eight actions per plan and reported far higher success on
# the same scene, and that has no accepted explanation yet. This measures the two regimes
# against each other directly, with no RL involved: same checkpoint, same scenes, same
# episodes, only the chunk length differs.
set -u

R=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
P=/home/jovyan/users/staroverov/pi05_molmospaces
V=$P/venvs/pi05_torch
MS=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces
OP=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/openpi
C=$P/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999
OUT=$P/pi05_runs/chunk_sweep

EPISODES=${EPISODES:-12}

mkdir -p "$OUT"
cd "$R" || exit 1

gpu=0
for scene in house10 house21; do
    for chunk in 1 2 4 8; do
        name="${scene}_chunk${chunk}"
        PI05_CHUNK_SIZE=$chunk setsid nohup env PYTHONUNBUFFERED=1 \
            PI05_CHUNK_SIZE=$chunk \
            HF_HOME=/home/jovyan/users/staroverov/.cache/huggingface \
            "$OP/.venv/bin/python" "$R/scripts/eval_pi05.py" \
            --checkpoint "$C" \
            --benchmark "$P/bench_collect_${scene}" \
            --episodes "$EPISODES" --workers 1 \
            --out "$OUT/$name" \
            --server-gpu $gpu --sim-gpu $gpu --egl-device $gpu \
            --port $((8900 + gpu)) --no-video \
            --server-python "$V/bin/python" \
            > "$OUT/${name}.log" 2>&1 < /dev/null &
        gpu=$((gpu + 1))
    done
done

echo "eight runs launched: two scenes x chunk in {1,2,4,8} at $(date -u)"
