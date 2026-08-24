#!/bin/bash
# Chunk length x conversion regime, on the held-out 128 episodes of the val benchmark.
#
# The existing chunk table (36.1/50.0/72.2/83.3 on house10, 88.9/86.1/77.8/58.3 on
# house21) was measured on 36 episodes of two scenes, and entirely under plan_time -- the
# only regime the pre-refactor serving path could produce. So it cannot answer either
# half of the question it is used for:
#
#   * is the chunk trend a property of the policy, or of those two scenes?
#   * does the trend survive step_time, which is the regime the labels actually describe?
#     action[k] = q_commanded[t+k] - q_state[t+k], a per-step displacement, so adding it
#     to the state at step t+k reconstructs the commanded target exactly at every k,
#     while plan_time is short by the distance travelled since the plan.
#
# 8 cells, one GPU each, one server each -- pi0.5 inference blocks its server's event
# loop, so a shared server would serialise and then time out.
#
#   GPU 0   chunk 1    plan_time  |  GPU 4   chunk 8    plan_time
#   GPU 1   chunk 1    step_time  |  GPU 5   chunk 8    step_time
#   GPU 2   chunk 4    plan_time  |  GPU 6   chunk 16   plan_time
#   GPU 3   chunk 4    step_time  |  GPU 7   chunk 16   step_time
#
# The two chunk-1 cells are a deliberate control: the regimes are identical at chunk 1
# (planning and execution happen on the same step, against the same state), so any
# difference between them is harness noise and bounds how much of the rest to believe.
#
# Each run starts and stops its own server. Nothing is left holding a card.
set -u

R=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
SIM=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python
OUT=${OUT:-/home/jovyan/users/staroverov/pi05_molmospaces/pi05_runs/sweep_chunk_conversion}
EPISODES=${EPISODES:-128}

mkdir -p "$OUT"
cd "$R" || exit 1

gpu=0
for chunk in 1 4 8 16; do
    for conv in plan_time step_time; do
        name="chunk${chunk}_${conv}"
        echo "GPU $gpu  port $((8080 + gpu))  $name"
        setsid nohup env PYTHONUNBUFFERED=1 "$SIM" scripts/run_eval.py \
            --scene val \
            --episodes "$EPISODES" \
            --gpu "$gpu" \
            --port "$((8080 + gpu))" \
            --tag "$name" \
            --set chunk_size="$chunk" \
            --set conversion="$conv" \
            --set out_dir="$OUT" \
            > "$OUT/${name}.log" 2>&1 < /dev/null &
        gpu=$((gpu + 1))
    done
done

echo "8 cells launched at $(date -u)"
echo "watch:   grep -h SUCCESS $OUT/*.log"
echo "collect: $SIM $R/scripts/summarize_sweep.py $OUT"
