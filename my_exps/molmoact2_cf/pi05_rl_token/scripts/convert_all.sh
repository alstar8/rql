#!/bin/bash
# Convert the downloaded demonstrations in batches of houses.
#
# Batching matters: the converter appends and records each finished trajectory, so
# between batches the LeRobot dataset on disk is complete and loadable. Training can
# start on whatever has been converted so far instead of waiting for the whole corpus,
# and an interrupted run resumes without redoing or duplicating work.
set -u

R=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
OP=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/openpi
SRC=/workspace-SR008.nfs2/users/staroverov/pi05_molmospaces/mbdata_pick/FrankaPickOmniCamConfig/part0/train
OUT=/workspace-SR008.nfs2/users/staroverov/pi05_molmospaces/lerobot_pick/all

# This is a CPU job. Hiding the GPUs keeps any imported framework from opening a CUDA
# context and sitting on memory that training needs -- which is exactly what happened
# when the resize helper still came from openpi and pulled in jax.
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

cd "$R" || exit 1
mapfile -t HOUSES < <(ls "$SRC" | grep '^house_' | sort -t_ -k2 -n)
echo "houses to convert: ${#HOUSES[@]}"

BATCH=100
for ((i = 0; i < ${#HOUSES[@]}; i += BATCH)); do
    slice=("${HOUSES[@]:i:BATCH}")
    echo "=== batch $((i / BATCH + 1)): houses $((i + 1))-$((i + ${#slice[@]})) at $(date -u +%H:%M:%S)"
    "$OP/.venv/bin/python" scripts/convert_demos_to_lerobot.py \
        --source "$SRC" --out "$OUT" --houses "${slice[@]}" 2>&1 | grep -vE "examples/s|ba/s"
done
echo "ALL BATCHES DONE at $(date -u)"
