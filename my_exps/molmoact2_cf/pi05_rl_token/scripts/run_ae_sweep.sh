#!/bin/bash
# Wait for token collection to finish, then train the four phase-1 autoencoders.
#
# Four checkpoints, as agreed: one per scene, plus one on all three together. Training
# them separately is what makes the later comparison possible -- whether a scene-specific
# encoder beats a general one is an empirical question per task, not something to assume.
#
# The four runs go on four GPUs at once; each is a small transformer over cached token
# sequences, so they fit alongside each other comfortably.
set -u

R=/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
P=/home/jovyan/users/staroverov/pi05_molmospaces
T=$P/rlt_tokens
OUT=$P/pi05_runs/token_ae
MS=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces

export RLT_VLA_TOKEN_DIM=2048   # pi0.5 tokens are 2048 wide, not MolmoAct2's 2560

mkdir -p "$OUT"
cd "$R" || exit 1

echo "waiting for token collection to finish..."
while pgrep -f serve_pi05_tokens.py > /dev/null 2>&1; do sleep 60; done
echo "collection finished at $(date -u)"

for scene in house2 house10 house21; do
    n=$(ls "$T/$scene"/*.npz 2>/dev/null | wc -l)
    echo "  $scene: $n shards, $(du -sh "$T/$scene" 2>/dev/null | cut -f1)"
done

# Per-scene encoders on GPUs 0-2, the combined one on GPU 3.
gpu=0
for scene in house2 house10 house21; do
    echo "launching AE for $scene on cuda:$gpu"
    setsid nohup env PYTHONUNBUFFERED=1 RLT_VLA_TOKEN_DIM=2048 \
        "$MS/.venv/bin/python" -m rlt.train_token_ae \
        --token_replay "$T/$scene/*.npz" \
        --out "$OUT/ae_$scene.pt" \
        --device "cuda:$gpu" \
        --steps 8000 \
        --max_sequences 6000 \
        > "$OUT/ae_$scene.log" 2>&1 < /dev/null &
    gpu=$((gpu + 1))
done

echo "launching AE for all three on cuda:$gpu"
setsid nohup env PYTHONUNBUFFERED=1 RLT_VLA_TOKEN_DIM=2048 \
    "$MS/.venv/bin/python" -m rlt.train_token_ae \
    --token_replay "$T/*/*.npz" \
    --out "$OUT/ae_combined.pt" \
    --device "cuda:$gpu" \
    --steps 8000 \
    --max_sequences 6000 \
    > "$OUT/ae_combined.log" 2>&1 < /dev/null &

echo "four autoencoder runs launched at $(date -u)"
wait
echo "ALL AUTOENCODERS DONE at $(date -u)"
