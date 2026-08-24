#!/usr/bin/env bash
# Phase 1 across all 4 GPUs, as a sweep rather than one run.
#
# Why a sweep and not one big job: the AE is a 6-34M parameter transformer over
# 481-token sequences. Measured at batch 32 it runs 34 steps/s using 2.3 GiB on
# a single 4090 -- one run cannot fill an H100, and raising the batch mostly
# buys more PCIe traffic (158 MB/step already), not more useful compute. The way
# to make four cards do useful work here is to answer several open questions at
# once. Two are worth answering:
#
#   z_dim     the paper never states the bottleneck width
#   capacity  the reference AE only reached recon ~4.5 from 6.7, a weak fit --
#             is the bottleneck too tight, or is the decoder too small?
#
# `wide_bottleneck` and `deep_decoder` separate exactly those two.
#
# Step 0 builds a shared memmap of the token corpus. Without it each of the
# eight runs decompresses its own ~20 GB off NFS before taking a single step.
#
#   bash scripts/launch_ae_sweep.sh
#   PER_GPU=1 STEPS=8000 bash scripts/launch_ae_sweep.sh
set -euo pipefail

B1K="${B1K:-/home/jovyan/users/staroverov/B1K/B1K_AIRI}"
RLT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-${B1K}/submodules/molmospaces/.venv/bin/python}"
RUNS="${RUNS:-${B1K}/submodules/rql/my_exps/molmoact2_cf/runs/rlt_pretrain_demo1k}"

STEPS="${STEPS:-20000}"        # paper says 2000-10000; we log every 100 so the
                               # plateau is visible and can be read off after
BATCH="${BATCH:-32}"
MAX_SEQ="${MAX_SEQ:-8000}"     # quota split evenly across the 8 shards
# Explicit list, so a card already holding the VLA server can be skipped:
#   GPUS="0 2 3" bash scripts/launch_ae_sweep.sh
GPUS="${GPUS:-0 1 2 3}"
PER_GPU="${PER_GPU:-2}"        # 2.3-6.1 GiB each, so memory is never the limit
STAGGER="${STAGGER:-10}"

CACHE="${CACHE:-${RLT}/runs/token_cache}"
OUT_ROOT="${OUT_ROOT:-${RLT}/runs/ae_sweep}"
LOG_DIR="${RLT}/logs/ae_sweep"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

cd "${RLT}"

# ---- 0. shared token memmap ------------------------------------------------ #
if [ ! -f "${CACHE}/tokens.npy" ]; then
    echo "building shared token cache at ${CACHE} (one time, a few minutes) ..."
    "${PY}" scripts/prepare_token_cache.py \
        --token_replay "${RUNS}/token_replay_*_s*.npz" \
        --out "${CACHE}" \
        --max_sequences "${MAX_SEQ}" 2>&1 | tee "${LOG_DIR}/prepare_cache.log"
else
    echo "reusing token cache ${CACHE}"
fi
echo

# name:z_dim:d_model:n_layers:lr
CONFIGS=(
    "reference:256:256:2:1e-4"        # what molmoact2_cf used -- the control
    "tight_bottleneck:128:256:2:1e-4" # is 256 wider than the task needs?
    "wide_bottleneck:512:384:3:1e-4"  # more room in z
    "deep_decoder:256:512:4:1e-4"     # same z, more capacity: isolates the cause
    "reference_lr3e4:256:256:2:3e-4"  # the reference run's loss was still noisy
    "tight_lr3e4:128:256:2:3e-4"
    "wide_lr3e4:512:384:3:3e-4"
    "deep_lr3e4:256:512:4:3e-4"
)

read -r -a GPU_LIST <<< "${GPUS}"
N_GPUS="${#GPU_LIST[@]}"
TOTAL=$(( N_GPUS * PER_GPU ))
[ "${TOTAL}" -gt "${#CONFIGS[@]}" ] && TOTAL="${#CONFIGS[@]}"
echo "launching ${TOTAL}/${#CONFIGS[@]} configs on GPU(s) [${GPUS}], ${PER_GPU} per GPU"
echo "steps=${STEPS} batch=${BATCH}  out=${OUT_ROOT}  logs=${LOG_DIR}"
echo

i=0
for cfg in "${CONFIGS[@]}"; do
    [ "${i}" -ge "${TOTAL}" ] && break
    IFS=':' read -r NAME Z_DIM D_MODEL N_LAYERS LR <<< "${cfg}"
    GPU="${GPU_LIST[$(( i % N_GPUS ))]}"

    CUDA_VISIBLE_DEVICES="${GPU}" nohup "${PY}" -m rlt.train_token_ae \
        --token_cache "${CACHE}" \
        --out "${OUT_ROOT}/${NAME}.pt" \
        --steps "${STEPS}" --batch_size "${BATCH}" \
        --z_dim "${Z_DIM}" --d_model "${D_MODEL}" --n_layers "${N_LAYERS}" --lr "${LR}" \
        --device cuda:0 --seed 0 \
        > "${LOG_DIR}/${NAME}.log" 2>&1 &
    echo "[${i}] ${NAME}  gpu=${GPU}  z=${Z_DIM} d=${D_MODEL} L=${N_LAYERS} lr=${LR}  pid $!"

    i=$(( i + 1 ))
    [ "${i}" -lt "${TOTAL}" ] && sleep "${STAGGER}"
done

echo
echo "watch:  bash scripts/watch_sweep.sh"
echo "stop:   pkill -f rlt.train_token_ae"
