#!/usr/bin/env bash
# Fresh corpora + RL Token from scratch, beta=1.
#
#   GPUs 0-3  desk_mug  (horizon 500, gate 56)
#   GPUs 4-7  kettle    (horizon 400, gate 40)
#
# Frozen pi0.5 checkpoint only. Token AE is trained on tokens collected in this run.
#
#   bash launch_beta1_from_scratch.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
B1K_TMP="${B1K_TMP:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp}"
STAGGER_SEC="${STAGGER_SEC:-60}"

chmod +x "${ROOT}/pipeline_beta1_scene.sh" "${ROOT}/stop_beta1_from_scratch.sh"

mkdir -p "${ROOT}/runs/beta1_from_scratch" "${B1K_TMP}"

setsid nohup bash "${ROOT}/pipeline_beta1_scene.sh" desk_mug 0,1,2,3 8500 500 \
  > "${B1K_TMP}/beta1_desk_mug_pipeline.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/beta1_from_scratch/desk_mug_pipeline.pid"
echo "[beta1] mug    GPUs 0-3  pid=$(cat "${ROOT}/runs/beta1_from_scratch/desk_mug_pipeline.pid")  log=${B1K_TMP}/beta1_desk_mug_pipeline.log"

sleep "${STAGGER_SEC}"

setsid nohup bash "${ROOT}/pipeline_beta1_scene.sh" kettle 4,5,6,7 8540 400 \
  > "${B1K_TMP}/beta1_kettle_pipeline.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/beta1_from_scratch/kettle_pipeline.pid"
echo "[beta1] kettle GPUs 4-7  pid=$(cat "${ROOT}/runs/beta1_from_scratch/kettle_pipeline.pid")  log=${B1K_TMP}/beta1_kettle_pipeline.log"
echo "[beta1] corpus notes: runs/beta1_from_scratch/<scene>/corpus_stats.md after collect"
echo "[beta1] stop: bash ${ROOT}/stop_beta1_from_scratch.sh"
