#!/usr/bin/env bash
# Mug: AC pretrain + online warmup=0 on GPUs 0-3.
# Kettle: jittered recollect + AE + AC pretrain + online warmup=0 on GPUs 4-7.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
B1K_TMP="${B1K_TMP:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp}"
mkdir -p "${ROOT}/runs/beta1_jitter_ac" "${B1K_TMP}"
chmod +x "${ROOT}/pipeline_mug_pretrain_online.sh" "${ROOT}/pipeline_kettle_jitter_ac.sh"

setsid nohup bash "${ROOT}/pipeline_mug_pretrain_online.sh" \
  > "${B1K_TMP}/beta1_jitter_ac_desk_mug_pipeline.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/beta1_jitter_ac/desk_mug_pipeline.pid"
echo "[jitter-ac] mug    GPUs 0-3  pid=$(cat "${ROOT}/runs/beta1_jitter_ac/desk_mug_pipeline.pid")"

sleep 8

setsid nohup bash "${ROOT}/pipeline_kettle_jitter_ac.sh" \
  > "${B1K_TMP}/beta1_jitter_ac_kettle_pipeline.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/beta1_jitter_ac/kettle_pipeline.pid"
echo "[jitter-ac] kettle GPUs 4-7  pid=$(cat "${ROOT}/runs/beta1_jitter_ac/kettle_pipeline.pid")"
echo "[jitter-ac] mug log    ${B1K_TMP}/beta1_jitter_ac_desk_mug_pipeline.log"
echo "[jitter-ac] kettle log ${B1K_TMP}/beta1_jitter_ac_kettle_pipeline.log"
