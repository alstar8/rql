#!/usr/bin/env bash
# V22 mug ConsensusFlow: GPU 5 no-AE, GPU 6 full AE finetune.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
B1K_TMP="${B1K_TMP:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp}"
mkdir -p "${ROOT}/runs/v22_cf" "${B1K_TMP}"
chmod +x "${ROOT}/pipeline_mug_cf_no_ae.sh" "${ROOT}/pipeline_mug_cf_ae_ft.sh"

setsid nohup bash "${ROOT}/pipeline_mug_cf_no_ae.sh" \
  > "${B1K_TMP}/v22_cf_mug_no_ae_pipeline.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/v22_cf/mug_no_ae_pipeline.pid"
echo "[v22] mug no-AE     GPU 5  pid=$(cat "${ROOT}/runs/v22_cf/mug_no_ae_pipeline.pid")"

sleep 8

setsid nohup bash "${ROOT}/pipeline_mug_cf_ae_ft.sh" \
  > "${B1K_TMP}/v22_cf_mug_ae_ft_pipeline.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/v22_cf/mug_ae_ft_pipeline.pid"
echo "[v22] mug AE-ft     GPU 6  pid=$(cat "${ROOT}/runs/v22_cf/mug_ae_ft_pipeline.pid")"
echo "[v22] no-AE log  ${B1K_TMP}/v22_cf_mug_no_ae_pipeline.log"
echo "[v22] AE-ft log  ${B1K_TMP}/v22_cf_mug_ae_ft_pipeline.log"
