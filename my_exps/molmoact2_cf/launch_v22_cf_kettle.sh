#!/usr/bin/env bash
# V22 kettle ConsensusFlow: GPU 2 no-AE, GPU 3 full AE finetune.
# Reuses V21 kettle GPU 1 offline trajectories (AC-pretrain buffer.npz).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
B1K_TMP="${B1K_TMP:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp}"
mkdir -p "${ROOT}/runs/v22_cf" "${B1K_TMP}"
chmod +x "${ROOT}/pipeline_kettle_cf_no_ae.sh" "${ROOT}/pipeline_kettle_cf_ae_ft.sh"

setsid nohup bash "${ROOT}/pipeline_kettle_cf_no_ae.sh" \
  > "${B1K_TMP}/v22_cf_kettle_no_ae_pipeline.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/v22_cf/kettle_no_ae_pipeline.pid"
echo "[v22] kettle no-AE  GPU 2  pid=$(cat "${ROOT}/runs/v22_cf/kettle_no_ae_pipeline.pid")"

sleep 8

setsid nohup bash "${ROOT}/pipeline_kettle_cf_ae_ft.sh" \
  > "${B1K_TMP}/v22_cf_kettle_ae_ft_pipeline.log" 2>&1 < /dev/null &
echo $! > "${ROOT}/runs/v22_cf/kettle_ae_ft_pipeline.pid"
echo "[v22] kettle AE-ft  GPU 3  pid=$(cat "${ROOT}/runs/v22_cf/kettle_ae_ft_pipeline.pid")"
echo "[v22] no-AE log  ${B1K_TMP}/v22_cf_kettle_no_ae_pipeline.log"
echo "[v22] AE-ft log  ${B1K_TMP}/v22_cf_kettle_ae_ft_pipeline.log"
