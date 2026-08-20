#!/usr/bin/env bash
# V20 rerun driver (post-fix): wait for the running fresh-scaffold collect to
# finish, then run the born-CFGRL pretrain chain, then launch V20 fresh.
#
# The collect itself is code-independent of the 2026-08-20 fixes (it runs the
# frozen VLA with 0 updates), so it is left to complete; the pretrain and the
# online run pick up the fixed cfgrl_actor_step (reference dropout), the
# deployed-head promotion gate, and the Bonferroni-corrected alpha.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
SRC_DIR="${SRC_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle}"
MERGED_CHUNK="${SRC_DIR}/chunk_replay_merged.npz"
MERGED_TOKEN="${SRC_DIR}/token_replay_merged.npz"

echo "[rerun $(date -Is)] waiting for collect merge artifacts in ${SRC_DIR}"
while true; do
  if [[ -f "${MERGED_CHUNK}" && -f "${MERGED_TOKEN}" ]]; then
    break
  fi
  if ! pgrep -f "collect_house0_kettle_offline.sh" >/dev/null 2>&1; then
    echo "[rerun $(date -Is)] collect not running; waiting up to 150s for restart"
    found=0
    for _ in 1 2 3 4 5; do
      sleep 30
      if pgrep -f "collect_house0_kettle_offline.sh" >/dev/null 2>&1; then
        found=1
        break
      fi
    done
    if (( found == 0 )); then
      echo "[rerun $(date -Is)] collect launcher exited without merged replays" >&2
      exit 1
    fi
  fi
  sleep 60
done

echo "[rerun $(date -Is)] merged replays present; waiting for collect launcher exit"
while pgrep -f "collect_house0_kettle_offline.sh" >/dev/null 2>&1; do
  sleep 15
done
sleep 30  # let server processes release GPUs

echo "[rerun $(date -Is)] starting born-CFGRL pretrain chain"
cd "${ROOT}"
bash pretrain_rlt_house0_kettle_cfgrl.sh

echo "[rerun $(date -Is)] pretrain done; launching V20 (fresh)"
GPU_IDS=0,1,2,3,4,5,6,7 FRESH=1 bash launch_v20_rlt_cfgrl.sh
echo "[rerun $(date -Is)] V20 learner exited"
