#!/usr/bin/env bash
# Stop both beta=1 from-scratch pipelines (mug GPUs 0-3, kettle GPUs 4-7).

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
RUN_ROOT="${ROOT}/runs/beta1_from_scratch"
GRACE_SEC="${GRACE_SEC:-60}"

declare -a PIDS=()

add_pid() {
  local pid="$1"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 0
  kill -0 "${pid}" 2>/dev/null || return 0
  PIDS+=("${pid}")
}

if [[ -d "${RUN_ROOT}" ]]; then
  shopt -s nullglob
  for pidfile in "${RUN_ROOT}"/*.pid "${RUN_ROOT}"/*/pids/*.pid; do
    read -r pid _rest < "${pidfile}" || pid=""
    add_pid "${pid}"
  done
fi

mapfile -t extra < <(ps -eo pid=,args= | grep -E 'pipeline_beta1_scene\.sh|pipeline_mug_pretrain_online\.sh|pipeline_kettle_jitter_ac\.sh|restart_mug_rl\.sh|restart_kettle_samepose\.sh|launch_jitter_ac\.sh|beta1_from_scratch|beta1_jitter_ac|[r]un_collect\.py --scene (desk_mug|kettle)|[r]un_train\.py --scene (desk_mug|kettle)|[r]un_eval\.py --scene (desk_mug|kettle)|[r]un_pretrain_ac\.py --scene (desk_mug|kettle)|[r]lt.train_token_ae' | awk '{print $1}')
for pid in "${extra[@]:-}"; do
  add_pid "${pid}"
done

declare -A seen=()
declare -a uniq=()
for pid in "${PIDS[@]}"; do
  [[ -n "${seen[$pid]:-}" ]] && continue
  seen[$pid]=1
  uniq+=("${pid}")
done
PIDS=("${uniq[@]}")

if (( ${#PIDS[@]} == 0 )); then
  echo "[stop-beta1] nothing running"
  exit 0
fi

echo "[stop-beta1] TERM ${PIDS[*]}"
for pid in "${PIDS[@]}"; do
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "${pgid}" && "${pgid}" == "${pid}" ]]; then
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  else
    kill -TERM "${pid}" 2>/dev/null || true
  fi
done

deadline=$((SECONDS + GRACE_SEC))
while (( SECONDS < deadline )); do
  alive=0
  for pid in "${PIDS[@]}"; do
    kill -0 "${pid}" 2>/dev/null && alive=$((alive + 1))
  done
  (( alive == 0 )) && break
  sleep 2
done

for pid in "${PIDS[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[stop-beta1] KILL ${pid}"
    kill -KILL "${pid}" 2>/dev/null || true
  fi
done
echo "[stop-beta1] stopped"
