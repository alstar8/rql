#!/usr/bin/env bash
# Stop only processes recorded by a launch_rlt_v4.sh run directory.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
RUN_DIR_INPUT="${1:-${ROOT}/runs/rlt_cf_v4_screen}"
GRACE_SEC="${GRACE_SEC:-120}"

if [[ ! -d "${RUN_DIR_INPUT}" ]]; then
  echo "[stop] run directory does not exist: ${RUN_DIR_INPUT}" >&2
  exit 1
fi
RUN_DIR="$(cd "${RUN_DIR_INPUT}" && pwd -P)"
PID_DIR="${RUN_DIR}/pids"
if [[ ! -d "${PID_DIR}" ]]; then
  echo "[stop] no PID directory: ${PID_DIR}"
  exit 0
fi

shopt -s nullglob
PID_FILES=("${PID_DIR}"/*.pid)
if (( ${#PID_FILES[@]} == 0 )); then
  echo "[stop] no PID manifests under ${PID_DIR}"
  exit 0
fi

declare -a PIDS=()
declare -A PID_FILES_BY_PID=()
declare -A SEEN=()

belongs_to_run() {
  local pid="$1"
  local environ="/proc/${pid}/environ"
  local cmdline="/proc/${pid}/cmdline"
  local entry
  # /proc/pid/environ is the *initial* exec env. Bash `export` after start
  # does not appear there, so the launch watchdog would survive stop_run
  # and immediately respawn trainers.
  if [[ -r "${environ}" ]]; then
    while IFS= read -r entry; do
      case "${entry}" in
        "RLT_CF_V4_RUN_DIR=${RUN_DIR}"|\
        "RLT_CF_V13_RUN_DIR=${RUN_DIR}"|\
        "RLT_CF_V14_RUN_DIR=${RUN_DIR}"|\
        "RLT_CF_V15_RUN_DIR=${RUN_DIR}"|\
        "RLT_CF_V16_RUN_DIR=${RUN_DIR}"|\
        "RLT_CF_V17_RUN_DIR=${RUN_DIR}"|\
        "RLT_CF_V18_RUN_DIR=${RUN_DIR}"|\
        "RLT_CF_V19_RUN_DIR=${RUN_DIR}")
          return 0
          ;;
      esac
    done < <(tr '\0' '\n' < "${environ}")
  fi
  if [[ -r "${cmdline}" ]]; then
    entry="$(tr '\0' ' ' < "${cmdline}")"
    [[ "${entry}" == *"${RUN_DIR}"* ]] && return 0
    [[ "${entry}" == *"launch_v19_rlt_cfgrl.sh"* ]] && return 0
  fi
  return 1
}

for pidfile in "${PID_FILES[@]}"; do
  read -r pid _rest < "${pidfile}" || pid=""
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || (( pid <= 1 )); then
    echo "[stop] removing malformed manifest ${pidfile}"
    rm -f "${pidfile}"
    continue
  fi
  if ! kill -0 "${pid}" 2>/dev/null; then
    echo "[stop] removing stale manifest ${pidfile} (pid ${pid})"
    rm -f "${pidfile}"
    continue
  fi
  if ! belongs_to_run "${pid}"; then
    echo "[stop] refusing pid ${pid}: process does not belong to ${RUN_DIR}" >&2
    continue
  fi
  if [[ -z "${SEEN[${pid}]:-}" ]]; then
    PIDS+=("${pid}")
    SEEN["${pid}"]=1
  fi
  PID_FILES_BY_PID["${pid}"]+="${pidfile} "
done

if (( ${#PIDS[@]} == 0 )); then
  echo "[stop] no live run-owned processes found"
  exit 0
fi

signal_process() {
  local signal_name="$1"
  local pid="$2"
  local pgid
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ -n "${pgid}" && "${pgid}" == "${pid}" ]]; then
    kill "-${signal_name}" -- "-${pgid}" 2>/dev/null || true
  else
    kill "-${signal_name}" "${pid}" 2>/dev/null || true
  fi
}

echo "[stop] sending TERM to ${#PIDS[@]} run-owned process groups"
for pid in "${PIDS[@]}"; do
  signal_process TERM "${pid}"
done

deadline=$((SECONDS + GRACE_SEC))
while (( SECONDS < deadline )); do
  alive=0
  for pid in "${PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  (( alive == 0 )) && break
  sleep 2
done

remaining=0
for pid in "${PIDS[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    remaining=$((remaining + 1))
    echo "[stop] pid ${pid} exceeded ${GRACE_SEC}s; sending KILL"
    signal_process KILL "${pid}"
  fi
done
(( remaining > 0 )) && sleep 2

failed=0
for pid in "${PIDS[@]}"; do
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[stop] WARNING: pid ${pid} is still alive" >&2
    failed=$((failed + 1))
  else
    for pidfile in ${PID_FILES_BY_PID["${pid}"]}; do
      rm -f "${pidfile}"
    done
  fi
done

if (( failed > 0 )); then
  echo "[stop] ${failed} process(es) could not be stopped" >&2
  exit 1
fi
echo "[stop] run stopped cleanly: ${RUN_DIR}"
