#!/usr/bin/env bash
# Read-only concise status for the controlled V13 run.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v13_controlled}"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
HELPER="${ROOT}/v13_harness.py"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[v13-status] run directory does not exist: ${RUN_DIR}" >&2
  exit 1
fi

pid_alive() {
  local pidfile="$1"
  local pid=""
  [[ -f "${pidfile}" ]] || return 1
  IFS=$' \t\r\n' read -r pid _ < "${pidfile}" || true
  [[ "${pid}" =~ ^[0-9]+$ ]] && (( pid > 1 )) && kill -0 "${pid}" 2>/dev/null
}

mapfile -t rows < <("${PYTHON}" "${HELPER}" variants --format tsv)
printf '%-25s %5s %7s %7s %8s %8s %5s %9s\n' \
  "variant" "gpu" "trainer" "server" "episodes" "envsteps" "gate" "resume"
printf '%-25s %5s %7s %7s %8s %8s %5s %9s\n' \
  "-------------------------" "-----" "-------" "-------" "--------" "--------" "-----" "---------"

for row in "${rows[@]}"; do
  IFS='|' read -r variant gpu cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
  trainer="down"
  server="inproc"
  pid_alive "${RUN_DIR}/pids/train_${variant}.pid" && trainer="up"
  if [[ -n "${port}" ]]; then
    server="down"
    pid_alive "${RUN_DIR}/pids/server_${variant}.pid" && server="up"
  fi
  status_args=(
    "${PYTHON}" "${HELPER}" training-status
    --out-dir "${RUN_DIR}/${variant}"
  )
  if [[ "${ae_mode}" == "1" ]]; then
    status_args+=(--ae)
  fi
  IFS=$'\t' read -r episodes env_steps cumulative window gate_on resume_state < <(
    "${status_args[@]}"
  )
  printf '%-25s %5s %7s %7s %8s %8s %5s %9s\n' \
    "${variant}" "${gpu}" "${trainer}" "${server}" "${episodes}" \
    "${env_steps}" "${gate_on}" "${resume_state}"
done

if pid_alive "${RUN_DIR}/pids/watchdog.pid"; then
  echo "[v13-status] watchdog: up"
else
  echo "[v13-status] watchdog: down"
fi
echo "[v13-status] full atomic snapshot: ${PYTHON} ${ROOT}/snapshot_run_status.py --run-dir ${RUN_DIR}"
