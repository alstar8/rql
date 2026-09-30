#!/usr/bin/env bash
# Report V25 distillation milestones as they land, and stop if the pipeline dies.
set -uo pipefail

R="${R:-/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/molmoact2_cf/runs/pick18_v25_distill}"
JOVYAN_RUN="${JOVYAN_RUN:-/home/jovyan/users/staroverov/v25_run}"
PIPE_PID="$(cat "${JOVYAN_RUN}/pipeline.pid" 2>/dev/null || cat "${R}/pids/pipeline.pid" 2>/dev/null || echo 0)"
PROGRESS="${R}/progress.jsonl"
seen_round=0

while true; do
  if [[ -f "${PROGRESS}" ]]; then
    last="$(tail -n 1 "${PROGRESS}" 2>/dev/null || true)"
    if [[ -n "${last}" ]]; then
      rnd="$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('round',0))" "${last}" 2>/dev/null || echo 0)"
      if [[ -n "${rnd}" && "${rnd}" -gt "${seen_round}" ]]; then
        echo "V25_MILESTONE round ${rnd} ${last}"
        seen_round="${rnd}"
      fi
    fi
  fi
  if [[ -f "${R}/eval_val1000_e100/vla_summary.json" ]]; then
    echo "V25_DONE eval_e100 at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 0
  fi
  if ! kill -0 "${PIPE_PID}" 2>/dev/null; then
    echo "V25_PIPELINE_GONE pid ${PIPE_PID} last_round=${seen_round} at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    exit 1
  fi
  sleep 120
done
