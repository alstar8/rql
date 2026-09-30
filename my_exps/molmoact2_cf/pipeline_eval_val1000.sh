#!/usr/bin/env bash
# Official MolmoPick val (1000 unique episodes): shared V24 gOn, then frozen pi0.5.
# One VLA server per GPU, 250-ep shards. Resume skips shards that already have result.json.
#
#   GPUS="0 1 2 3" bash pipeline_eval_val1000.sh
#   PHASES="vla" GPUS="0 1 2 3" bash pipeline_eval_val1000.sh   # VLA only
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v24_shared}"
EVAL_DIR="${EVAL_DIR:-${RUN_DIR}/eval_val1000}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/pick18_v24_shared_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
ACTOR="${ACTOR:-${RUN_DIR}/rl/shared_v24_s0/agent.pt}"
AE="${AE:-${RUN_DIR}/ae/ae_pick18_shared.pt}"
N_EP="${N_EP:-1000}"
N_SHARDS="${N_SHARDS:-4}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
PHASES="${PHASES:-gon vla}"

mkdir -p "${EVAL_DIR}/shards" "${EVAL_DIR}/pids" "${LOCAL_LOG}"
echo $$ > "${EVAL_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[val1000 $(date -u +%H:%M:%S)] $*"; }
has_phase() { [[ " ${PHASES} " == *" $1 "* ]]; }

if (( N_EP % N_SHARDS != 0 )); then
  echo "N_EP=${N_EP} must divide N_SHARDS=${N_SHARDS}" >&2
  exit 1
fi
if (( ${#GPUS[@]} < N_SHARDS )); then
  echo "need ${N_SHARDS} GPUs, got ${#GPUS[@]}: ${GPUS[*]}" >&2
  exit 1
fi
PER=$((N_EP / N_SHARDS))

cd "${CODE}"
for ((s = 0; s < N_SHARDS; s++)); do
  dest="${EVAL_DIR}/shards/shard${s}"
  if [[ -f "${dest}/benchmark.json" ]]; then
    log "shard ${s}: bench exists"
    continue
  fi
  start=$((s * PER))
  end=$((start + PER - 1))
  ids=$(seq -s, "${start}" "${end}")
  log "shard ${s}: val episodes ${start}-${end} -> ${dest}"
  "${SIM}" scripts/make_subset_benchmark.py --episodes "${ids}" --out "${dest}" \
    > "${LOCAL_LOG}/val1000_shard${s}_bench.log" 2>&1
done

run_shard() {
  local phase="$1" shard="$2" gpu="$3"
  local tag="${phase}_shard${shard}"
  local out="${EVAL_DIR}/${tag}"
  if [[ -f "${out}/result.json" ]]; then
    log "${tag}: exists, skip"
    return 0
  fi
  local port=$((9900 + gpu))
  local extra=()
  if [[ "${phase}" == "gon" ]]; then
    extra+=(--actor "${ACTOR}" --token-ae "${AE}" --set guidance_coef=-1)
  elif [[ "${phase}" == "actor" ]]; then
    extra+=(--actor "${ACTOR}" --token-ae "${AE}")
  fi
  log "${tag}: gpu=${gpu} port=${port} eps=${PER}"
  "${SIM}" scripts/run_eval.py \
    --scene val --episodes "${PER}" --gpu "${gpu}" --port "${port}" \
    --tag "${tag}" \
    --set "benchmark_override=${EVAL_DIR}/shards/shard${shard}" \
    --set "gate_step=0" \
    --set "gate_frac=0" \
    --set "horizon=500" \
    --set "save_video=false" \
    --set "out_dir=${EVAL_DIR}" \
    --set "checkpoint=${CKPT}" \
    "${extra[@]}" \
    >> "${LOCAL_LOG}/val1000_${tag}.log" 2>&1
}

run_phase() {
  local phase="$1"
  local pids=() fail=0
  log "${phase}: launching ${N_SHARDS} shards on GPUs ${GPUS[*]:0:${N_SHARDS}}"
  for ((s = 0; s < N_SHARDS; s++)); do
    run_shard "${phase}" "${s}" "${GPUS[$s]}" &
    pids+=($!)
    echo "${pids[-1]}" > "${EVAL_DIR}/pids/${phase}_shard${s}.pid"
  done
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "${phase} shard ${i}: ok"
    else
      log "${phase} shard ${i}: FAILED (see ${LOCAL_LOG}/val1000_${phase}_shard${i}.log)"
      fail=1
    fi
  done
  "${SIM}" - "${EVAL_DIR}" "${phase}" "${N_SHARDS}" <<'PY'
import json, math, sys
from pathlib import Path
root, phase, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
rows = []
for s in range(n):
    p = root / f"{phase}_shard{s}" / "result.json"
    if not p.exists():
        print(f"missing {p}", file=sys.stderr)
        sys.exit(1)
    rows.append(json.loads(p.read_text()))
succ = sum(r["successes"] for r in rows)
tot = sum(r["episodes_run"] for r in rows)
secs = sum(r["seconds"] for r in rows)
rate = succ / tot if tot else 0.0
z = 1.96
den = 1 + z * z / tot
centre = (rate + z * z / (2 * tot)) / den
half = z * math.sqrt(rate * (1 - rate) / tot + z * z / (4 * tot * tot)) / den
summary = {
    "phase": phase,
    "successes": succ,
    "episodes_run": tot,
    "success_rate": round(rate, 4),
    "ci95": [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)],
    "gpu_seconds": round(secs, 1),
    "gpu_hours": round(secs / 3600, 2),
    "shards": rows,
}
(root / f"{phase}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"{phase}: {succ}/{tot} = {100*rate:.1f}%  [{100*(centre-half):.1f}, {100*(centre+half):.1f}]  {secs/3600:.2f} GPU-h")
PY
  return "${fail}"
}

fail=0
has_phase gon && { run_phase gon || fail=1; }
has_phase actor && { run_phase actor || fail=1; }
has_phase vla && { run_phase vla || fail=1; }
log "val1000 finished (fail=${fail})"
exit "${fail}"
