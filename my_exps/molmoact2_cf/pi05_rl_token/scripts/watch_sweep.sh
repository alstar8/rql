#!/usr/bin/env bash
# One screen: GPU load, live processes, and where each sweep run has got to.
#
#   bash scripts/watch_sweep.sh
#   watch -n 30 bash scripts/watch_sweep.sh
#
# Every run that has a metrics file gets a row, always. An earlier version read
# `tail -1` and printed nothing when that came back empty (which happens when the
# file is caught between a write and the next flush) -- a live run would silently
# vanish from the table and look like it had died.
set -uo pipefail

RLT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${OUT_ROOT:-${RLT}/runs/ae_sweep}"

echo "=== GPUs ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo
echo "=== processes ==="
alive=$(pgrep -fc "rlt.train_token_ae" 2>/dev/null || echo 0)
echo "${alive} training process(es) alive"
pgrep -af "rlt.train_token_ae" 2>/dev/null | grep -o "ae_sweep/[a-z_0-9]*\.pt" | sed 's|ae_sweep/|  running: |' | sort

echo
echo "=== progress ==="
shopt -s nullglob
metrics=("${OUT_ROOT}"/*.metrics.jsonl)
if [ ${#metrics[@]} -eq 0 ]; then
    echo "  no runs under ${OUT_ROOT} yet"
else
    printf "  %-20s %8s %10s %8s %9s %8s\n" RUN STEP RECON "|z|" GRAD ELAPSED
    for m in "${metrics[@]}"; do
        name=$(basename "${m}" .metrics.jsonl)
        # Last non-empty line, so a torn read never hides a run.
        line=$(grep -v '^[[:space:]]*$' "${m}" 2>/dev/null | tail -1)
        if [ -z "${line}" ]; then
            printf "  %-20s %8s\n" "${name}" "starting"
            continue
        fi
        printf '%s' "${line}" | python3 -c "
import json, sys
raw = sys.stdin.read().strip()
try:
    d = json.loads(raw)
except Exception:
    print('  %-20s %8s' % ('${name}', 'writing'))
    sys.exit()
print('  %-20s %8d %10.4f %8.2f %9.2f %7.0fs' % (
    '${name}', d.get('step', 0), d.get('recon_loss', float('nan')),
    d.get('z_norm', float('nan')), d.get('grad_norm', float('nan')),
    d.get('elapsed_sec', 0)))
"
    done
fi

echo
echo "=== finished checkpoints ==="
ckpts=("${OUT_ROOT}"/*.pt)
if [ ${#ckpts[@]} -eq 0 ]; then
    echo "  none yet"
else
    for c in "${ckpts[@]}"; do echo "  $(basename "${c}")"; done
fi
