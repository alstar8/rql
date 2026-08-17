#!/usr/bin/env bash
# Fine-tune kettle residual actor on success-only chunks for V18.
# Reuses V17 token AE + re-encoded chunks + critic; flow ckpt copied as-is.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

SRC_PRETRAIN="${SRC_PRETRAIN:-${ROOT}/runs/rlt_pretrain_house0_kettle_z256_d512_l4}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle_v18_success_bc}"
LOCAL_LOG="${LOCAL_LOG_DIR:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_pretrain_house0_kettle_v18_success_bc_logs}"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
ACTOR_BC_STEPS="${ACTOR_BC_STEPS:-8000}"
ACTOR_BETA="${ACTOR_BETA:-5.0}"

mkdir -p "${OUT_DIR}" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${OUT_DIR}/logs"
exec > >(tee -a "${LOCAL_LOG}/launcher.log") 2>&1
echo "[v18-success-bc $(date -Is)] src=${SRC_PRETRAIN} out=${OUT_DIR} steps=${ACTOR_BC_STEPS}"

RESIDUAL_IN="${SRC_PRETRAIN}/rlt_cf_pretrain_house0_kettle_z256_d512_l4.pt"
FLOW_IN="${SRC_PRETRAIN}/rlt_cf_flow_pretrain_house0_kettle_z256_d512_l4.pt"
REENC="${SRC_PRETRAIN}/chunk_replay_reencoded.npz"
TOKEN_IN="${SRC_PRETRAIN}/rlt_token_house0_kettle_z256_d512_l4.pt"

for req in "${RESIDUAL_IN}" "${REENC}" "${TOKEN_IN}" "${FLOW_IN}"; do
  [[ -f "${req}" ]] || { echo "missing ${req}" >&2; exit 1; }
done

ln -sfn "${TOKEN_IN}" "${OUT_DIR}/rlt_token_house0_kettle_z256_d512_l4.pt"
ln -sfn "${REENC}" "${OUT_DIR}/chunk_replay_reencoded.npz"
cp -f "${FLOW_IN}" "${OUT_DIR}/rlt_cf_flow_pretrain_house0_kettle_z256_d512_l4.pt"

RESIDUAL_OUT="${OUT_DIR}/rlt_cf_pretrain_house0_kettle_z256_d512_l4.pt"

echo "[v18-success-bc] residual success-only actor BC"
"${PYTHON}" -u "${ROOT}/warmup_rlt_critic.py" \
  --rlt_ckpt "${RESIDUAL_IN}" \
  --chunk_replay "${REENC}" \
  --skip_reencode \
  --out_ckpt "${RESIDUAL_OUT}" \
  --device "${DEVICE}" \
  --steps 0 \
  --actor_bc_steps "${ACTOR_BC_STEPS}" \
  --actor_bc_success_only \
  --actor_beta "${ACTOR_BETA}" \
  --batch_size 64 \
  --log_every 200

cat > "${OUT_DIR}/README.md" <<EOF
# V18 success-only actor BC

- Residual critic from V17 kettle pretrain; actor BC ${ACTOR_BC_STEPS} on success chunks only
- Flow: copied from V17 (mixture pre-gate forced to 0 online)
EOF

echo "[v18-success-bc $(date -Is)] DONE -> ${RESIDUAL_OUT}"
