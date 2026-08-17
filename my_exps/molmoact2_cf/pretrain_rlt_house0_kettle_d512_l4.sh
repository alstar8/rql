#!/usr/bin/env bash
# Pretrain z=256 / d=512 / 4-layer RLT on house0 kettle offline collect.
# Adds residual actor BC (q_coef=0) so online actors are not cold.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

SRC_DIR="${SRC_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle_z256_d512_l4}"
LOCAL_LOG="${LOCAL_LOG_DIR:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_pretrain_house0_kettle_d512_l4_logs}"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"

TOKEN_STEPS="${TOKEN_STEPS:-8000}"
CRITIC_STEPS="${CRITIC_STEPS:-15000}"
ACTOR_BC_STEPS="${ACTOR_BC_STEPS:-5000}"
FLOW_STEPS_TRAIN="${FLOW_STEPS_TRAIN:-15000}"
N_CRITICS="${N_CRITICS:-10}"
Z_DIM="${Z_DIM:-256}"
TOKEN_D_MODEL="${TOKEN_D_MODEL:-512}"
TOKEN_LAYERS="${TOKEN_LAYERS:-4}"
TOKEN_HEADS="${TOKEN_HEADS:-4}"
HIDDEN="${HIDDEN:-256}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CRITIC_BATCH="${CRITIC_BATCH:-64}"
ACTOR_BETA="${ACTOR_BETA:-5.0}"

TOKEN_REPLAY="${TOKEN_REPLAY:-${SRC_DIR}/token_replay_merged.npz}"
CHUNK_REPLAY="${CHUNK_REPLAY:-${SRC_DIR}/chunk_replay_merged.npz}"
CHUNK_TOKEN_GLOB="${CHUNK_TOKEN_GLOB:-${SRC_DIR}}"

mkdir -p "${OUT_DIR}" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${OUT_DIR}/logs"

exec > >(tee -a "${LOCAL_LOG}/launcher.log") 2>&1
echo "[pretrain-kettle $(date -Is)] src=${SRC_DIR} out=${OUT_DIR}"
echo "[pretrain-kettle] arch z=${Z_DIM} d=${TOKEN_D_MODEL} layers=${TOKEN_LAYERS} actor_bc=${ACTOR_BC_STEPS}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "python missing: ${PYTHON}" >&2
  exit 1
fi
for required in "${TOKEN_REPLAY}" "${CHUNK_REPLAY}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required artifact: ${required}" >&2
    echo "  run: bash collect_house0_kettle_offline.sh" >&2
    exit 1
  fi
done

TOKEN_CKPT="${OUT_DIR}/rlt_token_house0_kettle_z256_d512_l4.pt"
RESIDUAL_CKPT="${OUT_DIR}/rlt_cf_pretrain_house0_kettle_z256_d512_l4.pt"
FLOW_CKPT="${OUT_DIR}/rlt_cf_flow_pretrain_house0_kettle_z256_d512_l4.pt"

echo "[pretrain-kettle] token AE warmup steps=${TOKEN_STEPS}"
"${PYTHON}" -u "${ROOT}/warmup_rlt_token.py" \
  --token_replay "${TOKEN_REPLAY}" \
  --out_ckpt "${TOKEN_CKPT}" \
  --device "${DEVICE}" \
  --steps "${TOKEN_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --n_critics "${N_CRITICS}" \
  --z_dim "${Z_DIM}" \
  --token_d_model "${TOKEN_D_MODEL}" \
  --token_layers "${TOKEN_LAYERS}" \
  --token_heads "${TOKEN_HEADS}" \
  --hidden "${HIDDEN}" \
  --use_cf_guide \
  --log_every 100

echo "[pretrain-kettle] residual critic + actor BC steps=${CRITIC_STEPS}+${ACTOR_BC_STEPS}"
"${PYTHON}" -u "${ROOT}/warmup_rlt_critic.py" \
  --rlt_ckpt "${TOKEN_CKPT}" \
  --chunk_replay "${CHUNK_REPLAY}" \
  --chunk_token_glob "${CHUNK_TOKEN_GLOB}" \
  --out_ckpt "${RESIDUAL_CKPT}" \
  --device "${DEVICE}" \
  --steps "${CRITIC_STEPS}" \
  --actor_bc_steps "${ACTOR_BC_STEPS}" \
  --actor_beta "${ACTOR_BETA}" \
  --batch_size "${CRITIC_BATCH}" \
  --encode_batch_size 8 \
  --log_every 100

REENC_CHUNKS="${OUT_DIR}/chunk_replay_reencoded.npz"
if [[ ! -f "${REENC_CHUNKS}" ]]; then
  echo "missing re-encoded chunks after residual warmup: ${REENC_CHUNKS}" >&2
  exit 1
fi

echo "[pretrain-kettle] flow critic+actor warmup steps=${FLOW_STEPS_TRAIN}"
"${PYTHON}" -u "${ROOT}/warmup_flow_critic.py" \
  --rlt_ckpt "${TOKEN_CKPT}" \
  --chunk_replay "${REENC_CHUNKS}" \
  --out_ckpt "${FLOW_CKPT}" \
  --device "${DEVICE}" \
  --steps "${FLOW_STEPS_TRAIN}" \
  --batch_size "${CRITIC_BATCH}" \
  --n_critics "${N_CRITICS}" \
  --flow_steps 10 \
  --guidance_coef 0.5 \
  --actor_beta "${ACTOR_BETA}" \
  --actor_q_coef 0.0 \
  --ref_dropout 0.0 \
  --log_every 100

cat > "${OUT_DIR}/README.md" <<EOF
# V17 kettle pretrain z=256 / d=512 / 4 layers

- Source: \`${SRC_DIR}\` (house0 kettle reference VLA rollouts)
- Token AE: z=${Z_DIM}, d=${TOKEN_D_MODEL}, layers=${TOKEN_LAYERS}
- Residual: critic ${CRITIC_STEPS} + actor BC ${ACTOR_BC_STEPS} (β=${ACTOR_BETA})
- Flow: ${FLOW_STEPS_TRAIN} steps with BC actor
EOF

echo "[pretrain-kettle $(date -Is)] DONE"
echo "  token=${TOKEN_CKPT}"
echo "  residual=${RESIDUAL_CKPT}"
echo "  flow=${FLOW_CKPT}"
