#!/usr/bin/env bash
# V20 pretrain chain (post-I1/I4): collect -> token warmup -> re-encode ->
# born-CFGRL flow pretrain at the exact V20 architecture.
#
# The residual stage is gone. The flow stage builds a born-CFGRL model
# (use_cfgrl=True, o_dim=16) at hidden=1024 / n_hidden_actor=10 /
# n_hidden_critic=5 / z_expand=512 / layernorm heads, so the V20 learner's
# prepare_cfgrl_model hits the as_cfgrl fast path and no pretrained weight is
# ever silently re-initialized.
#
# Prereq: collect_house0_kettle_offline.sh has produced
#   ${SRC_DIR}/token_replay_merged.npz and ${SRC_DIR}/chunk_replay_merged.npz
# from a FRESH scaffold (make_fresh_scaffold.py), not from any prior pretrain.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

SRC_DIR="${SRC_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle_cfgrl_v20}"
LOCAL_LOG="${LOCAL_LOG_DIR:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_pretrain_house0_kettle_cfgrl_logs}"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"

TOKEN_STEPS="${TOKEN_STEPS:-8000}"
FLOW_STEPS_TRAIN="${FLOW_STEPS_TRAIN:-15000}"
N_CRITICS="${N_CRITICS:-10}"
Z_DIM="${Z_DIM:-256}"
TOKEN_D_MODEL="${TOKEN_D_MODEL:-512}"
TOKEN_LAYERS="${TOKEN_LAYERS:-4}"
TOKEN_HEADS="${TOKEN_HEADS:-4}"
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
CRITIC_BATCH="${CRITIC_BATCH:-64}"

# V20 online architecture (must match launch_v20_rlt_cfgrl.sh / v20_runner).
HIDDEN="${HIDDEN:-1024}"
N_HIDDEN_ACTOR="${N_HIDDEN_ACTOR:-10}"
N_HIDDEN_CRITIC="${N_HIDDEN_CRITIC:-5}"
Z_EXPAND_DIM="${Z_EXPAND_DIM:-512}"
CFGRL_O_DIM="${CFGRL_O_DIM:-16}"
CFGRL_W="${CFGRL_W:-1.0}"
CFGRL_DROPOUT="${CFGRL_DROPOUT:-0.1}"
# Reference-action dropout for the CFGRL actor; matches the critic bootstrap
# regime (flow_critic_td_step ref_dropout=0.5) and the V20 online phase.
CFGRL_REF_DROPOUT="${CFGRL_REF_DROPOUT:-0.5}"

TOKEN_REPLAY="${TOKEN_REPLAY:-${SRC_DIR}/token_replay_merged.npz}"
CHUNK_REPLAY="${CHUNK_REPLAY:-${SRC_DIR}/chunk_replay_merged.npz}"
CHUNK_TOKEN_GLOB="${CHUNK_TOKEN_GLOB:-${SRC_DIR}}"

mkdir -p "${OUT_DIR}" "${LOCAL_LOG}"
ln -sfn "${LOCAL_LOG}" "${OUT_DIR}/logs"

exec > >(tee -a "${LOCAL_LOG}/launcher.log") 2>&1
echo "[pretrain-cfgrl $(date -Is)] src=${SRC_DIR} out=${OUT_DIR}"
echo "[pretrain-cfgrl] arch z=${Z_DIM} d=${TOKEN_D_MODEL} l=${TOKEN_LAYERS} hidden=${HIDDEN} nha=${N_HIDDEN_ACTOR} nhc=${N_HIDDEN_CRITIC} zexp=${Z_EXPAND_DIM} o=${CFGRL_O_DIM}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "python missing: ${PYTHON}" >&2
  exit 1
fi
for required in "${TOKEN_REPLAY}" "${CHUNK_REPLAY}"; do
  if [[ ! -f "${required}" ]]; then
    echo "missing required artifact: ${required}" >&2
    echo "  run: bash collect_house0_kettle_offline.sh (with a fresh scaffold)" >&2
    exit 1
  fi
done

TOKEN_CKPT="${OUT_DIR}/rlt_token_house0_kettle_cfgrl.pt"
FLOW_CKPT="${OUT_DIR}/rlt_cf_flow_pretrain_house0_kettle_cfgrl.pt"
REENC_CHUNKS="${OUT_DIR}/chunk_replay_reencoded.npz"

echo "[pretrain-cfgrl] token AE warmup steps=${TOKEN_STEPS}"
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
  --no_cf_guide \
  --log_every 100

echo "[pretrain-cfgrl] re-encode chunk zs with warmed encoder"
"${PYTHON}" -u "${ROOT}/reencode_chunk_replay.py" \
  --rlt_ckpt "${TOKEN_CKPT}" \
  --chunk_replay "${CHUNK_REPLAY}" \
  --chunk_token_glob "${CHUNK_TOKEN_GLOB}" \
  --out_replay "${REENC_CHUNKS}" \
  --device "${DEVICE}" \
  --encode_batch_size 8

echo "[pretrain-cfgrl] born-CFGRL flow critic+actor warmup steps=${FLOW_STEPS_TRAIN}"
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
  --hidden "${HIDDEN}" \
  --n_hidden_actor "${N_HIDDEN_ACTOR}" \
  --n_hidden_critic "${N_HIDDEN_CRITIC}" \
  --z_expand_dim "${Z_EXPAND_DIM}" \
  --layernorm_heads \
  --use_cfgrl \
  --cfgrl_o_dim "${CFGRL_O_DIM}" \
  --cfgrl_w "${CFGRL_W}" \
  --cfgrl_dropout "${CFGRL_DROPOUT}" \
  --cfgrl_ref_dropout "${CFGRL_REF_DROPOUT}" \
  --log_every 100

cat > "${OUT_DIR}/README.md" <<EOF
# V20 kettle pretrain (born-CFGRL, no previously trained artifacts)

- Source: \`${SRC_DIR}\` (house0 kettle reference VLA rollouts, fresh scaffold collect)
- Token AE: z=${Z_DIM}, d=${TOKEN_D_MODEL}, layers=${TOKEN_LAYERS}, steps=${TOKEN_STEPS}
- Re-encode: chunk z/next_z from the warmed encoder (no residual stage)
- Flow: ${FLOW_STEPS_TRAIN} steps, born-CFGRL at V20 arch
  (hidden=${HIDDEN}, n_hidden_actor=${N_HIDDEN_ACTOR}, n_hidden_critic=${N_HIDDEN_CRITIC},
   z_expand=${Z_EXPAND_DIM}, o_dim=${CFGRL_O_DIM}, dropout=${CFGRL_DROPOUT},
   ref_dropout=${CFGRL_REF_DROPOUT})
EOF

echo "[pretrain-cfgrl $(date -Is)] DONE"
echo "  token=${TOKEN_CKPT}"
echo "  flow=${FLOW_CKPT}"
