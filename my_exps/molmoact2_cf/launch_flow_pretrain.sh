#!/usr/bin/env bash
# Offline flow-time CF critic (+ BC actor) pretrain on demo1k reencoded chunks.
# Reuses token AE from residual demo1k; builds Q(s,x,t) + flow actor + guide.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

PRETRAIN_DIR="${PRETRAIN_DIR:-${ROOT}/runs/rlt_pretrain_demo1k}"
TOKEN_CKPT="${TOKEN_CKPT:-${PRETRAIN_DIR}/rlt_token_demo1k.pt}"
# Prefer reencoded buffer (z already filled); fall back to merged.
CHUNK_REPLAY="${CHUNK_REPLAY:-${PRETRAIN_DIR}/chunk_replay_reencoded.npz}"
if [[ ! -f "${CHUNK_REPLAY}" ]]; then
  CHUNK_REPLAY="${PRETRAIN_DIR}/chunk_replay_merged.npz"
fi
OUT_CKPT="${OUT_CKPT:-${PRETRAIN_DIR}/rlt_cf_flow_pretrain_demo1k.pt}"
DEVICE="${DEVICE:-cuda:0}"
STEPS="${STEPS:-15000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${MOLMOSPACES}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[flow-pretrain] missing python ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${TOKEN_CKPT}" ]]; then
  echo "[flow-pretrain] missing token ckpt ${TOKEN_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${CHUNK_REPLAY}" ]]; then
  echo "[flow-pretrain] missing chunk replay ${CHUNK_REPLAY}" >&2
  exit 1
fi

mkdir -p "${PRETRAIN_DIR}"
echo "[flow-pretrain] token=${TOKEN_CKPT}"
echo "[flow-pretrain] chunks=${CHUNK_REPLAY}"
echo "[flow-pretrain] out=${OUT_CKPT} device=${DEVICE} steps=${STEPS}"

exec "${PYTHON}" -u "${ROOT}/warmup_flow_critic.py" \
  --rlt_ckpt "${TOKEN_CKPT}" \
  --chunk_replay "${CHUNK_REPLAY}" \
  --out_ckpt "${OUT_CKPT}" \
  --device "${DEVICE}" \
  --steps "${STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --n_critics 10 \
  --flow_steps 10 \
  --guidance_coef 0.5 \
  --log_every 100
