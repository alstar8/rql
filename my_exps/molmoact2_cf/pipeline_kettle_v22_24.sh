#!/usr/bin/env bash
# V22_24 kettle: ConsensusFlow losses on the tested V22/V23 (flow_rlt) recipe.
#
#   V   corrector flow MLP v(z, x_t, t, a_ref): BC to the frozen pi0.5 chunk,
#       beta=100 endpoint anchor on its own unguided unroll, and -- stage 1 only
#       -- the one-step guided lookahead -mean_k Q_k(s, x + (v+sg(G))dt, t+dt).
#   G   guidance student W(z, x_t, t), zero-init at t=0, trained ONLY by
#       common-scale-normalized distillation of the target-critic ensemble
#       gradients; deployed as lambda*t*u_damp (radial clip, trust-weighted
#       conflict contraction against sg(v), residual damping).
#   Q   10-head timed ensemble Q(z, x, t), reverse-state TD (delta ~ U[0,1] or
#       k/N with equal probability), target clipped to [0, 1], bootstrap at
#       fresh noise (s', x0', 0); Polyak tau=0.005, actor EMA 0.999.
#
# One SHARED AC pretrain (cf_actor_coef=0, beta=100: the tested BC + tight
# specialist copy) on GPU 6, then two online arms in parallel with a weaker
# anchor (ONLINE_BETA, default 1) so V can actually leave ã:
#
#   stage 0 (GPU 6): cf_actor_coef=0 -- the supplement's "BC-v / distilled-G"
#                    routing; all value improvement must arrive through G.
#   stage 1 (GPU 4): cf_actor_coef=1 -- the full joint method.
#
# SKIP_PRETRAIN=1 reuses an existing ac_pretrain/agent.pt (the beta=100 run).
#
# Each arm gets a PAIRED eval: guidance on (lambda=0.5) vs off (lambda=0) on the
# same held-out bench -- the on/off gap is G's sample-time contribution.
#
# Offline trajectories: V21 kettle AC-pretrain buffer (skip VLA collect).
# Tokens for L_ro: V21 kettle corpus shards (buffer.npz does not store tokens).
#
#   bash pipeline_kettle_v22_24.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/v22_24/kettle}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/v22_24_kettle_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"
GPU0="${GPU0:-6}"  # shared pretrain, then stage 0
GPU1="${GPU1:-4}"  # stage 1
PORT0="${PORT0:-8560}"
PORT1="${PORT1:-8540}"

V21_KETTLE="${V21_KETTLE:-${ROOT}/runs/beta1_1gpu/kettle}"
SRC_AE="${SRC_AE:-${V21_KETTLE}/ae/ae_kettle.pt}"
VLA_TRAJ="${VLA_TRAJ:-${V21_KETTLE}/ac_pretrain/kettle_ac_pretrain/buffer.npz}"
TOKENS_GLOB="${TOKENS_GLOB:-${ROOT}/runs/beta1_jitter_ac/kettle/tokens/kettle/*.npz}"

HORIZON=400
EPISODES=300
WARMUP=0
PRETRAIN_EPS="${PRETRAIN_EPS:-100}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
EVAL_EPISODES="${EVAL_EPISODES:-16}"
ONLINE_BETA="${ONLINE_BETA:-1}"
AE_PATH="${RUN_DIR}/ae/ae_kettle.pt"
PRETRAIN_TAG="kettle_v22_24_pretrain"
S0_TAG="${S0_TAG:-kettle_v22_24_s0_beta${ONLINE_BETA}}"
S1_TAG="${S1_TAG:-kettle_v22_24_s1_beta${ONLINE_BETA}}"

if [[ ! -f "${SRC_AE}" ]]; then
  echo "[v22_24] missing V21 kettle AE ${SRC_AE}" >&2
  exit 1
fi
if [[ ! -f "${VLA_TRAJ}" ]]; then
  echo "[v22_24] missing V21 kettle replay ${VLA_TRAJ}" >&2
  exit 1
fi
shopt -s nullglob
hits=(${TOKENS_GLOB})
shopt -u nullglob
if (( ${#hits[@]} == 0 )); then
  echo "[v22_24] no kettle token shards at ${TOKENS_GLOB}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/ae" \
  "${RUN_DIR}/rl" "${RUN_DIR}/eval" "${RUN_DIR}/ac_pretrain"
cp -n "${SRC_AE}" "${AE_PATH}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[v22_24 $(date -u +%H:%M:%S)] $*"; }

wait_pidfile() {
  local pidfile="$1" name="$2"
  local pid
  pid="$(cat "${pidfile}")"
  log "wait ${name} pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
  wait "${pid}" 2>/dev/null || true
}

AGENT="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/agent.pt"
BUFFER="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/buffer.npz"

if [[ "${SKIP_PRETRAIN:-0}" == "1" ]]; then
  if [[ ! -f "${AGENT}" ]]; then
    log "ERROR: SKIP_PRETRAIN=1 but missing ${AGENT}"
    exit 1
  fi
  log "skip pretrain, reuse ${AGENT}"
else
  # --- shared AC pretrain: BC + tight-anchor V, live reverse-TD critic, distilled G ---
  log "pretrain v22_24 gpu=${GPU0} replay=${VLA_TRAJ} (cf_actor_coef=0, beta=100)"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_pretrain_ac.py \
      --scene kettle --encoder kettle \
      --episodes "${PRETRAIN_EPS}" --offline-steps "${OFFLINE_STEPS}" \
      --gpu "${GPU0}" --port "${PORT0}" --tag "${PRETRAIN_TAG}" \
      --set "algorithm=v22_24" \
      --set "init_random_ae=false" \
      --set "token_ae=${AE_PATH}" \
      --set "beta=100" \
      --set "cf_actor_coef=0" \
      --set "horizon=${HORIZON}" \
      --set "episode_pool=0-47" \
      --set "out_dir=${RUN_DIR}/ac_pretrain" \
      --set "checkpoint=${CKPT}" \
      --set "train_token_offline=false" \
      --set "train_token_online=false" \
      --set "ae_finetune=true" \
      --set "store_decision_tokens=false" \
      --set "vla_traj=${VLA_TRAJ}" \
      --set "token_replay=${TOKENS_GLOB}" \
      > "${LOCAL_LOG}/${PRETRAIN_TAG}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid"
  wait_pidfile "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid" "ac_pretrain"
  if [[ ! -f "${AGENT}" ]]; then
    log "ERROR: missing pretrained agent ${AGENT}"
    exit 1
  fi
fi

# --- online arms -------------------------------------------------------------
start_rl() {
  local gpu="$1" port="$2" coef="$3" tag="$4"
  log "online v22_24 gpu=${gpu} cf_actor_coef=${coef} beta=${ONLINE_BETA} ${tag}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene kettle --encoder kettle \
      --gpu "${gpu}" --port "${port}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${tag}" --init-actor "${AGENT}" \
      --set "algorithm=v22_24" \
      --set "init_random_ae=false" \
      --set "token_ae=${AE_PATH}" \
      --set "beta=${ONLINE_BETA}" \
      --set "cf_actor_coef=${coef}" \
      --set "horizon=${HORIZON}" \
      --set "seed=0" \
      --set "init_buffer=${BUFFER}" \
      --set "episode_pool=0-11" \
      --set "out_dir=${RUN_DIR}/rl" \
      --set "checkpoint=${CKPT}" \
      --set "train_token_offline=false" \
      --set "train_token_online=false" \
      --set "ae_finetune=true" \
      --set "store_decision_tokens=false" \
      --set "token_replay=${TOKENS_GLOB}" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

eval_one() {
  local gpu="$1" port="$2" tag="$3" gcoef="$4" etag="$5"
  log "eval ${etag} gpu=${gpu} guidance_coef=${gcoef}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene kettle --episodes "${EVAL_EPISODES}" --gpu "${gpu}" --port "${port}" \
      --actor "${RUN_DIR}/rl/${tag}/agent.pt" --token-ae "${AE_PATH}" \
      --tag "${etag}" \
      --set "horizon=${HORIZON}" \
      --set "guidance_coef=${gcoef}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${etag}.log" 2>&1 < /dev/null
}

eval_pair() {
  # Paired on/off intervention on the same held-out bench, same seed stream.
  local gpu="$1" port="$2" tag="$3"
  wait_pidfile "${RUN_DIR}/pids/${tag}.pid" "${tag}"
  local summary="${RUN_DIR}/rl/${tag}/summary.json"
  if [[ ! -f "${RUN_DIR}/rl/${tag}/agent.pt" ]]; then
    log "ERROR: ${tag} missing agent"
    return 1
  fi
  if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('actor_episodes') is not None else 1)" "${summary}"; then
    log "ERROR: ${tag} incomplete summary"
    return 1
  fi
  eval_one "${gpu}" "${port}" "${tag}" "-1" "${tag}_gOn"
  eval_one "${gpu}" "${port}" "${tag}" "0" "${tag}_gOff"
}

start_rl "${GPU0}" "${PORT0}" 0 "${S0_TAG}"
start_rl "${GPU1}" "${PORT1}" 1 "${S1_TAG}"
eval_pair "${GPU0}" "${PORT0}" "${S0_TAG}" &
eval_pair "${GPU1}" "${PORT1}" "${S1_TAG}" &
wait
log "v22_24 kettle done: stage0 (distilled-G routing) + stage1 (joint CF), paired evals"
