#!/usr/bin/env bash
# Kettle: ±2cm jitter benches, recollect tokens, train AE, pretrain actor-critic
# on collected train poses, then online RL from episode 0 (no warmup).
#
#   bash pipeline_kettle_jitter_ac.sh
# GPUs 4,5 collect then RL; GPU 6 frozen eval then actor eval; GPU 7 AE then pretrain then s1 eval.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/beta1_jitter_ac/kettle}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/beta1_jitter_ac_kettle_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

HORIZON=400
BETA=1.0
TARGET=100000
COLLECT_MAX_EPISODES="${COLLECT_MAX_EPISODES:-50}"
EPISODES=300
WARMUP=0
PRETRAIN_EPS="${PRETRAIN_EPS:-100}"
SKIP_EVAL="${SKIP_EVAL:-0}"
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
AE_PATH="${RUN_DIR}/ae/ae_kettle.pt"
TOKENS="${RUN_DIR}/tokens"
COLLECT_DIR="${RUN_DIR}/collect"
PRETRAIN_TAG="kettle_ac_pretrain"
S0_TAG="kettle_ae-kettle_beta1_s0"
S1_TAG="kettle_ae-kettle_beta1_s1"
VLA_TAG="kettle_vla_baseline"

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${TOKENS}" "${RUN_DIR}/ae" "${COLLECT_DIR}" \
  "${RUN_DIR}/rl" "${RUN_DIR}/eval" "${RUN_DIR}/ac_pretrain"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[kettle-ac $(date -u +%H:%M:%S)] $*"; }

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

start_collect() {
  local gpu="$1" port="$2" tag="$3"
  log "collect ${tag} gpu=${gpu} max_episodes=${COLLECT_MAX_EPISODES} (100 trajectories total)"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_collect.py \
      --scene kettle --gpu "${gpu}" --port "${port}" --target "${TARGET}" \
      --max-episodes "${COLLECT_MAX_EPISODES}" \
      --corpus "${TOKENS}" --out "${COLLECT_DIR}" \
      --set "tag=${tag}" \
      --set "horizon=${HORIZON}" \
      --set "checkpoint=${CKPT}" \
      --set "save_video=False" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

start_collect 4 8540 collect_kettle_a
sleep 20
start_collect 5 8541 collect_kettle_b
sleep 20

if [[ "${SKIP_EVAL}" != "1" ]]; then
  log "frozen VLA eval gpu=6 (jittered eval48, 16 specs)"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene kettle --episodes 16 --gpu 6 --port 8542 \
      --tag "${VLA_TAG}" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      > "${LOCAL_LOG}/${VLA_TAG}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${VLA_TAG}.pid"
else
  log "skip frozen eval (already running)"
fi

wait_pidfile "${RUN_DIR}/pids/collect_kettle_a.pid" "collect_a"
wait_pidfile "${RUN_DIR}/pids/collect_kettle_b.pid" "collect_b"

if [[ ! -f "${COLLECT_DIR}/collect_kettle_a/collect.json" ]]; then
  log "ERROR: collect failed"
  exit 1
fi
"${SIM}" scripts/summarize_collect.py "${COLLECT_DIR}" "${RUN_DIR}/corpus_stats.json" \
  | tee "${LOCAL_LOG}/corpus_stats.txt"

shopt -s nullglob
shards=("${TOKENS}/kettle"/*.npz)
shopt -u nullglob
if (( ${#shards[@]} == 0 )); then
  log "ERROR: no token shards"
  exit 1
fi

log "train AE gpu=7"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 \
  "${SIM}" -m rlt.train_token_ae \
    --token_replay "${TOKENS}/kettle/*.npz" \
    --out "${AE_PATH}" \
    --device "cuda:7" \
    --steps 8000 \
    --max_sequences 6000 \
    > "${LOCAL_LOG}/ae_kettle.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/ae.pid"
wait_pidfile "${RUN_DIR}/pids/ae.pid" "ae"
if [[ ! -f "${AE_PATH}" ]]; then
  log "ERROR: AE missing"
  exit 1
fi

log "pretrain AC gpu=7 episodes=${PRETRAIN_EPS}"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
  "${SIM}" scripts/run_pretrain_ac.py \
    --scene kettle --encoder kettle \
    --episodes "${PRETRAIN_EPS}" --offline-steps "${OFFLINE_STEPS}" \
    --gpu 7 --port 8543 --tag "${PRETRAIN_TAG}" \
    --set "horizon=${HORIZON}" \
    --set "beta=${BETA}" \
    --set "token_ae=${AE_PATH}" \
    --set "episode_pool=0-47" \
    --set "out_dir=${RUN_DIR}/ac_pretrain" \
    --set "checkpoint=${CKPT}" \
    > "${LOCAL_LOG}/${PRETRAIN_TAG}.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid"
wait_pidfile "${RUN_DIR}/pids/${PRETRAIN_TAG}.pid" "ac_pretrain"

AGENT="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/agent.pt"
BUFFER="${RUN_DIR}/ac_pretrain/${PRETRAIN_TAG}/buffer.npz"
if [[ ! -f "${AGENT}" ]]; then
  log "ERROR: missing pretrained actor"
  exit 1
fi

start_rl() {
  local gpu="$1" port="$2" seed="$3" tag="$4"
  log "online RL warmup=0 ${tag} gpu=${gpu}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene kettle --encoder kettle \
      --gpu "${gpu}" --port "${port}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${tag}" --init-actor "${AGENT}" \
      --set "horizon=${HORIZON}" \
      --set "beta=${BETA}" \
      --set "seed=${seed}" \
      --set "token_ae=${AE_PATH}" \
      --set "init_buffer=${BUFFER}" \
      --set "episode_pool=0-11" \
      --set "out_dir=${RUN_DIR}/rl" \
      --set "checkpoint=${CKPT}" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

start_rl 4 8550 0 "${S0_TAG}"
sleep 20
start_rl 5 8551 1 "${S1_TAG}"

if [[ -f "${RUN_DIR}/pids/${VLA_TAG}.pid" ]]; then
  wait_pidfile "${RUN_DIR}/pids/${VLA_TAG}.pid" "vla_baseline" || true
fi

eval_actor() {
  local tag="$1" gpu="$2" port="$3"
  wait_pidfile "${RUN_DIR}/pids/${tag}.pid" "${tag}"
  local summary="${RUN_DIR}/rl/${tag}/summary.json"
  local agent="${RUN_DIR}/rl/${tag}/agent.pt"
  if [[ ! -f "${agent}" ]]; then
    log "ERROR: ${tag} missing agent"
    return 1
  fi
  if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('actor_episodes') is not None else 1)" "${summary}"; then
    log "ERROR: ${tag} incomplete summary"
    return 1
  fi
  log "eval actor ${tag} gpu=${gpu}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene kettle --episodes 16 --gpu "${gpu}" --port "${port}" \
      --actor "${agent}" --token-ae "${AE_PATH}" \
      --tag "${tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${tag}_actor.log" 2>&1 < /dev/null
}

eval_actor "${S0_TAG}" 6 8552
eval_actor "${S1_TAG}" 7 8553
log "kettle jitter+pretrain+online done. corpus=${RUN_DIR}/corpus_stats.md"
