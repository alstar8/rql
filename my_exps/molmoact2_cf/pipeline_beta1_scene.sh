#!/usr/bin/env bash
# One-scene pipeline: collect frozen-pi0.5 tokens → train AE from scratch →
# RL Token beta=1 from zero → held-out actor eval.
#
#   bash pipeline_beta1_scene.sh desk_mug 0,1,2,3 8500 500
#   bash pipeline_beta1_scene.sh kettle    4,5,6,7 8540 400
#
# GPU layout inside the quartet:
#   0,1  two collectors (original PI05 recipe, target 2500 sequences each)
#   2    frozen-VLA eval (16 specs → 64 rollouts) in parallel with collect
#   3    AE after collect
# After collect+AE: RL seeds 0 and 1 on GPUs 0 and 1; actor eval on GPU 2 then 3.

set -euo pipefail

SCENE="${1:?scene}"
GPU_CSV="${2:?gpus}"
PORT_BASE="${3:?port base}"
HORIZON="${4:?horizon}"

IFS=',' read -r GPU_C0 GPU_C1 GPU_EVAL GPU_AE <<< "${GPU_CSV}"

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/beta1_from_scratch/${SCENE}}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/beta1_${SCENE}_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

ENCODER="${SCENE}"
TARGET="${TARGET:-2500}"
EPISODES="${EPISODES:-300}"
WARMUP="${WARMUP:-40}"
BETA="${BETA:-1.0}"

PORT_C0=$((PORT_BASE + 0))
PORT_C1=$((PORT_BASE + 1))
PORT_EVAL=$((PORT_BASE + 2))
PORT_RL0=$((PORT_BASE + 10))
PORT_RL1=$((PORT_BASE + 11))
PORT_ACTOR=$((PORT_BASE + 12))

TOKENS="${RUN_DIR}/tokens"
AE_PATH="${RUN_DIR}/ae/ae_${ENCODER}.pt"
COLLECT_DIR="${RUN_DIR}/collect"

mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${TOKENS}" "${RUN_DIR}/ae" "${COLLECT_DIR}" "${RUN_DIR}/rl" "${RUN_DIR}/eval"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[${SCENE} $(date -u +%H:%M:%S)] $*"; }

wait_pidfile() {
  local pidfile="$1" name="$2"
  if [[ ! -f "${pidfile}" ]]; then
    log "ERROR: missing pidfile ${pidfile} (${name})"
    return 1
  fi
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
  log "collect ${tag} gpu=${gpu} port=${port} target=${TARGET} horizon=${HORIZON}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_collect.py \
      --scene "${SCENE}" --gpu "${gpu}" --port "${port}" --target "${TARGET}" \
      --corpus "${TOKENS}" \
      --out "${COLLECT_DIR}" \
      --set "tag=${tag}" \
      --set "horizon=${HORIZON}" \
      --set "checkpoint=${CKPT}" \
      --set "save_video=False" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

start_eval_vla() {
  local tag="${SCENE}_vla_baseline"
  log "frozen VLA eval gpu=${GPU_EVAL} port=${PORT_EVAL} horizon=${HORIZON}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene "${SCENE}" --episodes 16 --gpu "${GPU_EVAL}" --port "${PORT_EVAL}" \
      --tag "${tag}" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

start_rl() {
  local gpu="$1" port="$2" seed="$3" tag="$4"
  log "RL Token beta=${BETA} ${tag} gpu=${gpu} horizon=${HORIZON}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene "${SCENE}" --encoder "${ENCODER}" \
      --gpu "${gpu}" --port "${port}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${tag}" \
      --set "horizon=${HORIZON}" \
      --set "beta=${BETA}" \
      --set "seed=${seed}" \
      --set "token_ae=${AE_PATH}" \
      --set "episode_pool=0-11" \
      --set "out_dir=${RUN_DIR}/rl" \
      --set "checkpoint=${CKPT}" \
      > "${LOCAL_LOG}/${tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${tag}.pid"
}

eval_actor() {
  local tag="$1" gpu="$2" port="$3"
  local train_pidf="${RUN_DIR}/pids/${tag}.pid"
  local summary="${RUN_DIR}/rl/${tag}/summary.json"
  local agent="${RUN_DIR}/rl/${tag}/agent.pt"
  local result="${RUN_DIR}/eval/${tag}_actor/result.json"
  wait_pidfile "${train_pidf}" "${tag}"
  if [[ ! -f "${summary}" || ! -f "${agent}" ]]; then
    log "ERROR: ${tag} finished without agent/summary"
    return 1
  fi
  if ! python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('actor_episodes') is not None else 1)" "${summary}"; then
    log "ERROR: ${tag} summary is incomplete (train was killed); not evaluating"
    return 1
  fi
  if [[ -f "${result}" ]]; then
    log "eval already present ${result}"
    return 0
  fi
  log "eval actor ${tag} gpu=${gpu}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene "${SCENE}" --episodes 16 --gpu "${gpu}" --port "${port}" \
      --actor "${agent}" --token-ae "${AE_PATH}" \
      --tag "${tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${tag}_actor.log" 2>&1 < /dev/null
}

# --- phase 1: collect + frozen eval ---
start_collect "${GPU_C0}" "${PORT_C0}" "collect_${SCENE}_a"
sleep 20
start_collect "${GPU_C1}" "${PORT_C1}" "collect_${SCENE}_b"
sleep 20
start_eval_vla

wait_pidfile "${RUN_DIR}/pids/collect_${SCENE}_a.pid" "collect_a"
wait_pidfile "${RUN_DIR}/pids/collect_${SCENE}_b.pid" "collect_b"

if [[ ! -f "${COLLECT_DIR}/collect_${SCENE}_a/collect.json" || ! -f "${COLLECT_DIR}/collect_${SCENE}_b/collect.json" ]]; then
  log "ERROR: collect.json missing under ${COLLECT_DIR}"
  exit 1
fi

"${SIM}" scripts/summarize_collect.py "${COLLECT_DIR}" "${RUN_DIR}/corpus_stats.json" \
  | tee "${LOCAL_LOG}/corpus_stats.txt"

shopt -s nullglob
shards=("${TOKENS}/${SCENE}"/*.npz)
shopt -u nullglob
if (( ${#shards[@]} == 0 )); then
  log "ERROR: no token shards in ${TOKENS}/${SCENE}"
  exit 1
fi
log "collected ${#shards[@]} npz shards"

# --- phase 2: AE from this corpus only ---
log "train AE on gpu=${GPU_AE} from ${TOKENS}/${SCENE}"
setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 \
  "${SIM}" -m rlt.train_token_ae \
    --token_replay "${TOKENS}/${SCENE}/*.npz" \
    --out "${AE_PATH}" \
    --device "cuda:${GPU_AE}" \
    --steps 8000 \
    --max_sequences 6000 \
    > "${LOCAL_LOG}/ae_${ENCODER}.log" 2>&1 < /dev/null &
echo $! > "${RUN_DIR}/pids/ae.pid"
wait_pidfile "${RUN_DIR}/pids/ae.pid" "ae"
if [[ ! -f "${AE_PATH}" ]]; then
  log "ERROR: AE missing ${AE_PATH}"
  exit 1
fi

S0_TAG="${SCENE}_ae-${ENCODER}_beta1_s0"
S1_TAG="${SCENE}_ae-${ENCODER}_beta1_s1"
start_rl "${GPU_C0}" "${PORT_RL0}" 0 "${S0_TAG}"
sleep 20
start_rl "${GPU_C1}" "${PORT_RL1}" 1 "${S1_TAG}"

wait_pidfile "${RUN_DIR}/pids/${SCENE}_vla_baseline.pid" "vla_baseline"

eval_actor "${S0_TAG}" "${GPU_EVAL}" "${PORT_ACTOR}"
eval_actor "${S1_TAG}" "${GPU_AE}" "$((PORT_ACTOR + 1))"

log "pipeline done. corpus=${RUN_DIR}/corpus_stats.md eval=${RUN_DIR}/eval/"
