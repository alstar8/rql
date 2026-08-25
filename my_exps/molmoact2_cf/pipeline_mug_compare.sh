#!/usr/bin/env bash
# Mug method comparison on one shared frozen-VLA buffer + frozen pi0.5 + one AE.
#
# Three arms, identical except for the actor architecture and the critic schedule:
#   v21        one-pass Gaussian actor mu(x, a_ref), TD critic offline + online (RLT paper)
#   flow_frz   reference-conditioned flow corrector, TD critic offline, FROZEN online
#   flow_td    reference-conditioned flow corrector, TD critic offline + online
#
# The question: should the RLT actor emit the action in one pass (V21) or be a
# flow-matching corrector trained with the CF-style loss (V22)? Arms 2 vs 3 also
# isolate whether the flow actor needs an online critic.
#
# All arms share: the same 100-trajectory buffer, the beta1_from_scratch mug AE
# (frozen encoder), beta=1, gate_step=0 (RL from the first env step),
# episode_pool 0-11, 300 episodes.
#
#   GPUS="4 1 7" bash pipeline_mug_compare.sh      # one GPU per arm: v21 flow_frz flow_td

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/compare_mug}"
LOCAL_LOG="${LOCAL_LOG:-${B1K_TMP}/compare_mug_logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r GPU_V21 GPU_FRZ GPU_TD GPU_CFAE GPU_CFNOAE _ <<< "${GPUS:-4 1 0 3 6}"
PORT_V21="${PORT_V21:-8611}"
PORT_FRZ="${PORT_FRZ:-8612}"
PORT_TD="${PORT_TD:-8613}"
PORT_CFAE="${PORT_CFAE:-8614}"
PORT_CFNOAE="${PORT_CFNOAE:-8615}"

V21_AE="${ROOT}/runs/beta1_from_scratch/desk_mug/ae/ae_desk_mug.pt"
BUFFER="${ROOT}/runs/beta1_jitter_ac/desk_mug/ac_pretrain/desk_mug_ac_pretrain/buffer.npz"
TOKEN_CORPUS="${ROOT}/runs/beta1_1gpu/desk_mug/tokens/desk_mug"

HORIZON=500
EPISODES="${EPISODES:-300}"
WARMUP=0
OFFLINE_STEPS="${OFFLINE_STEPS:-8000}"
BETA="${BETA:-1}"
AE_PATH="${RUN_DIR}/ae/ae_desk_mug.pt"

for f in "${V21_AE}" "${BUFFER}"; do
  [[ -f "${f}" ]] || { echo "missing ${f}"; exit 1; }
done
mkdir -p "${RUN_DIR}/pids" "${LOCAL_LOG}" "${RUN_DIR}/ae"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid"
cp -n "${V21_AE}" "${AE_PATH}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"
cd "${CODE}"

log() { echo "[compare $(date -u +%H:%M:%S)] $*"; }

wait_pidfile() {
  local pidfile="$1" name="$2" pid
  pid="$(cat "${pidfile}")"
  log "wait ${name} pid=${pid}"
  while kill -0 "${pid}" 2>/dev/null; do sleep 30; done
  wait "${pid}" 2>/dev/null || true
}

# run_arm <arm> <gpu> <port> <algorithm> <freeze_online> <compose> <ae_finetune>
run_arm() {
  local arm="$1" gpu="$2" port="$3" algo="$4" freeze="$5" compose="$6" ae_ft="$7"
  local pre_tag="mug_${arm}_pretrain" on_tag="mug_${arm}_s0"
  local pre_dir="${RUN_DIR}/ac_pretrain" on_dir="${RUN_DIR}/rl"

  # AE finetune (online only) needs the mug token corpus for reconstruction.
  local ae_online_sets=()
  if [[ "${ae_ft}" == "true" ]]; then
    [[ -d "${TOKEN_CORPUS}" ]] || { log "ERROR: missing token corpus ${TOKEN_CORPUS}"; return 1; }
    ae_online_sets+=(--set "ae_finetune=true" --set "store_decision_tokens=false" --set "token_replay=${TOKEN_CORPUS}/*.npz")
  else
    ae_online_sets+=(--set "ae_finetune=false")
  fi

  log "${arm}: pretrain ${algo} gpu=${gpu} compose=${compose} (replay shared buffer, offline actor+critic)"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_pretrain_ac.py \
      --scene desk_mug --encoder desk_mug \
      --episodes 100 --offline-steps "${OFFLINE_STEPS}" \
      --gpu "${gpu}" --port "${port}" --tag "${pre_tag}" \
      --set "algorithm=${algo}" \
      --set "token_ae=${AE_PATH}" \
      --set "beta=${BETA}" \
      --set "flow_actor_coef=0" \
      --set "flow_compose=${compose}" \
      --set "horizon=${HORIZON}" \
      --set "episode_pool=0-47" \
      --set "out_dir=${pre_dir}" \
      --set "checkpoint=${CKPT}" \
      --set "vla_traj=${BUFFER}" \
      --set "train_token_offline=false" \
      --set "train_token_online=false" \
      --set "ae_finetune=false" \
      > "${LOCAL_LOG}/${pre_tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${pre_tag}.pid"
  wait_pidfile "${RUN_DIR}/pids/${pre_tag}.pid" "${pre_tag}"

  local agent="${pre_dir}/${pre_tag}/agent.pt"
  [[ -f "${agent}" ]] || { log "ERROR: ${arm} missing pretrained agent"; return 1; }

  log "${arm}: online gpu=${gpu} freeze_critic=${freeze} compose=${compose} ae_finetune=${ae_ft}"
  setsid nohup env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene desk_mug --encoder desk_mug \
      --gpu "${gpu}" --port "${port}" \
      --episodes "${EPISODES}" --warmup-episodes "${WARMUP}" \
      --tag "${on_tag}" --init-actor "${agent}" \
      --set "algorithm=${algo}" \
      --set "token_ae=${AE_PATH}" \
      --set "beta=${BETA}" \
      --set "flow_actor_coef=1" \
      --set "flow_freeze_critic_online=${freeze}" \
      --set "flow_compose=${compose}" \
      --set "horizon=${HORIZON}" \
      --set "seed=0" \
      --set "init_buffer=${pre_dir}/${pre_tag}/buffer.npz" \
      --set "episode_pool=0-11" \
      --set "out_dir=${on_dir}" \
      --set "checkpoint=${CKPT}" \
      --set "train_token_offline=false" \
      --set "train_token_online=false" \
      "${ae_online_sets[@]}" \
      > "${LOCAL_LOG}/${on_tag}.log" 2>&1 < /dev/null &
  echo $! > "${RUN_DIR}/pids/${on_tag}.pid"
  wait_pidfile "${RUN_DIR}/pids/${on_tag}.pid" "${on_tag}"

  local on_agent="${on_dir}/${on_tag}/agent.pt"
  [[ -f "${on_agent}" ]] || { log "ERROR: ${arm} missing online agent"; return 1; }
  log "${arm}: eval gpu=${gpu}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_eval.py \
      --scene desk_mug --episodes 16 --gpu "${gpu}" --port "${port}" \
      --actor "${on_agent}" --token-ae "${AE_PATH}" \
      --tag "${on_tag}_actor" \
      --set "horizon=${HORIZON}" \
      --set "out_dir=${RUN_DIR}/eval" \
      --set "checkpoint=${CKPT}" \
      >> "${LOCAL_LOG}/${on_tag}_actor.log" 2>&1 < /dev/null
  log "${arm}: done"
}

# Run the requested arms concurrently, one GPU each. ARMS selects a subset so arms
# can be launched as GPUs free.
#   v21/flow_frz/flow_td : the one-pass vs flow-corrector arms (frozen vs TD critic)
#   cf_ae/cf_noae        : CF composition V = v_pi05_base + G(RLT), +/- token-AE finetune
ARMS="${ARMS:-v21 flow_frz flow_td cf_ae cf_noae}"
pids=()
for arm in ${ARMS}; do
  case "${arm}" in
    v21)      run_arm v21      "${GPU_V21}"   "${PORT_V21}"   rl_token false false false & pids+=($!) ;;
    flow_frz) run_arm flow_frz "${GPU_FRZ}"   "${PORT_FRZ}"   flow_rlt true  false false & pids+=($!) ;;
    flow_td)  run_arm flow_td  "${GPU_TD}"    "${PORT_TD}"    flow_rlt false false false & pids+=($!) ;;
    cf_ae)    run_arm cf_ae    "${GPU_CFAE}"  "${PORT_CFAE}"  flow_rlt false true  true  & pids+=($!) ;;
    cf_noae)  run_arm cf_noae  "${GPU_CFNOAE}" "${PORT_CFNOAE}" flow_rlt false true  false & pids+=($!) ;;
    *) echo "unknown arm ${arm}"; exit 1 ;;
  esac
done
wait "${pids[@]}"
log "arms done: ${ARMS}"
