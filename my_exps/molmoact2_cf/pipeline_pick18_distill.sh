#!/usr/bin/env bash
# V25: Iterated Action-Expert Distillation (IAD) on the Pick-18 pool.
#
# Folds the Stage A shared gOn policy (frozen pi0.5 + V + G, val 77.7%) back into a
# standalone pi0.5 checkpoint, so eval needs no V, no G, no AE. See V25_Distill.md.
#
# Each round: continue online RL (V/G warm-started, Stage A recipe) with the VLA servers
# on the CURRENT expert checkpoint, recording (observation, reference, teacher) at every
# corrected decision; then train a student action expert on that round's shards and make
# it the next round's expert. The backbone is never touched, so the shared AE and the RL
# state survive every swap.
#
#   round k: servers on E{k-1}, V/G resume round k-1, record teacher -> distill -> Ek
#   eval:    pure-VLA val-1000 on E5, E10, ..., E100 (and any EXTRA_EVALS)
#
# Reuses Stage A artifacts (no re-collect, no re-pretrain):
#   runs/pick18_v24_shared/ae/ae_pick18_shared.pt
#   runs/pick18_v24_shared/rl/shared_v24_s0/{agent.pt,buffer.npz}
#   runs/pick18_v24_shared/ae/tokens/*/*.npz        (token_replay for ae_finetune)
#
# Completed rounds are skipped, so a 100-round run continues from E2:
#   GPUS="0 1 2 3" ROUNDS=100 bash pipeline_pick18_distill.sh
#   PHASES="round1" GPUS="0 1 2 3" bash pipeline_pick18_distill.sh
#   PHASES="distill1" DISTILL_STEPS=200 bash pipeline_pick18_distill.sh   # smoke
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
CODE="${CODE:-${ROOT}/pi05_rl_token}"
B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
SRC_DIR="${SRC_DIR:-${ROOT}/runs/pick18_v24_shared}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/pick18_v25_distill}"
# RL checkpoints, metrics, and logs also live on jovyan: the workspace NFS
# still returns ENOSPC on small appends even when df shows leftover space.
JOVYAN_RUN="${JOVYAN_RUN:-/home/jovyan/users/staroverov/v25_run}"
LOCAL_LOG="${LOCAL_LOG:-${JOVYAN_RUN}/logs}"
SIM="${SIM:-${B1K_ROOT}/B1K_AIRI/submodules/molmospaces/.venv/bin/python}"
TORCH_PY="${TORCH_PY:-/home/jovyan/users/staroverov/pi05_molmospaces/venvs/pi05_torch/bin/python}"
CKPT="${CKPT:-/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999}"

read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
PHASES="${PHASES:-loop}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-6}"
COLLECTORS_PER_GPU="${COLLECTORS_PER_GPU:-8}"
VLA_SERVERS_PER_GPU="${VLA_SERVERS_PER_GPU:-2}"

ROUNDS="${ROUNDS:-100}"
START_ROUND="${START_ROUND:-1}"
ROUND_EPISODES="${ROUND_EPISODES:-600}"
PROBE_EPISODES="${PROBE_EPISODES:-36}"
ONLINE_BETA="${ONLINE_BETA:-1}"
DISTILL_STEPS="${DISTILL_STEPS:-12000}"
DISTILL_BATCH="${DISTILL_BATCH:-8}"
DISTILL_LR="${DISTILL_LR:-1e-5}"
EVAL_EVERY="${EVAL_EVERY:-10}"
EXTRA_EVALS="${EXTRA_EVALS:-5}"
CLEAN_DISTILL="${CLEAN_DISTILL:-1}"
PRUNE_EXPERTS="${PRUNE_EXPERTS:-1}"
# The workspace NFS is at capacity (~31T). New 7 GB experts and per-round distill
# shards go to the jovyan volume (785 GB free) so a round cannot fill the disk again.
EXPERT_ROOT="${EXPERT_ROOT:-/home/jovyan/users/staroverov/pi05_ckpt/v25_experts}"
DISTILL_ROOT="${DISTILL_ROOT:-/home/jovyan/users/staroverov/v25_distill_shards}"
ON_DIR="${ON_DIR:-${JOVYAN_RUN}/rl}"

OBJECTS=(remote ladle tissue spoon spatula pot soap_dispenser spray_bottle cup shaker fork bottle fruit bowl knife box)
read -r -a TASKS <<< "${TASKS:-desk_mug kettle ${OBJECTS[*]}}"

mkdir -p "${RUN_DIR}/pids" "${RUN_DIR}/summaries" "${LOCAL_LOG}" "${EXPERT_ROOT}" "${DISTILL_ROOT}" "${ON_DIR}"
ln -sfn "${LOCAL_LOG}" "${RUN_DIR}/logs"
echo $$ > "${RUN_DIR}/pids/pipeline.pid" || true
echo $$ > "${JOVYAN_RUN}/pipeline.pid"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${CODE}${PYTHONPATH:+:${PYTHONPATH}}"
export RLT_VLA_TOKEN_DIM=2048
export HF_HOME="${HF_HOME:-/home/jovyan/users/staroverov/.cache/huggingface}"

log() { echo "[pick18-v25-distill $(date -u +%H:%M:%S)] $*" || true; }
has_phase() { [[ " ${PHASES} " == *" $1 "* ]]; }

task_meta() {
  case "$1" in
    desk_mug) SCENE=desk_mug HORIZON=500 ;;
    kettle) SCENE=kettle HORIZON=400 ;;
    *) SCENE="${1}" HORIZON=500 ;;
  esac
}

task_pool_spec() {
  local parts=()
  local task
  for task in "${TASKS[@]}"; do
    task_meta "${task}"
    parts+=("${SCENE}:${HORIZON}")
  done
  local IFS=,
  echo "${parts[*]}"
}

AE="${SRC_DIR}/ae/ae_pick18_shared.pt"
AE_TOKENS="${SRC_DIR}/ae/tokens"
STAGEA_AGENT="${SRC_DIR}/rl/shared_v24_s0/agent.pt"
STAGEA_BUFFER="${SRC_DIR}/rl/shared_v24_s0/buffer.npz"
ON_DIR="${ON_DIR:-${JOVYAN_RUN}/rl}"

for f in "${AE}" "${STAGEA_AGENT}" "${STAGEA_BUFFER}"; do
  [[ -f "${f}" ]] || { log "ERROR: Stage A artifact missing: ${f}"; exit 1; }
done
[[ -d "${AE_TOKENS}" ]] || { log "ERROR: Stage A token replay dir missing: ${AE_TOKENS}"; exit 1; }

# Prefer an existing NFS copy (E1/E2/E5), otherwise the jovyan expert store.
expert_dir() {
  local k="$1"
  local nfs="${RUN_DIR}/expert_e${k}"
  local jov="${EXPERT_ROOT}/expert_e${k}"
  if [[ -f "${nfs}/model.safetensors" ]]; then
    echo "${nfs}"
  else
    echo "${jov}"
  fi
}

# expert checkpoint served in round k (1-indexed): round 1 serves the base pi0.5.
expert_for_round() {
  local k="$1"
  if (( k == 1 )); then
    echo "${CKPT}"
  else
    expert_dir "$((k - 1))"
  fi
}

# agent + buffer a round warm-starts V/G from.
prev_agent_for_round() {
  local k="$1"
  if (( k == 1 )); then echo "${STAGEA_AGENT}"; else echo "${ON_DIR}/v25_round$((k - 1))/agent.pt"; fi
}
prev_buffer_for_round() {
  local k="$1"
  if (( k == 1 )); then echo "${STAGEA_BUFFER}"; else echo "${ON_DIR}/v25_round$((k - 1))/buffer.npz"; fi
}

want_eval() {
  local k="$1"
  local extra
  (( k == ROUNDS )) && return 0
  if (( EVAL_EVERY > 0 )) && (( k % EVAL_EVERY == 0 )); then
    return 0
  fi
  for extra in ${EXTRA_EVALS}; do
    (( extra == k )) && return 0
  done
  return 1
}

keep_expert() {
  local n="$1"
  # Always keep the two Stage-A evals, the latest expert, and every eval point.
  (( n == 1 || n == 2 )) && return 0
  want_eval "${n}" && return 0
  return 1
}

record_round_progress() {
  local k="$1"
  local ed; ed="$(expert_dir "${k}")"
  "${SIM}" - "${k}" "${ON_DIR}/v25_round${k}/metrics.jsonl" \
    "${ed}/distill_summary.json" \
    "${RUN_DIR}/eval_val1000_e${k}/vla_summary.json" \
    "${RUN_DIR}/progress.jsonl" <<'PY'
import json, pathlib, sys
k = int(sys.argv[1])
metrics_p, distill_p, eval_p, out_p = map(pathlib.Path, sys.argv[2:6])
rows = []
if metrics_p.exists():
    for line in metrics_p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
succ = sum(int(r.get("success") or 0) for r in rows)
n = len(rows)
last = rows[-50:] if rows else []
row = {
    "round": k,
    "episodes": n,
    "online_sr": (succ / n) if n else None,
    "online_sr_last50": (sum(int(r.get("success") or 0) for r in last) / len(last)) if last else None,
}
if distill_p.exists():
    d = json.loads(distill_p.read_text())
    row["distill_decisions"] = d.get("decisions")
    row["distill_ref_loss"] = (d.get("self_check") or {}).get("reference")
    row["distill_teacher_loss"] = (d.get("self_check") or {}).get("teacher")
    row["distill_final_loss"] = d.get("final_loss")
    row["distill_hours"] = d.get("hours")
if eval_p.exists():
    e = json.loads(eval_p.read_text())
    row["val1000_sr"] = e.get("success_rate")
    row["val1000_successes"] = e.get("successes")
    row["val1000_ci95"] = e.get("ci95")
out_p.parent.mkdir(parents=True, exist_ok=True)
existing = []
if out_p.exists():
    for line in out_p.read_text().splitlines():
        if line.strip():
            prev = json.loads(line)
            if prev.get("round") != k:
                existing.append(prev)
existing.append(row)
out_p.write_text("".join(json.dumps(r) + "\n" for r in existing))
print(json.dumps(row))
PY
}

cleanup_after_round() {
  local k="$1"
  local data summary n_npz
  local ed; ed="$(expert_dir "${k}")"
  summary="${ed}/distill_summary.json"
  if [[ -f "${summary}" ]]; then
    cp -f "${summary}" "${RUN_DIR}/summaries/distill_e${k}.json"
  fi
  if [[ "${CLEAN_DISTILL}" == "1" ]]; then
    for data in "${DISTILL_ROOT}/round${k}/distill" "${RUN_DIR}/round${k}/distill"; do
      if [[ -d "${data}" ]]; then
        n_npz="$(ls "${data}"/*.npz 2>/dev/null | wc -l)"
        rm -f "${data}"/*.npz
        log "cleanup: removed ${n_npz} distill shards from ${data}"
      fi
    done
  fi
  if [[ "${PRUNE_EXPERTS}" == "1" && k -ge 2 ]]; then
    local prev=$((k - 1))
    if ! keep_expert "${prev}"; then
      local prev_dir
      for prev_dir in "${RUN_DIR}/expert_e${prev}" "${EXPERT_ROOT}/expert_e${prev}"; do
        if [[ -f "${prev_dir}/distill_summary.json" ]]; then
          cp -f "${prev_dir}/distill_summary.json" "${RUN_DIR}/summaries/distill_e${prev}.json"
        fi
        if [[ -f "${prev_dir}/model.safetensors" ]]; then
          rm -f "${prev_dir}/model.safetensors"
          log "cleanup: pruned ${prev_dir}/model.safetensors (not an eval checkpoint)"
        fi
      done
    fi
  fi
  # Drop replay buffers older than two rounds; keep metrics + agent for resume/debug.
  if (( k >= 4 )); then
    local old=$((k - 3))
    local old_dir="${ON_DIR}/v25_round${old}"
    if [[ -d "${old_dir}" ]]; then
      rm -f "${old_dir}/buffer.npz" "${old_dir}/actor_live.pt" "${old_dir}/agent_best.pt"
    fi
  fi
}

# --------------------------------------------------------------------------------------
# one online round: RL continues, servers on the current expert, recording on
# --------------------------------------------------------------------------------------
run_round() {
  local k="$1"
  local tag="v25_round${k}"
  local expert; expert="$(expert_for_round "${k}")"
  local init_actor; init_actor="$(prev_agent_for_round "${k}")"
  local init_buffer; init_buffer="$(prev_buffer_for_round "${k}")"
  local distill_out="${DISTILL_ROOT}/round${k}/distill"
  mkdir -p "${distill_out}"
  # A round that died mid-way may still have shards on the NFS path; keep using them.
  if [[ ! -e "${distill_out}"/.keep_nfs && -d "${RUN_DIR}/round${k}/distill" ]]; then
    local n_nfs
    n_nfs="$(ls "${RUN_DIR}/round${k}/distill"/distill_*.npz 2>/dev/null | wc -l)"
    if (( n_nfs > 0 )); then
      log "round${k}: moving ${n_nfs} NFS distill shards -> ${distill_out}"
      mv "${RUN_DIR}/round${k}/distill"/distill_*.npz "${distill_out}/" 2>/dev/null || true
    fi
  fi

  local episodes_done=0
  if [[ -f "${ON_DIR}/${tag}/metrics.jsonl" ]]; then
    episodes_done="$(python3 -c "import pathlib,sys
n=0
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line.strip(): n+=1
print(n)" "${ON_DIR}/${tag}/metrics.jsonl")"
  fi
  if [[ -f "${ON_DIR}/${tag}/agent.pt" && "${episodes_done}" -ge "${ROUND_EPISODES}" ]]; then
    log "round${k}: already finished ${episodes_done} eps, skip"
    return 0
  fi
  if [[ -f "${RUN_DIR}/summaries/distill_e${k}.json" || -f "$(expert_dir "${k}")/model.safetensors" || -f "$(expert_dir "${k}")/distill_summary.json" ]]; then
    log "round${k}: later distill already exists, skip"
    return 0
  fi
  [[ -f "${expert}" || -d "${expert}" ]] || { log "ERROR: round${k} expert missing: ${expert}"; exit 1; }
  [[ -f "${init_actor}" ]] || { log "ERROR: round${k} init actor missing: ${init_actor}"; exit 1; }
  [[ -f "${init_buffer}" ]] || { log "ERROR: round${k} init buffer missing: ${init_buffer}"; exit 1; }

  vla_ports=(); vla_gpus=(); egl_devices=()
  local base_port=9600
  local i s
  for i in "${!GPUS[@]}"; do
    for ((s = 0; s < VLA_SERVERS_PER_GPU; s++)); do
      vla_ports+=($((base_port + i * 10 + s)))
      vla_gpus+=("${GPUS[$i]}")
      egl_devices+=("${GPUS[$i]}")
    done
  done
  join_by() { local IFS="$1"; shift; echo "$*"; }
  local n_collectors=$(( ${#GPUS[@]} * COLLECTORS_PER_GPU ))
  local resume_args=()
  if [[ "${episodes_done}" -gt 0 ]]; then
    resume_args+=(--set "resume=true")
    log "round${k}: resume from ${episodes_done}/${ROUND_EPISODES}, ${n_collectors} collectors, ${#vla_ports[@]} VLA servers, egl_slots=${SLOTS_PER_GPU}, expert=${expert}"
  else
    log "round${k}: ${ROUND_EPISODES} eps over ${#TASKS[@]} tasks, ${n_collectors} collectors, ${#vla_ports[@]} VLA servers, egl_slots=${SLOTS_PER_GPU}, expert=${expert}, recording -> ${distill_out}"
  fi
  cd "${CODE}"
  set +e
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" RLT_VLA_TOKEN_DIM=2048 HF_HOME="${HF_HOME}" \
    "${SIM}" scripts/run_train.py \
      --scene desk_mug --encoder pick18_shared \
      --gpu "${GPUS[0]}" --port "${base_port}" \
      --episodes "${ROUND_EPISODES}" --warmup-episodes 0 \
      --tag "${tag}" --init-actor "${init_actor}" \
      --set "algorithm=v22_24" \
      --set "init_random_ae=false" \
      --set "token_ae=${AE}" \
      --set "beta=${ONLINE_BETA}" \
      --set "cf_actor_coef=0" \
      --set "cf_ref_conditioned=true" \
      --set "gate_step=0" \
      --set "gate_frac=0" \
      --set "seed=0" \
      --set "init_buffer=${init_buffer}" \
      --set "task_pool=$(task_pool_spec)" \
      --set "episode_pool=0-11" \
      --set "probe_episodes=${PROBE_EPISODES}" \
      --set "collectors=${n_collectors}" \
      --set "egl_slots=${SLOTS_PER_GPU}" \
      --set "vla_ports=$(join_by , "${vla_ports[@]}")" \
      --set "vla_gpus=$(join_by , "${vla_gpus[@]}")" \
      --set "egl_devices=$(join_by , "${egl_devices[@]}")" \
      --set "buffer_capacity=400000" \
      --set "out_dir=${ON_DIR}" \
      --set "checkpoint=${expert}" \
      --set "train_token_offline=false" \
      --set "train_token_online=false" \
      --set "ae_finetune=true" \
      --set "store_decision_tokens=false" \
      --set "token_replay=${AE_TOKENS}/*/*.npz" \
      --set "distill_out=${distill_out}" \
      "${resume_args[@]}" \
      >> "${LOCAL_LOG}/${tag}.log" 2>&1
  train_rc=$?
  set -e
  episodes_done=0
  if [[ -f "${ON_DIR}/${tag}/metrics.jsonl" ]]; then
    episodes_done="$(python3 -c "import pathlib,sys
n=0
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if line.strip(): n+=1
print(n)" "${ON_DIR}/${tag}/metrics.jsonl")"
  fi
  if (( train_rc != 0 )) || [[ ! -f "${ON_DIR}/${tag}/agent.pt" ]] || (( episodes_done < ROUND_EPISODES )); then
    log "ERROR: round${k} stopped at ${episodes_done}/${ROUND_EPISODES} eps rc=${train_rc} (see ${LOCAL_LOG}/${tag}.log)"
    exit 1
  fi
}

# --------------------------------------------------------------------------------------
# distill round k's shards into the next expert checkpoint (1 GPU)
# --------------------------------------------------------------------------------------
run_distill() {
  local k="$1"
  local data="${DISTILL_ROOT}/round${k}/distill"
  if ! ls "${data}"/distill_*.npz >/dev/null 2>&1; then
    data="${RUN_DIR}/round${k}/distill"
  fi
  local init; init="$(expert_for_round "${k}")"
  local out; out="$(expert_dir "${k}")"
  if [[ -f "${out}/model.safetensors" || -f "${out}/distill_summary.json" || -f "${RUN_DIR}/summaries/distill_e${k}.json" ]]; then
    log "distill${k}: ${out} exists, skip"
    return 0
  fi
  if ! ls "${data}"/distill_*.npz >/dev/null 2>&1; then
    log "ERROR: distill${k} has no shards in ${data} and no existing expert"
    exit 1
  fi
  log "distill${k}: train student on ${data} (init=${init}, steps=${DISTILL_STEPS}) -> ${out}"
  cd "${CODE}"
  setsid env PYTHONUNBUFFERED=1 PYTHONPATH="${CODE}" HF_HOME="${HF_HOME}" HF_HUB_OFFLINE=1 \
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" TORCHINDUCTOR_CUDAGRAPHS=0 \
    "${TORCH_PY}" scripts/train_expert_distill.py \
      --data "${data}" \
      --init "${init}" \
      --out "${out}" \
      --steps "${DISTILL_STEPS}" \
      --batch-size "${DISTILL_BATCH}" \
      --lr "${DISTILL_LR}" \
      > "${LOCAL_LOG}/distill_e${k}.log" 2>&1
  if [[ ! -f "${out}/model.safetensors" ]]; then
    log "ERROR: distill${k} produced no checkpoint (see ${LOCAL_LOG}/distill_e${k}.log)"
    exit 1
  fi
}

# --------------------------------------------------------------------------------------
# official MolmoPick val-1000 for a student expert, as a pure VLA (no actor/AE/guidance)
# --------------------------------------------------------------------------------------
run_eval() {
  local k="$1"
  local expert; expert="$(expert_dir "${k}")"
  local eval_dir="${RUN_DIR}/eval_val1000_e${k}"
  local eval_log="${LOCAL_LOG}/eval_e${k}"
  if [[ -f "${eval_dir}/vla_summary.json" ]]; then
    log "eval${k}: exists, skip"
    return 0
  fi
  [[ -f "${expert}/model.safetensors" ]] || { log "ERROR: eval${k} expert missing: ${expert}"; exit 1; }
  mkdir -p "${eval_dir}" "${eval_log}"
  if [[ ! -e "${eval_dir}/shards" && -d "${SRC_DIR}/eval_val1000/shards" ]]; then
    ln -sfn "${SRC_DIR}/eval_val1000/shards" "${eval_dir}/shards"
    log "eval${k}: reusing Stage A val-1000 shards"
  fi
  log "eval${k}: ${expert} on 1000 val episodes (pure VLA)"
  cd "${ROOT}"
  if ! env RUN_DIR="${RUN_DIR}" EVAL_DIR="${eval_dir}" CKPT="${expert}" \
      LOCAL_LOG="${eval_log}" PHASES="vla" GPUS="${GPUS[*]}" \
      bash pipeline_eval_val1000.sh; then
    log "ERROR: eval${k} failed"
    return 1
  fi
  # MolmoSpaces still dumps mp4/h5 even with save_video=false; drop them once
  # result.json exists so a 100-round loop cannot refill the NFS.
  rm -rf "${eval_dir}"/*/eval_output
}

run_one_round() {
  local k="$1"
  run_round "${k}"
  run_distill "${k}"
  if want_eval "${k}"; then
    run_eval "${k}" || return 1
  fi
  record_round_progress "${k}" || true
  cleanup_after_round "${k}"
}

fail=0
if has_phase loop; then
  log "loop: rounds ${START_ROUND}..${ROUNDS}, eval every ${EVAL_EVERY} plus [${EXTRA_EVALS}], ${#GPUS[@]} GPUs"
  local_k="${START_ROUND}"
  for ((local_k = START_ROUND; local_k <= ROUNDS; local_k++)); do
    run_one_round "${local_k}" || fail=1
    if (( fail != 0 )); then
      log "ERROR: stopping at round ${local_k}"
      break
    fi
  done
else
  if has_phase round1;   then run_round 1 || fail=1; fi
  if has_phase distill1; then run_distill 1 || fail=1; fi
  if has_phase round2;   then run_round 2 || fail=1; fi
  if has_phase distill2; then run_distill 2 || fail=1; fi
  if has_phase eval; then
    run_eval 1 || fail=1
    if [[ -f "${RUN_DIR}/expert_e2/model.safetensors" ]]; then
      run_eval 2 || fail=1
    fi
  fi
fi

log "pick18 V25 distill finished (fail=${fail})"
exit "${fail}"
