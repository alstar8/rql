#!/usr/bin/env bash
# Read-only evaluation of immutable V15 snapshots into V15 validation outputs.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v15_controlled}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
VAL_BENCHMARK="${BENCHMARK_ROOT}/val"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
HELPER="${ROOT}/v15_harness.py"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_eval_v15_controlled}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v15_controlled}"
V15_EVAL_EPISODES="${V15_EVAL_EPISODES:-}"
V15_EVAL_VARIANTS="${V15_EVAL_VARIANTS:-}"
V15_EVAL_POLICIES="${V15_EVAL_POLICIES:-}"

if [[ "${RUN_DIR}" != /* ]]; then
  RUN_DIR="${ROOT}/${RUN_DIR}"
fi
if [[ "${BENCHMARK_ROOT}" != /* ]]; then
  BENCHMARK_ROOT="${ROOT}/${BENCHMARK_ROOT}"
  VAL_BENCHMARK="${BENCHMARK_ROOT}/val"
fi
RUN_DIR="${RUN_DIR%/}"
BENCHMARK_ROOT="${BENCHMARK_ROOT%/}"
VAL_BENCHMARK="${VAL_BENCHMARK%/}"
if [[ "$(basename "${RUN_DIR}")" != "rlt_cf_v15_controlled" ]]; then
  echo "[v15-eval] RUN_DIR must be the isolated V15 run: ${RUN_DIR}" >&2
  exit 1
fi
case "${RUN_DIR}" in
  *rlt_cf_v13_controlled*|*rlt_cf_v14_controlled*)
    echo "[v15-eval] refusing V13 path: ${RUN_DIR}" >&2
    exit 1
    ;;
esac
if [[ ! -x "${PYTHON}" ]]; then
  echo "[v15-eval] Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/MANIFEST.json" ]]; then
  echo "[v15-eval] missing immutable V15 manifest: ${RUN_DIR}/MANIFEST.json" >&2
  exit 1
fi
if [[ ! -f "${VAL_BENCHMARK}/benchmark.json" ]]; then
  echo "[v15-eval] missing held-out validation benchmark: ${VAL_BENCHMARK}" >&2
  exit 1
fi

IFS=$'\t' read -r \
  V15_MODE \
  V15_MAX_VALID_EPISODES \
  V15_TARGET_ENV_STEPS \
  MANIFEST_SNAPSHOT_EPISODES \
  V15_AE_BATCH_SIZE \
  V15_AE_MICROBATCH_SIZE \
  V15_AE_MIN_SUCCESS_EPISODES \
  V15_MAX_UPDATE_SEC_PER_EPISODE \
  < <(
    "${PYTHON}" "${HELPER}" manifest-run-settings \
      --manifest "${RUN_DIR}/MANIFEST.json" \
      --format tsv
  )
V15_SNAPSHOT_EPISODES="${MANIFEST_SNAPSHOT_EPISODES}"
if [[ -z "${V15_EVAL_EPISODES}" ]]; then
  V15_EVAL_EPISODES="${MANIFEST_SNAPSHOT_EPISODES}"
fi
export V15_MODE V15_MAX_VALID_EPISODES V15_TARGET_ENV_STEPS
export V15_SNAPSHOT_EPISODES V15_AE_BATCH_SIZE V15_AE_MICROBATCH_SIZE
export V15_AE_MIN_SUCCESS_EPISODES
export V15_MAX_UPDATE_SEC_PER_EPISODE

IFS=',' read -r -a EVAL_EPISODES <<< "${V15_EVAL_EPISODES}"
if (( ${#EVAL_EPISODES[@]} == 0 )); then
  echo "[v15-eval] V15_EVAL_EPISODES is empty" >&2
  exit 1
fi
for episode in "${EVAL_EPISODES[@]}"; do
  if [[ ! "${episode}" =~ ^[0-9]+$ ]]; then
    echo "[v15-eval] invalid episode filter: ${episode}" >&2
    exit 1
  fi
done

echo "[v15-eval] validating manifest and controlled benchmark"
"${PYTHON}" "${HELPER}" validate-manifest \
  --manifest "${RUN_DIR}/MANIFEST.json" \
  --run-dir "${RUN_DIR}"
"${PYTHON}" "${ROOT}/generate_controlled_benchmark.py" \
  --output-root "${BENCHMARK_ROOT}" \
  --validate-only

manifest_gpu_ids="$("${PYTHON}" "${HELPER}" manifest-gpu-ids \
  --manifest "${RUN_DIR}/MANIFEST.json")"
IFS=',' read -r -a GPU_ARRAY <<< "${manifest_gpu_ids}"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "[v15-eval] manifest does not contain eight GPU assignments" >&2
  exit 1
fi
mapfile -t VARIANT_ROWS < <("${PYTHON}" "${HELPER}" variants --format tsv)
if (( ${#VARIANT_ROWS[@]} != 8 )); then
  echo "[v15-eval] expected eight variants, got ${#VARIANT_ROWS[@]}" >&2
  exit 1
fi

csv_contains() {
  local csv="$1"
  local needle="$2"
  local item
  [[ -n "${csv}" ]] || return 0
  local -a items=()
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

validate_variant_filters() {
  [[ -n "${V15_EVAL_VARIANTS}" ]] || return 0
  local -a requested=()
  local requested_variant row found
  IFS=',' read -r -a requested <<< "${V15_EVAL_VARIANTS}"
  for requested_variant in "${requested[@]}"; do
    found=0
    for row in "${VARIANT_ROWS[@]}"; do
      if [[ "${row%%|*}" == "${requested_variant}" ]]; then
        found=1
        break
      fi
    done
    if (( found == 0 )); then
      echo "[v15-eval] unknown variant filter: ${requested_variant}" >&2
      return 1
    fi
  done
}

validate_policy_filters() {
  [[ -n "${V15_EVAL_POLICIES}" ]] || return 0
  local -a requested=()
  local policy
  IFS=',' read -r -a requested <<< "${V15_EVAL_POLICIES}"
  for policy in "${requested[@]}"; do
    case "${policy}" in
      reference|reference_noise|actor|actor_guide) ;;
      *)
        echo "[v15-eval] unknown policy filter: ${policy}" >&2
        return 1
        ;;
    esac
  done
}

validate_variant_filters
validate_policy_filters

mkdir -p "${RUN_DIR}/validation" "${TMP_ROLLOUT_DIR}" "${RLT_EGL_LOCK_DIR}"
LOCK_FILE="${B1K_TMP}/rlt_cf_v15_controlled_eval.lock"
exec 9> "${LOCK_FILE}"
if ! flock -n 9; then
  echo "[v15-eval] another V15 evaluator holds ${LOCK_FILE}" >&2
  exit 1
fi

pid_first_field() {
  local pidfile="$1"
  local pid=""
  [[ -f "${pidfile}" ]] || return 1
  IFS=$' \t\r\n' read -r pid _ < "${pidfile}"
  if [[ ! "${pid}" =~ ^[0-9]+$ ]] || (( pid <= 1 )); then
    return 1
  fi
  printf '%s\n' "${pid}"
}

pid_is_live_v15_owned() {
  local pidfile="$1"
  local pid
  pid="$(pid_first_field "${pidfile}")" || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/environ" ]] || return 1
  local entry
  while IFS= read -r entry; do
    if [[ "${entry}" == "RLT_CF_V15_RUN_DIR=${RUN_DIR}" ]]; then
      return 0
    fi
  done < <(tr '\0' '\n' < "/proc/${pid}/environ")
  return 1
}

server_ready() {
  local variant="$1"
  local port="$2"
  local pidfile="${RUN_DIR}/pids/server_${variant}.pid"
  if ! pid_is_live_v15_owned "${pidfile}"; then
    return 1
  fi
  local response
  if ! response="$(curl -sf --max-time 3 "http://127.0.0.1:${port}/healthz" 2>/dev/null)"; then
    return 1
  fi
  [[ "${response}" == *'"status":"ok"'* ]]
}

eval_command() {
  local variant="$1"
  local episode="$2"
  local policy="$3"
  local seed="$4"
  mapfile -d '' -t GENERATED_COMMAND < <(
    "${PYTHON}" "${HELPER}" eval-command \
      --variant "${variant}" \
      --root "${ROOT}" \
      --run-dir "${RUN_DIR}" \
      --benchmark-val "${VAL_BENCHMARK}" \
      --tmp-rollout-dir "${TMP_ROLLOUT_DIR}" \
      --python-executable "${PYTHON}" \
      --episode "${episode}" \
      --policy "${policy}" \
      --seed "${seed}" \
      --format nul
  )
  if (( ${#GENERATED_COMMAND[@]} == 0 )); then
    echo "[v15-eval] failed to generate command for ${variant}" >&2
    return 1
  fi
}

run_one() {
  local variant="$1"
  local gpu="$2"
  local ae_mode="$3"
  local episode="$4"
  local policy="$5"
  local seed="$6"
  local snapshot_name
  snapshot_name="$(printf 'ep_%06d' "${episode}")"
  local source_dir="${RUN_DIR}/${variant}/snapshots/${snapshot_name}"
  local out_dir="${RUN_DIR}/validation/${variant}/${snapshot_name}/${policy}/seed_${seed}"

  if [[ ! -f "${source_dir}/rlt_cf.pt" || ! -f "${source_dir}/snapshot.json" ]]; then
    echo "[v15-eval] requested immutable snapshot is unavailable: ${source_dir}" >&2
    return 1
  fi
  if [[ "${ae_mode}" == "1" && ! -f "${source_dir}/molmo_ae_lora.pt" ]]; then
    echo "[v15-eval] requested AE adapter snapshot is unavailable: ${source_dir}" >&2
    return 1
  fi
  if "${PYTHON}" "${HELPER}" evaluation-complete \
    --out-dir "${out_dir}" \
    --expected-episodes 12 >/dev/null 2>&1; then
    echo "[v15-eval] skip complete ${variant}/${snapshot_name}/${policy}/seed_${seed}"
    return 0
  fi
  if [[ -e "${out_dir}" ]]; then
    # Another worker may already own this cell (parallel fan-out). Wait for it.
    local waited=0
    while (( waited < 7200 )); do
      if "${PYTHON}" "${HELPER}" evaluation-complete \
        --out-dir "${out_dir}" \
        --expected-episodes 12 >/dev/null 2>&1; then
        echo "[v15-eval] adopted in-flight complete ${variant}/${snapshot_name}/${policy}/seed_${seed}"
        return 0
      fi
      # Still incomplete and no live writer → hard fail.
      if ! pgrep -f "out_dir ${out_dir}|--out_dir ${out_dir}" >/dev/null 2>&1 \
        && ! pgrep -f "${out_dir}" >/dev/null 2>&1; then
        # pgrep on path is noisy; treat missing eval.log mtime progress as stale.
        if [[ ! -f "${out_dir}/eval.log" ]] || [[ $(( $(date +%s) - $(stat -c %Y "${out_dir}/eval.log") )) -gt 300 ]]; then
          echo "[v15-eval] refusing to overwrite incomplete output: ${out_dir}" >&2
          return 1
        fi
      fi
      sleep 15
      waited=$((waited + 15))
    done
    echo "[v15-eval] timed out waiting for in-flight ${out_dir}" >&2
    return 1
  fi
  case "${out_dir}" in
    "${RUN_DIR}/validation/"*) ;;
    *)
      echo "[v15-eval] refusing unsafe validation output path: ${out_dir}" >&2
      return 1
      ;;
  esac
  mkdir -p "${out_dir}"
  eval_command "${variant}" "${episode}" "${policy}" "${seed}"
  local -a command=("${GENERATED_COMMAND[@]}")
  local -a egl_env=()
  if [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    egl_env=(MUJOCO_EGL_DEVICE_ID="${gpu}")
  fi

  echo "[v15-eval] run ${variant}/${snapshot_name}/${policy}/seed_${seed} gpu=${gpu}"
  env \
    RLT_CF_V15_RUN_DIR="${RUN_DIR}" \
    RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
    RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-16}" \
    RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-2}" \
    RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${egl_env[@]}" \
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
    "${command[@]}" \
    > "${out_dir}/eval.log" 2>&1

  if ! "${PYTHON}" "${HELPER}" evaluation-complete \
    --out-dir "${out_dir}" \
    --expected-episodes 12 >/dev/null 2>&1; then
    echo "[v15-eval] evaluation did not produce a complete result: ${out_dir}" >&2
    return 1
  fi
}

# Fan out every (variant, episode, policy, seed) cell across GPUs instead of
# serializing policies inside each variant (that left most GPUs idle).
MAX_PARALLEL_EVALS="${MAX_PARALLEL_EVALS:-8}"
declare -a EVAL_PIDS=()
declare -a EVAL_NAMES=()
declare -a CELL_QUEUE=()
gpu_rr=0
failed=0

wait_for_eval_slot() {
  while (( ${#EVAL_PIDS[@]} >= MAX_PARALLEL_EVALS )); do
    local index surviving_pids surviving_names
    surviving_pids=()
    surviving_names=()
    for index in "${!EVAL_PIDS[@]}"; do
      if kill -0 "${EVAL_PIDS[$index]}" 2>/dev/null; then
        surviving_pids+=("${EVAL_PIDS[$index]}")
        surviving_names+=("${EVAL_NAMES[$index]}")
        continue
      fi
      if ! wait "${EVAL_PIDS[$index]}"; then
        echo "[v15-eval] cell failed: ${EVAL_NAMES[$index]}" >&2
        failed=1
      fi
    done
    EVAL_PIDS=("${surviving_pids[@]}")
    EVAL_NAMES=("${surviving_names[@]}")
    if (( ${#EVAL_PIDS[@]} >= MAX_PARALLEL_EVALS )); then
      sleep 2
    fi
  done
}

for row in "${VARIANT_ROWS[@]}"; do
  variant="${row%%|*}"
  if ! csv_contains "${V15_EVAL_VARIANTS}" "${variant}"; then
    continue
  fi
  IFS='|' read -r variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
  if [[ -n "${port}" ]] && ! server_ready "${variant}" "${port}"; then
    echo "[v15-eval] V15-owned server is unavailable for ${variant} on ${port}" >&2
    exit 1
  fi

  local_policies=()
  if [[ "${actor_mode}" == "vla_only" ]]; then
    local_policies=(reference)
  elif [[ "${variant}" == "residual_rlt_actor" || "${variant}" == "flow_rlt_actor" ]]; then
    local_policies=(reference reference_noise actor)
  elif [[ "${guide}" == "1" ]]; then
    local_policies=(reference actor actor_guide)
  else
    local_policies=(reference actor)
  fi

  for episode in "${EVAL_EPISODES[@]}"; do
    if (( episode == 400 )); then
      seeds=(20260831 20260832 20260833 20260834)
    else
      seeds=(20260831)
    fi
    for policy in "${local_policies[@]}"; do
      if ! csv_contains "${V15_EVAL_POLICIES}" "${policy}"; then
        continue
      fi
      for seed in "${seeds[@]}"; do
        CELL_QUEUE+=("${variant}|${ae_mode}|${episode}|${policy}|${seed}")
      done
    done
  done
done

if (( ${#CELL_QUEUE[@]} == 0 )); then
  echo "[v15-eval] filters selected no evaluation cells" >&2
  exit 1
fi

echo "[v15-eval] scheduling ${#CELL_QUEUE[@]} cells with max_parallel=${MAX_PARALLEL_EVALS} egl_per_gpu=${RLT_EGL_PER_GPU:-2}"

for cell in "${CELL_QUEUE[@]}"; do
  IFS='|' read -r variant ae_mode episode policy seed <<< "${cell}"
  wait_for_eval_slot
  gpu="${GPU_ARRAY[$((gpu_rr % ${#GPU_ARRAY[@]}))]}"
  gpu_rr=$((gpu_rr + 1))
  name="${variant}/ep_$(printf '%06d' "${episode}")/${policy}/seed_${seed}|gpu=${gpu}"
  run_one "${variant}" "${gpu}" "${ae_mode}" "${episode}" "${policy}" "${seed}" &
  EVAL_PIDS+=("$!")
  EVAL_NAMES+=("${name}")
done

for index in "${!EVAL_PIDS[@]}"; do
  if ! wait "${EVAL_PIDS[$index]}"; then
    echo "[v15-eval] cell failed: ${EVAL_NAMES[$index]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

echo "[v15-eval] all requested immutable-snapshot evaluations are complete"
