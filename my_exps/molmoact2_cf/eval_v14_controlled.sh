#!/usr/bin/env bash
# Read-only evaluation of immutable V14 snapshots into V14 validation outputs.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v14_controlled}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
VAL_BENCHMARK="${BENCHMARK_ROOT}/val"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
HELPER="${ROOT}/v14_harness.py"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_eval_v14_controlled}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v14_controlled}"
V14_EVAL_EPISODES="${V14_EVAL_EPISODES:-}"
V14_EVAL_VARIANTS="${V14_EVAL_VARIANTS:-}"
V14_EVAL_POLICIES="${V14_EVAL_POLICIES:-}"

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
if [[ "$(basename "${RUN_DIR}")" != "rlt_cf_v14_controlled" ]]; then
  echo "[v14-eval] RUN_DIR must be the isolated V14 run: ${RUN_DIR}" >&2
  exit 1
fi
case "${RUN_DIR}" in
  *rlt_cf_v13_controlled*)
    echo "[v14-eval] refusing V13 path: ${RUN_DIR}" >&2
    exit 1
    ;;
esac
if [[ ! -x "${PYTHON}" ]]; then
  echo "[v14-eval] Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/MANIFEST.json" ]]; then
  echo "[v14-eval] missing immutable V14 manifest: ${RUN_DIR}/MANIFEST.json" >&2
  exit 1
fi
if [[ ! -f "${VAL_BENCHMARK}/benchmark.json" ]]; then
  echo "[v14-eval] missing held-out validation benchmark: ${VAL_BENCHMARK}" >&2
  exit 1
fi

IFS=$'\t' read -r \
  V14_MODE \
  V14_MAX_VALID_EPISODES \
  V14_TARGET_ENV_STEPS \
  MANIFEST_SNAPSHOT_EPISODES \
  V14_AE_BATCH_SIZE \
  V14_AE_MICROBATCH_SIZE \
  V14_AE_MIN_SUCCESS_EPISODES \
  V14_MAX_UPDATE_SEC_PER_EPISODE \
  < <(
    "${PYTHON}" "${HELPER}" manifest-run-settings \
      --manifest "${RUN_DIR}/MANIFEST.json" \
      --format tsv
  )
V14_SNAPSHOT_EPISODES="${MANIFEST_SNAPSHOT_EPISODES}"
if [[ -z "${V14_EVAL_EPISODES}" ]]; then
  V14_EVAL_EPISODES="${MANIFEST_SNAPSHOT_EPISODES}"
fi
export V14_MODE V14_MAX_VALID_EPISODES V14_TARGET_ENV_STEPS
export V14_SNAPSHOT_EPISODES V14_AE_BATCH_SIZE V14_AE_MICROBATCH_SIZE
export V14_AE_MIN_SUCCESS_EPISODES
export V14_MAX_UPDATE_SEC_PER_EPISODE

IFS=',' read -r -a EVAL_EPISODES <<< "${V14_EVAL_EPISODES}"
if (( ${#EVAL_EPISODES[@]} == 0 )); then
  echo "[v14-eval] V14_EVAL_EPISODES is empty" >&2
  exit 1
fi
for episode in "${EVAL_EPISODES[@]}"; do
  if [[ ! "${episode}" =~ ^[0-9]+$ ]]; then
    echo "[v14-eval] invalid episode filter: ${episode}" >&2
    exit 1
  fi
done

echo "[v14-eval] validating manifest and controlled benchmark"
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
  echo "[v14-eval] manifest does not contain eight GPU assignments" >&2
  exit 1
fi
mapfile -t VARIANT_ROWS < <("${PYTHON}" "${HELPER}" variants --format tsv)
if (( ${#VARIANT_ROWS[@]} != 8 )); then
  echo "[v14-eval] expected eight variants, got ${#VARIANT_ROWS[@]}" >&2
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
  [[ -n "${V14_EVAL_VARIANTS}" ]] || return 0
  local -a requested=()
  local requested_variant row found
  IFS=',' read -r -a requested <<< "${V14_EVAL_VARIANTS}"
  for requested_variant in "${requested[@]}"; do
    found=0
    for row in "${VARIANT_ROWS[@]}"; do
      if [[ "${row%%|*}" == "${requested_variant}" ]]; then
        found=1
        break
      fi
    done
    if (( found == 0 )); then
      echo "[v14-eval] unknown variant filter: ${requested_variant}" >&2
      return 1
    fi
  done
}

validate_policy_filters() {
  [[ -n "${V14_EVAL_POLICIES}" ]] || return 0
  local -a requested=()
  local policy
  IFS=',' read -r -a requested <<< "${V14_EVAL_POLICIES}"
  for policy in "${requested[@]}"; do
    case "${policy}" in
      reference|reference_noise|actor|actor_guide) ;;
      *)
        echo "[v14-eval] unknown policy filter: ${policy}" >&2
        return 1
        ;;
    esac
  done
}

validate_variant_filters
validate_policy_filters

mkdir -p "${RUN_DIR}/validation" "${TMP_ROLLOUT_DIR}" "${RLT_EGL_LOCK_DIR}"
LOCK_FILE="${B1K_TMP}/rlt_cf_v14_controlled_eval.lock"
exec 9> "${LOCK_FILE}"
if ! flock -n 9; then
  echo "[v14-eval] another V14 evaluator holds ${LOCK_FILE}" >&2
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

pid_is_live_v14_owned() {
  local pidfile="$1"
  local pid
  pid="$(pid_first_field "${pidfile}")" || return 1
  kill -0 "${pid}" 2>/dev/null || return 1
  [[ -r "/proc/${pid}/environ" ]] || return 1
  local entry
  while IFS= read -r entry; do
    if [[ "${entry}" == "RLT_CF_V14_RUN_DIR=${RUN_DIR}" ]]; then
      return 0
    fi
  done < <(tr '\0' '\n' < "/proc/${pid}/environ")
  return 1
}

server_ready() {
  local variant="$1"
  local port="$2"
  local pidfile="${RUN_DIR}/pids/server_${variant}.pid"
  if ! pid_is_live_v14_owned "${pidfile}"; then
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
    echo "[v14-eval] failed to generate command for ${variant}" >&2
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
    echo "[v14-eval] requested immutable snapshot is unavailable: ${source_dir}" >&2
    return 1
  fi
  if [[ "${ae_mode}" == "1" && ! -f "${source_dir}/molmo_ae_lora.pt" ]]; then
    echo "[v14-eval] requested AE adapter snapshot is unavailable: ${source_dir}" >&2
    return 1
  fi
  if "${PYTHON}" "${HELPER}" evaluation-complete \
    --out-dir "${out_dir}" \
    --expected-episodes 12 >/dev/null 2>&1; then
    echo "[v14-eval] skip complete ${variant}/${snapshot_name}/${policy}/seed_${seed}"
    return 0
  fi
  if [[ -e "${out_dir}" ]]; then
    echo "[v14-eval] refusing to overwrite incomplete output: ${out_dir}" >&2
    return 1
  fi
  case "${out_dir}" in
    "${RUN_DIR}/validation/"*) ;;
    *)
      echo "[v14-eval] refusing unsafe validation output path: ${out_dir}" >&2
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

  echo "[v14-eval] run ${variant}/${snapshot_name}/${policy}/seed_${seed} gpu=${gpu}"
  env \
    RLT_CF_V14_RUN_DIR="${RUN_DIR}" \
    RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
    RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-8}" \
    RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-1}" \
    RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}" \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    "${egl_env[@]}" \
    HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
    "${command[@]}" \
    > "${out_dir}/eval.log" 2>&1

  if ! "${PYTHON}" "${HELPER}" evaluation-complete \
    --out-dir "${out_dir}" \
    --expected-episodes 12 >/dev/null 2>&1; then
    echo "[v14-eval] evaluation did not produce a complete result: ${out_dir}" >&2
    return 1
  fi
}

run_variant() {
  local row="$1"
  local variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port
  IFS='|' read -r variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
  if [[ -n "${port}" ]] && ! server_ready "${variant}" "${port}"; then
    echo "[v14-eval] V14-owned server is unavailable for ${variant} on ${port}" >&2
    return 1
  fi

  local -a policies=()
  if [[ "${actor_mode}" == "vla_only" ]]; then
    policies=(reference)
  elif [[ "${variant}" == "residual_rlt_actor" || "${variant}" == "flow_rlt_actor" ]]; then
    policies=(reference reference_noise actor)
  elif [[ "${guide}" == "1" ]]; then
    policies=(reference actor actor_guide)
  else
    policies=(reference actor)
  fi

  local episode policy seed
  local -a seeds=()
  for episode in "${EVAL_EPISODES[@]}"; do
    if (( episode == 400 )); then
      seeds=(20260831 20260832 20260833 20260834)
    else
      seeds=(20260831)
    fi
    for policy in "${policies[@]}"; do
      if ! csv_contains "${V14_EVAL_POLICIES}" "${policy}"; then
        continue
      fi
      for seed in "${seeds[@]}"; do
        run_one \
          "${variant}" \
          "${GPU_ARRAY[$gpu_index]}" \
          "${ae_mode}" \
          "${episode}" \
          "${policy}" \
          "${seed}"
      done
    done
  done
}

declare -a EVAL_PIDS=()
declare -a EVAL_NAMES=()
for row in "${VARIANT_ROWS[@]}"; do
  variant="${row%%|*}"
  if ! csv_contains "${V14_EVAL_VARIANTS}" "${variant}"; then
    continue
  fi
  run_variant "${row}" &
  EVAL_PIDS+=("$!")
  EVAL_NAMES+=("${variant}")
done
if (( ${#EVAL_PIDS[@]} == 0 )); then
  echo "[v14-eval] filters selected no variants" >&2
  exit 1
fi

failed=0
for index in "${!EVAL_PIDS[@]}"; do
  if ! wait "${EVAL_PIDS[$index]}"; then
    echo "[v14-eval] variant evaluation failed: ${EVAL_NAMES[$index]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

echo "[v14-eval] all requested immutable-snapshot evaluations are complete"
