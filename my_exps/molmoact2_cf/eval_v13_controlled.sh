#!/usr/bin/env bash
# Held-out validation of immutable V13 snapshot bundles. Never launches training.

set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")" && pwd -P)}"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/rlt_cf_v13_controlled}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
VAL_BENCHMARK="${BENCHMARK_ROOT}/val"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
HELPER="${ROOT}/v13_harness.py"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_eval_v13_controlled}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v13_controlled}"
FORCE="${FORCE:-0}"
V13_EVAL_EPISODES="${V13_EVAL_EPISODES:-0,100,200,400}"
V13_EVAL_VARIANTS="${V13_EVAL_VARIANTS:-}"
V13_EVAL_POLICIES="${V13_EVAL_POLICIES:-}"
V13_ALLOW_UNSAFE_AE_FORCE_DEPLOY="${V13_ALLOW_UNSAFE_AE_FORCE_DEPLOY:-0}"

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

if [[ ! "${FORCE}" =~ ^[01]$ ]]; then
  echo "[v13-eval] FORCE must be 0 or 1, got ${FORCE}" >&2
  exit 1
fi
if [[ ! "${V13_ALLOW_UNSAFE_AE_FORCE_DEPLOY}" =~ ^[01]$ ]]; then
  echo "[v13-eval] V13_ALLOW_UNSAFE_AE_FORCE_DEPLOY must be 0 or 1" >&2
  exit 1
fi
IFS=',' read -r -a EVAL_EPISODES <<< "${V13_EVAL_EPISODES}"
if (( ${#EVAL_EPISODES[@]} == 0 )); then
  echo "[v13-eval] V13_EVAL_EPISODES is empty" >&2
  exit 1
fi
for episode in "${EVAL_EPISODES[@]}"; do
  if [[ ! "${episode}" =~ ^[0-9]+$ ]]; then
    echo "[v13-eval] invalid episode milestone: ${episode}" >&2
    exit 1
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "[v13-eval] Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${RUN_DIR}/MANIFEST.json" ]]; then
  echo "[v13-eval] missing launch manifest: ${RUN_DIR}/MANIFEST.json" >&2
  exit 1
fi
if [[ ! -f "${VAL_BENCHMARK}/benchmark.json" ]]; then
  echo "[v13-eval] missing held-out validation benchmark: ${VAL_BENCHMARK}" >&2
  exit 1
fi

echo "[v13-eval] validating controlled benchmark before held-out evaluation"
"${PYTHON}" "${ROOT}/generate_controlled_benchmark.py" \
  --output-root "${BENCHMARK_ROOT}" \
  --validate-only

manifest_gpu_ids="$("${PYTHON}" "${HELPER}" manifest-gpu-ids \
  --manifest "${RUN_DIR}/MANIFEST.json")"
IFS=',' read -r -a GPU_ARRAY <<< "${manifest_gpu_ids}"
if (( ${#GPU_ARRAY[@]} != 8 )); then
  echo "[v13-eval] manifest does not contain eight GPU assignments" >&2
  exit 1
fi

mapfile -t VARIANT_ROWS < <("${PYTHON}" "${HELPER}" variants --format tsv)
if (( ${#VARIANT_ROWS[@]} != 8 )); then
  echo "[v13-eval] expected eight variants, got ${#VARIANT_ROWS[@]}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/validation" "${TMP_ROLLOUT_DIR}" "${RLT_EGL_LOCK_DIR}"
exec 9> "${RUN_DIR}/.eval_v13.lock"
if ! flock -n 9; then
  echo "[v13-eval] another V13 evaluator holds ${RUN_DIR}/.eval_v13.lock" >&2
  exit 1
fi

server_ready() {
  local port="$1"
  local response
  response="$(curl -sf --max-time 3 "http://127.0.0.1:${port}/healthz" 2>/dev/null || true)"
  [[ "${response}" == *'"status":"ok"'* ]]
}

csv_contains() {
  local csv="$1"
  local needle="$2"
  local item
  [[ -n "${csv}" ]] || return 0
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
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
    echo "[v13-eval] failed to generate command for ${variant}" >&2
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
    echo "[v13-eval] skip unavailable immutable snapshot: ${source_dir}"
    return 0
  fi
  if [[ "${ae_mode}" == "1" && ! -f "${source_dir}/molmo_ae_lora.pt" ]]; then
    echo "[v13-eval] missing AE adapter snapshot: ${source_dir}/molmo_ae_lora.pt" >&2
    return 1
  fi

  if [[ "${FORCE}" != "1" ]] && "${PYTHON}" "${HELPER}" evaluation-complete \
    --out-dir "${out_dir}" --expected-episodes 12 >/dev/null 2>&1; then
    echo "[v13-eval] skip complete ${variant}/${snapshot_name}/${policy}/seed_${seed}"
    return 0
  fi

  case "${out_dir}" in
    "${RUN_DIR}/validation/"*) ;;
    *)
      echo "[v13-eval] refusing unsafe validation output path: ${out_dir}" >&2
      return 1
      ;;
  esac
  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  eval_command "${variant}" "${episode}" "${policy}" "${seed}"
  local -a command=("${GENERATED_COMMAND[@]}")
  local -a egl_env=()
  if [[ "${gpu}" =~ ^[0-9]+$ ]]; then
    egl_env=(MUJOCO_EGL_DEVICE_ID="${gpu}")
  fi

  echo "[v13-eval] run ${variant}/${snapshot_name}/${policy}/seed_${seed} gpu=${gpu}"
  env \
    RLT_CF_V4_RUN_DIR="${RUN_DIR}" \
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
    --out-dir "${out_dir}" --expected-episodes 12 >/dev/null 2>&1; then
    echo "[v13-eval] evaluation did not produce a complete summary: ${out_dir}" >&2
    return 1
  fi
}

run_variant() {
  local row="$1"
  local variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port
  IFS='|' read -r variant gpu_index cf_mode actor_mode guide ae_mode checkpoint_kind updates port <<< "${row}"
  if [[ -n "${port}" ]] && ! server_ready "${port}"; then
    echo "[v13-eval] required HTTP server for ${variant} is unavailable on port ${port}" >&2
    return 1
  fi

  local -a POLICIES
  if [[ "${actor_mode}" == "vla_only" ]]; then
    POLICIES=(reference)
  elif [[ "${ae_mode}" == "1" && "${V13_ALLOW_UNSAFE_AE_FORCE_DEPLOY}" != "1" ]]; then
    # V13 AE adapters were trained in raw robot coordinates although Molmo's
    # action expert operates in native normalized coordinates.  Only the
    # adapter-disabled checkpoint gate is a valid legacy AE canary.
    POLICIES=(checkpoint_gate)
  elif [[ "${guide}" == "1" ]]; then
    POLICIES=(checkpoint_gate actor actor_guide)
  else
    POLICIES=(checkpoint_gate actor)
  fi
  if [[ "${variant}" == "residual_rlt_actor" || "${variant}" == "flow_rlt_actor" ]]; then
    POLICIES+=(reference_noise)
  fi

  local episode policy seed
  local -a SEEDS
  for episode in "${EVAL_EPISODES[@]}"; do
    if (( episode == 400 )); then
      SEEDS=(20260831 20260832 20260833 20260834)
    else
      SEEDS=(20260831)
    fi
    for policy in "${POLICIES[@]}"; do
      if ! csv_contains "${V13_EVAL_POLICIES}" "${policy}"; then
        continue
      fi
      for seed in "${SEEDS[@]}"; do
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
  if ! csv_contains "${V13_EVAL_VARIANTS}" "${variant}"; then
    continue
  fi
  run_variant "${row}" &
  EVAL_PIDS+=("$!")
  EVAL_NAMES+=("${variant}")
done

failed=0
for index in "${!EVAL_PIDS[@]}"; do
  if ! wait "${EVAL_PIDS[$index]}"; then
    echo "[v13-eval] variant evaluation failed: ${EVAL_NAMES[$index]}" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 1
fi

echo "[v13-eval] all requested held-out evaluations are complete"
