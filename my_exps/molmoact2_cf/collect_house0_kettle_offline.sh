#!/usr/bin/env bash
# Collect house0 kettle VLA reference rollouts with token export for V17 offline.
# Uses frozen MolmoAct2 + residual RLT scaffold (reference-only, 0 updates).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

B1K_ROOT="${B1K_ROOT:-/workspace-SR008.nfs2/users/staroverov/B1K}"
B1K_TMP="${B1K_TMP:-${B1K_ROOT}/tmp}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/rlt_pretrain_house0_kettle}"
LOCAL_LOG="${LOCAL_LOG_DIR:-${B1K_TMP}/rlt_pretrain_house0_kettle_collect_logs}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${ROOT}/runs/benchmarks/house0_kettle_v13}"
# Any residual ckpt works for collect; z is re-encoded after kettle AE warmup.
INIT_CKPT="${INIT_CKPT:-${ROOT}/runs/rlt_pretrain_demo1k_z256_d512_l4/rlt_cf_pretrain_demo1k_z256_d512_l4.pt}"
if [[ ! -f "${INIT_CKPT}" ]]; then
  INIT_CKPT="${ROOT}/runs/rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt"
fi

MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON="${PYTHON:-${MOLMOSPACES}/.venv/bin/python}"
SERVE_PYTHON="${SERVE_PYTHON:-${MOLMOACT2}/.venv/bin/python}"

NUM_SHARDS="${NUM_SHARDS:-8}"
EPISODES_PER_SHARD="${EPISODES_PER_SHARD:-150}"
BASE_PORT="${BASE_PORT:-8740}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"
RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR:-${B1K_TMP}/rlt_egl_locks_v17_collect}"
TMP_ROLLOUT_DIR="${TMP_ROLLOUT_DIR:-${B1K_TMP}/molmoact2_rlt_rollouts_v17_collect}"

mkdir -p "${OUT_DIR}/pids" "${LOCAL_LOG}" "${RLT_EGL_LOCK_DIR}" "${TMP_ROLLOUT_DIR}"
ln -sfn "${LOCAL_LOG}" "${OUT_DIR}/logs"

exec > >(tee -a "${LOCAL_LOG}/collect_launcher.log") 2>&1
echo "[v17-collect $(date -Is)] out=${OUT_DIR} eps/shard=${EPISODES_PER_SHARD} shards=${NUM_SHARDS}"

if [[ ! -f "${INIT_CKPT}" ]]; then
  echo "missing init ckpt: ${INIT_CKPT}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON}" || ! -x "${SERVE_PYTHON}" ]]; then
  echo "python missing" >&2
  exit 1
fi

if [[ -n "${GPU_IDS:-}" ]]; then
  IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
else
  GPU_ARRAY=(0 1 2 3 4 5 6 7)
fi
if (( ${#GPU_ARRAY[@]} < NUM_SHARDS )); then
  echo "need ${NUM_SHARDS} GPUs, got ${#GPU_ARRAY[@]}" >&2
  exit 1
fi

server_ready() {
  local port="$1" response
  response="$(curl -sf --max-time 3 "http://127.0.0.1:${port}/healthz" 2>/dev/null)" || return 1
  [[ "${response}" == *'"status":"ok"'* ]]
}

wait_for_server() {
  local port="$1" attempt
  for ((attempt=1; attempt<=SERVER_WAIT_ATTEMPTS; attempt++)); do
    server_ready "${port}" && return 0
    sleep 5
  done
  echo "server port ${port} not ready" >&2
  return 1
}

declare -a SERVER_PIDS=() TRAIN_PIDS=()

cleanup() {
  local pid
  for pid in "${TRAIN_PIDS[@]:-}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  for pid in "${SERVER_PIDS[@]:-}"; do kill -TERM "${pid}" 2>/dev/null || true; done
}
trap cleanup EXIT

for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPU_ARRAY[$shard]}"
  port=$((BASE_PORT + shard))
  slog="${LOCAL_LOG}/server_s${shard}_gpu${gpu}.log"
  echo "[v17-collect] server s${shard} gpu=${gpu} port=${port}"
  (
    cd "${MOLMOACT2}"
    exec setsid env CUDA_VISIBLE_DEVICES="${gpu}" HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
      "${SERVE_PYTHON}" "${ROOT}/serve.py" \
        --host 0.0.0.0 --port "${port}" --device cuda:0 --dtype bfloat16 \
        --disable_g --feature_mode tokens
  ) >> "${slog}" 2>&1 &
  SERVER_PIDS+=("$!")
  echo "$! server shard=${shard} gpu=${gpu} port=${port}" > "${OUT_DIR}/pids/server_s${shard}.pid"
  sleep 2
done

for ((shard=0; shard<NUM_SHARDS; shard++)); do
  wait_for_server $((BASE_PORT + shard))
done

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export RLT_EGL_LOCK_DIR RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT:-16}"
export RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU:-2}"
export RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC:-0.5}"
export RLT_VLA_PREFETCH="${RLT_VLA_PREFETCH:-1}"
export RLT_TOKEN_SAVE_STAGING="${RLT_TOKEN_SAVE_STAGING:-/tmp/rlt_token_stage_v17}"
mkdir -p "${RLT_TOKEN_SAVE_STAGING}"

for ((shard=0; shard<NUM_SHARDS; shard++)); do
  gpu="${GPU_ARRAY[$shard]}"
  port=$((BASE_PORT + shard))
  shard_dir="${OUT_DIR}/shard_${shard}"
  mkdir -p "${shard_dir}"
  tlog="${LOCAL_LOG}/train_s${shard}_gpu${gpu}.log"
  echo "[v17-collect] train s${shard} gpu=${gpu} port=${port}"
  (
    exec setsid env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
      MUJOCO_GL="${MUJOCO_GL}" \
      PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM}" \
      RLT_EGL_LOCK_DIR="${RLT_EGL_LOCK_DIR}" \
      RLT_EGL_MAX_CONCURRENT="${RLT_EGL_MAX_CONCURRENT}" \
      RLT_EGL_PER_GPU="${RLT_EGL_PER_GPU}" \
      RLT_EGL_COOLDOWN_SEC="${RLT_EGL_COOLDOWN_SEC}" \
      RLT_VLA_PREFETCH="${RLT_VLA_PREFETCH}" \
      RLT_TOKEN_SAVE_STAGING="${RLT_TOKEN_SAVE_STAGING}" \
      "${PYTHON}" -u "${ROOT}/train_rlt_online.py" \
        --config_name "house0_kettle_collect_s${shard}" \
        --rlt_ckpt "${INIT_CKPT}" \
        --benchmark_dir "${BENCHMARK_ROOT}/train" \
        --benchmark_pose_cycle 24 \
        --out_dir "${shard_dir}" \
        --replay_out "${shard_dir}/chunk_replay.npz" \
        --export_offline_tokens \
        --chunk_token_replay_out "${shard_dir}/chunk_token_replay.npz" \
        --token_ckpt_every_episodes "${TOKEN_CKPT_EVERY:-25}" \
        --server_host 127.0.0.1 \
        --server_port "${port}" \
        --device cuda:0 \
        --actor_mode vla_only \
        --cf_mode residual \
        --no_cf_guide \
        --updates_per_episode 0 \
        --max_updates_per_episode 0 \
        --max_valid_episodes "${EPISODES_PER_SHARD}" \
        --target_env_steps 1000000 \
        --horizon 400 \
        --seed $((20260815 + shard * 17)) \
        --tmp_rollout_dir "${TMP_ROLLOUT_DIR}/s${shard}" \
        --no_resume
  ) >> "${tlog}" 2>&1 &
  TRAIN_PIDS+=("$!")
  echo "$! train shard=${shard} gpu=${gpu}" > "${OUT_DIR}/pids/train_s${shard}.pid"
  sleep 5
done

echo "[v17-collect] waiting for ${#TRAIN_PIDS[@]} collectors"
fail=0
for pid in "${TRAIN_PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[v17-collect] collector pid=${pid} failed" >&2
    fail=1
  fi
done
if (( fail != 0 )); then
  exit 1
fi

echo "[v17-collect] merging shard NPZs"
"${PYTHON}" - <<PY
from pathlib import Path
from chunk_replay import ChunkReplay, TokenReplay

out = Path("${OUT_DIR}")
shards = sorted(out.glob("shard_*"))
if not shards:
    raise SystemExit("no shards")

def merge_tokens(name: str, dest: Path) -> None:
    merged = None
    for shard in shards:
        path = shard / name
        if not path.is_file():
            raise FileNotFoundError(path)
        buf = TokenReplay.load_npz(str(path))
        if merged is None:
            merged = buf
        else:
            for tokens, mask in zip(buf.tokens, buf.masks):
                merged.add(tokens, mask)
    assert merged is not None
    merged.save_npz(str(dest))
    print(f"wrote {dest} n={len(merged)}")

def merge_chunks(dest: Path) -> None:
    merged = None
    ep_offset = 0
    for shard in shards:
        path = shard / "chunk_replay.npz"
        buf = ChunkReplay.load_npz(str(path))
        if merged is None:
            merged = ChunkReplay(
                max_transitions=max(buf.max_transitions, 500_000),
                chunk_size=buf.chunk_size,
                action_dim=buf.action_dim,
                z_dim=buf.z_dim,
                pos_frac=buf.pos_frac,
                benchmark_pose_cycle=buf.benchmark_pose_cycle,
            )
        for row in buf.rows:
            row.episode_id = int(row.episode_id) + ep_offset
            merged.rows.append(row)
        ep_offset += int(buf.n_episodes)
        merged.n_episodes = ep_offset
    assert merged is not None
    merged.save_npz(str(dest))
    succ = sum(float(r.success) for r in merged.rows) / max(1, len(merged.rows))
    print(f"wrote {dest} n={len(merged)} eps={merged.n_episodes} chunk_succ={succ:.3f}")

merge_tokens("chunk_token_replay.npz", out / "chunk_token_replay_merged.npz")
# AE warmup expects token_replay_merged; same sequences as chunk tokens.
token_merged = out / "token_replay_merged.npz"
chunk_merged = out / "chunk_token_replay_merged.npz"
if token_merged.exists() or token_merged.is_symlink():
    token_merged.unlink()
token_merged.symlink_to(chunk_merged.resolve())
print(f"symlinked {token_merged} -> {chunk_merged}")
merge_chunks(out / "chunk_replay_merged.npz")
# Keep shard token files discoverable for --chunk_token_glob.
for shard in shards:
    src = shard / "chunk_token_replay.npz"
    dst = out / f"chunk_token_replay_kettle_{shard.name}.npz"
    if not dst.exists():
        dst.symlink_to(src.resolve())
print("merge done")
PY

cat > "${OUT_DIR}/README.md" <<EOF
# House0 kettle offline collect (V17)

- Benchmark: \`${BENCHMARK_ROOT}/train\` (\`pick up the kettle.\`)
- Shards: ${NUM_SHARDS} × ${EPISODES_PER_SHARD} reference VLA episodes
- Artifacts: token/chunk NPZs under this directory
EOF

echo "[v17-collect $(date -Is)] DONE -> ${OUT_DIR}"
trap - EXIT
cleanup
