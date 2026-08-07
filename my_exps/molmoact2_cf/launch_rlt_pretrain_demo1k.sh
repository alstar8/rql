#!/usr/bin/env bash
# Offline RLT pretrain: ~1k MolmoBot + ~1k DROID → token AE + critic.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd -P)"
cd "${ROOT}"

OUT_DIR="${OUT_DIR:-${ROOT}/runs/rlt_pretrain_demo1k}"
LOCAL_LOG="${LOCAL_LOG_DIR:-/workspace-SR008.nfs2/users/staroverov/B1K/tmp/rlt_pretrain_demo1k_logs}"
MOLMOACT2="${ROOT}/../../../molmoact2"
MOLMOSPACES="${ROOT}/../../../molmospaces"
PYTHON_MS="${MOLMOSPACES}/.venv/bin/python"
PYTHON_MA="${MOLMOACT2}/.venv/bin/python"
MOLMOBOT_DIR="${MOLMOBOT_DIR:-${MOLMOSPACES}/mbdata/FrankaPickOmniCamConfig/part0/train}"
DROID_REPO="${DROID_REPO:-IPEC-COMMUNITY/droid_lerobot}"
DROID_ROOT="${DROID_ROOT:-${OUT_DIR}/droid_lerobot_1k}"
NUM_EPISODES="${NUM_EPISODES:-1000}"
MAX_CHUNKS="${MAX_CHUNKS:-16}"
ENCODE_SHARDS="${ENCODE_SHARDS:-4}"
BASE_PORT="${BASE_PORT:-8700}"
TOKEN_STEPS="${TOKEN_STEPS:-8000}"
CRITIC_STEPS="${CRITIC_STEPS:-15000}"
N_CRITICS="${N_CRITICS:-10}"
SERVER_WAIT_ATTEMPTS="${SERVER_WAIT_ATTEMPTS:-240}"

mkdir -p "${OUT_DIR}" "${LOCAL_LOG}" "${OUT_DIR}/pids" "${DROID_ROOT}"
ln -sfn "${LOCAL_LOG}" "${OUT_DIR}/logs"

exec > >(tee -a "${LOCAL_LOG}/launcher.log") 2>&1
echo "[pretrain $(date -Is)] out=${OUT_DIR}"

if [[ ! -x "${PYTHON_MS}" ]]; then
  echo "MolmoSpaces python missing: ${PYTHON_MS}" >&2
  exit 1
fi
if [[ ! -x "${PYTHON_MA}" ]]; then
  echo "MolmoAct2 python missing: ${PYTHON_MA}" >&2
  exit 1
fi

# ---------- 1) Ensure MolmoBot index ----------
if [[ ! -f "${MOLMOBOT_DIR}/valid_trajectory_index.json" ]]; then
  echo "[pretrain] validating MolmoBot trajectories"
  "${PYTHON_MS}" "${MOLMOSPACES}/scripts/data/validate_trajectories.py" "${MOLMOBOT_DIR}"
fi

# ---------- 2) Warm HF meta for IPEC DROID ----------
echo "[pretrain] warming IPEC DROID meta cache"
NUM_EPISODES="${NUM_EPISODES}" "${PYTHON_MS}" - <<'PY'
import os
from droid_ipec_loader import IPECDroidEpisodes
n = int(os.environ["NUM_EPISODES"])
ds = IPECDroidEpisodes(list(range(n)))
print("droid episodes indexed", len(ds), "tasks", len(ds._tasks))
PY

# ---------- 3) Start encode servers (one per shard GPU) ----------
declare -a SERVER_PIDS=()
start_server() {
  local shard="$1"
  local gpu="$1"
  local port=$((BASE_PORT + shard))
  local logfile="${LOCAL_LOG}/encode_server_s${shard}_gpu${gpu}.log"
  local pidfile="${OUT_DIR}/pids/encode_server_s${shard}.pid"
  echo "[pretrain] encode server shard=${shard} gpu=${gpu} port=${port}"
  (
    cd "${MOLMOACT2}"
    exec setsid env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON_MA}" "${ROOT}/serve.py" \
        --host 0.0.0.0 --port "${port}" --device cuda:0 --dtype bfloat16 \
        --disable_g --feature_mode tokens
  ) > "${logfile}" 2>&1 &
  local pid=$!
  echo "${pid} encode_server shard=${shard} gpu=${gpu} port=${port}" > "${pidfile}"
  SERVER_PIDS+=("${pid}")
}

for ((s=0; s<ENCODE_SHARDS; s++)); do
  start_server "${s}"
  sleep 2
done

echo "[pretrain] waiting for encode servers"
for ((s=0; s<ENCODE_SHARDS; s++)); do
  port=$((BASE_PORT + s))
  ready=0
  for ((a=1; a<=SERVER_WAIT_ATTEMPTS; a++)); do
    if curl -sf --max-time 3 "http://127.0.0.1:${port}/act" | grep -q '"status":"ok"'; then
      ready=1
      break
    fi
    sleep 5
  done
  if (( ready == 0 )); then
    echo "[pretrain] ERROR server port ${port} not ready" >&2
    exit 1
  fi
  echo "[pretrain] server port=${port} READY"
done

# ---------- 4) Encode MolmoBot then DROID (sharded) ----------
declare -a ENC_PIDS=()
encode_one() {
  local source="$1"
  local shard="$2"
  local port=$((BASE_PORT + shard))
  local logfile="${LOCAL_LOG}/encode_${source}_s${shard}.log"
  local pidfile="${OUT_DIR}/pids/encode_${source}_s${shard}.pid"
  echo "[pretrain] encode source=${source} shard=${shard}/${ENCODE_SHARDS} port=${port}"
  setsid env \
    "${PYTHON_MS}" "${ROOT}/encode_offline_demo_tokens.py" \
      --source "${source}" \
      --out_dir "${OUT_DIR}" \
      --server_host 127.0.0.1 \
      --server_port "${port}" \
      --num_episodes "${NUM_EPISODES}" \
      --max_chunks "${MAX_CHUNKS}" \
      --shard_id "${shard}" \
      --num_shards "${ENCODE_SHARDS}" \
      --molmobot_dir "${MOLMOBOT_DIR}" \
      --droid_repo "${DROID_REPO}" \
      --droid_root "" \
      --seed 0 \
    > "${logfile}" 2>&1 &
  local pid=$!
  echo "${pid} encode ${source} shard=${shard}" > "${pidfile}"
  ENC_PIDS+=("${pid}")
}

wait_encoders() {
  local status_local=0
  for pid in "${ENC_PIDS[@]}"; do
    if ! wait "${pid}"; then
      echo "[pretrain] encode pid ${pid} failed" >&2
      status_local=1
    fi
  done
  ENC_PIDS=()
  return "${status_local}"
}

for ((s=0; s<ENCODE_SHARDS; s++)); do
  encode_one molmobot "${s}"
done
if ! wait_encoders; then
  echo "[pretrain] molmobot encode failed; stopping servers" >&2
  for pid in "${SERVER_PIDS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  exit 1
fi

for ((s=0; s<ENCODE_SHARDS; s++)); do
  encode_one droid "${s}"
done
if ! wait_encoders; then
  echo "[pretrain] droid encode failed; stopping servers" >&2
  for pid in "${SERVER_PIDS[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  exit 1
fi

echo "[pretrain] stopping encode servers"
for pid in "${SERVER_PIDS[@]}"; do
  kill -TERM "${pid}" 2>/dev/null || true
done
sleep 5
for pid in "${SERVER_PIDS[@]}"; do
  kill -KILL "${pid}" 2>/dev/null || true
done

# ---------- 5) Merge replays ----------
echo "[pretrain] merging token/chunk replays"
cd "${ROOT}"
"${PYTHON_MS}" - <<PY
from pathlib import Path
from chunk_replay import TokenReplay, ChunkReplay
out = Path("${OUT_DIR}")

def merge_tokens(pattern, dest):
    paths = sorted(out.glob(pattern))
    if not paths:
        raise SystemExit(f"no files for {pattern}")
    merged = TokenReplay()
    for p in paths:
        part = TokenReplay.load_npz(str(p))
        for t, m in zip(part.tokens, part.masks):
            merged.add(t, m)
        print(f"loaded {p.name} n={len(part)} total={len(merged)}")
    merged.save_npz(str(dest))
    print(f"wrote {dest} n={len(merged)}")

def merge_chunks(pattern, dest):
    paths = sorted(out.glob(pattern))
    if not paths:
        raise SystemExit(f"no files for {pattern}")
    merged = ChunkReplay(max_transitions=1_000_000)
    for p in paths:
        part = ChunkReplay.load_npz(str(p))
        for row in part.rows:
            merged.add(row)
        merged.n_episodes += part.n_episodes
        print(f"loaded {p.name} n={len(part)} total={len(merged)}")
    merged.save_npz(str(dest))
    print(f"wrote {dest} n={len(merged)} eps={merged.n_episodes}")

merge_tokens("token_replay_*.npz", out / "token_replay_merged.npz")
merge_tokens("chunk_token_replay_*.npz", out / "chunk_token_replay_merged.npz")
merge_chunks("chunk_replay_*.npz", out / "chunk_replay_merged.npz")
PY

# ---------- 6) Token AE warmup ----------
echo "[pretrain] token AE warmup steps=${TOKEN_STEPS}"
"${PYTHON_MS}" "${ROOT}/warmup_rlt_token.py" \
  --token_replay "${OUT_DIR}/token_replay_merged.npz" \
  --out_ckpt "${OUT_DIR}/rlt_token_demo1k.pt" \
  --device cuda:0 \
  --steps "${TOKEN_STEPS}" \
  --batch_size 4 \
  --n_critics "${N_CRITICS}" \
  --use_cf_guide \
  --log_every 100

# ---------- 7) Critic warmup ----------
echo "[pretrain] critic warmup steps=${CRITIC_STEPS}"
# Stream token shards (not the 45GB merged NPZ) to avoid OOM on re-encode.
"${PYTHON_MS}" -u "${ROOT}/warmup_rlt_critic.py" \
  --rlt_ckpt "${OUT_DIR}/rlt_token_demo1k.pt" \
  --chunk_replay "${OUT_DIR}/chunk_replay_merged.npz" \
  --chunk_token_glob "${OUT_DIR}" \
  --out_ckpt "${OUT_DIR}/rlt_cf_pretrain_demo1k.pt" \
  --device cuda:0 \
  --steps "${CRITIC_STEPS}" \
  --batch_size 64 \
  --encode_batch_size 8 \
  --log_every 100

cat > "${OUT_DIR}/README.md" <<EOF
# RLT offline pretrain (demo 1k + 1k)

- MolmoBot: ${NUM_EPISODES} trajs from FrankaPickOmniCamConfig
- DROID: ${NUM_EPISODES} episodes from ${DROID_REPO}
- Token AE steps: ${TOKEN_STEPS}
- Critic steps: ${CRITIC_STEPS}
- Checkpoint: \`rlt_cf_pretrain_demo1k.pt\`

Use with online CF:
\`\`\`bash
RLT_CKPT=${OUT_DIR}/rlt_cf_pretrain_demo1k.pt bash launch_rlt_v7.sh
\`\`\`
EOF

echo "[pretrain $(date -Is)] DONE ckpt=${OUT_DIR}/rlt_cf_pretrain_demo1k.pt"
