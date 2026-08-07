"""Collect frozen-VLA MolmoSpaces episodes for RLT token/chunk replay."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MUJOCO_GL", "egl")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from chunk_replay import ChunkReplay, TokenReplay  # noqa: E402
from rlt_models import ACTION_DIM, CHUNK_SIZE, Z_DIM, MolmoAct2RLTCF  # noqa: E402
from train_rlt_online import (  # noqa: E402
    _bench_size,
    _build_eval_policy,
    _default_bench,
    _server_health,
    _validate_server_features,
    _wait_for_server,
)
from molmo_spaces.evaluation.configs.evaluation_configs import (  # noqa: E402
    MolmoAct2PolicyEvalConfig,
)
from molmo_spaces.evaluation.eval_main import run_evaluation  # noqa: E402

log = logging.getLogger("molmoact2_cf.collect_token_replay")


def _save_replays(
    token_replay: TokenReplay,
    chunk_replay: ChunkReplay,
    token_path: Path,
    chunk_path: Path,
) -> None:
    token_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    token_tmp = token_path.with_name(f".{token_path.name}.tmp.npz")
    chunk_tmp = chunk_path.with_name(f".{chunk_path.name}.tmp.npz")
    token_replay.save_npz(str(token_tmp))
    chunk_replay.save_npz(str(chunk_tmp))
    os.replace(token_tmp, token_path)
    os.replace(chunk_tmp, chunk_path)


def collect(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.assets_dir:
        os.environ["MLSPACES_ASSETS_DIR"] = args.assets_dir

    health = _wait_for_server(
        args.server_host,
        args.server_port,
        args.server_wait_sec,
    )
    _validate_server_features(health)
    if health.get("enable_g"):
        raise RuntimeError(
            "Token replay collection requires a frozen G=0 server; "
            "restart serve.py with --disable_g"
        )

    device = torch.device(args.device)
    # The collector intentionally records z from a local random encoder. Raw
    # tokens are the durable artifact and can later be re-encoded after warmup.
    model = MolmoAct2RLTCF(
        use_cf_guide=False,
        tune_token_online=False,
    ).to(device)
    model.freeze_token_encoder()
    args.actor_mode = "vla_only"
    args.use_cf_guide = False
    policy, _exp_config = _build_eval_policy(
        args,
        model,
        device,
        prefer_server_z=False,
        retain_tokens=True,
    )
    policy.enable_rlt = False

    token_replay = TokenReplay(
        max_seq=args.token_max_seq,
        token_dim=model.feature_dim,
    )
    chunk_replay = ChunkReplay(
        max_transitions=args.replay_capacity,
        chunk_size=CHUNK_SIZE,
        action_dim=ACTION_DIM,
        z_dim=Z_DIM,
        pos_frac=args.pos_frac,
        seed=args.seed,
    )
    token_path = Path(args.token_replay_out)
    chunk_path = Path(args.chunk_replay_out)
    tmp_rollouts = (
        Path(args.tmp_rollout_dir)
        / f"rlt_token_collect_port{args.server_port}_pid{os.getpid()}"
    )
    tmp_rollouts.mkdir(parents=True, exist_ok=True)

    bench = Path(args.benchmark_dir) if args.benchmark_dir else _default_bench()
    n_bench = _bench_size(bench)
    shard_size = args.shard_size if args.shard_size > 0 else n_bench
    shard_start = args.start_episode % n_bench
    valid_episodes = 0
    skipped_episodes = 0
    successes = 0
    cycle = 0
    start = time.time()

    while valid_episodes < args.num_episodes:
        if args.max_attempts > 0 and cycle >= args.max_attempts:
            raise RuntimeError(
                f"Reached --max_attempts={args.max_attempts} with only "
                f"{valid_episodes}/{args.num_episodes} valid episodes"
            )
        if _server_health(args.server_host, args.server_port) is None:
            health = _wait_for_server(
                args.server_host,
                args.server_port,
                args.server_wait_sec,
            )
            _validate_server_features(health)
            if health.get("enable_g"):
                raise RuntimeError("Restarted server has G enabled; refusing collection")
            policy.prepare_model()

        episode_idx = shard_start + (cycle % shard_size)
        cycle += 1
        episode_dir = tmp_rollouts / f"ep_{cycle:06d}"
        shutil.rmtree(episode_dir, ignore_errors=True)
        episode_dir.mkdir(parents=True, exist_ok=True)
        rollout_ok = True
        try:
            results = run_evaluation(
                eval_config_cls=MolmoAct2PolicyEvalConfig,
                benchmark_dir=bench,
                task_horizon_steps=args.horizon,
                num_workers=1,
                use_wandb=False,
                preloaded_policy=policy,
                episode_idx=episode_idx,
                output_dir=episode_dir,
            )
            success = bool(results.success_count > 0)
            rollout_ok = bool(results.total_count > 0)
        except Exception as error:  # noqa: BLE001
            log.warning("Episode %d failed: %s", episode_idx, error)
            success = False
            rollout_ok = False
        trajectory = policy.pop_episode(success)
        shutil.rmtree(episode_dir, ignore_errors=True)
        if policy.fatal_error is not None:
            raise policy.fatal_error
        if not rollout_ok or trajectory["n_steps"] <= 0:
            skipped_episodes += 1
            log.warning(
                "Skipping invalid episode idx=%d steps=%d skipped=%d",
                episode_idx,
                trajectory["n_steps"],
                skipped_episodes,
            )
            continue
        if len(trajectory["token_batches"]) != len(trajectory["zs"]):
            raise RuntimeError(
                "The server did not return token_features at every chunk boundary. "
                "Start serve.py with --feature_mode tokens."
            )

        chunk_replay.add_episode_chunks(
            trajectory["zs"],
            trajectory["proprios"],
            trajectory["references"],
            trajectory["executed"],
            trajectory["rewards"],
            trajectory["masks"],
            success=success,
            gamma=args.gamma,
            episode_id=valid_episodes,
        )
        for tokens, mask in trajectory["token_batches"]:
            token_replay.add(tokens, mask)

        valid_episodes += 1
        successes += int(success)
        log.info(
            "episode=%d/%d idx=%d success=%s steps=%d token_sequences=%d chunks=%d sr=%.3f",
            valid_episodes,
            args.num_episodes,
            episode_idx,
            success,
            trajectory["n_steps"],
            len(token_replay),
            len(chunk_replay),
            successes / valid_episodes,
        )
        if (
            valid_episodes % args.save_every_episodes == 0
            or valid_episodes == args.num_episodes
        ):
            _save_replays(token_replay, chunk_replay, token_path, chunk_path)

    summary = {
        "valid_episodes": valid_episodes,
        "skipped_episodes": skipped_episodes,
        "successes": successes,
        "success_rate": successes / max(valid_episodes, 1),
        "token_sequences": len(token_replay),
        "chunk_transitions": len(chunk_replay),
        "token_replay": str(token_path),
        "chunk_replay": str(chunk_path),
        "encoder": "random_local_init",
        "elapsed_sec": time.time() - start,
    }
    summary_path = token_path.parent / "collect_token_replay_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(tmp_rollouts, ignore_errors=True)
    log.info("Done: %s", json.dumps(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server_host", type=str, default="localhost")
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--server_wait_sec", type=float, default=1800.0)
    parser.add_argument("--server_request_timeout_sec", type=float, default=120.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--benchmark_dir", type=str, default="")
    parser.add_argument(
        "--assets_dir",
        type=str,
        default=os.path.expanduser("~/.cache/molmospaces/assets"),
    )
    parser.add_argument(
        "--tmp_rollout_dir",
        type=str,
        default="/workspace-SR008.nfs2/users/staroverov/B1K/tmp/molmoact2_rlt_rollouts",
    )
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_attempts", type=int, default=0)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--shard_size", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--pos_frac", type=float, default=0.4)
    parser.add_argument("--replay_capacity", type=int, default=50_000)
    parser.add_argument("--token_max_seq", type=int, default=512)
    parser.add_argument("--save_every_episodes", type=int, default=10)
    parser.add_argument(
        "--token_replay_out",
        type=str,
        default=str(_HERE / "runs/rlt_token_collect/token_replay.npz"),
    )
    parser.add_argument(
        "--chunk_replay_out",
        type=str,
        default=str(_HERE / "runs/rlt_token_collect/chunk_replay.npz"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_episodes <= 0:
        parser.error("--num_episodes must be positive")
    if args.save_every_episodes <= 0:
        parser.error("--save_every_episodes must be positive")
    if args.replay_capacity <= 0:
        parser.error("--replay_capacity must be positive")
    return args


if __name__ == "__main__":
    collect(parse_args())
