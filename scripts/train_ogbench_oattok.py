#!/usr/bin/env python3
"""Train per-domain CRAFT-style OATTok tokenizers for OGBench50.

One independent tokenizer weight set per domain (shared across that domain's
5 singletask goals). Saves best-by-val-RMSE checkpoints for ConsensusDiscreteFlow.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

RQL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RQL_ROOT))

from agents.oattok_jax import (  # noqa: E402
    OATTok,
    create_tokenizer_state,
    save_tokenizer,
    tokenizer_train_step,
)

DOMAINS = [
    # dataset_name, h, sample_dim
    ("scene-play-v0", 5, 5),
    ("puzzle-3x3-play-v0", 5, 5),
    ("puzzle-4x4-play-v0", 5, 5),
    ("cube-double-play-v0", 5, 5),
    ("cube-triple-play-v0", 5, 5),
    ("cube-quadruple-play-v0", 5, 5),
    ("antmaze-large-navigate-v0", 1, 8),
    ("antmaze-giant-navigate-v0", 1, 8),
    ("humanoidmaze-medium-navigate-v0", 1, 21),
    ("humanoidmaze-large-navigate-v0", 1, 21),
]


def domain_id(dataset_name: str) -> str:
    return dataset_name.replace("-play-v0", "").replace("-navigate-v0", "")


def domain_hparams(did: str, sample_dim: int, base_steps: int) -> dict:
    """Per-domain capacity / schedule. Mazes need more capacity."""
    if did.startswith("humanoidmaze"):
        return dict(
            emb_dim=512,
            num_registers=16,
            encoder_depth=3,
            decoder_depth=5,
            steps=max(base_steps, 200_000),
            lr=3e-4,
        )
    if did.startswith("antmaze"):
        return dict(
            emb_dim=384,
            num_registers=16,
            encoder_depth=3,
            decoder_depth=5,
            steps=max(base_steps, 200_000),
            lr=3e-4,
        )
    # manipulation / play
    return dict(
        emb_dim=256,
        num_registers=12,
        encoder_depth=2,
        decoder_depth=4,
        steps=base_steps,
        lr=1e-4,
    )


def load_action_chunks(
    npz_path: Path,
    horizon: int,
    max_chunks: int,
    seed: int = 0,
) -> np.ndarray:
    data = np.load(npz_path)
    actions = data["actions"].astype(np.float32)
    terminals = data["terminals"].astype(np.float32)
    n = len(actions)
    terminal_locs = np.nonzero(terminals > 0)[0]
    if len(terminal_locs) == 0:
        terminal_locs = np.array([n - 1])
    initial_locs = np.concatenate([[0], terminal_locs[:-1] + 1])

    rng = np.random.default_rng(seed)
    chunks = []
    attempts = 0
    while len(chunks) < max_chunks and attempts < max_chunks * 40:
        attempts += 1
        ep = int(rng.integers(0, len(initial_locs)))
        start0, end0 = int(initial_locs[ep]), int(terminal_locs[ep])
        if end0 - start0 + 1 < horizon:
            continue
        start = int(rng.integers(start0, end0 - horizon + 2))
        chunks.append(actions[start : start + horizon])
    if not chunks:
        raise RuntimeError(f"No valid chunks from {npz_path}")
    return np.stack(chunks, axis=0)


def resolve_dataset_path(data_dir: Path, dataset_name: str) -> Path:
    if "puzzle-4x4" in dataset_name:
        alt = os.environ.get("OGBENCH_PUZZLE_4X4_100M_DIR", "")
        if alt:
            p = Path(alt)
            files = sorted(f for f in p.glob("*.npz") if "-val" not in f.name)
            if files:
                return files[0]
    if "cube-quadruple" in dataset_name:
        alt = os.environ.get("OGBENCH_CUBE_QUADRUPLE_100M_DIR", "")
        if alt:
            p = Path(alt)
            files = sorted(f for f in p.glob("*.npz") if "-val" not in f.name)
            if files:
                return files[0]
    path = data_dir / f"{dataset_name}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def eval_rmse(model, params, chunks: np.ndarray, batch_size: int = 512) -> float:
    rmses = []
    for i in range(0, len(chunks), batch_size):
        batch = jnp.asarray(chunks[i : i + batch_size])
        recons, _, _ = model.apply({"params": params}, batch, deterministic=True)
        err = np.asarray(recons) - chunks[i : i + batch_size]
        rmses.append(np.mean(err**2))
    return float(np.sqrt(np.mean(rmses)))


def train_one(
    dataset_name: str,
    horizon: int,
    sample_dim: int,
    data_dir: Path,
    save_dir: Path,
    base_steps: int,
    batch_size: int,
    seed: int,
    max_chunks: int,
    force: bool,
    num_registers: int | None = None,
    emb_dim: int | None = None,
):
    did = domain_id(dataset_name)
    hp = domain_hparams(did, sample_dim, base_steps)
    if num_registers is not None:
        if int(num_registers) < 1:
            raise ValueError(f"num_registers must be >= 1, got {num_registers}")
        hp["num_registers"] = int(num_registers)
    if emb_dim is not None:
        if int(emb_dim) < 1:
            raise ValueError(f"emb_dim must be >= 1, got {emb_dim}")
        hp["emb_dim"] = int(emb_dim)
    # Include K in the filename so ablations can share a save root safely.
    out_path = (
        save_dir / f"{did}_h{horizon}_d{sample_dim}_K{hp['num_registers']}.pkl"
    )
    # Backward-compatible alias used by existing RL launchers (no K suffix).
    legacy_path = save_dir / f"{did}_h{horizon}_d{sample_dim}.pkl"
    if not force:
        if out_path.is_file():
            print(f"[skip] {out_path} already exists (pass --force to overwrite)")
            return out_path
        # Production checkpoints predate the _K suffix; don't silently retrain.
        default_k = int(domain_hparams(did, sample_dim, base_steps)["num_registers"])
        if (
            num_registers is None
            and int(hp["num_registers"]) == default_k
            and legacy_path.is_file()
        ):
            print(f"[skip] {legacy_path} already exists (pass --force to overwrite)")
            return legacy_path

    npz_path = resolve_dataset_path(data_dir, dataset_name)
    print(
        f"=== training tokenizer {did} from {npz_path} "
        f"(steps={hp['steps']} emb={hp['emb_dim']} K={hp['num_registers']}) ==="
    )
    # Larger pool + held-out val for best-checkpoint selection.
    all_chunks = load_action_chunks(
        npz_path, horizon, max_chunks=max_chunks, seed=seed
    )
    assert all_chunks.shape[-1] == sample_dim, (all_chunks.shape, sample_dim)
    n_val = min(8192, max(1024, all_chunks.shape[0] // 20))
    val_chunks = all_chunks[:n_val]
    train_chunks = all_chunks[n_val:]
    print(f"chunks train={train_chunks.shape} val={val_chunks.shape}")

    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)

    # Build model with domain capacity; cosine LR over full schedule.
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=hp["lr"],
        warmup_steps=min(2000, hp["steps"] // 20),
        decay_steps=hp["steps"],
        end_value=hp["lr"] * 0.05,
    )
    model = OATTok(
        sample_dim=sample_dim,
        sample_horizon=horizon,
        num_registers=hp["num_registers"],
        emb_dim=hp["emb_dim"],
        encoder_depth=hp["encoder_depth"],
        decoder_depth=hp["decoder_depth"],
    )
    dummy = jnp.zeros((1, horizon, sample_dim), dtype=jnp.float32)
    params = model.init(init_rng, dummy, deterministic=True)["params"]
    tx = optax.adamw(learning_rate=schedule, weight_decay=1e-4, b1=0.9, b2=0.95)
    from flax.training import train_state

    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )

    n = train_chunks.shape[0]
    best_rmse = float("inf")
    best_params = None
    best_step = -1
    last = {}
    eval_every = 2000

    pbar = tqdm(range(hp["steps"]), desc=did)
    for step in pbar:
        rng, b_rng, s_rng = jax.random.split(rng, 3)
        idx = np.asarray(jax.random.randint(b_rng, (batch_size,), 0, n))
        batch = jnp.asarray(train_chunks[idx])
        state, info = tokenizer_train_step(
            state, batch, s_rng, num_registers=hp["num_registers"]
        )
        last = {k: float(v) for k, v in info.items()}

        if step % eval_every == 0 or step == hp["steps"] - 1:
            v_rmse = eval_rmse(model, state.params, val_chunks)
            last["val_rmse"] = v_rmse
            if v_rmse < best_rmse:
                best_rmse = v_rmse
                best_params = copy.deepcopy(
                    jax.tree_util.tree_map(np.asarray, state.params)
                )
                best_step = step
            pbar.set_postfix(
                {
                    "train_rmse": last.get("tok_rmse"),
                    "val_rmse": v_rmse,
                    "best": best_rmse,
                }
            )
        elif step % 200 == 0:
            pbar.set_postfix(last)

    if best_params is None:
        best_params = jax.tree_util.tree_map(np.asarray, state.params)
        best_rmse = eval_rmse(model, state.params, val_chunks)
        best_step = hp["steps"] - 1

    # Final report on val + a fresh train subsample.
    train_probe = train_chunks[: min(4096, len(train_chunks))]
    train_rmse = eval_rmse(model, best_params, train_probe)
    val_rmse = eval_rmse(model, best_params, val_chunks)

    meta = {
        "domain": did,
        "dataset_name": dataset_name,
        "sample_dim": sample_dim,
        "sample_horizon": horizon,
        "num_registers": hp["num_registers"],
        "emb_dim": hp["emb_dim"],
        "encoder_depth": hp["encoder_depth"],
        "decoder_depth": hp["decoder_depth"],
        "fsq_levels": (8, 5, 5, 5),
        "steps": hp["steps"],
        "best_step": best_step,
        "final_rmse": val_rmse,
        "train_probe_rmse": train_rmse,
        "val_rmse": val_rmse,
        "source_npz": str(npz_path),
        "lr_peak": hp["lr"],
    }
    save_dir.mkdir(parents=True, exist_ok=True)
    save_tokenizer(str(out_path), best_params, meta)
    # Keep legacy filename only when training the domain-default K, so existing
    # RL launchers keep working when save-dir is the production ogbench_oattok.
    default_k = int(domain_hparams(did, sample_dim, base_steps)["num_registers"])
    if int(hp["num_registers"]) == default_k:
        save_tokenizer(str(legacy_path), best_params, meta)
    print(
        f"saved {out_path} best_step={best_step} "
        f"val_rmse={val_rmse:.4f} train_probe_rmse={train_rmse:.4f}"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/workspace-SR008.nfs2/users/staroverov/ogbench/data",
    )
    parser.add_argument(
        "--save-dir",
        default=str(RQL_ROOT / "exp" / "ogbench_oattok"),
    )
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=500000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--num-registers",
        type=int,
        default=None,
        help="Override discrete token count K (num_registers). Default: domain table.",
    )
    parser.add_argument(
        "--emb-dim",
        type=int,
        default=None,
        help="Override embedding width. Default: domain table.",
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Optional domain id filter, e.g. scene antmaze-large",
    )
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    domains = DOMAINS
    if args.domains:
        want = set(args.domains)
        domains = [d for d in DOMAINS if domain_id(d[0]) in want]
        if not domains:
            raise SystemExit(f"No domains matched {args.domains}")

    for dataset_name, horizon, sample_dim in domains:
        train_one(
            dataset_name=dataset_name,
            horizon=horizon,
            sample_dim=sample_dim,
            data_dir=data_dir,
            save_dir=save_dir,
            base_steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            max_chunks=args.max_chunks,
            force=args.force,
            num_registers=args.num_registers,
            emb_dim=args.emb_dim,
        )


if __name__ == "__main__":
    main()
