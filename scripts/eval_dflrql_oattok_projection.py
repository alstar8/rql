#!/usr/bin/env python3
"""Paired eval: continuous DFLRQL9 vs post-hoc OATTok projection filter.

This is a **discrete-latent deployment filter** over a continuous flow actor:
  raw action  = DFLRQL9.sample_actions(...)
  projected   = clip(decode(FSQ(encode(raw))))
It is **not** a learned discrete policy (unlike DiscreteARIQL / CDF).

Examples:
  python scripts/eval_dflrql_oattok_projection.py \\
    --run-dir exp/rql/humanoidmaze-large-dflrql9-400k-ckpt/sd000_... \\
    --tokenizer exp/ogbench_oattok/humanoidmaze-large_h1_d21.pkl \\
    --num-episodes 50 --conditions raw projected

  python scripts/eval_dflrql_oattok_projection.py \\
    --checkpoint .../params_400000.pkl \\
    --run-dir .../sd000_... \\
    --tokenizer .../humanoidmaze-large_h1_d21.pkl
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.dflrql9 import DFLRQL9Agent, get_config
from agents.oattok_jax import OATTok, load_tokenizer
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import flatten
from utils.flax_utils import restore_agent

CONDITION_RAW = "raw"
CONDITION_PROJECTED = "projected"
VALID_CONDITIONS = (CONDITION_RAW, CONDITION_PROJECTED)


def latest_params_pkl(run_dir: Path) -> Path:
    cands = sorted(run_dir.glob("params_*.pkl"))
    if not cands:
        raise FileNotFoundError(f"No params_*.pkl under {run_dir}")
    return cands[-1]


def load_flags(run_dir: Path) -> dict:
    path = run_dir / "flags.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing flags.json in {run_dir}")
    return json.loads(path.read_text())


def config_from_flags(flags: dict | None) -> dict:
    """Rebuild DFLRQL9 config from a run's flags.json agent block."""
    cfg = dict(get_config())
    if not flags:
        return cfg
    agent_flags = flags.get("agent") or {}
    for key, value in agent_flags.items():
        if key == "agent_name":
            continue
        cfg[key] = value
    return cfg


def episode_env_seed(base_eval_seed: int, episode_idx: int) -> int:
    """Stable per-episode env seed shared across paired conditions."""
    return int(base_eval_seed) * 1_000_003 + int(episode_idx)


def actor_key_for_episode(base_eval_seed: int, episode_idx: int) -> jax.Array:
    """Initial actor PRNG for an episode (same for raw and projected)."""
    return jax.random.PRNGKey(episode_env_seed(base_eval_seed, episode_idx) ^ 0xA11CE)


def supply_rng_from_key(f, rng: jax.Array):
    """Like utils.evaluation.supply_rng but with an explicit starting key."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def assert_tokenizer_matches_policy(
    tok_meta: dict, policy_h: int, prim_action_dim: int
) -> None:
    sample_h = int(tok_meta["sample_horizon"])
    sample_dim = int(tok_meta["sample_dim"])
    if sample_h != int(policy_h):
        raise ValueError(
            f"Tokenizer sample_horizon={sample_h} does not match policy h={policy_h}. "
            f"Use the matching OATTok checkpoint (e.g. humanoidmaze-large_h1_d21.pkl)."
        )
    if sample_dim != int(prim_action_dim):
        raise ValueError(
            f"Tokenizer sample_dim={sample_dim} does not match policy "
            f"primitive action dim={prim_action_dim}."
        )


def build_oattok(tok_params: Any, tok_meta: dict) -> OATTok:
    return OATTok(
        sample_dim=int(tok_meta["sample_dim"]),
        sample_horizon=int(tok_meta["sample_horizon"]),
        num_registers=int(tok_meta["num_registers"]),
        emb_dim=int(tok_meta.get("emb_dim", 256)),
        encoder_depth=int(tok_meta.get("encoder_depth", 2)),
        decoder_depth=int(tok_meta.get("decoder_depth", 4)),
    )


class OATTokActionProjector:
    """Frozen encode/FSQ/decode round-trip as a discrete-latent deployment filter.

    Accepts / returns ActionChunkingWrapper shapes: (h, d). Internally applies
    OATTok on (B, h, d) with B=1.
    """

    def __init__(self, tokenizer: OATTok, tok_params: Any, clip_eps: float = 0.0):
        self.tokenizer = tokenizer
        self.tok_params = tok_params
        self.clip_eps = float(clip_eps)
        self._apply = jax.jit(self._project_btd)

    def _project_btd(self, actions_btd: jnp.ndarray) -> jnp.ndarray:
        # encode -> FSQ quant codes -> decode (same as OATTok.__call__)
        recons, _tokens, _quant = self.tokenizer.apply(
            {"params": self.tok_params},
            actions_btd,
            deterministic=True,
        )
        lo = -1.0 + self.clip_eps
        hi = 1.0 - self.clip_eps
        return jnp.clip(recons, lo, hi)

    def __call__(self, action_hd: np.ndarray | jnp.ndarray) -> np.ndarray:
        """Project (h, d) -> (h, d)."""
        arr = np.asarray(action_hd, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"Expected action shape (h, d) for ActionChunkingWrapper, got {arr.shape}"
            )
        btd = jnp.asarray(arr[None, ...])  # (1, h, d)
        out = self._apply(btd)
        return np.asarray(out[0], dtype=np.float32)


def clip_action_hd(
    action_hd: np.ndarray, clip_eps: float = 0.0
) -> np.ndarray:
    lo = -1.0 + float(clip_eps)
    hi = 1.0 - float(clip_eps)
    return np.clip(np.asarray(action_hd, dtype=np.float32), lo, hi)


def action_diff_metrics(
    raw_hd: np.ndarray, proj_hd: np.ndarray, sat_eps: float = 1e-5
) -> dict[str, float]:
    raw = np.asarray(raw_hd, dtype=np.float32)
    proj = np.asarray(proj_hd, dtype=np.float32)
    delta = proj - raw
    abs_delta = np.abs(delta)
    return {
        "proj_rmse": float(np.sqrt(np.mean(delta**2))),
        "proj_mae": float(np.mean(abs_delta)),
        "proj_max_abs": float(np.max(abs_delta)),
        "raw_sat_frac": float(np.mean(np.abs(raw) >= 1.0 - sat_eps)),
        "proj_sat_frac": float(np.mean(np.abs(proj) >= 1.0 - sat_eps)),
    }


def extract_episode_metrics(info: dict) -> dict[str, float]:
    flat = flatten(info)
    out: dict[str, float] = {}
    for key in (
        "success",
        "xy",
        "episode.return",
        "episode.length",
        "episode.final_reward",
    ):
        if key in flat:
            val = flat[key]
            out[key] = float(np.mean(val)) if np.ndim(val) else float(val)
    # Prefer explicit final xy if present as array.
    if "xy" in flat:
        xy = np.asarray(flat["xy"], dtype=np.float32).reshape(-1)
        if xy.size >= 2:
            out["final_xy_x"] = float(xy[0])
            out["final_xy_y"] = float(xy[1])
            out["final_xy_mean"] = float(np.mean(xy[:2]))
    return out


def run_episode(
    env,
    actor_fn: Callable,
    *,
    env_seed: int,
    eval_temperature: float,
    project_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    clip_eps: float = 0.0,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Roll out one episode; return end-of-episode metrics + per-step proj stats."""
    observation, info = env.reset(seed=int(env_seed))
    done = False
    step_metrics: list[dict[str, float]] = []
    while not done:
        raw = actor_fn(obs=observation, temperature=eval_temperature)
        raw = clip_action_hd(np.asarray(raw), clip_eps=clip_eps)
        if project_fn is None:
            action = raw
            step_metrics.append(
                {
                    "proj_rmse": 0.0,
                    "proj_mae": 0.0,
                    "proj_max_abs": 0.0,
                    "raw_sat_frac": float(np.mean(np.abs(raw) >= 1.0 - 1e-5)),
                    "proj_sat_frac": float(np.mean(np.abs(raw) >= 1.0 - 1e-5)),
                }
            )
        else:
            projected = project_fn(raw)
            projected = clip_action_hd(projected, clip_eps=clip_eps)
            step_metrics.append(action_diff_metrics(raw, projected))
            action = projected
        next_observation, _reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        observation = next_observation
    ep = extract_episode_metrics(info)
    if step_metrics:
        for key in step_metrics[0]:
            ep[f"mean_{key}"] = float(np.mean([m[key] for m in step_metrics]))
    ep["num_steps"] = float(len(step_metrics))
    return ep, step_metrics


def summarize_condition_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0] if k not in ("condition", "episode", "env_seed")]
    summary: dict[str, float] = {"num_episodes": float(len(rows))}
    for key in keys:
        vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
        if vals:
            summary[f"mean_{key}"] = float(np.mean(vals))
            summary[f"std_{key}"] = float(np.std(vals))
    return summary


def create_and_restore_agent(
    *,
    seed: int,
    config: dict,
    ex_observations: np.ndarray,
    ex_actions: np.ndarray,
    checkpoint: Path | str,
    restore_epoch: int | None,
) -> DFLRQL9Agent:
    agent = DFLRQL9Agent.create(seed, ex_observations, ex_actions, config)
    restore_path = str(checkpoint)
    if Path(restore_path).is_dir() and restore_epoch is None:
        raise ValueError(
            "--restore-epoch required when checkpoint path is a directory"
        )
    return restore_agent(agent, restore_path, restore_epoch)


def parse_conditions(raw: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in raw:
        for part in str(item).replace(",", " ").split():
            name = part.strip().lower()
            if name not in VALID_CONDITIONS:
                raise SystemExit(
                    f"Unknown condition {name!r}; expected one of {VALID_CONDITIONS}"
                )
            if name not in out:
                out.append(name)
    if not out:
        raise SystemExit("Provide at least one --conditions value")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Paired DFLRQL9 eval: continuous raw vs OATTok discrete-latent "
            "deployment filter (not a learned discrete actor)."
        )
    )
    p.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exp run dir with flags.json (and optionally params_*.pkl).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to params_*.pkl (or directory + --restore-epoch).",
    )
    p.add_argument("--restore-epoch", type=int, default=None)
    p.add_argument("--env-name", type=str, default=None)
    p.add_argument(
        "--tokenizer",
        "--tokenizer-path",
        dest="tokenizer",
        type=Path,
        required=True,
        help="Frozen OATTok .pkl (e.g. humanoidmaze-large_h1_d21.pkl).",
    )
    p.add_argument(
        "--ogbench-data-dir",
        type=str,
        default="/workspace-SR008.nfs2/users/staroverov/ogbench/data",
    )
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument(
        "--base-eval-seed",
        type=int,
        default=0,
        help="Base seed for paired episode env/actor keys.",
    )
    p.add_argument(
        "--eval-temperature",
        type=float,
        default=0.0,
        help="Actor temperature (0 = target_actor; default matches stock eval).",
    )
    p.add_argument(
        "--conditions",
        nargs="+",
        default=[CONDITION_RAW, CONDITION_PROJECTED],
        help="Subset of: raw projected",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output prefix; writes <prefix>_episodes.csv and <prefix>_summary.json",
    )
    p.add_argument("--clip-eps", type=float, default=0.0)
    p.add_argument(
        "--agent-seed",
        type=int,
        default=None,
        help="Seed for agent.create before restore (default: flags seed or 0).",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    conditions = parse_conditions(args.conditions)

    flags: dict = {}
    if args.run_dir is not None:
        flags = load_flags(args.run_dir)
        if args.checkpoint is None:
            if args.restore_epoch is not None:
                args.checkpoint = args.run_dir
            else:
                args.checkpoint = latest_params_pkl(args.run_dir)

    if args.checkpoint is None:
        raise SystemExit("Provide --checkpoint or --run-dir with params_*.pkl")

    env_name = args.env_name or flags.get("env_name")
    if not env_name:
        raise SystemExit("Provide --env-name or a --run-dir with flags.json")

    cfg = config_from_flags(flags)
    ogbench_data_dir = (
        args.ogbench_data_dir
        or flags.get("ogbench_data_dir")
        or "/workspace-SR008.nfs2/users/staroverov/ogbench/data"
    )
    agent_seed = (
        args.agent_seed
        if args.agent_seed is not None
        else int(flags.get("seed", 0))
    )

    tok_params, tok_meta = load_tokenizer(str(args.tokenizer))
    policy_h = int(cfg["h"])
    # Primitive dim inferred from tokenizer after assert; create needs ex_actions.
    print(f"env={env_name}")
    print(f"checkpoint={args.checkpoint}")
    print(f"tokenizer={args.tokenizer}")
    print(
        f"conditions={conditions} episodes={args.num_episodes} "
        f"base_eval_seed={args.base_eval_seed} eval_temperature={args.eval_temperature}"
    )
    print(
        "note: projected = discrete-latent deployment filter over continuous "
        "DFLRQL9; not a learned discrete policy"
    )

    # Separate envs per condition so paired resets cannot share wrapper state.
    envs: dict[str, Any] = {}
    train_dataset = None
    for cond in conditions:
        env, _eval_env, train_ds, _val = make_env_and_datasets(
            env_name,
            frame_stack=None,
            agent_config=cfg,
            dataset_dir=ogbench_data_dir,
        )
        envs[cond] = env
        if train_dataset is None:
            train_dataset = train_ds

    assert train_dataset is not None
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=train_dataset.size + 1
    )
    train_dataset.config = cfg
    # Match main.py: pass the full example batch into create() before restore.
    ex_batch = train_dataset.sample(1)
    ex_obs = np.asarray(ex_batch["observations"])
    ex_act = np.asarray(ex_batch["actions"])
    prim_dim = int(ex_act.shape[-1])

    assert_tokenizer_matches_policy(tok_meta, policy_h, prim_dim)
    tokenizer = build_oattok(tok_params, tok_meta)
    projector = OATTokActionProjector(
        tokenizer, tok_params, clip_eps=args.clip_eps
    )

    agent = create_and_restore_agent(
        seed=agent_seed,
        config=cfg,
        ex_observations=ex_obs,
        ex_actions=ex_act,
        checkpoint=args.checkpoint,
        restore_epoch=args.restore_epoch,
    )

    episode_rows: list[dict[str, Any]] = []
    for ep_idx in range(int(args.num_episodes)):
        env_seed = episode_env_seed(args.base_eval_seed, ep_idx)
        init_actor_key = actor_key_for_episode(args.base_eval_seed, ep_idx)
        for cond in conditions:
            # Fresh deterministic actor schedule from the same initial key.
            actor_fn = supply_rng_from_key(
                agent.sample_actions, rng=init_actor_key
            )
            project_fn = projector if cond == CONDITION_PROJECTED else None
            ep_metrics, _ = run_episode(
                envs[cond],
                actor_fn,
                env_seed=env_seed,
                eval_temperature=float(args.eval_temperature),
                project_fn=project_fn,
                clip_eps=args.clip_eps,
            )
            row = {
                "condition": cond,
                "episode": ep_idx,
                "env_seed": env_seed,
                **ep_metrics,
            }
            episode_rows.append(row)
            succ = ep_metrics.get("success", float("nan"))
            ret = ep_metrics.get("episode.return", float("nan"))
            print(
                f"ep={ep_idx} cond={cond} success={succ:.3f} "
                f"return={ret:.1f} proj_rmse={ep_metrics.get('mean_proj_rmse', 0):.4f}"
            )

    by_cond: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        by_cond[row["condition"]].append(row)
    summary = {
        "env_name": env_name,
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "tokenizer_meta": {
            k: tok_meta[k]
            for k in (
                "sample_dim",
                "sample_horizon",
                "num_registers",
                "domain",
            )
            if k in tok_meta
        },
        "base_eval_seed": int(args.base_eval_seed),
        "eval_temperature": float(args.eval_temperature),
        "num_episodes": int(args.num_episodes),
        "conditions": conditions,
        "terminology": (
            "projected is a discrete-latent deployment filter over continuous "
            "DFLRQL9; not a learned discrete policy"
        ),
        "per_condition": {
            cond: summarize_condition_rows(rows) for cond, rows in by_cond.items()
        },
    }

    out_prefix = args.output
    if out_prefix is None:
        ckpt_stem = Path(args.checkpoint).stem
        out_prefix = Path.cwd() / f"dflrql_oattok_proj_{ckpt_stem}"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(f"{out_prefix}_episodes.csv")
    json_path = Path(f"{out_prefix}_summary.json")

    fieldnames: list[str] = []
    for row in episode_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print("=== summary ===")
    print(json.dumps(summary["per_condition"], indent=2, sort_keys=True))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
