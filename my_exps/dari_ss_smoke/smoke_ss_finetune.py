#!/usr/bin/env python3
"""Stock DiscreteARIQL scheduled-sampling smoke: restore 200k, soft re-entry SS, short FT."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.discrete_ar_iql import DiscreteARIQLAgent, get_config
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.flax_utils import restore_agent, save_agent

RUN_DIR = Path(
    "exp/rql/humanoidmaze-large-dari-v6-awr5-smoke300k/sd000_20260715_213015"
)
CKPT = RUN_DIR / "params_200000.pkl"
OUT_DIR = Path("my_exps/dari_ss_smoke/run_ss_ft_from200k")
NUM_UPDATES = int(os.environ.get("SS_SMOKE_UPDATES", "800"))
BATCH = int(os.environ.get("SS_SMOKE_BATCH", "64"))
# Soft re-entry: ramp λ/p from resume step over this many updates.
SOFT_RAMP = int(os.environ.get("SS_SMOKE_SOFT_RAMP", "400"))
SEED = 0
HELD_OUT_SEED = 12345


def _f(x):
    return float(np.asarray(x))


def _fmt(d: dict, keys=None) -> str:
    keys = keys or sorted(d.keys())
    return "\n".join(f"  {k}: {_f(d[k]):.6f}" for k in keys if k in d)


def _param_shapes(params):
    return jax.tree_util.tree_map(lambda x: tuple(np.asarray(x).shape), params)


def _diag(agent, batch, seed):
    return agent.diagnose_teacher_vs_freerun(
        batch["observations"][0],
        batch["actions"],
        seed=seed,
        temperature=0.0,
        force_argmax=True,
    )


def _mean_info(infos):
    keys = infos[0].keys()
    return {k: float(np.mean([_f(i[k]) for i in infos])) for k in keys}


def main():
    os.chdir(ROOT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flags = json.loads((RUN_DIR / "flags.json").read_text())
    cfg = dict(get_config())
    for k, v in (flags.get("agent") or {}).items():
        if k in cfg or k == "tokenizer_path":
            cfg[k] = v
    cfg["batch_size"] = BATCH
    # Enable SS with soft re-entry from restored step (schedule uses network.step).
    # After restore, step≈200001; set start to that so frac ramps 0→1 over SOFT_RAMP.
    cfg["ss_start_steps"] = 200_000  # will refine after restore
    cfg["ss_ramp_steps"] = SOFT_RAMP
    cfg["ss_loss_coef"] = 0.5
    cfg["ss_prefix_prob_min"] = 0.1
    cfg["ss_prefix_prob_max"] = 0.35
    cfg["ss_pred_mode"] = "argmax"

    ogbench = flags.get("ogbench_data_dir") or "/workspace-SR008.nfs2/users/staroverov/ogbench/data"
    env_name = flags["env_name"]
    print(f"GPU visible: {jax.devices()}")
    print(f"env={env_name} ckpt={CKPT} batch={BATCH} updates={NUM_UPDATES}")
    print(f"SS soft-ramp={SOFT_RAMP} coef_max={cfg['ss_loss_coef']} p=[{cfg['ss_prefix_prob_min']},{cfg['ss_prefix_prob_max']}]")

    _, _, train_dataset, _ = make_env_and_datasets(
        env_name, frame_stack=None, agent_config=cfg, dataset_dir=ogbench
    )
    train_dataset = Dataset.create(**train_dataset)
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=train_dataset.size + 1
    )
    train_dataset.config = cfg
    ex = train_dataset.sample(1)

    agent = DiscreteARIQLAgent.create(SEED, ex["observations"], ex["actions"], cfg)
    shapes_before = _param_shapes(agent.network.params)
    step_before = int(agent.network.step)
    print(f"create(): network.step={step_before}")

    agent = restore_agent(agent, str(CKPT), None)
    step0 = int(agent.network.step)
    shapes_after = _param_shapes(agent.network.params)
    shape_ok = shapes_before == shapes_after
    print(f"restore: network.step={step0} param_tree_shape_match={shape_ok}")
    if not shape_ok:
        # Find diffs
        flat_b, _ = jax.tree_util.tree_flatten_with_path(shapes_before)
        flat_a, _ = jax.tree_util.tree_flatten_with_path(shapes_after)
        for (pb, sb), (pa, sa) in zip(flat_b, flat_a):
            if sb != sa:
                print(f"  SHAPE DIFF {pb}: {sb} vs {sa}")
        raise SystemExit("BLOCKER: param tree shape mismatch after restore")

    # Soft re-entry: ramp from current step so λ/p exercise during FT.
    # FrozenDict → replace config with updated start.
    new_cfg = dict(agent.config)
    new_cfg["ss_start_steps"] = step0
    new_cfg["ss_ramp_steps"] = SOFT_RAMP
    agent = agent.replace(config=type(agent.config)(**new_cfg))
    c0, p0 = DiscreteARIQLAgent.ss_schedule(
        step0, new_cfg["ss_start_steps"], SOFT_RAMP, 0.5, 0.1, 0.35
    )
    print(f"soft re-entry: ss_start={step0} ramp={SOFT_RAMP} → at step0 λ={_f(c0):.4f} p={_f(p0):.4f}")

    # Held-out batch (fixed RNG via dataset sample seed by resampling with fixed indices)
    rng = np.random.RandomState(HELD_OUT_SEED)
    # Use agent sampling path: sample once and keep.
    # ReplayBuffer.sample is random; freeze by keeping the numpy batch.
    held = train_dataset.sample(BATCH)
    # Make held-out immutable copy
    held = {k: np.array(v) for k, v in held.items()}

    print("\n=== BEFORE free-run diagnostic (held-out) ===")
    before = _diag(agent, held, jax.random.PRNGKey(HELD_OUT_SEED))
    print(_fmt(before))

    # One update to verify SS branch activates after ramp starts (step advances)
    # Warm JIT
    b0 = train_dataset.sample(BATCH)
    t_jit = time.time()
    agent, info0 = agent.update(b0)
    jax.block_until_ready(info0["total_loss"])
    jit_s = time.time() - t_jit
    print(f"\nfirst update (incl JIT): {jit_s:.2f}s step={int(agent.network.step)}")
    print(
        f"  ss_coef={_f(info0['ss_coef']):.4f} ss_p={_f(info0['ss_prefix_prob']):.4f} "
        f"replaced={_f(info0['ss_replaced_frac']):.4f} ss_acc={_f(info0['ss_token_acc']):.4f}"
    )

    # Training loop
    hist = []
    t0 = time.time()
    mem0 = None
    try:
        mem0 = jax.local_devices()[0].memory_stats()
    except Exception:
        pass

    log_every = max(50, NUM_UPDATES // 10)
    for i in range(1, NUM_UPDATES):
        batch = train_dataset.sample(BATCH)
        agent, info = agent.update(batch)
        if i % log_every == 0 or i == NUM_UPDATES - 1:
            row = {
                "i": i,
                "step": int(agent.network.step),
                "total_loss": _f(info["total_loss"]),
                "actor_loss": _f(info["actor_loss"]),
                "tf_actor_loss": _f(info["tf_actor_loss"]),
                "ss_actor_loss": _f(info["ss_actor_loss"]),
                "ss_coef": _f(info["ss_coef"]),
                "ss_prefix_prob": _f(info["ss_prefix_prob"]),
                "ss_replaced_frac": _f(info["ss_replaced_frac"]),
                "ss_token_acc": _f(info["ss_token_acc"]),
                "token_acc": _f(info["token_acc"]),
                "q_loss": _f(info["q_loss"]),
                "v_loss": _f(info["v_loss"]),
                "q": _f(info["q"]),
                "v": _f(info.get("v", 0.0)),
                "awr_ess": _f(info.get("awr_ess", info.get("ess", 0.0))),
                "awr_clip_frac": _f(info.get("awr_clip_frac", 0.0)),
                "v_mean": _f(info.get("v_mean", 0.0)),
                "grad_delta_actor": _f(info["grad_delta_actor"]),
                "grad_delta_q": _f(info["grad_delta_q"]),
                "grad_delta_v": _f(info["grad_delta_v"]),
            }
            # fill missing keys safely
            for k in list(row.keys()):
                if k in ("i", "step"):
                    continue
                if k not in info and k not in ("v", "awr_ess", "clip_frac"):
                    pass
            # pull optional keys from info
            for opt in ("v", "ess", "awr_ess", "clip_frac", "advantage"):
                if opt in info:
                    row[opt] = _f(info[opt])
            hist.append(row)
            print(
                f"[{i}/{NUM_UPDATES}] step={row['step']} loss={row['total_loss']:.4f} "
                f"λ={row['ss_coef']:.3f} p={row['ss_prefix_prob']:.3f} "
                f"repl={row['ss_replaced_frac']:.3f} ss_acc={row['ss_token_acc']:.3f} "
                f"tf_acc={row['token_acc']:.3f} q={row['q']:.3f} "
                f"|Δa|={row['grad_delta_actor']:.4f}"
            )

    jax.block_until_ready(agent.network.params["modules_actor"])
    elapsed = time.time() - t0
    ups = NUM_UPDATES / max(elapsed, 1e-6)
    mem1 = None
    try:
        mem1 = jax.local_devices()[0].memory_stats()
    except Exception:
        pass

    print(f"\n=== speed: {ups:.2f} updates/s over {elapsed:.1f}s ({NUM_UPDATES} updates) ===")
    if mem1:
        print(
            f"memory: bytes_in_use={mem1.get('bytes_in_use')} "
            f"peak={mem1.get('peak_bytes_in_use')} "
            f"(pre={None if not mem0 else mem0.get('bytes_in_use')})"
        )

    print("\n=== AFTER free-run diagnostic (same held-out) ===")
    after = _diag(agent, held, jax.random.PRNGKey(HELD_OUT_SEED))
    print(_fmt(after))

    # Exposure gap deltas
    gap_keys = [
        "tf_token_acc",
        "fr_token_acc",
        "tf_seq_exact",
        "fr_seq_exact",
        "tf_prefix_exact_1",
        "fr_prefix_exact_1",
        "tf_prefix_exact_4",
        "fr_prefix_exact_4",
        "tf_prefix_exact_8",
        "fr_prefix_exact_8",
        "tf_action_rmse_gt",
        "fr_action_rmse_gt",
        "tf_action_corr_gt",
        "fr_action_corr_gt",
    ]
    print("\n=== BEFORE → AFTER deltas (held-out) ===")
    deltas = {}
    for k in gap_keys:
        if k in before and k in after:
            d = _f(after[k]) - _f(before[k])
            deltas[k] = d
            print(f"  {k}: {_f(before[k]):.4f} → {_f(after[k]):.4f}  (Δ={d:+.4f})")

    # Exposure gap: TF - FR token acc (positive = exposure bias)
    gap_b = _f(before["tf_token_acc"]) - _f(before["fr_token_acc"])
    gap_a = _f(after["tf_token_acc"]) - _f(after["fr_token_acc"])
    seq_gap_b = _f(before["tf_seq_exact"]) - _f(before["fr_seq_exact"])
    seq_gap_a = _f(after["tf_seq_exact"]) - _f(after["fr_seq_exact"])
    print(f"\nexposure gap token (TF-FR): {gap_b:.4f} → {gap_a:.4f} (Δ={gap_a-gap_b:+.4f})")
    print(f"exposure gap seq   (TF-FR): {seq_gap_b:.4f} → {seq_gap_a:.4f} (Δ={seq_gap_a-seq_gap_b:+.4f})")

    # Checkpoint save/restore roundtrip
    save_epoch = int(agent.network.step)
    save_agent(agent, str(OUT_DIR), save_epoch)
    agent2 = DiscreteARIQLAgent.create(SEED, ex["observations"], ex["actions"], dict(agent.config))
    agent2 = restore_agent(agent2, str(OUT_DIR), save_epoch)
    step_rt = int(agent2.network.step)
    # Compare a leaf
    a = np.asarray(agent.network.params["modules_actor"]["Dense_0"]["kernel"] if False else
                   list(jax.tree_util.tree_leaves(agent.network.params["modules_actor"]))[0])
    b = np.asarray(list(jax.tree_util.tree_leaves(agent2.network.params["modules_actor"]))[0])
    max_abs = float(np.max(np.abs(a - b)))
    print(f"\nsave/restore: step {save_epoch}→{step_rt} actor_leaf_max_abs_diff={max_abs:.3e}")

    # Stability summary from hist
    if hist:
        losses = [h["total_loss"] for h in hist]
        ss_coefs = [h["ss_coef"] for h in hist]
        print("\n=== stability (logged) ===")
        print(f"  total_loss first/last/min/max: {losses[0]:.4f}/{losses[-1]:.4f}/{min(losses):.4f}/{max(losses):.4f}")
        print(f"  ss_coef first/last: {ss_coefs[0]:.4f}/{ss_coefs[-1]:.4f}")
        print(f"  ss_replaced last: {hist[-1]['ss_replaced_frac']:.4f}")
        print(f"  grad_delta_actor last: {hist[-1]['grad_delta_actor']:.6f}")
        finite = all(np.isfinite(h["total_loss"]) for h in hist)
        print(f"  all_finite_losses: {finite}")

    improved = (gap_a < gap_b - 1e-4) or (seq_gap_a < seq_gap_b - 1e-4)
    fr_improved = (_f(after["fr_token_acc"]) > _f(before["fr_token_acc"]) + 1e-4) or (
        _f(after["fr_seq_exact"]) > _f(before["fr_seq_exact"]) + 1e-4
    )
    print("\n=== VERDICT ===")
    print(f"  param_restore_ok: {shape_ok}")
    print(f"  network.step_restored: {step0}")
    print(f"  ss_ramp_exercised: {ss_coefs[0] < ss_coefs[-1] if hist else False}")
    print(f"  exposure_gap_reduced: {improved}")
    print(f"  freerun_metrics_improved: {fr_improved}")
    print(f"  short_FT_helps_exposure: {improved or fr_improved}")

    report = {
        "step0": step0,
        "step_final": int(agent.network.step),
        "num_updates": NUM_UPDATES,
        "batch": BATCH,
        "soft_ramp": SOFT_RAMP,
        "ups": ups,
        "before": {k: _f(v) for k, v in before.items()},
        "after": {k: _f(v) for k, v in after.items()},
        "deltas": deltas,
        "gap_token_before": gap_b,
        "gap_token_after": gap_a,
        "gap_seq_before": seq_gap_b,
        "gap_seq_after": seq_gap_a,
        "restore_max_abs": max_abs,
        "hist": hist,
        "exposure_gap_reduced": improved,
        "freerun_improved": fr_improved,
    }
    out_json = OUT_DIR / "smoke_report.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
