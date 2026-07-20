#!/usr/bin/env python3
"""Reproducible ConsensusFlow / OGBench-50 aggregation with bootstrap CIs.

Authoritative paper protocol for OGBench-50 paper results (tables +
results-gains supplement figure; domain learning curves use the same checkpoint):
  - seeds exactly 0, 1, 2
  - final_checkpoint_at_2m: evaluation/success at exactly 2,000,000 steps
  - then mean over seeds, then mean over tasks

Also aggregates humanoidmaze-large ConsensusFlow ablations at a labeled budget
(default 1M; shorter than the 2M paper protocol — never treat as equivalent).
Short-budget ablations likewise use only their exact final checkpoint.

Outputs JSON (+ optional CSV/MD) under my_exps/ by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ogbench50_config import expand_tasks, load_tasks_config  # noqa: E402
from plot_ogbench50_dflrql_vs_baseline import (  # noqa: E402
    AGENT_FOR_METHOD,
    METHOD_CSV_KEY,
    build_result_index,
    final_metrics_for_task,
    parse_eval_csv,
)

RQL_ROOT = SCRIPT_DIR.parent
DEFAULT_SAVE_DIR = RQL_ROOT / "exp"
DEFAULT_CSV = RQL_ROOT / "my_exps" / "ogbench50_all50_metrics_2m.csv"
DEFAULT_OUT = RQL_ROOT / "my_exps" / "consensusflow_stats_final_2m.json"

PAPER_SEEDS = (0, 1, 2)
PAPER_REPORT_STEP = 2_000_000
OGBENCH_METHODS = ("baseline", "dflrql6", "dflrql7", "dflrql8", "dflrql9")

# Controlled humanoidmaze-large ablations (common 1M budget; full method reused).
ABLATION_SPECS = {
    "full": {
        "run_group": "humanoidmaze-large-dflrql9-2m",
        "reuse_from_2m": True,
        "description": "Full ConsensusFlow/v9 defaults (reused from matched 2M runs)",
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.01,
            "conflict_power": 2.0,
            "residual_coef": 0.25,
            "ensemble_ct": 10,
        },
    },
    "no_guidance": {
        "run_group": "humanoidmaze-large-cf-ablation-noguidance-1m",
        "reuse_from_2m": False,
        "description": "guidance_coef=0 (no guidance term at train/sample)",
        "flags": {
            "guidance_coef": 0.0,
            "distill_coef": 1.0,
            "consensus_floor": 0.01,
            "conflict_power": 2.0,
            "residual_coef": 0.25,
            "ensemble_ct": 10,
        },
    },
    "lambda02": {
        "run_group": "humanoidmaze-large-cf-ablation-nocrf-lambda02-1m",
        "reuse_from_2m": False,
        "description": (
            "guidance_coef=0.2 under no-CRF safety-off "
            "(conflict_power=0, residual_coef=0, consensus_floor=0)"
        ),
        "flags": {
            "guidance_coef": 0.2,
            "distill_coef": 1.0,
            "consensus_floor": 0.0,
            "conflict_power": 0.0,
            "residual_coef": 0.0,
            "ensemble_ct": 10,
        },
    },
    "lambda10": {
        "run_group": "humanoidmaze-large-cf-ablation-nocrf-lambda10-1m",
        "reuse_from_2m": False,
        "description": (
            "guidance_coef=1.0 under no-CRF safety-off "
            "(conflict_power=0, residual_coef=0, consensus_floor=0)"
        ),
        "flags": {
            "guidance_coef": 1.0,
            "distill_coef": 1.0,
            "consensus_floor": 0.0,
            "conflict_power": 0.0,
            "residual_coef": 0.0,
            "ensemble_ct": 10,
        },
    },
    "no_conflict": {
        "run_group": "humanoidmaze-large-cf-ablation-noconflict-1m",
        "reuse_from_2m": False,
        "description": (
            "conflict_power=0 → kill_frac=1-trust^0=0 for all trust "
            "(disables trust-weighted BC-conflict projection; residual damping remains)"
        ),
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.01,
            "conflict_power": 0.0,
            "residual_coef": 0.25,
            "ensemble_ct": 10,
        },
    },
    "no_residual": {
        "run_group": "humanoidmaze-large-cf-ablation-noresidual-1m",
        "reuse_from_2m": False,
        "description": "residual_coef=0 (disables residual damping when BC aligns)",
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.01,
            "conflict_power": 2.0,
            "residual_coef": 0.0,
            "ensemble_ct": 10,
        },
    },
    "no_floor": {
        "run_group": "humanoidmaze-large-cf-ablation-nofloor-1m",
        "reuse_from_2m": False,
        "description": (
            "consensus_floor=0 disables the batch-relative scale-free floor "
            "(closest controlled switch). Distill target remains "
            "q_grad/(||q_grad||+1e-6) unit normalization; not a raw-gradient ablation."
        ),
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.0,
            "conflict_power": 2.0,
            "residual_coef": 0.25,
            "ensemble_ct": 10,
        },
    },
    "no_crf": {
        "run_group": "humanoidmaze-large-cf-ablation-nocrf-1m",
        "reuse_from_2m": False,
        "description": (
            "Combined safety-off ablation: conflict_power=0, residual_coef=0, "
            "consensus_floor=0 (guidance_coef retained at 0.5)"
        ),
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.0,
            "conflict_power": 0.0,
            "residual_coef": 0.0,
            "ensemble_ct": 10,
        },
    },
    "nocrf_k2": {
        "run_group": "humanoidmaze-large-cf-ablation-nocrf-k2-1m",
        "reuse_from_2m": False,
        "description": "no_crf + ensemble_ct=2 (K=2 under p=β=c=0)",
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.0,
            "conflict_power": 0.0,
            "residual_coef": 0.0,
            "ensemble_ct": 2,
        },
    },
    "nocrf_k5": {
        "run_group": "humanoidmaze-large-cf-ablation-nocrf-k5-1m",
        "reuse_from_2m": False,
        "description": "no_crf + ensemble_ct=5 (K=5 under p=β=c=0)",
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.0,
            "conflict_power": 0.0,
            "residual_coef": 0.0,
            "ensemble_ct": 5,
        },
    },
    "nocrf_k20": {
        "run_group": "humanoidmaze-large-cf-ablation-nocrf-k20-1m",
        "reuse_from_2m": False,
        "description": "no_crf + ensemble_ct=20 (K=20 under p=β=c=0)",
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.0,
            "conflict_power": 0.0,
            "residual_coef": 0.0,
            "ensemble_ct": 20,
        },
    },
    "single_critic": {
        "run_group": "humanoidmaze-large-cf-ablation-singlecritic-1m",
        "reuse_from_2m": False,
        "description": "ensemble_ct=1 (single critic; consensus/trust signals collapse)",
        "flags": {
            "guidance_coef": 0.5,
            "distill_coef": 1.0,
            "consensus_floor": 0.01,
            "conflict_power": 2.0,
            "residual_coef": 0.25,
            "ensemble_ct": 1,
        },
    },
}

DIAGNOSTIC_KEYS = (
    "training/trust",
    "training/conflict_kill_frac",
    "training/residual_damp",
    "training/guidance_retained",
    "training/behavior_conflict_fraction",
    "training/safe_w_norm",
    "training/w_norm",
    "training/consensus_target_norm",
)


@dataclass
class SeedMetric:
    seed: int
    success: float | None
    mode: str
    max_step: int
    run_dir: str | None
    flags_match: bool | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def domain_of_task(task_id: str) -> str:
    # antmaze-giant-task1 → antmaze-giant; cube-double-task3 → cube-double
    if "-task" in task_id:
        return task_id.rsplit("-task", 1)[0]
    return task_id


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = values.size
    for i in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        means[i] = float(sample.mean())
    low = float(np.quantile(means, alpha / 2))
    high = float(np.quantile(means, 1 - alpha / 2))
    return {
        "mean": float(values.mean()),
        "ci_low": low,
        "ci_high": high,
        "n": int(n),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }


def paired_bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if a.size == 0:
        return {
            "mean_delta": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
        }
    delta = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = delta.size
    for i in range(n_boot):
        sample = delta[rng.integers(0, n, size=n)]
        means[i] = float(sample.mean())
    return {
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(means, alpha / 2)),
        "ci_high": float(np.quantile(means, 1 - alpha / 2)),
        "n": int(n),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
    }


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    out: dict[str, Any] = {"n_paired": int(a.size)}
    if a.size < 2:
        out["note"] = "insufficient paired samples"
        return out
    delta = a - b
    out["mean_delta"] = float(delta.mean())
    out["std_delta"] = float(delta.std(ddof=1)) if a.size > 1 else float("nan")
    # Paired t-test (scipy if available; else manual).
    try:
        from scipy import stats

        t_res = stats.ttest_rel(a, b, nan_policy="omit")
        out["paired_t"] = {
            "statistic": float(t_res.statistic),
            "pvalue": float(t_res.pvalue),
        }
        try:
            if np.allclose(delta, 0.0):
                out["wilcoxon"] = {
                    "statistic": 0.0,
                    "pvalue": 1.0,
                    "note": "all paired deltas are zero",
                }
            else:
                w_res = stats.wilcoxon(
                    delta, zero_method="wilcox", alternative="two-sided"
                )
                out["wilcoxon"] = {
                    "statistic": float(w_res.statistic),
                    "pvalue": float(w_res.pvalue),
                }
        except ValueError as exc:
            out["wilcoxon"] = {"error": str(exc)}
    except ImportError:
        # Manual paired t
        n = a.size
        se = out["std_delta"] / math.sqrt(n)
        t_stat = out["mean_delta"] / se if se > 0 else float("nan")
        out["paired_t"] = {
            "statistic": float(t_stat),
            "pvalue": None,
            "note": "scipy unavailable; pvalue not computed",
        }
    wins = int(np.sum(a > b + 1e-12))
    losses = int(np.sum(a < b - 1e-12))
    ties = int(n - wins - losses) if (n := a.size) else 0
    out["win_loss_tie"] = {"win": wins, "loss": losses, "tie": ties}
    return out


def load_source_csv_task_means(
    csv_path: Path,
    seeds: tuple[int, ...] = PAPER_SEEDS,
) -> dict[str, dict[str, Any]]:
    """Parse ogbench50_all50_metrics_2m.csv into per-method per-task seed means."""
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    methods = ("baseline", "v6", "v7", "v8", "v9")
    out: dict[str, dict[str, Any]] = {m: {"tasks": {}, "grand_values": []} for m in methods}
    for row in rows:
        task_id = row["task_id"]
        domain_id = row["domain_id"]
        for m in methods:
            seed_vals = []
            for s in seeds:
                key = f"{m}_seed{s}"
                if key in row and row[key] not in (None, ""):
                    seed_vals.append(float(row[key]))
            if not seed_vals:
                continue
            mean = float(np.mean(seed_vals))
            out[m]["tasks"][task_id] = {
                "domain_id": domain_id,
                "seed_values": seed_vals,
                "seeds_present": [
                    s
                    for s in seeds
                    if f"{m}_seed{s}" in row and row[f"{m}_seed{s}"] not in (None, "")
                ],
                "mean": mean,
                "n": len(seed_vals),
            }
            out[m]["grand_values"].append(mean)
    for m in methods:
        vals = out[m]["grand_values"]
        out[m]["grand_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[m]["n_tasks"] = len(vals)
        incomplete = [
            {"task_id": tid, "n": info["n"], "seeds_present": info["seeds_present"]}
            for tid, info in out[m]["tasks"].items()
            if info["n"] < len(seeds)
        ]
        out[m]["incomplete_tasks"] = incomplete
        # Domain means
        by_dom: dict[str, list[float]] = defaultdict(list)
        for tid, info in out[m]["tasks"].items():
            by_dom[info["domain_id"]].append(info["mean"])
        out[m]["domain_means"] = {d: float(np.mean(v)) for d, v in sorted(by_dom.items())}
    return out


def recompute_from_raw(
    save_dir: Path,
    seeds: tuple[int, ...] = PAPER_SEEDS,
    report_step: int = PAPER_REPORT_STEP,
) -> dict[str, dict[str, Any]]:
    """Recompute exact final-checkpoint success from raw eval.csv."""
    tasks = expand_tasks(load_tasks_config())
    indexes = {
        seed: build_result_index(
            str(save_dir),
            seed,
            prefer_step=report_step,
            prefer_checkpoints=None,
        )
        for seed in seeds
    }
    methods = OGBENCH_METHODS
    out: dict[str, dict[str, Any]] = {
        METHOD_CSV_KEY[m]: {"tasks": {}, "grand_values": []} for m in methods
    }
    for task in tasks:
        for method in methods:
            csv_key = METHOD_CSV_KEY[method]
            seed_vals: list[float] = []
            seeds_present: list[int] = []
            run_dirs: dict[str, str] = {}
            for seed in seeds:
                success, _ = final_metrics_for_task(
                    str(save_dir),
                    task,
                    method,
                    seed,
                    indexes[seed],
                    report_step,
                    require_exact_step=True,
                    use_log_fallback=False,
                )
                if success is None:
                    continue
                seed_vals.append(float(success))
                seeds_present.append(seed)
                rec = indexes[seed].get(task.env_name, {}).get(method)
                if rec is not None:
                    run_dirs[str(seed)] = str(rec.run_dir)
            if not seed_vals:
                continue
            mean = float(np.mean(seed_vals))
            out[csv_key]["tasks"][task.task_id] = {
                "domain_id": task.domain_id,
                "env_name": task.env_name,
                "seed_values": seed_vals,
                "seeds_present": seeds_present,
                "mean": mean,
                "n": len(seed_vals),
                "run_dirs": run_dirs,
            }
            out[csv_key]["grand_values"].append(mean)
    for csv_key, payload in out.items():
        vals = payload["grand_values"]
        payload["grand_mean"] = float(np.mean(vals)) if vals else float("nan")
        payload["n_tasks"] = len(vals)
        incomplete = [
            {"task_id": tid, "n": info["n"], "seeds_present": info["seeds_present"]}
            for tid, info in payload["tasks"].items()
            if info["n"] < len(seeds)
        ]
        payload["incomplete_tasks"] = incomplete
        by_dom: dict[str, list[float]] = defaultdict(list)
        for tid, info in payload["tasks"].items():
            by_dom[info["domain_id"]].append(info["mean"])
        payload["domain_means"] = {d: float(np.mean(v)) for d, v in sorted(by_dom.items())}
        payload["agent_name"] = AGENT_FOR_METHOD.get(
            next(m for m in methods if METHOD_CSV_KEY[m] == csv_key), csv_key
        )
    return out


def compare_csv_vs_raw(
    csv_agg: dict[str, dict[str, Any]],
    raw_agg: dict[str, dict[str, Any]],
    atol: float = 1e-6,
) -> dict[str, Any]:
    report: dict[str, Any] = {"methods": {}, "all_match": True}
    for method in ("baseline", "v6", "v7", "v8", "v9"):
        csv_mean = csv_agg[method]["grand_mean"]
        raw_mean = raw_agg[method]["grand_mean"]
        delta = abs(csv_mean - raw_mean) if (
            not math.isnan(csv_mean) and not math.isnan(raw_mean)
        ) else float("nan")
        match = (not math.isnan(delta)) and delta <= atol
        mismatches = []
        csv_tasks = csv_agg[method]["tasks"]
        raw_tasks = raw_agg[method]["tasks"]
        for tid in sorted(set(csv_tasks) | set(raw_tasks)):
            c = csv_tasks.get(tid, {}).get("mean")
            r = raw_tasks.get(tid, {}).get("mean")
            if c is None or r is None:
                mismatches.append({"task_id": tid, "csv": c, "raw": r})
                match = False
            elif abs(c - r) > atol:
                mismatches.append({"task_id": tid, "csv": c, "raw": r, "abs_delta": abs(c - r)})
                match = False
        report["methods"][method] = {
            "csv_grand_mean": csv_mean,
            "raw_grand_mean": raw_mean,
            "abs_delta": delta,
            "match": match,
            "n_task_mismatches": len(mismatches),
            "task_mismatches_head": mismatches[:10],
        }
        report["all_match"] = report["all_match"] and match
    return report


def stats_for_method_vs_baseline(
    method_agg: dict[str, Any],
    baseline_agg: dict[str, Any],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    # Align on shared tasks
    shared = sorted(set(method_agg["tasks"]) & set(baseline_agg["tasks"]))
    m_vals = np.array([method_agg["tasks"][t]["mean"] for t in shared], dtype=float)
    b_vals = np.array([baseline_agg["tasks"][t]["mean"] for t in shared], dtype=float)
    return {
        "n_shared_tasks": len(shared),
        "method_bootstrap_ci": bootstrap_ci(m_vals, n_boot=n_boot, seed=seed),
        "baseline_bootstrap_ci": bootstrap_ci(b_vals, n_boot=n_boot, seed=seed + 1),
        "paired_delta_bootstrap_ci": paired_bootstrap_ci(
            m_vals, b_vals, n_boot=n_boot, seed=seed + 2
        ),
        "paired_tests": paired_tests(m_vals, b_vals),
        "domain_means": method_agg.get("domain_means", {}),
        "incomplete_tasks": method_agg.get("incomplete_tasks", []),
        "grand_mean": method_agg.get("grand_mean"),
    }


def find_best_run_dir(root: Path, seed: int, env_name: str, agent_name: str = "dflrql9") -> Path | None:
    candidates = sorted(root.glob(f"sd{seed:03d}_*"))
    best = None
    best_step = -1
    for run_dir in candidates:
        flags_path = run_dir / "flags.json"
        eval_path = run_dir / "eval.csv"
        if not flags_path.is_file() or not eval_path.is_file():
            continue
        flags = json.loads(flags_path.read_text())
        if flags.get("env_name") != env_name:
            continue
        if flags.get("agent", {}).get("agent_name") != agent_name:
            continue
        rows = parse_eval_csv(eval_path)
        if not rows:
            continue
        max_step = max(r["step"] for r in rows)
        if max_step >= best_step:
            best_step = max_step
            best = run_dir
    return best


def flags_match_expected(flags: dict, expected: dict[str, float], atol: float = 1e-9) -> bool:
    agent = flags.get("agent") or {}
    for key, want in expected.items():
        got = agent.get(key)
        if got is None:
            return False
        if abs(float(got) - float(want)) > atol:
            return False
    return True


def final_checkpoint_from_run(
    run_dir: Path,
    report_step: int,
) -> SeedMetric:
    eval_path = run_dir / "eval.csv"
    rows = parse_eval_csv(eval_path)
    max_step = max((r["step"] for r in rows), default=0)
    by_step = {r["step"]: r["success"] for r in rows}
    success = float(by_step[report_step]) if report_step in by_step else None
    mode = "final_checkpoint" if success is not None else "missing_final_checkpoint"
    return SeedMetric(
        seed=-1,
        success=success,
        mode=mode,
        max_step=max_step,
        run_dir=str(run_dir),
    )


def summarize_train_diagnostics(run_dir: Path, last_n: int = 20) -> dict[str, float]:
    train_path = run_dir / "train.csv"
    if not train_path.is_file():
        return {}
    with train_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    tail = rows[-last_n:]
    out: dict[str, float] = {}
    for key in DIAGNOSTIC_KEYS:
        vals = []
        for row in tail:
            if key in row and row[key] not in (None, ""):
                try:
                    vals.append(float(row[key]))
                except ValueError:
                    continue
        if vals:
            out[key.replace("training/", "") + "_last_mean"] = float(np.mean(vals))
    return out


def aggregate_ablations(
    save_dir: Path,
    budget_step: int,
    seeds: tuple[int, ...] = PAPER_SEEDS,
    env_name: str = "humanoidmaze-large-navigate-singletask-v0",
) -> dict[str, Any]:
    rql_root = save_dir / "rql"
    results: dict[str, Any] = {
        "budget_step": budget_step,
        "budget_label": (
            f"final_checkpoint_at_{budget_step // 1000}k"
            f" (exact step {budget_step}; "
            f"{'SHORTER THAN 2M PAPER PROTOCOL' if budget_step < PAPER_REPORT_STEP else 'full 2M paper protocol'})"
        ),
        "seeds": list(seeds),
        "env_name": env_name,
        "variants": {},
        "protocol_note": (
            "Do not compare mismatched budgets as equivalent. "
            "Ablation suite uses a common budget; full method is reused from "
            "humanoidmaze-large-dflrql9-2m only when flags exactly match v9 defaults "
            "and the exact final checkpoint at the ablation budget exists. "
            "OGBench paper values require exact 2,000,000-step checkpoints; "
            "the 1M suite is short-budget evidence, not a protocol-equivalent substitute."
        ),
    }
    for name, spec in ABLATION_SPECS.items():
        root = rql_root / spec["run_group"]
        seed_metrics: list[dict[str, Any]] = []
        successes: list[float] = []
        for seed in seeds:
            if not root.is_dir():
                seed_metrics.append(
                    asdict(
                        SeedMetric(
                            seed=seed,
                            success=None,
                            mode="missing_run_group",
                            max_step=0,
                            run_dir=None,
                            flags_match=False,
                        )
                    )
                )
                continue
            run_dir = find_best_run_dir(root, seed, env_name)
            if run_dir is None:
                seed_metrics.append(
                    asdict(
                        SeedMetric(
                            seed=seed,
                            success=None,
                            mode="missing_run",
                            max_step=0,
                            run_dir=None,
                            flags_match=False,
                        )
                    )
                )
                continue
            flags = json.loads((run_dir / "flags.json").read_text())
            match = flags_match_expected(flags, spec["flags"])
            metric = final_checkpoint_from_run(run_dir, budget_step)
            metric.seed = seed
            metric.flags_match = match
            payload = asdict(metric)
            payload["diagnostics"] = summarize_train_diagnostics(run_dir)
            payload["offline_steps"] = flags.get("offline_steps")
            payload["agent_flags"] = {
                k: (flags.get("agent") or {}).get(k) for k in spec["flags"]
            }
            seed_metrics.append(payload)
            if metric.success is not None and metric.mode == "final_checkpoint" and match:
                successes.append(float(metric.success))
        arr = np.array(successes, dtype=float) if successes else np.array([], dtype=float)
        results["variants"][name] = {
            "run_group": spec["run_group"],
            "description": spec["description"],
            "expected_flags": spec["flags"],
            "reuse_from_2m": spec["reuse_from_2m"],
            "seeds": seed_metrics,
            "n_complete_matched": len(successes),
            "mean": float(arr.mean()) if arr.size else None,
            "std": float(arr.std(ddof=1)) if arr.size > 1 else (0.0 if arr.size == 1 else None),
            "bootstrap_ci": bootstrap_ci(arr, seed=17) if arr.size else None,
        }
    # Paired tests vs full (seed-aligned)
    full = results["variants"]["full"]
    full_by_seed = {
        s["seed"]: s["success"]
        for s in full["seeds"]
        if s.get("success") is not None and s.get("mode") == "final_checkpoint" and s.get("flags_match")
    }
    comparisons: dict[str, Any] = {}
    for name, variant in results["variants"].items():
        if name == "full":
            continue
        a, b = [], []
        for s in variant["seeds"]:
            if (
                s.get("success") is not None
                and s.get("mode") == "final_checkpoint"
                and s.get("flags_match")
                and s["seed"] in full_by_seed
            ):
                a.append(float(full_by_seed[s["seed"]]))
                b.append(float(s["success"]))
        comparisons[name] = {
            "full_minus_ablation_paired_tests": paired_tests(np.array(a), np.array(b)),
            "full_minus_ablation_bootstrap_ci": paired_bootstrap_ci(
                np.array(a), np.array(b), seed=23
            )
            if a
            else None,
        }
    results["comparisons_vs_full"] = comparisons
    return results


def write_markdown_summary(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# ConsensusFlow empirical aggregation",
        "",
        f"Generated: `{payload['generated_utc']}`",
        "",
        "## OGBench-50 final_checkpoint_at_2m (seeds 0–2)",
        "",
        "**Protocol change (2026-07-16): only the exact 2,000,000-step "
        "checkpoint is authoritative. Earlier paper_last3 aggregates are legacy-only.**",
        "",
        f"Source CSV: `{payload['ogbench']['source_csv']}`",
        f"Source CSV sha256: `{payload['ogbench']['source_csv_sha256']}`",
        f"CSV↔raw all_match: `{payload['ogbench']['csv_vs_raw'].get('all_match')}`",
        f"Legacy paper_last3 CSV: `{payload['ogbench'].get('legacy_paper_last3_csv')}`",
        f"Raw recompute save_dir: `{payload['ogbench']['raw_save_dir']}`",
        "",
        "| Method | Final-2M grand mean | Bootstrap 95% CI (tasks) |",
        "|---|---:|---:|",
    ]
    for method in ("baseline", "v6", "v7", "v8", "v9"):
        raw_m = payload["ogbench"]["from_raw"][method]["grand_mean"]
        ci = payload["ogbench"]["statistics"][method]["method_bootstrap_ci"]
        lines.append(
            f"| {method} | {raw_m:.6f} | "
            f"[{ci['ci_low']:.6f}, {ci['ci_high']:.6f}] |"
        )
    v9_stats = payload["ogbench"]["statistics"]["v9"]
    lines += [
        "",
        "### v9 vs baseline (per-task seed-means)",
        "",
        f"- Δ mean: {v9_stats['paired_tests'].get('mean_delta')}",
        f"- Win/loss/tie: {v9_stats['paired_tests'].get('win_loss_tie')}",
        f"- Paired Δ bootstrap 95% CI: {v9_stats['paired_delta_bootstrap_ci']}",
        f"- Paired t: {v9_stats['paired_tests'].get('paired_t')}",
        f"- Wilcoxon: {v9_stats['paired_tests'].get('wilcoxon')}",
        "",
        "## humanoidmaze-large ConsensusFlow ablations",
        "",
    ]
    ablations = payload.get("ablations") or {}
    if ablations.get("skipped"):
        lines.append("_Ablation aggregation skipped in this run._")
    else:
        lines += [
            f"Budget: {ablations.get('budget_label')}",
            "",
            "| Variant | Mean | Std | n | Run group |",
            "|---|---:|---:|---:|---|",
        ]
        for name, var in (ablations.get("variants") or {}).items():
            lines.append(
                f"| {name} | {var['mean']} | {var['std']} | {var['n_complete_matched']} | `{var['run_group']}` |"
            )
        lines += ["", "### Seed-level", ""]
        for name, var in (ablations.get("variants") or {}).items():
            lines.append(f"#### {name}")
            for s in var["seeds"]:
                lines.append(
                    f"- seed {s['seed']}: success={s['success']} mode={s['mode']} "
                    f"flags_match={s['flags_match']} max_step={s['max_step']}"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, default=DEFAULT_SAVE_DIR)
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ablation-budget", type=int, default=1_000_000)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    args = parser.parse_args()

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    # Source CSV is the authoritative final-checkpoint@2M table (regenerated
    # from raw eval.csv). Legacy paper_last3 CSV is archived separately.
    csv_agg = load_source_csv_task_means(args.source_csv, PAPER_SEEDS)
    if args.skip_raw:
        raise ValueError(
            "--skip-raw is incompatible with the authoritative final-checkpoint "
            "protocol; always recompute from raw eval.csv"
        )
    raw_agg = recompute_from_raw(args.save_dir, PAPER_SEEDS, PAPER_REPORT_STEP)
    csv_vs_raw = compare_csv_vs_raw(csv_agg, raw_agg)
    legacy_last3_csv = RQL_ROOT / "my_exps" / "ogbench50_all50_metrics_2m_paper_last3_legacy.csv"
    legacy_comparison = None
    if legacy_last3_csv.is_file():
        legacy_agg = load_source_csv_task_means(legacy_last3_csv, PAPER_SEEDS)
        legacy_comparison = compare_csv_vs_raw(legacy_agg, raw_agg)

    statistics = {
        method: stats_for_method_vs_baseline(
            raw_agg[method],
            raw_agg["baseline"],
            n_boot=args.n_boot,
            seed=100 + i,
        )
        for i, method in enumerate(("baseline", "v6", "v7", "v8", "v9"))
    }

    ablations = (
        {"skipped": True}
        if args.skip_ablations
        else aggregate_ablations(args.save_dir, args.ablation_budget, PAPER_SEEDS)
    )

    payload: dict[str, Any] = {
        "title": "ConsensusFlow empirical aggregation (authoritative)",
        "generated_utc": generated,
        "paper_seed_protocol": {
            "seeds": list(PAPER_SEEDS),
            "metric_primary": "final_checkpoint_at_2m",
            "metric_definition": (
                "For each (task, seed): evaluation/success at exactly step "
                "2000000. Then mean over seeds 0,1,2. Then mean over tasks."
            ),
            "protocol_changed_utc": "2026-07-16T19:25:00Z",
            "protocol_change_note": (
                "All OGBench paper results (tables + results-gains figure) "
                "use only the final 2,000,000-step checkpoint. paper_last3 "
                "is superseded and "
                "must not be copied into paper figures, tables, or claims."
            ),
            "ablation_budget_step": args.ablation_budget,
            "ablation_budget_note": (
                "Ablations use a common shorter budget than 2M when "
                f"ablation_budget={args.ablation_budget} < 2000000; "
                "they use only the exact final checkpoint at that budget. "
                "Label clearly and never compare mismatched budgets as equivalent "
                "or use 1M ablations as final-2M OGBench results."
            ),
        },
        "ogbench": {
            "source_csv": str(args.source_csv),
            "source_csv_sha256": sha256_file(args.source_csv) if args.source_csv.is_file() else None,
            "source_protocol": "final_checkpoint_at_2m",
            "legacy_paper_last3_csv": str(legacy_last3_csv) if legacy_last3_csv.is_file() else None,
            "legacy_paper_last3_csv_sha256": (
                sha256_file(legacy_last3_csv) if legacy_last3_csv.is_file() else None
            ),
            "raw_save_dir": str(args.save_dir),
            "report_step": PAPER_REPORT_STEP,
            "report_checkpoint": PAPER_REPORT_STEP,
            "from_csv": {
                m: {
                    "grand_mean": csv_agg[m]["grand_mean"],
                    "n_tasks": csv_agg[m]["n_tasks"],
                    "incomplete_tasks": csv_agg[m]["incomplete_tasks"],
                    "domain_means": csv_agg[m]["domain_means"],
                }
                for m in ("baseline", "v6", "v7", "v8", "v9")
            },
            "from_raw": {
                m: {
                    "grand_mean": raw_agg[m]["grand_mean"],
                    "n_tasks": raw_agg[m]["n_tasks"],
                    "incomplete_tasks": raw_agg[m]["incomplete_tasks"],
                    "domain_means": raw_agg[m]["domain_means"],
                    "per_task": {
                        tid: {
                            "domain_id": info["domain_id"],
                            "env_name": info["env_name"],
                            "mean": info["mean"],
                            "n": info["n"],
                            "seed_values": info["seed_values"],
                            "seeds_present": info["seeds_present"],
                        }
                        for tid, info in raw_agg[m]["tasks"].items()
                    },
                }
                for m in ("baseline", "v6", "v7", "v8", "v9")
            },
            "csv_vs_raw": csv_vs_raw,
            "legacy_paper_last3_vs_final_checkpoint": legacy_comparison,
            "statistics": statistics,
        },
        "ablations": ablations,
        "ablation_switch_documentation": {
            k: {"run_group": v["run_group"], "description": v["description"], "flags": v["flags"]}
            for k, v in ABLATION_SPECS.items()
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    md_path = args.out.with_suffix(".md")
    write_markdown_summary(payload, md_path)

    # Compact CSV of OGBench grand means + ablation means
    csv_out = args.out.with_name(args.out.stem + "_summary.csv")
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "name", "mean", "std_or_ci_low", "ci_high", "n", "notes"])
        for m in ("baseline", "v6", "v7", "v8", "v9"):
            ci = statistics[m]["method_bootstrap_ci"]
            w.writerow(
                [
                    "ogbench50_final_checkpoint_2m",
                    m,
                    raw_agg[m]["grand_mean"],
                    ci["ci_low"],
                    ci["ci_high"],
                    ci["n"],
                    "task-level bootstrap CI",
                ]
            )
        if not args.skip_ablations:
            for name, var in ablations["variants"].items():
                ci = var.get("bootstrap_ci") or {}
                w.writerow(
                    [
                        f"hl_ablation_{args.ablation_budget}",
                        name,
                        var.get("mean"),
                        ci.get("ci_low"),
                        ci.get("ci_high"),
                        var.get("n_complete_matched"),
                        var.get("run_group"),
                    ]
                )

    print(f"wrote {args.out}")
    print(f"wrote {md_path}")
    print(f"wrote {csv_out}")
    print("OGBench grand means (raw):")
    for m in ("baseline", "v6", "v7", "v8", "v9"):
        print(f"  {m}: {raw_agg[m]['grand_mean']:.6f}  incomplete={raw_agg[m]['incomplete_tasks']}")
    print(f"CSV↔raw all_match: {csv_vs_raw.get('all_match')}")
    if legacy_comparison is not None:
        print(
            "Legacy paper_last3 vs final_checkpoint all_match: "
            f"{legacy_comparison.get('all_match')} (expected False)"
        )
    if not args.skip_ablations:
        print("Ablations:")
        for name, var in ablations["variants"].items():
            print(
                f"  {name}: mean={var['mean']} n={var['n_complete_matched']} "
                f"group={var['run_group']}"
            )


if __name__ == "__main__":
    main()
