#!/usr/bin/env python3
"""Write RQL / ConsensusFlow / CF-noCRF metrics under the RQL validation protocol.

RQL validation protocol (see benchmarks/ogbench50_tasks.json):
  - 50 OGBench singletask tasks
  - 50 eval episodes per checkpoint
  - mean evaluation/success over the last three 100k-spaced checkpoints
    ending at the report budget (800k/900k/1M or 1.8M/1.9M/2.0M)
  - aggregate: per-(task, seed) → mean over seeds → mean over 50 tasks

Local CF / CF-noCRF runs only have seeds {0,1,2}; RQL published uses 4 seeds.
This script reports the matched local seed set (default 0 1 2) for all three
methods, and optionally an RQL-only 4-seed headline when seed 3 is available.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from ogbench50_config import expand_tasks  # noqa: E402
from plot_ogbench50_dflrql_vs_baseline import (  # noqa: E402
    aggregate_headline_stats,
    checkpoint_label,
    collect_multiseed_metrics,
    report_checkpoints,
)


def _wlt(a: np.ndarray, b: np.ndarray) -> dict[str, int]:
    """Win/loss/tie of a vs b on paired task means (a − b)."""
    win = loss = tie = 0
    for x, y in zip(a, b):
        d = float(x) - float(y)
        if abs(d) < 1e-12:
            tie += 1
        elif d > 0:
            win += 1
        else:
            loss += 1
    return {"win": win, "loss": loss, "tie": tie}


def _coverage(rows: list[dict], key: str, seeds: list[int]) -> dict:
    n_tasks = sum(1 for r in rows if int(r.get(f"{key}_n") or 0) > 0)
    n_full = sum(1 for r in rows if int(r.get(f"{key}_n") or 0) == len(seeds))
    return {
        "tasks_with_any_seed": n_tasks,
        "tasks_with_all_seeds": n_full,
        "n_seeds_requested": len(seeds),
    }


def _domain_means(rows: list[dict], key: str) -> dict[str, float]:
    by_domain: dict[str, list[float]] = {}
    for row in rows:
        m = row.get(f"{key}_mean")
        if m is None:
            continue
        by_domain.setdefault(row["domain_id"], []).append(float(m))
    return {d: float(np.mean(vs)) for d, vs in sorted(by_domain.items())}


def summarize_budget(
    save_dir: Path,
    report_step: int,
    seeds: list[int],
    baseline_log_root: Path,
    v6_log_root: Path,
    v9_log_root: Path,
) -> tuple[list[dict], dict]:
    tasks = expand_tasks()
    rows, per_seed = collect_multiseed_metrics(
        str(save_dir),
        tasks,
        seeds,
        baseline_log_root,
        v6_log_root,
        report_step=report_step,
        paper_protocol=True,
        last3_average=True,
        v9_log_root=v9_log_root,
        include_v9=True,
        include_nocrf=True,
    )
    # Drop v6 from the written view by ignoring it in the summary.
    headline = aggregate_headline_stats(per_seed)

    # Row-paired comparisons (only tasks where both methods have a mean).
    def row_paired(key_a: str, key_b: str) -> dict:
        aa, bb = [], []
        for row in rows:
            va = row.get(f"{key_a}_mean")
            vb = row.get(f"{key_b}_mean")
            if va is None or vb is None:
                continue
            aa.append(float(va))
            bb.append(float(vb))
        if not aa:
            return {"n": 0}
        a = np.asarray(aa)
        b = np.asarray(bb)
        return {
            "n": len(a),
            "mean_delta": float(np.mean(a - b)),
            "win_loss_tie": _wlt(a, b),
        }

    def method_block(csv_key: str, label: str) -> dict:
        mean, std = headline.get(csv_key, (float("nan"), float("nan")))
        seed_vals = [float(x) for x in per_seed.get(csv_key, [])]
        return {
            "label": label,
            "csv_key": csv_key,
            "grand_mean": None if np.isnan(mean) else float(mean),
            "seed_std": None if np.isnan(std) else float(std),
            "per_seed_50task_means": seed_vals,
            "coverage": _coverage(rows, csv_key, seeds),
            "domain_means": _domain_means(rows, csv_key),
        }

    summary = {
        "protocol": "rql_validation_last3",
        "protocol_definition": {
            "source": "benchmarks/ogbench50_tasks.json eval_protocol",
            "metric": "mean evaluation/success over last three 100k-spaced checkpoints",
            "checkpoints": list(report_checkpoints(report_step)),
            "checkpoint_label": checkpoint_label(report_step, last3=True),
            "eval_episodes": 50,
            "n_tasks": 50,
            "seeds": seeds,
            "aggregation": (
                "per (task, seed) last3 mean → mean over seeds → mean over 50 tasks; "
                "headline seed_std is std of per-seed 50-task means"
            ),
            "note": (
                "RQL paper reports 4 seeds; local CF / CF-noCRF currently have "
                "seeds {0,1,2}. Matched comparison uses the shared seed set."
            ),
        },
        "report_step": report_step,
        "methods": {
            "RQL": method_block("baseline", "RQL baseline"),
            "CF": method_block("v9", "ConsensusFlow (dflrql9)"),
            "CF_noCRF": method_block("v9_nocrf", "ConsensusFlow no-CRF"),
        },
        "comparisons": {
            "CF_minus_RQL": row_paired("v9", "baseline"),
            "CF_noCRF_minus_RQL": row_paired("v9_nocrf", "baseline"),
            "CF_minus_CF_noCRF": row_paired("v9", "v9_nocrf"),
        },
    }
    return rows, summary


def write_csv(rows: list[dict], path: Path, keys: tuple[str, ...]) -> None:
    keep_prefix = (
        "task_id",
        "domain_id",
        "env_name",
        "report_step",
        "report_checkpoints",
    )
    fieldnames: list[str] = []
    for row in rows[:1]:
        for k in row:
            if k in keep_prefix or any(k.startswith(p) for p in keys):
                fieldnames.append(k)
    # Stable order
    ordered = [k for k in keep_prefix if k in (rows[0] if rows else {})]
    for key in keys:
        ordered.extend(
            k
            for k in (rows[0] if rows else {})
            if k.startswith(key) and k not in ordered
        )
    # Also keep delta columns for the compared methods.
    for extra in ("delta_v9_mean", "delta_v9_nocrf_mean"):
        if rows and extra in rows[0] and extra not in ordered:
            ordered.append(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_markdown(summary: dict, path: Path) -> None:
    proto = summary["protocol_definition"]
    lines = [
        "# RQL validation protocol metrics",
        "",
        f"Generated: `{summary['generated_utc']}`",
        "",
        "## Protocol",
        "",
        f"- Checkpoints: **{proto['checkpoint_label']}** (`{proto['checkpoints']}`)",
        f"- Seeds: `{proto['seeds']}`",
        f"- Eval episodes / checkpoint: `{proto['eval_episodes']}`",
        f"- Tasks: `{proto['n_tasks']}`",
        f"- Aggregation: {proto['aggregation']}",
        "",
        f"> {proto['note']}",
        "",
        "## Grand means (seed-mean of 50-task means)",
        "",
        "| Method | Mean | Seed std | Coverage (all-seed tasks) |",
        "|---|---:|---:|---:|",
    ]
    for name in ("RQL", "CF", "CF_noCRF"):
        m = summary["methods"][name]
        cov = m["coverage"]
        lines.append(
            f"| {m['label']} | {m['grand_mean']:.6f} | {m['seed_std']:.6f} | "
            f"{cov['tasks_with_all_seeds']}/{cov['tasks_with_any_seed']} |"
        )
    lines += ["", "## Pairwise (task means)", ""]
    for cname, c in summary["comparisons"].items():
        wlt = c.get("win_loss_tie", {})
        lines.append(
            f"- **{cname}**: Δ={c.get('mean_delta'):+.6f}, "
            f"W/L/T={wlt.get('win')}/{wlt.get('loss')}/{wlt.get('tie')} "
            f"(n={c.get('n')})"
        )
    lines += ["", "## Domain means", "", "| Domain | RQL | CF | CF no-CRF |", "|---|---:|---:|---:|"]
    domains = sorted(
        set(summary["methods"]["RQL"]["domain_means"])
        | set(summary["methods"]["CF"]["domain_means"])
        | set(summary["methods"]["CF_noCRF"]["domain_means"])
    )
    for d in domains:
        lines.append(
            f"| {d} | "
            f"{summary['methods']['RQL']['domain_means'].get(d, float('nan')):.4f} | "
            f"{summary['methods']['CF']['domain_means'].get(d, float('nan')):.4f} | "
            f"{summary['methods']['CF_noCRF']['domain_means'].get(d, float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=SCRIPT_DIR.parent / "exp",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--report-steps",
        type=int,
        nargs="+",
        default=[1_000_000, 2_000_000],
        help="Budgets at which to apply last3 (default: 1M JSON protocol + 2M suite).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=SCRIPT_DIR.parent / "my_exps",
    )
    parser.add_argument(
        "--baseline-log-root",
        type=Path,
        default=SCRIPT_DIR.parent / "exp" / "baseline_50_2000000",
    )
    parser.add_argument(
        "--v6-log-root",
        type=Path,
        default=SCRIPT_DIR.parent / "exp" / "dflrql6_50_2000000",
    )
    parser.add_argument(
        "--v9-log-root",
        type=Path,
        default=SCRIPT_DIR.parent / "exp" / "dflrql9_50_2000000",
    )
    args = parser.parse_args()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    combined: dict = {
        "generated_utc": generated,
        "protocol": "rql_validation_last3",
        "seeds": args.seeds,
        "budgets": {},
    }

    for step in args.report_steps:
        tag = f"{step // 1000}k"
        print(f"=== RQL last3 @ {tag} seeds={args.seeds} ===")
        rows, summary = summarize_budget(
            args.save_dir,
            step,
            args.seeds,
            args.baseline_log_root,
            args.v6_log_root,
            args.v9_log_root,
        )
        summary["generated_utc"] = generated
        csv_path = args.out_dir / f"ogbench50_rql_protocol_last3_{tag}.csv"
        json_path = args.out_dir / f"ogbench50_rql_protocol_last3_{tag}.json"
        md_path = args.out_dir / f"ogbench50_rql_protocol_last3_{tag}.md"
        write_csv(rows, csv_path, ("baseline", "v9", "v9_nocrf"))
        json_path.write_text(json.dumps(summary, indent=2) + "\n")
        write_markdown(summary, md_path)
        combined["budgets"][tag] = {
            "report_step": step,
            "checkpoint_label": summary["protocol_definition"]["checkpoint_label"],
            "methods": {
                k: {
                    "grand_mean": v["grand_mean"],
                    "seed_std": v["seed_std"],
                    "coverage": v["coverage"],
                }
                for k, v in summary["methods"].items()
            },
            "comparisons": summary["comparisons"],
            "csv": str(csv_path),
            "json": str(json_path),
            "md": str(md_path),
        }
        for name, block in summary["methods"].items():
            print(
                f"  {name}: {block['grand_mean']:.6f} ± {block['seed_std']:.6f} "
                f"(all-seed tasks {block['coverage']['tasks_with_all_seeds']}/50)"
            )
        for cname, c in summary["comparisons"].items():
            wlt = c["win_loss_tie"]
            print(
                f"  {cname}: Δ={c['mean_delta']:+.6f} "
                f"W/L/T={wlt['win']}/{wlt['loss']}/{wlt['tie']}"
            )
        print(f"  wrote {csv_path}")
        print(f"  wrote {json_path}")
        print(f"  wrote {md_path}")

    # Optional RQL-only 4-seed headline at 2M when seed 3 exists.
    if 2_000_000 in args.report_steps:
        rql_seeds = [0, 1, 2, 3]
        print(f"=== RQL-only 4-seed last3 @ 2000k seeds={rql_seeds} ===")
        rows4, summary4 = summarize_budget(
            args.save_dir,
            2_000_000,
            rql_seeds,
            args.baseline_log_root,
            args.v6_log_root,
            args.v9_log_root,
        )
        rql4 = summary4["methods"]["RQL"]
        combined["rql_only_4seed_2000k"] = {
            "seeds": rql_seeds,
            "grand_mean": rql4["grand_mean"],
            "seed_std": rql4["seed_std"],
            "coverage": rql4["coverage"],
            "note": (
                "RQL paper seed count; CF/CF-noCRF lack seed 3 so this is "
                "RQL-only reference, not a matched three-way compare."
            ),
        }
        print(
            f"  RQL@4seeds: {rql4['grand_mean']:.6f} ± {rql4['seed_std']:.6f} "
            f"(all-seed tasks {rql4['coverage']['tasks_with_all_seeds']}/50)"
        )

    out = args.out_dir / "ogbench50_rql_protocol_last3_summary.json"
    out.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
