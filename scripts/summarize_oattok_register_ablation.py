#!/usr/bin/env python3
"""Summarize OATTok num_registers (K) ablation checkpoints.

Reads ``*_K{K}.pkl`` metas under a save dir, writes a JSON table and an optional
val_rmse vs K plot.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RQL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RQL_ROOT))


_K_RE = re.compile(r"_K(\d+)\.pkl$")


def _load_meta(path: Path) -> dict:
    with path.open("rb") as f:
        blob = pickle.load(f)
    if isinstance(blob, dict) and "meta" in blob:
        meta = dict(blob["meta"])
    elif isinstance(blob, dict):
        meta = {k: v for k, v in blob.items() if k != "params"}
    else:
        raise ValueError(f"Unrecognized tokenizer pickle: {path}")
    meta["path"] = str(path)
    if "num_registers" not in meta:
        m = _K_RE.search(path.name)
        if m:
            meta["num_registers"] = int(m.group(1))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=RQL_ROOT / "exp" / "ogbench_oattok_register_ablation",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="JSON summary path (default: <save-dir>/register_ablation_summary.json)",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional PNG path for val_rmse vs K",
    )
    args = parser.parse_args()
    save_dir = args.save_dir
    out_path = args.out or (save_dir / "register_ablation_summary.json")
    plot_path = args.plot

    rows = []
    for path in sorted(save_dir.glob("*_K*.pkl")):
        try:
            meta = _load_meta(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] skip {path.name}: {exc}", file=sys.stderr)
            continue
        rows.append(
            {
                "domain": meta.get("domain"),
                "num_registers": int(meta.get("num_registers", -1)),
                "val_rmse": float(meta.get("val_rmse", meta.get("final_rmse", np.nan))),
                "train_probe_rmse": float(meta.get("train_probe_rmse", np.nan)),
                "best_step": int(meta.get("best_step", -1)),
                "steps": int(meta.get("steps", -1)),
                "emb_dim": int(meta.get("emb_dim", -1)),
                "sample_dim": int(meta.get("sample_dim", -1)),
                "sample_horizon": int(meta.get("sample_horizon", -1)),
                "path": str(path),
            }
        )

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)

    winners = {}
    for domain, domain_rows in by_domain.items():
        domain_rows = sorted(domain_rows, key=lambda r: r["num_registers"])
        by_domain[domain] = domain_rows
        finite = [r for r in domain_rows if np.isfinite(r["val_rmse"])]
        if finite:
            best = min(finite, key=lambda r: r["val_rmse"])
            winners[domain] = {
                "best_num_registers": best["num_registers"],
                "best_val_rmse": best["val_rmse"],
                "path": best["path"],
            }

    summary = {
        "save_dir": str(save_dir),
        "n_checkpoints": len(rows),
        "by_domain": {k: v for k, v in by_domain.items()},
        "winners": winners,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")

    # Pretty table
    print("\n=== OATTok register ablation (val_rmse) ===")
    print(f"{'domain':<24} {'K':>4} {'val_rmse':>10} {'train_rmse':>10} {'best_step':>10}")
    for domain, domain_rows in sorted(by_domain.items()):
        for r in domain_rows:
            print(
                f"{domain:<24} {r['num_registers']:>4} "
                f"{r['val_rmse']:>10.4f} {r['train_probe_rmse']:>10.4f} "
                f"{r['best_step']:>10}"
            )
        if domain in winners:
            w = winners[domain]
            print(
                f"  -> best K={w['best_num_registers']} "
                f"val_rmse={w['best_val_rmse']:.4f}"
            )

    if plot_path is not None and rows:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for domain, domain_rows in sorted(by_domain.items()):
            ks = [r["num_registers"] for r in domain_rows]
            ys = [r["val_rmse"] for r in domain_rows]
            ax.plot(ks, ys, marker="o", linewidth=2, label=domain)
        ax.set_xlabel("num_registers (K discrete tokens)")
        ax.set_ylabel("val action RMSE")
        ax.set_title("OATTok register ablation on OGBench")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
