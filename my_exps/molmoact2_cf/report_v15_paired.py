#!/usr/bin/env python3
"""Summarize V15 held-out paired ΔSR from validation outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    if n <= 0:
        return 0.0, 0.0, 1.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return p, max(0.0, center - margin), min(1.0, center + margin)


def load_cell(out_dir: Path) -> tuple[int, int] | None:
    summary = out_dir / "validation_summary.json"
    metrics = out_dir / "metrics.jsonl"
    if summary.exists():
        payload = json.loads(summary.read_text())
        n = int(payload.get("valid_episodes") or payload.get("n") or 0)
        s = int(payload.get("successes") or 0)
        if n > 0:
            return s, n
    if metrics.exists():
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        if not rows:
            return None
        last = rows[-1]
        n = int(last.get("valid_episodes") or 0)
        s = int(last.get("successes") or 0)
        if n > 0:
            return s, n
    # Fall back to counting episode success files if present.
    successes = list(out_dir.glob("**/success.json")) + list(out_dir.glob("**/result.json"))
    if successes:
        ok = 0
        for path in successes:
            try:
                payload = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            if payload.get("success") or payload.get("succeeded"):
                ok += 1
        return ok, len(successes)
    return None


def main() -> None:
    root = Path(__file__).resolve().parent
    run = root / "runs" / "rlt_cf_v15_controlled" / "validation"
    arms = [
        "residual_vla_baseline",
        "residual_rlt_actor",
        "residual_vla_cf",
        "residual_rlt_cf",
        "flow_vla_baseline",
        "flow_rlt_actor",
        "flow_rlt_cf",
        "molmo_ae_lora_actor",
    ]
    episodes = [0, 100, 200, 400]
    report: dict = {"run": str(run), "cells": []}
    print(f"{'arm':28s} {'ep':>4} {'policy':16s} {'s/n':>6} {'sr':>6} {'wilson':>18}")
    for arm in arms:
        for ep in episodes:
            for policy in ("reference", "reference_noise", "actor", "actor_guide"):
                seed_dirs = sorted(
                    (run / arm / f"ep_{ep:06d}" / policy).glob("seed_*")
                )
                if not seed_dirs:
                    continue
                succ = 0
                n = 0
                for seed_dir in seed_dirs:
                    cell = load_cell(seed_dir)
                    if cell is None:
                        continue
                    s, nn = cell
                    succ += s
                    n += nn
                if n <= 0:
                    continue
                sr, lo, hi = wilson_ci(succ, n)
                row = {
                    "arm": arm,
                    "episode": ep,
                    "policy": policy,
                    "successes": succ,
                    "n": n,
                    "sr": sr,
                    "wilson_lo": lo,
                    "wilson_hi": hi,
                }
                report["cells"].append(row)
                print(
                    f"{arm:28s} {ep:4d} {policy:16s} {succ:2d}/{n:<3d} "
                    f"{sr:6.3f} [{lo:5.3f},{hi:5.3f}]"
                )

    # Paired deltas where both reference and actor exist for same arm/ep.
    print("\nPaired ΔSR (actor - reference):")
    by_key = {(c["arm"], c["episode"], c["policy"]): c for c in report["cells"]}
    deltas = []
    for arm, ep, _ in sorted({(c["arm"], c["episode"], None) for c in report["cells"]}):
        ref = by_key.get((arm, ep, "reference"))
        act = by_key.get((arm, ep, "actor"))
        noise = by_key.get((arm, ep, "reference_noise"))
        if ref and act:
            d = act["sr"] - ref["sr"]
            deltas.append({"arm": arm, "episode": ep, "delta_actor_ref": d})
            print(f"  {arm:28s} ep{ep}: actor-ref={d:+.3f}")
        if noise and act:
            d2 = act["sr"] - noise["sr"]
            deltas.append({"arm": arm, "episode": ep, "delta_actor_noise": d2})
            print(f"  {arm:28s} ep{ep}: actor-noise={d2:+.3f}")
    report["deltas"] = deltas

    out = root / "runs" / "rlt_cf_v15_controlled" / "plots" / "v15_paired_policy_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
