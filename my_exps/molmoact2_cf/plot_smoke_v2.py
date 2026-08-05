"""Regenerate smoke_v2 success-rate curves from episode logs / metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


_STEP_RE = re.compile(
    r"valid_eps=(?P<ep>\d+).*success=(?P<success>True|False)"
    r".*g=(?P<g>True|False).*sr=(?P<sr>[0-9.]+)"
)


def parse_episode_rows(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _STEP_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "ep": int(m.group("ep")),
                "success": m.group("success") == "True",
                "g": m.group("g") == "True",
                "sr": float(m.group("sr")),
            }
        )
    # Keep last row per episode index (logs can repeat on resume).
    by_ep: dict[int, dict] = {}
    for row in rows:
        by_ep[row["ep"]] = row
    return [by_ep[k] for k in sorted(by_ep)]


def rolling(xs: list[float], w: int) -> np.ndarray:
    out = np.full(len(xs), np.nan, dtype=np.float64)
    for i in range(len(xs)):
        lo = max(0, i + 1 - w)
        out[i] = float(np.mean(xs[lo : i + 1]))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs/molmoact2_cf_smoke_v2",
    )
    args = p.parse_args()
    run_dir: Path = args.run_dir
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    online_rows = parse_episode_rows(run_dir / "online_short.log")
    g0 = json.loads((run_dir / "eval_g0/summary.json").read_text())
    gon = json.loads((run_dir / "eval_g_on/summary.json").read_text())
    g0_sr = float(g0["cumulative_success_rate"])
    gon_sr = float(gon["cumulative_success_rate"])

    eps = [r["ep"] for r in online_rows]
    succ = [1.0 if r["success"] else 0.0 for r in online_rows]
    cum = np.cumsum(succ) / np.arange(1, len(succ) + 1)
    roll5 = rolling(succ, 5)
    g_start = next((r["ep"] for r in online_rows if r["g"]), None)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=140)
    ax.plot(eps, cum, color="#1f4e79", lw=2.2, label="Online cumulative SR")
    ax.plot(eps, roll5, color="#2a9d8f", lw=1.8, ls="--", label="Online rolling 5-ep SR")
    ax.scatter(
        [e for e, s in zip(eps, succ) if s > 0.5],
        [c for c, s in zip(cum, succ) if s > 0.5],
        s=28,
        c="#1f4e79",
        zorder=3,
        label="Success ep",
    )
    ax.scatter(
        [e for e, s in zip(eps, succ) if s <= 0.5],
        [c for c, s in zip(cum, succ) if s <= 0.5],
        s=28,
        facecolors="none",
        edgecolors="#1f4e79",
        zorder=3,
        label="Fail ep",
    )
    ax.axhline(g0_sr, color="#c0392b", ls=":", lw=1.6, label=f"Fixed-seed G=0 ({g0_sr:.0%})")
    ax.axhline(gon_sr, color="#e67e22", ls=":", lw=1.6, label=f"Fixed-seed G-on ({gon_sr:.0%})")
    if g_start is not None:
        ax.axvline(g_start, color="#7f8c8d", ls="-.", lw=1.2, label=f"G enabled (ep {g_start})")
    ax.set_xlabel("Valid episodes")
    ax.set_ylabel("Success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, max(eps) + 1 if eps else 1)
    ax.set_title(
        f"MolmoAct2 + CF smoke_v2 — Pick success rate\n"
        f"final cum SR={cum[-1]:.1%} over {len(eps)} eps"
        if len(eps)
        else "MolmoAct2 + CF smoke_v2 — Pick success rate"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    out_png = plots / "sr_curves.png"
    fig.savefig(out_png)
    plt.close(fig)

    summary = {
        "eval_g0_sr": g0_sr,
        "eval_g_on_sr": gon_sr,
        "online_n_eps": len(eps),
        "online_cum_sr": float(cum[-1]) if len(cum) else 0.0,
        "n_successes": int(np.sum(succ)),
        "g_enabled_from_ep": g_start,
        "rolling5_final": float(roll5[-1]) if len(roll5) else 0.0,
        "early_sr_eps1_10": float(np.mean(succ[:10])) if len(succ) >= 10 else None,
        "late_sr_eps21_40": float(np.mean(succ[20:40])) if len(succ) >= 40 else None,
        "stop_reason": "smoke_cap_max_valid_episodes",
    }
    (plots / "sr_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"wrote": str(out_png), **summary}, indent=2))


if __name__ == "__main__":
    main()
