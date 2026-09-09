#!/usr/bin/env python3
"""Pick-18 last-10 SR, β=100: V21 / V22 / V22+V23 / V22.24.

Mean ± std across the 18 tasks. Offline probe after AC pretrain occupies the
first 10% of the x-axis; online stored episodes start at 0 (rolling last-10 of
online only, no probe in the window).

Writes:
  runs/pick18/plots/v22_family_sr_curves.pdf|.png
  runs/pick18/plots/v22_family_sr_curves.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/molmoact2_cf")
OUT = ROOT / "runs" / "pick18" / "plots"
ONLINE_N = 300
OFFLINE_FRAC = 0.10
OFFLINE_W = ONLINE_N * OFFLINE_FRAC / (1.0 - OFFLINE_FRAC)
WINDOW = 10
TASKS = [
    "kettle",
    "remote",
    "ladle",
    "tissue",
    "spoon",
    "spatula",
    "desk_mug",
    "pot",
    "soap_dispenser",
    "spray_bottle",
    "cup",
    "shaker",
    "fork",
    "bottle",
    "fruit",
    "bowl",
    "knife",
    "box",
]
ARMS = [
    ("v21", ROOT / "runs" / "pick18", "{task}_v21_s0", r"V21"),
    ("v22", ROOT / "runs" / "pick18", "{task}_cf_ae_s0", r"V22"),
    ("v22v23", ROOT / "runs" / "pick18_v22v23", "{task}_cf_ae_s0", r"V22+V23"),
    ("v22_24", ROOT / "runs" / "pick18_v22_24", "{task}_v22_24_s0", r"V22.24"),
]
COLOR = {
    "v21": "#7f7f7f",
    "v22": "#d55e00",
    "v22v23": "#0072b2",
    "v22_24": "#009e73",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def online_successes(path: Path) -> np.ndarray:
    succ = []
    if not path.exists():
        return np.array([], dtype=float)
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "success" not in row:
            continue
        succ.append(float(row["success"]))
        if len(succ) >= ONLINE_N:
            break
    return np.asarray(succ, dtype=float)


def rolling_last10(succ: np.ndarray) -> np.ndarray:
    """Last-10 of stored online episodes only. Episode 0 is the first online ep."""
    out = np.full(ONLINE_N, np.nan)
    n = min(len(succ), ONLINE_N)
    for i in range(n):
        w = succ[max(0, i - WINDOW + 1) : i + 1]
        out[i] = float(w.mean())
    return out


def load_arm(run: Path, tag_t: str) -> dict:
    probes = []
    curves = []
    n_online = []
    for task in TASKS:
        tag = tag_t.format(task=task)
        dest = run / task / "rl" / tag
        summary = load_json(dest / "summary.json")
        pe = int(summary["probe_episodes"]) if summary and summary.get("probe_episodes") else 0
        ps = int(summary["probe_successes"]) if summary and summary.get("probe_successes") is not None else 0
        probe_sr = (ps / pe) if pe else np.nan
        probes.append(probe_sr)
        succ = online_successes(dest / "metrics.jsonl")
        n_online.append(len(succ))
        curves.append(rolling_last10(succ))
    stacked = np.stack(curves, axis=0)
    probes_a = np.asarray(probes, dtype=float)
    return {
        "probe_mean": float(np.nanmean(probes_a)),
        "probe_std": float(np.nanstd(probes_a, ddof=1)),
        "probe_n": int(np.sum(~np.isnan(probes_a))),
        "online_mean": np.nanmean(stacked, axis=0),
        "online_std": np.nanstd(stacked, axis=0, ddof=1),
        "online_n": np.sum(~np.isnan(stacked), axis=0),
        "end_mean": float(np.nanmean(stacked[:, -1])),
        "end_std": float(np.nanstd(stacked[:, -1], ddof=1)),
        "n_online": n_online,
        "probes": probes_a.tolist(),
    }


def plot_arm(ax, arm: dict, color: str, label: str) -> None:
    x_off = np.linspace(-OFFLINE_W, 0.0, 32)
    mu_p, sd_p = arm["probe_mean"], arm["probe_std"]
    ax.fill_between(
        x_off,
        np.clip(mu_p - sd_p, 0.0, 1.0),
        np.clip(mu_p + sd_p, 0.0, 1.0),
        color=color,
        alpha=0.16,
        lw=0,
        zorder=1,
    )
    ax.plot(x_off, np.full_like(x_off, mu_p), color=color, lw=1.8, zorder=3)
    x = np.arange(ONLINE_N, dtype=float)
    mu, sd = arm["online_mean"], arm["online_std"]
    ax.fill_between(
        x,
        np.clip(mu - sd, 0.0, 1.0),
        np.clip(mu + sd, 0.0, 1.0),
        color=color,
        alpha=0.16,
        lw=0,
        zorder=1,
    )
    ax.plot(x, mu, color=color, lw=1.9, label=label, zorder=3)
    ax.plot([0.0], [mu_p], marker="o", ms=4.0, color=color, zorder=4, clip_on=False)


def downsample(mu: np.ndarray, sd: np.ndarray, step: int = 10) -> dict:
    idx = list(range(0, ONLINE_N, step))
    if idx[-1] != ONLINE_N - 1:
        idx.append(ONLINE_N - 1)
    return {
        "episode": idx,
        "mean": [round(float(mu[i]), 4) for i in idx],
        "std": [round(float(sd[i]), 4) for i in idx],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for key, run, tag, lab in ARMS:
        loaded[key] = load_arm(run, tag)
        loaded[key]["label"] = lab
        n_ok = sum(n == ONLINE_N for n in loaded[key]["n_online"])
        print(
            f"{key:8} n_tasks={loaded[key]['probe_n']} complete_online={n_ok}/{len(TASKS)}  "
            f"probe {100 * loaded[key]['probe_mean']:.1f}±{100 * loaded[key]['probe_std']:.1f}%  "
            f"end last-10 {100 * loaded[key]['end_mean']:.1f}±{100 * loaded[key]['end_std']:.1f}%"
        )

    fig, ax = plt.subplots(figsize=(6.4, 3.55))
    ax.axvspan(-OFFLINE_W, 0.0, color="#f2f2f2", zorder=0)
    ax.axvline(0.0, color="0.45", ls="--", lw=0.85, zorder=2)
    ax.text(
        -OFFLINE_W / 2.0,
        1.02,
        "offline",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.35",
    )
    ax.text(
        ONLINE_N / 2.0,
        1.02,
        "online",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="0.35",
    )
    for key, _run, _tag, lab in ARMS:
        plot_arm(ax, loaded[key], COLOR[key], lab)
    ax.set_xlim(-OFFLINE_W, ONLINE_N)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks([-OFFLINE_W / 2.0, 0, 100, 200, 300])
    ax.set_xticklabels(["probe", "0", "100", "200", "300"])
    ax.set_xlabel("online episode")
    ax.set_ylabel("success rate (last 10 ep.)")
    ax.set_title(r"Pick-18, $\beta{=}100$: last-10 SR (mean $\pm$ std across 18 tasks)", pad=10)
    ax.legend(loc="lower right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "v22_family_sr_curves.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "v22_family_sr_curves.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    dump = {
        "beta": 100,
        "offline_width_episodes": OFFLINE_W,
        "offline_frac": OFFLINE_FRAC,
        "online_episodes": ONLINE_N,
        "window": WINDOW,
        "n_tasks": len(TASKS),
        "arms": {},
    }
    for key, _run, _tag, lab in ARMS:
        a = loaded[key]
        dump["arms"][key] = {
            "label": lab,
            "probe_mean": round(a["probe_mean"], 4),
            "probe_std": round(a["probe_std"], 4),
            "end_mean": round(a["end_mean"], 4),
            "end_std": round(a["end_std"], 4),
            "online": downsample(a["online_mean"], a["online_std"]),
        }
    (OUT / "v22_family_sr_curves.json").write_text(json.dumps(dump, indent=2) + "\n")
    print("wrote", OUT / "v22_family_sr_curves.png")


if __name__ == "__main__":
    main()
