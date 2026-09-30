#!/usr/bin/env python3
"""Pick-18 mean metrics: V21 vs AWR vs PPO vs V22.24.

Mean ± std across the 18 tasks. Offline probe after AC pretrain occupies the
first 10% of the x-axis; online stored episodes start at 0 (rolling last-10 of
online only, no probe in the window). Curve is the equal-weight mean of all 18
tasks; it is NaN at an episode unless every task has a stored episode there.

V21 and V22.24 are the finished β=100 Pick-18 sweeps. AWR and PPO may still
be online.

Writes:
  runs/pick18/plots/onepass_vs_flow_sr.pdf|.png
  runs/pick18/plots/onepass_vs_flow_macro.pdf|.png
  runs/pick18/plots/onepass_vs_flow_sr.json
"""

from __future__ import annotations

import json
import re
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
PROBE_RE = re.compile(r"probe (\d+)/(\d+).*success=(\d+)")
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
    {
        "key": "v21",
        "run": ROOT / "runs" / "pick18",
        "tag": "{task}_v21_s0",
        "label": r"V21",
        "eval_suffix": "_actor",
        "log_dir": ROOT / "runs" / "pick18" / "logs",
    },
    {
        "key": "awr",
        "run": ROOT / "runs" / "pick18_awr",
        "tag": "{task}_awr_s0",
        "label": r"AWR",
        "eval_suffix": "_actor",
        "log_dir": ROOT / "runs" / "pick18_awr" / "logs",
    },
    {
        "key": "ppo",
        "run": ROOT / "runs" / "pick18_ppo",
        "tag": "{task}_ppo_s0",
        "label": r"PPO",
        "eval_suffix": "_actor",
        "log_dir": ROOT / "runs" / "pick18_ppo" / "logs",
    },
    {
        "key": "v22_24",
        "run": ROOT / "runs" / "pick18_v22_24",
        "tag": "{task}_v22_24_s0",
        "label": r"V22.24",
        "eval_suffix": "_gOn",
        "log_dir": ROOT / "runs" / "pick18_v22_24" / "logs",
    },
]
COLOR = {
    "v21": "#7f7f7f",
    "awr": "#e69f00",
    "ppo": "#d55e00",
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
    out = np.full(ONLINE_N, np.nan)
    n = min(len(succ), ONLINE_N)
    for i in range(n):
        w = succ[max(0, i - WINDOW + 1) : i + 1]
        out[i] = float(w.mean())
    return out


def probe_from_log(log_dir: Path, tag: str) -> float:
    path = log_dir / f"{tag}.log"
    if not path.exists():
        return float("nan")
    seen: dict[int, int] = {}
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return float("nan")
    for match in PROBE_RE.finditer(text):
        seen[int(match.group(1))] = int(match.group(3))
    if not seen:
        return float("nan")
    n = max(seen)
    if n <= 0:
        return float("nan")
    return float(sum(seen.values()) / n)


def load_probe(dest: Path, log_dir: Path, tag: str) -> float:
    summary = load_json(dest / "summary.json")
    if summary and summary.get("probe_episodes"):
        pe = int(summary["probe_episodes"])
        ps = int(summary.get("probe_successes") or 0)
        if pe > 0:
            return ps / pe
    return probe_from_log(log_dir, tag)


def load_heldout(run: Path, task: str, tag: str, eval_suffix: str) -> float:
    path = run / task / "eval" / f"{tag}{eval_suffix}" / "result.json"
    row = load_json(path)
    if not row or not row.get("episodes_run"):
        return float("nan")
    return float(row["success_rate"])


def load_arm(spec: dict) -> dict:
    probes = []
    online_rates = []
    last10s = []
    helds = []
    curves = []
    n_online = []
    per_task = []
    for task in TASKS:
        tag = spec["tag"].format(task=task)
        dest = spec["run"] / task / "rl" / tag
        probe_sr = load_probe(dest, spec["log_dir"], tag)
        succ = online_successes(dest / "metrics.jsonl")
        n = int(len(succ))
        online_rate = float(succ.mean()) if n else float("nan")
        last10 = float(succ[-WINDOW:].mean()) if n else float("nan")
        held = load_heldout(spec["run"], task, tag, spec["eval_suffix"])
        probes.append(probe_sr)
        online_rates.append(online_rate)
        last10s.append(last10)
        helds.append(held)
        n_online.append(n)
        curves.append(rolling_last10(succ))
        per_task.append(
            {
                "task": task,
                "probe": None if np.isnan(probe_sr) else round(probe_sr, 4),
                "online_n": n,
                "online": None if np.isnan(online_rate) else round(online_rate, 4),
                "last10": None if np.isnan(last10) else round(last10, 4),
                "heldout": None if np.isnan(held) else round(held, 4),
            }
        )
    stacked = np.stack(curves, axis=0)
    n_at = np.sum(~np.isnan(stacked), axis=0)
    mean = np.full(ONLINE_N, np.nan)
    std = np.full(ONLINE_N, np.nan)
    full = n_at == len(TASKS)
    if np.any(full):
        mean[full] = stacked[:, full].mean(axis=0)
        std[full] = stacked[:, full].std(axis=0, ddof=1)
    probes_a = np.asarray(probes, dtype=float)
    online_a = np.asarray(online_rates, dtype=float)
    last_a = np.asarray(last10s, dtype=float)
    held_a = np.asarray(helds, dtype=float)
    held_n = int(np.sum(~np.isnan(held_a)))
    held_mean = float(held_a.mean()) if held_n == len(TASKS) else float("nan")
    held_std = float(held_a.std(ddof=1)) if held_n == len(TASKS) else float("nan")
    return {
        "probe_mean": float(np.nanmean(probes_a)),
        "probe_std": float(np.nanstd(probes_a, ddof=1)),
        "probe_n": int(np.sum(~np.isnan(probes_a))),
        "online_mean": float(np.nanmean(online_a)),
        "online_std": float(np.nanstd(online_a, ddof=1)),
        "last10_mean": float(np.nanmean(last_a)),
        "last10_std": float(np.nanstd(last_a, ddof=1)),
        "heldout_mean": held_mean,
        "heldout_std": held_std,
        "heldout_n": held_n,
        "curve_mean": mean,
        "curve_std": std,
        "n_at": n_at,
        "n_online": n_online,
        "min_n": int(min(n_online) if n_online else 0),
        "complete_n": int(sum(n == ONLINE_N for n in n_online)),
        "probes": probes_a.tolist(),
        "per_task": per_task,
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
    mu, sd = arm["curve_mean"], arm["curve_std"]
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


def downsample(mu: np.ndarray, sd: np.ndarray, n_at: np.ndarray, step: int = 10) -> dict:
    idx = list(range(0, ONLINE_N, step))
    if idx[-1] != ONLINE_N - 1:
        idx.append(ONLINE_N - 1)
    return {
        "episode": idx,
        "mean": [None if np.isnan(mu[i]) else round(float(mu[i]), 4) for i in idx],
        "std": [None if np.isnan(sd[i]) else round(float(sd[i]), 4) for i in idx],
        "n": [int(n_at[i]) for i in idx],
    }


def finite(x: float) -> float | None:
    return None if x != x else round(float(x), 4)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = {}
    for spec in ARMS:
        loaded[spec["key"]] = load_arm(spec)
        a = loaded[spec["key"]]
        print(
            f"{spec['key']:8} probe {100 * a['probe_mean']:.1f}±{100 * a['probe_std']:.1f}%  "
            f"online {100 * a['online_mean']:.1f}±{100 * a['online_std']:.1f}%  "
            f"last-10 {100 * a['last10_mean']:.1f}±{100 * a['last10_std']:.1f}%  "
            f"held-out {100 * a['heldout_mean']:.1f}% ({a['heldout_n']}/18)  "
            f"complete {a['complete_n']}/{len(TASKS)} min_n={a['min_n']}"
        )

    fig, ax = plt.subplots(figsize=(6.4, 3.55))
    ax.axvspan(-OFFLINE_W, 0.0, color="#f2f2f2", zorder=0)
    ax.axvline(0.0, color="0.45", ls="--", lw=0.85, zorder=2)
    ax.text(-OFFLINE_W / 2.0, 1.02, "offline", ha="center", va="bottom", fontsize=8.0, color="0.35")
    ax.text(ONLINE_N / 2.0, 1.02, "online", ha="center", va="bottom", fontsize=8.0, color="0.35")
    for spec in ARMS:
        plot_arm(ax, loaded[spec["key"]], COLOR[spec["key"]], spec["label"])
    ax.set_xlim(-OFFLINE_W, ONLINE_N)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xticks([-OFFLINE_W / 2.0, 0, 100, 200, 300])
    ax.set_xticklabels(["probe", "0", "100", "200", "300"])
    ax.set_xlabel("online episode")
    ax.set_ylabel("success rate (last 10 ep.)")
    ax.set_title(r"Pick-18 last-10 SR (mean $\pm$ std across 18 tasks)", pad=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "onepass_vs_flow_sr.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "onepass_vs_flow_sr.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    metrics = ["probe", "online", "last-10", "held-out"]
    keys = ["probe_mean", "online_mean", "last10_mean", "heldout_mean"]
    x = np.arange(len(metrics), dtype=float)
    width = 0.18
    offsets = np.linspace(-1.5, 1.5, len(ARMS)) * width
    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    for spec, dx in zip(ARMS, offsets):
        a = loaded[spec["key"]]
        vals = [a[k] if a[k] == a[k] else np.nan for k in keys]
        err = []
        for k in keys:
            std_k = k.replace("_mean", "_std")
            std = a[std_k]
            err.append(0.0 if std != std or a[k] != a[k] else std)
        bars = ax.bar(
            x + dx,
            vals,
            width,
            yerr=err,
            color=COLOR[spec["key"]],
            label=spec["label"],
            capsize=2.0,
            error_kw={"elinewidth": 0.8, "ecolor": "0.25"},
            zorder=3,
        )
        for bar, val, key in zip(bars, vals, keys):
            if a[key] != a[key]:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                min(val + 0.03, 1.02),
                f"{100 * val:.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="0.25",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(["probe\n(offline)", "online SR\n(stored)", "last-10", "held-out\neval48"])
    ax.set_ylabel("success rate")
    ax.set_ylim(0.0, 1.12)
    ax.set_title(r"Pick-18 mean $\pm$ std across 18 tasks", pad=8)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.axhline(0.0, color="0.7", lw=0.6, zorder=1)
    ppo_h = loaded["ppo"]["heldout_n"]
    awr_min = loaded["awr"]["min_n"]
    ax.text(
        0.0,
        -0.22,
        rf"V21 / V22.24: $\beta{{=}}100$, 300 ep. + eval48. "
        rf"PPO online finished (held-out {ppo_h}/18). "
        rf"AWR online in progress (min {awr_min}/300). V22.24 held-out is $G$ on.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        color="0.4",
    )
    fig.tight_layout()
    fig.savefig(OUT / "onepass_vs_flow_macro.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "onepass_vs_flow_macro.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    dump = {
        "note": (
            "Equal-weight mean over 18 Pick-18 tasks. V21 and V22.24 are finished "
            "β=100 stage-0. PPO online is finished (300); held-out eval48 is partial. "
            "AWR is still online. Last-10 curves are defined only while all 18 "
            "tasks have a stored episode at that index. Held-out is eval48 (64 "
            "rollouts); V22.24 is guidance on (λ=0.5)."
        ),
        "offline_width_episodes": OFFLINE_W,
        "offline_frac": OFFLINE_FRAC,
        "online_episodes": ONLINE_N,
        "window": WINDOW,
        "n_tasks": len(TASKS),
        "arms": {},
    }
    for spec in ARMS:
        a = loaded[spec["key"]]
        dump["arms"][spec["key"]] = {
            "label": spec["label"].replace(r"$", "").replace("{", "").replace("}", ""),
            "probe_mean": finite(a["probe_mean"]),
            "probe_std": finite(a["probe_std"]),
            "online_mean": finite(a["online_mean"]),
            "online_std": finite(a["online_std"]),
            "last10_mean": finite(a["last10_mean"]),
            "last10_std": finite(a["last10_std"]),
            "heldout_mean": finite(a["heldout_mean"]),
            "heldout_std": finite(a["heldout_std"]),
            "heldout_n": a["heldout_n"],
            "complete_n": a["complete_n"],
            "min_n": a["min_n"],
            "n_online": a["n_online"],
            "online": downsample(a["curve_mean"], a["curve_std"], a["n_at"]),
            "per_task": a["per_task"],
        }
    (OUT / "onepass_vs_flow_sr.json").write_text(json.dumps(dump, indent=2) + "\n")
    print("wrote", OUT / "onepass_vs_flow_sr.png")
    print("wrote", OUT / "onepass_vs_flow_macro.png")


if __name__ == "__main__":
    main()
