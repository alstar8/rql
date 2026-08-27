#!/usr/bin/env python3
"""Pick-18 SR plots and metrics: frozen-VLA 100-traj collect vs V21 / V22 (cf_ae).

Compares beta=100 (`runs/pick18`) and beta=1 (`runs/pick18_beta1`). Writes:

  runs/pick18/plots/metrics.json
  runs/pick18/plots/metrics.md
  runs/pick18/plots/pick18_sr_heldout.pdf|.png
  runs/pick18/plots/pick18_sr_online.pdf|.png
  runs/pick18/plots/pick18_sr_curves.pdf|.png
  runs/pick18/plots/pick18_sr_macro.pdf|.png
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
SHORT = {
    "desk_mug": "mug",
    "soap_dispenser": "soap",
    "spray_bottle": "spray",
}
METHODS = ["v21", "cf_ae"]
SWEEPS = {
    "beta100": ROOT / "runs" / "pick18",
    "beta1": ROOT / "runs" / "pick18_beta1",
}
COLLECT = {
    "desk_mug": ROOT / "runs" / "beta1_1gpu" / "desk_mug" / "collect" / "collect_desk_mug_a" / "collect.json",
    "kettle": None,  # merged a+b below
}
C = {
    "collect": "#000000",
    "v21_100": "#7f7f7f",
    "cf_100": "#d55e00",
    "v21_1": "#56b4e9",
    "cf_1": "#0072b2",
}
LABEL = {
    "collect": r"frozen $\pi_{0.5}$ (100-traj collect)",
    "v21_100": r"V21 one-pass $\beta{=}100$",
    "cf_100": r"V22 cf_ae $\beta{=}100$",
    "v21_1": r"V21 one-pass $\beta{=}1$",
    "cf_1": r"V22 cf_ae $\beta{=}1$",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.4,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def metrics_series(path: Path) -> list[float]:
    if not path.exists():
        return []
    succ = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        succ.append(float(row.get("success", 0.0)))
    return succ


def last10(succ: list[float]) -> float | None:
    if not succ:
        return None
    w = succ[-10:]
    return float(np.mean(w))


def collect_sr(task: str) -> dict:
    if task == "kettle":
        a = load_json(ROOT / "runs" / "beta1_jitter_ac" / "kettle" / "collect" / "collect_kettle_a" / "collect.json")
        b = load_json(ROOT / "runs" / "beta1_jitter_ac" / "kettle" / "collect" / "collect_kettle_b" / "collect.json")
        n = int(a["episodes_run"]) + int(b["episodes_run"])
        s = int(a["successes"]) + int(b["successes"])
        return {"successes": s, "episodes": n, "rate": s / n if n else None}
    if task == "desk_mug":
        row = load_json(COLLECT["desk_mug"])
        n = int(row["episodes_run"])
        s = int(row["successes"])
        return {"successes": s, "episodes": n, "rate": s / n if n else None}
    row = load_json(ROOT / "runs" / "pick_objects" / task / "collect" / f"collect_{task}" / "collect.json")
    n = int(row["episodes_run"])
    s = int(row["successes"])
    return {"successes": s, "episodes": n, "rate": s / n if n else None}


def method_row(run: Path, task: str, method: str) -> dict:
    on_tag = f"{task}_{method}_s0"
    dest = run / task
    metrics = dest / "rl" / on_tag / "metrics.jsonl"
    summary = load_json(dest / "rl" / on_tag / "summary.json")
    progress = load_json(dest / "rl" / on_tag / "progress.json")
    evalp = load_json(dest / "eval" / f"{on_tag}_actor" / "result.json")
    succ = metrics_series(metrics)
    online_n = len(succ)
    online_s = int(sum(succ))
    if summary:
        online_n = int(summary.get("actor_episodes", online_n))
        online_s = int(summary.get("actor_successes", online_s))
    held = None
    if evalp and evalp.get("episodes_run"):
        held = {
            "successes": int(evalp["successes"]),
            "episodes": int(evalp["episodes_run"]),
            "rate": float(evalp["success_rate"]),
            "ci95": evalp.get("ci95"),
        }
    ep_done = online_n
    if progress and "episodes_done" in progress:
        ep_done = int(progress["episodes_done"])
    complete = bool(held) and ep_done >= 300
    return {
        "online_successes": online_s,
        "online_episodes": online_n,
        "online_rate": (online_s / online_n) if online_n else None,
        "sr_last10": last10(succ),
        "success_series": succ,
        "heldout": held,
        "episodes_done": ep_done,
        "complete": complete,
    }


def gather() -> dict:
    tasks = {}
    for task in TASKS:
        entry = {"collect": collect_sr(task), "beta100": {}, "beta1": {}}
        for sweep, run in SWEEPS.items():
            for method in METHODS:
                entry[sweep][method] = method_row(run, task, method)
        tasks[task] = entry
    return {"tasks": tasks, "task_order": TASKS}


def fmt_frac(s, n, rate) -> str:
    if s is None or n is None or not n:
        return "—"
    return f"{s}/{n} = {100 * rate:.1f}%"


def fmt_held(held) -> str:
    if not held:
        return "—"
    return fmt_frac(held["successes"], held["episodes"], held["rate"])


def write_metrics(data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    slim = {"task_order": data["task_order"], "tasks": {}}
    for task, entry in data["tasks"].items():
        slim["tasks"][task] = {
            "collect": entry["collect"],
            "beta100": {
                m: {k: v for k, v in entry["beta100"][m].items() if k != "success_series"}
                for m in METHODS
            },
            "beta1": {
                m: {k: v for k, v in entry["beta1"][m].items() if k != "success_series"}
                for m in METHODS
            },
        }
    (OUT / "metrics.json").write_text(json.dumps(slim, indent=2) + "\n")

    lines = [
        "# Pick-18 success rates",
        "",
        "Frozen $\\pi_{0.5}$ collect is 100 train-jitter trajectories per task. "
        "V21 = one-pass Gaussian RL-Token. V22 = cf_ae (flow composition + AE finetune). "
        "Online is 300 stored actor episodes (`gate_step=0`). Held-out is eval48, "
        "`--episodes 16` → 64 rollouts.",
        "",
        "## Held-out eval48",
        "",
        "| Task | 100-traj collect | V21 $\\beta{=}100$ | V22 $\\beta{=}100$ | V21 $\\beta{=}1$ | V22 $\\beta{=}1$ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    macros = {k: [0, 0] for k in ("collect", "v21_100", "cf_100", "v21_1", "cf_1")}
    n_complete = {k: 0 for k in macros}
    for task in TASKS:
        e = data["tasks"][task]
        c = e["collect"]
        v100 = e["beta100"]["v21"]["heldout"]
        c100 = e["beta100"]["cf_ae"]["heldout"]
        v1 = e["beta1"]["v21"]["heldout"]
        c1 = e["beta1"]["cf_ae"]["heldout"]
        lines.append(
            f"| {task} | {fmt_frac(c['successes'], c['episodes'], c['rate'])} "
            f"| {fmt_held(v100)} | {fmt_held(c100)} | {fmt_held(v1)} | {fmt_held(c1)} |"
        )
        macros["collect"][0] += c["successes"]
        macros["collect"][1] += c["episodes"]
        n_complete["collect"] += 1
        for key, held in (("v21_100", v100), ("cf_100", c100), ("v21_1", v1), ("cf_1", c1)):
            if held:
                macros[key][0] += held["successes"]
                macros[key][1] += held["episodes"]
                n_complete[key] += 1
    lines.append(
        "| **macro** | "
        + " | ".join(
            fmt_frac(macros[k][0], macros[k][1], macros[k][0] / macros[k][1] if macros[k][1] else None)
            + (f" ({n_complete[k]}/18)" if n_complete[k] < 18 else "")
            for k in ("collect", "v21_100", "cf_100", "v21_1", "cf_1")
        )
        + " |"
    )
    lines += [
        "",
        "## Online actor SR (300 stored episodes)",
        "",
        "| Task | V21 $\\beta{=}100$ | last-10 | V22 $\\beta{=}100$ | last-10 | V21 $\\beta{=}1$ | last-10 | V22 $\\beta{=}1$ | last-10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        e = data["tasks"][task]
        cells = []
        for sweep, method in (
            ("beta100", "v21"),
            ("beta100", "cf_ae"),
            ("beta1", "v21"),
            ("beta1", "cf_ae"),
        ):
            r = e[sweep][method]
            cells.append(fmt_frac(r["online_successes"], r["online_episodes"], r["online_rate"]))
            cells.append("—" if r["sr_last10"] is None else f"{100 * r['sr_last10']:.0f}%")
        lines.append(f"| {task} | " + " | ".join(cells) + " |")
    incomplete = []
    for task in TASKS:
        for sweep, method in (
            ("beta100", "v21"),
            ("beta100", "cf_ae"),
            ("beta1", "v21"),
            ("beta1", "cf_ae"),
        ):
            r = data["tasks"][task][sweep][method]
            if not r["complete"]:
                incomplete.append(f"{sweep}/{task}/{method} (ep {r['episodes_done']}, heldout={'yes' if r['heldout'] else 'no'})")
    lines += ["", "## Status", ""]
    if incomplete:
        lines.append("Incomplete:")
        for item in incomplete:
            lines.append(f"- {item}")
    else:
        lines.append("All 18 tasks × V21/V22 × $\\beta\\in\\{1,100\\}$ finished (36 + 36 evals).")
    (OUT / "metrics.md").write_text("\n".join(lines) + "\n")


def fig_heldout(data: dict) -> None:
    names = [SHORT.get(t, t) for t in TASKS]
    x = np.arange(len(TASKS))
    width = 0.16
    series = {
        "collect": [data["tasks"][t]["collect"]["rate"] for t in TASKS],
        "v21_100": [
            (data["tasks"][t]["beta100"]["v21"]["heldout"] or {}).get("rate") for t in TASKS
        ],
        "cf_100": [
            (data["tasks"][t]["beta100"]["cf_ae"]["heldout"] or {}).get("rate") for t in TASKS
        ],
        "v21_1": [(data["tasks"][t]["beta1"]["v21"]["heldout"] or {}).get("rate") for t in TASKS],
        "cf_1": [(data["tasks"][t]["beta1"]["cf_ae"]["heldout"] or {}).get("rate") for t in TASKS],
    }
    fig, ax = plt.subplots(figsize=(11.2, 3.15))
    offsets = {"collect": -2, "v21_100": -1, "cf_100": 0, "v21_1": 1, "cf_1": 2}
    for key, off in offsets.items():
        vals = [np.nan if v is None else v for v in series[key]]
        bars = ax.bar(x + off * width, vals, width=width, color=C[key], label=LABEL[key], zorder=3)
        if key == "collect":
            for b in bars:
                b.set_hatch("//")
                b.set_edgecolor("white")
                b.set_linewidth(0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("held-out success (eval48)")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.7, len(TASKS) - 0.3)
    ax.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Pick-18 held-out SR: 100-traj collect vs V21 / V22 at $\\beta{=}100$ and $\\beta{=}1$", pad=18)
    fig.tight_layout()
    fig.savefig(OUT / "pick18_sr_heldout.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "pick18_sr_heldout.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_online(data: dict) -> None:
    names = [SHORT.get(t, t) for t in TASKS]
    x = np.arange(len(TASKS))
    width = 0.18
    keys = [
        ("v21_100", "beta100", "v21"),
        ("cf_100", "beta100", "cf_ae"),
        ("v21_1", "beta1", "v21"),
        ("cf_1", "beta1", "cf_ae"),
    ]
    fig, ax = plt.subplots(figsize=(11.2, 3.05))
    for i, (key, sweep, method) in enumerate(keys):
        vals = []
        for t in TASKS:
            r = data["tasks"][t][sweep][method]["online_rate"]
            vals.append(np.nan if r is None else r)
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=C[key], label=LABEL[key], zorder=3)
    collect = [data["tasks"][t]["collect"]["rate"] for t in TASKS]
    ax.scatter(x, collect, marker="_", s=90, color=C["collect"], zorder=4, label=LABEL["collect"])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right")
    ax.set_ylabel("online actor SR (300 ep.)")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.7, len(TASKS) - 0.3)
    ax.legend(loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Pick-18 online SR (stored actor episodes) vs frozen-VLA collect", pad=18)
    fig.tight_layout()
    fig.savefig(OUT / "pick18_sr_online.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "pick18_sr_online.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def rolling(succ: list[float], k: int = 10) -> np.ndarray:
    out = np.full(len(succ), np.nan)
    for i in range(len(succ)):
        w = succ[max(0, i - k + 1) : i + 1]
        out[i] = float(np.mean(w))
    return out


def fig_curves(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.35), sharey=True)
    specs = [
        (axes[0], "beta100", r"$\beta{=}100$"),
        (axes[1], "beta1", r"$\beta{=}1$"),
    ]
    for ax, sweep, title in specs:
        for method, color, lab in (
            ("v21", C["v21_100"] if sweep == "beta100" else C["v21_1"], "V21 one-pass"),
            ("cf_ae", C["cf_100"] if sweep == "beta100" else C["cf_1"], "V22 cf_ae"),
        ):
            curves = []
            for task in TASKS:
                succ = data["tasks"][task][sweep][method]["success_series"]
                if len(succ) < 10:
                    continue
                curves.append(rolling(succ[:300], 10))
            if not curves:
                continue
            n = min(len(c) for c in curves)
            stacked = np.stack([c[:n] for c in curves])
            mean = np.nanmean(stacked, axis=0)
            lo = np.nanpercentile(stacked, 25, axis=0)
            hi = np.nanpercentile(stacked, 75, axis=0)
            x = np.arange(1, n + 1)
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(x, mean, color=color, lw=1.8, label=lab)
        ax.set_title(f"{title}: last-10 SR (macro mean, IQR band)", pad=4)
        ax.set_xlabel("online episode")
        ax.set_xlim(1, 300)
        ax.set_ylim(-0.03, 1.05)
        ax.legend(loc="lower right", frameon=False)
    axes[0].set_ylabel("success rate (last 10 ep.)")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT / "pick18_sr_curves.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "pick18_sr_curves.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def fig_macro(data: dict) -> None:
    def held_macro(sweep, method):
        s = n = 0
        for task in TASKS:
            held = data["tasks"][task][sweep][method]["heldout"]
            if not held:
                continue
            s += held["successes"]
            n += held["episodes"]
        return (s / n) if n else np.nan, s, n

    def online_macro(sweep, method):
        s = n = 0
        for task in TASKS:
            r = data["tasks"][task][sweep][method]
            if not r["online_episodes"]:
                continue
            s += r["online_successes"]
            n += r["online_episodes"]
        return (s / n) if n else np.nan, s, n

    cs = sum(data["tasks"][t]["collect"]["successes"] for t in TASKS)
    cn = sum(data["tasks"][t]["collect"]["episodes"] for t in TASKS)

    def pack(key, triple):
        rate, s, n = triple
        return key, rate, f"{s}/{n}"

    rows = [
        ("collect", cs / cn, f"{cs}/{cn}"),
        pack("v21_100", held_macro("beta100", "v21")),
        pack("cf_100", held_macro("beta100", "cf_ae")),
        pack("v21_1", held_macro("beta1", "v21")),
        pack("cf_1", held_macro("beta1", "cf_ae")),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.35))
    ax = axes[0]
    names = [LABEL[k] for k, _, _ in rows]
    vals = [v for _, v, _ in rows]
    cols = [C[k] for k, _, _ in rows]
    counts = [c for _, _, c in rows]
    bars = ax.bar(range(len(rows)), vals, color=cols, width=0.72, zorder=3)
    bars[0].set_hatch("//")
    bars[0].set_edgecolor("white")
    for i, (v, c) in enumerate(zip(vals, counts)):
        if not np.isnan(v):
            ax.text(i, v + 0.02, c, ha="center", fontsize=6.4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(["frozen\n$\\pi_{0.5}$", "V21\n$\\beta{=}100$", "V22\n$\\beta{=}100$", "V21\n$\\beta{=}1$", "V22\n$\\beta{=}1$"], fontsize=7.0)
    ax.set_ylabel("macro held-out SR")
    ax.set_ylim(0, 1.18)
    ax.set_title("(a) Held-out eval48, summed over tasks", pad=3)

    ax = axes[1]
    on_rows = [
        ("v21_100", online_macro("beta100", "v21")),
        ("cf_100", online_macro("beta100", "cf_ae")),
        ("v21_1", online_macro("beta1", "v21")),
        ("cf_1", online_macro("beta1", "cf_ae")),
    ]
    ax.axhline(cs / cn, color=C["collect"], ls="--", lw=1.0, zorder=1)
    ax.text(-0.45, cs / cn + 0.035, f"collect {100 * cs / cn:.1f}%", fontsize=6.4, color=C["collect"], ha="left")
    names = ["V21\n$\\beta{=}100$", "V22\n$\\beta{=}100$", "V21\n$\\beta{=}1$", "V22\n$\\beta{=}1$"]
    vals = [p[0] for _, p in on_rows]
    cols = [C[k] for k, _ in on_rows]
    counts = [f"{p[1]}/{p[2]}" for _, p in on_rows]
    ax.bar(range(4), vals, color=cols, width=0.68, zorder=3)
    for i, (v, c) in enumerate(zip(vals, counts)):
        ax.text(i, v + 0.02, c, ha="center", fontsize=6.4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=7.0)
    ax.set_ylabel("macro online SR")
    ax.set_ylim(0, 1.18)
    ax.set_title("(b) Online actor, summed over tasks", pad=3)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "pick18_sr_macro.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(OUT / "pick18_sr_macro.png", dpi=220, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    data = gather()
    OUT.mkdir(parents=True, exist_ok=True)
    write_metrics(data)
    fig_heldout(data)
    fig_online(data)
    fig_curves(data)
    fig_macro(data)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
