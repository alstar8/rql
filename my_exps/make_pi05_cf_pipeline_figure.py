#!/usr/bin/env python3
"""One-page figure for the π₀.₅ + CF fine-tune pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_DIR = Path(__file__).resolve().parent


def _box(ax, xy, w, h, text, *, fc="#f4f6f8", ec="#2c3e50", fontsize=9, lw=1.2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#1a1a1a",
        wrap=True,
    )
    return patch


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#34495e",
        )
    )


def main():
    fig = plt.figure(figsize=(11.5, 7.2), dpi=160)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        r"$\pi_{0.5}$ + ConsensusFlow target-task fine-tune",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color="#1a1a1a",
    )
    ax.text(
        0.5,
        0.91,
        "Freeze VLM  ·  LoRA/BC on action expert  ·  RL endpoint $G$  ·  sparse success critic",
        ha="center",
        va="top",
        fontsize=9,
        color="#555555",
    )

    # Architecture row
    ax.text(0.02, 0.84, "A. Deploy graph", fontsize=11, fontweight="bold")
    _box(ax, (0.02, 0.62), 0.16, 0.16, "obs + language", fc="#eef2f5")
    _box(ax, (0.22, 0.62), 0.18, 0.16, "VLM backbone\n(FROZEN)", fc="#dfe6ee", ec="#7f8c8d")
    _box(ax, (0.44, 0.62), 0.20, 0.16, "Action expert $v$\n+ LoRA (BC only)", fc="#d5ecd8", ec="#2e7d32")
    _box(ax, (0.68, 0.62), 0.14, 0.16, "Endpoint\n$G_a$", fc="#ffe0b2", ec="#e65100")
    _box(ax, (0.85, 0.62), 0.13, 0.16, "action\nchunk $a$", fc="#f5f5f5")
    _box(ax, (0.68, 0.42), 0.14, 0.14, "Q ensemble\n(+ CQL)", fc="#e1bee7", ec="#6a1b9a")

    _arrow(ax, (0.18, 0.70), (0.22, 0.70))
    _arrow(ax, (0.40, 0.70), (0.44, 0.70))
    _arrow(ax, (0.64, 0.70), (0.68, 0.70))
    _arrow(ax, (0.82, 0.70), (0.85, 0.70))
    _arrow(ax, (0.75, 0.56), (0.75, 0.62))
    ax.text(0.78, 0.55, "DPG / trust", fontsize=7, color="#6a1b9a")
    ax.text(0.54, 0.58, r"$a_v=T_v(h,z)$", fontsize=8, color="#2e7d32")
    ax.text(0.72, 0.80, r"$a=\mathrm{clip}(a_v+G)$", fontsize=8, color="#e65100")

    # Phases
    ax.text(0.02, 0.36, "B. Training phases (limited online)", fontsize=11, fontweight="bold")
    phases = [
        (0.02, "Phase 0", "Buffer hygiene\nstratify $\\mathcal{D}^+/\\mathcal{D}^-$\nsparse $r\\in\\{0,1\\}$", "#eceff1"),
        (0.26, "Phase 1", "Offline: critic + CQL\nLoRA BC on $v$\n$G\\equiv 0$ init", "#d5ecd8"),
        (0.50, "Phase 2", "Offline residual RL\ntrain $G$ + dual $\\delta$\nBC-only LoRA", "#ffe0b2"),
        (0.74, "Phase 3", "Budgeted online\nstratified replay\ngrow $\\delta$ cautiously", "#ffcdd2"),
    ]
    for x, title, body, color in phases:
        _box(ax, (x, 0.08), 0.22, 0.24, f"{title}\n{body}", fc=color, fontsize=8)

    for x0, x1 in ((0.24, 0.26), (0.48, 0.50), (0.72, 0.74)):
        _arrow(ax, (x0, 0.20), (x1, 0.20))

    ax.text(
        0.50,
        0.02,
        r"Metric: role-gap $\Delta=J(v,G)-J(v,0)$; guard $J(v,0)$ against LoRA forgetting",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    png = OUT_DIR / "pi05_cf_finetune_pipeline_figure.png"
    pdf = OUT_DIR / "pi05_cf_finetune_pipeline_figure.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
