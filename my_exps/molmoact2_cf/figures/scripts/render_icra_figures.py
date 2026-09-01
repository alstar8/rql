#!/usr/bin/env python3
"""Camera-ready ICRA figures: architecture + training, with real env stills."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage, TextArea, VPacker
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath
from PIL import Image


FIG_ROOT = Path(__file__).resolve().parent.parent
ENV = FIG_ROOT / "env"
OUT = FIG_ROOT / "icra"

COLORS = {
    "frozen": "#3B82C4",
    "train": "#E07A3D",
    "ae": "#0F9D7A",
    "critic": "#7C3AED",
    "data": "#C4A35A",
    "ink": "#172033",
    "muted": "#475569",
    "rule": "#CBD5E1",
    "white": "#FFFFFF",
}


def tint(color: str, white: float = 0.90) -> tuple[float, float, float]:
    r, g, b = mpl.colors.to_rgb(color)
    return (r * (1 - white) + white, g * (1 - white) + white, b * (1 - white) + white)


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 8.2,
            "axes.linewidth": 0.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def box(ax, x, y, w, h, color, *, dashed=False, fill=0.92, lw=1.15, radius=0.35, z=1.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=tint(color, fill),
        edgecolor=color,
        linewidth=lw,
        linestyle=(0, (3.5, 2.4)) if dashed else "solid",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def txt(ax, x, y, s, *, size=8, weight="medium", color=None, ha="center", va="center", z=5):
    ax.text(
        x,
        y,
        s,
        fontsize=size,
        fontweight="bold" if weight == "bold" else "normal",
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        zorder=z,
        linespacing=1.22,
    )


def arrow(ax, pts, *, color=None, dashed=False, lw=1.15, ms=9, z=2):
    path = MplPath(list(pts), [MplPath.MOVETO] + [MplPath.LINETO] * (len(pts) - 1))
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=(0, (3.5, 2.4)) if dashed else "solid",
        color=color or COLORS["muted"],
        capstyle="round",
        joinstyle="round",
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)


def photo(ax, path: Path, xy, zoom: float, *, label: str | None = None):
    img = Image.open(path).convert("RGB")
    artist = OffsetImage(img, zoom=zoom)
    if label:
        packed = VPacker(
            children=[
                TextArea(
                    label,
                    textprops={"fontsize": 6.2, "fontweight": "bold", "color": COLORS["ink"]},
                ),
                artist,
            ],
            align="center",
            pad=0.4,
            sep=1.5,
        )
        ab = AnnotationBbox(packed, xy, frameon=False, zorder=6)
    else:
        ab = AnnotationBbox(
            artist,
            xy,
            frameon=True,
            pad=0.12,
            bboxprops={"edgecolor": COLORS["rule"], "linewidth": 0.8},
            zorder=6,
        )
    ax.add_artist(ab)


def chip(ax, x, y, w, h, text, color):
    box(ax, x, y, w, h, color, fill=0.78, lw=0.7, radius=0.2, z=4)
    txt(ax, x + w / 2, y + h / 2, text, size=6.0, weight="bold")


def render_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.2, 7.15), dpi=200)
    ax.set_xlim(0, 162)
    ax.set_ylim(0, 71.5)
    ax.axis("off")

    # Observation
    box(ax, 1.2, 6.5, 26.5, 62, COLORS["data"], fill=0.94, dashed=True, radius=0.55)
    txt(ax, 14.4, 65.6, "Observation  $s$  (chunk start)", size=9.2, weight="bold")
    photo(ax, ENV / "head_reach.png", (14.4, 51.6), 0.28, label="Head")
    photo(ax, ENV / "left_wrist_reach.png", (14.4, 33.4), 0.28, label="Left wrist")
    photo(ax, ENV / "right_wrist_press.png", (14.4, 15.4), 0.28, label="Right wrist")
    box(ax, 3.0, 7.6, 23.0, 6.2, COLORS["data"], fill=0.82, radius=0.25)
    txt(
        ax,
        14.5,
        10.7,
        "Turn on the radio receiver\nthat's on the table\nin the living room.",
        size=6.0,
    )

    # Frozen VLM
    box(ax, 30.0, 6.5, 24.8, 62, COLORS["frozen"], dashed=True, fill=0.94, radius=0.55)
    chip(ax, 31.0, 63.6, 10.2, 3.2, "frozen", COLORS["frozen"])
    txt(ax, 48.8, 65.2, r"PaliGemma  $\pi_{0.5}$", size=8.6, weight="bold", ha="right")
    box(ax, 31.6, 38.5, 21.6, 22.5, COLORS["frozen"], fill=0.86)
    txt(ax, 42.4, 56.2, "PaliGemma", size=8.2, weight="bold")
    txt(ax, 42.4, 52.6, "gemma_2b  ·  frozen", size=6.4, color=COLORS["muted"])
    txt(ax, 42.4, 46.4, "3 RGB + prompt\n→ prefix tokens", size=6.6)
    box(ax, 31.6, 22.0, 21.6, 14.0, COLORS["frozen"], fill=0.88, dashed=True)
    txt(ax, 42.4, 29.0, "prefix KV cache\nfor the expert", size=6.6)
    box(ax, 31.6, 8.4, 21.6, 11.2, COLORS["frozen"], fill=0.86, dashed=True)
    txt(ax, 42.4, 14.0, "proprio  23-d\n(pad → 32)  for Q", size=6.5)

    arrow(ax, [(27.8, 48.0), (30.0, 48.0)], color=COLORS["frozen"])

    # Action sandwich + LoRA actor
    box(ax, 57.2, 6.5, 44.4, 62, COLORS["train"], dashed=True, fill=0.94, radius=0.55)
    chip(ax, 58.4, 63.6, 28.8, 3.2, "LoRA actor  ·  one-way sandwich", COLORS["train"])
    txt(
        ax,
        79.4,
        60.0,
        "LoRA rank 32 / α 32 on attn+FFN only  ·  width 1024  ·  B=0  ·  lora_scale 0→1",
        size=6.1,
        color=COLORS["muted"],
    )

    box(ax, 59.0, 51.6, 18.0, 6.6, COLORS["frozen"], fill=0.86)
    txt(ax, 68.0, 54.9, "action_in_proj\n32 → 1024  frozen", size=6.2, weight="bold")
    box(ax, 80.6, 51.6, 18.8, 6.6, COLORS["frozen"], fill=0.86)
    txt(ax, 90.0, 54.9, "time_mlp  (adaRMS)\n$t$ → cond  frozen", size=6.2, weight="bold")

    box(ax, 59.0, 38.6, 40.4, 11.4, COLORS["train"], fill=0.86)
    txt(ax, 79.2, 46.4, "gemma_300m  +  LoRA", size=8.0, weight="bold")
    txt(ax, 79.2, 42.0, "base weights frozen  ·  LoRA on attn/FFN\nprefix KV from PaliGemma", size=6.2)

    box(ax, 59.0, 30.4, 40.4, 6.6, COLORS["frozen"], fill=0.86)
    txt(ax, 79.2, 33.7, "action_out_proj   1024 → 32-d velocity $v_t$   frozen", size=6.3, weight="bold")

    arrow(ax, [(68.0, 51.6), (68.0, 50.0)], color=COLORS["frozen"], ms=8)
    arrow(ax, [(90.0, 51.6), (90.0, 50.0)], color=COLORS["frozen"], ms=8)
    arrow(ax, [(79.2, 38.6), (79.2, 37.0)], color=COLORS["frozen"], ms=8)
    arrow(ax, [(54.8, 29.0), (57.2, 29.0)], color=COLORS["frozen"], dashed=True)

    box(ax, 59.0, 19.6, 19.4, 9.0, COLORS["frozen"], fill=0.88, dashed=True)
    txt(ax, 68.7, 24.1, r"$\tilde{a}=\mathrm{Euler}(v_{\mathrm{frozen}},\varepsilon)$" "\n10 steps  stop-grad", size=6.0)
    box(ax, 80.0, 19.6, 19.4, 9.0, COLORS["train"], fill=0.86)
    txt(ax, 89.7, 24.1, r"$a=\mathrm{Euler}(v_{\mathrm{live}},\varepsilon)$" "\nsame $\\varepsilon$  live grad", size=6.0)

    box(ax, 59.0, 8.2, 40.4, 9.8, COLORS["train"], fill=0.90, dashed=True)
    txt(
        ax,
        79.2,
        13.1,
        r"$L_{\pi}=\lambda_{\pi}(-\min_k Q_k)+\beta\|a-\tilde{a}\|^2+L_{\mathrm{bc}}$"
        "\n"
        r"$\beta=100$ per-dim   ·   $\lambda_{\pi}=0$ until online"
        "\nprojections frozen so lora_scale=0 is an exact pt12 copy",
        size=6.0,
    )

    # Token AE
    box(ax, 104.0, 6.5, 26.8, 62, COLORS["ae"], dashed=True, fill=0.94, radius=0.55)
    chip(ax, 105.2, 63.6, 14.5, 3.2, "AE  /  RL token", COLORS["ae"])
    txt(ax, 117.4, 65.2, r"$z$  for Q only", size=9.0, weight="bold")
    box(ax, 106.0, 44.0, 22.8, 16.5, COLORS["ae"], fill=0.86)
    txt(ax, 117.4, 56.6, "CFTokenPool", size=8.0, weight="bold")
    txt(ax, 117.4, 51.0, "256-d · 4 heads · 2 layers\n8 queries → pooled $z$", size=6.4)
    box(ax, 106.0, 25.5, 22.8, 15.5, COLORS["ae"], fill=0.86, dashed=True)
    txt(ax, 117.4, 36.8, "CFTokenDecoder", size=8.0, weight="bold")
    txt(ax, 117.4, 31.2, r"$L_{\mathrm{ro}}=\mathrm{MSE}(\hat{p},\mathrm{sg}(\mathrm{prefix}))$", size=6.4)
    box(ax, 106.0, 8.4, 22.8, 14.2, COLORS["frozen"], fill=0.88, dashed=True)
    txt(ax, 117.4, 17.8, "critic state", size=7.4, weight="bold")
    txt(ax, 117.4, 12.4, r"$\phi(s)=[\mathrm{sg}(\mathrm{target\_pool}),\mathrm{proprio}]$" "\nonline: freeze_pool = 1", size=6.2)

    arrow(ax, [(54.8, 46.4), (54.8, 61.8), (104.0, 61.8), (104.0, 52.2)], color=COLORS["ae"], dashed=True)

    # Critic
    box(ax, 133.2, 6.5, 27.4, 62, COLORS["critic"], dashed=True, fill=0.94, radius=0.55)
    chip(ax, 134.4, 63.6, 16.8, 3.2, "chunk critic", COLORS["critic"])
    txt(ax, 146.9, 65.2, r"$Q(s,\mathrm{flatten}(a))$", size=9.0, weight="bold")
    box(ax, 135.0, 42.5, 23.8, 17.8, COLORS["critic"], fill=0.86)
    txt(ax, 146.9, 56.4, "CFTrunk  1536", size=8.0, weight="bold")
    txt(ax, 146.9, 50.2, "input  $[z,\\;a]$   1024+32+736\n10 heads   $Q=\\min_k Q_k$", size=6.4)
    box(ax, 135.0, 24.0, 23.8, 15.8, COLORS["critic"], fill=0.86)
    txt(ax, 146.9, 35.4, "TD  (live)", size=8.0, weight="bold")
    txt(
        ax,
        146.9,
        29.2,
        r"$y=\mathrm{clip}(r+\gamma(1-d)Q^-(s',a'),0,1)$"
        "\n"
        r"$\gamma=0.99$  per chunk, not $\gamma^{32}$",
        size=6.1,
    )
    box(ax, 135.0, 8.4, 23.8, 13.0, COLORS["frozen"], fill=0.88, dashed=True)
    txt(ax, 146.9, 16.8, "bootstrap $a'(s')$", size=7.2, weight="bold")
    txt(ax, 146.9, 12.0, "EMA LoRA  0.999\nactor uses sg(critic)", size=6.2)

    arrow(ax, [(101.6, 24.1), (104.0, 24.1), (104.0, 14.0), (133.2, 14.0)], color=COLORS["train"])
    arrow(ax, [(128.8, 14.8), (133.2, 14.8)], color=COLORS["ae"], dashed=True)

    # legend
    box(ax, 1.2, 0.6, 159.4, 5.0, COLORS["rule"], fill=0.97, lw=0.6, radius=0.25)
    txt(
        ax,
        81.0,
        3.1,
        "ice-blue  frozen (VLM, expert, projections, target pool, EMA LoRA)     "
        "coral  trainable (live LoRA, critic; pool only in AE)     "
        "dashed  stop-grad / auxiliary     "
        "solid  forward",
        size=6.4,
        color=COLORS["muted"],
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_training(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.2, 7.15), dpi=200)
    ax.set_xlim(0, 162)
    ax.set_ylim(0, 71.5)
    ax.axis("off")

    stages = [
        (1.2, COLORS["data"], "0   Data", "100 specialist rollouts"),
        (41.4, COLORS["ae"], "1   AE pretrain", "8000 steps"),
        (81.6, COLORS["frozen"], "2   AC pretrain", "8000 steps"),
        (121.8, COLORS["train"], "3   Online RL", "50 rounds"),
    ]
    for x, color, title, sub in stages:
        box(ax, x, 18.5, 38.8, 51.5, color, fill=0.94, radius=0.55)
        chip(ax, x + 1.2, 64.4, 22.5, 3.6, title, color)
        txt(ax, x + 31.5, 66.2, sub, size=6.6, color=COLORS["muted"], ha="right")

    # stage 0
    x = 1.2
    photo(ax, ENV / "cameras_approach.png", (20.6, 52.8), 0.18)
    box(ax, x + 2.0, 32.5, 34.8, 10.5, COLORS["data"], fill=0.86)
    txt(ax, x + 19.4, 39.6, "Replay  $\\mathcal{B}$", size=8.0, weight="bold")
    txt(ax, x + 19.4, 35.6, "9650 chunks  ·  32-step grid\ninstances 0–9  ·  50/50 success/fail", size=6.3)
    box(ax, x + 2.0, 21.0, 34.8, 10.0, COLORS["data"], fill=0.88, dashed=True)
    txt(
        ax,
        x + 19.4,
        26.0,
        r"$s$  3 RGB + proprio + prompt"
        "\n"
        r"$a\in\mathbb{R}^{32\times 23}$   $r\in\{0,1\}$   $s'$",
        size=6.3,
    )

    # stage 1
    x = 41.4
    box(ax, x + 3.5, 46.0, 31.8, 15.5, COLORS["ae"], fill=0.86)
    txt(ax, x + 19.4, 57.4, "token AE only", size=8.0, weight="bold")
    txt(ax, x + 19.4, 51.8, r"$L_{\mathrm{ro}}$  on sg(prefix tokens)" "\nencoder + decoder   ·   no Euler, no TD", size=6.4)
    box(ax, x + 3.5, 28.0, 31.8, 15.5, COLORS["ae"], fill=0.90, dashed=True)
    txt(
        ax,
        x + 19.4,
        35.8,
        "ae_only = 1\nactor_coef = 0\nfreeze_critic = 1   freeze_pool = 0",
        size=6.5,
    )
    chip(ax, x + 8.5, 21.5, 22.0, 4.2, "RL token $z$ ready", COLORS["ae"])

    # stage 2
    x = 81.6
    box(ax, x + 3.5, 46.0, 31.8, 15.5, COLORS["critic"], fill=0.88)
    txt(ax, x + 19.4, 57.4, "critic TD", size=8.0, weight="bold")
    txt(ax, x + 19.4, 51.8, r"$Q$ on $\mathrm{sg}(\mathrm{target\_pool}(s))$" "\n10 heads  ·  $a+\\varepsilon$  ($\\sigma=0.08$)", size=6.4)
    box(ax, x + 3.5, 31.0, 31.8, 12.5, COLORS["train"], fill=0.90, dashed=True)
    txt(ax, x + 19.4, 37.2, "LoRA BC + anchor\nactor_coef = 0  →  stays a pt12 copy", size=6.4)
    box(ax, x + 3.5, 21.0, 31.8, 7.6, COLORS["rule"], fill=0.92, dashed=True)
    txt(ax, x + 19.4, 24.8, "Probe: 1 ep on 308, not in replay", size=6.4)

    # stage 3
    x = 121.8
    photo(ax, ENV / "cameras_press.png", (141.2, 55.2), 0.14)
    box(ax, x + 2.2, 36.5, 34.4, 8.8, COLORS["train"], fill=0.86)
    txt(ax, x + 19.4, 40.9, "collect 1 ep  →  train $5\\times n_{\\mathrm{chunks}}$", size=6.6, weight="bold")
    box(ax, x + 2.2, 21.0, 34.4, 13.8, COLORS["train"], fill=0.90, dashed=True)
    txt(
        ax,
        x + 19.4,
        27.9,
        "publish EMA LoRA  (cf_live.npz)\nserve + client  instance 308\nactor_coef = 1   freeze_pool = 1\nreplay = 100 pretrain + online",
        size=6.2,
    )

    for x0, x1 in ((40.0, 41.4), (80.2, 81.6), (120.4, 121.8)):
        arrow(ax, [(x0, 45.0), (x1, 45.0)], color=COLORS["ink"], lw=1.35, ms=11)

    box(ax, 1.2, 1.0, 159.4, 15.8, COLORS["critic"], fill=0.95, radius=0.4)
    txt(ax, 81.0, 14.2, "Joint loss  (every stage;  $\\lambda_\\pi$  gates only  $-Q$)", size=8.0, weight="bold")
    txt(
        ax,
        81.0,
        8.4,
        r"$L = 4\,L_{\mathrm{td}} + \lambda_{\pi}(-\mathrm{mean}\,\min_k Q_k(s,a))"
        r" + \beta\|a-\tilde{a}\|^2 + L_{\mathrm{bc}} + L_{\mathrm{ro}}$",
        size=8.4,
    )
    txt(
        ax,
        81.0,
        3.4,
        r"$\lambda_{\pi}=0$ in AE / AC pretrain,  $1$ online."
        "   Anchor $\\beta=100$ and BC stay on."
        "   Kill if $|Q|>2$ or actor–ref RMSE $>0.04$.",
        size=6.6,
        color=COLORS["muted"],
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_filmstrip(out: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(10.8, 8.6), dpi=200)
    rows = (
        ("Approach  (chunk start)", "approach"),
        ("Reach  (mid episode)", "reach"),
        ("Press  (success chunk)", "press"),
    )
    cols = (
        ("Head", "head"),
        ("Left wrist", "left_wrist"),
        ("Right wrist", "right_wrist"),
    )
    for r, (row_title, stem) in enumerate(rows):
        for c, (col_title, cam) in enumerate(cols):
            ax = axes[r][c]
            img = Image.open(ENV / f"{cam}_{stem}.png").convert("RGB")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(COLORS["rule"])
                spine.set_linewidth(0.8)
            if r == 0:
                ax.set_title(col_title, fontsize=10, fontweight="bold", color=COLORS["ink"], pad=6)
            if c == 0:
                ax.set_ylabel(row_title, fontsize=8.5, fontweight="bold", color=COLORS["ink"])
    fig.suptitle(
        "BEHAVIOR-1K  turning_on_radio   ·   R1Pro   ·   instance 308 specialist rollout",
        fontsize=11,
        fontweight="bold",
        color=COLORS["ink"],
        y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0.01, 1.0, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


# cf_v23_ae2 online rounds 1–50 on instance 308. Source: V23_METHODS.md.
# Probe (1/1, 1324 steps) is not stored and is not in these series.
ONLINE_SUCCESS = [
    1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,
    1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
]
ONLINE_STEPS = [
    1333, 1368, 1460, 2276, 4300, 1843, 4300, 4300, 2927, 4300,
    4300, 4300, 3344, 1436, 997, 975, 4300, 1161, 1190, 1174,
    1157, 4300, 4300, 4300, 4300, 1197, 841, 4300, 1317, 858,
    1130, 965, 1161, 863, 1156, 4300, 2870, 874, 807, 871,
    1157, 1138, 1198, 1903, 1476, 1738, 871, 3029, 1483, 858,
]
TIMEOUT_STEPS = 4300
VLA_SR = 80.0  # frozen pt12 specialist, 8/10 on instance 308


def window_rate(xs: list[int], k: int | None) -> list[float]:
    out: list[float] = []
    for i in range(1, len(xs) + 1):
        w = xs[:i] if k is None else xs[max(0, i - k) : i]
        out.append(100.0 * sum(w) / len(w))
    return out


def _style_axes(ax) -> None:
    ax.tick_params(colors=COLORS["ink"], length=3)
    for spine_name, spine in ax.spines.items():
        spine.set_color(COLORS["rule"])
        spine.set_linewidth(0.7)
        if spine_name in ("top", "right"):
            spine.set_visible(False)
    ax.grid(axis="y", color=COLORS["rule"], lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def render_success(out: Path) -> None:
    n = len(ONLINE_SUCCESS)
    episodes = list(range(1, n + 1))
    last10 = window_rate(ONLINE_SUCCESS, 10)
    cum = window_rate(ONLINE_SUCCESS, None)
    n_ok = sum(ONLINE_SUCCESS)
    ok_steps = [s for s, y in zip(ONLINE_STEPS, ONLINE_SUCCESS) if y]
    med_ok = sorted(ok_steps)[len(ok_steps) // 2]
    bar_colors = [
        tint(COLORS["ae"], 0.42) if y else tint(COLORS["muted"], 0.58) for y in ONLINE_SUCCESS
    ]

    fig = plt.figure(figsize=(7.16, 4.72), dpi=200)
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[0.20, 2.22, 1.58],
        hspace=0.10,
        left=0.105,
        right=0.985,
        top=0.88,
        bottom=0.11,
    )
    ax_bar = fig.add_subplot(gs[0])
    ax_sr = fig.add_subplot(gs[1], sharex=ax_bar)
    ax_len = fig.add_subplot(gs[2], sharex=ax_bar)

    ax_bar.bar(episodes, [1] * n, width=0.92, color=bar_colors, edgecolor="none", zorder=2)
    ax_bar.set_yticks([])
    ax_bar.set_ylabel(" ", fontsize=7.0)
    ax_bar.tick_params(axis="x", labelbottom=False, length=0)
    for spine in ax_bar.spines.values():
        spine.set_visible(False)
    ax_bar.set_xlim(0.5, n + 0.5)
    ax_bar.set_ylim(0.0, 1.0)
    ax_bar.text(
        -0.02,
        0.5,
        "ep.",
        transform=ax_bar.transAxes,
        fontsize=7.0,
        color=COLORS["muted"],
        ha="right",
        va="center",
    )

    ax_sr.plot(episodes, last10, color=COLORS["train"], lw=2.15, solid_capstyle="round", zorder=3)
    ax_sr.plot(episodes, cum, color=COLORS["frozen"], lw=1.45, solid_capstyle="round", zorder=2)
    ax_sr.axhline(VLA_SR, color=COLORS["muted"], ls=(0, (3.5, 2.4)), lw=1.05, zorder=1)
    ax_sr.set_ylim(0, 108)
    ax_sr.set_ylabel("Success rate (%)")
    ax_sr.tick_params(axis="x", labelbottom=False, length=3, colors=COLORS["muted"])
    ax_sr.set_yticks([0, 20, 40, 60, 80, 100])
    _style_axes(ax_sr)
    ax_sr.legend(
        handles=[
            plt.Line2D([0], [0], color=COLORS["train"], lw=2.15, label="Last-10 SR"),
            plt.Line2D([0], [0], color=COLORS["frozen"], lw=1.45, label="Cumulative SR"),
            plt.Line2D(
                [0],
                [0],
                color=COLORS["muted"],
                lw=1.05,
                ls=(0, (3.5, 2.4)),
                label="Frozen specialist  8/10",
            ),
        ],
        loc="lower right",
        frameon=False,
        fontsize=7.2,
        borderaxespad=0.4,
    )
    ax_sr.text(
        0.015,
        0.97,
        "(a)",
        transform=ax_sr.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
        color=COLORS["ink"],
    )
    ax_sr.annotate(
        "10/10 from ep. 46",
        xy=(48, 100),
        xytext=(28, 92),
        fontsize=6.6,
        color=COLORS["train"],
        arrowprops={"arrowstyle": "-", "color": COLORS["train"], "lw": 0.7},
    )

    ok_x = [i + 1 for i, y in enumerate(ONLINE_SUCCESS) if y]
    fail_x = [i + 1 for i, y in enumerate(ONLINE_SUCCESS) if not y]
    ok_y = [ONLINE_STEPS[i - 1] for i in ok_x]
    fail_y = [ONLINE_STEPS[i - 1] for i in fail_x]
    ax_len.plot(episodes, ONLINE_STEPS, color=tint(COLORS["frozen"], 0.35), lw=0.9, zorder=1)
    ax_len.scatter(ok_x, ok_y, s=18, c=COLORS["ae"], zorder=3, linewidths=0, label="Success")
    ax_len.scatter(
        fail_x,
        fail_y,
        s=22,
        facecolors="none",
        edgecolors=COLORS["muted"],
        linewidths=0.9,
        zorder=3,
        label="Timeout",
    )
    ax_len.axhline(TIMEOUT_STEPS, color=COLORS["muted"], ls=(0, (3.5, 2.4)), lw=1.0, zorder=1)
    ax_len.axhline(med_ok, color=COLORS["ae"], ls=(0, (1.5, 1.8)), lw=0.9, zorder=1)
    ax_len.set_ylabel("Env steps")
    ax_len.set_xlabel("Online episode")
    ax_len.set_ylim(0, 4700)
    ax_len.set_xticks([1, 10, 20, 30, 40, 50])
    _style_axes(ax_len)
    ax_len.legend(loc="center right", frameon=False, fontsize=7.2, borderaxespad=0.4)
    ax_len.text(
        0.015,
        0.96,
        "(b)",
        transform=ax_len.transAxes,
        fontsize=9.0,
        fontweight="bold",
        va="top",
        color=COLORS["ink"],
    )
    ax_len.text(
        2.0,
        TIMEOUT_STEPS - 160,
        "timeout 4300",
        fontsize=6.2,
        color=COLORS["muted"],
        va="top",
    )
    ax_len.text(
        2.0,
        med_ok + 80,
        f"median success  {med_ok}",
        fontsize=6.2,
        color=COLORS["ae"],
        va="bottom",
    )

    fig.text(
        0.105,
        0.955,
        r"cf_v23_ae2  ·  turning_on_radio  ·  instance 308",
        fontsize=9.2,
        fontweight="bold",
        color=COLORS["ink"],
        ha="left",
        va="center",
    )
    fig.text(
        0.985,
        0.955,
        f"{n_ok}/50 = {100.0 * n_ok / n:.0f}%     last-10  10/10     pt12  8/10",
        fontsize=7.4,
        color=COLORS["muted"],
        ha="right",
        va="center",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    configure()
    render_architecture(OUT / "fig_architecture.png")
    render_training(OUT / "fig_training.png")
    render_filmstrip(OUT / "fig_task_filmstrip.png")
    render_success(OUT / "fig_success.png")
    print(OUT / "fig_architecture.png")
    print(OUT / "fig_training.png")
    print(OUT / "fig_task_filmstrip.png")
    print(OUT / "fig_success.png")


if __name__ == "__main__":
    main()
