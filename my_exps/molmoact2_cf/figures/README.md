# V19 ICRA-style schemes (TikZ)

Two figures in the visual language of RL Token (`papers/RLT`) and CFGRL
(`papers/CFGRL`): pastel panels, frozen VLA, RL token, product-policy
sampling. Environment stills are real MolmoSpaces house0 exo/wrist frames.

| File | Role |
| --- | --- |
| `architecture.tikz` | Frozen MolmoAct2 + RL token + CFGRL extractor |
| `training_loop.tikz` | CPI loop with brief \(\mathcal{L}_Q\), \(\mathcal{L}_{\mathrm{CFGRL}}\), \(w\)/\(p\) |
| `v19_fig_style.tex` | Shared palette / TikZ keys |
| `compile_figures.tex` | Standalone crop of both pictures |
| `v19_icra_figures.tex` | IEEEtran `figure*` wrappers + captions |
| `env/` | `exo_kettle.png`, `wrist_kettle.png` (from `real_*_h0.png`) |
| `compile_figures.pdf` | Compiled schemes |
| `preview_p1.png`, `preview_p2.png` | Raster previews |

Compile:

```bash
tectonic -X compile compile_figures.tex
# or, for IEEE column geometry:
tectonic -X compile v19_icra_figures.tex
```

ICRA reviewer loop (3 rounds) ended **accept with minor comments**
(Fig.1 4/5, Fig.2 5/5). Method non-goals kept out of the drawings:
no CF guide, no Molmo LoRA, no val-split eval, no \(o{=}0\) failure BC.
