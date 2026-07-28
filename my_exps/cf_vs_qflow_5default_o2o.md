# Matched 5-default-task offline→online comparison (Fig-6 style)

## Figure

- PDF: `assets/consensusflow/generated/cf_vs_qflow_fig6_o2o.pdf`
- PNG: `assets/consensusflow/generated/cf_vs_qflow_fig6_o2o.png`
- JSON: `generated/cf_vs_qflow_fig6_o2o.json`
- Regenerator: `scripts/plot_cf_vs_qflow_fig6.py`

```bash
python scripts/plot_cf_vs_qflow_fig6.py
# optional local Q-Flow overlay when complete:
python scripts/plot_cf_vs_qflow_fig6.py --include-local-qflow
```

## Task set (Q-Flow Fig. 6 / `online_ft_full.png`)

| Env | Default task | CF source (plotted) |
|---|---|---|
| antmaze-giant | task1 | **tune-C** `cftune-antmaze-C-rho0-lam1` (+ matched `-o2o1m`) |
| humanoidmaze-medium | task1 | `ogbench50-dflrql9-*` |
| antsoccer-arena | task4 | **S3 ens=2** `cftune-soccer-S3-std-a03` (+ matched `-o2o1m`) |
| cube-double | task2 | `ogbench50-dflrql9-*` |
| puzzle-4x4 | task4 | `ogbench50-dflrql9-*` |

## Protocol caveats

| | Q-Flow (paper Fig. 6) | ConsensusFlow (this plot) |
|---|---|---|
| Setting | **standard** (ens=2, no chunking) | mixed: baseline advanced-like; hard tasks use tuned overrides |
| Seeds | 8 | 3 |
| Online hparam search | yes (re-tuned) | no |
| Curve source | digitized from `online_ft_full.png` | local `evaluation/success` |

## Per-task success (fraction) — refreshed 2026-07-27 ~08:55 UTC

| Task | Q-Flow @1M | Q-Flow @2M | CF @1M | CF @2M | CF variant |
|---|---:|---:|---:|---:|---|
| antmaze-giant-task1 | 0.15 | 0.85 | **0.000** | **0.907** | tune-C ens=10, ρ=0, λ=1.0 |
| humanoidmaze-medium-task1 | 0.90 | 0.99 | 0.980 | 0.980 | baseline CF (ens=10) |
| antsoccer-arena-task4 | 0.25 | 0.80 | **0.327** | **0.440** | S3 ens=2, ρ=0, α=0.3, λ=0.5 |
| cube-double-task2 | 0.28 | 0.92 | 0.460 | 0.987 | baseline CF |
| puzzle-4x4-task4 | 0.20 | 0.75 | 0.113 | 1.000 | baseline CF |

5-task CF mean @2M: **0.863** (vs Q-Flow paper digitize ~0.862).

### Antmaze-giant tune C (`cftune-antmaze-C-rho0-lam1`, ens=10, ρ=0, λ=1.0, α=0.1) — COMPLETE

Offline @1M: **0.000** (all seeds, all 100k checkpoints).

Matched online `@2M` seeds: **0.90 / 0.98 / 0.84** → mean **0.907 ± 0.070**.

| step | mean success (n=3) |
|---:|---:|
| 1.0M | 0.000 |
| 1.5M | ~rising |
| 2.0M | **0.907** |

Other antmaze arms (worse): fig6-std ens=2 @2M **0.647** (high variance); OGBench-50 advanced o2o @2M **0.000**; tune A/E near zero.

### Antsoccer-arena S3 (`cftune-soccer-S3-std-a03`, ens=2, ρ=0, α=0.3, λ=0.5) — COMPLETE

Offline @1M seeds: **0.38 / 0.26 / 0.34** → mean **0.327**.

Matched online `@2M` seeds: **0.60 / 0.34 / 0.38** → mean **0.440 ± 0.140**.

Other soccer arms: OGBench-50 advanced ens=10 @2M **0.000**; fig6-std ens=2 still training (s0 online @1.3M succ≈0.52).

## ETA rollup (2026-07-27 ~08:55 UTC)

Throughput: Q-Flow ~50–110 it/s/job (median ~55); antsoccer ~55–80 it/s. Packing: Q-Flow ~43 live; Fig6-standard starved to **2** GPU slots (1 job/GPU launcher).

| Suite | Status | Est. wall |
|---|---|---|
| CF antmaze tune-C (ρ=0, λ=1) | **done** (3/3 @2M) | — |
| CF antsoccer S3 ens=2 | **done** (3/3 @2M) | — |
| CF hard-tasks HP tune | **done** | — |
| dflrql6 antmaze-giant task1 o2o | 3 live @1.6–1.7M | **~2 h** |
| Q-Flow OGBench-50 o2o | ~139/150 off, ~107/150 on; ~54 jobs / ~34M steps left; 43 live | **~4 h** |
| CF Fig6-standard (ens=2) | 17 jobs left (antsoccer partial + cube×3 + puzzle×3); 2 live | **~12–14 h** (critical path; accelerates after Q-Flow frees GPUs) |

**All-of-the-above finish: ~12–14 h** (bottleneck = Fig6-standard CF cube/puzzle + leftover antsoccer). Tune-C and S3 antsoccer are already in the figure.
