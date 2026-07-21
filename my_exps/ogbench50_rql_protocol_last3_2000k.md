# RQL validation protocol metrics

Generated: `2026-07-21T08:30:26Z`

## Protocol

- Checkpoints: **1800k/1900k/2000k** (`[1800000, 1900000, 2000000]`)
- Seeds: `[0, 1, 2]`
- Eval episodes / checkpoint: `50`
- Tasks: `50`
- Aggregation: per (task, seed) last3 mean → mean over seeds → mean over 50 tasks; headline seed_std is std of per-seed 50-task means

> RQL paper reports 4 seeds; local CF / CF-noCRF currently have seeds {0,1,2}. Matched comparison uses the shared seed set.

## Grand means (seed-mean of 50-task means)

| Method | Mean | Seed std | Coverage (all-seed tasks) |
|---|---:|---:|---:|
| RQL baseline | 0.562889 | 0.020265 | 50/50 |
| ConsensusFlow (dflrql9) | 0.592578 | 0.009697 | 50/50 |
| ConsensusFlow no-CRF | 0.590800 | 0.020480 | 50/50 |

## Pairwise (task means)

- **CF_minus_RQL**: Δ=+0.029689, W/L/T=28/15/7 (n=50)
- **CF_noCRF_minus_RQL**: Δ=+0.027911, W/L/T=25/17/8 (n=50)
- **CF_minus_CF_noCRF**: Δ=+0.001778, W/L/T=17/25/8 (n=50)

## Domain means

| Domain | RQL | CF | CF no-CRF |
|---|---:|---:|---:|
| antmaze-giant | 0.3600 | 0.3204 | 0.3476 |
| antmaze-large | 0.7249 | 0.8236 | 0.8342 |
| cube-double | 0.0796 | 0.1942 | 0.2053 |
| cube-quadruple | 0.5200 | 0.5813 | 0.4684 |
| cube-triple | 0.0342 | 0.0249 | 0.0333 |
| humanoidmaze-large | 0.4533 | 0.5627 | 0.5391 |
| humanoidmaze-medium | 0.9929 | 0.9902 | 0.9898 |
| puzzle-3x3 | 1.0000 | 1.0000 | 1.0000 |
| puzzle-4x4 | 0.6911 | 0.4818 | 0.5351 |
| scene | 0.7729 | 0.9467 | 0.9551 |
