# RQL validation protocol metrics

Generated: `2026-07-21T08:30:26Z`

## Protocol

- Checkpoints: **800k/900k/1000k** (`[800000, 900000, 1000000]`)
- Seeds: `[0, 1, 2]`
- Eval episodes / checkpoint: `50`
- Tasks: `50`
- Aggregation: per (task, seed) last3 mean → mean over seeds → mean over 50 tasks; headline seed_std is std of per-seed 50-task means

> RQL paper reports 4 seeds; local CF / CF-noCRF currently have seeds {0,1,2}. Matched comparison uses the shared seed set.

## Grand means (seed-mean of 50-task means)

| Method | Mean | Seed std | Coverage (all-seed tasks) |
|---|---:|---:|---:|
| RQL baseline | 0.552000 | 0.018106 | 50/50 |
| ConsensusFlow (dflrql9) | 0.566000 | 0.007564 | 50/50 |
| ConsensusFlow no-CRF | 0.570267 | 0.004946 | 50/50 |

## Pairwise (task means)

- **CF_minus_RQL**: Δ=+0.014000, W/L/T=30/14/6 (n=50)
- **CF_noCRF_minus_RQL**: Δ=+0.018267, W/L/T=25/18/7 (n=50)
- **CF_minus_CF_noCRF**: Δ=-0.004267, W/L/T=20/23/7 (n=50)

## Domain means

| Domain | RQL | CF | CF no-CRF |
|---|---:|---:|---:|
| antmaze-giant | 0.3724 | 0.2689 | 0.2227 |
| antmaze-large | 0.8236 | 0.8693 | 0.8853 |
| cube-double | 0.2107 | 0.3702 | 0.4009 |
| cube-quadruple | 0.5116 | 0.5329 | 0.5382 |
| cube-triple | 0.0453 | 0.0422 | 0.0413 |
| humanoidmaze-large | 0.3809 | 0.3653 | 0.3773 |
| humanoidmaze-medium | 0.9120 | 0.9240 | 0.9480 |
| puzzle-3x3 | 1.0000 | 1.0000 | 1.0000 |
| puzzle-4x4 | 0.3907 | 0.3200 | 0.3156 |
| scene | 0.8729 | 0.9671 | 0.9733 |
