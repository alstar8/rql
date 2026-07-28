# OGBench-50 checkpoint × protocol comparison

Generated: `2026-07-25T21:55:10Z`

Seeds `{0,1,2}`; 50 tasks. Headline = mean over task-means (each task-mean averages all available seeds for that task; only tasks with all 3 seeds are included).

## Protocols

- **final_checkpoint**: evaluation/success at exactly report_step; mean over seeds 0-2; mean over tasks with all seeds
- **rql_validation_last3**: mean evaluation/success over last three 100k-spaced checkpoints ending at report_step (offsets (200000, 100000, 0)); then mean over seeds; then mean over full-seed tasks
- **CF-noCRF**: CF-noCRF uses ogbench50-dflrql9-nocrf-* offline groups. No matched *-o2o1m suite exists for CF-noCRF (o2o rows are n/a).

## Results

| Setting | Protocol | RQL | ConsensusFlow | CF-noCRF | Coverage (RQL/CF/noCRF) |
|---|---|---:|---:|---:|---:|
| 1M offline (final checkpoint) | final_checkpoint | 0.5696 | 0.5837 | 0.5951 | 50/50/50 |
| 1M offline (RQL last3: 800k/900k/1M) | rql_validation_last3 | 0.5564 | 0.5695 | 0.5703 | 50/50/50 |
| 1M offline + 1M online (final checkpoint @2M) | final_checkpoint | 0.8215 | 0.8592 | n/a | 50/50/0 |
| 1M offline + 1M online (RQL last3: 1.8M/1.9M/2.0M) | rql_validation_last3 | 0.8214 | 0.8539 | n/a | 50/50/0 |
| 2M offline (final checkpoint) | final_checkpoint | 0.5672 | 0.5956 | 0.5885 | 39/50/50 |
| 2M offline (RQL last3: 1.8M/1.9M/2.0M) | rql_validation_last3 | 0.5795 | 0.5946 | 0.5908 | 39/50/50 |

## Pairwise (on shared full-seed tasks)

### 1M offline (final checkpoint)
- CF_minus_RQL: Δ=+0.0141, W/L/T=28/14/8 (n=50)
- CFnoCRF_minus_RQL: Δ=+0.0255, W/L/T=24/16/10 (n=50)
- CF_minus_CFnoCRF: Δ=-0.0113, W/L/T=18/25/7 (n=50)

### 1M offline (RQL last3: 800k/900k/1M)
- CF_minus_RQL: Δ=+0.0131, W/L/T=30/12/8 (n=50)
- CFnoCRF_minus_RQL: Δ=+0.0138, W/L/T=24/17/9 (n=50)
- CF_minus_CFnoCRF: Δ=-0.0008, W/L/T=23/21/6 (n=50)

### 1M offline + 1M online (final checkpoint @2M)
- CF_minus_RQL: Δ=+0.0377, W/L/T=19/11/20 (n=50)
- CFnoCRF_minus_RQL: n/a
- CF_minus_CFnoCRF: n/a

### 1M offline + 1M online (RQL last3: 1.8M/1.9M/2.0M)
- CF_minus_RQL: Δ=+0.0324, W/L/T=22/11/17 (n=50)
- CFnoCRF_minus_RQL: n/a
- CF_minus_CFnoCRF: n/a

### 2M offline (final checkpoint)
- CF_minus_RQL: Δ=+0.0484, W/L/T=19/9/11 (n=39)
- CFnoCRF_minus_RQL: Δ=+0.0354, W/L/T=17/12/10 (n=39)
- CF_minus_CFnoCRF: Δ=+0.0071, W/L/T=18/18/14 (n=50)

### 2M offline (RQL last3: 1.8M/1.9M/2.0M)
- CF_minus_RQL: Δ=+0.0319, W/L/T=20/12/7 (n=39)
- CFnoCRF_minus_RQL: Δ=+0.0250, W/L/T=19/13/7 (n=39)
- CF_minus_CFnoCRF: Δ=+0.0038, W/L/T=18/23/9 (n=50)

