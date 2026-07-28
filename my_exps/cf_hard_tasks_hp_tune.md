# CF hard-task investigation + HP tune (antmaze-giant / antsoccer)

## Diagnosis (from local curves + [antmaze agent](8def06ce-fb87-4e1a-80e0-6db1afcb6ab1) / [antsoccer agent](7b7f40ee-b325-4e30-aa6f-a0482dbb4186))

### antmaze-giant-task1
- **Advanced CF** (`ens=10, ρ=0.5, λ=0.5`): weak offline (~0.07) → **online collapse to 0.00**
- **Fig6-standard CF** (`ens=2, ρ=0, λ 0.2→1.0`): offline also ~0, but **online recovers** (seeds ~0.04 / 0.96 / ~0.40)
- Main gap: online λ stuck at 0.5; advanced ens/ρ may also over-dampen

### antsoccer-arena-task4
- **Advanced CF and RQL both never learn** (offline ≤0.08 noise, o2o=0.00)
- ObsPad (42→43) is CF-only and **not causal** (RQL fails identically without pad)
- Q-Flow paper only reports antsoccer under **standard** (`ens=2, ρ=0, λ=0.5`); antsoccer absent from advanced table
- Train diagnostics: `q_mean` → −200 (failure value), high `conflict_kill_frac`

## Tune launcher

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 JOBS_PER_GPU=2 MIN_FREE_MIB=5000 \
  bash cloud_job/run_cf_hard_tasks_hp_tune.sh
```

| Tag | Task | Change | Warmstart |
|---|---|---|---|
| **A** | antmaze online | λ=1.0, keep ens=10/ρ=0.5 | adv 1M ckpt |
| **C** | antmaze online | λ=1.0, ρ=0, ens=10 | adv 1M ckpt |
| **E** | antmaze online | λ=2.0, keep ens=10/ρ=0.5 | adv 1M ckpt |
| **S1** | antsoccer o2o | ens=2, ρ=0, λ=0.5 (Q-Flow std) | from scratch |
| **S2** | antsoccer o2o | ens=10, ρ=0, λ=0.5 | from scratch |
| **S3** | antsoccer o2o | ens=2, ρ=0, α=0.3, λ=0.5 | from scratch |

Groups: `cftune-antmaze-*`, `cftune-soccer-*` under `exp/rql/`.
Logs: `exp/cf_hard_tasks_hp_tune_{driver.log,logs}/`.

## Success criteria (seed-mean)
- antmaze online@2M: target ≥0.4 (Fig6-like foothold); stretch ≥0.7
- antsoccer offline@1M: target ≥0.15; online@2M stretch ≥0.5

## Early monitor (~15 min after relaunch)

- 16/16 live jobs healthy; no Tracebacks. 11 jobs still queued (S3 seeds 1–2 + soccer o2o).
- Antmaze restore @1.000001M ≈ 0.00–0.04 (expected cold start).
- Soccer offline only step-1 evals (=0). Keep all arms; first discriminative signal at ~1.1M (antmaze) / ~100–200k (soccer).

## Mid-run prune (~16:18 UTC, from recheck agent)

| cfg | latest | action |
|---|---|---|
| **S3** α=0.3 | seed0@100k=**0.12** | **KEEP** — launched seeds 1–2 |
| **S1** α=0.1 | mean≈0 @100–200k | **KILLED** |
| **S2** ens10 ρ0 | ≤0.02 | **KILLED** |
| **S5** α=0.2 / **S4** α=0.5 | new | **ADDED** (α sweep) |
| **A** λ=1 | @1.1M mean≈0.007 | keep weak |
| **C/E** | waiting more 1.1M evals | keep |

Antmaze A/C/E keepers adopted across dispatcher relaunch (no double-start).
