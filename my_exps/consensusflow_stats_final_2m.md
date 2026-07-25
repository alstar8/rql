# ConsensusFlow empirical aggregation

Generated: `2026-07-22T04:58:29Z`

## OGBench-50 final_checkpoint_at_2m (seeds 0–2)

**Protocol change (2026-07-16): only the exact 2,000,000-step checkpoint is authoritative. Earlier paper_last3 aggregates are legacy-only.**

Source CSV: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/ogbench50_all50_metrics_2m.csv`
Source CSV sha256: `c94b13292b42d6edea8390310121cd71b873756179a33e4612dfdf95f554dda3`
CSV↔raw all_match: `False`
Legacy paper_last3 CSV: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/ogbench50_all50_metrics_2m_paper_last3_legacy.csv`
Raw recompute save_dir: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/exp`

| Method | Final-2M grand mean | Bootstrap 95% CI (tasks) |
|---|---:|---:|
| baseline | 0.550933 | [0.444533, 0.654673] |
| v6 | 0.586000 | [0.480530, 0.690800] |
| v7 | 0.576800 | [0.469600, 0.680667] |
| v8 | 0.583333 | [0.474933, 0.689867] |
| v9 | 0.593467 | [0.486797, 0.696267] |

### v9 vs baseline (per-task seed-means)

- Δ mean: 0.04253333333333333
- Win/loss/tie: {'win': 26, 'loss': 13, 'tie': 11}
- Paired Δ bootstrap 95% CI: {'mean_delta': 0.04253333333333333, 'ci_low': -0.004000000000000008, 'ci_high': 0.08946999999999995, 'n': 50, 'n_boot': 10000, 'alpha': 0.05}
- Paired t: {'statistic': 1.722905100062272, 'pvalue': 0.09121203517615803}
- Wilcoxon: {'statistic': 237.5, 'pvalue': 0.033308536756606444}

## humanoidmaze-large ConsensusFlow ablations

Budget: final_checkpoint_at_1000k (exact step 1000000; SHORTER THAN 2M PAPER PROTOCOL)

| Variant | Mean | Std | n | Run group |
|---|---:|---:|---:|---|
| full | 0.4186666666666667 | 0.052814139520902255 | 3 | `ogbench50-dflrql9-humanoidmaze_large_task{task}` |
| no_guidance | 0.39333333333333337 | 0.010066445913694341 | 3 | `ogbench-hl5-cf-ablation-noguidance-task{task}-1m` |
| lambda02 | 0.42533333333333334 | 0.03801753981168865 | 3 | `ogbench-hl5-cf-ablation-nocrf-lambda02-task{task}-1m` |
| lambda10 | 0.33066666666666666 | 0.016653327995729075 | 3 | `ogbench-hl5-cf-ablation-nocrf-lambda10-task{task}-1m` |
| no_conflict | 0.42266666666666675 | 0.010066445913694341 | 3 | `ogbench-hl5-cf-ablation-noconflict-task{task}-1m` |
| no_residual | 0.38400000000000006 | 0.0471593044902064 | 3 | `ogbench-hl5-cf-ablation-noresidual-task{task}-1m` |
| no_floor | 0.3906666666666667 | 0.062139627721232285 | 3 | `ogbench-hl5-cf-ablation-nofloor-task{task}-1m` |
| no_crf | 0.42533333333333334 | 0.05430776494511013 | 3 | `ogbench50-dflrql9-nocrf-humanoidmaze_large_task{task}` |
| nocrf_k2 | 0.24266666666666667 | 0.07433258594542055 | 3 | `ogbench-hl5-cf-ablation-nocrf-k2-task{task}-1m` |
| nocrf_k5 | 0.38933333333333336 | 0.020132891827388682 | 3 | `ogbench-hl5-cf-ablation-nocrf-k5-task{task}-1m` |
| nocrf_k20 | 0.38133333333333336 | 0.015143755588800743 | 3 | `ogbench-hl5-cf-ablation-nocrf-k20-task{task}-1m` |
| single_critic | 0.0013333333333333333 | 0.002309401076758503 | 3 | `ogbench-hl5-cf-ablation-singlecritic-task{task}-1m` |

### Seed-level

#### full
- seed 0: success=0.40800000000000003 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000
- seed 1: success=0.476 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000
- seed 2: success=0.37200000000000005 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000

#### no_guidance
- seed 0: success=0.404 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.384 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.392 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### lambda02
- seed 0: success=0.46399999999999997 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.42400000000000004 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.388 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### lambda10
- seed 0: success=0.34400000000000003 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.312 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.336 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### no_conflict
- seed 0: success=0.43200000000000005 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.41200000000000003 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.42400000000000004 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### no_residual
- seed 0: success=0.34400000000000003 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.43600000000000005 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.372 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### no_floor
- seed 0: success=0.372 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.33999999999999997 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.4600000000000001 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### no_crf
- seed 0: success=0.368 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000
- seed 1: success=0.43200000000000005 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000
- seed 2: success=0.476 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000

#### nocrf_k2
- seed 0: success=0.192 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.328 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.20800000000000002 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### nocrf_k5
- seed 0: success=0.392 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.40800000000000003 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.368 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### nocrf_k20
- seed 0: success=0.392 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.388 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.364 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

#### single_critic
- seed 0: success=0.004 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 1: success=0.0 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000
- seed 2: success=0.0 mode=final_checkpoint flags_match=True n_tasks=5 max_step=1000000

