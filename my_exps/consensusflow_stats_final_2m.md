# ConsensusFlow empirical aggregation

Generated: `2026-07-21T08:11:37Z`

## OGBench-50 final_checkpoint_at_2m (seeds 0–2)

**Protocol change (2026-07-16): only the exact 2,000,000-step checkpoint is authoritative. Earlier paper_last3 aggregates are legacy-only.**

Source CSV: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/ogbench50_all50_metrics_2m.csv`
Source CSV sha256: `c94b13292b42d6edea8390310121cd71b873756179a33e4612dfdf95f554dda3`
CSV↔raw all_match: `True`
Legacy paper_last3 CSV: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/ogbench50_all50_metrics_2m_paper_last3_legacy.csv`
Raw recompute save_dir: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/exp`

| Method | Final-2M grand mean | Bootstrap 95% CI (tasks) |
|---|---:|---:|
| baseline | 0.550667 | [0.444530, 0.654400] |
| v6 | 0.586000 | [0.480530, 0.690800] |
| v7 | 0.576800 | [0.469600, 0.680667] |
| v8 | 0.583333 | [0.474933, 0.689867] |
| v9 | 0.593333 | [0.486793, 0.696137] |

### v9 vs baseline (per-task seed-means)

- Δ mean: 0.04266666666666666
- Win/loss/tie: {'win': 27, 'loss': 13, 'tie': 10}
- Paired Δ bootstrap 95% CI: {'mean_delta': 0.04266666666666666, 'ci_low': -0.003870000000000015, 'ci_high': 0.0896033333333333, 'n': 50, 'n_boot': 10000, 'alpha': 0.05}
- Paired t: {'statistic': 1.7286091421892391, 'pvalue': 0.09017624565194245}
- Wilcoxon: {'statistic': 250.5, 'pvalue': 0.032028368053185555}

## humanoidmaze-large ConsensusFlow ablations

Budget: final_checkpoint_at_1000k (exact step 1000000; SHORTER THAN 2M PAPER PROTOCOL)

| Variant | Mean | Std | n | Run group |
|---|---:|---:|---:|---|
| full | 0.4186666666666667 | 0.052814139520902255 | 3 | `ogbench50-dflrql9-humanoidmaze_large_task{task}` |
| no_guidance | None | None | 0 | `ogbench-hl5-cf-ablation-noguidance-task{task}-1m` |
| lambda02 | None | None | 0 | `ogbench-hl5-cf-ablation-nocrf-lambda02-task{task}-1m` |
| lambda10 | None | None | 0 | `ogbench-hl5-cf-ablation-nocrf-lambda10-task{task}-1m` |
| no_conflict | None | None | 0 | `ogbench-hl5-cf-ablation-noconflict-task{task}-1m` |
| no_residual | None | None | 0 | `ogbench-hl5-cf-ablation-noresidual-task{task}-1m` |
| no_floor | None | None | 0 | `ogbench-hl5-cf-ablation-nofloor-task{task}-1m` |
| no_crf | 0.42533333333333334 | 0.05430776494511013 | 3 | `ogbench50-dflrql9-nocrf-humanoidmaze_large_task{task}` |
| nocrf_k2 | None | None | 0 | `ogbench-hl5-cf-ablation-nocrf-k2-task{task}-1m` |
| nocrf_k5 | None | None | 0 | `ogbench-hl5-cf-ablation-nocrf-k5-task{task}-1m` |
| nocrf_k20 | None | None | 0 | `ogbench-hl5-cf-ablation-nocrf-k20-task{task}-1m` |
| single_critic | None | None | 0 | `ogbench-hl5-cf-ablation-singlecritic-task{task}-1m` |

### Seed-level

#### full
- seed 0: success=0.40800000000000003 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000
- seed 1: success=0.476 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000
- seed 2: success=0.37200000000000005 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000

#### no_guidance
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### lambda02
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### lambda10
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### no_conflict
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### no_residual
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### no_floor
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### no_crf
- seed 0: success=0.368 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000
- seed 1: success=0.43200000000000005 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000
- seed 2: success=0.476 mode=final_checkpoint flags_match=True n_tasks=5 max_step=2000000

#### nocrf_k2
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### nocrf_k5
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### nocrf_k20
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

#### single_critic
- seed 0: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 1: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None
- seed 2: success=None mode=incomplete_tasks flags_match=False n_tasks=0 max_step=None

