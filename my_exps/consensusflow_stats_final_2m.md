# ConsensusFlow empirical aggregation

Generated: `2026-07-19T08:12:32Z`

## OGBench-50 final_checkpoint_at_2m (seeds 0–2)

**Protocol change (2026-07-16): only the exact 2,000,000-step checkpoint is authoritative. Earlier paper_last3 aggregates are legacy-only.**

Source CSV: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/ogbench50_all50_metrics_2m.csv`
Source CSV sha256: `94ae51cf886a627cc3c8996cf4d8d00001372f821b68c2ada5ef35047b471194`
CSV↔raw all_match: `False`
Legacy paper_last3 CSV: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/ogbench50_all50_metrics_2m_paper_last3_legacy.csv`
Raw recompute save_dir: `/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/exp`

| Method | Final-2M grand mean | Bootstrap 95% CI (tasks) |
|---|---:|---:|
| baseline | 0.550667 | [0.444530, 0.654400] |
| v6 | 0.586000 | [0.480530, 0.690800] |
| v7 | 0.576800 | [0.469600, 0.680667] |
| v8 | 0.583333 | [0.474933, 0.689867] |
| v9 | 0.593733 | [0.486930, 0.696403] |

### v9 vs baseline (per-task seed-means)

- Δ mean: 0.04306666666666667
- Win/loss/tie: {'win': 27, 'loss': 13, 'tie': 10}
- Paired Δ bootstrap 95% CI: {'mean_delta': 0.04306666666666667, 'ci_low': -0.0035999999999999934, 'ci_high': 0.09026666666666668, 'n': 50, 'n_boot': 10000, 'alpha': 0.05}
- Paired t: {'statistic': 1.7424707560003396, 'pvalue': 0.0876997506793281}
- Wilcoxon: {'statistic': 250.5, 'pvalue': 0.032028368053185555}

## humanoidmaze-large ConsensusFlow ablations

Budget: final_checkpoint_at_1000k (exact step 1000000; SHORTER THAN 2M PAPER PROTOCOL)

| Variant | Mean | Std | n | Run group |
|---|---:|---:|---:|---|
| full | 0.7733333333333334 | 0.03055050463303896 | 3 | `humanoidmaze-large-dflrql9-2m` |
| no_guidance | 0.7066666666666667 | 0.023094010767584987 | 3 | `humanoidmaze-large-cf-ablation-noguidance-1m` |
| lambda02 | 0.7733333333333334 | 0.011547005383792525 | 3 | `humanoidmaze-large-cf-ablation-nocrf-lambda02-1m` |
| lambda10 | 0.8000000000000002 | 0.019999999999999962 | 3 | `humanoidmaze-large-cf-ablation-nocrf-lambda10-1m` |
| no_conflict | 0.8133333333333335 | 0.041633319989322626 | 3 | `humanoidmaze-large-cf-ablation-noconflict-1m` |
| no_residual | 0.82 | 0.10000000000000003 | 3 | `humanoidmaze-large-cf-ablation-noresidual-1m` |
| no_floor | 0.7999999999999999 | 0.12165525060596437 | 3 | `humanoidmaze-large-cf-ablation-nofloor-1m` |
| no_crf | 0.8533333333333334 | 0.02309401076758505 | 3 | `humanoidmaze-large-cf-ablation-nocrf-1m` |
| nocrf_k2 | 0.6266666666666666 | 0.13012814197295425 | 3 | `humanoidmaze-large-cf-ablation-nocrf-k2-1m` |
| nocrf_k5 | 0.84 | 0.06 | 3 | `humanoidmaze-large-cf-ablation-nocrf-k5-1m` |
| nocrf_k20 | 0.8533333333333332 | 0.08082903768654756 | 3 | `humanoidmaze-large-cf-ablation-nocrf-k20-1m` |
| single_critic | 0.0 | 0.0 | 3 | `humanoidmaze-large-cf-ablation-singlecritic-1m` |

### Seed-level

#### full
- seed 0: success=0.78 mode=final_checkpoint flags_match=True max_step=2000000
- seed 1: success=0.8 mode=final_checkpoint flags_match=True max_step=2000000
- seed 2: success=0.74 mode=final_checkpoint flags_match=True max_step=2000000

#### no_guidance
- seed 0: success=0.68 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.72 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.72 mode=final_checkpoint flags_match=True max_step=1000000

#### lambda02
- seed 0: success=0.78 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.76 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.78 mode=final_checkpoint flags_match=True max_step=1000000

#### lambda10
- seed 0: success=0.82 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.78 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.8 mode=final_checkpoint flags_match=True max_step=1000000

#### no_conflict
- seed 0: success=0.78 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.86 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.8 mode=final_checkpoint flags_match=True max_step=1000000

#### no_residual
- seed 0: success=0.92 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.72 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.82 mode=final_checkpoint flags_match=True max_step=1000000

#### no_floor
- seed 0: success=0.66 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.86 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.88 mode=final_checkpoint flags_match=True max_step=1000000

#### no_crf
- seed 0: success=0.84 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.88 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.84 mode=final_checkpoint flags_match=True max_step=1000000

#### nocrf_k2
- seed 0: success=0.76 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.62 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.5 mode=final_checkpoint flags_match=True max_step=1000000

#### nocrf_k5
- seed 0: success=0.9 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.84 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.78 mode=final_checkpoint flags_match=True max_step=1000000

#### nocrf_k20
- seed 0: success=0.94 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.84 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.78 mode=final_checkpoint flags_match=True max_step=1000000

#### single_critic
- seed 0: success=0.0 mode=final_checkpoint flags_match=True max_step=1000000
- seed 1: success=0.0 mode=final_checkpoint flags_match=True max_step=1000000
- seed 2: success=0.0 mode=final_checkpoint flags_match=True max_step=1000000

