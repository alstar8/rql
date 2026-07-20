# ConsensusFlow empirical aggregation

Generated: `2026-07-16T14:43:21Z`

## OGBench-50 paper_last3_at_2m (seeds 0–2)

Source CSV: `my_exps/ogbench50_all50_metrics_2m.csv`
CSV sha256: `dd33b3de578012bd27549fabd86ff48f25dd84a3dd7ccae8ca9b9ccb84766fa8`
Raw recompute save_dir: `exp`
CSV↔raw match: **True**

| Method | CSV grand mean | Raw grand mean | Bootstrap 95% CI (tasks) |
|---|---:|---:|---:|
| baseline | 0.553200 | 0.553200 | [0.449949, 0.655911] |
| v6 | 0.587489 | 0.587489 | [0.484086, 0.690713] |
| v7 | 0.580933 | 0.580933 | [0.475197, 0.684044] |
| v8 | 0.590222 | 0.590222 | [0.483599, 0.693467] |
| v9 | 0.591600 | 0.591600 | [0.487288, 0.691647] |

### v9 vs baseline (per-task seed-means)

- Δ mean: 0.03839999999999999
- Win/loss/tie: {'win': 29, 'loss': 15, 'tie': 6}
- Paired Δ bootstrap 95% CI: {'mean_delta': 0.03839999999999999, 'ci_low': -0.007335555555555571, 'ci_high': 0.08288888888888886, 'n': 50, 'n_boot': 10000, 'alpha': 0.05}
- Paired t: {'statistic': 1.639716738422991, 'pvalue': 0.10746937636795055}
- Wilcoxon: {'statistic': 311.5, 'pvalue': 0.032232385612084226}

## humanoidmaze-large ConsensusFlow ablations

_Ablation aggregation skipped in this run._
