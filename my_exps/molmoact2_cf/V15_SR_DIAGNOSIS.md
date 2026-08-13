# V15 train SR diagnosis (pre held-out)

Train SR is **not** policy efficacy. Summary at latest metrics:

| Arm | Eps | CumSR | WinSR | Phase | Gate | Emp block |
|---|---:|---:|---:|---|---|---|
| residual_vla_baseline | 400 | 0.240 | 0.200 | — | closed | — |
| residual_rlt_actor | 370 | 0.154 | 0.160 | clipped_q | closed | empirical_insufficient_episodes |
| residual_vla_cf | 360 | 0.144 | 0.180 | — | closed | — |
| residual_rlt_cf | 370 | 0.138 | 0.160 | clipped_q | closed | empirical_insufficient_episodes |
| flow_vla_baseline | 380 | 0.234 | 0.240 | — | closed | — |
| flow_rlt_actor | 370 | 0.135 | 0.100 | clipped_q | closed | empirical_insufficient_episodes |
| flow_rlt_cf | 360 | 0.139 | 0.160 | clipped_q | closed | empirical_insufficient_episodes |
| molmo_ae_lora_actor | 170 | 0.141 | 0.180 | — | closed | — |

## Mechanism

- Baselines: explore off → SR≈0.24
- `residual_vla_cf` (mixture=0, explore on): SR≈0.14–0.15 → explore tax explains the gap
- Actor arms match that band; gate never opened
- Empirical gate stuck on `empirical_insufficient_episodes` (per-chunk mixture)

Held-out paired eval (ep100/200) running via `eval_v15_controlled.sh`.
