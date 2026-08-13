# V15 methods: closed-loop coverage and calibrated gates

Status: **implementation complete; smoke/full launch via `launch_v15_controlled.sh`**.  
Run: `runs/rlt_cf_v15_controlled`  
Predecessor: immutable V13 (negative residual/flow efficacy) and in-progress V14 (AE contract repairs).

## Why V13 failed (summary)

1. Gate closed → training never executed learned actors; replay was frozen VLA + explore noise.
2. Offline Q maximization on that support produced over-optimistic residual Q and collapsed forced-actor held-out SR (−0.21 to −0.23).
3. Guide was distilled from the same critic and composed on a bad actor → unresolved.
4. AE arms were invalid by construction in V13 (coords/FIFO/dead head); not used as efficacy evidence.

## V15 changes

| Change | Default |
| --- | --- |
| Gate-off actor mixture | `p=0.25` (deterministic actor chunks) |
| Shadow actor logging + `actor_ref_mse` | always for residual/flow RLT |
| BC warmup then clipped Q | 50 eps BC (`q_coef=0`), then `residual_clip=0.02` |
| Actor CQL on proposed actions | `actor_cql_coef=0.1` |
| Empirical ΔSR Wilson LCB gate | required for residual/flow RLT; τ=0.0; min 16+16 eps |
| `residual_vla_cf` arm | CF guide on frozen VLA (`--guide_on_reference`) |
| Mid-run forced eval | snapshots `0,100,200,400` |
| AE | single provisional `molmo_ae_lora_actor` with V14 repairs (no AE+CF until V14 canary > 0) |

## Eight arms

1. `residual_vla_baseline`
2. `residual_rlt_actor` (+ mixture + conservative + empirical)
3. `residual_vla_cf` (guide on frozen VLA)
4. `residual_rlt_cf`
5. `flow_vla_baseline`
6. `flow_rlt_actor`
7. `flow_rlt_cf`
8. `molmo_ae_lora_actor` (provisional)

HTTP ports: `8700–8706` + in-process AE on GPU 7.

## Launch

```bash
# Smoke (2 episodes) — requires free GPUs 0–7 and ports 8700–8706
V15_MODE=smoke FRESH=1 bash launch_v15_controlled.sh

# Full 400-episode matrix
V15_MODE=full FRESH=1 bash launch_v15_controlled.sh

# Held-out eval at all snapshots (includes mid-run 100/200)
bash eval_v15_controlled.sh
```

Do not overwrite `runs/rlt_cf_v13_controlled` or `runs/rlt_cf_v14_controlled`.

**Launch note (2026-08-12):** V14 currently owns all eight GPUs and HTTP ports `8700–8705`. A V15 smoke dry-run validated trainer CLI + controlled benchmark, then correctly refused port `8700`. Start V15 after V14 releases the machine (or on a free 8-GPU host).