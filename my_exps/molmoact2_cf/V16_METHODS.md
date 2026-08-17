# V16 methods: paper Alg.1 collect + strong soft-β BC

Status: **long run live; mid-run repair applied 2026-08-13** (explore=0 on CF/AE, CF guide skip uses selected grads, AE actor-before-expensive-critic + 180s budget).  
Reset fresh: `residual_vla_cf`, `residual_rlt_cf`, `flow_rlt_cf`, `molmo_ae_lora_actor`. Kept/resumed: baselines + RLT actors.  
Run: `runs/rlt_cf_v16_controlled`  
Ports: `8710–8716` (does not collide with V15 `8700–8706`)  
Predecessor: V15 (mixture coverage + empirical gate; train SR drop diagnosed as explore tax)

## Why train SR dropped in V15

1. RLT arms always applied explore σ=0.02 on frozen-VLA chunks; baselines had explore off → ~0.10 SR gap.
2. `residual_vla_cf` (no mixture) matched actor-arm train SR → mixture was not the main train-SR culprit.
3. Per-chunk mixture broke empirical episode labels (`empirical_insufficient_episodes` after 350+ eps).
4. Gate never opened → held-out forced-actor efficacy still required (V13 ΔSR ≈ −0.21).

## V16 changes (RLT paper aligned)

| Change | Default |
| --- | --- |
| Always collect `π_θ` after BC warmup | `--always_collect_actor` |
| Gate-off mixture | **retired** (`p=0`) |
| Additive explore on RLT arms | **0** (use actor stochasticity) |
| Soft BC β | `actor_beta=2.0` |
| Reference dropout (train) | `0.5` |
| Hard residual ball | `residual_clip=0.02` (safety only) |
| Q/empirical gates | tag deploy / guide only; do **not** block actor collection |
| Updates / episode | `128` (was 8; fills ~60s CUDA budget; MuJoCo still dominates wall) |
| EGL concurrency | `RLT_EGL_PER_GPU=2`, `RLT_EGL_MAX_CONCURRENT=16` |
| VLA chunk prefetch | `RLT_VLA_PREFETCH=1` (hide `/act` behind late-chunk MuJoCo) |
| Metrics schema | `16` (+ episode collect labels, empirical counts) |

GPU idle is structural (MuJoCo-dominated wall). See [GPU_UTIL_BOTTLENECK.md](GPU_UTIL_BOTTLENECK.md).

## Eight arms

Same matrix as V15: residual/flow baselines, actors, CF, `residual_vla_cf`, provisional AE.

## Launch

```bash
# Smoke
V16_MODE=smoke FRESH=1 bash launch_v16_controlled.sh

# Full 400-episode matrix (stop V15 first if GPUs are busy)
V16_MODE=full FRESH=1 bash launch_v16_controlled.sh

# ~24h wall-clock long run (1000 eps / 600k steps; snapshots 0,100,200,400,700,1000)
# Packs INSTANCES_PER_GPU=3 HTTP replicas per GPU (AE stays 1).
V16_MODE=long FRESH=1 bash launch_v16_controlled.sh

bash eval_v16_controlled.sh
```

Packing defaults: `INSTANCES_PER_GPU=3`, `AE_INSTANCES_PER_GPU=1`, `RLT_EGL_PER_GPU=3`, `RLT_EGL_MAX_CONCURRENT=24`. Shard outputs live under `<variant>/shard_{0,1,2}/`.

Do not overwrite `runs/rlt_cf_v13_controlled`, `v14`, or `v15`.

Improved RLT+CF capacity test (z=256 / d=512 / 4 layers, non-overlapping chunks only): see [V16_RLT_IMPROVED.md](V16_RLT_IMPROVED.md).  
Follow-on (kettle offline + safe collect): see [V17_METHODS.md](V17_METHODS.md).
