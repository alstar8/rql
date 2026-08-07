# MolmoAct2 + CF full online run — metrics summary

**Date:** 2026-07-31  
**Run:** `runs/molmoact2_cf_full_8gpu/` (8-GPU shard)  
**Also:** aborted single-GPU `runs/molmoact2_cf_full/` (~20 eps before parallel relaunch)

## What happened

All 8 shards **exited “Done”**, but MolmoAct2 servers died around local ep **3–4** (`uvicorn` graceful Shutting down → later `Connection refused`). Train loops kept going and logged **steps=0** failures for the remaining ~110 eps/shard.

| | Count |
|---|---:|
| Logged episodes | 1000 |
| **Real rollouts** (`steps>0`) | **79** |
| Dead-server placeholders (`steps=0`) | 921 |
| Real successes | 14 |
| **Real SR** | **14/79 = 17.7%** |
| Contaminated logged cum SR | ~1–4% |

So the headline `metrics.jsonl` success rates are **not trustworthy** for learning — they mix real fails with server-down zeros.

## Per-shard (real episodes only)

| Shard | Real eps | Successes | Real SR |
|---:|---:|---:|---:|
| 0 | 15 | 5 | 33% |
| 1 | 10 | 0 | 0% |
| 2 | 14 | 2 | 14% |
| 3 | 12 | 1 | 8% |
| 4 | 8 | 2 | 25% |
| 5 | 5 | 2 | 40% |
| 6 | 8 | 2 | 25% |
| 7 | 7 | 0 | 0% |

## Offline / train signals (from `metrics.jsonl`)

At local 100 and 125 (mostly after servers died):

- Predicted **advantage** rose ~15 → ~19–22 (Q thinks \(a_v+G\) is better)
- **Residual RMS** stuck ~**0.031–0.032** (hitting `max_delta=0.05` tanh saturation)
- Critic `q_mean` wandered; no stable AUROC / SR coupling

This is the classic **optimistic \(Q\) + saturated residual** pattern: the dual/trust math looks “healthy” while env success does not improve.

## Vs baselines (same Pick-v1.1 protocol)

| Policy | SR (n) |
|---|---|
| MolmoAct2 baseline smoke | **60%** (6/10) |
| CF \(G=0\) smoke | **40%** (4/10) |
| CF capped \(G\) smoke | **50%** (5/10) |
| Full 8-GPU online (real only) | **~18%** (14/79) |

Online SR is **worse** than smoke, with high variance and too few valid episodes after server death.

## Did we learn something useful?

**Policy / SR: no.** No evidence that online CF raised Pick success. Real SR is far below the frozen MolmoAct2 smoke, and early→late splits (where n allows) do not show a clear lift.

**Engineering / scientific: yes, several hard lessons:**

1. **Server liveness must gate training** — don’t treat `steps=0` / connection errors as task failures; abort or restart the expert server.
2. **8× VLA servers are fragile** under long parallel load (all shards lost servers within ~first 4 real eps); need watchdog + restart, or in-process expert on each GPU.
3. **Proprio-only \(G\) + capped residual** can report large offline advantage while **saturating** and not moving sim success — need better features (VLM \(h\)), fail/success buffer hygiene, and/or light AE BC before trusting Phase-3 DPG.
4. **Logging every 100 env eps** is fine, but SR must be computed on **valid** rollouts only.

## Recommendation

Do **not** treat this run as a positive CF learning result. Next useful experiment: keep servers alive (watchdog), stop-on-server-death, log real SR every 25–50 valid eps, and ablate \(G=0\) vs capped \(G\) on the same live shards before scaling updates.
