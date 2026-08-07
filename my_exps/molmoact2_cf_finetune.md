# MolmoAct2 + ConsensusFlow Offline Warmup

**Date:** 2026-07-31  
**Scope:** Offline Phases 0–2 + same-10-eps Pick smoke (no online Phase 3).  
**Code:** [`molmoact2_cf/`](molmoact2_cf/)  
**Baseline:** MolmoAct2-DROID **6/10 (60%)** on first 10 Pick-v1.1 eps  
([`molmospaces_molmoact2_pick_baseline.md`](molmospaces_molmoact2_pick_baseline.md))

## Locks

| Lock | Choice |
|---|---|
| Warmup data | MolmoBot sim pick `FrankaPickOmniCamConfig` (`allenai/molmobot-data`) |
| Backbone | `allenai/MolmoAct2-DROID` |
| VLM | **Frozen** |
| Action expert | Soft-plastic via **BC only** (optional for smoke; default off for speed) |
| Residual | Endpoint \(G_a(s, a_v)\); RL only |
| Critic | Ensemble CQL on sparse success |
| Features (smoke) | Proprio state \(s\in\mathbb{R}^8\) (joint+gripper); matches DROID server |

## Role mapping

| CF | MolmoAct2 CF |
|---|---|
| \(v\) | Frozen continuous action expert (`predict_action`) |
| \(G\) | Zero-init MLP residual after `predict_action` |
| \(Q\) | Ensemble MLP + CQL |
| Deployed | \(a=\mathrm{clip}(a_v+G_\phi(s,\mathrm{sg}[a_v]))\) |

## Phases

### Phase 0 — Buffer

Download one train shard → stratified H5 replay \(\mathcal{D}^+\cup\mathcal{D}^-\) with \(p_+\approx 0.4\), sparse \(r_T\in\{0,1\}\).

### Phase 1 — Critic (+ optional AE BC)

- Train CQL critic on \((s,a)\) with Monte-Carlo sparse returns.
- Keep \(G\equiv 0\).
- Optional: light AE BC with frozen VLM (`--ae_bc_steps`); smoke default `0`.

### Phase 2 — Residual RL offline

\[
\mathcal{L}_G=-\mathbb{E}[Q(s,\mathrm{clip}(a_v+G))]+\lambda(\mathbb{E}\|G\|^2-\delta)
\]

Gradients only into \(G\) + Lagrange dual. During offline smoke, \(a_v\) is the **dataset action** (expert demo); at deploy, \(a_v\) is MolmoAct2 output.

### Smoke eval

Same first 10 `FrankaPickDroidMiniBench` eps, horizon 500, `G=0` vs `G` on; report SR + role-gap \(\Delta\).

**Pass criteria:** end-to-end runs; \(G=0\) does not hard-regress vs 6/10; \(G\) flat or better.

## Smoke results (2026-07-31)

| Policy | At-end / oracle (n=10) |
|---|---:|
| Baseline MolmoAct2-DROID | 6/10 |
| CF \(G=0\) | 4/10 |
| CF uncapped \(G\) | 0/10 |
| CF capped \(G\) (`max_delta=0.05`) | **5/10** |

Role-gap \(\Delta=+10\%\) vs \(G=0\). Full writeup: [`molmospaces_molmoact2_cf_smoke.md`](molmospaces_molmoact2_cf_smoke.md).

### Residual note

Absolute joint actions require a **tanh-capped** residual (`max_delta≈0.05` in normalized space). Uncapped \(G\) (RMS≈0.22) collapsed SR to 0/10.

## Full online train loop

Script: [`molmoact2_cf/train_full.py`](molmoact2_cf/train_full.py)

- Frozen MolmoAct2 server (`--disable_g`) provides \(a_v\)
- Client-side capped \(G\) + online CQL/\(G\) updates after each Pick env episode
- Metrics appended to `metrics.jsonl` **every 100 environment episodes**
- Default: full MiniBench **1000** eps → `runs/molmoact2_cf_full/`

### 8-GPU parallel (4 instances / GPU = 32 shards)

```bash
bash my_exps/molmoact2_cf/launch_parallel_full.sh \
  --num_gpus 8 --instances_per_gpu 4 --total_episodes 1000 --log_every 10
bash my_exps/molmoact2_cf/aggregate_parallel_metrics.sh \
  my_exps/molmoact2_cf/runs/molmoact2_cf_full_8x4
```

Each instance: MolmoAct2 server on `:8000+g` + `train_full` shard on the same GPU (~14GB×4 ≈ 56GB / 80GB).
Output: `runs/molmoact2_cf_full_8x4/shard_*/`.

## 100M env-step experiment

```bash
nohup bash my_exps/molmoact2_cf/launch_100m.sh > /tmp/molmoact2_cf_100m_launch.log 2>&1 &
```

- Global target: **100,000,000** valid Pick sim steps (8 × 12.5M)
- Metrics every **1,000,000** steps (+ every 100 valid eps)
- Server **watchdog** restarts dead expert servers
- Rollouts on `/tmp` then deleted
- Output: `runs/molmoact2_cf_100m/`
- Rough ETA: ~10–15 days at ~10 steps/s/GPU × 8
