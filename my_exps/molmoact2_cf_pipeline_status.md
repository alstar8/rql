# MolmoAct2 + ConsensusFlow — What We Did & Current Status

**Date:** 2026-08-01  
**Code:** [`molmoact2_cf/`](molmoact2_cf/)  
**Active run:** [`molmoact2_cf/runs/molmoact2_cf_100m_vlacrit_v3/`](molmoact2_cf/runs/molmoact2_cf_100m_vlacrit_v3/) (8×4 → 100M)  
**Smoke plot:** [`smoke_v2/plots/sr_curves.png`](molmoact2_cf/runs/molmoact2_cf_smoke_v2/plots/sr_curves.png)  
**Warmup:** [`molmoact2_cf_vlacrit_warmup_v3/`](molmoact2_cf/runs/molmoact2_cf_vlacrit_warmup_v3/)

---

## Why the previous training “stopped”

1. **Failed 100M v1** (`molmoact2_cf_100m_vlacrit/`): stopped on purpose — learning divergent (SR~18%, TD/CQL exploding). See [`STOPPED.md`](molmoact2_cf/runs/molmoact2_cf_100m_vlacrit/STOPPED.md).
2. **Smoke v2 online** (`molmoact2_cf_smoke_v2/online_short/`): **not a crash**. It hit the intentional smoke caps `--max_valid_episodes 40` / `--target_env_steps 20000` and exited cleanly. Servers were later torn down when the new 100M launcher started.

---

## Smoke_v2 results (refreshed plot)

![Success rate curves](molmoact2_cf/runs/molmoact2_cf_smoke_v2/plots/sr_curves.png)

| Arm | SR |
|---|---:|
| Fixed-seed G=0 (n=10) | **50%** |
| Fixed-seed G-on (n=10) | **30%** |
| Online short (40 eps / 16.4k steps) | **30%** cum |

Critic was **healthy** (MC~2e-3, CQL≥0, Q∈[0,1]), but **G was inert/harmful**: predicted adv≈2e-4, residual RMS stuck ~0.011, G-on below G=0.

### Algorithm drawbacks found in smoke_v2

1. **Local CQL inside the residual ball** flattened the only Q landscape G can climb.
2. **Absolute `-Q` actor** on a nearly flat sparse-return critic → tiny G gradients.
3. **No residual exploration** → critic never saw nearby actions.
4. **Weak G still deployed** once episode/health gates passed → SR regression vs G=0.

### Fixes landed for v3

| Fix | Detail |
|---|---|
| Far-OOD CQL | Samples **outside** residual radius; hinged, non-negative |
| Advantage actor | Maximize `Q(a_v+G) − Q(a_v)` with live critic mean |
| Residual explore | `explore_residual_std=0.02` while learning |
| Deploy gate | `g_min_advantage=0.005` plus critic health |
| Softer dual | `target_divergence=0.0025`, `initial_alpha=0.3` |
| Tests | 13 CPU tests green (`test_models.py`) |

---

## Active 100M v3 launch

```bash
# Already launched:
# nohup bash launch_100m.sh --instances_per_gpu 4 \
#   --out_root .../runs/molmoact2_cf_100m_vlacrit_v3 \
#   --ckpt .../runs/molmoact2_cf_vlacrit_warmup_v3/molmoact2_cf.pt \
#   > /tmp/molmoact2_cf_100m_vlacrit_v3_launch.log 2>&1 &
```

| Item | Value |
|---|---|
| Layout | 8 GPUs × 4 = **32 shards**, ports 8000–8031 |
| Target | **100M** valid env steps (3.125M / shard) |
| Buffer | `pick_buffer_vla_g0.npz` (matched MolmoAct2 G=0) |
| Init | `molmoact2_cf_vlacrit_warmup_v3/molmoact2_cf.pt` |
| Logs | `runs/molmoact2_cf_100m_vlacrit_v3/logs/` |

Watch:

```bash
grep -hE 'steps=|METRICS|critic_loss|g_adv' \
  molmoact2_cf/runs/molmoact2_cf_100m_vlacrit_v3/logs/train_g*_gpu*.log | tail
```

### Early health gates (first ~100 valid eps / shard)

- MC/TD finite, CQL ≥ 0, Q ∈ [0, 1]
- `g_adv` should rise above ~0.005 before G deploys widely
- Cum SR should stay nearer fixed-seed G=0 (~0.3–0.5) than the old ~0.18 death spiral
- If SR collapses or TD explodes again: stop, do **not** burn another week

---

## Related docs / scripts

- `plot_smoke_v2.py` — regenerate SR curves
- `warmup_vlacrit.sh` — VLA buffer (optional) + warmup_v3
- `launch_100m.sh` — 100M v3 launcher (requires warmup_v3 + VLA buffer)
- `launch_smoke_v2.sh` — fixed-seed + short online gate
