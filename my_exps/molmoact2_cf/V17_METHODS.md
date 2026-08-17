# V17 methods: kettle-matched offline + safe residual/flow RLT+CF

Status: **launch via** `bash launch_v17_chain.sh` → `runs/rlt_cf_v17_kettle`.  
Predecessor: V16 improved (d512/L4 on demo1k) — critic/AE healthy but **actor always-collect collapsed train SR** (~0.18→~0.08); gate never opened; offline was MolmoBot+DROID, online was house0 kettle.

## Diagnosis carried into V17

1. Offline/online domain mismatch (demo1k ≠ house0 kettle).
2. `--always_collect_actor` after short BC destroyed VLA SR before Q/empirical gates could help.
3. Per-chunk mixture (V15) broke empirical episode labels; V16 set mixture=0.
4. Residual actor entered online nearly cold (no offline actor BC); capacity bump alone did not help.

## V17 changes

| Area | V16 improved | V17 |
| --- | --- | --- |
| Offline data | MolmoBot+DROID demo1k | **house0 kettle** reference VLA collect (`collect_house0_kettle_offline.sh`) |
| Arch | z=256 / d=512 / 4 layers | **same** |
| Residual actor offline | none | **BC `q_coef=0`, β=5, 5k steps** |
| Flow actor offline | BC (demo1k) | BC on **kettle** re-encoded chunks, β=5 |
| Online collect | always π_θ after ep50 | **episode-level mixture p=0.15**; always-collect **off** |
| BC / Q schedule | BC 50 then q_coef=1 cliff | **BC 150** then **q_coef ramp 100 eps** |
| Soft β | 2.0 | **5.0** |
| Explore σ | 0 | 0 |
| Arms | residual+flow CF (4 shards) | **same** |

## Launch

```bash
# Full chain (stops V16 improved, collect → pretrain → online)
bash launch_v17_chain.sh

# Or stepwise:
bash collect_house0_kettle_offline.sh          # ~150 eps × 8 shards
bash pretrain_rlt_house0_kettle_d512_l4.sh      # AE + residual/flow
V17_MODE=long FRESH=1 bash launch_v17_rlt_cf.sh
```

Ports: `8730–8737`. Logs: `tmp/rlt_cf_v17_kettle_logs/`.  
Do not overwrite `rlt_cf_v16_*`.

## Success checks

- First ~150 eps: mostly `episode_collect_policy=reference`; cumSR stays near VLA (~0.15–0.20).
- Mixture episodes populate `empirical_actor_*` / `empirical_ref_*` without `empirical_insufficient_episodes`.
- Actor LCB / emp ΔSR should not dive to V16’s −0.03 / 0.07 actor SR while reference stays ~0.18.

## Collect note (2026-08-15)

Intermediate multi-GB `token_replay` / `chunk_token_replay` NPZ dumps every 10 eps stalled NFS collectors (~20+ min hangs, idle GPUs). Fix: `--token_ckpt_every_episodes` default **0** (flush tokens **only at end**); model/chunk ckpts still every 10.

**Outcome:** rollouts reached 150 eps/shard (SR ~0.20–0.26) but **final token flush killed** processes (0-byte `.token_replay.npz.tmp.npz`, no NPZs). See **V18_METHODS.md** for robust re-collect + online 80% plan.
