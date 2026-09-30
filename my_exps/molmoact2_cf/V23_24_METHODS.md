# V23_24

V22_24 on `turning_on_radio` (OmniGibson / BEHAVIOR-1K, instance **301**).
Frozen specialist `pi05-b1kpt12-cs32`. Inference never evaluates a critic.

This is not LoRA-on-the-expert (that is V23). Leave earlier radio runs
(`cf_v23_ae*`, CFGRL/AWAC) on disk; do not resume them.

```
RGB + proprio + prompt
        │
   frozen PaliGemma          → prefix tokens (stop-grad)
        │
   token AE  g_φ            → z ∈ R^{256}   (RL state only)
        │
   frozen gemma_300m        → ã            (specialist chunk)
        │
   V = (ã − x_t)/(1 − t) + MLP_θ     analytic OT + zero-init residual
   G_φ(s, x_t, t; V)                 bounded guidance, G ≡ 0 at init
        │
   a = Euler(V + G, noise)           N = 10, paper clock t=0 noise → t=1 data
        │
   Q_k(s, x, t)  k = 1..10           reverse-state TD, y clipped to [0, 1]
        │
   L = L_td + α L_bc + β ‖a_V − ã‖² + λ_π L_la + L_distill + L_ro
```

`λ_π` gates only the lookahead. Distillation, BC, the β-anchor, and TD stay
on in every phase. Because the OT skip is exact, `G = 0` copies the
specialist **at init** (no BC wait). Stage 0 (`λ_π = 0`) keeps the residual
at zero; all value improvement is sample-time `G`.

Code: `openpi_comet/` (`models/pi0_cf.py` `compute_cf_v23_24_loss`,
`training/cf_live.py`, `scripts/train.py`,
`scripts/run_cf_v23_24_radio.sh`). Algorithm reference:
`pi05_rl_token/rlt/v22_24.py`.

---

## Frozen VLA

PaliGemma (`gemma_2b`) and base `gemma_300m` (**no LoRA**) stay frozen.
Checkpoint `pi05-b1kpt12-cs32`. The VLA supplies prefix tokens and `ã`
(Euler through the frozen expert, OpenPI clock `t=1→0`).

## RL token

```
z     = g_φ(sg(prefix_tokens))          CFTokenPool, z_dim=256, 8 queries
L_ro  = MSE(decoder(z), sg(prefix_tokens))
s     = [ z_disk(prefix), normalize(proprio) ] ∈ R^{288}
```

`z_disk` is the target-pool snapshot after AE. Live AE trains `L_ro` only.
Proprio is 23-d padded to `action_dim=32`. Batch 8.

## Chunk MDP

One 32-step action chunk is one MDP step (30 Hz env, re-inference when the
queue is empty). Timeout 4300 env steps ≈ 135 chunks.

| | |
| --- | --- |
| `s` | `[z, proprio]` at chunk start |
| `a` | executed chunk, flat `32 × 32` padded |
| `ã` | specialist chunk for the same `s` |
| `r` | 1 iff the episode succeeds inside the chunk, else 0 |
| `done` | 1 if the episode ends inside the chunk |
| `s'` | next chunk start (no successor → drop, do not bootstrap) |

`cf_ae_replay.py` sets `chunk_reward = 1` iff any env step in the chunk has
`reward > 0.5`. Only the finishing success chunk is positive. `γ = 0.99`
**per chunk** (`cf_discount_per_chunk=1`), not `γ^32`.

Replay: 100 pt12 trajs (`outputs/pt12_cs32_radio_10x10_states`) for AE/AC;
online filter instance 301. Stratified fail / success-prefix / terminal.

## Networks

Paper clock: `t = 0` noise, `t = 1` data. Chunk dim 1024.

| module | signature | size |
| --- | --- | --- |
| **V** | `(ã−x_t)/(1−t) + MLP(s, x_t, t, ã)` | 4 × 512, no LN, **zero-init head** |
| **W** | `W(s, x_t, t) → R^{1024}` | 4 × 512, LN, **zero-init head** |
| **Q** | `Q_k(s, x, t)`, `K = 10` | 4 × 512, LN, per head |

A 4×512 MLP cannot learn 1024-d OT from scratch on radio (abandoned `sr0`
arm: `actor_ref_rmse` stuck at 0.85, probe 0/10). The analytic skip makes
`Euler(V)=ã` at init. `W=0` ⇒ `G=0` until distillation.

Adam `lr=3e-4` (warmup 200) on `(V residual, W, timed Q, live AE)`. EMA on
`V` 0.999. Critic Polyak `τ=0.005`. `W` is not Polyak'd. Freeze: VLM +
expert + `cf_target_*`.

## Deployed field

```
w          = W(s, x, t)
u          = clip_{‖·‖≤1}(w)
trust      = sg(‖u‖)
v̂         = sg(V) / ‖V‖
kill       = 1 − trust^{p}                         p = 2
u_cf       = clip_{‖·‖≤1}( u − kill · min(⟨u, v̂⟩, 0) · v̂ )
cos        = ⟨u, v̂⟩ / (‖u‖ + ε)
damp       = clip_{[0,1]}( 1 − ρ · max(cos, 0) · trust )     ρ = 0.25
G_φ        = λ · t · clip_{‖·‖≤1}(u_cf · damp)
```

`λ = 0.5`. Serve: EMA `V` + live `W`, `Euler(V+G)`. Anchor is on unguided
`Euler(V)`. Live payload: target pool, EMA `V`, live `W`. Paired eval:
`OPENPI_CF_APPLY_GUIDE=0` turns `G` off.

```
x_0 = ε,   x_{i+1} = x_i + (1/N) (V + G),   t_i = i/N,   N = 10
```

## Losses

**BC** (`α=1`). OT path. With the skip this is MSE(residual, 0) at init.

```
x_t = (1−t)ε + t ã ,   L_bc = MSE(V(s,x_t,t,ã), ã−ε)
```

**Anchor.** Raw sum, not per-dim (`cf_anchor_normalize=0`). `β=100` AC,
`β=1` online.

```
a_V = Euler(V, ε'; guided=False)
L_β = β · mean_batch ∑_j (a_V − ã)²_j
```

**Lookahead** (`λ_π = cf_actor_coef`). `G` and critic weights are stop-grad.

```
x⁺ = x_t + (V + sg(G)) · min(1/N, 1−t)
L_la = − mean_k Q_k(s, x⁺, min(t+1/N, 1))
```

Stage 0: `λ_π=0` (AC and default online). Stage 1 would set `λ_π=1`.

**Distill** (`cf_distill_coef=1`). Only loss on `W`. One target head per row:

```
z_k = ∇_{x} Q̄_k / (‖∇_x Q̄_k‖ + 0.01 · mean_batch ‖∇_x Q̄_k‖)
L_distill = mean_batch ‖W − z_k‖²
```

**Reverse-state TD.** From executed `a` at `t=1`, integrate `V+G` backward
a random `δ ~ U[0,1]` or `k/N`. Bootstrap at fresh noise `(s', ε', 0)`:

```
y = clip( r + γ (1−done) mean_k Q̄_k(s', ε', 0), 0, 1 )
L_td = mean (y − Q_k(s, x, t))²
```

Two reverse samples per step. `cf_critic_coef=1`. Kill if `|Q|>2` or
`actor_ref_rmse>0.2` after a sidecar `${EXP}_V_COPY_OK` (first non-AE log
with rmse ≤ 0.2). With the OT skip that sidecar is written immediately.

## Training protocol

1. **Data.** 100 pt12 `state_action.npz`.
2. **AE.** 8000 steps, `ae_only=1`, `ae_coef=1`, `freeze_critic=1`.
3. **AC.** 8000 steps, `λ_π=0`, `β=100`, live TD + distill + `L_ro`, all 100
   trajs. Residual stays ~0; `W` grows.
4. **Probe.** 10 actor-only episodes on 301, not stored. `G` on.
5. **Online.** 1 collector, 300 × {1 ep on 301 + `UTD=5 × n_chunks`}.
   Stage 0, `β=1`, replay 301.
6. **Eval.** 301, paired `λ=0.5` vs `APPLY_GUIDE=0`, 10 eps each.

Driver `openpi_comet/scripts/run_cf_v23_24_radio.sh`, watchdog
`cf_v23_24_watchdog.sh`, exp `cf_v23_24_301`. Tests:
`JAX_PLATFORMS=cpu python openpi_comet/scripts/test_cf_v23_24.py`.

## Defaults

`OPENPI_CF_V23_INSTANCE=301`, `ACTOR_MAX_COEF=0`, `OPENPI_CF_ONLINE_BETA=1`,
`OPENPI_CF_ANCHOR_NORMALIZE=0`, `OPENPI_CF_GUIDANCE_COEF=0.5`,
`OPENPI_CF_CRITIC_COEF=1`, `OPENPI_CF_DISCOUNT_PER_CHUNK=1`,
`OPENPI_CF_LR=3e-4`, `OPENPI_CF_UTD=5`, `OPENPI_CF_V23_ROUNDS=300`,
`OPENPI_CF_KILL_REF_RMSE=0.2`.

---

## Radio vs kettle

| | V22_24 kettle | V23_24 radio |
| --- | --- | --- |
| Stack | PyTorch `pi05_rl_token` | JAX `openpi_comet` |
| Frozen VLA | droid pick-full | `pi05-b1kpt12-cs32` |
| `V` | learned 4×512 OT | **analytic OT + zero residual** |
| `C` / dim | 8 / 64 | 32 / 1024 padded |
| Reward | discounted in-chunk sum | sparse 0/1 success chunk |
| `γ` | `0.99^8` | `0.99` per chunk |
| Batch / collectors | 256 / 4 | 8 / 1 |
| Split | jittered held-out | instance 301 only |

---

## Log (`cf_v23_24_301`, snapshot 2026-09-10 22:38 UTC)

AE 8k + AC 8k (`β=100`, `λ_π=0`) on the 100-traj pt12 buffer, then online
stage 0 `β=1` on 301. Ckpt **25292**, live **25293**. No KILL marker.
`actor_ref_rmse = 0` from the first AC log (`V_COPY_OK` written).

### Train

| step | phase | `L_bc` | rmse | `w_norm` | `q_mean` | `L_td` | `g_over_v` | `L_distill` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | AE | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 8000 | AC start | 0 | 0 | 0 | 0.13 | 0.12 | 0 | 0.98 |
| 12000 | AC | 0.005 | 0 | 0.88 | 0.45 | 0.002 | 0.0071 | 0.19 |
| 16000 | AC end | 0 | 0 | 0.94 | 0.41 | 0.002 | 0.0068 | 0.06 |
| 20000 | online | 0 | 0 | 0.89 | 0.26 | 0.003 | 0.0062 | 0.09 |
| 25280 | last | 0.002 | 0 | 0.92 | 0.21 | 0.004 | **0.0072** | 0.07 |

`W` left zero (0 → 0.94). `Q` stayed in `[0, 1]`. Logged `g_over_v` is
mean(`‖G‖/‖V‖`) on the BC path: **0.72%**. `G = λ t clip(W)` with `λ=0.5`,
so `‖G‖ ≤ 0.5` while OT `V` is `O(√1024)`. Stage 0 does not move `V`; online
only updated `Q` (0.41 → 0.21) as 301 timeouts entered replay.

### Probe (10 actor-only, not stored, `G` on)

**9/10.** Timeout on ep 6 (4300). Wins: 1093, 1334, 1638, 1803, 1813, 1965,
2105, 2187, 2489 steps. Frozen pt12 on 301 is **5/10**. That gap is sample-time
`G` on an OT copy of pt12.

### Online stored (22 / 300)

**14/22 = 63.6%.** Last-10 **5/10 = 50%.** Peak last-10 was 10/13 (77%) around
ep 13, then timeout pairs 14–15, 18–19, 21. All eight losses are full 4300
timeouts. Win mean 1884 steps (1226–3913).

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| win | 1 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 |
| steps | 1420 | 1288 | 4300 | 4300 | 1226 | 3913 | 2698 | 2180 | 4300 | 1325 | 1330 | 1870 | 2072 | 4300 | 4300 | 1514 | 1328 | 4300 | 4300 | 1254 | 4300 | 1358 |

Round 23 collected a 4300 timeout at 20:24 UTC, then train/serve died:
container NVML `No visible GPU devices` while the host still has two idle
RTX 5090s. Driver is alive and retrying; ~286 serve deaths. That timeout is
**not** in replay (`rm -rf` on retry). Paired G-on / G-off eval has not run.

### Does the policy differ?

| Vs | `V` | `G` | Deployed |
| --- | --- | --- | --- |
| Frozen pt12 | Same: `Euler(V)=ã` | On, `λ=0.5`, `‖G‖/‖V‖=0.72%` | Yes at sample time (probe 9/10 vs 5/10) |
| AC step 16k | Same, rmse 0 | Same scale (`w_norm` 0.94 → 0.92) | No. Online moved `Q`, not the actor field |
| Abandoned `sr0` | Random MLP, rmse 0.85 | Irrelevant | 0/10 probe, 0/40 online. Do not mix |

Deployed policy is **pt12 plus a unit-ball `G` at 0.7% of `‖V‖`**. Enough to
beat the 5/10 floor; not a new online actor.

### vs other radio arms

| arm | probe | online | last-10 | `V` copy |
| --- | ---: | --- | ---: | --- |
| V23_24 current (stage 0, `β=1`) | **9/10** | 14/22 (63.6%) | 50% | rmse 0 |
| V23_24 `sr0` (no OT skip, abandoned) | 0/10 | 0/40 | 0% | rmse 0.85 |
| Frozen pt12 | 5/10 | — | — | identity |
| V23 LoRA copy-hold | — | stuck ~67% | ~67% | LoRA `B=0` |
