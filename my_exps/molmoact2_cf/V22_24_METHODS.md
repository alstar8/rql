# V22_24

A flow policy on a frozen π0.5 VLA. The improvement channel is split: a
behavior velocity `V` stays a specialist copy, a bounded guidance field `G` is
distilled from an ensemble of time-conditioned critics, and (optionally) `V`
absorbs `G` through a one-step lookahead. Inference never evaluates a critic.

```
RGB + proprio + prompt
        │
   frozen PaliGemma          → prefix tokens (stop-grad)
        │
   token AE  g_φ            → z ∈ R^{256}   (RL state only)
        │
   frozen action expert     → ã            (specialist chunk)
        │
   V_θ(s, x_t, t, ã)        behavior velocity
   G_φ(s, x_t, t; V)        bounded guidance, G ≡ 0 at init
        │
   a = Euler(V + G, noise)                 N = 10 steps
        │
   Q_k(s, x, t)  k = 1..10                  reverse-state TD, y clipped to [0, 1]
        │
   L = L_td + α L_bc + β ‖a_V − ã‖² + λ_π L_la + L_distill + L_ro
```

`λ_π` gates only the lookahead. Distillation, BC, the β-anchor, and TD stay
on in every phase. `G = 0` copies the specialist exactly once `V` has been
BC-pretrained.

Code: `pi05_rl_token/rlt/v22_24.py` (`algorithm=v22_24`).

---

## Frozen VLA

PaliGemma (`gemma_2b`, token width 2048) and the base action expert stay frozen.
Checkpoint `pi05_droid_finetune_pick_full_v3_39999`. The VLA supplies (i) prefix
tokens for the RL token, (ii) the reference chunk `ã`. Online RL does not update
VLM or expert weights.

## RL token

Encoder–decoder bottleneck on stop-grad prefix tokens (`RLTokenAE`,
`z_dim=256`, `d_model=256`, 4 heads, 2 layers):

```
z     = g_φ(sg(prefix_tokens))
L_ro  = MSE(decoder(z), sg(prefix_tokens))
```

RL always reads

```
s = φ(obs) = [ z_disk(prefix), normalize(proprio) ] ∈ R^{272}
```

`z_disk` is the on-disk encoder. Live AE weights train `L_ro` only; they do not
move the critic's `z`. `token_batch_size=8`.

## Chunk MDP

One inference commits one action chunk. That chunk is one MDP step.

| | |
| --- | --- |
| `s` | `[z, proprio]` at chunk start |
| `a` | executed chunk, flat (`C × 8`) |
| `ã` | specialist chunk for the same `s` |
| `r` | discounted sum of env rewards inside the chunk (`γ^i` per env step) |
| `done` | 1 if the episode ends inside the chunk |
| `s'` | next chunk start (truncated tails with no successor are dropped, not bootstrapped) |

Replay starts from 100 frozen-VLA episodes (kettle: 3977 rows, 17 reward
rows). Online rows are appended. Sampling is uniform over stored chunk
transitions (`batch_size=256`). Capacity 400k; nothing is evicted.

Kettle: `C = 8`, action is 7 arm deltas + gripper, `horizon=400`,
`gate_step=0` (RL from the first env step). Train pool `0-11`, pretrain collect
pool `0-47`, held-out eval is a disjoint jittered bench.

## Networks

All MLPs: ReLU, no dropout. Time `t` is one extra concatenated feature.
PyTorch flow-matching clock: `t = 0` noise, `t = 1` data.

| module | signature | size |
| --- | --- | --- |
| **V** `FlowActor` | `v(s, x_t, t, ã) → R^{64}` | 4 × 512, no LayerNorm, `ref_dim = 64` |
| **G student** `W` | `W(s, x_t, t) → R^{64}` | 4 × 512, LayerNorm on |
| **Q** `EnsembleCritic` | `Q_k(s, x, t) → R`, `K = 10` | 4 × 512, LayerNorm on, per head |

`W`'s last Linear is zero-initialised, so `G ≡ 0` until distillation moves it.
Adam `lr = 3e-4` on two optimizers: `(V, W, AE)` and `Q`. EMA on `V` with
`ema = 0.999`. Critic Polyak `τ = 0.005`.

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

`λ = 0.5` (`cf_guidance_coef`). The `t` gate shuts guidance off at noise.
Radial clip, trust-weighted conflict contraction against `sg(V)`, and residual
damping keep `G` from cancelling `V`. Eval can override `λ` (`guidance_coef`);
`λ = 0` is the paired intervention (guide off).

Forward unroll, `dt = 1/N`, `N = 10`:

```
x_0 = ε ~ N(0, I)
x_{i+1} = x_i + dt · ( V_θ(s, x_i, t_i, ã) + G_φ(s, x_i, t_i; V_θ) )
t_i = i / N
a   = x_N
```

Serve uses the EMA copy of `V` (`explore=False`). The executed policy is
`Euler(V + G)`; the anchor (below) is applied to the *unguided* unroll
`Euler(V)` so `V` cannot learn to cancel `G`.

## Losses

### Behavior cloning

OT path to the specialist chunk:

```
ε ~ N(0, I),   t ~ U(0, 1)
x_t = (1 − t) ε + t ã
L_bc = MSE( V_θ(s, x_t, t, ã),  ã − ε )
```

`α = 1`.

### Endpoint anchor

```
a_V = Euler(V_θ, ε'; guided=False)
L_β = β · mean_batch  ∑_j (a_V − ã)²_j
```

Sum of per-dimension squares, not divided by `dim(a)`. `β = 100` in AC
pretrain. Online kettle currently uses `β = 1` (a `β = 100` online pair was
started and stopped at ~30 episodes).

### One-step guided lookahead (`λ_π`)

```
g     = sg( G_φ(s, x_t, t; V) )
Δ     = min(1/N, 1 − t)
x⁺    = x_t + (V + g) · Δ
t⁺    = min(t + 1/N, 1)
L_la  = − mean_k Q_k(s, x⁺, t⁺)
```

`G` is stop-grad; `V` is the only module that absorbs the Q signal. Critic
weights are not in the actor optimizer, so `−Q` cannot inflate `Q`.

`λ_π = cf_actor_coef`:

- **0** — `V` gets BC + anchor only. All value improvement must arrive through
  the distilled `G`.
- **1** — full joint method.

AC pretrain always uses `λ_π = 0`.

### Distillation

One of the `K = 10` *target* heads is sampled per row. Common-scale-normalized
gradient of that head w.r.t. `x_t`:

```
k ~ Unif{1..K}
g_k = ∇_{x_t} Q̄_k(s, x_t, t)
m_B = mean_batch ‖g_k‖
z_k = g_k / ( ‖g_k‖ + c · m_B )                 c = 0.01
L_distill = mean_batch ‖ W(s, x_t, t) − z_k ‖²
```

`cf_distill_coef = 1`. Target heads (`Q̄`) are stop-grad. This is the only
loss on `W`.

### Reverse-state TD

The critic is not trained at the executed endpoint `(a, 1)` alone. From the
executed chunk, integrate `V + G` *backwards* a random fraction `δ`:

```
δ ~ U[0, 1]  or  k/N , k ∈ {0..N}     (equal probability, per row)
x ← a,  t ← 1
for i = 1..N:
    x ← x − (V + G) · (δ / N)
    t ← max(t − δ/N, 0)
```

Bootstrap at fresh noise, not an actor unroll:

```
ε' ~ N(0, I)
y  = clip( r + γ^C (1 − done) ( mean_k Q̄_k(s', ε', 0) − ρ std_k Q̄_k ),  0, 1 )
```

`γ = 0.99` per env step, so the chunk discount is `γ^C = 0.99^8 ≈ 0.923`.
`ρ = 0`. Expectile `τ_e = 0.5` (MSE on all 10 heads):

```
L_td = mean_{k, batch} (y − Q_k(s, x, t))²
```

Live `Q` is not clipped. Two critic steps per actor step
(`critic_updates_per_actor=2`). Online `UTD = 5` per absorbed row, with
actor/critic updates every 100 stored env steps.

## Training protocol

1. **Data.** 100 frozen-VLA rollouts → decision-chunk replay. Token shards for
   `L_ro` (the npz buffer does not store prefix tokens).
2. **AE.** Scene AE is trained first and frozen as `z_disk`. `L_ro` continues
   during AC and online on a live copy (`ae_finetune=true`).
3. **AC pretrain.** 8000 steps, `λ_π = 0`, `β = 100`, live reverse-TD, live
   distillation. `V` becomes a specialist copy; `W` grows under the critic.
4. **Probe.** 10 actor-only episodes, **not** written to replay.
5. **Online.** 4 collectors, 3 EGL slots, 300 stored episodes. Two arms from the
   same AC checkpoint:
   - Stage 0: `λ_π = 0` (distilled-`G` routing)
   - Stage 1: `λ_π = 1` (lookahead + distillation)
6. **Eval.** Held-out bench, paired `λ = 0.5` vs `λ = 0` on the same seed
   stream. The on/off gap is `G`'s sample-time contribution.

Publish the **EMA** `V` plus live `W`. Kill if `|Q| > 2` or `actor_ref_rmse`
leaves the chunk scale (~0.2).

---

## Kettle log (this implementation)

AC pretrain, 8000 steps, `β = 100`, `λ_π = 0`, 1.73 h, buffer 3977 rows:

| step | `L_bc` | `actor_ref_rmse` | `w_norm` | `q_mean` | `L_td` | `L_ro` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.04 | 1.021 | 0.00 | −0.13 | 0.047 | 0.025 |
| 2000 | 1.21 | 0.012 | 0.30 | 0.11 | 0.0004 | 0.026 |
| 8000 | 0.33 | **0.0039** | 0.44 | 0.47 | 0.0001 | 0.019 |

`Q` stayed in `[0, 1]`. `G` left zero under distillation (`w_norm` 0 → 0.44).
`g_over_v` settled near 0.03 (guidance small relative to `V`).

Probe after that checkpoint (10 actor-only episodes, not stored):

| arm | probe |
| --- | ---: |
| Stage 0, then online `β = 1` | **10/10** |
| Stage 1, then online `β = 1` | **8/10** |
| Stage 0, online `β = 100` (stopped) | 5/10 |
| Stage 1, online `β = 100` (stopped) | 6/10 |

Online, stopped 2026-09-02 (node lost the extra GPUs). Both `β = 1` arms
finished 300 + paired held-out on the 8×H100 (`launch_v22_24_8gpu.sh`).
`β = 100` arms were not resumed. Held-out of those stopped checkpoints
started 2026-09-09 (`eval_kettle_beta100.sh`) and was paused the same day
after S0 finished; S1 G-on was killed mid-eval, S1 G-off never started.

| arm | eps | stored SR | last-10 | G on | G off |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stage 0 `β = 1` | 300 | 211/300 = 70.3% | 1.0 | 59/64 | 41/64 |
| Stage 1 `λ_π = 1`, `β = 1` | 300 | 194/300 = 64.7% | 0.4 | **64/64** | 28/64 |
| Stage 0 `β = 100` (stopped) | 31 | 16/31 = 52% | 0.2 | 28/64 | 11/64 |
| Stage 1 `β = 100` (stopped) | 37 | 23/37 = 62% | 0.6 | paused | — |

Stopped-arm counts come from `metrics.jsonl`. `actor_ref_rmse` ≈ 0.004,
`w_norm` ≈ 0.38 on both β=100 arms at the stop. Resume S1 evals with
`bash eval_kettle_beta100.sh` (skips existing `result.json`).

## Pick-18 sweep (`runs/pick18_v22_24/`, stage 0, `β = 100`)

Same recipe as the kettle AC pretrain, then 300 online with `λ_π = 0`.
Paired held-out eval48 (64 rollouts). All 18 tasks finished (bottle/knife
`G`-off completed 2026-09-06).

| | probe | online (300) | gOn (`λ=0.5`) | gOff (`λ=0`) |
| --- | ---: | ---: | ---: | ---: |
| **Macro** | **107/180 = 59.4%** | **4003/5400 = 74.1%** | **995/1152 = 86.4%** | **672/1152 = 58.3%** |

| task | probe | online | last-10 | gOn | gOff |
| --- | ---: | ---: | ---: | ---: | ---: |
| bottle | 0/10 | 0/300 | 0.0 | 0/64 | 0/64 |
| bowl | 10/10 | 293/300 | 1.0 | 64/64 | 64/64 |
| box | 9/10 | 244/300 | 0.8 | 55/64 | 54/64 |
| cup | 10/10 | 294/300 | 1.0 | 64/64 | 64/64 |
| desk_mug | 10/10 | 289/300 | 1.0 | 64/64 | 32/64 |
| fork | 10/10 | 298/300 | 1.0 | 64/64 | 64/64 |
| fruit | 0/10 | 129/300 | 0.4 | 44/64 | 29/64 |
| kettle | 10/10 | 207/300 | 0.7 | **59/64** | 11/64 |
| knife | 1/10 | 176/300 | 0.7 | 50/64 | 5/64 |
| ladle | 9/10 | 259/300 | 0.9 | 60/64 | 54/64 |
| pot | 0/10 | 208/300 | 0.9 | 62/64 | 31/64 |
| remote | 10/10 | 286/300 | 1.0 | 61/64 | 45/64 |
| shaker | 0/10 | 131/300 | 0.5 | 51/64 | 12/64 |
| soap_dispenser | 5/10 | 263/300 | 0.9 | 62/64 | 26/64 |
| spatula | 10/10 | 285/300 | 0.8 | 64/64 | 63/64 |
| spoon | 10/10 | 290/300 | 1.0 | 64/64 | 64/64 |
| spray_bottle | 0/10 | 141/300 | 0.4 | 53/64 | 4/64 |
| tissue | 3/10 | 210/300 | 1.0 | 54/64 | 50/64 |

Bottle collapsed (0 probe, 0 online, 0 held-out). Kettle gOn **59/64 = 92.2%**
beats V22_23 kettle held-out 52/64 = 81.3%. The on/off gap is `G`'s
sample-time contribution.

## Pick-18 sweep, online `β = 1` (finished)

Same AC checkpoints as `runs/pick18_v22_24/` (8000 steps, `λ_π = 0`,
`β = 100`). Online is stage 0 (`λ_π = 0`) with **`β = 1`**, then paired
held-out eval. Run dir `runs/pick18_v22_24_beta1/`. Finished 2026-09-07
on the 8×H100 (`launch_v22_24_8gpu.sh`).

| | probe | online (300) | gOn (`λ=0.5`) | gOff (`λ=0`) |
| --- | ---: | ---: | ---: | ---: |
| **Macro** | 112/180 = 62.2% | 3644/5400 = 67.5% | **1069/1152 = 92.8%** | 651/1152 = 56.5% |

β=100 wins stored online (−6.6 pp). β=1 wins held-out G-on (+6.4 pp; 10
tasks up, 7 tied, only spray_bottle down). G-off is a wash. The extra
held-out is guidance at sample time, not a stronger `V`. Bottle is the
rescue: gOn 59/64 vs β=100 0/64.

Last-10 SR vs episode, mean ± std across the 18 tasks (offline AC probe
then online from 0): V21, V22, V22+V23, V22.24 at $\beta{=}100$.
`runs/pick18/plots/v22_family_sr_curves.pdf`.

---

## Is it better than V22_23? (kettle only)

Same scene, same frozen VLA, same AE, same 8000-step AC then 300-online
recipe. V22_23 kettle is `pick18_v22v23/kettle` (`flow_rlt`, compose, live TD,
`β = 100`): probe **2/10**, online **185/300 = 61.7%**, last-10 **0.7**,
held-out **52/64 = 81.3%**.

### Offline / probe — yes

After AC pretrain the deployed policy is already much stronger than V22_23's
specialist copy:

- Probe **5–10/10** vs **2/10**. The 10/10 and 8/10 probes are the same AC
  checkpoint with `G` on; V22_23's compose field is driven to 0 by BC, so
  probe is essentially the frozen VLA (collect floor 14–17%).
- Distillation moves `W` during the 8000 steps (`w_norm` 0 → 0.44). That is
  the sample-time improvement channel while `λ_π = 0`.
- `V`'s copy is tight (`actor_ref_rmse = 0.0039`) but not as exact as compose
  identity (`7×10^{-5}`). That is expected: a reference-conditioned corrector
  has to *learn* the OT field, compose adds it analytically.
- Reverse-TD `Q` is time-conditioned and bounded (`q_mean → 0.47`,
  `L_td → 10^{-4}`). It is a different object than V22_23's endpoint `Q`
  (`q_mean ≈ 0.04` at step 8000), so those two scalars are not ranked against
  each other.

### Online — not yet, and not on the finished metric

Matched **early** kettle episodes, V22_24 is ahead:

| first N stored | V22_23 | V22_24 S0 `β=1` | V22_24 S1 `β=1` | V22_24 S0 `β=100` | V22_24 S1 `β=100` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 2/10 (last-10 0.2) | **10/10** (1.0) | 4/10 (0.3) | 9/10 (0.8) | 8/10 (0.7) |
| 23 | 8/23 (0.4) | **19/23** (0.8) | 4/14 (0.2) | 14/23 (0.3) | 15/23 (0.7) |
| 31 | 8/31 (0.1) | — | — | 16/31 (0.2) | **19/31** (0.5) |

V22_23 then climbed: 29/100 at ep 100, **63/150** with last-10 = 1.0, finish
**185/300** and **81% held-out**. V22_24 has no 100+ episode number and no
held-out eval. Stage 1 at `β = 1` is currently *worse* than V22_23 at the
same count (lookahead + weak anchor). Stage 0 at `β = 1` is the promising
early curve; it is also the arm that cannot be compared to V22_23's finished
held-out yet.

**Claim that is supported today:** V22_24 is better at the *offline* phase
on kettle (probe), and the Pick-18 stage-0 `β=100` kettle held-out with `G`
on is already 59/64 vs V22_23's 52/64. **Claim that is not supported yet:**
the kettle `β=1` online arms (the ones in the ablation table) are a better
finished online method. Wait for 300 episodes and the paired `λ`-on / `λ`-off
held-out eval on those arms.
