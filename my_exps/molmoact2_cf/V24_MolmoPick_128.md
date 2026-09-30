# V24_MolmoPick_128

A **single** V22_24 ConsensusFlow policy shared across a task pool, instead of
one specialist per task. The end goal is the 128-task MolmoPick train pool
with eval on the 1000-episode validation benchmark; the path there is staged,
and this document now describes **Stage A**, the pilot that decides whether
the shared design holds up at all:

> **Stage A — shared Pick-18.** One shared `V`/`W`/`Q`/AE over the 18 tasks
> that already have a finished specialist number (β=1 sweep, macro **92.8%
> gOn / 56.5% gOff**, `runs/pick18_v22_24_beta1/`). Same bench, same episode
> counts, same recipe — the only change is sharing. If the shared policy
> matches the specialists here, Stage C (128 tasks) is worth its ~10× cost.
> If it collapses, the collapse is cheap and Stage B hardens the weak part.

**Stage A finished 2026-09-13: pass.** Shared held-out gOn **1081/1152 = 93.8%**
vs specialist **92.8%** (`runs/pick18_v24_shared/`). Official MolmoPick val
(1000 unique episodes): shared gOn **777/1000 = 77.7%** vs frozen pi0.5
**635/1000 = 63.5%**.

**Stage B finished 2026-09-14: `λ_π=1` does not beat Stage A.** Same AE,
pretrain, and merged buffer; online `cf_actor_coef=1`. Held-out gOn
**1021/1152 = 88.6%** (−5.2 pp). Stage 0 (`λ_π=0`) stays the default.

```
frozen pi0.5 (pick_full_v3_39999)
        │
   18 pick tasks (the Pick-18 pool, train benches)
        │
   shared token AE (all 18 corpora)  →  re-collect 100 VLA rollouts/task
        │                                encoded by the SHARED AE
   merge 18 buffers  →  30k-step AC pretrain (β=100, λ_π=0, ref-conditioned W/Q)
        │
   online RL: one learner, collectors cycle tasks round-robin
   (β=1, λ_π=0 — the stage-0 recipe), 1800 episodes = 100/task
        │
   per-task paired eval (λ on/off), same protocol as the specialist sweep
```

Code: `pi05_rl_token/rlt/v22_24.py` (`algorithm=v22_24`), pipeline
`pipeline_pick18_v24_shared.sh`. The VLA is never updated; RL learns `V`,
`W`, `Q`, and the live token AE only.

---

## Method (as implemented)

This section is the code, not the plan. Everything below is what
`V2224Agent` in `rlt/v22_24.py` actually does on the Stage A run.

### Frozen backbone and RL state

- **VLA**: `pi05_droid_finetune_pick_full_v3_39999` (PaliGemma `gemma_2b` +
  flow action expert), frozen. It supplies (i) the stop-grad prefix tokens
  the RL token is read from, and (ii) the reference chunk `ã` for the same
  state. No VLA weight moves during RL.
- **RL token AE** (`RLTokenAE`, `z_dim=256`, `d_model=256`, 4 heads, 2
  layers): encoder appends a learned `<rl>` embedding to the prefix token
  sequence and reads out that position; decoder reconstructs the sequence
  from that one vector (`L_ro`). The on-disk encoder (`z_disk`) is trained
  once on all 18 corpora and **frozen** for the critic's state. A live copy
  trains `L_ro` only (`ae_finetune=true`, `ae_finetune_coef=1.0`) and never
  feeds the critic.
- **State**: `s = [z_disk(prefix), normalize(proprio)] ∈ R^{272}`.
- **Chunk MDP**: one decision = one executed chunk. `C = 8` env steps,
  action is 7 arm deltas + gripper (`chunk_dim = 64`). Reward is the
  discounted sum of env rewards inside the chunk (`γ^i` per env step,
  `γ = 0.99`); the bootstrap carries `γ^C`. Truncated tails with no
  successor are dropped, not bootstrapped. `stride = C`, so every stored row
  is one decision — no spliced windows.

### Networks (`rlt/networks.py`)

One shared set across all 18 tasks. All MLPs are ReLU, no dropout; time `t`
is one extra concatenated feature (`t = 0` noise, `t = 1` data).

| module | signature | size |
| --- | --- | --- |
| **V** `FlowActor` | `v(s, x_t, t, ã) → R^{64}` | 4 × 512, no LayerNorm, `ref_dim = 64` |
| **G student** `W` | `W(s, x_t, t, ã) → R^{64}` | 4 × 512, LayerNorm on, zero-init last layer |
| **Q** `EnsembleCritic` | `Q_k(s, x, t, ã) → R`, `K = 10` | 4 × 512, LayerNorm on, per head |

`cf_ref_conditioned=true` is the Stage A change: `W` and `Q` also read the
VLA reference chunk `ã`, the same conditioning `V` always had. Without it a
shared `Q` averages "grasp the mug" with "grasp the kettle" at the same
`(s, x, t)`, and a distilled `W` inherits that average. The reference is the
specialist's plan, so it identifies the task far more directly than the
256-dim `z` bottleneck (trained for reconstruction, not discrimination) can.

Two optimizers: Adam `lr = 3e-4` on `(V, W, AE)` and Adam `lr = 3e-4` on
`Q`. EMA on `V` (`0.999`); critic Polyak `τ = 0.005`. `W`'s last Linear is
zero-initialised, so `G ≡ 0` until distillation moves it.

### Deployed field

Inference integrates `V + G` from fresh noise by Euler, `N = 10` steps,
`dt = 1/N`. The critic is never evaluated at test time.

```
w          = W(s, x, t, ã)
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
Radial clip, trust-weighted conflict contraction against `sg(V)`, and
residual damping keep `G` from cancelling `V`. Eval can override `λ`
(`guidance_coef`); `λ = 0` is the paired intervention (guide off). Serve
uses the EMA copy of `V` (`explore=False`).

### Losses (`actor_step`, one joint step for V and G)

1. **Flow-matching BC** on the OT path to the reference chunk:
   `x_t = (1−t)ε + tã`, `target = ã − ε`, `L_bc = MSE(V(s, x_t, t, ã),
   target)`, weight `α = 1`.
2. **Endpoint anchor** on `V`'s own **unguided** unroll:
   `a_V = Euler(V, ε'; guided=False)`, `L_β = β · mean_batch Σ_j (a_V − ã)²_j`.
   The anchor is applied to the unguided unroll so `V` cannot learn to
   cancel `G`. `β = 100` in AC pretrain, `β = 1` online.
3. **One-step guided lookahead** (`λ_π = cf_actor_coef`, **0 in Stage A**):
   `g = sg(G_φ(s, x_t, t; V))`, `Δ = min(1/N, 1−t)`, `x⁺ = x_t + (V + g)·Δ`,
   `L_la = −mean_k Q_k(s, x⁺, t⁺)`. `G` is stop-grad; `V` is the only module
   that would absorb the Q signal. With `λ_π = 0` this term is off and all
   value improvement must arrive through the distilled `G`.
4. **Distillation into `W`**: one of the `K = 10` *target* heads is sampled
   per row. Common-scale-normalized gradient of that head w.r.t. `x_t`:
   `z_k = g_k / (|g_k| + c · m_B)` with `c = cf_consensus_floor = 0.01` and
   `m_B` the batch-mean gradient norm. `L_distill = Σ_j (w − z_k)²_j`,
   weight `cf_distill_coef = 1.0`.
5. **AE reconstruction** (`ae_finetune=true`): `L_ro` on a small token batch
   (`token_batch_size = 8`), weight `ae_finetune_coef = 1.0`.

Total actor loss: `α·L_bc + λ_π·L_la + L_β + L_distill + L_ro`.

### Critic: reverse-state TD (`critic_step`)

The 10-head timed critic is trained by reverse-state TD. The target
bootstraps at fresh noise `(s′, x0′, 0)` — no actor unroll in the backup —
and is clipped to `[0, 1]` (one terminal +1 means the true return never
leaves that range). The critic is evaluated at `(x^t, t)` reached by
integrating `v + G` **backwards** from the executed chunk by a random
fraction `δ` — `δ ~ U[0,1]` or `k/N` with equal probability, per row. The
loss is expectile-weighted (`cf_expectile = 0.5`, i.e. plain MSE).

### Online loop (UTD)

One learner iteration per absorbed transition batch: `utd = 5` iterations,
each being `critic_updates_per_actor = 2` critic updates followed by one
actor update, batch `256`. With several collectors the updates happen in the
learner process; collectors only store what they closed. `update_every_steps
= 0` means update on every absorbed row (the sequential-trainer cadence).

### What Stage A is not

- **Not CFGRL.** CFGRL trains a classifier-free conditional denoiser on
  optimality/goal labels and mixes cond vs uncond scores with a test-time
  weight. It does not train a 10-head critic, does not distill `∇_x Q`, and
  does not run online TD. The `cf_*` flags here are **ConsensusFlow**, not
  CFGRL.
- **Not Stage 1.** `λ_π = 0` online, so `V` gets BC + anchor only; the
  one-step lookahead that would pull `V` toward `Q` is off. Stage B tested
  `λ_π = 1` and lost 5.2 pp on held-out gOn.
- **Not zero-shot.** The VLA was pick-finetuned on MolmoSpaces, then RL ran
  on 18 jittered val scenes (same house/object as 18 of the 1000 val
  episodes, different jitter).

## Stage A protocol

`pipeline_pick18_v24_shared.sh`, six phases, each skipped when its output
exists:

1. **AE** (1 GPU, ~1 h). Symlink the 18 token corpora into one tree, train
   one AE (`-m rlt.train_token_ae`, 8000 steps, `max_sequences=18000` ≈ 1000
   per task ≈ 72 GB RAM as float16; the loader takes an equal quota per
   shard, so the mix stays balanced).
2. **COLLECT** (queue over GPUs × 3 slots, 18 jobs). Per task: 100
   frozen-VLA episodes on the train bench (`episode_pool=0-47`),
   `run_pretrain_ac.py --episodes 100 --offline-steps 0` with
   `token_ae=<shared>` — collect-only, writes `collect/<task>/buffer.npz`.
   The per-task `runs/pick18/*/vla_buffer.npz` buffers live in per-task AE
   space and are **not** reused; merging them would mix 18 different
   encodings of the same prefix.
3. **MERGE** (CPU, minutes). `scripts/merge_replays.py` concatenates the 18
   buffers, refusing mismatched field shapes (a different encoder or chunk
   size is a different MDP).
4. **PRETRAIN** (1 GPU, replay-only). 30,000 offline steps on the merged
   buffer: `β=100`, `λ_π=0`, `cf_ref_conditioned=true`, `ae_finetune=true`,
   `token_replay=<symlinked glob>`. 30k steps ≈ 77 passes/row at batch 256
   over ~63k rows, matching the specialist per-row budget (8k steps was a
   per-specialist budget over ~1.6k rows).
5. **ONLINE** (all GPUs). One run: `task_pool` = all 18 `scene:horizon`
   entries (kettle 400, rest 500), 1800 episodes + 180 probes (100 + 10 per
   task), `β=1`, `λ_π=0`, `cf_ref_conditioned=true`, 16 collectors (4/GPU on
   4 cards), `vla_ports`/`vla_gpus`/`egl_devices` across the cards,
   `init_actor`/`init_buffer` from the pretrain. Episode N runs
   `task_list[N % 18]`, so the mixed pool is exactly round-robin. Per-task
   curves are logged (`task/<scene>/sr_last10`, `task/<scene>/probe_sr`).
6. **EVAL** (queue over GPUs, 36 jobs). Per task, paired `λ=0.5` vs `λ=0` on
   the held-out eval bench, 16 specs (= 64 rollouts) — the same protocol as
   the specialist sweep, with the shared agent and shared AE.

Kill criteria unchanged: `|Q| > 2` or `actor_ref_rmse` leaving the chunk
scale (~0.2).

Replay capacity stays 400k (`buffer_capacity`): 18 tasks × 100 episodes ×
~60 rows ≈ 108k pretrain rows + ~108k online rows fits with room. The
128-task pool needs ~1.5M — a Stage C change, made there.

## Decision gate

Compare per task and in macro against the finished β=1 specialist sweep
(`runs/pick18_v22_24_beta1/`):

| | probe | online | gOn (λ=0.5) | gOff (λ=0) |
| --- | ---: | ---: | ---: | ---: |
| Specialist macro (β=1) | 62.2% | 67.5% | **92.8%** | 56.5% |
| **Shared Stage A** | 50.0% | 65.3% | **93.8%** | 57.2% |

- **Pass** → Stage C: shared gOn within a few points of 92.8% macro, no task
  collapsing to 0 that the specialists solved, and the on/off gap still
  positive (guidance still earns its keep when shared).
- **Mixed** (macro close, specific tasks collapse) → the per-task curves say
  which: AE confusion (bad `z` for that task), reward-row starvation (its
  successes drowned in the shared buffer), or guidance conflict. Stage B
  targets exactly that part.
- **Fail** (macro drops ≫10 pp) → sharing hurts on its own; Stage C is off
  until the mechanism is understood. The pilot cost ~2 days, not ~3 weeks.

**Outcome: pass.** Held-out gOn is **1081/1152 = 93.8%** vs the specialist
macro **1069/1152 = 92.8%** (+1.0 pp). No specialist-solved task collapsed
to 0 at eval. The G on/off gap is **+36.6 pp** (specialists +36.3 pp), so
guidance still earns its keep when shared. Online train SR is 2.2 pp behind
(65.3% vs 67.5%) on ~100 episodes/task vs the specialists' 300.

---

## Stage A results (finished 2026-09-13)

Run dir `runs/pick18_v24_shared/`. Pipeline
`pipeline_pick18_v24_shared.sh`. Online ran on 4×H100 with 16 collectors
(4 per card), resumed once from episode 272/1800 after a reboot
(`resume=true`). Eval: 36 paired jobs, all `result.json` present, pipeline
`fail=0`.

### Pretrain

Shared AE: 8000 steps on 17,882 sequences (257 shards, equal quota). Replay
merge: **63,333** rows from 18×100 frozen-VLA episodes. AC pretrain: 30,000
steps, `β=100`, `λ_π=0`, `cf_ref_conditioned=true`, 0.91 h, **1,037** reward
rows, final `critic≈5e-4`, `q_mean≈0.26`, BC RMSE ≈0.42.

### Probe and online

| | probe (180) | online (1800) | last-10 |
| --- | ---: | ---: | ---: |
| Shared | **90/180 = 50.0%** | **1176/1800 = 65.3%** | **0.70** |
| Specialist β=1 | 112/180 = 62.2% | 3644/5400 = 67.5% | — |

Online SR by 300-episode window (shared, mixed 18-task pool):

| episodes | SR |
| ---: | ---: |
| 1–300 | 55.3% |
| 301–600 | 64.7% |
| 601–900 | 64.7% |
| 901–1200 | 66.7% |
| 1201–1500 | 69.0% |
| 1501–1800 | **71.7%** |

Kill checks at the last online step: `q_mean≈0.18`, `actor_ref_rmse≈0.005`,
`w_norm≈0.26`, `g_over_v≈0.008`, `buffer/reward_rows=2213`. `|Q|` never
left `[0,1]`; V stayed a tight copy of `ã`.

Per-task last-10 at the end of online (~84–85 episodes each):

| task | last-10 | task | last-10 |
| --- | ---: | --- | ---: |
| spoon, remote, fork, desk_mug | 1.0 | soap_dispenser | 0.7 |
| spatula, cup, bowl | 0.9 | tissue, pot | 0.6 |
| ladle, kettle, box | 0.8 | knife | 0.5 |
| | | fruit, shaker | 0.3 |
| | | bottle, spray_bottle | 0.2 |

### Held-out paired eval (16 specs = 64 rollouts / task)

Same protocol as `runs/pick18_v22_24_beta1/`. Shared agent + shared AE.
`gOn` = trained `λ=0.5`; `gOff` = `guidance_coef=0`.

| | gOn (`λ=0.5`) | gOff (`λ=0`) |
| --- | ---: | ---: |
| **Shared macro** | **1081/1152 = 93.8%** | **659/1152 = 57.2%** |
| Specialist macro | 1069/1152 = 92.8% | 651/1152 = 56.5% |
| Δ vs specialist | **+12 / +1.0 pp** | +8 / +0.7 pp |

gOn vs specialist per task: **6 up, 6 tie, 6 down**.

| task | shared gOn | spec gOn | Δ | shared gOff | spec gOff | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| desk_mug | 64/64 | 64/64 | 0 | 56/64 | 45/64 | +11 |
| kettle | 60/64 | 60/64 | 0 | 17/64 | 20/64 | −3 |
| remote | 63/64 | 62/64 | +1 | 49/64 | 49/64 | 0 |
| ladle | 59/64 | 64/64 | −5 | 57/64 | 58/64 | −1 |
| tissue | 58/64 | 61/64 | −3 | 31/64 | 37/64 | −6 |
| spoon | 64/64 | 64/64 | 0 | 63/64 | 63/64 | 0 |
| spatula | 64/64 | 64/64 | 0 | 63/64 | 63/64 | 0 |
| pot | 63/64 | 63/64 | 0 | 24/64 | 29/64 | −5 |
| soap_dispenser | 64/64 | 63/64 | +1 | 14/64 | 3/64 | +11 |
| spray_bottle | 44/64 | 32/64 | **+12** | 0/64 | 0/64 | 0 |
| cup | 63/64 | 64/64 | −1 | 64/64 | 64/64 | 0 |
| shaker | 59/64 | 60/64 | −1 | 12/64 | 9/64 | +3 |
| fork | 63/64 | 64/64 | −1 | 63/64 | 62/64 | +1 |
| bottle | 62/64 | 59/64 | +3 | 0/64 | 0/64 | 0 |
| fruit | 47/64 | 44/64 | +3 | 26/64 | 33/64 | −7 |
| bowl | 64/64 | 64/64 | 0 | 64/64 | 63/64 | +1 |
| knife | 61/64 | 55/64 | +6 | 2/64 | 4/64 | −2 |
| box | 59/64 | 62/64 | −3 | 54/64 | 49/64 | +5 |

The extra held-out is still **G at sample time**, not a stronger `V`: gOff
is a wash (+0.7 pp). Spray_bottle and knife are the main shared-policy
wins on gOn; ladle is the main loss (−5). Bottle stays a G-only rescue
(62/64 on, 0/64 off), same pattern as the β=1 specialists.

Eval artifacts: `runs/pick18_v24_shared/eval/shared_v24_s0_<task>_gOn|gOff/result.json`.
Online summary: `runs/pick18_v24_shared/rl/shared_v24_s0/summary.json`.

### Official val-1000 vs Pick-v1 (Using MolmoSpaces Data)

The 18-task held-out above is our own jittered eval benches (64 rollouts /
task). The number that matches the [MolmoSpaces leaderboard](https://molmospaces.allen.ai/leaderboard)
is the official **Pick-v1 (MS-Pick)** val set: 1000 unique episodes of
`FrankaPickDroidMiniBench_json_benchmark_20251231` under
`molmospaces-bench-v1` (leaderboard pin `20260408`). Same episodes for
shared gOn and frozen pi0.5, horizon 500, oracle-style `success_count`.

| Policy | Success | 95% CI | GPU-h |
| --- | ---: | ---: | ---: |
| Shared gOn Stage A `shared_v24_s0` | **777/1000 = 77.7%** | 75.0–80.2 | 11.1 |
| Frozen pi0.5 `pick_full_v3_39999` | 635/1000 = 63.5% | 60.5–66.4 | 13.0 |

Artifacts: `runs/pick18_v24_shared/eval_val1000/gon_summary.json`,
`vla_summary.json`.

That 77.7% is **not** a zero-shot VLA score. The VLA was pick-finetuned on
MolmoSpaces, then RL ran on 18 jittered val scenes (same house/object as 18
of the 1000 val episodes, different jitter). The leaderboard therefore
puts us in **Using MolmoSpaces Data**, next to MolmoBot, not next to
PrimeR0 / Phoenix / π₀.₅ DROID (those are the other split). The fair
claim on this bench is the +14.2 pp lift over the same frozen checkpoint.
Against the ID group, shared gOn sits between MolmoBot (~92%) and
MolmoBot-π₀ (64.0%); the frozen pick-ft run is a wash with MolmoBot-π₀.
Horizon is 500 vs the docs' 450 (Phoenix also used 500). Combined 9-task
scores are a different table.

Oracle rates below: leaderboard Pick-v1 with “Split by MolmoSpaces Data”
on, scraped 2026-09-15, plus our two val-1000 rows. Leaderboard ± is their
95% interval; ours is Wilson half-width at n=1000.

| # | Policy | Authors | Oracle SR | Type | Notes |
| ---: | --- | --- | ---: | --- | --- |
| 1 | MolmoBot-Img | Ai2 | 92.8 ±1.8 | VLA | sim, multi-task |
| 2 | MolmoBot | Ai2 | 92.0 ±1.8 | VLA | sim, multi-task |
| 3 | MolmoBot-f3 | Ai2 | 91.9 ±1.8 | VLA | sim, multi-task |
| **4** | **Shared gOn Stage A** | this work | **77.7 ±2.6** | RL on frozen pi0.5 | pick specialist; 18 jittered val scenes; `λ_π=0` |
| 5 | MolmoBot-π₀ | Ai2 | 64.0 ±3.0 | VLA | sim ft of π₀ |
| **6** | **Frozen pick-ft pi0.5** | this work | **63.5 ±3.0** | VLA | `pi05_droid_finetune_pick_full_v3_39999` |
| 7 | π₀.₅-MolmoSpaces | Ai2 | 45.0 ±3.1 | VLA | sim ft of π₀.₅ |

Two known risks Stage A measures at 18-task scale before they matter at 128:

- **Reward-row imbalance.** Uniform task sampling gives the strong tasks
  ~100× more reward rows than the weak ones (worse than kettle was). TD
  mostly reads reward rows; watch `buffer/reward_rows` and per-task
  `sr_last10` for starvation.
- **AE generalization.** The 256-dim bottleneck has only ever been evaluated
  in-distribution. Stage A still trains and tests on the same 18 scenes, so
  it does *not* measure unseen-house generalization. Stage B ran the `λ_π=1`
  second arm, not a held-out-house AE; that probe is still open before
  Stage C. What Stage A does measure is whether one AE can serve 18 scenes
  *at all* without per-task collapse.

## Stage B results (finished 2026-09-14)

The `λ_π=1` second arm on the shared 18-task policy. Pipeline
`pipeline_pick18_v24_stageB.sh`. Same shared AE, same merged buffer, same
30k-step pretrain (`shared_v24_pretrain`); only online changes:
`cf_actor_coef=1` (one-step guided lookahead on), tag `shared_v24_s1`.
4×H100, 16 collectors. Eval: 36 paired jobs, all `result.json` present.

This is the arm the original V24 draft wanted as the default. Stage A had
already chosen stage 0 from the kettle specialist (`λ_π=1` last-10 0.4,
weaker V). Stage B tests whether sharing plus ref-conditioned `W`/`Q`
changes that.

### Probe and online

| | probe (180) | online (1800) | last-10 |
| --- | ---: | ---: | ---: |
| **Stage B `λ_π=1`** | **95/180 = 52.8%** | **1111/1800 = 61.7%** | **0.50** |
| Stage A `λ_π=0` | 90/180 = 50.0% | 1176/1800 = 65.3% | 0.70 |

15.85 h wall (4 GPUs). Online SR by 300-episode window:

| episodes | Stage B | Stage A |
| ---: | ---: | ---: |
| 1–300 | 50.3% | 55.3% |
| 301–600 | 51.7% | 64.7% |
| 601–900 | 62.7% | 64.7% |
| 901–1200 | 66.0% | 66.7% |
| 1201–1500 | 67.7% | 69.0% |
| 1501–1800 | **72.0%** | 71.7% |

Slow start, last window matches Stage A. Kill checks at the last online
step: `q_mean≈0.14`, `actor_ref_rmse≈0.005`, `w_norm≈0.27`,
`g_over_v≈0.009`, `buffer/reward_rows=2148`. `|Q|` never left `[0,1]`; V
stayed a tight copy of `ã`. No kill.

Per-task last-10 at the end of online (100 episodes each):

| task | last-10 | task | last-10 |
| --- | ---: | --- | ---: |
| spoon, spatula, desk_mug, ladle, remote, fork, bowl | 1.0 | box | 0.7 |
| soap_dispenser, cup, kettle, tissue, pot | 0.8 | knife | 0.6 |
| | | shaker | 0.4 |
| | | fruit | 0.3 |
| | | spray_bottle | 0.2 |
| | | bottle | 0.1 |

Same weak cluster as Stage A (bottle, spray_bottle, fruit, shaker).

### Held-out paired eval (16 specs = 64 rollouts / task)

Same protocol as Stage A. Shared `shared_v24_s1` agent + the Stage A AE.
`gOn` = trained `λ=0.5`; `gOff` = `guidance_coef=0`.

| | gOn (`λ=0.5`) | gOff (`λ=0`) | on/off gap |
| --- | ---: | ---: | ---: |
| **Stage B `λ_π=1`** | **1021/1152 = 88.6%** | **704/1152 = 61.1%** | **+27.5 pp** |
| Stage A `λ_π=0` | 1081/1152 = 93.8% | 659/1152 = 57.2% | +36.6 pp |
| Δ vs Stage A | **−60 / −5.2 pp** | **+45 / +3.9 pp** | −9.1 pp |
| Specialist β=1 | 1069/1152 = 92.8% | 651/1152 = 56.5% | +36.3 pp |

Guidance still earns its keep (88.6 vs 61.1), but lookahead **hurts gOn**
and only slightly helps `V`. gOn vs Stage A: **6 up, 3 tie, 9 down**.

| task | Stage A gOn | Stage B gOn | Δ | Stage A gOff | Stage B gOff | Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| desk_mug | 64/64 | 64/64 | 0 | 56/64 | 63/64 | +7 |
| kettle | 60/64 | 48/64 | **−12** | 17/64 | 16/64 | −1 |
| remote | 63/64 | 64/64 | +1 | 49/64 | 50/64 | +1 |
| ladle | 59/64 | 53/64 | −6 | 57/64 | 47/64 | −10 |
| tissue | 58/64 | 57/64 | −1 | 31/64 | 43/64 | +12 |
| spoon | 64/64 | 64/64 | 0 | 63/64 | 63/64 | 0 |
| spatula | 64/64 | 64/64 | 0 | 63/64 | 61/64 | −2 |
| pot | 63/64 | 62/64 | −1 | 24/64 | 45/64 | **+21** |
| soap_dispenser | 64/64 | 62/64 | −2 | 14/64 | 30/64 | +16 |
| spray_bottle | 44/64 | 45/64 | +1 | 0/64 | 1/64 | +1 |
| cup | 63/64 | 64/64 | +1 | 64/64 | 64/64 | 0 |
| shaker | 59/64 | 37/64 | **−22** | 12/64 | 12/64 | 0 |
| fork | 63/64 | 64/64 | +1 | 63/64 | 64/64 | +1 |
| bottle | 62/64 | 55/64 | −7 | 0/64 | 0/64 | 0 |
| fruit | 47/64 | 33/64 | **−14** | 26/64 | 26/64 | 0 |
| bowl | 64/64 | 63/64 | −1 | 64/64 | 62/64 | −2 |
| knife | 61/64 | 62/64 | +1 | 2/64 | 2/64 | 0 |
| box | 59/64 | 60/64 | +1 | 54/64 | 55/64 | +1 |

The gOn losses are concentrated: shaker −22, fruit −14, kettle −12, bottle
−7, ladle −6. The gOff gains (pot +21, soap_dispenser +16, tissue +12,
desk_mug +7) are a stronger `V`, but they do not pay for the gOn drop, and
the G-only rescues (bottle 55/64 on vs 0/64 off, spray_bottle 45 vs 1) stay
G-only. Same kettle pattern as the specialist stage-1 arm.

**Decision:** Stage C uses Stage A's recipe (`β=1`, `λ_π=0`). `λ_π=1` is
not the shared-policy default.

Eval artifacts: `runs/pick18_v24_shared/eval_stageB/shared_v24_s1_<task>_gOn|gOff/result.json`.
Online summary: `runs/pick18_v24_shared/rl/shared_v24_s1/summary.json`.

Still open from the original Stage B outline (not run): AE trained on more
scenes and probed on held-out houses; reward-row balancing; `buffer_capacity`
to ~1.5M (needed at 128 tasks).

## Stage C (128 tasks) — outline

The 128-task train pool (stratified over all 18 instructions, seeded draw,
no val contamination), 100 VLA rollouts/task, same shared architecture, eval
on the 128 train tasks and the official 1000-episode val benchmark, paired
λ on/off. Budgets scale from Stage A measurements: collect ~12,800 episodes,
AC pretrain ~100k steps, online 12,800 episodes, eval sharded over GPUs.
