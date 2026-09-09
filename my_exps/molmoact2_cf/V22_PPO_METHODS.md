# V22 PPO

On-policy PPO (Schulman et al. 2017) on the **same one-pass Gaussian actor** as
V21 RL Token. The frozen π0.5, scene token AE, chunk MDP, frozen-VLA collect,
probe, online episode count, and held-out eval48 bench are identical to V21 /
AWR / V22_24. The actor objective is not: PPO never reads replay, never
backprops through a Q, and never applies Eq. 5.

This is the missing on-policy baseline: same π, same chunk as an MDP step, GAE
on the collector trajectory, clipped importance-sampling surrogate.

Code: `pi05_rl_token/rlt/ppo.py` (`PPOAgent`, `algorithm=ppo`).
Launcher: `pipeline_pick18_ppo.sh`. Checkpoint:
`pi05_droid_finetune_pick_full_v3_39999`. Chunk 8, `step_time`,
`rl_action_space=delta`. **Default: RL from env step 0** (`gate_step=0`).
Train and eval must use the same gate.

```
RGB, proprio, language
        │
   pi0.5 PaliGemma (frozen)          → prefix tokens (968, 2048)
        │
   Token encoder g_φ  →  z_rl        ← phase-1 AE, frozen during RL
        │
   x = (z_rl, proprio)
        │
   pi0.5 action expert → ã (delta reference chunk)   ← frozen
        │
   Gaussian π_θ(a | x, ã)   one pass, same Actor MLP as V21
   V_ψ(x)                   GAE critic (state only; no twin Q)
        │
   AC pretrain: BC μ → ã + clipped TD on V
        │
   10 actor-only probe episodes (not stored in replay; seed sr_last10)
        │
   Online PPO from step 0, warmup=0
   act() stores log π_old; GAE(λ) on each collector episode;
   K epochs of clip(r, 1−ε, 1+ε) once ppo_horizon rows are ready
```

The learner path is `consume_on_policy`, not UTD-on-replay. `actor_step` on a
shuffled batch raises. That is the point of this baseline. Sequential training
buffers mid-episode closes and runs GAE once on the finished episode; a one-row
close would turn GAE(λ) into TD(0). Parallel training already sends one finished
episode per payload. Missing `log π` is refused (recomputing it under the learner
would be a fake π_old). A warmup episode with no log π is skipped, not trained.

## Frozen VLA

Same as V21 / V22_24. PaliGemma (`gemma_2b`, token width 2048) and the base
action expert stay frozen. The VLA supplies (i) prefix tokens for the RL
token, (ii) the reference chunk `ã`. Online RL does not update VLM or expert
weights.

## RL token

Same scene AE as the Pick-18 V21/V22_24 sweeps (`z_dim=256`). RL always reads

```
s = φ(obs) = [ z_disk(prefix), normalize(proprio) ] ∈ R^{272}
```

`z_disk` is frozen. No `L_ro`, no `ae_finetune` (matched to V21, not V22_24).

## Chunk MDP

One inference commits one action chunk. That chunk is one MDP step. Same
table as V22_24:

| | |
| --- | --- |
| `s` | `[z, proprio]` at chunk start |
| `a` | executed chunk, flat (`C × 8`) |
| `ã` | specialist chunk for the same `s` |
| `r` | discounted sum of env rewards inside the chunk (`γ^i` per env step) |
| `done` | 1 if the episode ends inside the chunk |
| `s'` | next chunk start (truncated tails with no successor are dropped) |

Replay still starts from the **same** 100 frozen-VLA episodes and still
appends online rows (capacity 400k) so a run is inspectable and resumable.
**The actor does not train on that buffer.** Each collector payload is one
episode; GAE is computed on that trajectory; PPO epochs wait until
`ppo_horizon` on-policy rows have accumulated. `C = 8`, `gate_step=0`, train
pool `0-11`, pretrain collect pool `0-47`, held-out eval is the disjoint
jittered bench.

## Networks

All MLPs: ReLU, no dropout, **same width as V21** (not the 4×512 flow
stack). Time is not an input; this is a one-pass Gaussian.

| module | signature | size |
| --- | --- | --- |
| **π** `Actor` | `μ(s, ã) → R^{64}`, `π = N(μ, σ²I)` | 2 × 256, no LayerNorm |
| **V** `Value` | `V(s) → R` | 2 × 256, no LayerNorm |

`σ = 0.02` (fixed, not learned). The Gaussian entropy is therefore a
constant; there is no entropy bonus, because its gradient would be zero.
Adam `lr = 3e-4` on two optimizers: `(π)` and `(V)`. Critic Polyak `τ = 0.005`
on V. No actor EMA (matched to V21). No twin Q: PPO's critic is V.

## Losses

### AC pretrain (not PPO)

Frozen-VLA chunks are not draws from this Gaussian, so a PPO ratio against
them would be a fake. Pretrain is V21's BC plus clipped TD on V:

```
L_BC = MSE( μ(s, ã_drop), ã )
y    = clip( r + γ^C (1 − done) V̄(s'),  0, 1 )
L_V  = MSE(V(s), y)
```

Reference dropout `0.5` is on during BC only, so a pathway without `ã`
exists after pretrain. It is **off** on the PPO update: the ratio must use
the same `ã` the collector passed to `act()`.

### Online (PPO)

`act()` stores `log π_θ(a | s, ã)` on the decision. Collectors pickle that
scalar with the row, so `π_old` is the collector's policy, not a recompute
under the learner's later weights. The update raises if any on-policy row is
missing that scalar.

GAE(λ) on one **finished episode** (`γ^C = 0.99^8 ≈ 0.923`, `λ = 0.95`), not
on a mid-episode close. Timeouts bootstrap from `V(s')`; terminals do not.
Advantages are stop-grad and (by default) mean-std normalised over the PPO
batch:

```
δ_t = r_t + γ^C (1 − d_t) V(s'_t) − V(s_t)
A_t = δ_t + γ^C λ (1 − d_t) A_{t+1}
R_t = A_t + V(s_t)
```

Once `ppo_horizon = 256` chunk rows are ready, `K = 4` epochs of minibatches
of 64:

```
r     = exp( log π_θ(a | s, ã) − log π_old )
L_π   = − mean min( r A,  clip(r, 1−ε, 1+ε) A )     ε = 0.2
V_clip = V_old + clip(V − V_old, −ε, ε)
L_V   = ½ mean max( (V − R)², (V_clip − R)² )
L     = L_π + c_v L_V                                 c_v = 0.5
```

Gradients are clipped to `0.5`. V21's Euclidean `β ‖a − ã‖²` is **off**.
PPO's trust region is the clip. `update_every_steps` / `utd` do not apply;
the on-policy horizon replaces them.

## Training protocol

Identical schedule to V21 / V22_24 Pick-18:

1. **Data.** 100 frozen-VLA rollouts → decision-chunk replay. Reuse
   `runs/pick18/<task>/vla_buffer.npz` (do not re-collect).
2. **AE.** Scene AE from `runs/pick18/<task>/ae/`. Frozen as `z_disk`.
3. **AC pretrain.** 8000 steps, BC + TD on V. Not PPO.
4. **Probe.** 10 actor-only episodes, **not** written to replay.
5. **Online.** 4 collectors, 3 EGL slots, 300 stored episodes,
   `warmup=0`, `gate_step=0`, `episode_pool=0-11`.
6. **Eval.** Held-out eval48, `--episodes 16` → 64 rollouts, same gate.
   No guidance on/off pair: there is no `G`.

Rolling metric is **last-10** stored-actor SR (`sr_last10`). Kill if
`|V| > 2` or `actor_ref_rmse` leaves the chunk scale (~0.2).

```
GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 bash pipeline_pick18_ppo.sh
```

Run dir: `runs/pick18_ppo/`. Per task: `ac_pretrain/<task>_ppo_pretrain/`,
`rl/<task>_ppo_s0/`, `eval/<task>_ppo_s0_actor/`.

Kettle-only (reuse V21 kettle AE + buffer):

```
bash pipeline_kettle_ppo.sh
```

## What this isolates

| | V21 RL Token | AWR | **PPO (this)** | V22_24 ConsensusFlow |
| --- | --- | --- | --- | --- |
| Actor | one-pass Gaussian | same Gaussian | **same Gaussian** | flow `V + G`, N=10 |
| Actor loss | `-Q(s, a_π) + β‖a−ã‖²` | `-w log π(a_replay)` | **clipped IS surrogate** | BC + anchor + distill + optional `-Q` lookahead |
| Critic grads into π | yes (reparam) | no | **no** (stop-grad A) | only via distilled `G` / lookahead |
| Data for π | replay | replay | **on-policy rollouts** | replay |
| VLA pull | explicit `β` | implicit KL to replay | BC pretrain + clip | `β` on unguided Euler(`V`) |
| Extra nets vs V21 | — | state-value `V` + twin Q | state-value `V` | `FlowActor`, `Guidance`, 10-head timed Q |

A fair Pick-18 row is therefore: same AE, same 100-traj buffer, same 8000
+ probe + 300 + eval48, report probe / online / last-10 / held-out next to
V21 `β=1`, AWR, and V22_24 stage-0.

## Results

Not run yet. Fill from `runs/pick18_ppo/` the same way as
`V21_METHODS.md` (probe after offline AC → held-out eval48, 64 rollouts).

| | probe | online (300) | last-10 | held-out eval48 |
| --- | ---: | ---: | ---: | ---: |
| **Pick-18 macro** | — | — | — | — |
