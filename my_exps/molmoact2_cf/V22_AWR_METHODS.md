# V22 AWR

Advantage-Weighted Regression (Peng et al. 2019) on the **same one-pass
Gaussian actor** as V21 RL Token. The frozen π0.5, scene token AE, chunk MDP,
data, and eval bench are identical to V21 / V22_24. Only the actor objective
changes: the critic is stop-grad, and the actor is weighted regression onto
the **replay** action instead of V21's `-Q(s, a) + β ‖a − ã‖²` (Eq. 5) or
V22_24's flow + distilled guidance.

This is the missing one-pass baseline: same π, same Q-scale, no critic
gradients through the actor, no flow field, no `G`.

Code: `pi05_rl_token/rlt/awr.py` (`AWRAgent`, `algorithm=awr`).
Launcher: `pipeline_pick18_awr.sh`. Checkpoint:
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
   twin Q_ψ(x, a)           TD, target clipped to [0, 1]
   V_ψ(x)                   TD on x only (AWR baseline)
        │
   A = sg( min Q(x, a_replay) − V(x) )
   w = exp(A / τ) , clipped, mean-normalised
   L_π = E[ −w log π(a_replay | x, ã) ]
        │
   AC pretrain on frozen-VLA train rollouts (AWR + TD, not unweighted BC)
        │
   10 actor-only probe episodes (not stored in replay; seed sr_last10)
        │
   Online RL from step 0, warmup=0, init from that actor + buffer
```

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

Replay starts from the **same** 100 frozen-VLA episodes as V21/V22_24.
Online rows are appended. Sampling is uniform (`batch_size=256`). Capacity
400k; nothing is evicted. `C = 8`, `gate_step=0`, train pool `0-11`,
pretrain collect pool `0-47`, held-out eval is the disjoint jittered bench.

## Networks

All MLPs: ReLU, no dropout, **same width as V21** (not the 4×512 flow
stack). Time is not an input; this is a one-pass Gaussian.

| module | signature | size |
| --- | --- | --- |
| **π** `Actor` | `μ(s, ã) → R^{64}`, `π = N(μ, σ²I)` | 2 × 256, no LayerNorm |
| **Q** `DoubleCritic` | `Q_i(s, a) → R`, `i = 1, 2` | 2 × 256, no LayerNorm |
| **V** `Value` | `V(s) → R` | 2 × 256, no LayerNorm |

`σ = 0.02` (fixed, not learned). Adam `lr = 3e-4` on two optimizers:
`(π)` and `(Q, V)`. Critic Polyak `τ = 0.005` on both Q and V. No actor
EMA (matched to V21).

The extra `V` is AWR's baseline. It is not a third Q: it never sees `a`.
Capacity added vs V21 is one state-only MLP, not a flow ensemble.

## Losses

### Critic (same backup as V21, plus V)

```
a'  ~ π(· | s', ã')
y_Q = clip( r + γ^C (1 − done) min_i Q̄_i(s', a') ,  0, 1 )
y_V = clip( r + γ^C (1 − done) V̄(s') ,            0, 1 )
L_Q = MSE(Q_1, y_Q) + MSE(Q_2, y_Q)
L_V = MSE(V, y_V)
```

`γ = 0.99` per env step, so `γ^C = 0.99^8 ≈ 0.923`. Live Q and V are not
clipped. Two critic steps per actor step. Online `UTD = 5` per absorbed
row, actor/critic updates every 100 stored env steps.

### Actor (AWR, not Eq. 5)

```
A = sg( min_i Q_i(s, a) − V(s) )          a = replay action
w = exp(A / τ)                            clip at 20, then w ← w / mean(w)
L_π = E[ −w  log π_θ(a | s, ã_drop) ]
```

`τ = 1.0` (`awr_temp`). Returns live in `[0, 1]`; Peng et al.'s `β = 0.05`
on this scale saturates the clip and becomes a max-advantage one-hot.
`τ = 1` is AWAC's `λ` on unit-scale rewards. Reference dropout `0.5` on
the actor *input* matches V21; the replay action is never dropped.

`log π` is the Gaussian NLL of the **stored** chunk under `N(μ(s, ã), σ²I)`.
The actor never samples for the loss and never backprops through Q. That is
the whole point of this baseline.

V21's Euclidean `β ‖a − ã‖²` is **off**. AWR's constraint is the implicit
KL to the replay behavior (Peng et al., §3). The buffer already holds the
specialist chunks from the 100-traj collect, then the actor's own online
rows. Weighting those rows by advantage *is* the pull toward good
behavior; adding Eq. 5 on top would make a hybrid, not AWR.

AC pretrain calls the same `actor_step` (the hook is named `actor_bc_step`
for the shared trainer). It is **not** unweighted MSE onto `ã`. On
frozen-VLA data `a = ã`, so AWR reduces to advantage-weighted cloning of
the specialist: successful trajectories get `w > 1` once Q and V separate,
failed ones get `w < 1`. Unweighted BC is the `A ≡ 0` special case.

## Training protocol

Identical schedule to V21 / V22_24 Pick-18:

1. **Data.** 100 frozen-VLA rollouts → decision-chunk replay. Reuse
   `runs/pick18/<task>/vla_buffer.npz` (do not re-collect).
2. **AE.** Scene AE from `runs/pick18/<task>/ae/`. Frozen as `z_disk`.
3. **AC pretrain.** 8000 steps, AWR + TD, `τ = 1`. `V` and `Q` live.
4. **Probe.** 10 actor-only episodes, **not** written to replay.
5. **Online.** 4 collectors, 3 EGL slots, 300 stored episodes,
   `warmup=0`, `gate_step=0`, `episode_pool=0-11`.
6. **Eval.** Held-out eval48, `--episodes 16` → 64 rollouts, same gate.
   No guidance on/off pair: there is no `G`.

Rolling metric is **last-10** stored-actor SR (`sr_last10`). Kill if
`|Q| > 2` or `actor_ref_rmse` leaves the chunk scale (~0.2).

```
GPUS="0 1 2 3 4 5 6 7" SLOTS_PER_GPU=2 bash pipeline_pick18_awr.sh
```

Run dir: `runs/pick18_awr/`. Per task: `ac_pretrain/<task>_awr_pretrain/`,
`rl/<task>_awr_s0/`, `eval/<task>_awr_s0_actor/`.

Kettle-only (reuse V21 kettle AE + buffer):

```
bash pipeline_kettle_awr.sh
```

## What this isolates

| | V21 RL Token | **AWR (this)** | V22_24 ConsensusFlow |
| --- | --- | --- | --- |
| Actor | one-pass Gaussian | **same Gaussian** | flow `V + G`, N=10 |
| Actor loss | `-Q(s, a_π) + β‖a−ã‖²` | **`-w log π(a_replay)`** | BC + anchor + distill + optional `-Q` lookahead |
| Critic grads into π | yes (reparam) | **no** | only via distilled `G` / lookahead |
| VLA pull | explicit `β` | implicit KL to replay | `β` on unguided Euler(`V`) |
| Extra nets vs V21 | — | state-value `V` | `FlowActor`, `Guidance`, 10-head timed Q |

A fair Pick-18 row is therefore: same AE, same 100-traj buffer, same 8000
+ probe + 300 + eval48, report probe / online / last-10 / held-out next to
V21 `β=1` and V22_24 stage-0.

## Results

Not run yet. Fill from `runs/pick18_awr/` the same way as
`V21_METHODS.md` (probe after offline AC → held-out eval48, 64 rollouts).

| | probe | online (300) | last-10 | held-out eval48 |
| --- | ---: | ---: | ---: | ---: |
| **Pick-18 macro** | — | — | — | — |
