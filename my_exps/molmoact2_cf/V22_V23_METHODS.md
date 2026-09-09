# V22+V23: one method, two actor backbones

This is the paper method. It is **V23's recipe** (frozen π0.5, RL-token
critic, chunk TD, β-anchor, BC, AE → AC → online). Pick (V22) is the same
method with a **small actor instantiation**: the velocity field is a tiny
MLP on the RL token instead of LoRA inside `gemma_300m`. Nothing else is aW
new algorithm.

Per-run logs stay in [`V23_METHODS.md`](V23_METHODS.md) (radio,
`cf_v23_ae2`) and [`V22_METHODS.md`](V22_METHODS.md) (Molmo Pick). Failed
radio actors (`G` injection, OT-compose MLP) stay in
[`V23_attempts.md`](V23_attempts.md) — they are not the method.

```
RGB + proprio + prompt
        │
   frozen PaliGemma          → prefix tokens
        │
   token AE  g_φ, d_φ        → z   (critic state only; sg(target_pool) at RL)
        │
   frozen action expert      → ã   (specialist chunk)
        │
   v_θ  (identity at init)   → a = Euler(v_θ, noise)     10 steps
        │
   Q(s, flatten(a))          min over heads, TD clip [0,1]
        │
   L = 4 L_td + λ_π (−Q) + β‖a−ã‖² + L_bc + L_ro
```

`λ_π` gates only `−Q`. Anchor (`β=100`) and BC stay on in every phase.
`Δv = 0` (LoRA `B=0` / `G=0`) copies the specialist exactly.

## Shared method (from V23)

### Frozen VLA

PaliGemma (`gemma_2b`) and the base action expert stay frozen. The VLA
supplies (i) prefix tokens for the RL token, (ii) the reference chunk `ã`.
Online RL does not update VLM or base expert weights.

### RL token

Encoder–decoder on stop-grad prefix tokens:

```
L_ro = MSE(decoder(z_live), sg(prefix_tokens))
```

RL always reads `φ(s) = [ sg(target_pool(prefix)), normalize(proprio) ]`.
Live pool trains `L_ro` only. Online `freeze_pool=1` so AE finetune cannot
move the critic's `z`.

### Chunk MDP

One inference commits one action chunk. That chunk is one MDP step:

| | |
| --- | --- |
| `s` | obs at chunk start (images + proprio + prompt) |
| `a` | executed chunk, flat |
| `ã` | specialist / frozen-expert chunk for the same `s` |
| `r` | 1 if the episode succeeds inside the chunk, else 0 |
| `done` | 1 if the episode ends inside the chunk |
| `s'` | next chunk start (or terminal obs) |

Replay starts from 100 frozen-VLA (or specialist) episodes. Online npz are
appended. Stratify success/fail **episodes** 50/50 when the buffer is
imbalanced.

### Critic

`Q(s, flatten(a))` on the chunk, not per-step `(a_t, t)`. Min over heads.
Live TD, target clipped `[0, 1]`. Bootstrap `a'(s')` from the **EMA actor**
(ema 0.999). Critic Polyak `τ=0.005`.

```
y     = clip(r + γ (1−done) min_k Q_target_k(s', a'(s')), 0, 1)
L_td  = MSE(Q_heads(s, a + ε), y)
```

`γ = 0.99` **per decision chunk**. Actor uses **stop-grad critic weights**
so `−Q` cannot inflate Q. Live Q is not clipped.

### Actor (the only place the benches differ)

Ten Euler steps, same noise for `a` and `ã` when both are unrolled.

```
ã = Euler(v_ref,  noise)     # stop-grad specialist
a = Euler(v_θ,    noise)     # live policy; Δ=0 copies ã
```

Identity at init:

- **Radio:** LoRA `B=0`, `lora_scale=0` is an exact `gemma_300m` copy.
- **Pick:** compose `G=0` (or a BC-converged corrector) copies `ã`.

### Loss

```
t ∼ U(0,1)
x_t   = (1−t) · noise + t · ã
L_bc  = MSE(v_θ(x_t, t), v_ref(x_t, t))

L_π   = λ_π · (−mean min_k Q_k(s, a))
      + β · mean‖a − ã‖² / dim(a)
      + L_bc

L     = 4 L_td + L_π + L_ro
```

`λ_π = 0` in AE / AC pretrain, `1` online. `β=100` per dimension.

### Training protocol

1. **Data.** 100 frozen-VLA / specialist rollouts → decision-chunk replay.
2. **AE pretrain.** 8000 steps, `L_ro` only. No Euler, no TD.
3. **AC pretrain.** 8000 steps, `λ_π=0`, freeze pool, live critic TD.
   BC + anchor keep `v_θ` a specialist copy.
4. **Probe.** Live actor, **not** written to replay.
5. **Online.** Collect then `UTD=5` off-policy updates. `λ_π=1`, freeze
   pool, live TD, `L_ro` on. Publish the **EMA** actor for serve.

Kill if `|Q| > 2` or actor–ref RMSE is huge relative to the chunk
(radio threshold `0.04`).

## Small architecture change: Pick vs radio

The method is the block above. Pick cannot LoRA the OpenPI
`gemma_300m` (MolmoSpaces serves a frozen π0.5 and a small PyTorch
head). So `v_θ` is a compact flow MLP on the RL token. That is the
whole architectural delta.

| | Radio (V23) | Pick (V22 instantiation) |
| --- | --- | --- |
| Env | BEHAVIOR-1K `turning_on_radio`, R1Pro | MolmoSpaces Pick, Franka |
| Chunk `a` | `32 × 23` | `8 × 8` (delta) |
| `v_ref` | frozen `gemma_300m` (`lora_scale=0`) | frozen π0.5 chunk `ã` (OT base `ã − noise` if compose) |
| **`v_θ`** | **LoRA rank 32 on expert attn/ffn** | **`FlowActor` MLP, 4×512, on `(z, x_t, t)` + optional `ã`** |
| Identity | `B=0` | `G=0` (compose) |
| Critic | CFTrunk 1536, **10** heads | DoubleCritic, **2** heads, hidden 256 |
| `z` encoder | `CFTokenPool` 256-d, 8 queries | `RLTokenAE` (`z_dim=256`) |
| Serve | EMA LoRA written onto the expert (`cf_live.npz`) | EMA `FlowActor`; VLA still emits `ã` |
| Code | `openpi_comet/` `compute_cf_v23_loss` | `pi05_rl_token/` `FlowRLTAgent` |

Radio LoRA is required because a separate `G` on this expert stayed at
`G / x_ref ≈ 1%` and never left the specialist floor
([`V23_attempts.md`](V23_attempts.md)). Pick's 8-d delta chunk is small
enough that the MLP corrector moves; LoRA inside that stack is not
wired, so it is not used.

Do not describe Pick as a second method. Do not bring back radio `G` /
`ot_compose_integrate` as the Pick analogue — those paths are dead on
radio. Pick's `flow_compose=true` arm is the identity-at-init MLP, not
the abandoned radio guide.

### Pick-only knobs (not architecture)

These are bench scale, not a different algorithm:

- Probe 10 episodes (radio: 1 on instance 308).
- Online 300 stored episodes, last-10 SR, held-out 64 (radio: 50×1-ep
  smoke on 308, no held-out pool).
- `gate_step=0` (RL from `t=0`). Older mug tables used catalog gate 56.
- Current Pick code bootstraps with `γ^C` (`C=8` → `0.99^8 ≈ 0.92`).
  The **method** is per-chunk `γ=0.99` as on radio; `γ^C` on Pick is a
  leftover, not a claim. Radio cannot use `γ^{32}`.

### Which Pick arm is the V23 twin

[`V22_METHODS.md`](V22_METHODS.md) compared several actors. For this
unified method use **compose + live TD** (`flow_compose=true`,
`flow_freeze_critic_online=false`), optionally with AE finetune
(`cf_ae`). That is `V = v_π0.5_base + G`, `G=0` at init — the same
identity story as LoRA `B=0`.

One-pass Gaussian (V21) is a baseline, not this method.

## Instantiation details

### Radio (`cf_v23_ae2`)

- Specialist `pi05-b1kpt12-cs32`. Freeze VLM, base expert,
  `action_{in,out}_proj`, `time_mlp_*`. Train `.*lora.*` and live `cf_*`.
- `ã` is re-unrolled frozen expert (collect-time chunk is not stored as
  `reference`; Euler noise can differ).
- TD action noise `σ=0.08`. `cf_critic_coef=4` in the joint loss.
- Timeout 4300 env steps. Success `q_score.final ≥ 1`.
- See [`V23_METHODS.md`](V23_METHODS.md) for freeze filter, EMA shadow,
  kill, and the finished 50-round log.

### Pick (`flow_rlt`)

- Frozen `pi05_droid_finetune_pick_full_v3_39999`. RL state
  `x = (z_rl, proprio)`.
- `FlowActor`: `v(x, x_t, t)` (compose) or `v(x, x_t, t, ã)` (corrector).
  Euler 10 steps, PyTorch time `t=0` noise → `t=1` data (same 10-step
  FM; opposite OpenPI clock).
- Compose BC drives `G → 0` so `V` reproduces `ã`. Online `λ_π` grows
  `G`.
- `beta=100` on the **sum** of per-dim squares in the current code
  (radio normalizes by `32×23`). Keep β in the per-dimension sense in
  the paper (`β=100` after dividing by `dim(a)`).
- See [`V22_METHODS.md`](V22_METHODS.md) for mug arms and the 18-object
  collect table.

## What not to claim

- That Pick LoRA-s the action expert (it does not).
- That radio uses the Pick `G` MLP (abandoned; LoRA is the actor).
- That critic head count or `γ^C` vs per-chunk γ is a new method.
- That V21's one-pass Gaussian is this method.
