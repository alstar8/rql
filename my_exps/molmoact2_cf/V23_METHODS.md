# V23: cf_ae adapted to BEHAVIOR-1K `turning_on_radio` (openpi_comet / pi0.5)

V23 is V22 `cf_ae` on BEHAVIOR-1K `turning_on_radio` in the `openpi_comet` JAX
stack. The base VLA is the Compet-pt12 pi0.5 specialist (`pi05-b1kpt12-cs32`,
action horizon 32). Modules (CFTokenPool, CFTrunk, ensemble critic) keep the
V17 shapes. The **actor** is LoRA on the frozen pi0.5 **action expert**
(`gemma_300m`), not an OT residual MLP `G` and not `v_θ − G` injection.

- Freeze the VLM and the base action expert. LoRA adapters on `gemma_300m` are
  the trainable flow (V22's `G`, but added to the real `v_θ`).
- `lora_scale=0` Euler is the frozen pt12 chunk `ã`. `lora_scale=1` Euler is
  the executed chunk `a`. Same noise ⇒ `Δ=0` copies pi0.5 exactly.
- Actor loss is V22 Eq. 5: `E[ −Q(s, a) + β‖a − ã‖² ]` plus velocity BC
  `MSE(v_live, v_frozen)` on the OT path to `ã`, `β=100`. Live Q is not clipped;
  the TD target is. `actor_coef` gates only `−Q`.
- Critic: live TD, target clipped `[0, 1]`, bootstrap `a'(s')` from the EMA
  LoRA actor (0.999). Discount is **per decision chunk** (`cf_gamma=0.99`),
  not `γ^32`.
- AE `L_ro` on the live encoder; RL uses a **frozen** target-pool snapshot.
- Serve publishes the EMA LoRA into the live expert slots (`cf_live.npz`).

**Current job: `cf_v23_ae2`.** Do not resume `cf_v23_radio` (injection),
`cf_v23_ot` (analytic OT + MLP `G`), or `cf_v23_ae` (LoRA ×10 / `γ^32` /
cosine LR to 0). Those GraphDefs or hyperparameters are incompatible with
this recipe.

Code: `openpi_comet/` (`models/pi0_cf.py` `compute_cf_v23_loss`,
`training/cf_ae_replay.py`, `training/cf_live.py`, `scripts/train.py`,
`scripts/run_cf_v23_radio.sh`). Checkpoints:
`openpi_comet/checkpoints/cf_v23_ae2/`. Online rollouts:
`openpi_comet/outputs/cf_v23_ae2_online/round_N/`. Driver:
`outputs/cf_v23_ae2_driver.log`. Status: `outputs/cf_v23_ae2_status.log`.

## What G is (and is not)

`G` on V23 is **LoRA on the frozen pi0.5 action expert**, not an MLP on RL
tokens and not a second copy of `gemma_300m`.

| Piece | Role |
| --- | --- |
| Frozen `gemma_300m` (`lora_scale=0`) | pt12 expert `ã`. Never trained. |
| Live LoRA (`lora_scale=1`) | `G`. `a = Euler(v_θ + ΔW)`. `Δ=0` copies pt12. |
| `cf_token_pool` (AE encoder) | RL `z` for **Q**, not the actor residual. |
| CFTrunk / critic heads | `Q(s, flatten(a_chunk))` on `[z, a]`. |
| `cf_guide_trunk` / `cf_guide_head` | Leftover OT-compose MLP. **Unused** by `compute_cf_v23_loss` and serve. |

V22's `G` was an unbounded MLP in **action** space on top of analytic OT
`ã − noise`, because that PyTorch stack could not backprop through pi0.5.
OpenPI has the real expert, so `V = v_frozen + Δ_LoRA`. The MLP `G` port
(`cf_v23_ot`) never left a ~1% residual and was abandoned.

**Who trains.** Freeze filter: VLM + **base action expert** + `cf_target_*`
(including target LoRA) stay frozen in every phase. Live LoRA, live critic,
and live AE encoder/decoder are the trainable set. Online does **not**
finetune pi0.5 weights; it only trains LoRA (`G`) plus critic/`L_ro`.

**`actor_coef`.** Scalar on Q-ascent only. Anchor and BC stay on.

```
L_π = actor_coef × (−Q(s, a)) + β‖a − ã‖² + BC
```

- `actor_coef=0` (AE + AC pretrain): copy `ã`. No `−Q`.
- `actor_coef=1` (online): raise Q of the emitted chunk, still held by `β=100`
  and BC. Critic TD, `L_ro`, freeze flags, and serve Euler are independent of
  this scalar.

**LoRA LR is ×1 in every phase.** `OPENPI_CF_LORA_LR_MULT=10` was a workaround
for `β=100` + BC winning over default Adam. Through 10-step Euler it blew the
pt12 copy (`actor_ref_rmse` 0.004 → 0.042 in ~200 online steps) while
`dq_da_rms` stayed ~3e-4. Logged `lora_lr_mult` must be **1**. Q-ascent is
allowed to move LoRA only after the critic actually depends on the action.

## Why `cf_v23_ae` failed (50 rounds on instance 308)

The first action-expert job (`cf_v23_ae`, finished 2026-08-30) used the V22
*loss string* on the wrong MDP numbers and a LoRA step that the Euler graph
cannot absorb. Frozen pt12 on 308 is **8/10 = 80%**. Online was **18/50 =
36%**, last-10 **4/10**. Probe 0/1 timeout. Failures were almost all 4300-step
timeouts. No `_KILLED` (rmse peak 0.063, kill was 0.10).

| Knob | `cf_v23_ae` (failed) | Effect |
| --- | --- | --- |
| Discount | `γ^32 ≈ 0.725` per chunk | Typical 308 success ~35 chunks. First-chunk return `0.725^35 ≈ 4e-5`. Q collapsed to `V(s)`. |
| LoRA Adam | ×10 online | Residual 2.4% → **22%** after round 1. Kill 0.10 never fired. |
| LR | cosine `1e-4 → 0` at 40k | Online starts at 16k; adapters frozen by ~round 40. |
| Replay | filtered to instance 308 | 10 trajs, **8 chunks with `r=1`**. Batch-8 TD never saw action ranking. |
| `dq_da_rms` | ~3e-4 the whole run | `−Q` through 10 expert Euler steps was transformer noise, not V22's MLP `∂Q/∂a`. |

Rounds 1–4 still succeeded because serve loads **EMA** LoRA (`ema=0.999`).
After ~1500 online steps the blown adapters mixed in; SR dropped below the
VLA floor and never recovered. Injection / OT-compose cousins stayed near the
80% floor only because `G` never moved (`q_gap≈0`). This LoRA×10 job is the
opposite: `G` moved a lot, and success **fell**.

Do not resume `checkpoints/cf_v23_ae/`.

## Current recipe: `cf_v23_ae2`

Same actor (LoRA on `gemma_300m`) and same 1-ep instance-308 smoke protocol.
Knobs that actually blocked SR were changed; the expert is still not V22's
action-space MLP.

| Knob | `cf_v23_ae` | `cf_v23_ae2` (now) |
| --- | --- | --- |
| Discount | `cf_gamma ** 32` | **per-chunk `cf_gamma=0.99`** (`cf_discount_per_chunk=1`). 35-chunk return `0.99^35 ≈ 0.70`. |
| LoRA LR | ×10 online | **×1** every phase |
| LR schedule | cosine to 0 at 40k | **`WarmupConstantSchedule` peak 1e-4** (warmup 200) |
| Replay filter | `OPENPI_CF_V23_INSTANCE=308` | **`OPENPI_CF_REPLAY_INSTANCE` empty = all 100 trajs**. Collect still 308. |
| Critic coef | 2 | **4** |
| TD action noise | 0.05 | **0.08** |
| Kill `actor_ref_rmse` | 0.10 | **0.04** (damage in the ×10 run started at ~0.04) |

Collect instance is still 308 (eval id 7). Replay at AE start: **9650 chunks
(2360 success / 7290 fail)** from the full pt12 10×10 buffer.

### Method vs V22 (only task / network leftovers)

| Change | Where | V22 |
| --- | --- | --- |
| Actor = LoRA on the pi0.5 action expert | `action_expert_variant=gemma_300m_lora` | `FlowActor` MLP `G` on analytic OT |
| Frozen expert unroll is **only** ã (`lora_scale=0`) | `_unroll_endpoint` | HTTP VLA chunk stored as `reference` |
| Live expert unroll emits a (`lora_scale=1`, same noise) | `_unroll_endpoint`, `sample_actions` | `_integrate` + `flow_compose=true` |
| OT-path BC: `t∼U(0,1)`, `MSE(v_live, v_frozen)` | `compute_cf_v23_loss` | `F.mse_loss(G + (ã-x0), ã-x0)` |
| `L_π = actor_coef(−Q) + β‖a−ã‖² + bc` (anchor always on) | `compute_cf_v23_loss` | same; `flow_actor_coef` gates only −Q |
| Live TD, target clip `[0, 1]`, `a'` from EMA LoRA | `compute_cf_v23_loss`, `cf_target_lora` | `critic_step` + `_integrate(actor_target)` |
| Discount **per chunk** `γ=0.99` | `cf_discount_per_chunk` | `gamma ** chunk` (`C=8` → `γ^8≈0.923`) |
| RL z from **frozen target pool**; live pool is L_ro only | `cf_target_token_pool`, polyak skipped when `freeze_pool` | on-disk encoder writes buffer z |
| Serve publishes target pool + **target LoRA** | `cf_live.py` payload `lora` | `actor_target` (EMA 0.999) + frozen encoder |
| LoRA target EMA 0.999; critic Polyak `τ=0.005` | `train.py` `polyak_cf_target_lora` | `cf_ema=0.999`, critic `tau=0.005` |
| `cf_critic_coef=4` (joint-loss analogue of extra critic SGD) | config | `critic_updates_per_actor=2` |
| `cf_w_l2_coef=0` | config | no ‖W‖² |
| `actor_coef=1` from round 1 | `run_cf_v23_radio.sh` | `flow_actor_coef=1`, warmup=0 |
| Train steps = `UTD × n_chunks` (`UTD=5`) | `count_chunks` | utd=5 per stored decision |
| Peak LR `1e-4`, **constant after warmup 200** | `WarmupConstantSchedule` | constant `3e-4` |
| LoRA Adam ×1 | `cf_lora_lr_mult` | N/A (MLP `G` has its own Adam) |

Keep (do not revert): `_critic_with_sg` so `-Q` cannot inflate the critic
(V22: critic is a separate Adam), frozen VLM + base expert, `beta=100`,
kill switch, velocity BC, per-chunk discount, constant LR, LoRA ×1.

Task/network only: chunk 32×23 vs 8×8, B1K proprio 23-d vs 16-d, `z` 1024 vs
256, CFTrunk 1536 + 10 Q heads vs 2-layer MLP twin-Q, batch 8 vs 256 (PaliGemma
still in the TD graph), 1-ep instance-308 smoke vs 300-ep `episode_pool=0-11`.
Anchor is divided by `32×23` so β=100 stays per-dimension (V22 uses the raw
sum over 64 dims).

Env knobs (defaults = `cf_v23_ae2`): `OPENPI_CF_FREEZE_CRITIC=0`,
`OPENPI_CF_FREEZE_POOL=1`, `OPENPI_CF_UNIT_BALL=0`, `OPENPI_CF_ACTOR_COEF`
(0 pretrain / 1 online), `OPENPI_CF_BC_COEF=1.0`, `OPENPI_CF_ANCHOR_BETA=100`,
`OPENPI_CF_UTD=5`, `OPENPI_CF_W_L2=0`, `OPENPI_CF_CRITIC_COEF=4`,
`OPENPI_CF_GAMMA=0.99`, `OPENPI_CF_DISCOUNT_PER_CHUNK=1`,
`OPENPI_CF_ACTOR_EMA=0.999`, `OPENPI_CF_V23_INSTANCE=308`,
`OPENPI_CF_REPLAY_INSTANCE=` (empty = all trajs),
`OPENPI_CF_KILL_REF_RMSE=0.04`, `OPENPI_CF_CHUNK_CRITIC=1`,
`OPENPI_CF_TD_ACTION_NOISE=0.08`, `OPENPI_CF_LORA_LR_MULT=1`.

### Code (action-expert path)

V22 compose is `V = v_pi05_base + G`. On OpenPI, `v_pi05_base` is the frozen
pt12 action expert and `G` is LoRA on that expert (rank 32, alpha 32,
`gemma_300m_lora`). Analytic OT `G` and `v_θ − G` injection are gone from the
V23 train/serve path.

| Piece | What it does |
| --- | --- |
| Config | `action_expert_variant=gemma_300m_lora` (attn+ffn rank 32 / alpha 32, **B=0 init**). Weight loader `missing_regex=".*cf_.*|pointnet.*|.*lora.*"` (pt12 is LoRA-free). Exp default from `OPENPI_CF_V23_EXP` (`cf_v23_ae2`). Peak LR `1e-4` constant after warmup 200, LoRA updates ×1, `save_interval=1000`, batch 8. |
| Freeze | `cf_v23_freeze_filter`: train live `cf_*` **and** `.*lora.*`; freeze VLM, base `gemma_300m`, `action_{in,out}_proj`, `time_mlp_*`, and `cf_target_*`. |
| Critic | `Q(s, flatten(a))` on CFTrunk. Not per-step `(a_t, t)`. TD action noise 0.08. Log `dq_da_rms`, `q_success`, `q_fail`, `chunk_discount`. |
| Forward `lora_scale` | Threaded through `lora.Einsum` / `lora.FeedForward` and gemma `Attention` / `Block` / `Module` (`nn.scan` extra `broadcast`). `_euler_v` passes it into `PaliGemma.llm`. `0` zeros the LoRA delta without swapping weights. Prefix LLM calls leave the default (`1`); VLM is `gemma_2b` (no adapters). |
| Train-log `lora_scale` | Separate: `train.py` multiplies LoRA **grads** by this scalar (V17 freeze-after-warmup). V23 keeps it at 1. AE logs `lora_scale=1` even while Euler is off — that is the grad multiplier, not the forward scale. |
| ã | `_unroll_endpoint(..., stop_v=True, lora_scale=0)` — frozen pt12 expert, stop-grad. `guide_fn` is ignored (the expert *is* the actor). |
| a | `_unroll_endpoint(..., stop_v=False, lora_scale=1)` — live LoRA, **same noise** as ã. Grads through the 10-step Euler. AE-only (`cf_ae_only=1`) skips both unrolls. |
| BC | OpenPI time `t∼U(0,1)`, `x_t = t·x0 + (1-t)ã`, `MSE(v_live, sg(v_frozen))`. Logged as `lora_rms` / `guide_rms` (Δv RMS). |
| EMA shadow | `CFLoraShadow` (`cf_target_lora`): copies live LoRA at init into `t0..tN` Params so Orbax checkpoints them. `to_lora_pure_dict` rebuilds the nested adapter tree via `getattr(leaf, "value", leaf)` (JIT leaves may already be arrays). Frozen by `cf_target_*`. |
| `a'(s')` | `_fork_with_lora_pure` splits `self`, overlays `cf_target_lora.to_lora_pure_dict()` onto the flattened State **pure dict**, `replace_by_pure_dict` + `nnx.merge`, then unrolls `lora_scale=1`, stop-grad. Do **not** call `Param.replace` — under `nnx.value_and_grad` leaves are `ShapedArray` / `JVPTracer`. |
| Serve | `sample_actions` is live-expert Euler (`lora_scale=1`). V23 does **not** subtract G and does **not** OT-compose. `OPENPI_CF_APPLY_GUIDE` / `apply_guide` is ignored when `cf_v23=True`. `cf_live.npz` payload key `lora` is the **target** adapters (`_extract_lora_pure`); `load_cf_live_into_model` writes them onto live LoRA via `CF_LIVE_LORA_FILTER` and re-JITs. Default live path is `checkpoints/${EXP}/cf_live.npz`. |
| Polyak | Critic `τ=0.005` on `CF_V23_TARGET_PAIRS`. Actor EMA via `polyak_cf_target_lora` after that loop. `freeze_pool=1` still skips pool polyak. |
| Guide MLP | `cf_guide_trunk` / `cf_guide_head` stay in the graph (V17 shapes / opt_state) but are unused by the expert actor. |
| OT helper | `ot_compose_integrate` remains in `pi0_cf.py` for tests. V23 loss and `sample_actions` do not call it. |
| LR | `WarmupConstantSchedule` in `training/optimizer.py`. Cosine-to-zero is SFT-only; do not use it for RL. |
| Replay instance | `OPENPI_CF_REPLAY_INSTANCE` in `data_loader.py`. Empty = full 100-traj buffer. Collect instance is `OPENPI_CF_V23_INSTANCE` (serve/client only). |
| Tests | `test_v23_lora_filter_excludes_shadow`; `test_chunk_critic_depends_on_action`; `test_warmup_constant_lr_does_not_hit_zero`; `test_per_chunk_discount_keeps_credit_over_radio_horizon`. |

Three incompatible checkpoint families: injection (`cf_v23_radio`), OT-compose
(`cf_v23_ot`), action-expert ×10 (`cf_v23_ae`). Resume only `cf_v23_ae2`.

## What V22 cf_ae is (reference)

- Frozen VLA produces the reference chunk `ã` (stored on the replay row).
- Actor = CF composition `V = v_pi05_base + G` (`flow_compose=true`). In V22
  `v_pi05_base` is the analytic OT line `ã − noise` because the PyTorch RLT
  stack cannot run pi0.5 inside the actor; `G` is an unbounded MLP. Euler
  **emits** the chunk. OpenPI V23 uses the real action expert as `v_pi05_base`
  and LoRA as `G`.
- Actor loss: `E[ −Q(s, a) + beta * ||a − ã||² ]` with `beta=100`, plus
  OT-path BC. Live Q is not clipped; only the TD *target* is. `actor_coef`
  gates only −Q; the anchor and BC stay on in pretrain.
- Critic: twin Q, TD every utd step, target clipped `[0, 1]`, bootstrap
  `a'(s')` from the target actor. Discount `γ^C` with `C=8`. **Not frozen
  online.** Constant lr `3e-4`.
- AE reconstruction finetune; the `z_rl` written to the buffer stays the frozen
  on-disk encoder.
- Learner: utd=5 actor updates + 2 critic updates **per stored decision chunk**.
- Batch 256, critic is a cheap MLP on cached `(z_rl, proprio, a)`.
- 300 episodes, `episode_pool=0–11`, metric = last-10 SR. Mug gate0: online
  96.3%, last-10 100%, held-out 100%. Frozen-critic cousins are worse
  (external frozen V21 critic 67% online).

## V23 mapping (V22 → openpi_comet)

| V22 (MolmoSpaces) | V23 (`cf_v23_ae2`) |
| --- | --- |
| Frozen HTTP pi0.5 emits ã | Frozen `gemma_300m` (`lora_scale=0`) Euler emits ã |
| `G` MLP + analytic OT emits a | LoRA on the action expert (`lora_scale=1`) Euler emits a |
| Token AE `z_rl` (encoder frozen in buffer) | CFTokenPool = V22 TokenEncoder. RL uses `sg(target_pool(s))`. Live pool: `L_ro` only |
| proprio concat | normalized 23-d state concatenated to pooled features |
| DoubleCritic twin-Q, TD online | `cf_critic_heads` ensemble (10 heads), TD online |
| 100 VLA episodes buffer | 100 collected `state_action.npz` (pt12 rollouts, **all instances**) |
| 300 online eps, pool 0–11 | 50 rounds × **1 ep** on instance 308 (smoke) |
| `actor_target` EMA 0.999 | `cf_target_lora` EMA 0.999, critic τ=0.005 |
| `gamma ** 8 ≈ 0.923` per chunk | `cf_gamma=0.99` per chunk (`0.99^35 ≈ 0.70`) |
| constant lr 3e-4 | warmup 200 then constant 1e-4 |

## Decision-chunk MDP

The env steps at 30 Hz control but the policy commits to a 32-step action chunk
(re-inference only when the queue is empty; queue cleared at episode reset, so
chunk boundaries align to a 32-step grid from t=0). V23 treats each 32-step
chunk as one MDP decision:

- `s`     = observation at chunk start (3 images + 256-d proprio → 23-d state + prompt tokens)
- `a`     = the 32×23 action chunk executed
- `r`     = 1 if the episode terminated with success inside the chunk else 0
- `done`  = 1 if the episode ended (terminated or truncated) inside the chunk
- `s'`    = observation at the start of the next chunk (or terminal obs)

This is the granularity at which the actor acts, so the TD backup is
well-defined and matches V22's `(s, a_chunk, r, s')` transitions.

`state_action.npz` already stores everything needed per step: `state` (256-d),
`action` (N,1,23), `next_state`, `next_action`, `reward`, `done`, `truncated`,
`actor_obs__0..3` (proprio + head/left/right images), `next_actor_obs__*`,
`metadata` (`success`, `prompt`). The V23 replay dataset
(`openpi/training/cf_ae_replay.py`) chunks each episode on the 32-step grid and
emits one sample per decision chunk. `n_chunks = ceil(T / 32)` (timeout 4300 →
135 chunks → 675 train steps at utd=5).

## Model / loss (`pi0_cf.py`, `cf_v23=True`)

State features: `phi(s) = [ target_pool(prefix_tokens), normalize(state_23) ]`
(1024 + 32 padded). Critic is V22-style `Q(s, flatten(a_chunk))`: CFTrunk on
`[phi, a_flat]` (32×32 padded = 1024-d), **not** a per-step mean over
`(a_t, t_emb)`. The live pool is trained by `L_ro` only. `cf_guide_*` is
unused by the expert actor.

Critic (twin-Q TD, V22-style; **live online**):

- `y = r + γ_chunk * (1 - done) * min_k Q_target_k(s', a'(s'))`, clipped to `[0, 1]`.
- `ã(s')` = frozen expert Euler at `s'`. `a'(s')` = EMA LoRA expert (stop-grad).
- `γ_chunk = cf_gamma = 0.99` when `cf_discount_per_chunk=True`. Legacy
  `cf_gamma ** action_horizon` is off.
- Loss: MSE over all 10 heads, weighted `cf_critic_coef=4`. TD sees replay `a`
  plus Gaussian noise (`OPENPI_CF_TD_ACTION_NOISE=0.08`) so Q cannot ignore
  the action.
- `dq_da_rms` = RMS of `∂ min_k Q_k(s,a) / ∂a` (stop-grad critic weights).
  If this stays ~0, the actor has nothing to climb.
- `q_success` / `q_fail` = mean Q of batch rows with episode-success vs fail
  labels. With per-chunk discount these should separate; they did not under
  `γ^32`.
- `OPENPI_CF_FREEZE_CRITIC=1` zeros the TD term.

Actor (Eq. 5 anchor + raw Q-ascent + velocity BC on the action expert):

- `ã` = 10-step frozen-expert Euler (`lora_scale=0`, stop-grad).
- `a` = 10-step live-expert Euler (`lora_scale=1`), **same noise** as `ã`.
  LoRA B is **zero-init** so `a=ã` at step 0 (V22 `G=0`).
- Actor Q uses **stop-grad critic weights** (`_critic_with_sg`). Ascent is
  **raw** `−mean min_k Q_k(s, flatten(a))`. `q_gap = Q_guided − Q_ref` is
  logged only. State features are `sg(target_pool(s))`.
- BC: sample OpenPI-time `t∼U(0,1)`, `x_t=t·x0+(1-t)ã`,
  `MSE(v_live(x_t,t), sg(v_frozen(x_t,t)))`.
- `L_actor = cf_actor_coef * (−Q) + beta * ||a − ã||² / (32×23)
  + cf_bc_coef * MSE(Δv)`, `beta = 100`. Anchor and BC stay on when
  `actor_coef=0`.
- LoRA Adam updates are scaled by `OPENPI_CF_LORA_LR_MULT=1`. Logged
  `lora_grad_norm` and `lora_lr_mult`.
- `L_ro = cf_ae_coef * MSE(decoder(z_live), sg(prefix_tokens))`.
- `L = cf_critic_coef * L_td + L_actor + L_ro`. No ‖W‖² (`cf_w_l2_coef=0`).

Freeze filter (`cf_v23_freeze_filter`): train live `cf_*` **and** action-expert
LoRA (`G`); freeze VLM, **base** `gemma_300m`, and `cf_target_*` (including
`cf_target_lora`). The base expert is never an online finetune target. Pool /
critic freezes are zeroed Adam updates so resume `opt_state` shape is unchanged.
`freeze_pool=1` stops polyak on the target pool so AE finetune cannot move RL z.

Target networks: explicit `cf_target_*` plus `cf_target_lora`. LoRA polyak
`ema=0.999`; critic `tau=0.005`. Serve loads the **target** pool and **target
LoRA** under the live names.

`cf_q_clip_max` is still on the config; the actor term does not clip live Q
(V22). TD targets stay clipped `[0, 1]`.

## Training protocol (AE + live-TD 1-ep)

1. **Data**: 100 `state_action.npz` from the pt12 collector
   (`outputs/pt12_cs32_radio_10x10_states`), public_test instances 0–9.
2. **AE pretrain**: 8000 steps, `cf_ae_only=1`, `cf_ae_coef=1`, `freeze_critic=1`,
   `freeze_pool=0`, `actor_coef=0`. Encoder+decoder train `L_ro` on stop-grad
   Paligemma prefix tokens. No Euler unroll, no TD.
3. **AC pretrain**: 8000 more steps (target 16000), `actor_coef=0`,
   `freeze_pool=1`, `freeze_critic=0`, `ae_coef=0`. Critic TD on
   `sg(target_pool(s))`. LoRA BC + anchor keep the expert a copy of pt12.
4. **Probe**: 1 episode on instance 308, live expert, **not** written to replay.
5. **Online**: 50 rounds × {collect 1 ep on 308 + train `5 × n_chunks` steps}:
   - `actor_coef=1`, `freeze_critic=0`, `freeze_pool=1`, `ae_coef=1`
   - `freeze_pool=1` stops polyak on the target pool. Live encoder trains
     `L_ro`; RL z is the frozen snapshot (`sg(target_pool(s))`)
   - One serve + one client (`OPENPI_CF_V23_INSTANCE=308` → eval id 7)
   - Publish `cf_live.npz` (frozen encoder / critic trunk / **target LoRA**)
   - Replay = 100 pretrain (all instances) + all online npz, stratified
     50/50 success/fail
   - Kill: `|actor_q| > 2` or `actor_ref_rmse > 0.04` or non-finite grads

Success = `q_score.final >= 1`; timeout = 4300 env steps (~143 s sim). Typical
pt12 success on 308 is ~800–1343 steps (~30–45 s). Slow successes (2000–3200)
still count as `q_score=1`.

## Results: `cf_v23_ae2` (live)

Started 2026-08-31 16:08 local (`13:08 UTC`) from pt12 (`OPENPI_CF_V23_EXP=
cf_v23_ae2`, instance 308, `REPLAY_INSTANCE` empty). Driver
`outputs/cf_v23_ae2_driver.log`, status `outputs/cf_v23_ae2_status.log`,
watchdog `~/cf_v23_ae2_watchdog.log`. No `_KILLED` / `_DONE`.

Heartbeat at start: `lora_lr_mult=1 discount_per_chunk=1 critic_coef=4
td_noise=0.08 kill_rmse=0.04 replay_instance=all`.

| Phase | When (local) | Result |
| --- | --- | --- |
| AE 8000 | 16:08–16:36 | done (`ae_only=1`) |
| AC → 16000 | 16:36–17:55 | done. `chunk_discount=0.99`, `lora_lr_mult=1`. End: `rmse=0.014`, LoRA residual **3.4%**, `q_gap≈0`, `q_success=0.34` / `q_fail=0.16` |
| Probe | 17:55–18:00 | **1/1** success, 1324 steps (ckpt 15999) |
| Online | 18:00– | running. Step **20454** after round 10. Round 11 collecting (ckpt 20454) |

pt12 VLA on 308: **8/10 = 80%**.

### Online rounds (as of 2026-08-31 20:44 local)

| Round | Result | Env steps | Ckpt after train |
| ---: | --- | ---: | ---: |
| probe | success | 1324 | 15999 |
| 1 | success | 1333 | 16208 |
| 2 | success | 1368 | 16422 |
| 3 | success | 1460 | 16651 |
| 4 | success (slow) | 2276 | 17010 |
| 5 | fail | 4300 | 17684 |
| 6 | success | 1843 | 17972 |
| 7 | fail | 4300 | 18647 |
| 8 | fail | 4300 | 19321 |
| 9 | success (slow) | 2927 | 19780 |
| 10 | fail | 4300 | 20454 |
| 11 | collecting | — | — |

**Online 6/10 = 60%**, last-10 **6/10**. Below the 80% VLA floor, above the
×10 job's 36%. Failures are timeouts, not crashes. R4 and R9 are slow
successes (`q_score=1`).

### Train diagnostics vs pt12 copy

Frozen pt12 (old run, AC end): `rmse≈0.004`, residual ≈2%, `q_gap≈0`.
`cf_v23_ae` after one ×10 round: `rmse=0.042`, residual **22%**.

| Burst | rmse | residual | q_gap | dq_da | q_success | q_fail | actor_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AC end 15980 | 0.014 | 3.4% | ~0 | 4e-4 | 0.34 | 0.16 | 0.25 |
| R1 | 0.017 | 11% | 0.027 | 4e-4 | 0.43 | 0.17 | 0.30 |
| R5 | 0.015 | 8.7% | 0.029 | 4e-4 | 0.51 | 0.23 | 0.41 |
| R8 | 0.012 | 7.8% | 0.024 | 4e-4 | 0.66 | 0.36 | 0.56 |
| R10 (latest) | 0.023 | 7.6% | 0.015 | 3e-4 | 0.73 | 0.46 | 0.61 |

The policy has **not** left the VLA manifold: residual ~8%, `rmse` 0.012–0.023
vs kill 0.04. That is a small, stable LoRA residual, not the ×10 blow-up.
Q of success rows rose (0.34 → 0.73) and now separates from fail (0.16 → 0.46)
— per-chunk discount + full buffer are doing something the `γ^32` / 308-only
critic did not. `dq_da_rms` is still ~3e-4, so the actor is not climbing a
real `∂Q/∂a`.

## Problems (open)

These are why 60% on 308 is not yet a learned radio policy.

1. **SR is still below the pt12 floor.** 6/10 vs 8/10. Last four finished
   rounds are 1/4 (R7–R8 timeout, R9 slow win, R10 timeout). n=10 on one
   instance is noisy, but the recent timeouts are the same failure mode as
   `cf_v23_ae` after the copy drifted.

2. **`dq_da_rms` is still ~3e-4.** The critic ranks episodes (`q_success` >
   `q_fail`) more than it ranks **actions**. `−Q` through 10 transformer Euler
   steps is still a weak, noisy direction. `q_gap` 0.015–0.03 is LoRA being
   slightly preferred, not a V22-style action Jacobian.

3. **The actor is still expert LoRA, not V22's MLP `G`.** Gradients go through
   the 10-step `gemma_300m` Euler. That is the same autodiff path that ×10
   destroyed. ×1 keeps the copy intact; it also means Q-ascent barely moves
   the policy. Matching V22's *mechanism* would be an action-space residual
   on a stored `reference`, which this job does not train.

4. **Slow successes (R4=2276, R9=2927).** Frozen pt12 typically finishes
   308 in ~800–1343 steps. Longer wins still `q_score=1` but look like a
   degraded copy, not a better radio policy.

5. **Batch 8, re-unrolled ã, PaliGemma in the TD graph.** V22 critic is a
   cheap MLP on cached `(z, a)` at batch 256 with the collect-time VLA chunk
   stored as `reference`. Train still re-unrolls frozen pi0.5, so Euler noise
   can differ from collect.

6. **1-ep instance-308 is a smoke test.** Do not call last-10 on n=10 a
   cf_ae result. V22's metric is 300 eps on `episode_pool=0–11` plus held-out
   eval48.

7. **Ops.** A leftover `cf_v23_ae_301` watchdog was `pkill`ing every
   `train.py` in the container; it was killed when `ae2` launched. Host
   `cf_v23_watchdog.sh` on disk was briefly 0 bytes (2026-08-30) and had to
   be restored from git before the ae2 watchdog would start. `outputs/
   cf_v23_ae2_DONE` is the only clean stop; do not start a second driver on
   the same GPU.

If SR stays ≤60% after ~20 rounds with residual still ~8% and `dq_da` still
~3e-4, the next lever is the critic's action dependence (cached `z`, stored
`reference`, larger batch / separate critic SGD) — not another LoRA LR
multiplier.

## Not yet ported (intentional leftovers)

- Cache pooled `z` in replay so critic TD does not re-run PaliGemma. Batch is
  still 8. The **encoder snapshot** used for RL now matches V22 (frozen target
  pool); only the VLM forward is still live.
- Store the collect-time VLA chunk as `reference` (V22 replay field). Train
  still re-unrolls frozen pi0.5 for ã, so Euler noise can differ from collect.
- Multi-instance `episode_pool` / last-10 as the official metric. 1 training
  episode on 308 is a smoke test; do not call it a cf_ae result.
- Two separate critic SGD steps on fresh batches. `cf_critic_coef=4` is the
  joint-loss analogue; V22's learner samples twice.
- `ema_decay` on the full TrainState (`None`). Serve already uses the target
  LoRA (`cf_target_lora`) and target pool.
- Train LoRA on `action_in_proj` / `action_out_proj` / `time_mlp_*`. Those stay
  frozen so `lora_scale=0` is an exact pt12 copy. Plasticity is the expert
  transformer LoRA only.
- Delete `ot_compose_integrate` / `cf_guide_*`. The helper and MLP remain so
  tests and V17 GraphDef shapes stay intact; the V23 actor path ignores both.
- Action-space MLP `G` on analytic OT (V22 `FlowActor`). Still not ported;
  LoRA-in-Euler is the stand-in.

## Files

- `openpi_comet/src/openpi/training/cf_ae_replay.py`  — decision-chunk replay
- `openpi_comet/src/openpi/training/cf_live.py`       — atomic CF live publish (target LoRA)
- `openpi_comet/src/openpi/training/data_loader.py`   — `OPENPI_CF_REPLAY_INSTANCE`
- `openpi_comet/src/openpi/training/optimizer.py`     — `WarmupConstantSchedule`
- `openpi_comet/src/openpi/models/pi0_cf.py`          — V23 LoRA expert loss + freeze + EMA shadow + per-chunk discount
- `openpi_comet/src/openpi/models/lora.py`            — `lora_scale` on Einsum / FeedForward
- `openpi_comet/src/openpi/models/gemma.py`           — thread `lora_scale` through the expert
- `openpi_comet/src/openpi/policies/policy.py`        — CF live poll + re-JIT `sample_actions`
- `openpi_comet/src/openpi/shared/eval_b1k_wrapper.py`— clear action queue on CF reload
- `openpi_comet/src/openpi/training/config.py`        — `pi05_b1k-turning_on_radio_cf_v23` (`gemma_300m_lora`)
- `openpi_comet/scripts/train.py`                     — rebind + freeze-pool/critic zeroing + LoRA polyak + kill 0.04
- `openpi_comet/scripts/serve_b1k.py`                 — serve with `OPENPI_CF_LIVE_PATH`
- `openpi_comet/scripts/run_cf_v23_radio.sh`          — 1-ep live-TD orchestrator (`OPENPI_CF_V23_EXP`)
- `openpi_comet/scripts/cf_v23_watchdog.sh`           — host watchdog
- `openpi_comet/scripts/test_cf_token_ae.py`          — AE + OT helper + LoRA-filter + LR + discount tests

## Watchdog

Action-expert run (`OPENPI_CF_V23_EXP=cf_v23_ae2`): heartbeats
`outputs/cf_v23_ae2_status.log`, stdout `outputs/cf_v23_ae2_driver.log`.
The host-side watchdog (`cf_v23_watchdog.sh`) checks every 20 min: driver
alive (restarts via `docker exec` if dead), and progress advancing. If nothing
progresses for 35 min it kills stuck phase processes; if still stuck at 80 min
it restarts the driver. Completed rounds are skipped via `round_N/.done`.
`outputs/cf_v23_ae2_DONE` ends the watchdog. `outputs/cf_v23_ae2_KILLED` also
ends it (do not restart). Do not run a second `cf_v23_watchdog.sh` pointed at
another exp in the same container — it `pkill`s every `train.py`.
