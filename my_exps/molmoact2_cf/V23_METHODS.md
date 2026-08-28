# V23: cf_ae adapted to BEHAVIOR-1K `turning_on_radio` (openpi_comet / pi0.5)

V23 is V22 `cf_ae` on BEHAVIOR-1K `turning_on_radio` in the `openpi_comet` JAX
stack. The base VLA is the Compet-pt12 pi0.5 specialist (`pi05-b1kpt12-cs32`,
action horizon 32). Modules (CFTokenPool, CFTrunk, ensemble critic) stay the
V17 shapes. The **actor** is the pi0.5 **action expert** (`gemma_300m` + LoRA),
not an OT residual `G` and not `v_θ − G` injection:

- Freeze the VLM and the base action expert. LoRA adapters on `gemma_300m` are
  the trainable flow (V22's `G`, but added to the real `v_θ`).
- `lora_scale=0` Euler is the frozen pt12 chunk `ã`. `lora_scale=1` Euler is
  the executed chunk `a`. Same noise ⇒ `Δ=0` copies pi0.5 exactly.
- Actor loss is V22 Eq. 5: `E[ −Q(s, a) + β‖a − ã‖² ]` plus velocity BC
  `MSE(v_live, v_frozen)` on the OT path to `ã`, `β=100`. Live Q is not clipped;
  the TD target is. `actor_coef` gates only `−Q`.
- Critic: live TD, target clipped `[0, 1]`, bootstrap `a'(s')` from the EMA
  LoRA actor (0.999). Discount is `γ^C` with `γ=0.99` per env step, `C=32`.
- AE `L_ro` on the live encoder; RL uses a **frozen** target-pool snapshot
  (V22's on-disk encoder that fills the buffer).
- Serve publishes the EMA LoRA into the live expert slots (`cf_live.npz`).

**OT-compose job (`cf_v23_ot`) and injection-era jobs are incompatible.** Those
trained a tiny `G` (or `v_θ − G`) while the expert stayed frozen; G never left
a ~1% residual. New jobs must AE-pretrain / AC-pretrain from the VLA weights
with `gemma_300m_lora`.

Code: `openpi_comet/` (`models/pi0_cf.py` `compute_cf_v23_loss`,
`training/cf_ae_replay.py`, `training/cf_live.py`, `scripts/train.py`,
`scripts/run_cf_v23_radio.sh`). Checkpoints:
`openpi_comet/checkpoints/cf_v23_ae/`. Online rollouts:
`openpi_comet/outputs/cf_v23_ae_online/round_N/`. Driver:
`outputs/cf_v23_ae_driver.log`. Status: `outputs/cf_v23_ae_status.log`.

## Why the first V23 ports did not learn

Three failed recipes preceded the live-TD 1-ep run. They matter because V23
cloned the `cf_ae` *formula* and then ran the **losing cousins** of the V22 mug
table (frozen critic, overfit steps, bound G).

### 2026-08-26: AE-unfrozen / `actor_coef=1` offline

Diverged (`actor_q` ~1e5, `x_ref_rms` ~870) → probe/R1 0% SR. Reverted to a
frozen-VLA pretrain.

### 2026-08-26: frozen VLA + `beta=1` + mid-collect CF sync

- Frozen whole VLA (`cf_v23_freeze_filter`); `sg(v_theta)` in guided Euler.
- Pretrain `cf_actor_coef=0`; online `=1`.
- `cf_anchor_beta=1.0` (V22 uses **100**).
- 10 eps / 2000 train steps per round, 50 rounds, 2 workers.
- Mid-collect CF weight sync (5 slices × micro-train → `cf_live.npz`).

**Outcome:** probe 5/10, then online SR → 0% by round 3. Cause: one joint loss
over all live `cf_*` params with **no stop-grad on critic weights** for the
actor term. `-Q` inflated `cf_critic_heads` + the **shared** `cf_trunk`;
`actor_q` ~2e4, `guide_rms` ~93, `actor_ref_rmse` ~5.6 vs frozen `x_ref_rms`
~0.47. `beta=1` could not hold the trust region.

### 2026-08-27 morning: stop-grad restore, then frozen-critic 1-ep

Restored the V22 *trust region* (this is still in the code; do not revert):

- Stop-grad **critic weights** in actor Q (`_critic_with_sg`); dQ/dx still flows.
- Dedicated `cf_guide_trunk` (actor grads never hit the critic trunk).
- `OPENPI_CF_ANCHOR_BETA=100`, velocity BC (`cf_bc_coef=1`, paper-times 0.2 / 0.6 / 1.0).
- Kill switch: `|actor_q| > 2` or `actor_ref_rmse > 0.10` (was 0.05) or
  non-finite CF grads → no save/publish, write `outputs/cf_v23_radio_KILLED`.

Then the 1-ep smoke test ran the **losing V22 arm**: critic+pool frozen after
pretrain, unit-ball + `tmax=0.4`, `actor_coef` ramp 0.1 → 1 → 2, **2000 train
steps per episode**. `cf_guide_trunk` is a new module — old checkpoints are
incompatible; pretrain 1000–7000 was kept, online ckpts `>7000` from this abort
were wiped before the live-TD restart.

## 1-ep frozen-critic abort (instance 308)

Protocol: 1 ep/round, freeze_critic=1, unit-ball on, tmax=0.4, 2000 steps/ep,
actor_coef ramp. Probe 0/1 (timeout 4300; G≈0 after pretrain). Killed during
round 12–13 collect (ckpt ~31999). Trust region was healthy (`rmse` ~0.0035 vs
kill 0.10); G stayed ≤1% of `x_ref` through round 11 collect, then `w_norm`
0.04 → 0.18 at the end of R11 train.

| Round | Result | Env steps | actor_coef | G / x_ref at collect |
| ---: | --- | ---: | ---: | --- |
| probe | fail | 4300 | 0 | ~0 |
| 1 | success | 1366 | 0.1 | ~0 |
| 2 | success | 1339 | 0.1 | 0.2% |
| 3 | fail | 4300 | 0.1 | 0.2% |
| 4 | success | 1357 | 0.1 | 0.3% |
| 5 | fail | 4300 | 0.1 | 0.4% |
| 6 | success | 1355 | 1.0 | 0.5% |
| 7 | fail | 4300 | 1.0 | 0.5% |
| 8 | fail | 4300 | 1.0 | 0.5% |
| 9 | fail | 4300 | 1.0 | 0.4% |
| 10 | success (slow) | 1880 | 2.0 | 0.4% |
| 11 | fail | 4300 | 2.0 | 0.7% |

**Online SR 5/11 = 45.5%**, last-10 **4/10 = 40%**. pt12 VLA on 308 is **8/10 =
80%**. `actor_q` was a ~0.001 advantage (`Q_guided − Q_ref`); `td_loss` and
`q_next` logged as 0 (critic frozen). Successes were essentially frozen VLA
coin-flips; later rounds got worse as `actor_coef` ramped.

This is the V22 analogue of `flow_freeze_critic_online=true` / the external
frozen V21 critic arm (67% mug online vs 98% with live TD).

## OT-compose abort (`cf_v23_ot`, stopped 2026-08-28 14:32)

After the injection port, V23 was aligned to V22's analytic OT actor
`V = (ã − noise) + G` (`ot_compose_integrate`). Frozen pt12 Euler emitted `ã`;
a zero-init MLP `G` was the only trainable actor. `G=0` copied pi0.5. Live TD
+ AE + `β=100`. That is **not** V22 `cf_ae` on this stack: V22's `v_pi05_base`
is a stand-in for a frozen flow expert the PyTorch RLT code could not run;
OpenPI already has that expert.

Stopped during round-6 train (target 17614) to switch the actor to LoRA on
`gemma_300m`. Probe 0/1 timeout. Online **5/6 = 83.3%** (timeout on ep 6),
last-5 **4/5**. Same VLA-floor coin-flip as injection: `guide_rms / x_ref_rms`
stayed ~0.2–0.6%, `actor_ref_rmse` 0.001–0.003, `q_gap` ~0.001. G never
steered. Checkpoints `checkpoints/cf_v23_ot/` are incompatible with
`gemma_300m_lora`; do not resume.

| Ep | Result | Env steps | Chunks | Ckpt |
| ---: | --- | ---: | ---: | ---: |
| probe | fail | 4300 | — | 15999 |
| 1 | success | 933 | 30 | 16148 |
| 2 | success | 1233 | 39 | 16342 |
| 3 | success | 1623 | 51 | 16596 |
| 4 | success | 944 | 30 | 16745 |
| 5 | success | 1223 | 39 | 16939 |
| 6 | fail | 4300 | 135 | killed mid-train |

## Current recipe: V22 cf_ae (action-expert LoRA + live TD + AE)

Implemented against `pipeline_mug_cf_ae_gate0.sh` / `flow_rlt.py`. V22's
compose actor is `V = v_pi05_base + G`; OpenPI *has* the real action expert, so
`v_pi05_base` is frozen `gemma_300m` and `G` is LoRA on that expert (not an
analytic OT field and not a separate MLP). Injection-era and OT-compose
checkpoints (`cf_v23_radio`, `cf_v23_ot`) are incompatible; do not resume.

### Method vs V22 (only task / network leftovers)

| Change | Where | V22 |
| --- | --- | --- |
| Actor = LoRA on the pi0.5 action expert | `action_expert_variant=gemma_300m_lora` | `FlowActor` MLP `G` on analytic OT |
| Frozen expert unroll is **only** ã (`lora_scale=0`) | `_unroll_endpoint` | HTTP VLA chunk stored as `reference` |
| Live expert unroll emits a (`lora_scale=1`, same noise) | `_unroll_endpoint`, `sample_actions` | `_integrate` + `flow_compose=true` |
| OT-path BC: `t∼U(0,1)`, `MSE(v_live, v_frozen)` | `compute_cf_v23_loss` | `F.mse_loss(G + (ã-x0), ã-x0)` |
| `L_π = actor_coef(−Q) + β‖a−ã‖² + bc` (anchor always on) | `compute_cf_v23_loss` | same; `flow_actor_coef` gates only −Q |
| Live TD, target clip `[0,1]`, `a'` from EMA LoRA | `compute_cf_v23_loss`, `cf_target_lora` | `critic_step` + `_integrate(actor_target)` |
| Discount `γ^C`, `γ=0.99`, `C=32` | `cf_gamma ** action_horizon` | `gamma ** chunk` (`C=8`) |
| RL z from **frozen target pool**; live pool is L_ro only | `cf_target_token_pool`, polyak skipped when `freeze_pool` | on-disk encoder writes buffer z |
| Serve publishes target pool + **target LoRA** | `cf_live.py` payload `lora` | `actor_target` (EMA 0.999) + frozen encoder |
| LoRA target EMA 0.999; critic Polyak `τ=0.005` | `train.py` `polyak_cf_target_lora` | `cf_ema=0.999`, critic `tau=0.005` |
| `cf_critic_coef=2` (joint-loss analogue of 2 critic SGD) | config | `critic_updates_per_actor=2` |
| `cf_w_l2_coef=0` | config | no ‖W‖² |
| `actor_coef=1` from round 1 | `run_cf_v23_radio.sh` | `flow_actor_coef=1`, warmup=0 |
| Train steps = `UTD × n_chunks` (`UTD=5`) | `count_chunks` | utd=5 per stored decision |

Keep (do not revert): `_critic_with_sg` so `-Q` cannot inflate the critic
(V22: critic is a separate Adam), frozen VLM + base expert, `beta=100`, kill
switch, velocity BC.

Task/network only: chunk 32×23 vs 8×8, B1K proprio 23-d vs 16-d, `z` 1024 vs
256, CFTrunk 1536 + 10 Q heads vs 2-layer MLP twin-Q, batch 8 vs 256 (PaliGemma
still in the TD graph), 1-ep instance-308 smoke vs 300-ep `episode_pool=0-11`.
Anchor is divided by `32×23` so β=100 stays per-dimension (V22 uses the raw
sum over 64 dims).

Env knobs (defaults = V22 cf_ae): `OPENPI_CF_FREEZE_CRITIC=0`,
`OPENPI_CF_FREEZE_POOL=1`, `OPENPI_CF_UNIT_BALL=0`, `OPENPI_CF_ACTOR_COEF`
(0 pretrain / 1 online), `OPENPI_CF_BC_COEF=1.0`, `OPENPI_CF_ANCHOR_BETA=100`,
`OPENPI_CF_UTD=5`, `OPENPI_CF_W_L2=0`, `OPENPI_CF_CRITIC_COEF=2`,
`OPENPI_CF_GAMMA=0.99`, `OPENPI_CF_ACTOR_EMA=0.999`,
`OPENPI_CF_V23_INSTANCE=308`, `OPENPI_CF_KILL_REF_RMSE=0.10`.

### Code changes (this port)

V22 compose is `V = v_pi05_base + G`. On OpenPI, `v_pi05_base` is the frozen
pt12 action expert and `G` is LoRA on that expert (rank 32, alpha 32,
`gemma_300m_lora`). Analytic OT `G` and `v_θ − G` injection are gone from the
V23 train/serve path.

| Piece | What changed |
| --- | --- |
| Config | `action_expert_variant=gemma_300m_lora` (attn+ffn rank 32 / alpha 32). Weight loader `missing_regex=".*cf_.*|pointnet.*|.*lora.*"` (pt12 is LoRA-free; adapters init near 0). Exp default `cf_v23_ae`. Peak LR `1e-4`, warmup 200, `save_interval=1000`, `keep_period=1000`, `ema_decay=None`, batch 8. `cf_freeze_lora_after_warmup=False`. Config still has `apply_guide=True`; V23 `sample_actions` ignores it. |
| Freeze | `cf_v23_freeze_filter`: train live `cf_*` **and** `.*lora.*`; freeze VLM, base `gemma_300m`, `action_{in,out}_proj`, `time_mlp_*`, and `cf_target_*`. `cf_v23_lora_filter` is live adapters only (excludes `cf_target_lora`). |
| Forward `lora_scale` | Threaded through `lora.Einsum` / `lora.FeedForward` and gemma `Attention` / `Block` / `Module` (`nn.scan` extra `broadcast`). `_euler_v` passes it into `PaliGemma.llm`. `0` zeros the LoRA delta without swapping weights. Prefix LLM calls leave the default (`1`); VLM is `gemma_2b` (no adapters). |
| Train-log `lora_scale` | Separate: `train.py` multiplies LoRA **grads** by this scalar (V17 freeze-after-warmup). V23 keeps it at 1. AE logs `lora_scale=1` even while Euler is off — that is the grad multiplier, not the forward scale. |
| ã | `_unroll_endpoint(..., stop_v=True, lora_scale=0)` — frozen pt12 expert, stop-grad. `guide_fn` is ignored (the expert *is* the actor). |
| a | `_unroll_endpoint(..., stop_v=False, lora_scale=1)` — live LoRA, **same noise** as ã. Grads through the 10-step Euler. AE-only (`cf_ae_only=1`) skips both unrolls. |
| BC | OpenPI time `t∼U(0,1)`, `x_t = t·x0 + (1-t)ã`, `MSE(v_live, sg(v_frozen))`. Logged as `lora_rms` / `guide_rms` (Δv RMS). |
| EMA shadow | `CFLoraShadow` (`cf_target_lora`): copies live LoRA at init into `t0..tN` Params so Orbax checkpoints them. `to_lora_pure_dict` rebuilds the nested adapter tree via `getattr(leaf, "value", leaf)` (JIT leaves may already be arrays). Frozen by `cf_target_*`. |
| `a'(s')` | `_fork_with_lora_pure` splits `self`, overlays `cf_target_lora.to_lora_pure_dict()` onto the flattened State **pure dict**, `replace_by_pure_dict` + `nnx.merge`, then unrolls `lora_scale=1`, stop-grad. Do **not** call `Param.replace` — under `nnx.value_and_grad` leaves are `ShapedArray` / `JVPTracer` (first three AC attempts crashed on that). |
| Serve | `sample_actions` is live-expert Euler (`lora_scale=1`). V23 does **not** subtract G and does **not** OT-compose. `OPENPI_CF_APPLY_GUIDE` / `apply_guide` is ignored when `cf_v23=True`. `cf_live.npz` payload key `lora` is the **target** adapters (`_extract_lora_pure`); `load_cf_live_into_model` writes them onto live LoRA via `CF_LIVE_LORA_FILTER` and re-JITs. Default live path is `checkpoints/cf_v23_ae/cf_live.npz`. |
| Polyak | Critic `τ=0.005` on `CF_V23_TARGET_PAIRS` (`cf_guide_*` still listed there; unused by the expert). Actor EMA via `polyak_cf_target_lora` after that loop (`replace_by_pure_dict` on the shadow, same tracer-safe pattern). `freeze_pool=1` still skips pool polyak. `CF_V23_ACTOR_LIVE_NAMES` still names the unused guide MLP. |
| Guide MLP | `cf_guide_trunk` / `cf_guide_head` stay in the graph (V17 shapes / opt_state) but are unused by the expert actor. |
| OT helper | `ot_compose_integrate` remains in `pi0_cf.py` for tests. V23 loss and `sample_actions` do not call it. |
| Tests | `test_cf_token_ae.py` `test_v23_lora_filter_excludes_shadow`: live expert LoRA matches; `cf_target_lora.t0` does not. |

Do not resume `cf_v23_radio` (injection) or `cf_v23_ot` (OT compose): GraphDef
has no live LoRA / `cf_target_lora`. Three incompatible checkpoint families:
injection (`cf_v23_radio`), OT-compose (`cf_v23_ot`), action-expert (`cf_v23_ae`).

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
  `a'(s')` from the target actor. Discount `γ^C`. **Not frozen online.**
- AE reconstruction finetune; the `z_rl` written to the buffer stays the frozen
  on-disk encoder.
- Learner: utd=5 actor updates + 2 critic updates **per stored decision chunk**.
- Batch 256, critic is a cheap MLP on cached `(z_rl, proprio, a)`.
- 300 episodes, `episode_pool=0–11`, metric = last-10 SR. Mug gate0: online
  96.3%, last-10 100%, held-out 100%. Frozen-critic cousins are worse
  (external frozen V21 critic 67% online).

## V23 mapping (V22 → openpi_comet)

| V22 (MolmoSpaces)                  | V23 (openpi_comet / BEHAVIOR-1K)                        |
|------------------------------------|---------------------------------------------------------|
| Frozen HTTP pi0.5 emits ã          | Frozen `gemma_300m` (`lora_scale=0`) Euler emits ã      |
| `G` MLP + analytic OT emits a      | LoRA on the action expert (`lora_scale=1`) Euler emits a |
| Token AE `z_rl` (encoder frozen in buffer) | CFTokenPool = V22 TokenEncoder. RL uses `sg(target_pool(s))`. Live pool: `L_ro` only |
| proprio concat                     | normalized 23-d state concatenated to pooled features   |
| DoubleCritic twin-Q, TD online     | `cf_critic_heads` ensemble (10 heads), TD online        |
| 100 VLA episodes buffer            | 100 collected `state_action.npz` (pt12 rollouts)        |
| 300 online eps, pool 0–11          | 50 rounds × **1 ep** on instance 308 (smoke)            |
| `actor_target` EMA 0.999           | `cf_target_lora` EMA 0.999, critic τ=0.005              |

## Decision-chunk MDP (the key data adaptation)

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
(1024 + 32 padded). Critic trunk `h = CFTrunk([phi, a_t, t_emb])`. The live
pool is trained by `L_ro` only. The `cf_guide_*` MLP is unused by the expert
actor (kept so the critic graph stays the V17 shape).

Critic (twin-Q TD, V22-style; **live online**):

- `y = r + γ^C * (1 - done) * min_k Q_target_k(s', a'(s'))`, clipped to `[0, 1]`.
- `ã(s')` = frozen expert Euler at `s'`. `a'(s')` = EMA LoRA expert (stop-grad).
- `γ = 0.99` per env step, `C = 32` (`cf_gamma ** action_horizon`).
- Loss: MSE over all 10 heads, weighted `cf_critic_coef=2` (V22's 2 critic
  updates per actor). `OPENPI_CF_FREEZE_CRITIC=1` zeros this term.

Actor (Eq. 5 anchor + raw Q-ascent + velocity BC on the action expert):

- `ã` = 10-step frozen-expert Euler (`lora_scale=0`, stop-grad).
- `a` = 10-step live-expert Euler (`lora_scale=1`), **same noise** as `ã`.
  LoRA=0 ⇒ `a=ã`.
- Actor Q uses **stop-grad critic weights** (`_critic_with_sg`). Ascent is
  **raw** `−mean min_k Q_k(s, a)` — not an advantage of the VLA chunk.
  `q_gap = Q_guided − Q_ref` is logged only. State features are
  `sg(target_pool(s))`.
- BC: sample OpenPI-time `t∼U(0,1)`, `x_t=t·x0+(1-t)ã`,
  `MSE(v_live(x_t,t), sg(v_frozen(x_t,t)))`.
- `L_actor = cf_actor_coef * (−Q) + beta * ||a − ã||² / (32×23)
  + cf_bc_coef * MSE(Δv)`, `beta = 100`. Anchor and BC stay on when
  `actor_coef=0`.
- `L_ro = cf_ae_coef * MSE(decoder(z_live), sg(prefix_tokens))`.
- `L = cf_critic_coef * L_td + L_actor + L_ro`. No ‖W‖² (`cf_w_l2_coef=0`).

Freeze filter (`cf_v23_freeze_filter`): train live `cf_*` **and** action-expert
LoRA; freeze VLM, base expert, and `cf_target_*` (including `cf_target_lora`).
Pool/critic freezes are zeroed Adam updates so resume `opt_state` shape is
unchanged. `freeze_pool=1` stops polyak on the target pool so AE finetune
cannot move RL z.

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
   Paligemma prefix tokens. No Euler unroll, no TD. New encoder (`<rl>` readout)
   + causal decoder; old live-TD / OT-compose checkpoints are incompatible.
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
   - Replay = 100 pretrain + all online npz, stratified 50/50 success/fail
   - Kill: `|actor_q| > 2` or `actor_ref_rmse > 0.10` or non-finite grads

The injection-era live-TD no-AE job (through ep 18, ckpt 13626) and the
OT-compose job (`cf_v23_ot`) are archived. Those checkpoints are incompatible
with the LoRA action-expert actor.

Driver: `outputs/cf_v23_ae_driver.log`. Status:
`outputs/cf_v23_ae_status.log`. Success = `q_score.final >= 1`; timeout =
4300 env steps (~143 s sim). Typical success ~1320–1510 steps (~45 s).

## Results: injection-era live-TD 1-ep, no AE (archived)

Stopped 2026-08-27 18:13 UTC during round 19 collect (ckpt 13626). Actor was
`v = sg(v_θ) − G`, **not** OT compose and **not** the LoRA action expert.
Incompatible with this code. Recipe then: `freeze_critic=0 freeze_pool=1
unit_ball=0 tmax=1.0 actor_coef=1.0 utd=5`. Archived as the V22 **cf_noae**
analogue of that injection port.

| Ep | Result | Env steps | Chunks | Train Δ | Ckpt | Cum SR | Last-10 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | fail | 4300 | 135 | 675 | 8673 | 0/1 = 0% | 0/1 |
| 2 | success | 1321 | 42 | 210 | 8882 | 1/2 = 50% | 1/2 |
| 3 | success | 1470 | 46 | 230 | 9111 | 2/3 = 67% | 2/3 |
| 4 | success | 1349 | 43 | 215 | 9325 | 3/4 = 75% | 3/4 |
| 5 | success | 1325 | 42 | 210 | 9534 | 4/5 = 80% | 4/5 |
| 6 | success | 1445 | 46 | 230 | 9763 | 5/6 = 83% | 5/6 |
| 7 | success | 1321 | 42 | 210 | 9972 | 6/7 = 86% | 6/7 |
| 8 | success | 1461 | 46 | 230 | 10201 | 7/8 = 88% | 7/8 |
| 9 | fail | 4300 | 135 | 675 | 10875 | 7/9 = 78% | 7/9 |
| 10 | success (slow) | 2252 | 71 | 355 | 11229 | 8/10 = 80% | 8/10 |
| 11 | success | 1475 | 47 | 235 | 11463 | 9/11 = 82% | 9/10 |
| 12 | success | 1514 | 48 | 240 | 11702 | 10/12 = 83% | 9/10 |
| 13 | success | 1372 | 43 | 215 | 11916 | 11/13 = 85% | 9/10 |
| 14 | success (slow) | 2006 | 63 | 315 | 12230 | 12/14 = 85.7% | 9/10 |
| 15 | success | 1354 | 43 | 215 | 12444 | 13/15 = 86.7% | 9/10 |
| 16 | fail | 4300 | 135 | 675 | 13118 | 13/16 = 81.2% | 8/10 |
| 17 | success | 1625 | 51 | 255 | 13372 | 14/17 = 82.4% | 8/10 |
| 18 | success | 1618 | 51 | 255 | 13626 | **15/18 = 83.3%** | **8/10** |

Probe (not stored): 0/1 timeout, leftover from the frozen-critic job. Episode 1
is the first G≈0 collect off pretrain (VLA coin-flip on 308). After that: 15/17.
Timeouts: 1, 9, 16. Slow successes (10, 14) still `q_score=1`.

### Comparison

| Arm | Critic | Pool | Actor | Steps / ep | n | Online SR | Last-10 |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| pt12 VLA | — | — | frozen expert | — | 10 | 8/10 = 80% | — |
| V23 frozen critic | frozen | frozen | injection G, unit-ball | 2000 | 11 | 5/11 = 45.5% | 4/10 |
| V23 live TD (no AE) | TD | frozen | injection `v_θ − G` | 5 × n_chunks | 18 | 15/18 = 83.3% | 8/10 |
| V23 OT compose (`cf_v23_ot`) | TD | frozen | analytic OT + G | 5 × n_chunks | 6 | 5/6 = 83.3% | 4/5 |
| **V23 action expert (`cf_v23_ae`)** | **TD** | **frozen** | **LoRA on `gemma_300m`** | **5 × n_chunks** | **running** | — | — |
| V22 cf_noae (mug) | TD | frozen encoder | OT + G MLP | utd=5 / row | 300 | 286/300 = 95.3% | 9/10 |
| V22 cf_ae gate0 (mug) | TD | frozen encoder | OT + G MLP | utd=5 / row | 300 | 289/300 = 96.3% | 10/10 |

Reads:

- Unfreezing TD recovered SR from 45% to the VLA floor (83.3% vs 80%). Last-10
  ended at **8/10**, matching the VLA line. n=18 on one instance is still noisy;
  this is not cf_ae's 300-episode mug curve.
- Failures are timeouts (4300), not crashes. Slow successes still `q_score=1`.
- Driver-log lines at steps `19960+` are leftover from the **wiped**
  frozen-critic job (`td_loss=0`, no `q_gap`). Filter this run to steps
  7000–13626.

### Train diagnostics (last ~8 logs of each burst)

`actor_q` is now raw Q of the guided chunk (~0.26 → 0.36), not a 0.001
advantage. `q_next` tracks `q_data` (TD is live). G is still a small residual
(`guide_rms / x_ref_rms ≈ 0.8%`). Trust region holds (`rmse` 0.003 vs kill 0.10).

| Burst | actor_q | q_gap | rmse | guide_rms | w_norm | q_data | q_next | td_loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pretrain 7000–8000 | 0 | 0 | 0 | 0 | 0 | 0.26 | 0.27 | 0.0025 |
| R1 | 0.27 | 0.0003 | 0.0024 | 0.0034 | 0.018 | 0.27 | 0.28 | 0.0024 |
| R4 | 0.30 | 0.0004 | 0.0028 | 0.0029 | 0.016 | 0.30 | 0.32 | 0.0020 |
| R8 | 0.33 | 0.0004 | 0.0026 | 0.0030 | 0.016 | 0.33 | 0.35 | 0.0020 |
| R9 (fail) | 0.35 | 0.0006 | 0.0028 | 0.0030 | 0.017 | 0.35 | 0.36 | 0.0018 |
| R13 | 0.36 | 0.0008 | 0.0030 | 0.0035 | 0.019 | 0.36 | 0.37 | 0.0023 |
| R14 | 0.36 | 0.0008 | 0.0035 | 0.0034 | 0.018 | 0.36 | 0.38 | 0.0020 |
| R18 | 0.41 | 0.0011 | 0.0040 | 0.0031 | 0.017 | 0.41 | 0.42 | 0.0025 |

Max over the run: `actor_q` 0.46, `rmse` 0.0073, `guide_rms` 0.0059, `w_norm`
0.031. Kill switch never fired.

**Verdict (injection):** critic learned (Q of replay actions rose ~0.26 → 0.41).
The actor never grew a residual — `q_gap` stayed ~0.001, so the executed chunk
was still ≈ the VLA chunk. SR matching VLA + a live critic is the intended V22
*start* without AE (`cf_noae`). It is not evidence that G is steering.

**Verdict (OT compose `cf_v23_ot`):** same story with the V22 *formula* on the
wrong network. `G` stayed a ~1% residual; 5/6 on 308 is the VLA floor at n=6.
That is why the actor is now the pi0.5 action expert.

## Results: `cf_v23_ae` (live)

Started 2026-08-28 14:32 local from pt12 (`--overwrite`). Driver
`outputs/cf_v23_ae_driver.log`, status `outputs/cf_v23_ae_status.log`,
watchdog `~/cf_v23_ae_watchdog.log` (pid 970570). Instance 308, `β=100`,
live TD after AE. No `_KILLED` / `_DONE`.

**AE pretrain** (`ae_only=1`, 8000 steps, freeze_critic=1, freeze_pool=0):
**done** 2026-08-28 15:00 local. `recon_loss` 8.35 → **~0.62**; `z_norm` 32.6
→ **~12.7**; actor/TD terms 0 (phase skips Euler); `cf_grad_finite=1`. Wall
~28 min, ~5.4 it/s. Orbax dirs `1000…7000`; the 7999 save was dropped later
by `keep_period=1000` once `9000` appeared. Resume used GraphDef from 7000
plus train state at 7999.

**AC pretrain** (target 16000, `actor_coef=0`, freeze_pool=1, TD on): first
JIT of the 10-step LoRA unroll + EMA fork. First three attempts crashed
12:01–12:03 UTC on `_fork_with_lora_pure` (`Param.replace` under
`nnx.value_and_grad` → `ShapedArray` / `JVPTracer` has no `replace`). Fixed
by overlaying EMA adapters on the split State pure dict. Driver restarted
15:06 local; skipped AE (already at 7999) and resumed AC.

**Status 2026-08-28 15:24 local:** train.py `--resume` to 16000, step
**~9680 / 16000**, ckpt **9000**, `cf_live.npz` version **9000**. ~1.7 it/s
(~11 s / 20 steps) → ~1 h to 16000. Trust region holds (`actor_ref_rmse`
0.005–0.019 vs kill 0.10). `q_gap≈0` (same-noise copy; `actor_coef=0` so
`actor_loss` is anchor+BC only). `lora_rms / x_ref_rms` ~3–4% velocity
residual (`~0.015 / 0.41`). Critic left the AE-end negative Q (`q_data`
−0.09 → ~0.04). `cf_grad_finite=1`. Next: finish 16000, then probe on 308.

| Step | actor_ref_rmse | lora_rms | bc_loss | anchor | actor_q | q_data | q_next | td_loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8000 | 0.0019 | 0.008 | 0.0001 | 0.0004 | −0.095 | −0.095 | −0.104 | 0.0034 |
| 8060 | 0.0174 | 0.018 | 0.0005 | 0.078 | 0.019 | 0.019 | −0.062 | 0.023 |
| 8500 | 0.0057 | 0.015 | 0.0003 | 0.007 | 0.035 | 0.035 | 0.030 | 0.011 |
| 9000 | 0.0109 | 0.017 | 0.0008 | 0.043 | 0.021 | 0.021 | 0.029 | 0.0028 |
| 9500 | 0.0188 | 0.017 | 0.0003 | 0.088 | 0.034 | 0.034 | 0.039 | 0.0058 |
| 9560 | 0.0087 | 0.014 | 0.0002 | 0.022 | 0.034 | 0.034 | 0.044 | 0.0035 |
| 9680 | 0.0124 | 0.019 | 0.0006 | 0.032 | 0.043 | 0.043 | 0.060 | 0.0053 |

## Not yet ported (intentional leftovers)

- Cache pooled `z` in replay so critic TD does not re-run PaliGemma. Batch is
  still 8. The **encoder snapshot** used for RL now matches V22 (frozen target
  pool); only the VLM forward is still live.
- Store the collect-time VLA chunk as `reference` (V22 replay field). Train
  still re-unrolls frozen pi0.5 for ã, so Euler noise can differ from collect.
- Multi-instance `episode_pool` / last-10 as the official metric. 1 training
  episode on 308 is a smoke test; do not call it a cf_ae result.
- Two separate critic SGD steps on fresh batches. `cf_critic_coef=2` is the
  joint-loss analogue; V22's learner samples twice.
- `ema_decay` on the full TrainState (`None`). Serve already uses the target
  LoRA (`cf_target_lora`) and target pool.
- Train LoRA on `action_in_proj` / `action_out_proj` / `time_mlp_*`. Those stay
  frozen so `lora_scale=0` is an exact pt12 copy. Plasticity is the expert
  transformer LoRA only.
- Delete `ot_compose_integrate` / `cf_guide_*`. The helper and MLP remain so
  tests and V17 GraphDef shapes stay intact; the V23 actor path ignores both.

## Files

- `openpi_comet/src/openpi/training/cf_ae_replay.py`  — decision-chunk replay
- `openpi_comet/src/openpi/training/cf_live.py`       — atomic CF live publish (target LoRA)
- `openpi_comet/src/openpi/models/pi0_cf.py`          — V23 LoRA expert loss + freeze + EMA shadow
- `openpi_comet/src/openpi/models/lora.py`            — `lora_scale` on Einsum / FeedForward
- `openpi_comet/src/openpi/models/gemma.py`           — thread `lora_scale` through the expert
- `openpi_comet/src/openpi/policies/policy.py`        — CF live poll + re-JIT `sample_actions`
- `openpi_comet/src/openpi/shared/eval_b1k_wrapper.py`— clear action queue on CF reload
- `openpi_comet/src/openpi/training/config.py`        — `pi05_b1k-turning_on_radio_cf_v23` (`gemma_300m_lora`)
- `openpi_comet/scripts/train.py`                     — rebind + freeze-pool/critic zeroing + LoRA polyak + kill
- `openpi_comet/scripts/serve_b1k.py`                 — serve with `OPENPI_CF_LIVE_PATH`
- `openpi_comet/scripts/run_cf_v23_radio.sh`          — 1-ep live-TD orchestrator (`cf_v23_ae`)
- `openpi_comet/scripts/cf_v23_watchdog.sh`           — host watchdog (`OPENPI_CF_V23_INSTANCE=308`)
- `openpi_comet/scripts/test_cf_token_ae.py`          — AE + OT helper + LoRA-filter tests

## Watchdog

Action-expert run (`OPENPI_CF_V23_EXP=cf_v23_ae`): heartbeats
`outputs/cf_v23_ae_status.log`, stdout `outputs/cf_v23_ae_driver.log`.
The host-side watchdog (`cf_v23_watchdog.sh`) checks every 20 min: driver
alive (restarts via `docker exec` if dead), and progress advancing. If nothing
progresses for 35 min it kills stuck phase processes; if still stuck at 80 min
it restarts the driver. Completed rounds are skipped via `round_N/.done`.
`outputs/cf_v23_ae_DONE` ends the watchdog. `outputs/cf_v23_ae_KILLED` also
ends it (do not restart).
