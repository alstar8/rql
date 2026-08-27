# V23: cf_ae adapted to BEHAVIOR-1K `turning_on_radio` (openpi_comet / pi0.5)

V23 ports the proven V22 `cf_ae` recipe (MolmoSpaces `desk_mug`, PyTorch) to the
BEHAVIOR-1K `turning_on_radio` task in the `openpi_comet` JAX stack. The base VLA is
the Compet-pt12 pi0.5 specialist (`pi05-b1kpt12-cs32`, action horizon 32, LoRA-free
full checkpoint). The ConsensusFlow machinery lives in
`openpi_comet/src/openpi/models/pi0_cf.py` (V17 lineage); V23 replaces the V17 loss
with the V22 `cf_ae` objective while keeping the V17 architecture (CFTokenPool,
CFTrunk, ensemble critic heads, zero-init guide head).

**Current default (2026-08-27 afternoon):** 1 episode per round on public_test
instance 308, live TD critic, frozen token pool, unbounded G, `tmax=1`,
`actor_coef=1` from episode 0, train steps = `utd × n_chunks` (`utd=5`).
Orchestrator: `openpi_comet/scripts/run_cf_v23_radio.sh`. Through online episode
14 (round 15 collecting): **12/14 = 85.7%**, last-10 **9/10**, vs pt12 VLA
**8/10 = 80%** on the same instance. The frozen-critic cousin of this protocol
was **5/11 = 45%** and was killed.

Code: `openpi_comet/` (`models/pi0_cf.py` `compute_cf_v23_loss`,
`training/cf_ae_replay.py`, `training/cf_live.py`, `scripts/train.py`,
`scripts/run_cf_v23_radio.sh`). Checkpoints:
`openpi_comet/checkpoints/cf_v23_radio/`. Online rollouts:
`openpi_comet/outputs/cf_v23_radio_online/round_N/`.

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

## Current recipe (2026-08-27 afternoon): live TD + UTD

Implemented against the V22 winning arm (`pipeline_mug_cf_ae_gate0.sh` /
`flow_rlt.py`), then the 1-ep protocol was relaunched from pretrain step 7000
(`from-pretrain`; probe metrics already on disk → skipped).

### Code changes vs the frozen-critic abort

| Change | Where | V22 analogue |
| --- | --- | --- |
| `OPENPI_CF_FREEZE_CRITIC=0` online; TD stays live | `run_cf_v23_radio.sh`, `pi0_cf.py`, `train.py` | `flow_freeze_critic_online=false` |
| `OPENPI_CF_FREEZE_POOL=1` zeros Adam on `cf_token_pool` only | `is_v23_pool_path`, `train.py` | z_rl encoder written to the buffer stays frozen |
| Pretrain: both freezes off (pool+critic train, `actor_coef=0`) | `phase_pretrain` | AC pretrain on 100 VLA trajs |
| Actor is raw `-Q(s, x_guided)`, not `Q_guided − Q_ref` | `compute_cf_v23_loss` | `L_π = −Q(s, a) + β‖a−ã‖²`. `q_ref` / `q_gap` still logged |
| `OPENPI_CF_UNIT_BALL=0`; G is a raw MLP, scale=1 | `_v23_guide_with` | unbounded compose G |
| `OPENPI_CF_GUIDE_TMAX=1.0` (G on all 10 Euler steps) | `_v23_guide_with` | no late-Euler gate |
| `actor_coef=1` from round 1; ramp deleted (`ACTOR_RAMP_ROUNDS=0`) | `run_cf_v23_radio.sh` | `flow_actor_coef=1`, warmup=0 |
| Train steps = `UTD × n_chunks` (`UTD=5`, min 50), not 2000 | `count_chunks`, `collect_and_train_round` | utd=5 actor updates per stored decision |
| Skip completed rounds via `round_N/.done`, not ckpt arithmetic | `phase_online` | — |
| Serve publishes **Polyak target** guide (`cf_target_guide_*`) under live names | `cf_live.py` `CF_LIVE_SOURCE_NAMES` | eval uses `actor_target` (EMA 0.999) |
| Train step rebinds freeze/tmax/unit_ball/actor_coef from the process config | `train.py` | Orbax GraphDef is the pretrain one |

Keep (do not revert): dedicated `cf_guide_trunk`, `_critic_with_sg` so `-Q`
cannot inflate the critic, frozen VLA, `beta=100`, kill switch, velocity BC.

Env knobs (defaults = live-TD 1-ep): `OPENPI_CF_FREEZE_CRITIC=0`,
`OPENPI_CF_FREEZE_POOL=1`, `OPENPI_CF_UNIT_BALL=0`, `OPENPI_CF_GUIDE_TMAX=1.0`,
`OPENPI_CF_ACTOR_COEF` (0 pretrain / 1 online), `OPENPI_CF_BC_COEF=1.0`,
`OPENPI_CF_ANCHOR_BETA=100`, `OPENPI_CF_UTD=5`, `OPENPI_CF_V23_INSTANCE=308`,
`OPENPI_CF_KILL_REF_RMSE=0.10`.

## What V22 cf_ae is (reference)

- Frozen VLA flow `v_theta` (Euler, 10 steps) produces the reference chunk `x_ref`.
- Actor = CF composition `V = v_base + G` (`flow_compose=true`); G is an
  unbounded MLP. V23 injects `v = sg(v_θ) − G` into the real pi0.5 Euler
  instead of emitting the chunk from an OT flow — a B1K choice, not a V22 clone.
- Actor loss: `E[ −Q(s, a) + beta * ||a − ã||² ]` with `beta=100`. Live Q is
  not clipped; only the TD *target* is.
- Critic: twin Q, TD every utd step, target clipped `[0, 1]`, bootstrap
  `a'(s')` from the target actor. **Not frozen online.**
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
| Flow VLA `v_theta` (frozen)        | pi0.5 pt12 action expert (**frozen**), Euler 10         |
| Token AE `z_rl` (encoder frozen in buffer) | CFTokenPool over VLM prefix tokens (**frozen online**) |
| proprio concat                     | normalized 23-d state concatenated to pooled features   |
| `W(x,z,t)` unbounded guide MLP     | dedicated `cf_guide_trunk` + zero-init `cf_guide_head`  |
| DoubleCritic twin-Q, TD online     | `cf_critic_heads` ensemble (10 heads), TD online        |
| 100 VLA episodes buffer            | 100 collected `state_action.npz` (pt12 rollouts)        |
| 300 online eps, pool 0–11          | 50 rounds × **1 ep** on instance 308 (smoke)            |
| `actor_live.pt` / `actor_target`   | `checkpoints/cf_v23_radio/cf_live.npz` (Polyak guide)   |

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

State features: `phi(s) = [ CFTokenPool(prefix_tokens), normalize(state_23) ]`
(1024 + 32 padded). Critic trunk `h = CFTrunk([phi, a_t, t_emb])`. Guide uses
a **separate** `cf_guide_trunk`.

Critic (twin-Q TD, V22-style; **live online**):

- `y = r + gamma_chunk * (1 - done) * min_k Q_target_k(s', a'(s'))`, clipped to `[0, 1]`.
- `a'(s')` = endpoint of a 10-step Euler unroll at `s'` with the *target* guide
  (stop-grad everywhere).
- `gamma_chunk = 0.99` per 32-step chunk (not `gamma^C`; kept so pretrained Q
  scale matches).
- Loss: MSE over all 10 heads. `OPENPI_CF_FREEZE_CRITIC=1` zeros this term and
  the Adam update on pool+trunk+heads (old abort). Live-TD leaves it on.

Actor (Eq. 5 anchor + raw Q-ascent + velocity BC):

- `x_ref` = 10-step Euler unroll with current `v_theta` only (stop-grad).
- `x_guided` = 10-step unroll with `v = sg(v_theta) − G`. G is a raw MLP
  (`cf_unit_ball=0`), gated by `tmax` (1.0 = every Euler step).
- Actor Q uses **stop-grad critic weights** (`_critic_with_sg`). Ascent is
  **raw** `−mean min_k Q_k(s, x_guided)` — not an advantage of the VLA chunk.
  `q_gap = Q_guided − Q_ref` is logged only.
- `L_actor = cf_actor_coef * (−Q) + beta * ||x_guided − x_ref||² / (32×23)
  + cf_bc_coef * MSE(G)`, `beta = 100`. Anchor is still **normalized by chunk
  dims** (V22 sums over dims); leave as-is until rmse says otherwise.
- Online: pool Adam is zeroed (`OPENPI_CF_FREEZE_POOL=1`); critic trunk+heads
  and guide trunk+head train. `L = L_td + L_actor + cf_w_l2 * ||W||^2`.

Freeze filter (`cf_v23_freeze_filter`): train live `cf_*` only; freeze whole
VLA + `cf_target_*`. Pool/critic freezes are zeroed Adam updates so resume
`opt_state` shape is unchanged.

Target networks: explicit `cf_target_*` (pool / critic trunk / guide trunk /
critic / guide), polyak `tau=0.005` after each optimizer step. Serve loads the
**target** guide under the live names.

`cf_q_clip_max` is still on the config and rebound each step; the actor term
no longer clips live Q (V22). TD targets stay clipped `[0, 1]`.

## Training protocol (current 1-ep job)

1. **Data**: 100 `state_action.npz` from the pt12 collector
   (`outputs/pt12_cs32_radio_10x10_states`), public_test instances 0–9.
2. **AC pretrain**: 8000 steps, batch 8, `cf_actor_coef=0`, both freezes off —
   critic TD + pool on the 100 pt12 trajs. LR 1e-4 on live CF modules. This
   restart resumed from step 7000 (keep_period checkpoints 1000–7000).
3. **Probe**: 1 episode on instance 308, guide on, **not** written to replay.
   Existing 0/1 timeout was skipped (`phase=probe already done`).
4. **Online**: 50 rounds × {collect 1 ep on 308 + train `5 × n_chunks` steps}:
   - `actor_coef=1`, `freeze_critic=0`, `freeze_pool=1`, `unit_ball=0`, `tmax=1`
   - One serve + one client (`OPENPI_CF_V23_INSTANCE=308` → eval id 7)
   - Publish `cf_live.npz` (pool / critic trunk / **target** guide) after each burst
   - Replay = 100 pretrain + all online npz, stratified 50/50 success/fail
   - Skip `round_N/.done`. Keep all 1-ep npz (disk is small).
   - Kill: `|actor_q| > 2` or `actor_ref_rmse > 0.10` or non-finite grads

Set `OPENPI_CF_V23_INSTANCE=""` for the old 10-instance / 2-worker / 5-slice
schedule. Do not resume wiped frozen-critic ckpts; this run's online ckpts are
8000, 9000, 10000, 11000, 12230, plus keep_period 1000–7000.

Driver: `outputs/cf_v23_radio_driver.log`. Status:
`outputs/cf_v23_radio_status.log`. Success = `q_score.final >= 1`; timeout =
4300 env steps (~143 s sim). Typical success ~1320–1510 steps (~45 s).

## Results: live-TD 1-ep (in progress, through episode 14)

Snapshot 2026-08-27 17:21 UTC. Round 14 done (ckpt 12230); round 15 collecting.
No `KILL` / `DONE` marker. Recipe in the status line:
`freeze_critic=0 freeze_pool=1 unit_ball=0 tmax=1.0 actor_coef=1.0 utd=5`.

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
| 14 | success (slow) | 2006 | 63 | 315 | 12230 | **12/14 = 85.7%** | **9/10** |

Probe (not stored): 0/1 timeout, leftover from the frozen-critic job. Episode 1
is the first G≈0 collect off pretrain (VLA coin-flip on 308). After that: 12/13.

### Comparison

| Arm | Critic | Pool | G | Steps / ep | n | Online SR | Last-10 |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| pt12 VLA (no G) | — | — | 0 | — | 10 | 8/10 = 80% | — |
| V23 frozen critic | frozen | frozen | unit-ball, tmax=0.4 | 2000 | 11 | 5/11 = 45.5% | 4/10 |
| **V23 live TD (this run)** | **TD** | **frozen** | **unbounded, tmax=1** | **5 × n_chunks** | **14** | **12/14 = 85.7%** | **9/10** |
| V22 cf_ae gate0 (mug) | TD | frozen encoder | unbounded compose | utd=5 / row | 300 | 289/300 = 96.3% | 10/10 |

Reads:

- Unfreezing TD recovered SR from 45% to the VLA floor (85.7% vs 80%). Last-10
  9/10 is the V22 metric and is at or above the 8/10 baseline. n=14 on one
  instance is still noisy; this is not cf_ae's 300-episode mug curve.
- Failures are timeouts (4300), not crashes. Slow successes (ep 10, 14) still
  `q_score=1`.
- Driver-log lines at steps `19960+` are leftover from the **wiped**
  frozen-critic job (`td_loss=0`, no `q_gap`). Filter this run to steps
  7000–12230.

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

Max over the run: `actor_q` 0.46, `rmse` 0.0073, `guide_rms` 0.0059, `w_norm`
0.031. Kill switch never fired.

**Verdict so far:** critic is learning (Q of replay actions rose ~0.26 → 0.36).
The actor has not grown a large residual — `q_gap` stays ~0.0008, so the
executed chunk is still ≈ the VLA chunk. SR matching VLA + a live critic is
the intended V22 *start*; it is not yet evidence that G is steering. Keep
reading last-10 vs the 80% line and whether `q_gap` / `w_norm` grow without
`rmse` hitting 0.10.

## Not yet ported (intentional)

- Cache pooled `z` in replay so critic TD does not re-run PaliGemma. Batch is
  still 8; this is why V22 TD is cheap enough to leave on at batch 256.
- AE reconstruction on the pool (V22 P1). Do not send `-Q` through the pool.
- `gamma^C` vs `cf_chunk_gamma=0.99` per 32-step chunk.
- Switch V23's `v = v_θ − G` injection back to V22's OT flow that *emits* the
  chunk. Leave unless injection stays impotent after G actually grows.
- Multi-instance `episode_pool` / last-10 as the official metric. 1 training
  episode on 308 is a smoke test; do not call it a cf_ae result.
- `ema_decay` on the full TrainState (`None`). Serve already uses the Polyak
  target guide.

## Files

- `openpi_comet/src/openpi/training/cf_ae_replay.py`  — decision-chunk replay
- `openpi_comet/src/openpi/training/cf_live.py`       — atomic CF live publish (target guide)
- `openpi_comet/src/openpi/models/pi0_cf.py`          — V23 loss + freeze path helpers + guide deploy
- `openpi_comet/src/openpi/policies/policy.py`        — CF live poll + re-JIT `sample_actions`
- `openpi_comet/src/openpi/shared/eval_b1k_wrapper.py`— clear action queue on CF reload
- `openpi_comet/src/openpi/training/config.py`        — `pi05_b1k-turning_on_radio_cf_v23`
- `openpi_comet/scripts/train.py`                     — rebind + freeze-pool/critic zeroing + polyak + kill
- `openpi_comet/scripts/serve_b1k.py`                 — serve with `OPENPI_CF_LIVE_PATH`
- `openpi_comet/scripts/run_cf_v23_radio.sh`          — 1-ep live-TD orchestrator
- `openpi_comet/scripts/cf_v23_watchdog.sh`           — host watchdog (`OPENPI_CF_V23_INSTANCE=308`)

## Watchdog

The orchestrator (`run_cf_v23_radio.sh`, inside `b1k-airi-dev`) heartbeats to
`outputs/cf_v23_radio_status.log`; stdout goes to
`outputs/cf_v23_radio_driver.log`. The host-side watchdog
(`cf_v23_watchdog.sh`) checks every 20 min: driver alive (restarts via
`docker exec` if dead), and progress advancing. If nothing progresses for 35 min
it kills stuck phase processes; if still stuck at 80 min it restarts the
driver. Completed rounds are skipped via `round_N/.done`.
`outputs/cf_v23_radio_DONE` ends the watchdog. `outputs/cf_v23_radio_KILLED`
also ends it (do not restart).
