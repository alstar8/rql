# V23: LoRA action-expert RL on `turning_on_radio` (openpi_comet / pi0.5)

Paper method (shared with Pick): [`V22_V23_METHODS.md`](V22_V23_METHODS.md).
This file is the radio instantiation and the `cf_v23_ae2` log.

V23 trains a residual on a frozen pi0.5 specialist (`pi05-b1kpt12-cs32`,
action horizon 32) in the `openpi_comet` JAX stack. The VLM (`gemma_2b`) and
the base action expert (`gemma_300m`) stay frozen. The trainable actor is
LoRA on `gemma_300m`. Failed actor variants and killed jobs are in
[`V23_attempts.md`](V23_attempts.md).

Code: `openpi_comet/` (`models/pi0_cf.py` `compute_cf_v23_loss`,
`training/cf_ae_replay.py`, `training/cf_live.py`, `scripts/train.py`,
`scripts/run_cf_v23_radio.sh`). Current experiment: `cf_v23_ae2`.
Checkpoints: `openpi_comet/checkpoints/cf_v23_ae2/`. Online rollouts:
`openpi_comet/outputs/cf_v23_ae2_online/round_N/`. Driver:
`outputs/cf_v23_ae2_driver.log`. Status: `outputs/cf_v23_ae2_status.log`.

## Method

### Actor

LoRA on the frozen action expert is the policy. Rank 32 / alpha 32 on attn
and ffn, **B = 0** at init (`gemma_300m_lora`). A forward `lora_scale`
threads through `lora.Einsum` / `lora.FeedForward` and gemma `Attention` /
`Block` / `Module` (`nn.scan` extra `broadcast`). Scale 0 zeros the LoRA
delta without swapping weights.

| Piece | Role |
| --- | --- |
| Frozen `gemma_300m` (`lora_scale=0`) | Expert chunk `ã`. Never trained. |
| Live LoRA (`lora_scale=1`) | Executed chunk `a`. Same Euler noise as `ã`. `Δ=0` copies the specialist exactly. |
| `cf_token_pool` | RL state `z` for **Q**, not the actor residual. |
| CFTrunk / 10 critic heads | `Q(s, flatten(a_chunk))` on `[z, a]`. |
| `cf_guide_trunk` / `cf_guide_head` | Present in the GraphDef / `opt_state`. **Unused** by `compute_cf_v23_loss` and `sample_actions`. |

Euler is 10 steps from OpenPI time `t=1` (noise) to `t=0` (data):

```
ã = Euler(v_frozen, noise)     # stop-grad
a = Euler(v_live,   noise)     # grads through the 10 expert steps
```

`action_in_proj`, `action_out_proj`, and `time_mlp_*` stay frozen so
`lora_scale=0` is an exact specialist copy. Plasticity is transformer LoRA
only.

### Freeze filter

`cf_v23_freeze_filter`: train live `cf_*` **and** `.*lora.*`; freeze VLM,
base `gemma_300m`, `action_{in,out}_proj`, `time_mlp_*`, and `cf_target_*`
(including the LoRA EMA shadow). Pool / critic freezes are **zeroed Adam
updates** so resume `opt_state` shape is unchanged.

### Critic

`Q(s, flatten(a))` on CFTrunk (`cf_hidden=1536`), 10 heads, `min` over
heads. Not per-step `(a_t, t)`.

State: `phi(s) = [ sg(target_pool(prefix_tokens)), normalize(proprio) ]`.
Pooled tokens are 1024-d; proprio is the 23-d BEHAVIOR-1K state padded to
`action_dim=32`. Chunk is `32×23=736` dims, so the trunk input is
`1024+32+736`.

Live TD, target clipped `[0, 1]`. Bootstrap `a'(s')` from the EMA LoRA
actor (`cf_target_lora`, ema 0.999). Critic Polyak `τ=0.005` on
`CF_V23_TARGET_PAIRS`. `freeze_pool=1` skips pool polyak so AE finetune
cannot move RL `z`.

TD sees the replay chunk plus Gaussian noise (`cf_td_action_noise=0.08`)
so Q cannot ignore the action. `dq_da_rms` is RMS of
`∂ min_k Q_k(s,a) / ∂a` with stop-grad critic weights. `q_success` /
`q_fail` are mean Q of batch rows with episode-success vs fail labels.

`OPENPI_CF_FREEZE_CRITIC=1` zeros the TD term.

Actor Q uses **stop-grad critic weights** (`_critic_with_sg`) so `−Q`
cannot inflate the critic. Live Q is not clipped; only the TD target is.
`cf_q_clip_max` remains on the config and is unused by the actor term.

### Token AE

`L_ro = MSE(decoder(z_live), sg(prefix_tokens))` on Paligemma prefix
tokens. Encoder: `CFTokenPool` (`d_model=256`, 4 heads, 2 layers, 8
queries). Decoder: `CFTokenDecoder`, same width. RL always reads
`sg(target_pool(s))`. Live pool trains `L_ro` only.

### Loss (`compute_cf_v23_loss`)

Joint loss, batch 8. Paligemma prefix is still in the TD graph.

```
γ_chunk = cf_gamma = 0.99          # per decision chunk (not γ^32)
y       = clip(r + γ_chunk (1−done) min_k Q_target_k(s', a'(s')), 0, 1)
L_td    = MSE(Q_heads(s, a + ε), y)

t ∼ U(0,1)
x_t     = t · x0 + (1−t) · ã
L_bc    = MSE(v_live(x_t, t), sg(v_frozen(x_t, t)))

L_π     = actor_coef · (−mean min_k Q_k(s, a))
        + β · mean‖a − ã‖² / (32×23)
        + L_bc

L_ro    = cf_ae_coef · MSE(decoder(z_live), sg(prefix_tokens))
L       = cf_critic_coef · L_td + L_π + L_ro
```

- `actor_coef` gates only `−Q`. Anchor (`β=100`) and BC stay on in every
  phase.
- `cf_anchor_normalize=1` divides the sum by `32×23` so `β=100` is
  per-dimension.
- `cf_w_l2_coef=0`.
- `cf_critic_coef=4`.
- LoRA Adam updates are scaled by `OPENPI_CF_LORA_LR_MULT=1`. The train-log
  field `lora_scale` is this **grad multiplier**, not the Euler forward
  scale. AE logs `lora_scale=1` even while Euler is off.
- `cf_ae_only=1` skips both unrolls and TD; trains `L_ro` only.

Logged: `actor_ref_rmse`, `lora_rms` / `guide_rms` (Δv RMS), `x_ref_rms`,
`q_gap = Q(a)−Q(ã)`, `dq_da_rms`, `q_success`, `q_fail`, `chunk_discount`,
`lora_lr_mult`, `lora_grad_norm`.

### Decision-chunk MDP

The env steps at 30 Hz but the policy commits to a 32-step action chunk
(re-inference only when the queue is empty; queue cleared at episode
reset, so chunk boundaries align to a 32-step grid from `t=0`). Each
32-step chunk is one MDP decision:

- `s`    = observation at chunk start (3 images + 256-d proprio → 23-d
  state + prompt tokens)
- `a`    = the 32×23 action chunk executed
- `r`    = 1 if the episode terminated with success inside the chunk else 0
- `done` = 1 if the episode ended (terminated or truncated) inside the chunk
- `s'`   = observation at the start of the next chunk (or terminal obs)

`state_action.npz` stores per step: `state` (256-d), `action` (N,1,23),
`next_state`, `next_action`, `reward`, `done`, `truncated`,
`actor_obs__0..3` (proprio + head/left/right images), `next_actor_obs__*`,
`metadata` (`success`, `prompt`). `cf_ae_replay.py` chunks each episode on
the 32-step grid. Short last chunks pad by repeating the final action.
`n_chunks = ceil(T / 32)` (timeout 4300 → 135 chunks → 675 train steps at
utd=5).

Replay is stratified 50/50 over successful / failed **episodes**. Collect
instance is `OPENPI_CF_V23_INSTANCE` (serve/client only). Train filter is
`OPENPI_CF_REPLAY_INSTANCE` (empty = full 100-traj buffer). Chunks are
extracted once per npz into `<root>/_cf_ae_chunks/` (memory-mapped,
incremental).

### EMA, serve, optimizer

- `CFLoraShadow` (`cf_target_lora`): copies live LoRA at init into `t0..tN`
  Params so Orbax checkpoints them. `to_lora_pure_dict` rebuilds the nested
  adapter tree via `getattr(leaf, "value", leaf)` (JIT leaves may already
  be arrays). Frozen by `cf_target_*`.
- `a'(s')`: `_fork_with_lora_pure` splits `self`, overlays
  `cf_target_lora.to_lora_pure_dict()` onto the flattened State **pure
  dict**, `replace_by_pure_dict` + `nnx.merge`, then unrolls
  `lora_scale=1`, stop-grad. Do **not** call `Param.replace` — under
  `nnx.value_and_grad` leaves are `ShapedArray` / `JVPTracer`.
- Serve: `sample_actions` is live-expert Euler (`lora_scale=1`).
  `OPENPI_CF_APPLY_GUIDE` / `apply_guide` is ignored when `cf_v23=True`.
  `cf_live.npz` payload key `lora` is the **target** adapters
  (`_extract_lora_pure`); `load_cf_live_into_model` writes them onto live
  LoRA via `CF_LIVE_LORA_FILTER` and re-JITs. Also publishes target pool
  under the live name. Default path `checkpoints/${EXP}/cf_live.npz`.
- Polyak: critic `τ=0.005` on `CF_V23_TARGET_PAIRS`, then
  `polyak_cf_target_lora` at ema 0.999. `ema_decay` on the full TrainState
  is `None`.
- LR: `WarmupConstantSchedule` peak `1e-4`, warmup 200, then constant.
  Cosine-to-zero is SFT-only.
- Weight loader: pt12 params,
  `missing_regex=".*cf_.*|pointnet.*|.*lora.*"`.
- Kill: `|actor_q| > 2` or `actor_ref_rmse > 0.04` or non-finite CF grads.
  Writes `outputs/${EXP}_KILLED`, train exits 2.

`ot_compose_integrate` remains in `pi0_cf.py` for tests. The V23 loss and
`sample_actions` do not call it.

### Training protocol (AE + live-TD, 1-ep)

1. **Data**: 100 `state_action.npz` from the pt12 collector
   (`outputs/pt12_cs32_radio_10x10_states`), public_test instances 0–9.
2. **AE pretrain**: 8000 steps, `cf_ae_only=1`, `cf_ae_coef=1`,
   `freeze_critic=1`, `freeze_pool=0`, `actor_coef=0`. Encoder+decoder
   train `L_ro` on stop-grad Paligemma prefix tokens. No Euler, no TD.
3. **AC pretrain**: 8000 more steps (target 16000), `actor_coef=0`,
   `freeze_pool=1`, `freeze_critic=0`, `ae_coef=0`. Critic TD on
   `sg(target_pool(s))`. LoRA BC + anchor keep the expert a copy of pt12.
4. **Probe**: 1 episode on instance 308, live expert, **not** written to
   replay.
5. **Online**: 50 rounds × {collect 1 ep on 308 + train `5 × n_chunks`
   steps}:
   - `actor_coef=1`, `freeze_critic=0`, `freeze_pool=1`, `ae_coef=1`
   - One serve + one client (`OPENPI_CF_V23_INSTANCE=308` → eval id 7)
   - Publish `cf_live.npz` (frozen encoder / critic trunk / **target LoRA**)
   - Replay = 100 pretrain (all instances) + all online npz, stratified
     50/50 success/fail
   - `OPENPI_CF_V23_EXP=cf_v23_ae2` (script default `cf_v23_ae` is a
     different checkpoint family; see attempts)

Success = `q_score.final >= 1`; timeout = 4300 env steps (~143 s sim).
Typical specialist success on 308 is ~800–1343 steps (~30–45 s). Slow
successes (2000–3200) still count as `q_score=1`.

### Defaults (`cf_v23_ae2`)

`OPENPI_CF_FREEZE_CRITIC=0`, `OPENPI_CF_FREEZE_POOL=1`,
`OPENPI_CF_UNIT_BALL=0`, `OPENPI_CF_ACTOR_COEF` (0 pretrain / 1 online),
`OPENPI_CF_BC_COEF=1.0`, `OPENPI_CF_ANCHOR_BETA=100`,
`OPENPI_CF_ANCHOR_NORMALIZE=1`, `OPENPI_CF_UTD=5`, `OPENPI_CF_W_L2=0`,
`OPENPI_CF_CRITIC_COEF=4`, `OPENPI_CF_GAMMA=0.99`,
`OPENPI_CF_DISCOUNT_PER_CHUNK=1`, `OPENPI_CF_ACTOR_EMA=0.999`,
`OPENPI_CF_V23_INSTANCE=308`, `OPENPI_CF_REPLAY_INSTANCE=` (empty = all
trajs), `OPENPI_CF_KILL_REF_RMSE=0.04`, `OPENPI_CF_CHUNK_CRITIC=1`,
`OPENPI_CF_TD_ACTION_NOISE=0.08`, `OPENPI_CF_LORA_LR_MULT=1`,
`OPENPI_CF_AE_STEPS=8000`, `OPENPI_CF_AC_STEPS=8000`.

Tests: `test_v23_lora_filter_excludes_shadow`,
`test_chunk_critic_depends_on_action`,
`test_warmup_constant_lr_does_not_hit_zero`,
`test_per_chunk_discount_keeps_credit_over_radio_horizon`.

### Current-code leftovers (intentional)

- Paligemma is re-run for every TD backup. Batch is 8. Pooled `z` is not
  cached in replay.
- Train re-unrolls the frozen expert for `ã`; the collect-time chunk is
  not stored as `reference`. Euler noise can differ from collect.
- Official metric here is 1 training episode on instance 308 (smoke).
  There is no multi-instance `episode_pool` / held-out eval in this
  recipe.
- Two separate critic SGD steps on fresh batches are not implemented;
  `cf_critic_coef=4` is the joint-loss analogue.
- `cf_guide_*` and `ot_compose_integrate` stay so GraphDef / tests keep
  their shapes; the actor path ignores both.

## Experiment: `cf_v23_ae2`

Started 2026-08-31 13:08 UTC from pt12 (`OPENPI_CF_V23_EXP=cf_v23_ae2`,
instance 308, `REPLAY_INSTANCE` empty). Finished
**2026-09-01 02:10 UTC**, step **33019**, 50/50 rounds. No `_KILLED`.
Kill `rmse=0.04` never fired (online `actor_ref_rmse` peak ~0.032).

Heartbeat at start: `lora_lr_mult=1 discount_per_chunk=1 critic_coef=4
td_noise=0.08 kill_rmse=0.04 replay_instance=all`. AE start replay:
**9650 chunks (2360 success / 7290 fail)** from
`outputs/pt12_cs32_radio_10x10_states` + `outputs/cf_v23_ae2_online`.

Frozen specialist on 308: **8/10 = 80%**.

| Phase | When (UTC) | Result |
| --- | --- | --- |
| AE 8000 | 13:08–13:36 | done (`ae_only=1`) |
| AC → 16000 | 13:36–14:55 | done. `chunk_discount=0.99`, `lora_lr_mult=1`. End: `rmse=0.014`, LoRA residual **3.4%**, `q_gap≈0`, `q_success=0.34` / `q_fail=0.16` |
| Probe | 14:55–15:00 | **1/1**, 1324 steps (ckpt 15999) |
| Online | 15:00–02:10 | **37/50 = 74%**, last-10 **10/10**, first-10 6/10. Final step 33019 |

Timeouts (4300 steps): rounds 5, 7, 8, 10, 11, 12, 17, 22, 23, 24, 25,
28, 36 (13). Last-10 recovered to 90% by episode 35 and 100% from
episode 46. Successful episodes: min 807, median 1174, mean 1416 steps.
Late successes are often 807–1198 steps.

### Online rounds

| Round | Result | Env steps | Ckpt after train |
| ---: | --- | ---: | ---: |
| probe | success | 1324 | 15999 |
| 1 | success | 1333 | 16208 |
| 2 | success | 1368 | 16422 |
| 3 | success | 1460 | 16651 |
| 4 | success | 2276 | 17010 |
| 5 | fail | 4300 | 17684 |
| 6 | success | 1843 | 17973 |
| 7 | fail | 4300 | 18647 |
| 8 | fail | 4300 | 19321 |
| 9 | success | 2927 | 19780 |
| 10 | fail | 4300 | 20454 |
| 11 | fail | 4300 | 21128 |
| 12 | fail | 4300 | 21802 |
| 13 | success | 3344 | 22326 |
| 14 | success | 1436 | 22550 |
| 15 | success | 997 | 22709 |
| 16 | success | 975 | 22863 |
| 17 | fail | 4300 | 23537 |
| 18 | success | 1161 | 23721 |
| 19 | success | 1190 | 23910 |
| 20 | success | 1174 | 24094 |
| 21 | success | 1157 | 24278 |
| 22 | fail | 4300 | 24952 |
| 23 | fail | 4300 | 25626 |
| 24 | fail | 4300 | 26300 |
| 25 | fail | 4300 | 26974 |
| 26 | success | 1197 | 27163 |
| 27 | success | 841 | 27297 |
| 28 | fail | 4300 | 27971 |
| 29 | success | 1317 | 28180 |
| 30 | success | 858 | 28314 |
| 31 | success | 1130 | 28493 |
| 32 | success | 965 | 28647 |
| 33 | success | 1161 | 28831 |
| 34 | success | 863 | 28965 |
| 35 | success | 1156 | 29149 |
| 36 | fail | 4300 | 29823 |
| 37 | success | 2870 | 30272 |
| 38 | success | 874 | 30411 |
| 39 | success | 807 | 30540 |
| 40 | success | 871 | 30679 |
| 41 | success | 1157 | 30863 |
| 42 | success | 1138 | 31042 |
| 43 | success | 1198 | 31231 |
| 44 | success | 1903 | 31530 |
| 45 | success | 1476 | 31764 |
| 46 | success | 1738 | 32038 |
| 47 | success | 871 | 32177 |
| 48 | success | 3029 | 32651 |
| 49 | success | 1483 | 32885 |
| 50 | success | 858 | 33019 |

Last-10 SR after each round (%): 100, 100, 100, 100, 80, 83.3, 71.4,
62.5, 66.7, 60, 50, 40, 40, 40, 50, 50, 50, 60, 60, 70, 80, 80, 70, 60,
50, 50, 60, 50, 50, 50, 50, 60, 70, 80, 90, 80, 80, 90, 90, 90, 90, 90,
90, 90, 90, 100, 100, 100, 100, 100.

Cumulative SR ends at 74%.

### Train diagnostics vs the frozen copy

Residual = `lora_rms / x_ref_rms`. Kill is `actor_ref_rmse > 0.04`.

| Burst | rmse | residual | q_gap | dq_da | q_success | q_fail | actor_q |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AC end 15980 | 0.014 | 3.4% | ~0 | 4e-4 | 0.34 | 0.16 | 0.25 |
| R1 ~16200 | 0.021 | 9.4% | 0.025 | 4e-4 | 0.31 | 0.13 | 0.26 |
| R5 ~17680 | 0.015 | 8.7% | 0.029 | 4e-4 | 0.51 | 0.23 | 0.41 |
| R10 ~20440 | 0.023 | 7.6% | 0.015 | 3e-4 | 0.73 | 0.46 | 0.61 |
| R12 ~21800 | 0.011 | 6.0% | 0.008 | 2e-4 | 0.79 | 0.49 | 0.67 |
| ~25000 | 0.019 | 6.2% | 0.006 | 2e-4 | 0.74 | 0.52 | 0.64 |
| ~31000 | 0.016 | 6.3% | 0.011 | 2e-4 | 0.76 | 0.52 | 0.66 |
| End 33000 | 0.015 | 6.4% | 0.009 | 3e-4 | 0.79 | 0.44 | 0.60 |

The served policy stayed a small residual on the frozen expert: `rmse`
0.008–0.023 vs kill 0.04, LoRA velocity ~6% of `x_ref` at the end. Q of
success vs fail chunks separated (`q_success` 0.34 → 0.79, `q_fail`
0.16 → 0.44). `dq_da_rms` stayed ~3e-4 the whole run — Q ranks
**episodes** more than **actions**, so `−Q` through 10 expert Euler steps
is a weak, noisy direction. Last-10 10/10 is real; it is still
specialist-like behavior (slightly perturbed copy), not a large LoRA
rewrite.

## Files

- `openpi_comet/src/openpi/training/cf_ae_replay.py`  — decision-chunk replay
- `openpi_comet/src/openpi/training/cf_live.py`       — atomic CF live publish (target LoRA)
- `openpi_comet/src/openpi/training/data_loader.py`   — `OPENPI_CF_REPLAY_INSTANCE`
- `openpi_comet/src/openpi/training/optimizer.py`     — `WarmupConstantSchedule`
- `openpi_comet/src/openpi/models/pi0_cf.py`          — V23 LoRA expert loss + freeze + EMA shadow + per-chunk discount
- `openpi_comet/src/openpi/models/lora.py`            — `lora_scale` on Einsum / FeedForward
- `openpi_comet/src/openpi/models/gemma.py`           — thread `lora_scale` through the expert; `gemma_300m_lora` B=0
- `openpi_comet/src/openpi/policies/policy.py`        — CF live poll + re-JIT `sample_actions`
- `openpi_comet/src/openpi/shared/eval_b1k_wrapper.py`— clear action queue on CF reload
- `openpi_comet/src/openpi/training/config.py`        — `pi05_b1k-turning_on_radio_cf_v23`
- `openpi_comet/scripts/train.py`                     — rebind + freeze-pool/critic zeroing + LoRA polyak + kill 0.04
- `openpi_comet/scripts/serve_b1k.py`                 — serve with `OPENPI_CF_LIVE_PATH`
- `openpi_comet/scripts/run_cf_v23_radio.sh`          — 1-ep live-TD orchestrator (`OPENPI_CF_V23_EXP`)
- `openpi_comet/scripts/cf_v23_watchdog.sh`           — host watchdog
- `openpi_comet/scripts/test_cf_token_ae.py`          — AE + LoRA-filter + LR + discount tests

## Watchdog

Action-expert run (`OPENPI_CF_V23_EXP=cf_v23_ae2`): heartbeats
`outputs/cf_v23_ae2_status.log`, stdout `outputs/cf_v23_ae2_driver.log`.
The host-side watchdog (`cf_v23_watchdog.sh`) checks every 20 min: driver
alive (restarts via `docker exec` if dead), and progress advancing. If
nothing progresses for 35 min it kills stuck phase processes; if still
stuck at 80 min it restarts the driver. Completed rounds are skipped via
`round_N/.done`. `outputs/cf_v23_ae2_DONE` ends the watchdog.
`outputs/cf_v23_ae2_KILLED` also ends it (do not restart).

Do not run a second `cf_v23_watchdog.sh` pointed at another exp in the
same container — `kill_phase_processes` `pkill`s every `train.py`.
