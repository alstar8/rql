# V23: LoRA action-expert RL on `turning_on_radio` (openpi_comet / pi0.5)

V23 trains a residual on a frozen pi0.5 specialist (`pi05-b1kpt12-cs32`,
action horizon 32) in the `openpi_comet` JAX stack. The VLM (`gemma_2b`) and
the base action expert (`gemma_300m`) stay frozen. The trainable actor is
LoRA on `gemma_300m` plus a residual action-decode MLP.

Code: `openpi_comet/` (`models/pi0_cf.py` `compute_cf_v23_loss`,
`training/cf_ae_replay.py`, `training/cf_live.py`, `scripts/train.py`,
`scripts/run_cf_v23_radio.sh`).

**Current experiment: `cf_v23_ae2_301_cfgrl6`** — CFGRL with serve cap on
`w·delta`, `w=1` until 5 wins and last-10 SR ≥ 0.4, uncond BC=1, opt-embed
LR ×3. Seeded from cfgrl5 AC **15000** (LoRA still B=0). Leave `cfgrl5` and
earlier radio runs on disk; do not resume them.

Prior radio runs (artifacts left on disk, do not resume): `cf_v23_ae2`
(308, bootstrap TD, β=100), `cf_v23_ae2_301_qc` (301, CQL, saturated Q;
AC **15000** is the shared seed), `cf_v23_ae2_301_mc` (MC + DDPG, actor
walked off), `cf_v23_ae2_301_awac` (AWAC, LoRA-only), `cf_v23_ae2_301_aout`
(AWAC + MLP; episode-level y, slope gate skipped), `cf_v23_ae2_301_term`
(chunk-reward MC + slope-gated AWAC; SR stuck at copy-hold 67%).

Checkpoints: `/data/DATASETS/behavior/openpi_comet/v23/checkpoints/${EXP}/`.
Online rollouts: `.../outputs/${EXP}_online/round_N/`.
Driver/status/watchdog: `.../logs/${EXP}_{driver,status,watchdog}.log`.

## Method

### Actor

LoRA rank 32 / alpha 32 on attn and ffn, **B = 0** at init
(`gemma_300m_lora`). A forward `lora_scale` threads through `lora.Einsum` /
`lora.FeedForward` and gemma `Attention` / `Block` / `Module`. Scale 0
zeros the LoRA delta without swapping weights. The decode MLP (`cf_action_out`) is last-layer zero-init, so it copies the
specialist Linear until trained. CFGRL adds a 3-token optimality table
(`cf_opt_embed`: ∅ / o=0 / o=1) with zero-init projections into AdaRMS cond
and suffix hidden.

| Piece | Role |
| --- | --- |
| Frozen `gemma_300m` (`lora_scale=0`) | Expert chunk `ã`. Never trained. |
| Live LoRA (`lora_scale=1`) | Transformer residual on the expert. Shared across o. |
| `cf_opt_embed` | CFGRL o ∈ {∅,0,1} embeddings. Zero-init proj so ã is exact at init. |
| `cf_action_out` | Residual MLP: `v = action_out_proj(h) + mlp(h)`. Live path only. |
| Frozen `action_out_proj` | Specialist Linear decode used by ã and as the MLP skip. |
| `cf_token_pool` | RL state `z` for **Q**, not the actor residual. |
| CFTrunk / 10 critic heads | `Q(s, flatten(a_chunk))` on `[z, a]`. |
| `cf_guide_trunk` / `cf_guide_head` | GraphDef / `opt_state` only. Unused by the loss and `sample_actions`. |

Euler is 10 steps from OpenPI time `t=1` (noise) to `t=0` (data):

```
ã = Euler(v_frozen, noise)     # stop-grad; Linear only
a = Euler(v_live,   noise)     # grads through 10 expert steps + MLP
# v_live = action_out_proj(h_lora) + mlp(h_lora)
# v_frozen = action_out_proj(h_base)
```

`action_in_proj`, `action_out_proj`, and `time_mlp_*` stay frozen so
`lora_scale=0` is an exact specialist copy. Plasticity is transformer LoRA
plus the residual decode MLP.

### Freeze filter

`cf_v23_freeze_filter`: train live `cf_*` **and** `.*lora.*`; freeze VLM,
base `gemma_300m`, `action_{in,out}_proj`, `time_mlp_*`, and `cf_target_*`
(including the LoRA EMA shadow and `cf_target_action_out`). Pool / critic
freezes are **zeroed Adam updates** so resume `opt_state` shape is unchanged.

### Critic

`Q(s, flatten(a))` on CFTrunk (`cf_hidden=1536`), 10 heads, `min` over
heads. Not per-step `(a_t, t)`.

State: `phi(s) = [ sg(target_pool(prefix_tokens)), normalize(proprio) ]`.
Pooled tokens are 1024-d; proprio is the 23-d BEHAVIOR-1K state padded to
`action_dim=32`. Chunk is `32×23=736` dims, trunk input `1024+32+736`.

Monte Carlo target **`y = chunk_reward`** (1 only on the finishing success
chunk). Fail episodes: all `y=0`. Success prefixes: `y=0`. Episode-level
`y = episode_success` on every chunk made Q a state classifier
(`q_success−q_fail` large, `dq_da_rms~1e-3`, AWAC `A≈0`) — that is the
aout bug. Bootstrap remains behind `OPENPI_CF_MC_TARGET=0`.

TD action noise defaults to **0**. Nonzero smear (`0.08` plus `0.25×std`)
is larger than CQL local `σ=0.05`, so `Q(s, a+ε)=y` and local CQL cannot
create a slope. CQL still adds shuffle, isotropic Gaussian, and **local**
`a+N(0, 0.05)` negatives. Far negatives (shuffle + Gaussian) are MSE'd to
0; local negatives only enter the CQL logsumexp so Q must slope near the
specialist.

CQL / mismatch / AWAC are weighted by **chunk reward**
(`OPENPI_CF_CQL_SUCCESS_ONLY=1` / `OPENPI_CF_AWAC_SUCCESS_ONLY=1`).

`dq_da_rms` is RMS of `∂ min_k Q_k(s,a) / ∂a` with stop-grad critic
weights. `q_success` is mean Q on **reward=1** rows. `q_fail` is mean Q
on **episode-fail** rows. `q_local` is mean Q on the local negatives.
`positive_frac` is the batch fraction of `y=1` rows.

**Ready** is 1 when `q_success−q_fail ≥ 0.3` **and** `dq_da_rms ≥ 0.01`.
There is no AWAC slope bypass (`require_slope` was deleted). Until ready,
`actor_coef_eff=0` so AWAC / −Q are off; only β=10 and BC=1 hold the copy.
When ready, `actor_coef_eff=1` and `(β_eff, bc_eff)` drop to `(1, 0.1)`.
Logged `critic_ready` / `actor_coef_eff` / `β_eff` are **means over the
log interval** (binary gate per batch), so values like 0.65 mean 65% of
logged batches passed.

`OPENPI_CF_FREEZE_CRITIC=1` zeros the TD term. Actor Q uses stop-grad
critic weights (`_critic_with_sg`). Live Q is not clipped; only the TD
target is. `cf_q_clip_max` is unused by the actor term.

### Token AE

`L_ro = MSE(decoder(z_live), sg(prefix_tokens))` on Paligemma prefix
tokens. Encoder: `CFTokenPool` (`d_model=256`, 4 heads, 2 layers, 8
queries). Decoder: `CFTokenDecoder`, same width. RL always reads
`sg(target_pool(s))`. Live pool trains `L_ro` only.

### Loss (`compute_cf_v23_loss`)

Joint loss, batch 8. Paligemma prefix is still in the TD graph.

```
y       = chunk_reward             # 1 only on the finishing success chunk
L_td    = MSE(Q_heads(s, a), y)    # TD action noise default 0

a_far   = {roll(a), N(0, 2σ_a)}    # shuffle + isotropic
a_loc   = a + N(0, 0.05)           # LoRA-scale (n=2)
L_cql   = E_positive[ softplus(logsumexp Q(s, a_far∪a_loc) − Q(s,a)) ]
L_mis   = E_positive[ MSE(Q(s, a_far), 0) ]     # not a_loc

t ∼ U(0,1)
x_t     = t · x0 + (1−t) · ã
L_bc    = MSE(v_live(x_t, t), sg(v_frozen(x_t, t)))

A       = sg Q(s, a_data) − sg Q(s, ã)
w       = clip(exp(A / τ), 0, 10) · 1_{r=1}     # τ=0.2; stop-grad
L_awac  = Σ w · MSE(a, a_data) / Σ w              # per-dim MSE on replay

# CFGRL (Frans et al.). o∈{0,1} then 10% dropout to ∅.
# label=episode: o=1 on every chunk of a successful episode.
# label=advantage: o=1 iff Q(s,a_data)≥Q(s,ã).
t ∼ U(0,1)
x_t_fm  = t · x0 + (1−t) · a_data
L_cfgrl = masked_mean_i keep_i · MSE_i(v_live(x_t_fm, t, o), x0 − a_data)
keep    = 1 unless (online and episode-fail)   # OPENPI_CF_CFGRL_SKIP_ONLINE_FAIL

L_π     = actor_coef · L_cfgrl + bc_eff · L_bc     # CFGRL: uncond BC holds LoRA
        | actor_coef_eff · (q_ascent_coef · (−mean[Q(s, a) − sg Q(s, ã)])
          + 1_awac · L_awac)
        + β_eff · mean‖a − ã‖² / (32×23)
        + bc_eff · L_bc                             # AWAC path (ready-gated)

L_ro    = cf_ae_coef · MSE(decoder(z_live), sg(prefix_tokens))
L       = cf_critic_coef · L_td + cf_cql_coef · (L_cql + L_mis) + L_π + L_ro
```

- `actor_coef_eff = actor_coef · critic_ready` on the AWAC path. Until ready
  the actor terms are **off**; only anchor and BC hold `a≈ã`. CFGRL does
  **not** wait on ready: `L_π = actor_coef · L_cfgrl + bc_eff · L_bc` with
  `bc_eff = OPENPI_CF_CFGRL_BC_COEF` (default 1) on uncond `v_live(∅)` vs
  `v_frozen`. AWAC / −Q / endpoint-β stay zero so the two CFG factors can
  differ. Uncond BC keeps shared LoRA on ã; the o-table owns the residual.
- Default `OPENPI_CF_Q_ASCENT_COEF=0` (no DDPG through Euler).
  `OPENPI_CF_CFGRL=1` replaces AWAC with classifier-free flow matching.
- `OPENPI_CF_CFGRL_W` is serve-only. Default schedule: `w=1` until
  `online_wins ≥ OPENPI_CF_CFGRL_W_MIN_WINS` (5) **and** last-10 SR ≥
  `OPENPI_CF_CFGRL_W_RAISE_SR` (0.4), then `w=3`. One lucky `w=1` copy-hold
  win must not jump to `w=3`. If last-10 SR drops below 0.4, `w` falls back
  to 1.
- Actor FM drops online-fail chunks (`OPENPI_CF_CFGRL_SKIP_ONLINE_FAIL=1`).
  Offline pt12 success **and** fail stay in the loss so `o=0` / `o=1` / `∅`
  remain distinct. The critic still trains on online timeouts (`y=0`).
- Serve Euler: `v = v_frozen(∅) + cap(w · (v_live(o=1)−v_live(∅)))`
  (`cf_v23_cfgrl_mix_frozen_capped`). The RMS cap binds the **served**
  residual `w·delta` (≤ 0.08), not `delta` before `w`.
- Online (`actor_coef=1`): CFGRL trains from round 1. AC pretrain
  (`actor_coef=0`) never enables the actor term.
- `cf_anchor_normalize=1` divides the sum by `32×23` so `β` is per-dimension.
- `cf_w_l2_coef=0`. `cf_critic_coef=4`, `cf_cql_coef=1`.
- LoRA Adam updates are scaled by `OPENPI_CF_LORA_LR_MULT=1`. The train-log
  field `lora_scale` is this **grad multiplier**, not the Euler forward
  scale.
- `cf_ae_only=1` skips both unrolls and TD; trains `L_ro` only.

Logged: `actor_ref_rmse`, `lora_rms` / `guide_rms` (Δv RMS), `x_ref_rms`,
`q_gap = Q(a)−Q(ã)`, `q_adv` (same), `dq_da_rms`, `q_success`, `q_fail`,
`q_local`, `q_shuf`, `q_rand`, `cql_gap`, `critic_ready`, `beta_eff`,
`bc_eff`, `awac_loss`, `awac_w_mean`, `awac_adv`, `actor_coef_eff`,
`positive_frac`, `action_out_rms`, `action_out_grad_norm`, `chunk_discount`,
`lora_lr_mult`, `lora_grad_norm`, `cfgrl_loss`, `cfgrl_w`, `cfgrl_opt_frac`,
`cfgrl_uncond_frac`, `cfgrl_opt_rms`, `cfgrl_opt_grad_norm`,
`cfgrl_keep_frac`. There is **no** rmse / `|actor_q|` / NaN
kill; `actor_ref_rmse` is logged only.

### Decision-chunk MDP

The env steps at 30 Hz but the policy commits to a 32-step action chunk
(re-inference only when the queue is empty; queue cleared at episode
reset). Each 32-step chunk is one MDP decision:

- `s`    = observation at chunk start
- `a`    = the 32×23 action chunk executed
- `r`    = 1 if the episode terminated with success inside the chunk else 0
- `done` = 1 if the episode ended inside the chunk
- `s'`   = observation at the start of the next chunk (or terminal obs)

`state_action.npz` stores per step: `state` (256-d), `action` (N,1,23),
`next_state`, `next_action`, `reward`, `done`, `truncated`,
`actor_obs__0..3`, `next_actor_obs__*`, `metadata` (`success`, `prompt`).
`cf_ae_replay.py` chunks each episode on the 32-step grid. Short last
chunks pad by repeating the final action. `n_chunks = ceil(T / 32)`
(timeout 4300 → 135 chunks → 675 train steps at utd=5).

Replay is 3-way stratified **fail / success-prefix / terminal**
(`reward≥0.5`) via `cf_ae_stratum_indices` / `cf_ae_map_stratified` so a
batch of 8 sees ~2–3 `y=1` rows. When both offline and online fails exist,
sampling is **4-way** (offline_fail / online_fail / prefix / terminal) so
specialist fails are not drowned by a growing timeout pool. Actor FM skips
online-fail rows (`OPENPI_CF_CFGRL_SKIP_ONLINE_FAIL=1`); specialist fails
still train `o=0`. Fallback is 50/50 over successful / failed
episodes if no terminals exist. Collect instance is
`OPENPI_CF_V23_INSTANCE`. Train filter `OPENPI_CF_REPLAY_INSTANCE` defaults
to the collect instance if unset; set it empty to use the full 100-traj
buffer. Chunks land in `<root>/_cf_ae_chunks/` (memory-mapped, incremental).

### EMA, serve, optimizer

- `CFLoraShadow` (`cf_target_lora`): copies live LoRA at init into `t0..tN`
  Params so Orbax checkpoints them. Frozen by `cf_target_*`.
- Serve: CFGRL Euler is frozen ã plus the capped live o-residual
  `cap(w · (v_live(o=1)−v_live(∅)))`. Shared LoRA cancels in that residual.
  The cap is on `w·delta` (RMS ≤ `cf_cfgrl_serve_delta_cap`). Capping
  `delta` then multiplying by `w` (cfgrl5) made `w=3` a 0.24 RMS
  excursion. Capping `v_live(o=1)−v_frozen(∅)` lets LoRA steal the cap
  (cfgrl4 rounds 14–31). `OPENPI_CF_APPLY_GUIDE` is ignored when
  `cf_v23=True`. `cf_live.npz`
  payload `lora` / `cf_action_out` are the **target** adapters; serve writes
  them onto live LoRA / MLP and re-JITs. Default path
  `checkpoints/${EXP}/cf_live.npz`.
- Polyak: critic `τ=0.005` on `CF_V23_TARGET_PAIRS` (includes
  `cf_action_out`), then `polyak_cf_target_lora` at ema 0.999. Full
  TrainState `ema_decay` is `None`.
- LR: `WarmupConstantSchedule` peak `1e-4`, warmup 200, then constant.
- Weight loader: pt12 params, `missing_regex=".*cf_.*|pointnet.*|.*lora.*"`.

`ot_compose_integrate` remains in `pi0_cf.py` for tests. The V23 loss and
`sample_actions` do not call it.

### Training protocol

1. **Data**: 100 `state_action.npz` from the pt12 collector
   (`outputs/pt12_cs32_radio_10x10_states`), public_test instances 0–9.
2. **AE pretrain** (once, on qc): 8000 steps, `cf_ae_only=1`, `ae_coef=1`,
   `freeze_critic=1`, `freeze_pool=0`, `actor_coef=0`.
3. **AC pretrain** (once, on qc): 8000 more steps to **15000**,
   `actor_coef=0`, `freeze_pool=1`, `freeze_critic=0`, `ae_coef=0`. Critic
   TD on `sg(target_pool(s))`. LoRA BC + anchor keep the expert a copy of
   pt12.
4. **Online** (term): 100 rounds × {collect 1 ep on 301 + train
   `5 × n_chunks` steps} from that AC 15000, after seeding the MLP GraphDef:
   - `actor_coef=1`, `freeze_critic=0`, `freeze_pool=1`, `ae_coef=1`
   - One serve + one client (`OPENPI_CF_V23_INSTANCE=301` → eval id 0)
   - Publish `cf_live.npz` (frozen encoder / critic trunk / **target LoRA**
     / **target action-out**)
   - Replay = 301-only, 3-way fail / prefix / terminal
   - `OPENPI_CF_V23_EXP=cf_v23_ae2_301_term`

Success = `q_score.final >= 1`; timeout = 4300 env steps (~143 s sim).
Frozen pt12 specialist on 301 is **5/10**. Typical term successes are
~1100–1800 steps; slow successes (2000–3200) still count.

### Defaults (`cf_v23_ae2_301_cfgrl6`)

`OPENPI_CF_FREEZE_CRITIC=0`, `OPENPI_CF_FREEZE_POOL=1`,
`OPENPI_CF_UNIT_BALL=0`, `OPENPI_CF_ACTOR_COEF` (0 pretrain / 1 online),
`OPENPI_CF_BC_COEF=1.0`, `OPENPI_CF_BC_COEF_LOOSE=0.1`,
`OPENPI_CF_ANCHOR_BETA=10`, `OPENPI_CF_ANCHOR_BETA_LOOSE=1`,
`OPENPI_CF_ANCHOR_NORMALIZE=1`, `OPENPI_CF_UTD=5`, `OPENPI_CF_W_L2=0`,
`OPENPI_CF_CRITIC_COEF=4`, `OPENPI_CF_GAMMA=0.99`,
`OPENPI_CF_DISCOUNT_PER_CHUNK=1`, `OPENPI_CF_ACTOR_EMA=0.999`,
`OPENPI_CF_V23_INSTANCE=301`, `OPENPI_CF_REPLAY_INSTANCE=301`,
`OPENPI_CF_CHUNK_CRITIC=1`,
`OPENPI_CF_TD_ACTION_NOISE=0`, `OPENPI_CF_TD_NOISE_FRAC=0`,
`OPENPI_CF_CQL_COEF=1`, `OPENPI_CF_CQL_N_ACTIONS=4`,
`OPENPI_CF_CQL_LOCAL_NOISE=0.05`, `OPENPI_CF_CQL_LOCAL_N=2`,
`OPENPI_CF_CQL_SUCCESS_ONLY=1`, `OPENPI_CF_MC_TARGET=1`,
`OPENPI_CF_ACTOR_ADVANTAGE=1`, `OPENPI_CF_CRITIC_READY_Q_GAP=0.3`,
`OPENPI_CF_CRITIC_READY_DQDA=0.01`, `OPENPI_CF_AWAC=0`,
`OPENPI_CF_AWAC_TEMP=0.2`, `OPENPI_CF_AWAC_CLIP=10`,
`OPENPI_CF_AWAC_SUCCESS_ONLY=1`, `OPENPI_CF_Q_ASCENT_COEF=0`,
`OPENPI_CF_CFGRL=1`, `OPENPI_CF_CFGRL_W=1`,
`OPENPI_CF_CFGRL_W_UNTIL_WIN=1`, `OPENPI_CF_CFGRL_W_EXTRAPOLATE=3`,
`OPENPI_CF_CFGRL_W_MIN_WINS=5`, `OPENPI_CF_CFGRL_W_RAISE_SR=0.4`,
`OPENPI_CF_CFGRL_COND_SCALE=32`, `OPENPI_CF_OPT_EMBED_LR_MULT=3`,
`OPENPI_CF_CFGRL_SKIP_ONLINE_FAIL=1`,
`OPENPI_CF_CFGRL_SERVE_FROZEN_UNCOND=1`,
`OPENPI_CF_CFGRL_SERVE_DELTA_CAP=0.08`,
`OPENPI_CF_CFGRL_OPT_DELTA_CAP=0.35`, `OPENPI_CF_CFGRL_OPT_DELTA_PENALTY=100`,
`OPENPI_CF_CFGRL_BC_COEF=1`,
`OPENPI_CF_CFGRL_DROPOUT=0.1`, `OPENPI_CF_CFGRL_LABEL=episode`,
`OPENPI_CF_ACTION_OUT_MLP=1`, `OPENPI_CF_LORA_LR_MULT=1`,
`OPENPI_CF_AE_STEPS=8000`, `OPENPI_CF_AC_STEPS=8000`.

GPU: sim **0**, serve+train **1**, `FSDP_DEVICES=1`, `TRAIN_GPUS=1`,
`TRAIN_MEM=0.90`. Driver is `python` +
`PYTHONPATH=openpi_comet/src:packages/openpi-client/src` — **not** `uv run`.
`OPENPI_DATA_HOME` = v23 `outputs/openpi_cache`. Keep `huggingface-hub` 0.34.4.

Tests (`openpi_comet/scripts/test_cf_token_ae.py`, CPU
`JAX_PLATFORMS=cpu`): `test_v23_lora_filter_excludes_shadow`,
`test_chunk_critic_depends_on_action`,
`test_warmup_constant_lr_does_not_hit_zero`,
`test_per_chunk_discount_keeps_credit_over_radio_horizon`,
`test_action_contrast_mismatch_heads_ignore_local`,
`test_action_contrast_success_weight_ignores_fail`,
`test_critic_ready_thresholds`,
`test_stratum_indices_split_prefix_and_terminal`,
`test_map_stratified_three_way_oversamples_terminal`,
`test_awac_weights_success_mask_and_clip`,
`test_action_out_mlp_zero_init_is_identity`,
`test_v23_freeze_trains_action_out_mlp`,
`test_v23_freeze_trains_opt_embed`,
`test_cfgrl_opt_ids_dropout_and_mapping`,
`test_cfgrl_mix_weighting`,
`test_cfgrl_mix_capped_disabled_is_plain_mix`,
`test_cfgrl_mix_capped_under_cap_is_identity`,
`test_cfgrl_mix_capped_bounds_delta_and_keeps_direction`,
`test_cfgrl_mix_capped_w_does_not_inflate_cap`,
`test_cfgrl_mix_frozen_capped_ignores_shared_lora`,
`test_cfgrl_serve_w_waits_for_wins_and_recent_sr`,
`test_cfgrl_mix_frozen_capped_equals_live_mix_when_lora_is_zero`,
`test_cfgrl_scale_deltas_lora_zero_is_identity`,
`test_cfgrl_actor_keep_drops_online_fail`,
`test_cfgrl_masked_mean_zero_grad_on_dropped`,
`test_cfgrl_serve_uncond_uses_frozen_specialist`,
`test_cf_ae_chunk_source_keeps_round_prefix`,
`test_stratum_splits_offline_and_online_fail`,
`test_map_stratified_four_way_keeps_offline_fail`,
`test_opt_embed_zero_init_is_identity`. No `require_slope=False` case.

### Current-code leftovers (intentional)

- Paligemma is still run for the current prefix (Q(s,a) / actor). MC
  targets skip the next-obs Paligemma unroll. Batch is 8. Pooled `z` is
  not cached in replay.
- Train re-unrolls the frozen expert for `ã`; the collect-time chunk is
  not stored as `reference`. Euler noise can differ from collect.
- Official metric is 1 training episode on instance 301. No multi-instance
  `episode_pool` / held-out eval in this recipe.
- Two separate critic SGD steps on fresh batches are not implemented;
  `cf_critic_coef=4` is the joint-loss analogue.
- `cf_guide_*` and `ot_compose_integrate` stay so GraphDef / tests keep
  their shapes; the actor path ignores both.
- Target LoRA EMA and critic Polyak still run. MC does not use `a'(s')`.
- CFGRL serve does three suffix unrolls per Euler step when frozen ∅ is
  on: frozen ∅, live ∅, live o=1. Live-∅ mix (frozen off) is two.

## Experiment: `cf_v23_ae2_301_cfgrl4` (CFGRL + serve delta cap + o-table barrier)

Fresh 100-round online from ckpt **17000** of `cf_v23_ae2_301_cfgrl`
(pre-walk-off LoRA), same recipe as cfgrl3 plus two magnitude constraints
that keep the CFGRL extrapolation inside the region where wins happened
(cfgrl3 post-mortem below):

1. **Serve o-residual cap** (`cf_v23_cfgrl_mix_frozen_capped`):
   `v = v_frozen(∅) + w · cap(v_live(o=1) − v_live(∅))` with per-row RMS
   ≤ 0.08. Direction of the o-residual is preserved. Capping
   `v_live(o=1)−v_frozen(∅)` (the cfgrl4 launch mix) lets shared LoRA
   dominate the vector; after `lora_rms` grew to ~0.21 the 0.08 cap
   scaled the o-component nearly to 0 and `w=3` went 0/18 (rounds 14–31).
2. **o-table barrier**: `10 · relu(cfgrl_opt_delta_rms − 0.35)²` added to
   the loss. Below 0.35 the loss is untouched (pure CFGRL); above, the
   penalty dominates and pulls the scaled o-deltas back. Stops the
   monotone o-table growth (0.36→1.57 in cfgrl3) that also forced the
   shared LoRA to compensate-drift.

Everything else as cfgrl3: skip-online-fail actor mask, 4-way replay,
`w=1` until first online win then `w=3`, frozen ∅ at serve, no BC/β.

Health: `cfgrl_opt_delta_rms` must plateau ≤ 0.35 (barrier active),
`cfgrl_opt_delta_barrier` > 0 only transiently, `keep_frac≈0.75`,
`actor_ref_rmse` in the 0.05 band, gripper outliers gone (|g| ≲ 1.1),
SR climbing past the 3/92 cfgrl3 ceiling.

Ops: 1 win at round 13 (`w=1`, 2789 steps), then 0/18 at `w=3` (rounds
14–31) under the LoRA-contaminated cap. Round 32 then spun ~28h:
container NVML died (`CUDA_ERROR_NO_DEVICE`) while status-log retries
fooled the watchdog into "progress ok". Mix switched to
`mix_frozen_capped` at ckpt **37659**; watchdog now restarts the
container on NVML failure and counts only npz/ckpt mtimes as progress.
Do not start a second watchdog.
Stopped 2026-09-09 at round 37 (step 40139, `online_wins=3`) to launch
cfgrl5 from zero. Leave artifacts; do not resume.

## Experiment: `cf_v23_ae2_301_cfgrl5` (from-scratch AE + AC + CFGRL)

Full driver `phase=all` on the current code (token AE, AC, frozen-ã
o-residual cap, o-table barrier). Reuses the existing 100-traj pt12
buffer (`outputs/pt12_cs32_radio_10x10_states`); does **not** copy
qc/cfgrl/cfgrl4 checkpoints.

1. **Wait**: skip collect if 100 `state_action.npz` already exist.
2. **Token AE**: 8000 steps, `cf_ae_only=1`, `ae_coef=1`, `freeze_pool=0`,
   `freeze_critic=1`, `actor_coef=0`. Trains `CFTokenPool` /
   `CFTokenDecoder` (`L_ro`). `--overwrite` from empty ckpt dir.
3. **AC pretrain**: 8000 more steps to **16000**, `actor_coef=0`,
   `freeze_pool=1`, `ae_coef=0`, critic+CQL on `sg(target_pool)`. CFGRL
   actor term is off so LoRA stays at B=0 (specialist copy).
4. **Probe** then **online**: 100 rounds on instance 301, CFGRL as
   cfgrl4 (`w=1` until a win then `w=3`, `mix_frozen_capped`, skip
   online-fail, cond_scale=32, opt-embed LR ×10, no BC/β/AWAC/−Q).

Started 2026-09-09, one host watchdog
`OPENPI_CF_V23_EXP=cf_v23_ae2_301_cfgrl5`,
`OPENPI_CF_V23_PHASE=all`, `OPENPI_CF_V23_ROUNDS=100`. Host log:
`/home/admin/cf_v23_ae2_301_cfgrl5_watchdog.log`.

**Outcome: 2/16, do not continue.** Probe and R1 succeeded at `w=1` (specialist
copy). `w` then jumped to 3. First 200 CFGRL steps blew `actor_ref_rmse`
0→0.86 (`cond_scale=32`, opt-embed LR ×10, no uncond BC). R2 walked off
(hands ~136 m, gripper outliers). Cap-then-`w` served a 0.24 RMS residual.
Later timeouts at ~30 m. Critic was healthy and unused. Stopped 2026-09-09
during round 16 train. Leave artifacts; do not resume.

## Experiment: `cf_v23_ae2_301_cfgrl6` (cap `w·delta`, delayed `w`, uncond BC)

Online from cfgrl5 AC ckpt **15000** (LoRA still B=0; critic already CQL'd).
Does not copy cfgrl5 online steps. Fixes from the cfgrl5 post-mortem:

1. **Serve cap on `w·delta`**, not `delta` then `×w`. Served residual RMS
   ≤ 0.08 at both `w=1` and `w=3` once the o-field is large. Small o-deltas
   still scale with `w` until they hit the cap.
2. **`w=1` until 5 online wins and last-10 SR ≥ 0.4**, then `w=3`. Falls
   back to `w=1` if last-10 drops below 0.4. One copy-hold win must not
   extrapolate.
3. **Uncond BC=1** (`OPENPI_CF_CFGRL_BC_COEF`): `MSE(v_live(∅), v_frozen)`
   holds shared LoRA on ã. Opt-embed LR ×3 (not ×10). Barrier penalty 100.

`phase=from-pretrain` (AE skip at 15000, AC to 16000, probe, 100 online).
Started 2026-09-09, one host watchdog
`OPENPI_CF_V23_EXP=cf_v23_ae2_301_cfgrl6`,
`OPENPI_CF_V23_PHASE=from-pretrain`, `OPENPI_CF_V23_ROUNDS=100`. Host log:
`/home/admin/cf_v23_ae2_301_cfgrl6_watchdog.log`.

## Experiment: `cf_v23_ae2_301_cfgrl3` (CFGRL, behavior FM + w after a win)

Fresh 100-round online from ckpt **17000** of `cf_v23_ae2_301_cfgrl`
(pre-walk-off LoRA). Leave cfgrl and cfgrl2 on disk; do not continue them.

Same as cfgrl2 (`cond_scale=32`, opt-embed LR ×10, no BC/β) plus the
fixes for that collapse:

1. **Actor FM skips online timeouts** (`keep=0` on `online ∧ ¬success`).
   `o=1` on episode success, `o=0` on specialist fails, 10% dropout to ∅.
   Critic still trains on the 4-way batch including timeouts. Full-batch
   FM (skip removed 2026-09-06) cloned those timeouts into the shared
   LoRA / o-embed (`actor_ref_rmse` 0.05→0.22–1.0 by step ~42k). Frozen-ã
   serve could not recover SR. Skip restored 2026-09-07; actor rolled
   back to ckpt **34000** (last skip-on save). `cfgrl_keep_frac≈0.75`.
2. **4-way replay** once online fails exist: offline_fail / online_fail /
   prefix / terminal. Specialist fails keep 1/4 of the batch so `o=0`
   does not drown in timeouts. Buffer ingest is real (root `*_online`
   stamps `observation.online`).
3. **Serve `w=1` until an online win**, then `w=3`, with **frozen ∅** at
   serve (`lora_scale=0` on the uncond Euler). Mixing two *live* LoRA
   fields at `w=3` was `3 v(o=1) − 2 v(∅_timeout)` and locked the right
   gripper open (`gR≈+1`, finger width 0.070) after round 5. Frozen ∅
   makes `w>1` extrapolate against ã, not against timeout LoRA.

Started 2026-09-06 from-online, one host watchdog
`OPENPI_CF_V23_EXP=cf_v23_ae2_301_cfgrl3`, `OPENPI_CF_V23_ROUNDS=100`,
`OPENPI_CF_CFGRL=1`, `OPENPI_CF_CFGRL_W=1`,
`OPENPI_CF_CFGRL_COND_SCALE=32`, `OPENPI_CF_OPT_EMBED_LR_MULT=10`,
`OPENPI_CF_CFGRL_LABEL=episode`,
`OPENPI_CF_AWAC=0`, `OPENPI_CF_Q_ASCENT_COEF=0`, `OPENPI_CF_LORA_LR_MULT=1`.
Host log: `/home/admin/cf_v23_ae2_301_cfgrl3_watchdog.log`.

**Outcome: 3/92, do not continue.** Wins at rounds 4 (`w=1`), 9 and 13
(`w=3`), then 0/79. Post-mortem (2026-09-08):

- Buffer/loss were correct while wired: 4-way sampling every round
  (`four_way=True`, strata counts logged), `keep_frac≈0.75`, o-table and
  LoRA grads nonzero, `cf_grad_finite=1`, critic healthy
  (`q_success≈0.7–0.9`, `q_fail≈0` = MC y=0, positive CQL gap).
- A mid-run code edit broke `observation.online` for 13 rounds
  (`keep_frac=1.0`, steps ~34300–43060); actor was restored to 34000 and
  the wiring fixed at 21:58 UTC. **SR never recovered after the restore.**
- Root cause of the plateau: the o-direction grows unbounded
  (`cfgrl_opt_delta_rms` 0.36→1.57 post-restore; `lora_rms` 0.17→0.41).
  Serve delta `rms(v(o=1)−v(∅))` measured on replay batches: **0.06 at the
  win-era ckpts (24000/34000) → 0.15–0.24 at 49000/69048**. `w=3` × that
  is a ~0.5–0.7 RMS excursion from ã (field RMS ≈1.08) → off-manifold
  actions (late rollouts: arm RMS up to 39, gripper outliers ±18) →
  timeouts → no new `o=1` data → nothing corrects the drift.
- Fixes in cfgrl4: serve delta cap 0.08 (win-era magnitude) + o-table
  barrier at 0.35.

## Experiment: `cf_v23_ae2_301_cfgrl2` (CFGRL, strong o)

Stopped: o-deltas were live (`cfgrl_opt_delta_rms` 0.08→0.50) but every
online episode timed out (0/36). Round 1 at `w=1` from 17000 already
failed; then `w=3` plus unmasked timeout FM cloned flailing into the
shared LoRA (`actor_ref_rmse` 0.08→0.25, grippers stuck open). Do not
continue this exp.

## Experiment: `cf_v23_ae2_301_cfgrl` (CFGRL actor)

Stopped: o-conditioning stayed dead (`cfgrl_opt_rms≈0.004`), LoRA cloned
fail timeouts, SR 12/28 after a 17000 restore still collapsed (R28 timeout).
Do not continue this exp.

Online only from AC ckpt **15000** of `cf_v23_ae2_301_aout` after
`scripts/seed_cf_v23_cfgrl.py` writes a new GraphDef with zero-init
`cf_opt_embed`. Do not Orbax-resume from term/aout/qc without that module.
Leave the term run on disk; do not continue it.

Loss / data vs term:

1. **CFGRL** unweighted flow matching to `a_data` with o ∈ {∅,0,1}.
   `o=1` on every chunk of a successful episode (not just the finishing
   chunk). 10% dropout to ∅.
2. **Serve guidance** `w=3`: `v=(1-w)v(∅)+w v(o=1)`. Tune `w` without
   retraining.
3. **No copy-hold actor.** While CFGRL is on, AWAC / −Q / L_bc / β are
   zeroed and the critic-ready gate does not block the actor.
4. Critic stays chunk-reward MC + CQL (same as term) for logging and the
   optional `label=advantage` mode.

Started 2026-09-05 from-online, one host watchdog
`OPENPI_CF_V23_EXP=cf_v23_ae2_301_cfgrl`, `OPENPI_CF_V23_ROUNDS=100`,
`OPENPI_CF_CFGRL=1`, `OPENPI_CF_CFGRL_W=3`,
`OPENPI_CF_CFGRL_LABEL=episode`, `OPENPI_CF_AWAC=0`,
`OPENPI_CF_Q_ASCENT_COEF=0`, `OPENPI_CF_ACTION_OUT_MLP=1`. Host log:
`/home/admin/cf_v23_ae2_301_cfgrl_watchdog.log`.

Useful actor learning: `cfgrl_opt_rms` leaving 0, `actor_ref_rmse` toward
~0.02–0.03, last-10 SR beating the 67% copy-hold / term baseline.

## Experiment: `cf_v23_ae2_301_term` (chunk-reward MC + slope gate)

Stopped in favor of CFGRL. Online from AC ckpt **15000** of
`cf_v23_ae2_301_aout`. Do not Orbax-resume term from qc without the MLP
GraphDef. Do not continue aout from ~39893.

Loss / data vs aout:

1. **MC target** `y = chunk_reward` (finishing success chunk only).
   CQL / AWAC masks use the same positive rows (`_positive_weight`).
2. **Ready always requires slope.** Deleted `require_slope` (AWAC no
   longer skips `dq_da`).
3. **TD action noise = 0.** CQL local `σ=0.05` supplies the neighborhood.
4. **Replay instance 301** plus 3-way fail / prefix / terminal
   oversampling.

Started 2026-09-04 from-online, one host watchdog
`OPENPI_CF_V23_EXP=cf_v23_ae2_301_term`, `OPENPI_CF_V23_ROUNDS=100`,
`OPENPI_CF_AWAC=1`, `OPENPI_CF_Q_ASCENT_COEF=0`,
`OPENPI_CF_ACTION_OUT_MLP=1`. Host log:
`/home/admin/cf_v23_ae2_301_term_watchdog.log`.

### What learned vs the pretrained specialist

Frozen pt12 on 301: **5/10**. Seed 15000 is that expert plus a tiny LoRA
residual from critic AC (`actor_ref_rmse≈0.009` vs LoRA-off).

Slope first crossed `dq_da_rms≥0.01` at step **26720** (after round 33
collect). Rounds **1–33** were copy-hold (`actor_coef_eff=0`): **22/33
(67%)**. Rounds **34+** are AWAC-on collect: **~67%**, same as the copy.
Success length ~1400 steps vs pt12's mixed 1k–3.5k. The 10/10 streak in
rounds 22–31 is still copy-hold.

Actor distance to the frozen expert stayed copy-scale: `actor_ref_rmse`
**0.009 → ~0.01–0.022** (old kill line was 0.04). `action_out` last linear
kernel RMS ~0.0014 (still near zero-init). `awac_adv` median ~0.015 →
weight `exp(A/0.2)≈1.08` (almost uniform BC when the gate is on). The
critic is what moved: `dq_da_rms` 0.001 → 0.05–0.25, `q_local` well below
data Q, `q_fail≈0`. Occasional `q_success` dips to ~0.03–0.25 (empty
positive rows / overshoot) drop logged `critic_ready` and raise `β_eff`
back toward 10, which is the gate working.

Useful actor learning would be `actor_ref_rmse` toward ~0.02–0.03 **and**
last-10 SR beating the 67% copy-hold baseline.

### Snapshot (2026-09-05 ~00:29 UTC+3)

Not DONE. One watchdog, driver alive.

- **SR 51/76** (67%). Last 10 **6/10**. Round 76 succeeded in 1187 steps.
  Ckpt **44469**, last logged step **44460**.
- Last train window: `critic_ready=0.25`, `actor_coef_eff=0.25`,
  `dq_da_rms=0.075`, `q_success≈0.87`, `q_fail≈0`, `β_eff=7.8`,
  `actor_ref_rmse=0.011`, `action_out_rms=0.0039`.
- R1–10 **7/10** (same as aout seed, not RL). Overall holds vs aout's
  collapse after R33 (aout R34–45 was 2/12).

## Prior experiments (why this recipe)

Leave all of these on disk. Do not restart aout or AWAC.

**`cf_v23_ae2`** (308, bootstrap TD, β=100, no CQL). 2026-08-31 →
2026-09-01, step 33019, **37/50 = 74%**, last-10 10/10. Frozen 308
specialist 8/10. `actor_ref_rmse` 0.008–0.023, `dq_da_rms~3e-4` the whole
run — Q ranked episodes, last-10 10/10 is a slightly perturbed copy.

**`cf_v23_ae2_301_qc`** (301, CQL, β=10, bootstrap TD). Online **20/46 =
43.5%**. Specialist 301 **5/10**. Mid-run `q_success≈q_fail≈1`,
`dq_da_rms` 1e-4–2e-3: CQL mesa around the specialist. AC **15000** is the
shared online seed. Round 47 hit `No visible GPU devices` (NVML; docker
restart). Latest ckpt 39033 — do not resume term from this GraphDef
(no `cf_action_out`).

**`cf_v23_ae2_301_mc`** (online from qc 15000). `y = episode_success`,
local CQL, DDPG advantage. Rounds 1–5 all timeout. Critic worked
(`q_success≈0.91`, `q_fail≈0.01`) but −Q ran from round 1 because the
gate only dropped β/BC and did **not** zero the actor term.
`actor_ref_rmse` 0.014 → **0.12**. Stopped at 17696. Kill was documented
but missing in `train.py` (now removed on purpose).

**`cf_v23_ae2_301_awac`** (same seed, AWAC, LoRA-only, −Q off). Gated
`actor_coef_eff` until Q-gap 0.3; **skipped `dq_da`**. First 8 rounds 6/8,
then an old rmse kill at 17200 (`actor_ref_rmse=0.0410`). Continued from
17003 toward 100 rounds. Same stall class as aout (ranking critic, AWAC
≈ success-masked BC).

**`cf_v23_ae2_301_aout`** (AWAC + MLP from qc 15000). R1–10 **7/10**,
R1–16 **11/16**, R1–45 **19/45 (42%)**, last 10 at R45 **1/10**. Stopped
collecting R48 from **39893**. `actor_coef_eff=1` from ~15100,
`dq_da_rms≈0.0014`, `awac_adv` mean **0.003**. Causes: (1) episode-level
y; (2) `require_slope=not cf_awac`; (3) TD noise 0.08 + 0.25×std; (4)
replay = all 10 instances. `actor_ref_rmse` 0.019→0.032,
`action_out_rms` 0→0.0055.

## Files

- `openpi_comet/src/openpi/training/cf_ae_replay.py`  — 3-way stratified chunk replay
- `openpi_comet/src/openpi/training/cf_live.py`       — atomic CF live publish (target LoRA + MLP + opt embed)
- `openpi_comet/src/openpi/training/data_loader.py`   — `OPENPI_CF_REPLAY_INSTANCE`
- `openpi_comet/src/openpi/training/optimizer.py`     — `WarmupConstantSchedule`
- `openpi_comet/src/openpi/models/pi0_cf.py`          — V23 loss, CFGRL opt embed + guided Euler, freeze, EMA shadow, action-out MLP, chunk-reward MC
- `openpi_comet/src/openpi/models/lora.py`            — `lora_scale` on Einsum / FeedForward
- `openpi_comet/src/openpi/models/gemma.py`           — thread `lora_scale`; `gemma_300m_lora` B=0
- `openpi_comet/src/openpi/policies/policy.py`        — CF live poll + re-JIT `sample_actions`
- `openpi_comet/src/openpi/shared/eval_b1k_wrapper.py`— clear action queue on CF reload
- `openpi_comet/src/openpi/training/config.py`        — `pi05_b1k-turning_on_radio_cf_v23`; TD noise defaults 0
- `openpi_comet/scripts/train.py`                     — rebind + freeze-pool/critic zeroing + LoRA polyak
- `openpi_comet/scripts/serve_b1k.py`                 — serve with `OPENPI_CF_LIVE_PATH`
- `openpi_comet/scripts/run_cf_v23_radio.sh`          — 1-ep live-TD orchestrator; replay defaults to collect instance
- `openpi_comet/scripts/cf_v23_watchdog.sh`           — host watchdog (passes MLP, replay instance, TD noise 0)
- `openpi_comet/scripts/seed_cf_v23_cfgrl.py`     — seed opt-embed GraphDef from aout AC
- `openpi_comet/scripts/seed_cf_v23_action_out.py`    — seed new GraphDef from an AC params dir
- `openpi_comet/scripts/test_cf_token_ae.py`          — AE + LoRA-filter + strata + ready + CFGRL tests

## Watchdog

Host `cf_v23_watchdog.sh` heartbeats `logs/${EXP}_status.log`, stdout
`logs/${EXP}_driver.log`. Check every 20 min: driver alive (restarts via
`docker exec` if dead), progress advancing. Stuck 35 min → kill phase
processes; stuck 80 min → restart driver. Completed rounds skipped via
`round_N/.done`. `logs/${EXP}_DONE` ends the watchdog.

Do **not** run a second `cf_v23_watchdog.sh` in the same container —
`kill_phase_processes` `pkill`s every `train.py`. If in-container
`nvidia-smi -L` fails with `Failed to initialize NVML: Unknown Error`,
`docker restart b1k-airi-dev` then re-test JAX on GPU 1. Restart docker
only when NVML is actually down.
