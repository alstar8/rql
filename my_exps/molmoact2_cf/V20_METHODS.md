# V20: Central Online RLT Learner + Offline Molmo AE CFGRL

V20 replaces the V19 fleet of 32 independent learners with **one central
learner** that owns a single RLT actor+critic net and a single pooled replay.
All 32 workers are rollout-only: they stream trajectories to the learner,
receive the latest actor weights, and never update model parameters or
normalization statistics themselves.

The method alternates two phases:

- **Online RLT phase.** Workers collect episodes with the current incumbent
  policy; the learner trains the RLT critic (ensemble TimeCQL) and the RLT
  actor (CFGRL flow policy) on the pooled replay. The MolmoVLA (VLM + action
  expert) remains frozen online.
- **Offline Molmo AE phase.** Every `rounds_per_offline` online rounds, the
  learner fine-tunes the Molmo action expert (AE) via CFGRL flow matching on
  the collected online image trajectories (LoRA-as-CFG; the VLM stays
  frozen). The updated AE is then served as the reference policy for
  subsequent rounds.

V20 uses `runs/rlt_cf_v20_kettle` and does not read or modify the V19
directory.

## Roles

- One **learner** process owns `replay/pooled.npz`, the RLT actor, the RLT
  critic, all optimizer state, and all model checkpoints.
- Thirty-two **workers** run MolmoSpaces episodes in CUDA-free isolated
  children. Each worker holds a read-only copy of a checkpoint for inference
  and a local HTTP session to its GPU's VLA server. Workers write immutable
  episode NPZ files and publish lightweight journal records; they do not hold
  optimizer state.
- One frozen **MolmoAct2 HTTP server per GPU** supplies the VLA action
  reference `ã` and encodes VLM token embeddings through the frozen RL-token
  encoder \(g_\phi\) to produce `z_rl`. Four rollout workers share each server.
- Environment rollouts execute in CUDA-free isolated children. An EGL crash
  invalidates one trial rather than the learner.

Collection and probes both use the **incumbent** checkpoint recorded in
`round_state.json`. The incumbent starts as the pretrained reference
(`incumbent_mode="reference"`) and advances to a trained actor
(`incumbent_mode="actor"`) whenever a candidate wins the paired-probe
promotion gate (see [Evaluation and promotion](#evaluation-and-promotion)), so
on-policy data is actually collected once the actor earns it.

## RL token: what is trained when

The RL token is the RLT readout \(g_\phi\), not the Molmo VLM and not the
action expert. `RLTokenEncoder` maps VLA token embeddings \(\bar z_{1:M}\) to
\(z_{\mathrm{rl}}\in\mathbb{R}^{256}\) by appending a learned RL-token
embedding, running a 4-layer / 4-head transformer (\(d=512\)), and reading out
the last position. A paired decoder \(d_\phi\) exists only to reconstruct
\(\bar z_{1:M}\) from \(z_{\mathrm{rl}}\) during warmup. Actor and critic state
is \(x=(z_{\mathrm{rl}}, s^p)\), passed through a learned `z_expand` adapter
(\(256 \to 512\)). That adapter is **not** the RL token.

```
RGB, proprio, language
        │
   MolmoAct2 VLM (always frozen)          → token embeddings z̄_{1:M}
        │
   Token AE encoder g_φ  →  z_rl ∈ R^{256}   ← trained once, then frozen
        │
   z_expand adapter                           ← trained online; not g_φ
        │
   x = (z_expand(z_rl), proprio)
        │
   EnsembleTimeCQL Q_ψ  and  FlowVelocityActor v_θ(· | x, ã, o)
```

### Pretrain (once, before V20)

The pretrain chain is `collect_house0_kettle_offline.sh` →
`pretrain_rlt_house0_kettle_cfgrl.sh`, run from scratch:

1. **Fresh-scaffold collect.** Collection needs an RLT checkpoint only as a
   token-export scaffold (the frozen VLA supplies all actions,
   `actor_mode=vla_only`, 0 RLT updates). The scaffold is built by
   `make_fresh_scaffold.py`: random weights, identity normalization
   statistics. No previously trained artifact is consumed anywhere in the
   chain; the only pretrained component is the MolmoAct2 VLA itself.
2. **Token-encoder warmup** (`warmup_rlt_token.py`): 8,000 reconstruction
   steps (batch 4) on the collected `TokenReplay`,
   \[
   \mathcal{L}_{\mathrm{ro}}
   =
   \mathbb{E}\Big[\sum_i
   \big\| d_\phi(z_{\mathrm{rl}}, \bar z_{1:i-1})_i
   - \bar z_i\big\|_2^2\Big].
   \]
   There is no success filter: reconstruction sees both successful and failed
   VLA rollouts. The target encoder is copied from the online encoder after
   warmup so TD bootstrapping does not start from a random readout.
3. **Re-encode** (`reencode_chunk_replay.py`): every chunk's \(z\) and
   \(next\_z\) is re-encoded by the warmed encoder, streaming the per-shard
   token NPZs; episode boundaries set `terminal` and zero the final
   \(next\_z\).
4. **Born-CFGRL flow pretrain** (`warmup_flow_critic.py --use_cfgrl`):
   15,000 steps of critic TD + CFGRL actor updates on the re-encoded replay.
   The model is constructed at the exact V20 architecture (actor/critic
   `hidden=1024`, `n_hidden_actor=10`, `n_hidden_critic=5`,
   `z_expand_dim=512`, LayerNorm heads, CFGRL `o_dim=16`) directly from the
   token checkpoint via `from_token_ckpt_as_flow`, with `o_embed`
   zero-initialized so \(v(\cdot, o)\) is identical across conditions at
   birth. The actor is pretrained with `cfgrl_actor_step` (success labels,
   condition dropout 0.1, reference-action dropout 0.5), so the
   o-conditioned head the online run inherits is already trained in the same
   reference regime the online phase uses.

**Collect data** (`runs/rlt_pretrain_house0_kettle/`):

| Item | Value |
| --- | --- |
| Trajectories | **1200** (8 shards × 150 valid episodes) |
| Task | house0 `pick up the kettle.` on `house0_kettle_v13/train` (**episode 0 / pose 0 only**, matching V20 online) |
| Policy | frozen MolmoAct2 reference VLA (`actor_mode=vla_only`, 0 RLT updates) |
| Horizon / chunk | 400 / 8 |
| Success predicate | grasp+lift held for **5 consecutive policy steps** |
| Replay | `token_replay_merged.npz`, `chunk_replay_merged.npz` |

**Architecture handoff.** `MolmoAct2RLTCF.save` persists all architectural
parameters (`hidden`, `n_hidden_actor`, `n_hidden_critic`, `z_expand_dim`,
`layernorm_heads`, `use_cfgrl`, `cfgrl_o_dim`, `cfgrl_w`), and
`prepare_cfgrl_model` validates the loaded checkpoint against the V20 CLI
architecture. Because the flow pretrain is born-CFGRL at that architecture,
the handoff is a pure load: every pretrained weight (token encoder, actor,
critic) is carried into the online run. `as_cfgrl` fails loudly
(`RuntimeError`) if a checkpoint's architecture ever prevents a full
actor/critic transfer, so a silent re-initialization cannot recur.

### Serving and collection (every online / probe episode)

Each GPU HTTP server loads the frozen encoder from the flow pretrain
checkpoint (`FLOW_CKPT`). `feature_mode=rl_token`. For every observation:

1. Frozen Molmo VLM produces token embeddings.
2. Frozen \(g_\phi\) encodes them with `encode_z(..., detach=True)` → \(z_{\mathrm{rl}}\).
3. The worker stores that \(z_{\mathrm{rl}}\) (and \(next\_z\)) in the episode
   NPZ. No token optimizer, no reconstruction loss, no encoder gradient.

After an offline AE reload the **encoder weights stay the same**. Because the
VLM is frozen and \(g_\phi\) reads VLM token embeddings (not AE internals),
the \(z\) coordinates are invariant to AE LoRA updates.

When a server is (re)started with an AE LoRA checkpoint, `serve.py` first
reconstructs the PEFT wrapper from the `lora_config` stored in the checkpoint
metadata (`apply_ae_lora_to_model`), then strictly loads the trainable weights
(`load_ae_trainable_state`), freezes all parameters, and switches to eval
mode.

### Online RLT phase

`prepare_cfgrl_model` calls `freeze_token_encoder()`: `token_ae` has
`requires_grad=False` and `tune_token_online=False`. Every critic TD step and
CFGRL actor step reads **precomputed** `z` / `next_z` from `ChunkReplay` with
stop-grad (`_batch_state(..., detach_token=True)`). `token_step` is never
called. If a frozen token parameter receives a gradient, training fails closed
(`_assert_frozen_token_and_adapter_grads`).

What trains online on top of \(z_{\mathrm{rl}}\) is the `z_expand` adapter,
the flow actor, and the TimeCQL critic.

### Offline Molmo AE phase

The token encoder stays frozen. `ae_cfgrl_actor_step` uses stored `z` from
`ImageChunkReplay` with `detach_token=True`. Images are consumed only to train
the Molmo AE LoRA; they are not used to re-fit \(g_\phi\). The VLM stays
frozen.

## Online phase: central RLT updates

Online collection is single-pose: every worker runs benchmark episode 0
(target pose 0) with its own per-wave seed. Pretrain collect uses the same
episode ID, so the seed replay is also pose 0. Replay stratification still
over-samples target-pose (pose-0) positives on the 50% → 75% → 90% schedule
described under [Replay provenance](#replay-provenance).

Each worker episode produces one trajectory of chunks:

- `z`, `proprio`, `reference_actions` (frozen served reference), 
  `executed_actions`, `rewards`, `action_mask`, `success`, `pose_idx`,
  `worker_id`, `round_id`, `policy_version`.

Workers append episodes to `replay/journal/worker_*/episode_*.npz` and publish
a journal row. The learner imports completed journal entries into the pooled
replay after each wave; import is idempotent by `trajectory_uid`.

After each import wave, the learner runs a budgeted update
(`_run_online_update`): up to `updates_per_wave` (default **32**) critic+actor
step pairs, each on a fresh stratified batch, stopping early at the
`max_update_sec_per_wave` wall-clock cap (default **120 s**):

1. **Critic TD step** (`flow_critic_td_step`). The critic is an ensemble
   TimeCQL (\(K=10\) heads) over chunk actions, trained with TD + MC + CQL +
   ranking losses. Target critics are soft-updated (\(\tau=0.005\)). The TD
   bootstrap samples next-chunk actions from the current actor at \(w=1\)
   with **reference-action dropout**: for each row the served reference is
   zeroed with probability 0.5 (`reference_present`) before sampling, and
   small target noise (0.02) is added.
2. **Actor CFGRL step** (`cfgrl_actor_step`). Labels come from
   `cfgrl_condition_and_target`: while the critic is unhealthy,
   episode-success labels are used (CFGRL-GCBC) — conditional rows
   (\(o=\text{POS}\)) flow-match toward their executed action, all other rows
   are unconditional and flow-match toward the normalized reference `ã`.
   Once the critic passes its health check, labels switch to advantage
   \(A(x,a)=Q(x,a)-Q(x,\tilde a)\) with \(o=\text{POS}\) iff \(A\ge 0\).
   Failed chunks map to unconditional rather than a dedicated negative head,
   so the shared backbone is not trained mostly on unsuccessful actions.
   Condition dropout (`--cfgrl_dropout`, default **0.1**) retargets a fraction
   of conditional rows to unconditional, giving the unconditional head state
   coverage beyond failure rows. **Reference-action dropout**
   (`--ref_dropout`, default **0.5**) zeroes the reference input on a random
   half of the actor-training rows — the RLT paper's anti-copying mechanism —
   so the actor keeps a reference-free generation pathway and the critic's
   bootstrap (which samples in the same 50/50 regime) stays in-distribution.

The actor optimizer owns the flow actor and the `z_expand` adapter; the critic
optimizer owns the critic ensemble. Round reports log `critic_td`
(`q_td_loss`), `q_mean`, `q_target`, `q_rank_gap`, `critic_healthy`,
`actor_loss`, `cfgrl_pos_frac`, and the realized update count/seconds.

## Evaluation and promotion

After the last wave of a round, the learner publishes the **candidate**
checkpoint with the deployment guidance weight baked in
(`model.cfgrl_w = w_deploy`, default **1.0**), then runs a **paired probe**:

- Every worker runs **two** episodes — one with the candidate, one with the
  current incumbent — using the same paired seed (`pair_seed`), so both
  rollouts share initial conditions and outcomes are paired per worker. All
  workers are pinned to the same benchmark episode (`--target_pose_idx 0`,
  `--benchmark_episode_idx 0`); pairing varies the seed, not the pose.
  Promotion decisions are therefore specific to the target pose, and the
  24-pose benchmark only enters through the seed/pretrain data.
- The probe reports `probe_sr` (candidate) and `incumbent_sr` side by side.

After the probe, worker 0 rolls **one held-out validation episode** on
`house0_kettle_v13/val` episode 0 (`--val_benchmark_dir`,
`--val_episode_idx 0`) with the candidate at \(w_{\mathrm{deploy}}\) and
writes `videos/round_{RRR}_candidate.mp4` plus a JSON sidecar (success,
checkpoint, seed). After each offline AE phase that actually updates the
served reference, the same worker saves
`videos/round_{RRR}_offline_ae.mp4` of the new reference policy. Other
workers skip the video phase so the 32-way barrier still completes. On
resume, missing videos for already-completed rounds are recorded before
the next unfinished round starts.

The promotion gate (`_maybe_promote` → `decide_promotion`) combines:

- **McNemar's exact test** (one-sided) on the paired outcomes (discordant
  pairs candidate-only vs incumbent-only). Because one test runs per round
  and a run has at most `--max_rounds` rounds, each test uses the
  Bonferroni-corrected level `--promotion_alpha / max_rounds` (nominal
  default 0.05), controlling the family-wise false-promotion rate over the
  whole run; both alphas are logged per decision;
- a **gain check**: the point estimate of the paired success-rate gain must
  clear `--promotion_min_gain` (default 0.03), and the conservative
  Clopper–Pearson gain lower bound (`paired_gain_lower_bound`, computed at
  the corrected alpha) must be positive;
- **candidate diagnostics** (`_diagnose_candidate`): the **unconditional
  head's** clone MSE against the reference (`uncond_ref_mse`, sampled at
  \(w=0\)) must stay below `--clone_mse_max` (0.02) — the CFG anchor stays
  meaningful — and the **deployed conditional head's** MSE against the
  reference (`cond_ref_mse`, sampled at \(w=1\)) must stay below
  `--cond_ref_mse_max` (0.5, a loose excursion cap), so the policy that is
  actually deployed is gated directly. Normalized actions of both heads must
  additionally stay within `--max_normalized_action` (12.0).

On promotion the candidate is copied to `incumbent_v{N+1}.pt`, the run state
is republished with `incumbent_mode="actor"`, and subsequent collection waves
roll the new incumbent — the loop is genuinely online: better actors earn
control of the data stream. All decisions are appended to
`reports/promotions.jsonl` with the full paired counts, p-value, gain bound,
and diagnostics.

**Success predicate.** An episode counts as successful only when the
grasp+lift condition (object in contact only with the robot, lifted above
`succ_pos_threshold`) holds for `succ_hold_steps` **consecutive policy
steps** (5 for the kettle benchmarks, configured per-episode in the benchmark
JSON and copied into `PickTaskConfig`). A momentary contact no longer
terminates the episode as a success.

## Offline phase: Molmo AE update (LoRA-as-CFG)

After `rounds_per_offline` online rounds, V20 enters an offline phase:

1. Freeze the current RLT actor and critic. The RL token encoder is already
   frozen and is not updated in this phase.
2. Train on the collected **online** image trajectories
   (`replay/image_pooled.npz`). The seed/pretrain replay contains no images,
   so it does not contribute to this phase; image data accumulates from V20
   worker journals only. Stored `z` / `next_z` from the frozen encoder are
   used as-is; the pooled replay is not re-encoded.
3. Update the Molmo action expert LoRA adapters with **CFGRL flow matching**
   (`ae_cfgrl_actor_step`). The frozen base AE is the unconditional policy
   (the reference \(\tilde a\)); base+LoRA is the positive-conditioned
   policy. Each step sub-selects the **successful rows** of the batch and
   applies the standard conditional flow-matching loss toward the executed
   actions \(a_{\text{data}}\):
   \[
   \mathcal{L}_{\mathrm{AE}}
   =
   \mathbb{E}_{(x, a)\,\text{success},\, t}
   \big\|
     v_{\text{base+LoRA}}(x_t, t \mid x)
     - \big(a_{\text{data}} - a_{\text{source}}\big)
   \big\|^2,
   \qquad
   x_t = (1-t)\,a_{\text{source}} + t\,a_{\text{data}} .
   \]
   The source \(a_{\text{source}}\) is the **recorded** collection-time AE
   noise (`source_native` stored in the image replay), not a fresh Gaussian
   draw. The (source, action) pairs are therefore coupled — the recorded
   noise generated the reference \(\tilde a\) that the executed action was
   conditioned on. This is the rectified-flow / I-CFM setting: the regressed
   marginal velocity field still transports the Gaussian source to the
   executed-action marginal, but it is a deliberate departure from
   fresh-noise conditional flow matching and is recorded here for
   provenance.
   There is no Q term and no backpropagation through ODE integration — one
   velocity evaluation per row — so memory stays flat and the RLT critic
   never backprops into the AE. Steps with no positive row in the batch are
   skipped and counted. A diagnostic (`ae_lora_delta_mse`) measures
   \(\|v_{\text{lora}} - v_{\text{base}}\|^2\), the CFG delta a deployment
   weight would scale.
4. **Accumulation.** With `--ae_accumulate` (default on), each offline phase
   starts from `ae_trainable_latest.pt`, so LoRA improves iteratively across
   phases instead of retraining from the base AE.
5. After the update, the learner writes `ae_trainable_r{round}.pt`, copies it
   to `ae_trainable_latest.pt`, and signals the launcher via
   `coordination/ae_reload_request.json`; the launcher restarts the per-GPU
   HTTP servers with the updated AE, so the **served reference policy** for
   the next online phase is the improved one. Note the served AE is the
   PEFT model with LoRA **enabled** (`serve.py` `predict`), i.e. the
   composition \(v = v_{\text{base}} + w \cdot (v_{\text{lora}} -
   v_{\text{base}})\) is only ever realized at \(w=1\); there is no
   serve-time guidance-weight knob for the AE. (`predict_reference` disables
   the adapter to expose the \(w=0\) base AE for reference logging.)

The offline phase is not a separate run; it is a mode switch inside the same
learner. The RLT actor and critic are not updated during the offline AE phase.

## Replay provenance

Every chunk stores:

- `trajectory_uid`: globally unique episode identity;
- `pose_idx`: explicit benchmark pose identity;
- `source_policy`: offline reference, online reference, online actor, or
  offline AE;
- `worker_id`, `round_id`, and `policy_version`.

Legacy `episode_id` remains for compatibility but is not used to infer pose or
trajectory identity. `merge_chunk_replay_provenance.py` rebuilds the seed
replay from the eight original shard files before global IDs are assigned,
fails closed unless it observes the expected pose-0 counts, and — after
validating identical row order and action content — copies `z` and `next_z`
from the re-encoded pretrain replay, so seed state coordinates are compatible
with the frozen encoder in the flow checkpoint.

Workers first write one immutable episode NPZ, then atomically reserve its UID
in `replay/journal_index`, and finally publish their wave result. The learner
imports only completed-wave journals. Re-import is idempotent by UID, and the
pooled replay is atomically replaced after each wave.

Conditional CFGRL samples are selected trajectory-first and
temporal-bin-first, which prevents long successful trajectories from
dominating. Positives use a target-pose fraction of 50%, rising to 75% after
16 and 90% after 32 target-pose successes. Other-pose successes remain
anchors.

## Coordination and resume

`round_state.json` is atomically replaced and carries a monotonically
increasing generation plus a unique operation ID. Each worker holds an
operation-specific `flock`, writes one atomic done record, and ignores already
completed work after restart. The learner's barrier accepts only matching
operation IDs.

Episode UIDs are allocated under a lock. Journal UID reservation uses
exclusive file creation. Checkpoint, replay, state, and report writes use
same-directory temporary files followed by `os.replace`.

The launcher restarts failed worker processes, and restarts the per-GPU
servers when `coordination/ae_reload_request.json` is updated (serving the
latest AE trainable checkpoint). A resumed learner reuses completed wave
records, imported UIDs, and the latest actor/critic checkpoint. Stale workers
cannot publish into a newer wave.

## Theoretical notes

- **Unconditional head vs. behavior marginal.** CFGRL's product-policy
  sampler interprets guidance as reweighting by \(p(o{=}1\mid x,a)\), which
  requires the unconditional policy to approximate the behavior marginal
  \(p(a\mid x)\) — in the CFGRL paper the unconditional policy is trained on
  the *same dataset actions* as the conditional one. V20 instead trains the
  uncond head toward the stored reference actions `ã` (including
  advantage-negative rows and dropout-retargeted rows, whose targets are
  `ã`, not the executed action). While collection runs the frozen reference
  this is exact (executed = served); once a promoted actor collects, the
  uncond head is a **reference-prior head**, not the behavior marginal, and
  intermediate-\(w\) composition implements "guidance away from the served
  reference" rather than CFGRL's optimality reweighting. At the deployed
  \(w=1\) the uncond head is unused in sampling; its practical roles are the
  promotion clone gate and the option of intermediate-\(w\) probes.
- **Reference-action dropout matches the RLT paper.** Dropout is applied
  *during actor training* (`--ref_dropout`, 0.5): a random half of the rows
  zero the reference input, so the actor keeps an independent
  action-generation pathway instead of copying \(\tilde a\). The critic's TD
  bootstrap samples next actions in the same 50/50 regime, so bootstrap
  evaluation is in-distribution for the actor. The CFGRL objective provides a
  second anti-copying force, since the conditional head is regressed toward
  positive *executed* actions rather than toward \(\tilde a\).
- **Both CFG heads are gated.** `clone_ok` thresholds `uncond_ref_mse` (the
  \(w=0\) anchor) and `cond_ok` thresholds `cond_ref_mse` (the deployed
  \(w=1\) head, loose cap `--cond_ref_mse_max` 0.5). The anchor gate keeps
  the CFG composition meaningful; the cond gate bounds the deployed policy's
  excursion from the reference without blocking genuine improvement.
- **Multiple-testing correction.** One paired promotion test runs per round;
  with at most `--max_rounds` tests per run, each uses the Bonferroni level
  `--promotion_alpha / max_rounds`, controlling the family-wise
  false-promotion rate. (The look-based alpha-spending machinery in
  `v20_harness.py` belongs to a different confirm-at-looks protocol and
  remains unused.) The monotone-incumbent design further bounds the damage
  of any single wrong promotion — a promoted actor becomes the baseline for
  the next paired probe.
- **GCBC labels are episode-level.** With success labels, every chunk of a
  successful episode is \(o=\text{POS}\), including non-causal early chunks —
  the standard GCBC caveat. The advantage-label switch is the remedy and
  activates automatically once the critic passes its health check.
- **Deployment weight.** Improvement over the reference exists only for
  \(w>0\); candidates and incumbents are therefore saved and probed with
  \(w = w_{\text{deploy}} = 1.0\), the policy that would actually be deployed.
- **Reference drift across offline phases.** After an AE reload, new
  collection comes from a different reference policy than the rows already in
  the replay. The \(z\) coordinates stay consistent (frozen VLM + frozen
  \(g_\phi\)), but `reference_actions` and the behavior distribution shift;
  the critic and the uncond head mix references across rounds. Provenance
  (`source_policy`, `policy_version`) is stored, so per-round analysis is
  possible, but the training loss does not distinguish them.

## Launch and stopping rules

Fresh launch:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 FRESH=1 bash launch_v20_rlt_cfgrl.sh
```

Resume:

```bash
FRESH=0 bash launch_v20_rlt_cfgrl.sh
```

Key knobs (env / CLI): `V20_UPDATES_PER_WAVE` and
`V20_MAX_UPDATE_SEC_PER_WAVE` (per-wave gradient budget), `V20_W_DEPLOY`
(deployment guidance weight), `V20_CFGRL_DROPOUT` (condition dropout),
`V20_REF_DROPOUT` (reference-action dropout), `V20_PROMOTION_ALPHA` /
`V20_PROMOTION_MIN_GAIN` / `V20_CLONE_MSE_MAX` / `V20_COND_REF_MSE_MAX` /
`V20_MAX_NORMALIZED_ACTION` (promotion gate), `V20_AE_ACCUMULATE` (LoRA
accumulation across offline phases), `V20_ROUNDS_PER_OFFLINE` (offline
schedule).

The run stops after **100** online rounds (`V20_MAX_ROUNDS`, default 100), a
configured number of offline AE phases, an explicit signal, or a fatal
invariant failure. A
normal learner exit publishes a stop phase so workers terminate.

**No mocks in the real path.** The `--mock` flag exists only for the
self-contained `mock-smoke` subcommand (synthetic trajectories that exercise
the promotion machinery end-to-end); the launcher above never sets it, and
the learner fails closed if the CLI mock flag and the run config disagree.
All online rollouts execute the real MolmoSpaces environment and the real
served VLA; the only synthetic data path is the smoke test.
