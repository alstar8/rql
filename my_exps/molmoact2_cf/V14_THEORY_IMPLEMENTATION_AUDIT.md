# V14 Theory and Implementation Audit

V14 is a clean successor to the immutable V13 controlled run.  V13
checkpoints, replay files, snapshots, manifests, and validation outputs remain
read-only.  V14 keeps the same house-0 kettle train/validation factorization so
policy changes are tested on identical controlled poses.

## Falsifiable invariants

### Evaluation

1. Training collection success rate is diagnostic only.
2. Policy efficacy is measured on ep400 held-out pose/seed cells.
3. `actor` is compared with the frozen-reference `checkpoint_gate` cell.
4. `actor_guide` is compared with `actor` using the same pose and seed.
5. V13 AE forced-policy cells are invalid by construction and are never treated
   as negative efficacy evidence.

### Molmo action-expert coordinates

1. Molmo source states, velocities, BC targets, critic flow states, and guide
   corrections use the native `franka_droid` q01-q99-normalized coordinates.
2. Raw robot actions exist only at replay and environment boundaries.
3. With guide `G=0`, custom Euler integration must match native Molmo integration
   for identical context, source seed, number of steps, action mask, and adapter
   state.
4. A non-finite or shape-invalid native/raw action is a fatal error; there is no
   silent fallback.

### Gradient and cache ownership

1. The VLM and its KV output stay frozen and detached.
2. Action-expert context K/V projections are created with gradients when LoRA is
   trained.
3. Modulation caches are invalidated on adapter transitions, checkpoint load,
   and optimizer updates.
4. Base-then-current and current-then-base call orders produce the same
   same-source outputs.

### Randomness and resume

1. Rollout source noise and exploration use explicit episode/chunk counters.
2. Frozen/current/guided comparisons share source noise.
3. Episode `e + 24` does not replay episode `e` noise.
4. Optimizer, replay sampler, and process RNG state are restored with model and
   AE checkpoints.

### Replay

1. A long failure tail cannot evict every successful AE transition.
2. AE RL updates are disabled until both outcomes are present.
3. Save/load preserves row order, outcome composition, and sampler state.
4. V14 retains at least one full 24-pose cycle and records host-memory use.

### Critic and deployment

1. Every critic head receives ranking gradients.
2. Actor/guide objectives use a recorded robust lower-tail aggregate; hard
   minimum remains diagnostic only.
3. Health covers per-head action sensitivity, saturation, outcome separation,
   and minimum-head ownership.
4. A dominant dead minimum head blocks deployment.
5. Actor-vs-reference and guide-vs-actor gate decisions are independent.
6. Critic bootstrap actions use an explicit target policy.
7. The advantage and sensitivity thresholds remain `0.003`.

## Required evidence

- Focused CPU/static tests and eight-arm wiring probes.
- Real-checkpoint GPU parity, cache-order, gradient, and raw-output probes.
- A short V14 smoke satisfying every launch acceptance criterion.
- The eight-arm 400-episode controlled matrix and ep400 held-out evaluation.
- Paired confidence intervals, exact McNemar evidence, and per-pose outcomes.

## Evidence index

### V13 ep400 policy baseline

- Completed safe cells: 56/56 original cells, 672 held-out rollouts.  The
  matched noisy-reference controls are tracked separately and expand the final
  baseline to 64 cells.
- Report:
  `runs/rlt_cf_v13_controlled/plots/v13_paired_policy_report.json`.
- Residual actor vs frozen reference: -0.2292 paired SR
  (95% paired interval [-0.3627, -0.0956], McNemar p=0.00342).
- Flow actor vs frozen reference: -0.2083 paired SR
  (95% paired interval [-0.3244, -0.0922], McNemar p=0.00195).
- Guide-vs-actor deltas were statistically unresolved.  This evidence forbids
  lowering the gate thresholds or treating training SR as efficacy.

### Real Molmo AE contract probe

Command:

```bash
CUDA_VISIBLE_DEVICES=7 ../../../molmoact2/.venv/bin/python \
  probe_v14_ae_contract.py --device cuda:0 --lora-rank 2 \
  --output /tmp/v14_ae_contract_probe.json
```

Verdict: **VERIFIED**.

- Native/custom same-source `G=0` endpoint max absolute delta: `0.0`.
- Native action round-trip max absolute delta: `7.8124e-4` (bf16 boundary).
- Base and current adapter call-order deltas: `0.0`.
- Adapter/base treatment delta: `0.015625`.
- All 584 intended trainable LoRA tensors had finite, nonzero gradients;
  context K and V projections each contributed two tensors.
- Contract: 15x8 native q01-q99 actions, padded to 15x32, one observation
  step, 15 deployment actions.

### Frozen-checkpoint critic aggregate probe

Command:

```bash
../../../molmospaces/.venv/bin/python probe_v14_critic_aggregate.py \
  --run-dir runs/rlt_cf_v13_controlled \
  --output /tmp/v14_critic_aggregate_probe.json
```

Verdict: **VERIFIED** for the diagnosed hard-min failure.

- The AE actor critic's dead head owned the hard minimum for 100% of sampled
  transitions; hard-min action sensitivity was `7.81e-9`.
- The AE-CF critic reproduced the same 100% ownership failure with
  `4.87e-14` sensitivity.
- A bottom-three mean (fraction 0.25, minimum two heads) restored nonzero
  sensitivity and success/failure separation while remaining pessimistic.
  This aggregate is fixed for V14 actor, guide, target, and gate paths; hard
  minimum is diagnostic only.

### Static, unit, and eight-arm wiring verification

Verdict: **VERIFIED**.

- The focused CPU suite completed with `122 passed, 1 skipped`; the post-smoke
  regression subset completed with `60 passed`.
- Python compilation, shell syntax checks, `git diff --check`, and IDE lint
  diagnostics were clean.
- `validate_v13_wiring.py --output runs/v14_wiring_report.json` reported
  `VERIFIED` for all eight module/optimizer/gradient wirings.
- The V14 launcher manifest records SHA-256 hashes for the learning code,
  benchmark files, initial checkpoints, exact commands, GPU UUIDs, and all
  fixed hyperparameters.

### Real AE update stress smoke

Verdict: **VERIFIED** for the complete update path.

The two-episode stress smoke temporarily used one successful episode as the
update warm-up criterion so that the real AE update path could be exercised
before the production run. Production keeps the required three-success
criterion.

- All eight arms completed two controlled episodes with no fatal errors.
- Both AE arms ran batch-16 full-endpoint actor updates as four microbatches.
- AE actor: 8 critic updates and 3 actor updates; total LoRA gradient norm
  `0.0393`.
- AE-CF: 6 critic updates, 3 actor updates, and 2 guide updates; total LoRA
  gradient norm `0.0779`.
- AE-CF critic-gradient minimum norm was `0.00707`, with nonzero fraction
  `1.0`; the guide was therefore not trained from a dead gradient.
- Critic statistics were exactly identity (`mean_abs_max=0`, `std_min=1`,
  `std_max=1`) in native Molmo q01-q99 coordinates.
- Source padded-coordinate max magnitude was `0.0`.
- Peak observed AE memory was about 25.7 GiB; observed GPU utilization reached
  69%.
- Immutable summary: `runs/v14_stress_smoke_status.json`.

The stress smoke also exposed and fixed three last-mile defects before the
production launch:

1. Compact AE replay was already normalized and was being normalized a second
   time by the endpoint critic.
2. The shared stop/status tools recognized only the legacy V4 ownership
   marker, so they refused V14-owned processes.
3. A completed logging cadence was persisted a second time during normal
   shutdown.

### Full controlled V14 matrix

Status as of the first persisted cadence: **RUNNING, HEALTHY**.

- Run: `runs/rlt_cf_v14_controlled`.
- Manifest mode: 400 episodes, 250,000-step ceiling, snapshots at episodes
  0/100/200/400.
- Production AE settings: batch 16, microbatch 4, three successful episodes
  before updates, 30-second soft per-episode update budget.
- Eight trainers and six HTTP servers are owned by the V14 watchdog; the two
  AE arms use in-process Molmo backends.
- At episode 10, all eight metrics files had exactly one valid cadence row and
  no malformed rows. No fatal/non-finite/OOM errors had been observed.
- The AE arms retained 39 successful and 504 failed replay rows from two
  distinct successful episodes; training remained intentionally blocked until
  the third distinct successful episode.
- Status snapshot:
  `runs/rlt_cf_v14_controlled/status_snapshots/status_20260812T130928Z.json`.

The ep400 held-out efficacy verdict remains pending. Training SR and critic
predictions are not accepted as substitutes for the paired validation matrix.
