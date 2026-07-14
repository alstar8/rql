# DFL-RQL: RQL with a Linear Guidance Critic

Implementation of the "Dual-Flow Linear Actor-Critic (DFL-AC)" idea from
`chat-export-1782809854722.json`, corrected and adapted to the RQL codebase
(`agents/dflrql.py`). Compared against the tuned RQL baseline on
`humanoidmaze-large-navigate-singletask-v0`
(`cloud_job/run_train_rql_baseline.sh`, run group `humanoidmaze-large-rql-tuned`).

## Original proposal (from the chat)

1. **Synchronized dual-flow critic ("Value Flow").** The critic value `q^f`
   undergoes its own flow-matching ODE `dq^f/df = u_psi(q^f, x^f, s, f)`,
   synchronized with the actor's flow time, from a random prior `q^0` to the
   final Q-value `q^1`. Trained with a critic flow-matching loss plus a
   terminal Bellman loss.
2. **Linear guidance critic.** `Q_lin(s, x^f, f) = W_phi(s,f)^T x^f + B_phi(s,f)`,
   whose action gradient is analytically `W_phi(s,f)` (no backprop through a
   deep critic). The actor ODE becomes `dx^f/df = v_theta + lambda * W_phi`.
   `W` should be scheduled to vanish at `f=0` ("random action gets no push")
   and grow toward `f=1`.
3. Four losses: actor FM, critic FM, linear alignment, terminal Bellman.

## Corrections

1. **The separate value flow is dropped.** RQL's critic is already
   `Q(s, x^f, f)` — a value defined *along* the actor's flow time, trained at
   intermediate flow states produced by RQL's reversal mechanism, and anchored
   by an expectile TD target at the data action (`f=1`). This is exactly the
   "value that sharpens as the action forms". A second ODE over the scalar `q`
   with a random prior is ill-posed offline: its flow-matching target `q*` is
   the TD target itself, so the construction reduces to TD regression with
   extra integration cost and no extra learning signal. We therefore keep
   RQL's ensemble expectile critic as the value learner and implement only the
   novel component on top of it.
2. **`W` must be direction-normalized.** `Q_lin` regresses to Q-values, which
   on humanoidmaze-large (discount 0.995, rewards in [-1, 0]) are O(-100), so
   the raw least-squares `W` has magnitude far larger than the O(1) BC
   velocity `v_theta`; adding `lambda * W` directly (as written in the chat)
   pushes actions out of distribution immediately. We use only the direction:
   `g(s,f) = W(s,f) / (||W(s,f)|| + eps)`, and let the explicit time gate and
   `lambda` control speed. This preserves the original "direction + speed"
   intuition ("critic indicates + or - direction; schedule controls speed")
   while making the magnitude well-behaved.
3. **Explicit time gate.** Instead of hoping the network learns
   `||W|| -> 0` at `f=0`, we multiply the guidance by `f`:
   `guided velocity = v_theta(x^f,s,f) + lambda * f * g(s,f)`.
4. **Consistent dynamics everywhere.** The guided velocity is used in all
   three places the flow is integrated, so training matches inference:
   - inference sampling (`compute_flow_actions`),
   - RQL's reversal that constructs critic training states,
   - the one-step lookahead `q_pe` inside the actor loss.
5. **Alignment loss only for the linear head.** `W, B` are trained purely by
   regressing `Q_lin(s, x^t, t) = W^T x^t + B` to the (stop-gradient) mean of
   the target critic ensemble at the same interpolant points `x^t` used by the
   BC flow loss. Over many noise draws per state this makes `W` the amortized
   local linear fit of Q in action space, i.e. an averaged
   `grad_x Q(s, x^t, t)` — the guidance signal the chat wanted, obtained
   without per-step backprop through the deep critic. The linear head gets no
   Bellman loss of its own (correction of the chat's redundant fourth loss —
   the ensemble critic already carries the Bellman anchoring).
6. **Tuned hyperparameters.** The chat/README command uses `alpha=10,
   expectile=0.9`; the tuned humanoidmaze-large baseline uses `alpha=0.3,
   expectile=0.5, rho=0.0, h=1, discount=0.995`. We keep the tuned values so
   the only difference from the baseline is the guidance mechanism.

## Final method

Everything identical to RQL except:

- New module `guidance`: MLP `(s, f) -> (W in R^d, B in R)`.
- Guided velocity `v_theta + lambda * f * W/||W||` used in sampling, reversal,
  and the actor's `q_pe` lookahead (guidance stop-gradded in the actor loss so
  it does not shape `W`).
- New loss: `align_loss = E[(W^T x^t + B - mean_ensemble Q_target(s,x^t,t))^2]`,
  added with coefficient `align_coef`.

New hyperparameters: `guidance_coef` (lambda, default 1.0 — total extra
displacement over the flow is about `lambda * 0.5` in action L2 norm),
`align_coef` (default 1.0).

## Experiment

- Env: `humanoidmaze-large-navigate-singletask-v0`, 1M offline steps,
  batch 256, seeds 0..7 (one per GPU), same tuned hyperparameters as baseline.
- Launcher: `cloud_job/run_train_rql_dfl.sh` (tmux session `dflrql`).
- Baseline reference: `exp/rql_humanoidmaze-large-rql-tuned_seed*_gpu*.log`
  (seed 0 final eval: success 0.86).

## v1 results (8 seeds, 1M steps)

- Baseline final success: 0.725 +- 0.100; DFL-RQL v1: 0.738 +- 0.062
  (delta +0.013). Lower variance, but slower learning between 0.4M and 0.8M
  steps (see `my_exps/dflrql_vs_baseline.png`).
- Diagnosis: (a) `W` from the global linear fit of Q is a blunt gradient
  estimate — a single direction per (s, f) fit across all noise draws, which
  underfits when Q is not globally linear in the action; (b) guidance is
  active from training step 0, when `W` is still noise, dragging early
  learning below the baseline.

## v2: DFL-RQL2 (`agents/dflrql2.py`)

Two changes, targeting the two v1 weaknesses:

1. **Gradient distillation instead of the linear fit.** The guidance head
   `W(s, f)` (no bias output anymore) is trained to regress the
   unit-normalized action gradient of the target critic ensemble mean at the
   BC interpolant points:
   `distill_loss = E[ || W(s,t) - g/||g|| ||^2 ]`,
   `g = grad_x mean_ensemble Q_target(s, x_t, t)` (stop-gradded).
   This keeps the chat's core promise — no critic backprop during flow
   integration (the gradient is computed once per training batch, amortized
   into `W`) — while giving a local, exact steering signal instead of a
   global linear approximation. Logged `w_grad_cos` tracks how well `W`
   matches the true gradient direction.
2. **Guidance warm-up.** Guidance strength is multiplied by
   `ramp = clip(train_step / guidance_warmup_steps, 0, 1)` (default 100k of
   1M steps), so early training is exactly plain RQL while `W` is being
   distilled, and guidance blends in only once it is meaningful. The ramp is
   part of the agent state (`network.step`), so eval-time sampling uses the
   same schedule.

Guided velocity: `v_theta + guidance_coef * ramp * f * W/||W||`, used in
sampling, reversal, and the `q_pe` lookahead as in v1.

New hyperparameters: `distill_coef` (default 1.0, replaces `align_coef`),
`guidance_warmup_steps` (default 100000).

### v2 experiment

- Same protocol: humanoidmaze-large, 1M steps, batch 256, seeds 0..7, tuned
  hyperparameters.
- Launcher: `cloud_job/run_train_rql_dfl2.sh` (tmux session `dflrql2`),
  run group `humanoidmaze-large-dflrql2`.

## v2 results (8 seeds, 1M steps)

- Final success 0.680 +- 0.073 — worse than v1 (0.738) and baseline (0.725).
- The distillation itself worked: `w_grad_cos` stayed at 0.94-0.98 all run.
- The mid-training drag got worse, not better: at 700k, baseline 0.585 vs
  v1 0.465 vs v2 0.270 mean success. More accurate constant-magnitude
  guidance = more consistent drag off the data manifold.
- Late slopes (last 200k, per 100k steps): baseline +0.023 (plateaued),
  v1 +0.088, v2 +0.145 — v2 was still climbing steeply at the 1M cutoff.
  Guidance helps late (good critic), hurts early/mid (immature critic).

## v3: DFL-RQL3 (`agents/dflrql3.py`)

Diagnosis of v1/v2: unit-normalizing `W` makes guidance a *constant-force*
push, applied at full strength even where the critic is indifferent
(`grad_x Q ~ 0`) or where the BC action is already optimal. This breaks
stationarity — the flow endpoint is displaced by ~`coef * f` regardless of
critic preference — and permanently fights the BC term. v2's accurate
distillation aligned that constant force with an immature critic's gradient,
making mid-training worse.

Changes vs v2 (same conditions as baseline otherwise):

1. **Magnitude-aware distillation.** Target is the batch-relative raw
   gradient `g / mean_batch(||g||)` instead of the unit direction `g/||g||`.
   Dividing by the batch-mean norm keeps the target scale-stable across
   training (Q magnitude grows all run) while preserving *relative*
   magnitude: `W ~ 0` where the critic is locally flat, large where it
   strongly prefers a direction. Guidance at use is
   `coef * ramp * f * W` (no normalization; `||W||` capped at
   `guidance_norm_cap=3` for robustness). Stationarity is restored:
   if the critic is satisfied with the BC action, guidance vanishes.
2. **`guidance_coef` 1.0 -> 0.5** to further cut mid-training drag.
3. **`guidance_warmup_steps` 100k -> 300k**, matching when the critic
   becomes informative on humanoidmaze-large (baseline success starts
   moving ~300-400k).

New/changed hyperparameters: `guidance_coef=0.5`,
`guidance_warmup_steps=300000`, `guidance_norm_cap=3.0`.

### v3 experiment

- Same protocol: humanoidmaze-large, 1M steps, batch 256, seeds 0..7, tuned
  hyperparameters, same venv and eval protocol as baseline.
- Launcher: `cloud_job/run_train_rql_dfl3.sh` (tmux session `dflrql3`),
  run group `humanoidmaze-large-dflrql3`.

## v3 results (8 seeds, 1M steps)

- Final success 0.460 +- 0.151 — by far the worst variant
  (baseline 0.725, v1 0.738, v2 0.680).
- Monotonic pattern across versions: the more faithfully guidance follows
  the critic's raw gradient, the worse the outcome
  (v1 linear-fit direction > v2 exact unit direction > v3 magnitude-aware).
- Unit normalization in v1/v2 was an implicit *trust region*. v3 removed it:
  `w_norm_max` reached 4-5 (capped 3 at use), i.e. up to 3x stronger pushes
  precisely in high-gradient regions — and large action-gradients of a
  TD-learned critic are disproportionately overestimation artifacts.
  Magnitude-proportional pushing = uncontrolled gradient ascent on a learned
  Q, the failure mode offline RL methods specifically guard against.
- Structural point: RQL's actor loss already does policy improvement
  (`q_pe` backprops through the critic) balanced against BC by the tuned
  `alpha`. Additive guidance is a second, untuned improvement channel that
  shifts this equilibrium — harmful while the critic is immature.
- Positive finding stable across v1-v3: late-phase slopes (last 200k, per
  100k steps): baseline +0.023 (plateaus after 800k), v1 +0.088, v2 +0.145,
  v3 +0.116. Guidance reliably accelerates improvement once the critic is
  mature.

## v4: DFL-RQL4 (`agents/dflrql4.py`)

Keep guidance only where three experiments say it helps: the late phase.

Changes vs v2 (same conditions as baseline otherwise):

1. **Revert to v2's unit-direction distillation** — best-verified mechanism
   (cosine to true critic gradient 0.94-0.98), with normalization restored
   as the implicit trust region. v3's magnitude-aware variant is abandoned.
2. **Late-onset ramp replaces early warm-up:**
   `ramp = clip((step - 500k) / (700k - 500k), 0, 1)`.
   Training is bit-identical to plain RQL for the first 500k steps; guidance
   blends in linearly during 500k-700k and is fully active for the last 300k
   — exactly the phase where the baseline plateaus (+0.023/100k) and guided
   variants kept climbing (+0.09..0.15/100k).
3. **`guidance_coef` = 0.5** to soften the equilibrium shift when guidance
   engages mid-run.

New/changed hyperparameters: `guidance_coef=0.5`,
`guidance_ramp_start=500000`, `guidance_ramp_end=700000`
(replaces `guidance_warmup_steps`; no norm cap — unit direction).

### v4 experiment

- Same protocol: humanoidmaze-large, 1M steps, batch 256, seeds 0..7, tuned
  hyperparameters, same venv and eval protocol as baseline.
- Launcher: `cloud_job/run_train_rql_dfl4.sh` (tmux session `dflrql4`),
  run group `humanoidmaze-large-dflrql4`.

## v4 results (8 seeds, 1M steps)

- Final success 0.752 +- 0.066 — best variant so far
  (baseline 0.725, v1 0.738, v2 0.680, v3 0.460). Delta +0.028.
- Curve anatomy: identical to baseline through 400k (as designed), then an
  *engagement dip* while guidance blends in (-0.11 at 600k, -0.25 at 700k,
  -0.19 at 800k vs baseline), then steep recovery (+0.134/100k late slope,
  still climbing at the 1M cutoff).
- Second diagnostic: `w_grad_cos` *degrades* over training (0.976 -> 0.942)
  — distillation gets worse exactly as the critic matures.

## v5: DFL-RQL5 (`agents/dflrql5.py`)

Theory revision. Both v4 problems trace to one flaw: the guidance head
`W(s, f)` cannot see the current flow point `x`, while the true steering
signal `grad_x Q(s, x, f)` depends on it. Consequences:

- The distillation target is evaluated at random interpolant points `x_t`
  the head cannot condition on, so the best `W(s, f)` can learn is the
  conditional *mean* gradient over the noise distribution — irreducible
  regression error that grows as the critic sharpens and its gradient
  becomes more x-dependent (the declining cosine).
- At use, a constant-per-(s, f) push shifts every flow sample identically:
  samples on the wrong side of a mode are pushed further away, and
  multimodal action distributions are translated wholesale instead of
  steered per-sample. Plausible mechanism for the engagement dip: when
  guidance switches on, it is locally wrong for a large fraction of samples
  and the policy must re-equilibrate.

Change vs v4 (single change, to isolate the effect):

1. **x-conditioned guidance head** `W(s, x, f)` — same input layout as the
   actor. The distillation becomes pointwise well-posed: regress
   `W(s, x_t, t)` onto unit `grad_x Q_target(s, x_t, t)` at the same points
   (a deterministic function of the inputs). Inference cost unchanged: one
   extra forward pass per flow step, no critic backprop during integration.

Kept from v4: unit-direction trust region, `guidance_coef=0.5`, late-onset
ramp 500k -> 700k, and all baseline conditions.

### v5 experiment

- Same protocol: humanoidmaze-large, 1M steps, batch 256, seeds 0..7, tuned
  hyperparameters, same venv and eval protocol as baseline.
- Launcher: `cloud_job/run_train_rql_dfl5.sh` (tmux session `dflrql5`),
  run group `humanoidmaze-large-dflrql5`.

## v5 results (8 seeds, 1M steps)

- Final success 0.778 +- 0.067 — best variant
  (baseline 0.725, v1 0.738, v2 0.680, v3 0.460, v4 0.752). Delta +0.053.
  Best return too (-1644 vs baseline -1674). Seed 6 reached 0.90.
- x-conditioning worked as predicted: smaller engagement dip than v4
  (0.41 vs 0.33 mean success at 700k) and stronger late recovery
  (0.78 vs 0.75 at 1M; late slope +0.121/100k, still climbing at cutoff).

## v6: DFL-RQL6 (`agents/dflrql6.py`) — schedule-free, transferable

Goal: remove the two remaining non-transferable elements of v5 with one
mathematically grounded object. The step ramp (500k/700k) is tuned to this
environment's learning curve, and unit-normalizing W discards all confidence
information.

**Ensemble consensus vector.** For the K target-critic ensemble members with
action gradients `g_k = grad_x Q_k(s, x, f)`, define

    T(s, x, f) = (1/K) sum_k g_k / ||g_k||,      ||T|| <= 1.

- Direction of T = the ensemble's average improvement direction.
- Norm of T = directional agreement, dimensionless and in [0, 1]:
  * at initialization members are independent, gradient directions are
    near-independent random unit vectors -> E||T||^2 ~ 1/K (weak guidance
    exactly while the critic is untrustworthy);
  * as TD training aligns members, ||T|| -> 1 where they agree.

This subsumes both v5 mechanisms in a transferable way: maturity gating is
emergent (per state, flow point, and time — not a global step schedule), and
the trust region is the bound ||T|| <= 1 (magnitude = confidence, not raw
gradient size — avoiding v3's overestimation-chasing). Where ensemble
members disagree (OOD actions), guidance vanishes: a principled pessimism
mechanism.

**Single-backward distillation.** Regressing W(s, x_t, t) onto
`unit(g_k)` for one uniformly sampled member k per batch element has
conditional mean exactly T, so the MSE-optimal head converges to the
consensus vector with a single backward pass per training step
(amortization performs the ensemble average). Implemented as one VJP with a
one-hot member cotangent. No critic backprop during flow integration.

**Guided dynamics** (sampling, reversal, q_pe lookahead):

    dx/df = v_theta(x, s, f) + guidance_coef * f * Proj_{||.||<=1} W(s, x, f)

The projection enforces the range of T, not a tuned cap. Removed
hyperparameters: `guidance_ramp_start`, `guidance_ramp_end`,
`guidance_norm_cap`. Remaining: `guidance_coef=0.5`, `distill_coef=1.0`
(both dimensionless, carried unchanged from verified versions).

### v6 experiment (2M steps)

- humanoidmaze-large, **2M offline steps** (v1-v5 late slopes suggested 1M
  truncates guided variants), batch 256, seeds 0..7, tuned hyperparameters,
  same venv and eval protocol as baseline.
- Fair comparison: baseline rerun for 2M steps in parallel
  (run group `humanoidmaze-large-rql-tuned-2m`), both sharing the 8 GPUs
  (2 processes per GPU).
- Launcher: `cloud_job/run_train_rql_dfl6.sh` (tmux session `dflrql6`),
  run group `humanoidmaze-large-dflrql6`; baseline via
  `RQL_OFFLINE_STEPS=2000000 RQL_RUN_GROUP=humanoidmaze-large-rql-tuned-2m
  bash cloud_job/run_train_rql_baseline.sh` (tmux session `rql_baseline_2m`).

## v6 results (8 seeds, 2M steps) — clear win

| method | final success | return |
|---|---|---|
| RQL baseline (2M) | 0.630 +- 0.075 | -1667 +- 53 |
| **DFL-RQL v6 (2M)** | **0.882 +- 0.042** | **-1467 +- 38** |

- **Delta +0.252 vs 2M baseline; +0.157 vs the 1M baseline (0.725).**
  v6 wins on *all 8 seeds* (v6 range 0.82-0.96 vs baseline 0.50-0.72) and
  has the lowest seed variance of any variant.
- The 2M baseline *degrades* after 1M (0.725 -> 0.630, consistent with
  critic over-training/plateau), while v6 keeps improving monotonically —
  consensus gating turns the maturing ensemble into usable guidance instead
  of a liability.
- At the paper protocol (1M steps) v6 already leads: 0.818 +- 0.062 vs
  baseline 0.705 (+0.113) — the schedule-free gating engages earlier than
  the hardcoded 500k ramp of v4/v5 where the ensemble agrees earlier.
- Early-training prediction of the theory confirmed: `w_norm` at
  initialization settled at ~0.36 ~ 1/sqrt(K)=0.32 for K=10 (independent
  members), i.e. guidance self-attenuates without any schedule.

**Method summary (final):** RQL + a guidance head `W(s, x, f)` distilled
from the target-critic ensemble consensus vector
`T = (1/K) sum_k unit(grad_x Q_k)`, applied as
`dx/df = v_theta + guidance_coef * f * Proj_{||.||<=1} W` in all flow
integrations. Two dimensionless hyperparameters (`guidance_coef=0.5`,
`distill_coef=1.0`); no schedules; transfers across environments unchanged.

## v7: DFL-RQL7 (`agents/dflrql7.py`) — soft consensus + batch-relative uncertainty

**First attempt failed** (archived under
`exp/rql/_archive/humanoidmaze-large-dflrql7_broken_*`): finished-seed mean
~0.70 vs v6 0.82; aggregate with incomplete seeds looked ~0.50. Root causes
from TensorBoard:

1. `advantage_gate ≈ 0.50` for the whole run (`adv_delta_q ~ 1e-3` while
   `|Q|` is O(100)), so `sigmoid(β ΔQ)` permanently halved guidance —
   recreating v2's mid-training drag.
2. Absolute `std(Q)/|mean Q|` uncertainty gate stayed ≈0.997 (never fired
   on discounted maze returns).
3. Unguided reversal + `consensus_tau=0.1` weakened late guidance
   (`w_norm@1M ≈ 0.39` vs v6 `≈ 0.65`).

**Fixed v7** keeps only the transferable pieces that do not fight v6:

1. **Soft-consensus distillation** with small `consensus_tau=0.01`
   (near unit direction; `||W||` still learns agreement via the ensemble
   average of soft units).
2. **Batch-relative uncertainty gate**
   `clip(1 - κ (std(Q)/mean_batch(std(Q)) - 1), 0, 1)` with `κ=0.5` —
   attenuates only relatively uncertain batch elements; scale-free across
   reward magnitudes.
3. **Guided reversal restored** (same dynamics as inference / `q_pe`).
4. **Advantage sigmoid removed.**

Kept from v6: `guidance_coef=0.5`, `distill_coef=1.0`, schedule-free.

### v7 experiment (humanoidmaze-large, 1M)

- Same protocol as v1–v5: 1M steps, batch 256, seeds 0..7, tuned hparams.
- Launcher: `cloud_job/run_train_rql_dfl7.sh` (tmux session `dflrql7`),
  run group `humanoidmaze-large-dflrql7`.
- Plot: `scripts/plot_dflrql_vs_baseline.py` → `my_exps/dflrql_vs_baseline.png`.

## v6 on the full 50-task OGBench suite (paper protocol)

- All 10 environments from `hyperparameters.sh` x 5 singletask variants
  (`singletask-task1-v0` .. `singletask-task5-v0`) = 50 tasks.
- Paper protocol: 1M offline steps, batch 256, per-environment tuned
  hyperparameters from `hyperparameters.sh`, seed 0, 8 GPUs (task queue,
  ~6-7 sequential runs per GPU).
- Note: for puzzle-4x4 and cube-quadruple the paper uses 100M datasets
  (`--ogbench_dataset_dir`); those are not cached locally, so the standard
  play-v0 datasets are used — flagged for interpretation of those 10 tasks.
- Launcher: `cloud_job/run_train_rql_dfl6_ogbench50.sh`
  (tmux session `dflrql6_50`), run groups `ogbench50-dflrql6/<env>`.

## v8: DFL-RQL8 (`agents/dflrql8.py`) — scale-free consensus and behavior-conflict projection

The completed v7 50-task run improves the aggregate over both baseline and
v6, but its gains and regressions split by domain.  V7 is strongest on
puzzle-4x4 and cube-quadruple, while its value-uncertainty gate weakens the
guidance that made v6 strong on navigation.  The gate averages roughly 0.87
and requires a full target-critic ensemble forward at every flow integration
step, so it behaves more like a costly global attenuation than a selective
trust signal.

V8 keeps the transferable v6 consensus core and makes two task-independent
changes.

1. **Scale-invariant soft consensus.**  For the sampled ensemble-member
   gradient `g_k`, the target is

       g_k / (||g_k|| + c * mean_batch(||g_k||)),

   with dimensionless `c=0.1`.  Typical gradients remain near unit norm,
   locally flat gradients vanish, and uniform reward/Q rescaling leaves the
   target unchanged.  This preserves v7's useful stationarity behavior
   without its absolute `consensus_tau`.
2. **Behavior-conflict projection.**  If `u` is the distilled consensus and
   `v` is the behavior-flow velocity at the same point, apply

       u_safe = u - min(<u, v/||v||>, 0) * v/||v||.

   This is the minimum-L2 projection onto the half-space that does not oppose
   the learned behavior flow.  It removes only the conflicting component;
   aligned and orthogonal critic-improvement components retain their full
   strength.  Unlike a sigmoid or positive-cosine gate, it does not
   permanently halve generic guidance.

The resulting dynamics are used consistently in reversal, `q_pe`, and
sampling:

    dx/df = v_theta + guidance_coef * f * u_safe.

V8 removes v7's `consensus_tau`, `uncertainty_coef`, and repeated critic
forwards during integration.  It logs `q_grad_scale`,
`consensus_target_norm`, `behavior_alignment_cos`,
`behavior_conflict_fraction`, `guidance_retained`, and `safe_w_norm`.

Experiment order:

1. `cloud_job/run_train_rql_dfl8.sh`: humanoidmaze-large, 2M steps, seeds
   0–7; refresh `my_exps/dflrql_vs_baseline.png`.
2. `cloud_job/run_train_rql_dfl8_ogbench50.sh`: all 50 tasks, 2M steps,
   seeds 0,1,2; write `my_exps/ogbench50_v8_vs_baseline_common_max.png` and
   extend the all-method 50-task plots.
3. `cloud_job/run_train_rql_dfl8_pipeline.sh` enforces this sequence and only
   starts phase 2 after all phase-1 paper checkpoints exist.

## v8 results (paper @ 2M, seeds 0–2)

- Aggregate: baseline **0.520**, v6 **0.572**, v7 **0.581**, v8 **0.590**
  (Δv8−base **+0.070**); common_max Δ ≈ **+0.078**, seed std ≈ **±0.014**.
- Wins 29 / lose 6 / tie 15 vs baseline.  Worst regressions:
  `antmaze-giant-task2`, `puzzle-4x4-task5/3`, `humanoidmaze-large-task3/4`.
- Vs v7: loses on some puzzle/HL/cube-quad tasks; wins on others
  (puzzle-4x4-task4/2, antmaze-large-task2, scene-task2).
- Diagnostics: soft floor `c=0.1` under-drives `w_norm` (~0.50–0.58 vs v6
  ~0.60–0.69); hard conflict kill barely changes magnitude
  (`guidance_retained`~0.98–0.99) but blocks mature critics from overriding BC.

## v9: DFL-RQL9 (`agents/dflrql9.py`) — trust-weighted conflict + near-hard consensus

V9 inherits v8's scale-free soft consensus and behavior-aware dynamics, with
three task-agnostic fixes (no env schedules, no privileged eval signals, no
task-specific hyperparameters):

1. **Near-hard relative floor** `consensus_floor=0.01`.  Locally flat
   gradients still vanish; typical-batch targets stay near unit norm, so
   `||W||` recovers the v6 agreement signal without an absolute tau.
2. **Trust-weighted conflict projection.**  With `u = Proj_{||.||<=1}(W)`,
   `trust = ||u||`, and unit behavior velocity `v̂`,

       kill = (1 - trust^p) * min(<u, v̂>, 0)
       u_conf = u - kill * v̂

   Immature critics remain BC-safe as in v8; fully agreed ensembles may
   override BC as in v6.  Default `conflict_power=2`.
3. **Light residual damping** when BC and consensus already agree:

       damp = 1 - ρ * max(cos(u, v), 0) * trust
       u_safe = damp * Proj(u_conf)

   with `residual_coef=0.25`, so aligned directions are not double-pushed.

Shared RQL hyperparameters stay identical to the baseline comparison.
Logs add `conflict_kill_frac`, `residual_damp`, and `trust` on top of the
v8 diagnostic suite.

Experiment order:

1. `cloud_job/run_train_rql_dfl9.sh`: humanoidmaze-large, 2M steps, seeds
   0–7; refresh `my_exps/dflrql_vs_baseline.png`.
2. `cloud_job/run_train_rql_dfl9_ogbench50.sh`: all 50 tasks, 2M steps,
   seeds 0,1,2; write `my_exps/ogbench50_v9_vs_baseline_common_max.png` and
   extend the all-method 50-task plots.
3. `cloud_job/run_train_rql_dfl9_pipeline.sh` enforces this sequence and only
   starts phase 2 after all phase-1 paper checkpoints exist.
