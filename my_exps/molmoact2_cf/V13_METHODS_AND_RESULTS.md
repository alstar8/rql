# V13 methods and results

Status: **training complete, held-out eval complete** (56/56 safe cells, 672 rollouts).  
Run: `runs/rlt_cf_v13_controlled`  
Benchmark: `runs/benchmarks/house0_kettle_v13` (24 train poses, 12 held-out val poses)  
Report generated from: `plots/v13_paired_policy_report.json` (2026-08-12T11:09:47Z)  
This run is **immutable**. Do not overwrite it. Do not treat training success rate as policy efficacy.

## Protocol

Eight arms trained in parallel on eight H100s for **400 valid episodes** (horizon 500, target 250k env steps, train seed 20260813). Residual and flow arms query a frozen MolmoAct2 HTTP server; Molmo AE arms run the action expert in-process.

**Gate.** Learned actor (and guide, if present) deploys only if paired lower confidence bound ≥ `0.003` and action sensitivity ≥ `0.003`. Gate never opened on any V13 arm.

**Held-out eval.** Episode-400 snapshots, 12 val poses × 4 seeds (`20260831`–`20260834`), zero exploration.

| Policy | What it executes |
| --- | --- |
| `reference` | Frozen VLA canary (baselines only). |
| `checkpoint_gate` | Whatever the snapshot gate would deploy. Gate was closed, so this is the **frozen reference**. |
| `actor` | Forced learned actor, no guide. Counterfactual; not what training deployed. |
| `actor_guide` | Forced actor + consensus guide. Counterfactual. |

Molmo AE `actor` / `actor_guide` cells were **not** run: V13 AE weights were trained in raw robot coordinates while MolmoAct2 integrates in q01–q99 native coordinates. Those cells are invalid by construction.

## Shared algorithm

- **Residual family.** Action = frozen VLA chunk \(\tilde a\), plus optional residual \(\Delta_\pi\) and residual consensus field \(\Delta_g\). Actor: `ChunkGaussianActor`. Critic: `EnsembleCQL` (10 heads). Guide: `CFGradientGuide`.
- **Flow family.** Action = Euler endpoint of \(\dot x = v_\theta\) (and \(+G_\phi\) if CF). Actor: `FlowVelocityActor`. Critic: `EnsembleTimeCQL`. Guide: `FlowCFGuide`. \(G_\phi = \lambda t\,\mathrm{safe}(W_\phi; v)\) with \(\lambda=0.5\), 10 flow steps.
- **Molmo AE family.** Same flow critic/guide, but velocity is the Molmo action expert with LoRA (rank 16, alpha 32). Gate-off reference: same expert with adapters disabled.
- Consensus teacher: per-batch-element target-critic gradient, magnitude-preserving normalization. Raw student \(W_\phi\) is trained; the bounded field is used at deploy.
- Exploration during training: Gaussian std `0.02`. Validation: none.
- BC coefficient toward the frozen reference: `1.0`.

## The eight methods

### 1. `residual_vla_baseline` — residual frozen-VLA canary

- GPU 0, HTTP `:8700`. No online updates.
- Action: \(a=\tilde a_{\mathrm{VLA}}\).
- Role: stable residual-family reference.

### 2. `residual_rlt_actor` — residual actor

- GPU 1, HTTP `:8701`. Trainable: residual actor + critic. 8 updates/episode.
- Action if gate on: \(a=\tilde a+\Delta_\pi\). Gate stayed off, so training executed the frozen VLA plus exploration.

### 3. `residual_rlt_cf` — residual actor + consensus flow

- GPU 2, HTTP `:8702`. Trainable: actor, critic, `CFGradientGuide`.
- Action if gate on: \(a=\tilde a+\Delta_\pi+\Delta_g\).

### 4. `flow_vla_baseline` — flow frozen-VLA canary

- GPU 3, HTTP `:8703`. No online updates.
- Action: \(a=\tilde a_{\mathrm{VLA}}\). Should match arm 1 within paired variation.

### 5. `flow_rlt_actor` — lightweight flow actor

- GPU 4, HTTP `:8704`. Trainable: `FlowVelocityActor` + time critic.
- Action if gate on: endpoint of \(\dot x=v_\theta\).

### 6. `flow_rlt_cf` — lightweight flow actor + CF

- GPU 5, HTTP `:8705`. Trainable: flow actor, time critic, `FlowCFGuide`.
- Action if gate on: endpoint of \(\dot x=v_\theta+G_\phi\).

### 7. `molmo_ae_lora_actor` — Molmo action-expert LoRA (extension)

- GPU 6, in-process. Trainable: AE LoRA + AE-consistent time critic. 4 updates/episode.
- Not the frozen-action-expert CF_VLA claim. V13 implementation of this arm is **not a valid policy test** (wrong action coordinates, FIFO replay, dead critic head).

### 8. `molmo_ae_lora_cf` — Molmo AE LoRA + CF (extension)

- GPU 7, in-process. Trainable: AE LoRA, time critic, `FlowCFGuide`.
- Same invalid-by-construction caveat as arm 7 for forced actor/guide eval.

## Training results (400 episodes, not efficacy)

Training rollouts used exploration on RLT arms and a clean reference on VLA baselines, so these numbers are **not** comparable as policy quality.

| Arm | Episodes | Env steps | Cumulative SR | Last-100 window SR | Gate | Final paired LCB | Critic healthy |
| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| residual VLA | 400 | 172,378 | 0.238 | 0.32 | off by design | — | n/a |
| residual actor | 400 | 182,483 | 0.148 | 0.14 | never opened | −0.00049 | yes |
| residual CF | 400 | 180,691 | 0.163 | 0.12 | never opened | −0.00062 | yes |
| flow VLA | 400 | 172,676 | 0.230 | 0.20 | off by design | — | n/a |
| flow actor | 400 | 181,477 | 0.155 | 0.16 | never opened | −0.0108 | yes |
| flow CF | 400 | 181,615 | 0.153 | 0.12 | never opened | −0.0073 | yes |
| AE LoRA actor | 400 | 189,266 | 0.083 | 0.08 | never opened | 0.0 (dead) | **no** |
| AE LoRA CF | 400 | 189,266 | 0.083 | 0.08 | never opened | 0.0 (dead) | **no** |

Skipped episodes: 0 on every arm. Snapshots exist at episodes 0, 100, 200, 400.

## Held-out results (ep400, 12 poses × 4 seeds = 48 rollouts)

Wilson 95% intervals. `checkpoint_gate` is frozen-reference evaluation because the snapshot gate is closed.

| Arm | Policy | Successes | SR | 95% Wilson |
| --- | --- | ---: | ---: | --- |
| residual VLA | reference | 9/48 | 0.188 | [0.102, 0.319] |
| residual actor | checkpoint_gate (frozen ref) | 13/48 | 0.271 | [0.166, 0.410] |
| residual actor | **forced actor** | 2/48 | 0.042 | [0.012, 0.140] |
| residual CF | checkpoint_gate (frozen ref) | 7/48 | 0.146 | [0.072, 0.272] |
| residual CF | **forced actor** | 2/48 | 0.042 | [0.012, 0.140] |
| residual CF | **forced actor+guide** | 3/48 | 0.063 | [0.021, 0.168] |
| flow VLA | reference | 10/48 | 0.208 | [0.117, 0.343] |
| flow actor | checkpoint_gate (frozen ref) | 11/48 | 0.229 | [0.133, 0.365] |
| flow actor | **forced actor** | 1/48 | 0.021 | [0.004, 0.109] |
| flow CF | checkpoint_gate (frozen ref) | 11/48 | 0.229 | [0.133, 0.365] |
| flow CF | **forced actor** | 5/48 | 0.104 | [0.045, 0.222] |
| flow CF | **forced actor+guide** | 4/48 | 0.083 | [0.033, 0.196] |
| AE LoRA actor | checkpoint_gate (frozen AE) | 0/48 | 0.000 | [0.000, 0.074] |
| AE LoRA CF | checkpoint_gate (frozen AE) | 0/48 | 0.000 | [0.000, 0.074] |
| AE LoRA actor | forced actor | — | — | **invalid by construction** |
| AE LoRA CF | forced actor / actor+guide | — | — | **invalid by construction** |

Frozen HTTP VLA canaries sit around 0.15–0.27 held-out SR. Forced learned residual/flow actors are much worse. Frozen in-process AE (adapters off) scored **0/48**, so even the AE reference path in V13 is not a working canary.

## Paired policy comparisons

Same pose and seed. Positive delta would mean the treatment is better.

| Comparison | Δ SR | 95% paired CI | McNemar p | Verdict |
| --- | ---: | --- | ---: | --- |
| residual actor vs frozen ref | **−0.229** | [−0.363, −0.096] | 0.00342 | learned actor **worse** |
| residual CF actor vs frozen ref | −0.104 | [−0.209, +0.001] | 0.125 | unresolved / negative trend |
| residual CF actor+guide vs actor | +0.021 | [−0.071, +0.113] | 1.0 | unresolved |
| flow actor vs frozen ref | **−0.208** | [−0.324, −0.092] | 0.00195 | learned actor **worse** |
| flow CF actor vs frozen ref | −0.125 | [−0.263, +0.013] | 0.146 | unresolved / negative trend |
| flow CF actor+guide vs actor | −0.021 | [−0.130, +0.088] | 1.0 | unresolved |

## Known V13 implementation failures (do not read as algorithm disproof for AE)

These are why AE forced-policy cells were skipped and why the 0.003 gate must stay:

1. **Wrong AE coordinates.** Raw robot actions were treated as Molmo q01–q99 flow states; guided output was not unnormalized.
2. **FIFO AE replay (capacity 128).** Successes were evicted; the last 128 rows were late failures only → critic collapse.
3. **Non-policy RNG for AE source noise.** Only 24 unique AE trajectories were reused for 400 episodes.
4. **Hard-min critic.** One dead head owned `q_min` (100% of AE-actor samples in the later probe) → action-insensitive gates.
5. **AE context K/V LoRA created under `no_grad`;** modulation cache survived adapter toggles and optimizer steps.

Residual and flow forced-actor evals are valid counterfactuals of the V13 residual/flow implementation. They show the learned policies were worse than the frozen VLA on this held-out set. That is a negative efficacy result, not a harness skip.

## Verdicts

| Claim | Verdict |
| --- | --- |
| Eight-arm V13 matrix ran to 400 episodes with provenance | **VERIFIED** |
| Gate never opened (`paired_lcb` always `< 0.003`) | **VERIFIED** |
| Training SR is not policy efficacy | **VERIFIED** (protocol) |
| Residual learned actor beats frozen VLA on held-out | **NOT VERIFIED** (significantly worse) |
| Flow learned actor beats frozen VLA on held-out | **NOT VERIFIED** (significantly worse) |
| Consensus guide helps over actor-only | **INCONCLUSIVE** |
| Molmo AE LoRA / AE+CF policy efficacy | **INCONCLUSIVE** (invalid by construction; frozen AE canary also 0/48) |
| Lower the 0.003 safety gate | **NOT VERIFIED** — forbidden by this evidence |

## Artifacts

- Launch manifest: `runs/rlt_cf_v13_controlled/MANIFEST.json`
- Per-arm metrics: `runs/rlt_cf_v13_controlled/<arm>/metrics.jsonl`
- Held-out cells: `runs/rlt_cf_v13_controlled/validation/`
- Paired report: `runs/rlt_cf_v13_controlled/plots/v13_paired_policy_report.json`
- Per-pose outcomes: `runs/rlt_cf_v13_controlled/plots/v13_per_pose_outcomes.json`
- Theory contract: `V13_THEORY_IMPLEMENTATION_AUDIT.md`
- Successor repairs: `V14_THEORY_IMPLEMENTATION_AUDIT.md`
