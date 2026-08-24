# V22: flow actor on the RL-Token state (corrected ConsensusFlow)

Mug-focused. Frozen pretrained **pi0.5**, RL state `x = (z_rl, proprio)`, and a
**flow-matching actor** instead of V21's one-pass Gaussian. The question: should
the RLT actor emit the action in one pass, or be a flow corrector trained with a
ConsensusFlow-style loss?

Code: `pi05_rl_token/` (`rlt/flow_rlt.py` `FlowRLTAgent`, `rlt/networks.py`
`FlowActor` + `DoubleCritic`). Checkpoint:
`pi05_droid_finetune_pick_full_v3_39999`. Chunk 8, `step_time`,
`rl_action_space=delta`, horizon 500.

**Default: RL from env step 0** (no frozen-VLA prefix). Set `gate_step=N` to let
pi0.5 run the first N steps, or `gate_step=-1` for the old scene catalog (mug 56).

## Why the first ConsensusFlow did not learn

The original V22 (`rlt/consensusflow.py`) let its actor maximize an unbounded
ensemble-Q lookahead. `actor_q` left [0, 1] within a few hundred offline steps
while the flow-BC term stayed O(1); the flow never fit the VLA chunk, and Euler
integration from noise emitted joint deltas up to 1e9 — orders of magnitude past
the ~0.2 rad the scene allows. Held-out evals: mug no-AE 22/64 = 34.4%, mug
AE-ft 3/64 = 4.7%, kettle no-AE 1/64 = 1.6%, kettle AE-ft 0/64 = 0%.

## Corrected method (`flow_rlt`)

The Q signal is a **DoubleCritic** trained by TD with the target clipped to
[0, 1]. The actor keeps the paper's Eq. 5 anchor, `-Q(s, a) + beta *
||a - ã||²` with **beta=100** (the calibrated delta-space value). The anchor
gradient grows linearly with deviation while the critic's is piecewise constant,
so the ascent stays in the trust region where the critic is meaningful. A frozen
critic alone is not enough — a ReLU MLP extrapolates without bound off its data
manifold, and a bare `-Q` ascent finds those rays within a few dozen steps.

The flow actor is **conditioned on the VLA reference chunk** (`ref_dim =
chunk_dim`), the same conditioning the RL-Token actor uses. A reference-
conditioned flow is a corrector: it copies the reference at BC convergence and
the critic bends the endpoint online. Two actor variants share this code:

- **Flow corrector** (`flow_compose=false`): the network is the full velocity
  `v(x, x_t, t, ã)`.
- **CF composition** (`flow_compose=true`): the network is an unconditioned
  guidance field `G(x, x_t, t)`; the analytic base velocity toward the reference
  (`ã - noise`) is added, so `V = v_pi05_base + G` and the policy is an exact
  pi0.5 copy at init (G ≈ 0). This is the "V = pi05 actor + G(RLT actor)" arm.

Critic schedule is a per-arm flag: `flow_freeze_critic_online` freezes the
offline-pretrained critic for the online phase (arm 2) or keeps TD running
(arm 3). An external `rlt_critic` checkpoint is always frozen.

- AC pretrain: `scripts/run_pretrain_ac.py`, 100 frozen-VLA train episodes,
  8000 offline steps, `flow_actor_coef=0` (pure BC + anchor → a tight copy).
- Probe: 10 actor-only episodes immediately after pretrain
  (`probe_episodes=10`). They are **not** written to replay; they only seed
  `sr_last10`.
- Online: `scripts/run_train.py`, 300 stored episodes, warmup=0, `gate_step=0`,
  `flow_actor_coef=1` (Q-ascent + anchor), `episode_pool=0-11`. Rolling metric
  is **last-10** stored-actor SR (`sr_last10`).
- Held-out eval: `agent.pt` on eval48, `--episodes 16` → 64 rollouts, same gate.

## Controlled mug comparison (beta=100, shared buffer + AE + pi0.5)

`pipeline_mug_compare.sh`. All arms share the same 100-trajectory buffer
(`beta1_jitter_ac` mug collect, 3112 rows), the same `beta1_from_scratch` mug AE
(frozen encoder), frozen pi0.5, beta=100, 300 episodes. That table used catalog
**gate 56**. Run dir `runs/compare_mug/`. Probe 10, not stored. Last-10 is
recomputed from each `metrics.jsonl` success series (last 10 stored actor
episodes).

| Arm | Actor | Critic online | AE ft | Gate | Probe | Online SR | Last-10 | Held-out eval48 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| **v21** | one-pass Gaussian | TD | no | 56 | 8/10 | 283/300 = 94.3% | 9/10 = 90% | **64/64 = 100%** |
| **flow_frz** | flow corrector | frozen | no | 56 | 8/10 | 284/300 = 94.7% | 9/10 = 90% | 59/64 = 92.2% |
| **flow_td** | flow corrector | TD | no | 56 | 6/10 | 294/300 = 98.0% | 9/10 = 90% | 63/64 = 98.4% |
| **cf_ae** | v_base + G | TD | yes | 56 | 5/10 | 293/300 = 97.7% | **10/10 = 100%** | 62/64 = 96.9% |
| **cf_noae** | v_base + G | TD | no | 56 | 7/10 | 286/300 = 95.3% | 9/10 = 90% | **64/64 = 100%** |

Reads:

- **One-pass vs flow** (v21 vs flow_td): flow_td is a little stronger online
  (98.0% vs 94.3%) and starts faster (last-10 **80% vs 60%** at stored ep 10);
  both saturate held-out (100% / 98.4%). End-of-run last-10 is 90% for both.
  The gap is small.
- **Frozen vs online critic** (flow_frz vs flow_td): an online critic helps
  overall SR (94.7% → 98.0%) and held-out (92.2% → 98.4%); end last-10 is 90%
  for both.
- **Corrector vs composition** (flow_td vs cf_noae): comparable (98.0% vs
  95.3% online, both ~98–100% held-out, last-10 90%).
- **AE finetune** (cf_ae vs cf_noae): helps online (97.7% vs 95.3%) and is the
  only arm still at last-10 **100%** at episode 300; held-out is the other way
  (96.9% vs 100%).

## Gate ablation — cf_ae from step 0

`pipeline_mug_cf_ae_gate0.sh`, run dir `runs/compare_mug_gate0/`. Same cf_ae
recipe (compose, TD critic, AE finetune, shared actor+buffer) with
**`gate_step=0`**. This is now the default for new experiments.

| Arm | Gate | Probe | Online SR | Last-10 | Held-out eval48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cf_ae | 56 | 5/10 | 293/300 = 97.7% | 10/10 = 100% | 62/64 = 96.9% |
| **cf_ae** | **0** | 7/10 | 289/300 = 96.3% | **10/10 = 100%** | **64/64 = 100%** |

Dropping the pi0.5 prefix does not hurt on mug: online is a hair lower, last-10
and held-out saturate. New runs therefore start RL at step 0 unless a prefix is
requested.

## Corrected-V22 standalone test (frozen external critic)

`pipeline_mug_flow_rlt.sh`, run dir `runs/v22_cf/mug_flow_rlt/`. Uses the
finished V21 `beta1_jitter_ac` critic as a frozen external critic, reference-
conditioned flow corrector, beta=100, base-AE finetune online. Catalog gate 56.
This was the first evidence the fix works before the controlled comparison.

| Metric | s0 |
| --- | ---: |
| Probe (not stored) | 4/10 |
| Online SR | 202/300 = 67.3% |
| Last-10 SR | 5/10 = 50% |
| Held-out eval48 | 42/64 = 65.6% |

Lower than the comparison arms because its critic is a *frozen* pre-trained one
(no online TD) and it predates the shared-buffer controlled setup.
