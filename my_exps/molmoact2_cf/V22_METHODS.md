# V22: flow actor on the RL-Token state (corrected ConsensusFlow)

Paper method (shared with radio): [`V22_V23_METHODS.md`](V22_V23_METHODS.md).
Pick is that method with a small `FlowActor` MLP instead of LoRA on the
expert. This file is the mug comparison and the 18-object collect table.

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
||a - ã||²` with **default beta=1**. Some mug-comparison and Pick-18 runs used
`beta=100`; those are labeled below. The anchor
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

## Controlled mug comparison (those runs used beta=100, shared buffer + AE + pi0.5)

`pipeline_mug_compare.sh`. All arms share the same 100-trajectory buffer
(`beta1_jitter_ac` mug collect, 3112 rows), the same `beta1_from_scratch` mug AE
(frozen encoder), frozen pi0.5, **beta=100** (not the current default), 300 episodes. That table used catalog
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
conditioned flow corrector, **beta=100**, base-AE finetune online. Catalog gate 56.
This was the first evidence the fix works before the controlled comparison.

| Metric | s0 |
| --- | ---: |
| Probe (not stored) | 4/10 |
| Online SR | 202/300 = 67.3% |
| Last-10 SR | 5/10 = 50% |
| Held-out eval48 | 42/64 = 65.6% |

Lower than the comparison arms because its critic is a *frozen* pre-trained one
(no online TD) and it predates the shared-buffer controlled setup.

## Molmo Pick-v1.1 object set (18 tasks)

The 18-task sweep uses **MolmoSpaces Pick-v1.1** (`FrankaPickDroidMiniBench`,
shipped val benchmark `molmospaces-bench-v1/20260408`). Episodes 0–127 are the
held-out slice. Each **metadata category** in that slice contributes **one**
episode: the first time the category appears. Mug (val 9) and kettle (val 0)
already had jittered benches (`desk_mug`, `kettle`); the other 16 were built by
`scripts/make_pick_object_benches.py` (manifest
`pi05_rl_token/assets/benches/pick_objects/manifest.json`).

Construction matches mug/kettle: the chosen val episode is repeated with **±2 cm
XY jitter**, 48 train specs (seed 0) and 48 eval specs (seed 1000; unjittered
template dropped so the halves are disjoint). Frozen pi0.5 then rolls **100 train
trajectories** per task (horizon 500, kettle 400). Catalog `gate_step` is 56 for
mug and 40 for the rest; **new runs use `gate_step=0`**. **Default `beta=1`**.
The first Pick-18 V21/cf_ae sweep (`runs/pick18/`) used **`beta=100`**; the
repeat (`runs/pick18_beta1/`) used **`beta=1`**.

Collect SR is frozen-pi0.5 on those 100 train traj (not held-out eval). Mug:
`runs/beta1_1gpu/desk_mug` (`collect_desk_mug_a`). Kettle: `runs/beta1_jitter_ac`
shards a+b. Other 16: `runs/pick_objects/<obj>/collect/`. Macro **1028/1800 =
57.1%**.

| Scene | Val ep | House | Language | 100-traj collect |
| --- | ---: | ---: | --- | ---: |
| kettle | 0 | 0 | pick up the kettle. | 14/100 = 14.0% |
| remote | 1 | 1 | pick up the remote. | 66/100 = 66.0% |
| ladle | 2 | 10 | pick up the ladle. | 89/100 = 89.0% |
| tissue | 3 | 10 | pick up the tissue. | 64/100 = 64.0% |
| spoon | 5 | 101 | pick up the spoon. | 99/100 = 99.0% |
| spatula | 6 | 101 | pick up the spatula. | 96/100 = 96.0% |
| desk_mug | 9 | 104 | pick up the mug. | 52/100 = 52.0% |
| pot | 10 | 104 | pick up the pot. | 52/100 = 52.0% |
| soap_dispenser | 12 | 105 | pick up the bottle. | 48/100 = 48.0% |
| spray_bottle | 14 | 106 | pick up the bottle. | 1/100 = 1.0% |
| cup | 18 | 108 | pick up the cup. | 100/100 = 100.0% |
| shaker | 20 | 109 | pick up the pepper. | 12/100 = 12.0% |
| fork | 29 | 115 | pick up the fork. | 99/100 = 99.0% |
| bottle | 41 | 126 | pick up the bottle. | 0/100 = 0.0% |
| fruit | 43 | 127 | pick up the apple. | 52/100 = 52.0% |
| bowl | 44 | 128 | pick up the bowl. | 100/100 = 100.0% |
| knife | 54 | 136 | pick up the knife. | 9/100 = 9.0% |
| box | 55 | 137 | pick up the box. | 75/100 = 75.0% |

Language is the benchmark instruction, not always the scene name:
`soap_dispenser` / `spray_bottle` / `bottle` all say "pick up the bottle"
(`bottle` is the wine-bottle episode that the older `wine_bottle` scene rejected
as wrong-object); `shaker` is "pick up the pepper."; `fruit` is "pick up the
apple." `spray_bottle` is val 14 (same episode as `atomizer`). `ladle` is val 2
(same episode as `house10`). Later repeats of the same category in val 0–127 are
not used.

## Pick-18 V21 vs V22 (cf_ae), $\beta{=}100$ and $\beta{=}1$

`pipeline_pick18_v21_cfae.sh`. Per task: scene AE, shared 100-traj VLA buffer,
8000-step AC pretrain, 10 probe (not stored), 300 online episodes, `gate_step=0`,
held-out eval48. V21 = one-pass Gaussian. V22 = cf_ae (compose + AE finetune).

Plots: `runs/pick18/plots/pick18_sr_heldout.png`, `pick18_sr_online.png`,
`pick18_sr_curves.png`, `pick18_sr_macro.png`. Numbers: `runs/pick18/plots/metrics.md`.
All 72 held-out evals are 64 rollouts. Tasks sorted by 100-traj collect SR.
Each RL cell is **probe after offline AC pretrain → held-out eval48**. Probe is
10 actor-only episodes, not stored.

| Task | collect | V21 $\beta{=}100$ | V22 $\beta{=}100$ | V21 $\beta{=}1$ | V22 $\beta{=}1$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| bottle | 0/100 = 0.0% | 0/10 → 0/64 = 0.0% | 0/10 → 0/64 = 0.0% | 0/10 → 0/64 = 0.0% | 0/10 → 0/64 = 0.0% |
| spray_bottle | 1/100 = 1.0% | 0/10 → 1/64 = 1.6% | 0/10 → 1/64 = 1.6% | 0/10 → 7/64 = 10.9% | 0/10 → 2/64 = 3.1% |
| knife | 9/100 = 9.0% | 0/10 → 0/64 = 0.0% | 0/10 → 57/64 = 89.1% | 0/10 → 46/64 = 71.9% | 2/10 → 3/64 = 4.7% |
| shaker | 12/100 = 12.0% | 6/10 → 19/64 = 29.7% | 2/10 → 33/64 = 51.6% | 3/10 → 27/64 = 42.2% | 2/10 → 23/64 = 35.9% |
| kettle | 14/100 = 14.0% | 6/10 → 21/64 = 32.8% | 2/10 → 50/64 = 78.1% | 4/10 → 61/64 = 95.3% | 4/10 → 54/64 = 84.4% |
| soap_dispenser | 48/100 = 48.0% | 0/10 → 64/64 = 100.0% | 4/10 → 64/64 = 100.0% | 0/10 → 57/64 = 89.1% | 1/10 → 45/64 = 70.3% |
| desk_mug | 52/100 = 52.0% | 8/10 → 63/64 = 98.4% | 5/10 → 64/64 = 100.0% | 5/10 → 57/64 = 89.1% | 5/10 → 39/64 = 60.9% |
| pot | 52/100 = 52.0% | 4/10 → 9/64 = 14.1% | 5/10 → 57/64 = 89.1% | 0/10 → 54/64 = 84.4% | 4/10 → 49/64 = 76.6% |
| fruit | 52/100 = 52.0% | 0/10 → 24/64 = 37.5% | 2/10 → 40/64 = 62.5% | 0/10 → 10/64 = 15.6% | 5/10 → 0/64 = 0.0% |
| tissue | 64/100 = 64.0% | 7/10 → 33/64 = 51.6% | 6/10 → 59/64 = 92.2% | 4/10 → 58/64 = 90.6% | 8/10 → 49/64 = 76.6% |
| remote | 66/100 = 66.0% | 8/10 → 63/64 = 98.4% | 4/10 → 64/64 = 100.0% | 8/10 → 52/64 = 81.2% | 8/10 → 60/64 = 93.8% |
| box | 75/100 = 75.0% | 3/10 → 62/64 = 96.9% | 8/10 → 60/64 = 93.8% | 6/10 → 63/64 = 98.4% | 5/7† → 56/64 = 87.5% |
| ladle | 89/100 = 89.0% | 7/10 → 62/64 = 96.9% | 9/10 → 63/64 = 98.4% | 6/10 → 63/64 = 98.4% | 7/7† → 59/64 = 92.2% |
| spatula | 96/100 = 96.0% | 10/10 → 64/64 = 100.0% | 10/10 → 63/64 = 98.4% | 7/7† → 63/64 = 98.4% | 10/10 → 58/64 = 90.6% |
| spoon | 99/100 = 99.0% | 7/10 → 64/64 = 100.0% | 10/10 → 64/64 = 100.0% | 8/10 → 63/64 = 98.4% | 10/10 → 61/64 = 95.3% |
| fork | 99/100 = 99.0% | 4/10 → 64/64 = 100.0% | 10/10 → 64/64 = 100.0% | 1/10 → 58/64 = 90.6% | 10/10 → 57/64 = 89.1% |
| cup | 100/100 = 100.0% | 10/10 → 64/64 = 100.0% | 10/10 → 64/64 = 100.0% | 10/10 → 34/64 = 53.1% | 7/7† → 53/64 = 82.8% |
| bowl | 100/100 = 100.0% | 10/10 → 64/64 = 100.0% | 10/10 → 64/64 = 100.0% | 10/10 → 64/64 = 100.0% | 7/8† → 57/64 = 89.1% |
| **held-out macro** | **1028/1800 = 57.1%** | **741/1152 = 64.3%** | **931/1152 = 80.8%** | **837/1152 = 72.7%** | **725/1152 = 62.9%** |
| **probe macro** | — | 90/180 = 50.0% | 97/180 = 53.9% | 65/170† = 38.2% | 69/140† = 49.3% |
| **online macro** | — | 3346/5400 = 62.0% | 4195/5400 = 77.7% | 3165/5400 = 58.6% | 2436/5400 = 45.1% |

† Five $\beta{=}1$ jobs crashed mid-probe and resumed into online, so those
probe counts are out of 7 or 8 logged episodes rather than 10. Held-out macros
still use all 18×64. Probe macros skip the † cells.

Reads:

- **Best arm:** V22 at $\beta{=}100$ (80.8% held-out, 77.7% online). V21 prefers
  $\beta{=}1$ on held-out (72.7% vs 64.3%); V22 prefers 100 (80.8% vs 62.9%).
- **Hard zeros:** `bottle` (wine bottle, 0/100 collect) stays 0 for every actor.
  `spray_bottle` barely moves (1% collect → 2–11%).
- **Where cf_ae wins at $\beta{=}100$:** kettle (32.8% → 78.1%), pot (14.1% →
  89.1%), tissue (51.6% → 92.2%), knife (0% → 89.1%), fruit (37.5% → 62.5%).
- **Where $\beta{=}1$ hurts V22:** mug 100% → 60.9%, knife 89.1% → 4.7%, fruit
  62.5% → 0%, cup last-10 100% → 20%. The weaker anchor lets the flow leave the
  VLA trust region on several scenes.
