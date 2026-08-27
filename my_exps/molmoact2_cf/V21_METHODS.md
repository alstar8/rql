# V21: RL Token — one-pass Gaussian actor on the RL state

Frozen pretrained **pi0.5**, a scene-specific token AE trained from scratch on
frozen-VLA tokens, then **RL Token** with a one-pass Gaussian actor. This is the
reference one-pass method for the actor-architecture comparison.

Code: `pi05_rl_token/` (`rlt/agent.py` `RLTokenAgent`, `rlt/networks.py`
`Actor` + `DoubleCritic`). Checkpoint: `pi05_droid_finetune_pick_full_v3_39999`.
Chunk 8, `step_time`, `rl_action_space=delta`. **Default: RL from env step 0**
(no frozen-VLA prefix). Set `gate_step=N` to let pi0.5 run the first N steps,
`gate_frac=0.1` for ~10% of the horizon snapped down to a chunk boundary (48 of
500, 40 of 400), or `gate_step=-1` for the old scene catalog (mug 56, kettle 40).
`gate_frac > 0` wins over `gate_step`. Train and eval must use the same gate.

## Method

```
RGB, proprio, language
        │
   pi0.5 PaliGemma (frozen)          → prefix tokens (968, 2048)
        │
   Token encoder g_φ  →  z_rl        ← phase-1 AE, frozen during RL
        │
   x = (z_rl, proprio)
        │
   pi0.5 action expert → ã (delta reference chunk)   ← frozen
        │
   Gaussian π_θ(a | x, ã)   one pass, conditioned on the reference
   twin Q_ψ(x, a)           TD, target clipped to [0, 1]
        │
   AC pretrain on frozen-VLA train rollouts (actor off, then BC + TD)
        │
   10 actor-only probe episodes (not stored in replay; seed sr_last10)
        │
   Online RL from step 0, warmup=0, init from that actor + buffer
```

The actor emits the action chunk in **one forward pass** and is anchored to the
VLA reference by the Eq. 5 term `-Q(s, a) + beta * ||a - ã||²`. **Default
`beta=1`** (paper value). Some later mug-comparison and Pick-18 runs were
launched with `beta=100`; those are labeled below.

- AE training: `python -m rlt.train_token_ae`, 8000 steps, cap 6000 sequences.
  Tokens are recorded for the whole episode.
- AC pretrain: `scripts/run_pretrain_ac.py`, 100 frozen-VLA train episodes,
  then 8000 offline steps.
- Probe: 10 actor-only episodes immediately after pretrain
  (`probe_episodes=10`). They are **not** written to replay; they only seed
  `sr_last10`.
- Online: `scripts/run_train.py`, 300 stored episodes, warmup=0, `gate_step=0`,
  `episode_pool=0-11`. Rolling metric is **last-10** stored-actor SR
  (`sr_last10`).
- Held-out eval: `agent.pt` on eval48, `--episodes 16` → 64 rollouts, same gate.

## Mug (desk_mug, ±2 cm jitter, horizon 500)

AE: `runs/beta1_from_scratch/desk_mug/ae/ae_desk_mug.pt`, fit on a fresh
frozen-VLA collect (139 traj, 79 succ, 56.8% collect SR, 5120 token seqs).
Frozen held-out VLA on the same eval bench: **37/64 = 57.8%**.

Original `beta1_jitter_ac` run (beta=1, catalog gate 56),
`runs/beta1_jitter_ac/desk_mug/`. These seeds predate `probe_episodes`; last-10
is recomputed from the stored-actor success series.

| Metric | s0 | s1 |
| --- | ---: | ---: |
| Online actor SR | 278/300 = 92.7% | 292/300 = 97.3% |
| Last-10 SR | 8/10 = 80% | 10/10 = 100% |
| Held-out eval48 | 15/15 = 100% | — |

## Kettle (house0, horizon 400)

Recollects tokens on jittered train48 poses (100 traj, 14 succ, 14.0% SR) and
trains a new AE. Frozen held-out VLA baseline: **7/64 = 10.9%**. Historical run
used catalog gate 40.

`beta1_jitter_ac` kettle online jobs stopped early. Last-10 is recomputed from
the stored-actor series.

| Metric | s0 | s1 |
| --- | ---: | ---: |
| Online actor SR | 23/57 = 40.4% | 17/50 = 34.0% |
| Last-10 SR | 8/10 = 80% | 6/10 = 60% |
| Held-out eval48 | **62/64 = 96.9%** | — |

The finished 1-GPU kettle run (`runs/beta1_1gpu/kettle`, probe off) reached
238/300 = 79.3% online, last-10 **9/10 = 90%**, same 62/64 held-out.

## Controlled mug comparison (those runs used beta=100, shared buffer + AE + pi0.5)

The V21 arm of `pipeline_mug_compare.sh` — same 100-trajectory buffer
(`beta1_jitter_ac` mug collect, 3112 rows), same `beta1_from_scratch` AE,
frozen pi0.5, **beta=100** (not the current default), 300 episodes. That table used catalog gate 56 (the
default at the time). Run dir `runs/compare_mug/`. Probe 10, not stored.

| Arm | Gate | Probe | Online SR | Last-10 | Held-out eval48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **v21** (one-pass Gaussian, TD critic) | 56 | 8/10 | 283/300 = 94.3% | 9/10 = 90% | **64/64 = 100%** |

New runs default to `gate_step=0`. See `V22_METHODS.md` for the flow-actor arms
and the gate-0 cf_ae ablation.

## Molmo Pick-v1.1 object set (18 tasks)

Same construction as the V22 sweep: per-task AE, 100 frozen-VLA train traj,
`gate_step=0`, 300 online episodes, held-out eval48 (64 rollouts). V21 is the
one-pass Gaussian arm of `pipeline_pick18_v21_cfae.sh`. **Default `beta=1`**;
`runs/pick18/` used `beta=100`, `runs/pick18_beta1/` used `beta=1`.

Plots: `runs/pick18/plots/` (`pick18_sr_heldout`, `pick18_sr_online`,
`pick18_sr_curves`, `pick18_sr_macro`, `metrics.md`). All 18×2×2 held-out evals
are 64 rollouts. Tasks sorted by 100-traj collect SR. Each RL cell is
**probe after offline AC pretrain → held-out eval48**. Probe is 10 actor-only
episodes, not stored. See `V22_METHODS.md` for the matching cf_ae table.

| Task | collect | V21 $\beta{=}100$ | V21 $\beta{=}1$ |
| --- | ---: | ---: | ---: |
| bottle | 0/100 = 0.0% | 0/10 → 0/64 = 0.0% | 0/10 → 0/64 = 0.0% |
| spray_bottle | 1/100 = 1.0% | 0/10 → 1/64 = 1.6% | 0/10 → 7/64 = 10.9% |
| knife | 9/100 = 9.0% | 0/10 → 0/64 = 0.0% | 0/10 → 46/64 = 71.9% |
| shaker | 12/100 = 12.0% | 6/10 → 19/64 = 29.7% | 3/10 → 27/64 = 42.2% |
| kettle | 14/100 = 14.0% | 6/10 → 21/64 = 32.8% | 4/10 → 61/64 = 95.3% |
| soap_dispenser | 48/100 = 48.0% | 0/10 → 64/64 = 100.0% | 0/10 → 57/64 = 89.1% |
| desk_mug | 52/100 = 52.0% | 8/10 → 63/64 = 98.4% | 5/10 → 57/64 = 89.1% |
| pot | 52/100 = 52.0% | 4/10 → 9/64 = 14.1% | 0/10 → 54/64 = 84.4% |
| fruit | 52/100 = 52.0% | 0/10 → 24/64 = 37.5% | 0/10 → 10/64 = 15.6% |
| tissue | 64/100 = 64.0% | 7/10 → 33/64 = 51.6% | 4/10 → 58/64 = 90.6% |
| remote | 66/100 = 66.0% | 8/10 → 63/64 = 98.4% | 8/10 → 52/64 = 81.2% |
| box | 75/100 = 75.0% | 3/10 → 62/64 = 96.9% | 6/10 → 63/64 = 98.4% |
| ladle | 89/100 = 89.0% | 7/10 → 62/64 = 96.9% | 6/10 → 63/64 = 98.4% |
| spatula | 96/100 = 96.0% | 10/10 → 64/64 = 100.0% | 7/7† → 63/64 = 98.4% |
| spoon | 99/100 = 99.0% | 7/10 → 64/64 = 100.0% | 8/10 → 63/64 = 98.4% |
| fork | 99/100 = 99.0% | 4/10 → 64/64 = 100.0% | 1/10 → 58/64 = 90.6% |
| cup | 100/100 = 100.0% | 10/10 → 64/64 = 100.0% | 10/10 → 34/64 = 53.1% |
| bowl | 100/100 = 100.0% | 10/10 → 64/64 = 100.0% | 10/10 → 64/64 = 100.0% |
| **held-out macro** | **1028/1800 = 57.1%** | **741/1152 = 64.3%** | **837/1152 = 72.7%** |
| **probe macro** | — | 90/180 = 50.0% | 65/170† = 38.2% |
| **online macro** | — | 3346/5400 = 62.0% | 3165/5400 = 58.6% |

† `spatula` $\beta{=}1$ crashed mid-probe and resumed into online; 7 logged
probes, all successes. Held-out macros still use all 18×64.

V21 is stronger at $\beta{=}1$ than at 100 on held-out. `bottle` stays 0/64.
`knife` at $\beta{=}100$ is 0/64 held-out (19% online); cf_ae on the same
buffer is 57/64 = 89.1% — see `V22_METHODS.md`. `cup` at $\beta{=}1$ is the
main V21 regression vs collect (10/10 probe, then 34/64 = 53.1% held-out).

## V21 $\beta{=}1$ Step=10% (frozen-VLA prefix)

Same Pick-18 recipe as `runs/pick18_beta1/` (reuses those AEs and V21 pretrained
actors) except online + eval set `gate_frac=0.1`. The frozen VLA runs a prefix
equal to 10% of the horizon, snapped **down** to a chunk boundary so the
handover is a decision point: **48 of 500** (9.6%), **40 of 400** on kettle
(10.0%). Pretrain stays at gate 0 (actor off). Launcher:
`pipeline_pick18_v21_stepfrac.sh`. Run dir `runs/pick18_beta1_step10/`.
