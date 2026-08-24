# V21: RL Token — one-pass Gaussian actor on the RL state

Frozen pretrained **pi0.5**, a scene-specific token AE trained from scratch on
frozen-VLA tokens, then **RL Token** with a one-pass Gaussian actor. This is the
reference one-pass method for the actor-architecture comparison.

Code: `pi05_rl_token/` (`rlt/agent.py` `RLTokenAgent`, `rlt/networks.py`
`Actor` + `DoubleCritic`). Checkpoint: `pi05_droid_finetune_pick_full_v3_39999`.
Chunk 8, `step_time`, `rl_action_space=delta`. **Default: RL from env step 0**
(no frozen-VLA prefix). Set `gate_step=N` to let pi0.5 run the first N steps, or
`gate_step=-1` for the old scene catalog (mug 56, kettle 40).

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
VLA reference by the Eq. 5 term `-Q(s, a) + beta * ||a - ã||²`. `beta=1` for the
original mug/kettle runs; later work uses `beta=100` (calibrated for the delta
action space).

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

## Controlled mug comparison (beta=100, shared buffer + AE + pi0.5)

The V21 arm of `pipeline_mug_compare.sh` — same 100-trajectory buffer
(`beta1_jitter_ac` mug collect, 3112 rows), same `beta1_from_scratch` AE,
frozen pi0.5, beta=100, 300 episodes. That table used catalog gate 56 (the
default at the time). Run dir `runs/compare_mug/`. Probe 10, not stored.

| Arm | Gate | Probe | Online SR | Last-10 | Held-out eval48 |
| --- | ---: | ---: | ---: | ---: | ---: |
| **v21** (one-pass Gaussian, TD critic) | 56 | 8/10 | 283/300 = 94.3% | 9/10 = 90% | **64/64 = 100%** |

New runs default to `gate_step=0`. See `V22_METHODS.md` for the flow-actor arms
and the gate-0 cf_ae ablation.
