# V25 — Distill the RL correction back into pi0.5 (Iterated Action-Expert Distillation)

The current parallel recipe is **keep500** (closed-loop stitch, keep 500 best teachers,
copy student → ã only if student G-off beats frozen G-off). Method and numbers:
[V25_keep500_report.md](V25_keep500_report.md). Artifacts:
`/home/jovyan/users/staroverov/v25_keep500`.

## Parallel experts (ungated / wipe; superseded)

The 600-episode stop-and-distill loop did not hold the QUORUM gain: E2 reached
**69.9%** on val-1000, then E5 dropped to **66.2%**. That recipe is stopped.

This revision keeps **two action experts in the same online run**:

| expert | role |
|---|---|
| **Frozen** | HTTP servers. Produces the reference chunk `ã` that QUORUM (V+G) conditions on. Not trained. |
| **Learning** | A second pi0.5 action expert. Flow-matches the *deployed* QUORUM chunk (`Euler(V+G)`, `explore=False`). |

Every **50 actor episodes** (one round) the frozen servers copy the learning expert's
weights in place (`POST /reload`). Collectors pause for a 2×18-task greedy probe of
**base G-off**, **gOn**, and **student G-off** (`rounds.jsonl`) before the copy. The
student target is two successive gOn 8-step chunks stitched into the VLA's 16-step
horizon. Distill shards are **wiped after each round's student save**, so the next
round's student trains only on that round's teachers. Collectors keep running after the copy; the next replan sees the new `ã`.

The PaliGemma backbone is never written, so the token AE and the RL state survive every
copy. Eval is the learning expert as a **pure VLA** on the official 1000-episode
MolmoPick val (no V, no G, no AE).

```
GPU 0–3:  frozen expert servers + collectors (RLT / QUORUM)
GPU 0:    learning expert, flow-matching QUORUM teachers as shards arrive
ep 50, 100, …:  save student → copy expert tensors into every frozen server
```

Launch: `GPUS="0 1 2 3" bash pipeline_pick18_v25_parallel.sh`
Artifacts: `/home/jovyan/users/staroverov/v25_parallel` (workspace NFS is full).
Per-round table: `rounds.jsonl` / `rounds.md` / `progress.jsonl`.

Same Pick-18 pool and Stage A warm-start as before: shared AE, `shared_v24_s0` V/G,
1800 online episodes, then val-1000.

## Parallel V25 results (finished, fail=0)

Pipeline ran 2026-09-18 21:03 → 2026-09-19 15:03 UTC. Online QUORUM 14.4 h
(36 swaps), then the last student as a **pure VLA** on the official 1000-episode
val (13.4 GPU-h). GPUs idle.

| policy | eval | number | vs frozen |
|---|---|---|---|
| frozen π0.5 | val-1000 | 63.5% | — |
| sequential E2 (600-ep then distill) | val-1000 | 69.9% | +6.4 pp |
| sequential E5 (stopped) | val-1000 | 66.2% | +2.7 pp |
| **parallel student (36×50-ep copy)** | val-1000 | **62.9%** [59.9, 65.8] | **−0.6 pp** |
| Stage A gOn (frozen + V + G) | val-1000 | 77.7% | +14.2 pp |
| this run, online QUORUM | Pick-18 1800 eps | 84.2% (1515/1800) | n/a (train mix) |

The teacher stayed strong (online 84.2%, probe 32/36). Copying the student into
the frozen expert every 50 episodes did **not** collapse RLT. The student as a
standalone VLA, however, is at or slightly **below** frozen π0.5: the QUORUM
+14.2 pp did not survive without V and G. Sequential E2, with a long offline
distill on a *fixed* frozen expert, still holds the best distilled number (69.9%).

Round-1 SR is 96% because the frozen expert is still the original π0.5. After
the first copy (round 2) SR is 88%; it then drifts to a ~84% cumulative plateau.
Lowest window: round 24 (72%). Student flow-matching loss at swap is noisy
(0.001–0.157) and does **not** track online SR. `actor_ref_rmse` stays ~0.0043;
`g_over_v` slowly falls 0.0081 → 0.0066.

Weak tasks (all 100 online eps): fruit 26%, spray_bottle 45%, shaker 58%. Strong:
cup 100%, desk_mug / spoon 99%, fork / bowl 98%.

| Round | Episodes | Succ | SR | Cum SR | RMSE | Q | G/V | Gen | Student step | Loss | Swap UTC |
|------:|----------|-----:|---:|-------:|-----:|--:|----:|----:|-------------:|-----:|----------|
| 1 | 1-50 | 48/50 | 96.0% | 96.0% | 0.00438 | 0.160 | 0.0081 | 1 | 3193 | 0.0277 | 2026-09-18 21:58:28 |
| 2 | 51-100 | 44/50 | 88.0% | 92.0% | 0.00456 | 0.162 | 0.0081 | 2 | 6755 | 0.0059 | 2026-09-18 22:53:07 |
| 3 | 101-150 | 42/50 | 84.0% | 89.3% | 0.00465 | 0.157 | 0.0080 | 3 | 10573 | 0.0165 | 2026-09-18 23:53:43 |
| 4 | 151-200 | 43/50 | 86.0% | 88.5% | 0.00461 | 0.161 | 0.0080 | 4 | 14077 | 0.0559 | 2026-09-19 00:43:04 |
| 5 | 201-250 | 46/50 | 92.0% | 89.2% | 0.00460 | 0.167 | 0.0079 | 5 | 17257 | 0.0012 | 2026-09-19 01:28:21 |
| 6 | 251-300 | 41/50 | 82.0% | 88.0% | 0.00457 | 0.166 | 0.0081 | 6 | 21300 | 0.0889 | 2026-09-19 02:23:41 |
| 7 | 301-350 | 42/50 | 84.0% | 87.4% | 0.00457 | 0.160 | 0.0078 | 7 | 24923 | 0.0289 | 2026-09-19 03:00:20 |
| 8 | 351-400 | 41/50 | 82.0% | 86.8% | 0.00455 | 0.168 | 0.0079 | 8 | 28897 | 0.0124 | 2026-09-19 03:19:06 |
| 9 | 401-450 | 40/50 | 80.0% | 86.0% | 0.00455 | 0.168 | 0.0079 | 9 | 32885 | 0.0027 | 2026-09-19 03:37:55 |
| 10 | 451-500 | 44/50 | 88.0% | 86.2% | 0.00456 | 0.168 | 0.0075 | 10 | 36935 | 0.0112 | 2026-09-19 03:57:01 |
| 11 | 501-550 | 43/50 | 86.0% | 86.2% | 0.00453 | 0.167 | 0.0075 | 11 | 40499 | 0.1573 | 2026-09-19 04:13:50 |
| 12 | 551-600 | 44/50 | 88.0% | 86.3% | 0.00456 | 0.171 | 0.0078 | 12 | 44119 | 0.0301 | 2026-09-19 04:30:55 |
| 13 | 601-650 | 39/50 | 78.0% | 85.7% | 0.00444 | 0.175 | 0.0076 | 13 | 48588 | 0.0030 | 2026-09-19 04:52:00 |
| 14 | 651-700 | 40/50 | 80.0% | 85.3% | 0.00441 | 0.171 | 0.0076 | 14 | 52657 | 0.0397 | 2026-09-19 05:11:12 |
| 15 | 701-750 | 40/50 | 80.0% | 84.9% | 0.00452 | 0.177 | 0.0074 | 15 | 56631 | 0.0080 | 2026-09-19 05:29:56 |
| 16 | 751-800 | 43/50 | 86.0% | 85.0% | 0.00439 | 0.179 | 0.0074 | 16 | 59960 | 0.0444 | 2026-09-19 05:45:39 |
| 17 | 801-850 | 43/50 | 86.0% | 85.1% | 0.00441 | 0.187 | 0.0075 | 17 | 63399 | 0.1102 | 2026-09-19 06:01:53 |
| 18 | 851-900 | 38/50 | 76.0% | 84.6% | 0.00440 | 0.176 | 0.0072 | 18 | 67552 | 0.0158 | 2026-09-19 06:21:30 |
| 19 | 901-950 | 45/50 | 90.0% | 84.8% | 0.00436 | 0.177 | 0.0072 | 19 | 70681 | 0.0162 | 2026-09-19 06:36:17 |
| 20 | 951-1000 | 37/50 | 74.0% | 84.3% | 0.00434 | 0.179 | 0.0073 | 20 | 74970 | 0.0758 | 2026-09-19 06:56:31 |
| 21 | 1001-1050 | 41/50 | 82.0% | 84.2% | 0.00436 | 0.180 | 0.0072 | 21 | 78830 | 0.0323 | 2026-09-19 07:15:00 |
| 22 | 1051-1100 | 41/50 | 82.0% | 84.1% | 0.00435 | 0.180 | 0.0072 | 22 | 82840 | 0.0336 | 2026-09-19 07:33:54 |
| 23 | 1101-1150 | 43/50 | 86.0% | 84.2% | 0.00431 | 0.182 | 0.0070 | 23 | 86480 | 0.0050 | 2026-09-19 07:51:05 |
| 24 | 1151-1200 | 36/50 | 72.0% | 83.7% | 0.00428 | 0.179 | 0.0071 | 24 | 91502 | 0.0017 | 2026-09-19 08:14:46 |
| 25 | 1201-1250 | 45/50 | 90.0% | 83.9% | 0.00427 | 0.184 | 0.0070 | 25 | 94490 | 0.0014 | 2026-09-19 08:28:53 |
| 26 | 1251-1300 | 39/50 | 78.0% | 83.7% | 0.00418 | 0.182 | 0.0071 | 26 | 98542 | 0.0241 | 2026-09-19 08:47:59 |
| 27 | 1301-1350 | 45/50 | 90.0% | 83.9% | 0.00435 | 0.183 | 0.0071 | 27 | 101870 | 0.0165 | 2026-09-19 09:03:40 |
| 28 | 1351-1400 | 40/50 | 80.0% | 83.8% | 0.00424 | 0.178 | 0.0069 | 28 | 105929 | 0.1048 | 2026-09-19 09:22:51 |
| 29 | 1401-1450 | 43/50 | 86.0% | 83.9% | 0.00424 | 0.184 | 0.0069 | 29 | 109142 | 0.0026 | 2026-09-19 09:38:01 |
| 30 | 1451-1500 | 43/50 | 86.0% | 83.9% | 0.00431 | 0.185 | 0.0070 | 30 | 112541 | 0.0310 | 2026-09-19 09:54:04 |
| 31 | 1501-1550 | 45/50 | 90.0% | 84.1% | 0.00424 | 0.187 | 0.0066 | 31 | 115622 | 0.0110 | 2026-09-19 10:08:36 |
| 32 | 1551-1600 | 44/50 | 88.0% | 84.2% | 0.00428 | 0.185 | 0.0069 | 32 | 118888 | 0.0844 | 2026-09-19 10:24:02 |
| 33 | 1601-1650 | 41/50 | 82.0% | 84.2% | 0.00429 | 0.186 | 0.0067 | 33 | 122513 | 0.0101 | 2026-09-19 10:41:11 |
| 34 | 1651-1700 | 42/50 | 84.0% | 84.2% | 0.00424 | 0.187 | 0.0068 | 34 | 126110 | 0.0034 | 2026-09-19 10:58:14 |
| 35 | 1701-1750 | 44/50 | 88.0% | 84.3% | 0.00414 | 0.187 | 0.0067 | 35 | 129108 | 0.0265 | 2026-09-19 11:12:24 |
| 36 | 1751-1800 | 40/50 | 80.0% | 84.2% | 0.00421 | 0.184 | 0.0066 | 36 | 133228 | 0.0186 | 2026-09-19 11:31:49 |

---

Stage A/B leave the improved policy split across three pieces that all have to be
present at eval time: the frozen pi0.5 action expert, the learned flow actor `V`,
and the guidance field `G` (plus the token AE that builds the RL state). The deployed
policy is `Euler(V + G)` on top of the frozen reference chunk. That is fine for a
benchmark number, but it is not a VLA you can ship.

**V25 folds the correction back into the VLA.** A *student* copy of the pi0.5 action
expert is trained to imitate the deployed teacher `(frozen expert + V + G)`, so that
at eval time there is no `V`, no `G`, no AE — just pi0.5 with a better action expert.

> **The loop (iterated distillation).** Keep the RL stack training on top of the
> *current* expert. Every round, distill the deployed teacher into the student, then
> **swap**: the student becomes the expert the VLA server serves, `V`/`G` keep training
> on top of it, and the next round distills the further-improved teacher. The anchor
> loss `β‖a_V − ã‖²` is what makes this sound — after a swap the reference `ã` is the
> improved policy, so `V` re-centres on it instead of double-counting the correction.

This is the user's proposed mechanism: a separate pi0.5 action expert that learns from
`frozen expert + V + G`, and once per N update steps replaces the frozen expert.

## Why the action expert only

The RL state is `z = AE(prefix tokens)` from the **frozen PaliGemma backbone**. If the
backbone moved, every prefix token, the AE, and the merged buffer would all be
invalidated. The action expert (`gemma_300m` + `action_in/out_proj` + `state_proj` +
the adaRMS time MLP) is the only part that touches actions and the only part that is
trained. The backbone stays bit-identical, so the AE and the RL state survive every
swap unchanged.

## Teacher and target

At each decision point (every `chunk_size=8` env steps) the teacher is:

```
reference ã  = current expert chunk (server, delta space, 16×8)
state   s    = [AE(tokens), proprio]
teacher a*   = Euler(V + G)(s, fresh noise), explore=False    # the DEPLOYED chunk, 8×8
```

**The teacher is the deployed policy, not the executed one.** Collectors run with
`explore=True`, which on the flow agents integrates the *online* actor; the 77.7% Stage A
number belongs to `explore=False`, which integrates the *EMA target*. Distilling what the
collector executed would therefore fit exploration noise rather than the policy whose gain
is being preserved. The corrector takes a second, noise-free pass at the same state
(`record_deployed`, one 10-step MLP unroll — negligible next to the VLA call that produced
the reference) and that greedy chunk is the target. The executed chunk is recorded too, as
`executed`, so the gap is measurable rather than assumed.

Measured on the first round-1 shards, this is not a cosmetic distinction:

| quantity | mean abs (delta space) |
|---|---|
| `\|teacher − executed\|` — greedy vs exploring | 0.0041 |
| `\|teacher − reference\|` — the correction being distilled | 0.0026 |
| `\|executed − reference\|` | 0.0038 |

The exploration noise is **1.5× the correction signal**, so distilling `executed` would
have spent most of its capacity on noise.

The student is trained with the model's own flow-matching loss so that
`student.sample_actions(obs) ≈ a*`. Supervision is masked to the first `chunk_size=8`
steps and the real 8 action dims — the teacher only commits 8 steps, and eval replans
every 8, so the tail of the 16-step horizon keeps its initialised behaviour.

Both `ã` and `a*` are recorded at every corrected decision (see below); `ã` feeds the
frozen-reproduction self-check, `a*` is the distillation target.

## Recording

`--set distill_out=<dir>` makes each collector's `Pi05RLPolicy` write one shard per
~256 corrected decisions:

```
external_cam (224,224,3) u8 | wrist_cam (224,224,3) u8 | state (8,) f32
instruction str | reference (16,8) f32 | teacher (8,8) f32
```

~300 KB per decision, ~60 decisions per episode → ~11 GB per 600-episode round on NFS.

## Student training (`scripts/train_expert_distill.py`)

- Load the pi0.5 checkpoint exactly as the server does (`create_trained_policy`,
  config `pi05_droid_finetune`), so preprocessing is *the same code*, not a copy.
- Freeze `paligemma_with_expert.paligemma.*` (vision tower + language model +
  projector); train `gemma_expert.*` + the action/state/time projections.
- Dataset = the round's recorded shards. The raw observation dict goes through the
  policy's own `_input_transform` (DroidInputs → quantile Normalize → Tokenize →
  Pad-to-32); the recorded teacher chunk goes through the same transform's action
  path, then is padded 8→16 along the horizon.
- Loss = `model.forward(obs, actions)` flow-matching MSE, masked to steps `[0,8)` and
  dims `[0,8)`. AdamW, small lr, bf16, gradient checkpointing.
- Save a **full** pi0.5 checkpoint (`model.safetensors` + `config.json` + `assets/`),
  so the server and the pure-VLA eval load it with zero code changes.

**Self-check (gate before any training).** With the student still equal to the frozen
checkpoint, the flow-matching loss on the recorded *reference* chunks must be low (the
reference is a genuine sample of the frozen model — this validates the whole
preprocessing + action-normalisation chain), and the loss on *teacher* chunks must be
higher (the correction moved the action). Training then drives the teacher loss down.
If the reference loss is not low, the pipeline is wrong and nothing trains.

`freeze_backbone` additionally refuses to run if any parameter outside
`paligemma_with_expert` matched no entry in the trainable list. Without that check a stale
prefix silently freezes part of the head: `time_mlp_in/out` (the adaRMS flow-time
conditioning) were being frozen because the list called them `action_time_mlp_*`, so the
distillation would have run with 691M of the intended 693M parameters and no error.

## Pre-flight validation

Run once before committing GPU time (`_distill_probe.py`, `_distill_smoke_data.py`):

| check | result |
|---|---|
| transform + training forward produce a finite masked loss | yes |
| self-check direction on real frozen-model samples | reference 0.041 < teacher 0.150 |
| student checkpoint reloads through the serving path | yes, actions finite |
| **backbone bit-identical after training** | **603/603 tensors, max diff 0.0** |
| gradients reach the action output | mean \|Δaction\| 4.2e-4 after 20 steps |

The backbone check is the load-bearing one: it is what makes the token AE and the RL state
still valid after a swap, so it is verified rather than assumed.

## The experiment (Pick-18, warm-started from Stage A)

Reuses Stage A artifacts — shared AE, `shared_v24_s0` agent (val **77.7%**), merged
buffer. No re-collect, no re-pretrain.

```
round 1:  servers on E0 (frozen pi0.5), V/G resume Stage A, 600 eps on Pick-18,
          recording on  →  distill → E1   (one-shot distillation of the 77.7% teacher)
round 2:  servers on E1, V/G resume round-1, 600 eps, recording on
          →  distill (init from E1) → E2   (one iteration of the swap loop)
eval:     E1 and E2 each on the official 1000-episode MolmoPick val as a *pure VLA*
          (no actor, no AE, no guidance) — same shards as the Stage A/B evals.
```

Rounds are separate `run_train.py` invocations (resume agent+buffer, point
`--set checkpoint=` at the latest expert). That avoids hot-swapping weights into a live
server; the swap is just "next round's servers load the new checkpoint".

## Read-out (run finished, fail=0)

Same 1000-episode MolmoPick val, pure VLA for E1/E2 (no V, no G, no AE):

| policy | eval | number | vs frozen |
|---|---|---|---|
| frozen pi0.5 | val-1000 | 63.5% | — |
| **E1 (one-shot distill)** | val-1000 | **66.9%** [63.9, 69.7] | **+3.4 pp** |
| **E2 (one swap iteration)** | val-1000 | **69.9%** [67.0, 72.7] | **+6.4 pp** |
| Stage A gOn (E0+V+G) | val-1000 | 77.7% | +14.2 pp |

One-shot distillation recovered about a quarter of the RL gain into a standalone VLA.
The swap loop added another **+3.0 pp** (E1 → E2); CIs overlap, but the point estimates
move in the same direction as the online rates. Together E2 holds **45% of the Stage A
gain** without any extra modules at inference. The remaining 7.8 pp still lives in V+G.

Online, the swap did **not** collapse (the fallback of resetting V/G at each swap was
not needed):

| round | online SR (600 Pick-18 eps) | last 100 | first 50 after swap |
|---|---|---|---|
| round 1 on E0 | 84.3% (506/600) | 87.0% | — |
| round 2 on E1 | 88.0% (528/600) | 89.0% | 86.0% (vs round-1 last 50: 84%) |

Student training self-checks were in the right direction both rounds, and the
flow-matching loss dropped:

| expert | decisions | ref loss | teacher loss | final loss | hours |
|---|---|---|---|---|---|
| E1 from E0 | 13,172 | 0.011 | 0.030 | 0.0035 | 1.16 |
| E2 from E1 | 11,427 | 0.009 | 0.016 | 0.0080 | 1.17 |

Checkpoints: `runs/pick18_v25_distill/expert_e{1,2}/` — loadable by `serve_pi05_http.py`
with `--checkpoint` and no other flags.

## Continuation to 100 rounds

Same recipe, resumed from E2: 600 Pick-18 episodes of online RL with recording, distill
the deployed teacher into the action expert, swap the VLA server to that student, repeat.

A val-1000 every round would add ~13 days of eval on top of ~18 days of train+distill, and
keeping every 7.4 GB expert plus every 3.5 GB shard dump would not fit the remaining
~840 GB. The 100-round run therefore:

- **skips** rounds 1–2 (already done)
- **evals** as a pure VLA on the same 1000-episode shards at **E5, E10, E20, …, E100**
- **deletes** distill shards after each successful distill
- **keeps** expert weights only at those eval points (plus E1, E2, and the latest)

Progress is appended to `runs/pick18_v25_distill/progress.jsonl`. Expected remaining wall
clock on 4 GPUs: **~18–20 days** (98 × ~4.2 h train+distill + 11 × ~3.2 h val-1000).

## Cost

Wall clock was **~16 h** sequential (22:07 → 14:08 UTC): two 600-ep rounds ≈ 4 h each,
two distillations ≈ 1.2 h each, two val-1000s ≈ 3.2 h each. Matches the pre-run estimate.
