# V25 keep500 — gated parallel distillation of QUORUM into π0.5

**Run:** `v25_keep500` / tag `shared_v25_keep500`
**Artifacts:** `/home/jovyan/users/staroverov/v25_keep500`
**Pipeline:** `bash pipeline_pick18_v25_parallel.sh` (defaults now point at this recipe)
**Online:** 2026-09-20 15:16 → 2026-09-21 16:28 UTC (25.1 h, 1800 episodes)
**Val-1000:** started 2026-09-21 16:28 UTC, still running at the time of this report

The goal is to put Stage A QUORUM (`Euler(vθ + G)` over 10 flow steps) into a **trainable π0.5 action expert**, so inference does not need V or G. Eval is that student as a **pure VLA** on the official 1000-episode MolmoPick val.

This recipe keeps the closed-loop 16-step stitch, **does not wipe** the teacher buffer (keeps 500 best episodes), and **copies the student into ã only when it beats the frozen expert** on a held-out G-off probe.

---

## 1. Why this recipe

Three earlier attempts:

| recipe | what it did | result |
|---|---|---|
| Sequential IAD | freeze ã, distill 600 eps, swap, repeat | E2 **69.9%** val-1000 (best distilled); E5 fell to 66.2% |
| Parallel, unguarded copy | two experts, `/reload` every 50 eps, keep all shards | online QUORUM **84.2%**, student val **62.9%** (−0.6 pp vs frozen 63.5%) |
| Parallel + stitch + **wipe** | same, but delete all teachers each round | SR collapse: online 88% → 34%, gOn probe 100% → 50% (stopped) |

The wipe run showed the failure mode clearly. Round 1 was the only healthy distill (student 63.9% vs original π0.5 52.8%, gOn 100%). After that, every 50 episodes the unfinished student **replaced** ã, the 100% teachers were deleted, and the student trained on gOn of a worse base. Round 2 copied a **53%** student over a **67%** ã because copy happened before `student_off` was measured.

Keep500 is the gated, recency-buffered version of the parallel loop. ã stays the original π0.5 until the student is actually better G-off.

Baselines (same val-1000, same checkpoint):

| policy | val-1000 |
|---|---|
| frozen π0.5 | 63.5% |
| sequential E2 (600-ep distill, fixed ã) | 69.9% |
| unguarded parallel student (36 copies) | 62.9% [59.9, 65.8] |
| Stage A gOn (frozen + V + G) | 77.7% |

---

## 2. Method

### 2.1 Two action experts

| expert | process | role |
|---|---|---|
| **Frozen ã** | 4× HTTP `serve_pi05_http` (GPUs 0–3) | π0.5 action head that QUORUM conditions on. Not trained in place. |
| **Student** | `train_expert_online.py` on GPU 0 | Same action head + time/action projections. Flow-matches deployed gOn chunks. PaliGemma backbone frozen. |

RLT/QUORUM (V + G, `cf_actor_coef=0`, `beta=1`) runs on the collectors against frozen ã. The token AE and critic stay valid across a copy because only `gemma_expert.*` and the action/time projections are written (`POST /reload` → `copy_expert_weights`).

### 2.2 Teacher: closed-loop stitch

gOn commits `chunk_size=8` of the VLA’s 16-step chunk, then replans. The student head is 16 steps from **one** observation, so two successive commits are concatenated at the first observation:

```
teacher[0:8]  = gOn(obs_t)           # G-corrected 8 at the current state
teacher[8:16] = gOn(obs_{t+8})       # next decision, new images / new ã / new G
reference     = ã(obs_t)  (16 steps) # unused open-loop tail of the first VLA call
```

The last unpaired 8-step commit of an episode is padded with that plan’s unused VLA tail so the prefix is not dropped. Collectors explore; the stored `teacher` is the **greedy** (`explore=False`) chunk at the same state. Flow-matching uses the full 16-step `teacher`.

This is **not** a 16-step open-loop unroll from `obs_t`. Steps 8–15 depend on `obs_{t+8}`, which the student does not see. Empirically `|teacher − reference|` is ~0.0026 on the prefix (G residual) and ~0.025 on the tail (closed-loop replan vs unused ã tail).

### 2.3 Teacher buffer: keep 500 best

Each finished **episode** is one `distill_*.npz` tagged `_ok{0|1}_n{steps}`. After every 50-episode round the buffer is pruned to 500 trajectories:

1. successes before failures
2. among successes, **shorter** first (more efficient)
3. among failures, **longer** first (more teacher steps)

Until 500 files exist, nothing is dropped. After that, each round briefly sits at ~550 then prunes back to 500. Probe episodes are not recorded (`record_distill=False`).

### 2.4 Gated copy

Every 50 actor episodes collectors hold. Protocol:

1. Save the student; snapshot weights to `student_round/` **before** the student resumes training.
2. Prune the teacher buffer to 500.
3. Probe **base_off** — current frozen ã, G off, greedy, 2 episodes × 18 tasks (n=36).
4. Probe **gOn** — same ã, G on, `explore=False`.
5. Load `student_round` into the four frozen servers; probe **student_off** (G off).
6. **Copy** `student_round` → `frozen_expert` and keep it on the servers **only if**
   `student_off.sr > base_off.sr` on a complete probe.
7. Otherwise reload `frozen_expert` (or the original π0.5 checkpoint if no copy has ever succeeded).

Ties and incomplete probes do not copy. n=36 has SE ≈ 8 pp, so the gate is noisy; it is still enough to block the wipe-run disaster (copying a 53% student over a 67% ã).

### 2.5 Setup

Same Pick-18 pool and Stage A warm-start as V24/V25:

- Tasks (100 online eps each): desk_mug, kettle, remote, ladle, tissue, spoon, spatula, pot, soap_dispenser, spray_bottle, cup, shaker, fork, bottle, fruit, bowl, knife, box
- Horizons 500 (kettle 400), `episode_pool=0-11`, `chunk_size=8`, `gate_step=0`
- Init actor/buffer: `runs/pick18_v24_shared/rl/shared_v24_s0`
- Token AE: `ae_pick18_shared.pt`, `token_replay` on
- 16 collectors, 3 EGL slots/GPU, 4 frozen VLA servers
- Student: AdamW 1e-5, batch 4, backbone frozen
- 1800 online episodes, then val-1000 of `student_expert/` as a pure VLA (no V, no G)

---

## 3. Results

### 3.1 Headline

| quantity | keep500 | unguarded parallel | sequential E2 |
|---|---|---|---|
| Online gOn SR (train mix, 1800 eps) | **76.5%** (1377/1800) | 84.2% | n/a (stop-and-distill) |
| Copies into ã | **8 / 36** | 36 / 36 | 5 sequential swaps |
| Mean round probe gOn | 85.0% | ~84% online blocks | — |
| Mean round probe frozen G-off | 69.8% | — | — |
| Mean round probe student G-off | 60.0% | ~base | E2 val 69.9% |
| Student val-1000 | **in progress** (see §3.5) | 62.9% | **69.9%** |
| Distill files at end | 500 | all shards kept | per-round offline corpus |

Online 76.5% is **below** unguarded parallel (84.2%) and below Stage A val gOn (77.7%), but **far above** the wipe collapse (last window 34%). The gate did what it was for: ã was not replaced by a worse student on 28 of 36 rounds.

### 3.2 Online QUORUM (50-episode windows)

First window 94% (original π0.5 + G). After the first copy, 86–84%. The run then sits in the 70–86% band; lowest windows 62% (after copies at 300 and 650). Last window 76%. Cumulative 76.5%.

G/V 0.00814 → 0.00591 (−27%). G trust 0.257 → 0.187. `actor_ref_rmse` 0.00446 → 0.00419 (V still hugs ã). Probe at the start of online: 33/36.

### 3.3 Per-task online gOn (100 eps each)

| task | SR | task | SR |
|---|---|---|---|
| spray_bottle | 9% | spatula | 88% |
| fruit | 23% | pot | 90% |
| shaker | 48% | spoon / soap / remote | 91% |
| knife | 58% | box | 92% |
| bottle | 66% | fork | 94% |
| ladle | 79% | bowl | 95% |
| tissue | 81% | desk_mug | 98% |
| kettle | 84% | cup | 99% |

Weak tasks match the unguarded run (fruit, spray_bottle, shaker). Spray_bottle at 9% is worse than that run’s 45%.

### 3.4 Round probes and the eight copies

n=36 (2×18), all 36 rounds complete. Copy iff student G-off > frozen G-off.

**Copied (8):**

| Round | ep | frozen G-off | gOn | student G-off | gap | student step |
|------:|---:|-------------:|----:|--------------:|----:|-------------:|
| 1 | 50 | 52.8% | 91.7% | 58.3% | +5.5 pp | 3231 |
| 3 | 150 | 55.6% | 88.9% | 72.2% | +16.6 pp | 15936 |
| 6 | 300 | 69.4% | 91.7% | 80.6% | +11.2 pp | 35231 |
| 13 | 650 | 77.8% | 94.4% | 80.6% | +2.8 pp | 74617 |
| 21 | 1050 | 69.4% | 88.9% | 77.8% | +8.4 pp | 127133 |
| 29 | 1450 | 61.1% | 86.1% | 77.8% | +16.7 pp | 179593 |
| 33 | 1650 | 58.3% | 80.6% | 63.9% | +5.6 pp | 207604 |
| 34 | 1700 | 58.3% | 75.0% | 61.1% | +2.8 pp | 214960 |

Round 1 is the same pattern as sequential E1/wipe R1: student already above original π0.5, gOn still high, copy is justified. Later copies at 2.8 pp (R13, R34) are inside probe noise.

**Skipped (28), including the ones the wipe run would have copied:**

| Round | ep | frozen | student | note |
|------:|---:|-------:|--------:|---|
| 2 | 100 | 55.6% | 41.7% | wipe R2 copied 53% over 67% |
| 16 | 800 | 77.8% | 27.8% | student collapse, ã kept |
| 22 | 1100 | 66.7% | 38.9% | |
| 28 | 1400 | 69.4% | 19.4% | worst student probe |
| 32 | 1600 | 55.6% | 30.6% | |
| 35 | 1750 | 69.4% | 69.4% | tie → no copy |
| 36 | 1800 | 69.4% | 55.6% | last student, not copied |

gOn probe: 91.7% (R1) → peak 94.4% (R10/R12/R13) → **72.2%** (R36). It does **not** fall off a cliff, but it is not Stage A (77.7% val / ~97% early train). After R30, gOn (66.7%) is **below** frozen G-off (72.2%): G is no longer helping that ã.

Frozen G-off itself **rose** from 52.8% (original π0.5, n=36) to a peak of 86.1% (R11), then ended at 69.4%. The eight copies did move ã. They also moved the teacher distribution the student is imitating.

Student G-off mean 60.0%, min 19.4%, max 80.6%. It never stably matched gOn. Flow-matching loss at save is uncorrelated with probe SR (0.001–0.18).

Keep-best: 86 files after R1, 500/536 at R10, then 500/550 every later round.

### 3.5 Val-1000 (in progress)

Eval loads **`student_expert/`** (last save, generation 37, step 231115), not `frozen_expert/` (last accepted copy, R34). The last two rounds did not copy; the student kept training on the 500-best buffer after that.

At the time of writing (~3.3 h into val, 764/1000 episodes):

| shard | done | SR |
|------:|-----:|---:|
| 0 | 62/189 | 32.8% |
| 1 | 65/197 | 33.0% |
| 2 | 57/185 | 30.8% |
| 3 | 64/193 | 33.2% |
| **partial** | **248/764** | **32.5%** [29.2, 35.9] |

This is **not** the final number. House order can bias a partial. Even so, 32% with 764 episodes is already incompatible with 63.5% frozen / 62.9% unguarded student at the same eval. The last round probe (student 55.6% G-off on the train mix) already said this checkpoint is not a good standalone VLA.

---

## 4. Reading of the run

**What worked**

- The gate stopped the wipe-run death spiral. ã was not replaced 28 times; gOn stayed in the 80–94% probe band for most of the 1800 episodes.
- Keep-500 retained gen-0 (original π0.5 + G) teachers instead of deleting them. Student loss no longer exploded after every round (wipe R2 loss 0.088 after deleting 86 shards).
- Round-1 copy is the same healthy first distill as sequential E1: student > original π0.5, gOn still ~92%.

**What did not**

- The student still does not absorb G. Mean student G-off 60% vs mean gOn 85% vs sequential E2 val 69.9% after a long distill on a **fixed** ã.
- Eight copies still smear ã. Frozen G-off jumps around (53 → 86 → 58 → 69). After R30 G sometimes **hurts**. Online 76.5% < unguarded 84.2%, where ã was replaced constantly but G kept compensating.
- Closed-loop stitch is a misspecified 16-step target: the tail is gOn(`obs_{t+8}`) trained as `student(obs_t)[8:16]`. Prefix G (~0.002) is smaller than that tail (~0.025). The student can fit hindsight instead of G.
- Val is the **last student**, which failed the last gate (55.6% < 69.4%). A fair “best distilled ã” number would eval `frozen_expert/` after R34 or the best student_off checkpoint (R6/R13 at 80.6% n=36).

**Compared to sequential E2**

E2 froze original π0.5 for 600 episodes, distilled offline, then evaluated. Keep500 lets ã move (8 times) and trains online on a mixed, pruned, closed-loop-stitched buffer. The gate is necessary to not collapse, but it is not sufficient to beat E2.

---

## 5. Full round table

| R | ep | frozen G-off | gOn | student G-off | copied | keep | loss |
|--:|---:|-------------:|----:|--------------:|:------:|-----:|-----:|
| 1 | 50 | 52.8 | 91.7 | 58.3 | yes | 86/86 | 0.177 |
| 2 | 100 | 55.6 | 83.3 | 41.7 | | 136/136 | 0.020 |
| 3 | 150 | 55.6 | 88.9 | 72.2 | yes | 186/186 | 0.084 |
| 4 | 200 | 72.2 | 91.7 | 69.4 | | 236/236 | 0.035 |
| 5 | 250 | 75.0 | 86.1 | 66.7 | | 286/286 | 0.019 |
| 6 | 300 | 69.4 | 91.7 | 80.6 | yes | 336/336 | 0.051 |
| 7 | 350 | 75.0 | 86.1 | 55.6 | | 386/386 | 0.037 |
| 8 | 400 | 77.8 | 91.7 | 61.1 | | 436/436 | 0.063 |
| 9 | 450 | 72.2 | 86.1 | 72.2 | | 486/486 | 0.025 |
| 10 | 500 | 72.2 | 94.4 | 69.4 | | 500/536 | 0.002 |
| 11 | 550 | 86.1 | 91.7 | 66.7 | | 500/550 | 0.003 |
| 12 | 600 | 77.8 | 94.4 | 61.1 | | 500/550 | 0.002 |
| 13 | 650 | 77.8 | 94.4 | 80.6 | yes | 500/550 | 0.038 |
| 14 | 700 | 72.2 | 80.6 | 66.7 | | 500/550 | 0.014 |
| 15 | 750 | 75.0 | 80.6 | 58.3 | | 500/550 | 0.003 |
| 16 | 800 | 77.8 | 91.7 | 27.8 | | 500/550 | 0.059 |
| 17 | 850 | 75.0 | 88.9 | 69.4 | | 500/550 | 0.001 |
| 18 | 900 | 66.7 | 83.3 | 55.6 | | 500/550 | 0.007 |
| 19 | 950 | 75.0 | 83.3 | 61.1 | | 500/550 | 0.020 |
| 20 | 1000 | 77.8 | 88.9 | 61.1 | | 500/550 | 0.026 |
| 21 | 1050 | 69.4 | 88.9 | 77.8 | yes | 500/550 | 0.001 |
| 22 | 1100 | 66.7 | 91.7 | 38.9 | | 500/550 | 0.008 |
| 23 | 1150 | 75.0 | 86.1 | 66.7 | | 500/550 | 0.012 |
| 24 | 1200 | 66.7 | 83.3 | 58.3 | | 500/550 | 0.002 |
| 25 | 1250 | 66.7 | 88.9 | 55.6 | | 500/550 | 0.009 |
| 26 | 1300 | 66.7 | 83.3 | 52.8 | | 500/550 | 0.027 |
| 27 | 1350 | 72.2 | 80.6 | 66.7 | | 500/550 | 0.037 |
| 28 | 1400 | 69.4 | 88.9 | 19.4 | | 500/550 | 0.003 |
| 29 | 1450 | 61.1 | 86.1 | 77.8 | yes | 500/550 | 0.001 |
| 30 | 1500 | 72.2 | 66.7 | 55.6 | | 500/550 | 0.032 |
| 31 | 1550 | 75.0 | 75.0 | 52.8 | | 500/550 | 0.073 |
| 32 | 1600 | 55.6 | 66.7 | 30.6 | | 500/550 | 0.015 |
| 33 | 1650 | 58.3 | 80.6 | 63.9 | yes | 500/550 | 0.009 |
| 34 | 1700 | 58.3 | 75.0 | 61.1 | yes | 500/550 | 0.039 |
| 35 | 1750 | 69.4 | 77.8 | 69.4 | | 500/550 | 0.034 |
| 36 | 1800 | 69.4 | 72.2 | 55.6 | | 500/550 | 0.014 |

Probe SR in percent, n=36. `keep` is files kept / files present at prune time.

---

## 6. What would likely raise SR next

Ranked by how directly they attack the remaining failure (student does not absorb G; ã still moves too early):

1. **Sidecar ã.** Never `/reload`. Distill gOn of original π0.5 for the full 1800 (or 600 then stop, like E2). Gate already said “don’t copy” 28 times; making that the default removes the 8 noisy copies.
2. **Mask flow-matching to steps `[0, 8)`.** Keep recording the stitch for analysis, but do not train the tail that the student cannot condition on.
3. **Eval `frozen_expert/` and the best student_off snapshot**, not only the last `student_expert/`. R6/R13 student_off 80.6% (n=36) is the candidate, not generation 37.
4. **Stricter gate** if copies remain: require `student_off ≥ gOn − 5 pp` and gOn not down vs the last copy, not only `student_off > base_off`.
5. **Freeze ã for ≥600 episodes** before the first legal copy (E2’s horizon).

---

## 7. Files

| path | what |
|---|---|
| `pipeline_pick18_v25_parallel.sh` | launch (keep500 defaults) |
| `pi05/distill_recorder.py` | stitch + per-episode npz + keep-best |
| `pi05/round_eval.py` | probes + gated copy |
| `pi05/expert_swap.py` | save snapshot, restore frozen |
| `scripts/train_expert_online.py` | student FM loop |
| `.../v25_keep500/rl/shared_v25_keep500/{summary.json,rounds.jsonl,metrics.jsonl}` | online logs |
| `.../v25_keep500/{student_expert,frozen_expert,student_round,distill}` | checkpoints + 500 teachers |
| `.../v25_keep500/eval_val1000/` | val-1000 shards (running) |
