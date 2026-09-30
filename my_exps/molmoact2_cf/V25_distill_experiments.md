# V25 distillation experiments

Goal: fold Stage A QUORUM (`Euler(vθ + G)` over 10 flow steps) into the π0.5 action expert, so eval is a pure VLA. No V, no G, no token autoencoder. Same official 1000-episode MolmoPick val unless noted.

The PaliGemma backbone stays frozen. Only `gemma_expert` and the action, state, and time projections are trained. The teacher is the greedy deployed chunk (`explore=False`), not the exploring rollout.

## Baselines

| policy | val-1000 |
|---|---|
| frozen π0.5 | 63.5% |
| Stage A gOn (frozen π0.5 + V + G) | 77.7% |

The guidance gain on this val is 14.2 points. A distilled expert that merely copies π0.5 sits at 63.5%. One that fully absorbed gOn would sit at 77.7%.

## Results

| run | what changed | val-1000 |
|---|---|---|
| sequential E1 | 600 eps, fixed original π0.5, offline distill | 66.9% [63.9, 69.7] |
| sequential E2 | one swap, then another 600 on the new expert | **69.9%** [67.0, 72.7] |
| sequential E5 | same loop continued | 66.2% |
| unguarded parallel | copy the student into ã every 50 eps, keep all shards | 62.9% [59.9, 65.8] |
| keep500 last student | gated copy, stitch, keep 500 shortest successes | 32.8% [30.0, 35.8] |
| keep500 frozen expert | weights after the last accepted copy (round 34) | 50.4% [47.3, 53.5] |
| fixed600, step 1000 | fixed ã, 600 eps, early-stopped | 54.2% [51.1, 57.3] |
| fixed600, 12k steps (s12k) | same data, full budget, loss on steps `[0, 8)` | **71.5%** [68.6, 74.2] |
| iter2 | 600 more eps with s12k as ã, distill 12k from s12k | 71.3% [68.4, 74.0] |
| DAgger | s12k rolls out, original gOn labels, mixed corpus, 16-step loss | 71.0% [68.1, 73.7] |
| train48 | 864 eps, layouts 0–47, fixed ã, 8-step loss, 12k steps | 69.3% [66.4, 72.1] |
| endpoint | same 600 eps as s12k, clean-action loss, time in (0, 0.25] | **74.4%** [71.6, 77.0] |

Online success on the Pick-18 train mix, where it was recorded:

| run | online |
|---|---|
| unguarded parallel, gOn | 84.2% (1515/1800) |
| keep500, gOn | 76.5% (1377/1800) |
| fixed600, gOn on the original expert | 85.0% (510/600) |
| iter2, gOn on s12k | 90.3% (600 eps) |
| DAgger, s12k executing, labels from original gOn | 82.5% (495/600) |
| train48, gOn on the original expert | 85.1% (735/864) |

## What each run did

### Sequential E1–E5

Stop-and-distill. The reference expert stays fixed for 600 episodes. Collectors record the greedy 8-step gOn chunk. An offline flow-matching run trains the action expert, then the servers load that student for the next round.

E1 recovered about a quarter of the guidance gain (66.9%). One swap added another 3 points (E2, 69.9%). Continuing the loop to E5 fell back to 66.2%. The long offline distill on a fixed expert is the part that worked. Further swaps did not keep the gain.

Details and the round tables are in [V25_Distill.md](V25_Distill.md).

### Unguarded parallel

Two experts in one online run. Every 50 episodes the learning expert is copied into the frozen servers, and the next teachers are gOn of that new expert. Online gOn stayed high (84.2%). The last student as a pure VLA was 62.9%, half a point under frozen π0.5. The QUORUM gain did not survive without V and G.

### Wipe

Same parallel loop, but every teacher shard is deleted after each round. Round 1 was a normal distill. After that the unfinished student replaced ã, the good teachers were gone, and the student trained on gOn of a worse base. Online success collapsed from about 88% to 34%. The run was stopped. Copying before the student is measured, and deleting the only good teachers, is the failure mode.

### keep500

Gated version of the parallel loop. The student is copied into ã only when a 36-episode G-off probe is strictly higher. The buffer keeps the 500 shortest successful episodes. The training target is a closed-loop stitch: steps 0–7 are gOn at the current observation, steps 8–15 are gOn at the next observation, and the loss averages the whole horizon.

The gate refused 28 of 36 copies, so the run did not collapse. It still failed as a distillation. The stitched tail is about ten times larger than the guidance residual on the prefix, and the student never sees the second observation. Shortest-success pruning drops the hard tasks. The student kept training for about 231k steps. Val used the last student, which the last gate had already rejected: **32.8%**. The last accepted expert, evaluated on its own, is **50.4%**, also below frozen π0.5. The eight noisy copies damaged the reference.

Full round table: [V25_keep500_report.md](V25_keep500_report.md).

### fixed600

Return to the E2 conditions, collected in parallel. Original π0.5 stays the reference for 600 episodes on layouts 0–11. No copy. All episodes are kept. The recorded stitch tail is ignored. The loss is flow matching on steps `[0, 8)` and the 8 real action dimensions.

The first training run early-stopped at step 1000 because a noisy holdout rose after that. That checkpoint is **54.2%**. Retraining the same shards for the full 12k steps, with no holdout, produced **s12k at 71.5%**. That was the best distilled VLA until the endpoint run, and it is above E2 (69.9%). The self-check gap was small (reference loss 0.025, teacher loss 0.032). The final training loss was 0.0018. The student had fit the recorded chunks.

### iter2

One more swap, done properly relative to keep500. s12k becomes ã. Another 600 episodes of gOn are collected (90.3% online). Distill starts from s12k and runs 12k steps on steps `[0, 8)`. Val is **71.3%**. The new teacher was barely different from s12k (reference loss 0.015, teacher loss 0.020), so the second round had little left to absorb.

### DAgger

s12k executes. The original π0.5 and the Stage A actor stay frozen and supply the greedy gOn label at the states the student visits. V and G are not updated. Distill starts from s12k and trains on the union of those labels and the fixed600 shards (25,923 decisions).

This distill supervised all 16 steps (`loss_mask` [0, 16]), unlike s12k. The self-check already had the teacher loss below the reference loss (0.022 vs 0.024): s12k was already as close to gOn as to the raw expert. Val is **71.0%**. Visiting the student's own states did not add a correction the 12k-step fit could use.

### train48

Same fixed-anchor recipe as s12k, on the full train split. 864 episodes, layouts 0–47, one pass per task. Each task walks the layout pool on its own (`episode // n_tasks`), so this is all 48 repeats rather than the 8 layouts that `episode % 48` would have reused. Loss on steps `[0, 8)`, 12k steps, no early stop, init from the original π0.5. 17,274 decisions. Online gOn 85.1%. Val **69.3%**.

More layouts did not raise val. The final loss was 0.0012, so the model fit the larger set. The extra repeats averaged the student down relative to the 12-layout corpus.

### Endpoint

Same 12,129 decisions as s12k. Init is the original π0.5. Loss is still restricted to steps `[0, 8)`. Two changes address the size of the guidance shift, which is about 0.002 in action space:

- Flow time is drawn in (0.001, 0.25], so the noisy action stays close to the teacher chunk. The default Beta(1.5, 1) schedule puts most of the loss near pure noise, where a 0.002 shift is invisible.
- An endpoint term matches the decoded clean action to the teacher on those 8 steps (`endpoint_coef=1`). With `x_t = t * noise + (1-t) * action` and velocity target `noise - action`, the clean action is `x_t - t * v`.

12k steps, no early stop. The paired self-check, taken before training, was reference 0.036 and teacher 0.103 (ratio 2.87), a much clearer gap than the uniform-time runs. Val is **74.4%** [71.6, 77.0].

That is 10.9 points over frozen π0.5 and 2.9 points over s12k. It holds about 77% of the Stage A gain (14.2 points). The interval reaches 77.0%, just under the gOn point estimate of 77.7%.

## Reading

The recipe that works is a fixed original expert, greedy 8-step targets, a full 12k-step offline distill, and a loss that can see the small guidance shift. Endpoint training on the original 600-episode corpus is the best number.

These variants did not beat that:

- Copying the student into ã on a short, noisy probe. keep500 ended at 32.8% for the student and 50.4% for the copied expert.
- Deleting teachers between rounds. The wipe run collapsed.
- Early stopping on a small holdout. Step 1000 of the winning corpus scored 54.2%.
- A second swap, once the student already matches gOn. iter2 stayed at 71.3%.
- DAgger on top of that student. 71.0%, and the self-check showed no new residual.
- Labeling all 48 train layouts. 69.3%, below the 12-layout distill.

On the train mix the student was already close to gOn before the endpoint run (s12k executing at 82.5% in the DAgger collection, against about 85% for gOn). The remaining val gap after endpoint training is a few points, not the 14-point gap distillation started from.

## Files

| path | run |
|---|---|
| `pipeline_pick18_v25_parallel.sh` | keep500 |
| `pipeline_pick18_v25_fixed600.sh` | fixed 600, layouts 0–11 |
| `pipeline_pick18_v25_dagger.sh` | DAgger rollout |
| `pipeline_pick18_v25_train48.sh` | layouts 0–47 |
| `/home/jovyan/users/staroverov/v25_endpoint/` | endpoint distill and val-1000 |
| `pi05_rl_token/scripts/train_expert_distill.py` | offline distill (`--supervise-steps`, `--endpoint-coef`, `--time-max`) |
