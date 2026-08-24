# pi0.5 + RL Token on MolmoSpaces

Everything needed to evaluate the frozen pi0.5 policy, and to train RL Token on top of
it, without asking anyone how. Settings live in files, not in command lines.

```
pi05/
  config.py      every knob: paths, scenes, chunk size, gate, RL hyperparameters
  model.py       the action path: what the model emits, how a chunk is spent, how an
                 action reaches the environment. Read this one first.
  client.py      talking to the frozen VLA server (one contract, both entry points)
  policy.py      the classes MolmoSpaces builds and drives: eval and training
  rl_token.py    the gate, and the point where the RL correction enters the chunk
  eval.py        the evaluation run
  train.py       the RL training run
  vla_server.py  loading pi0.5 and capturing its prefix tokens (the RL state)
  action_space.py, proximity.py, convert.py, merge.py, ...   supporting pieces

scripts/
  run_eval.py            evaluate               <- start here
  run_train.py           train RL Token
  serve_pi05_http.py     the frozen VLA server (started for you by both of the above)
```

---

## 1. Quick start

Use the simulator's interpreter. The three virtualenvs are not interchangeable and
`config.py` names all of them.

```bash
cd /home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
SIM=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python

# See every setting and the resolved paths, run nothing.
$SIM scripts/run_eval.py --show

# Evaluate the frozen VLA on one scene.
$SIM scripts/run_eval.py --scene house21

# The held-out 128-episode slice of the shipped benchmark.
$SIM scripts/run_eval.py --scene val --episodes 128
```

`run_eval.py` starts the frozen VLA on its own GPU, runs the benchmark, prints the
success rate with a 95% Wilson interval, and **stops the server again**, including when
the run fails. Nothing is left holding a card.

Results land in `<out_dir>/<tag>/`:

```
config.json     the exact settings this run used
result.json     successes, episodes, success_rate, ci95, seconds
server.log      the frozen VLA's log
eval_output/    per-episode videos and HDF5 trajectories
```

The default `tag` names the regime, e.g. `house21_vla_chunk1_plan_time`, because two runs
are only comparable if those agree.

### Changing a setting

Edit the field in `pi05/config.py`. That is the point: the value ends up in the
repository rather than in somebody's shell history. For a genuine one-off:

```bash
$SIM scripts/run_eval.py --scene house10 --set chunk_size=8 --set episodes=36
```

`--set` accepts any field of `EvalConfig` and coerces it to the field's type. An unknown
name lists the ones that exist.

---

## 2. The scenes

A scene is one validation-benchmark episode repeated with a �2 cm horizontal jitter on
the object (`scripts/make_repeat_benchmark.py`). The jitter is for variety, not for a
train/test split: this is RL on a single episode, and the point is to squeeze that episode.

**The three in `SELECTED_SCENES` were chosen by measurement on 2026-08-20**, to span the
bands the project owner asked for, and each was checked for *why* it fails:

| band | scene | val ep / house | measured | how it fails |
|---|---|---|---|---|
| ~0% | `atomizer` | 14 / 106 | 0/12 probed | hand ends up on the atomizer at success-range geometry and never lifts it |
| 20�30% | `sink_dispenser` | 25 / 112 | **14/64 = 21.9%** (95% 13.5�33.4) | reaches the dispenser, stalls with the fingers on the body |
| 40�60% | `desk_mug` | 9 / 104 | **38/64 = 59.4%** (95% 47.1�70.5) | reaches the mug, closes beside it |

Every one fails at the **grasp**, which is the thing RL is being asked to fix. That was
verified against recorded ground truth, not by eye � see step 3 of the pipeline below.

### Rejected, and why it matters

`wine_bottle` (ep 41) looked ideal � a clean 0/64 � and was wrong. The recorded target
points show the bottle sitting in the corner of the wrist view while the fingers close on
a different object nearby; the hand reaches the actual target on only 9 of 64 rollouts.
No correction to the action chunk can change which object the model chose, so the scene
is not an RL target. It is kept in `SCENES`, marked rejected, as the worked example.

### Superseded

`house2`, `house10` and `house21` are still in `SCENES` so older runs can be reproduced,
but they are **not** selected. Their original selection came from a profiling run recorded
as contaminated, with no artefacts left to recheck, and their rates were measured under
chunk 1 + `plan_time`. Re-measured in the current regime they give **0% / 82% / 96%** �
two of them saturated, with no headroom for RL to show anything.

`--scene val` uses the shipped 1000-episode benchmark instead; its first 128 episodes are
the held-out slice the regime sweep in section 4 was measured on.

---

## 3. The model, and how an action reaches the arm

Full detail is in the docstring of `pi05/model.py`. In brief:

```
images + instruction + joint state
          |
          v
  PaliGemma (gemma_2b, 2048 wide)  reads the scene -> KV cache
          |                        its last hidden state, (968, 2048), IS the RL state
          v
  action expert (gemma_300m)       10 flow-matching steps -> 16 actions x 8 dims
          |
          v
  [0:7] joint DELTAS,  [7] gripper 0..1
```

**The single most important fact in this project:** the model emits joint *deltas*, and
MolmoSpaces executes *absolute* joint targets. The conversion happens in
`pi05/model.py` and nowhere else:

```python
absolute_arm = action[:7] + current_arm_qpos
gripper      = 255 if action[7] > 0.5 else 0
```

Feed raw deltas to the simulator and the arm is commanded toward a folded pose every
step. That is what the stock evaluation bridge does, and it is why the published pi0.5
baseline on this benchmark reads 0/10. With the conversion in place the *unmodified*
pretrained checkpoint scores 2/8 on the same benchmark. The 0% was a bug, not a model.

The server sends deltas and nothing else. It reports `converts_delta_to_absolute`, and
the client refuses to run against a server that converts — adding the arm state twice
looks like a bad policy rather than a misconfiguration, so it is made impossible instead
of documented.

---

## 4. chunk_size and conversion: settled

The model predicts 16 actions. `chunk_size` is how many are executed before asking again;
`conversion` decides which arm state each of them is added to. **A sweep over the
held-out 128-episode val benchmark on 2026-08-20 settled both**, one server per cell,
eight cells in parallel, nothing else varied:

| | chunk 1 | chunk 4 | chunk 8 | chunk 16 |
|---|---|---|---|---|
| `plan_time` | 40.6% | 51.6% | 44.5% | **0.0%** |
| `step_time` | 37.5% | 51.6% | **62.5%** | **63.3%** |

Read it by row. What decides the outcome is **`conversion`, not chunk length**: under the
faithful regime longer chunks help and then saturate, while under `plan_time` the
accumulated lag eventually destroys the policy outright.

| comparison | | Fisher p |
|---|---|---|
| chunk 1 `plan_time` → chunk 16 `step_time` | 40.6% → 63.3% | **0.0004** |
| chunk 1 `plan_time` → chunk 8 `step_time` | 40.6% → 62.5% | **0.0007** |
| chunk 8 `plan_time` → chunk 8 `step_time` | 44.5% → 62.5% | **0.0057** |
| chunk 16 `plan_time` → chunk 16 `step_time` | 0.0% → 63.3% | **4e-33** |
| chunk 8 `step_time` vs chunk 16 `step_time` | 62.5% vs 63.3% | 1.0 — tied |

**The defaults are now `chunk_size = 8`, `conversion = "step_time"`.** 8 over 16 because
they are statistically tied and 8 re-plans twice as often for the same wall clock, so it
reacts sooner when the scene does something the plan did not expect. It is also ~40%
faster end to end than chunk 1 (5519 s vs 9367 s for 128 episodes) — most of the cost is
the VLA forward pass, and a longer chunk needs fewer of them.

That is **+22 points over the previous default, from a serving-side change alone.** No
retraining, no new data, same checkpoint.

Two calibration notes before comparing any future number against this table:

- chunk 1 `plan_time` and chunk 1 `step_time` are the **same configuration** — the two
  conversions coincide at chunk 1, which `test_pi05_model.py` proves. They ran as
  separate jobs, so their gap is pure run-to-run noise from the flow-matching sampler:
  40.6% vs 37.5%, p=0.70. **Treat gaps below ~5 points as noise.**
- chunk 1 `plan_time` reproduced the previously published PyTorch-converted baseline of
  **40.6% exactly**, which is the evidence that moving the conversion out of the server
  did not change behaviour.

### The earlier per-scene table

| chunk_size | house10 | house21 |
|---|---|---|
| 1 | 36.1% | **88.9%** |
| 2 | 50.0% | 86.1% |
| 4 | 72.2% | 77.8% |
| 8 | **83.3%** | 58.3% |

36 episodes per cell. Both trends are significant and point in opposite directions, which
is what made chunk length look like it had no single right answer. **It was measured
entirely under `plan_time`.** It describes how the lag interacts with two particular
scenes, not how chunk length behaves — do not read a policy out of it.

**A long chunk is not wrong in itself**, and it is worth being precise about why, because
the opposite was believed here for a while. Each frame stores

```
action[k] = q_commanded[t+k] − q_state[t+k]
```

a displacement measured at its own step — semantically a joint velocity. openpi's DROID
data config says so explicitly and deliberately applies no further delta transform:
*"We assume joint velocity actions, so we should not apply an additional delta
transform"* (`openpi/src/openpi/training/config.py:450`). A velocity chunk is meant to run
open loop: element k says "move at this rate at step *t+k*" and needs no particular
absolute pose to be meaningful. That is the ordinary pi0 / pi0.5 regime.

### conversion: the knob that actually decides correctness

Because this environment executes absolute joint targets rather than velocities, each
delta has to be added to *some* arm state, and which one is the real question:

| `conversion` | command sent at step *t+k* | |
|---|---|---|
| `step_time` | `q_state[t+k] + action[k]` | reconstructs `q_commanded[t+k]` **exactly, for every k**. Faithful at any chunk length. **The measured winner.** |
| `plan_time` | `q_state[t] + action[k]` = `q_commanded[t+k] − (q_state[t+k] − q_state[t])` | short by the distance travelled since the plan: 0.027 rad/step, 0.32 rad by k=15. Exact only at k=0. **Every number measured before 2026-08-20 used this.** |

So the drift usually blamed on "long chunks" belongs to `plan_time` specifically.
`chunk_size = 1` was called the safe setting because under the pre-refactor serving path
`plan_time` was the only thing the code could do, so "chunk 1" and "exact" happened to
coincide. They are separate choices, the refactor separates them, and the sweep then
showed the conversion was carrying the effect all along.

The 0/128 at chunk 16 `plan_time` is that arithmetic reaching its conclusion: by k=15 the
command is short by 0.32 rad, more than the motion being commanded, so the arm is driven
somewhere the plan never intended and no episode recovers.

At `chunk_size = 1` the two are identical, which is why the distinction went unnoticed.
`tests/test_pi05_model.py` pins both, their equivalence at chunk 1, and the fact that
`plan_time` reproduces the pre-refactor server-side conversion exactly. `plan_time` is
kept only for reproducing those earlier runs.

---

## 5. How RL Token trains

`pi05/train.py` carries the full description; this is the summary.

**The VLA is frozen throughout.** Nothing updates it. The only things that learn are the
actor and two critic heads, all small MLPs (`rlt/networks.py`).

### What goes in

Per decision, with C = `chunk_size`:

| | |
|---|---|
| **RL state** `x` | `[z_rl, proprio]`. `z_rl` is the phase-1 encoder's compression of the VLA's prefix tokens, `(968, 2048)` → `z_dim`. `proprio` is 16 numbers: 7 joint positions + gripper, and their velocities. |
| **reference** `a_ref` | the first C actions of the VLA's own chunk, flattened |
| **action** `a` | what was actually committed: `a_ref` before the gate or during warmup, otherwise `a ~ N(mu(x, a_ref), sigma^2)` |
| **reward** `r` | discounted sum over the window. `+1` exactly once, on the step the task first reports success; the episode ends there (`end_on_success=True`). |

A transition `<x, a, a_ref, r, x'>` closes C env steps after the decision that opened it,
and `utd` learner iterations run per stored transition — each being
`critic_updates_per_actor` critic updates and one actor update.

### Stride

**One decision per executed chunk.** The paper subsamples with a stride of 2, which
stores overlapping windows spliced from two committed chunks and pairs them with a
reference the robot never executed. At stride = C every stored row is one decision: the
action is the chunk that was chosen, the reference is what the VLA proposed for that same
state, and nothing is stitched. `stride: 0` in `RLConfig` means "same as chunk_size" and
the trainer refuses anything else.

### Which actions of the chunk execute

The same C as evaluation, through the same `ChunkExecutor`. This is not a detail: the
first RL matrix trained at chunk 8 and was evaluated at chunk 1, and since chunk size
alone moves the success rate by 30–50 points depending on the scene, "RL helped" could
not be separated from "the chunk changed". That run was discarded. `chunk_size` and
`conversion` are written into every run directory so the pairing can always be checked.

### The gate

Default: RL drives from env step 0. Set `gate_step=N` to let the frozen VLA run
the first N env steps, or `gate_step=-1` to use the scene catalog handover (mug 56,
kettle 40, �). Once open, a latched gate stays open for the rest of the episode.

A distance gate was tried first and abandoned for a measured reason: the evaluator
records the object's pose from the **benchmark spec**, not its live position, so the
number stops describing reality as soon as the policy nudges the object, and on an
elongated object the centre stays far while the gripper is already at a graspable end. Do
not resurrect the spec-pose version; a real distance gate needs the live MuJoCo pose
pulled out during the rollout. `pi05/proximity.py` keeps both gates and the measurements.

Also: `obs/extra/grasp_state_pickup_obj.touching` read 0 across 64 rollouts where contact
was visible on video. Do not trust that flag.

### Which space the actor refines

`rl_action_space` decides what `a_ref` means:

| | |
|---|---|
| `delta` **(default)** | the raw joint deltas the model emits; conversion happens after the actor, at the environment boundary. |
| `absolute` | absolute joint targets. What the discarded 8-run matrix used. Requires `conversion = plan_time`, so it is incompatible with the settled default regime and survives only to reproduce that matrix. |

`delta` also satisfies the requirement the actor was specified under: **it is shown
exactly the actions that would be executed at real inference, and emits exactly as many
as will be executed.** The reference is the VLA's own `chunk[:chunk_size]` — the plan the
robot would otherwise have run — and the actor emits `chunk_size × 8`. Keeping one
decision per executed chunk (stride == `chunk_size`, enforced in `train.py`) is what
makes the first half true; see *Stride* above for what breaks otherwise.

This is **not cosmetic**. The actor is a plain MLP that emits the chunk outright, not a
residual on the reference, so at initialisation it outputs values near zero. Near-zero
deltas barely move the arm; near-zero absolute joint targets command a completely
different pose. `sigma = 0.02` is large next to deltas of ~0.027 rad and small next to
absolute targets of ~2 rad, and the `beta` pull toward the reference scales the same way.

---

## 6. Launching training

```bash
cd <rl_token>
SIM=/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python

# Check what would run.
$SIM scripts/run_train.py --scene house21 --encoder house21 --show

# Run it, detached -- SSH drops under load.
OUT=/home/jovyan/users/staroverov/pi05_molmospaces/pi05_runs/rl
setsid nohup env PYTHONUNBUFFERED=1 $SIM scripts/run_train.py \
    --scene house21 --encoder house21 --gpu 0 --port 8600 \
    > $OUT/house21.log 2>&1 < /dev/null &
```

`--encoder` picks a phase-1 autoencoder from `pi05_runs/token_ae/`: `house2`, `house10`,
`house21` or `combined`. Encoders trained on MolmoAct2 tokens are **unusable** — that
backbone is 2560 wide and pi0.5 is 2048. `pi05/rl_token.py` sets `RLT_VLA_TOKEN_DIM=2048`
itself so no launcher has to remember.

**One frozen-VLA server per run, one GPU each.** pi0.5 inference blocks the server's
event loop, so two runs sharing a server serialise and then time out during the
handshake, and episodes are skipped silently. The trainer starts and stops its own
server; it does not outlive the run.

Several cells in parallel — one per GPU:

```bash
gpu=0
for enc in house2 house10 house21 combined; do
    setsid nohup env PYTHONUNBUFFERED=1 $SIM scripts/run_train.py \
        --scene house21 --encoder $enc --gpu $gpu --port $((8600 + gpu)) \
        > $OUT/house21_$enc.log 2>&1 < /dev/null &
    gpu=$((gpu + 1))
done
```

Output in `<out_dir>/<tag>/`:

```
config.json     the exact settings
agent.pt        actor + critics; feed to run_eval.py --actor
buffer.npz      the replay buffer, so a run can be resumed or replayed offline
metrics.jsonl   per-episode metrics
tb/             TensorBoard
summary.json    warmup rate vs actor rate
```

### Reading the result

The **warmup rate** is the frozen VLA measured in place, on the same episodes, in the
same regime — the only baseline that is a fair comparison. The **actor rate** is what RL
achieved. Then confirm on held-out episodes:

```bash
$SIM scripts/run_eval.py --scene house21 \
    --actor    $OUT/house21_ae-house21_chunk1_gate40/agent.pt \
    --token-ae /home/jovyan/users/staroverov/pi05_molmospaces/pi05_runs/token_ae/ae_house21.pt
```

**at the same `chunk_size` and `conversion` the actor trained under.** `run_eval.py`
evaluates the actor at its mean, with no exploration noise.

A difference in rates is not a result until it survives a test:

```bash
$SIM scripts/compare_rates.py <successes_a> <n_a> <successes_b> <n_b>   # Fisher exact
```

---

## 6b. The full pipeline, in order

Each step has one entry point and hands its output to the next. Every one of them is
driven by `EvalConfig` / `RLConfig`, so the execution regime is identical throughout —
which is the property that makes the final numbers mean anything.

| # | step | entry point | output |
|---|---|---|---|
| 1 | rank candidate scenes | `scripts/run_profile.py --episodes 0-127 --repeats 4 --shard i --num-shards 8` | `scene_profile/shard*/profile.json`; merge with `--summarize` |
| 2 | re-measure the shortlist, with video | `scripts/run_probe.py --band zero --repeats 6 --gpus 0,1,2,3` | `probe/ep<N>/result.json` + videos |
| 3 | **check it goes for the right object** | `scripts/check_target.py --run <run>/eval_output` | per-rollout `ON TARGET` / `NOT AT OBJ` |
| 4 | promote the winners | `scripts/promote_scene.py --episode N --name X --note "..."` | two benchmarks + a config entry |
| 5 | measure them honestly | `scripts/run_eval.py --scene X --episodes 16` | `result.json` over 64 rollouts + video |
| 6 | choose the gate | `scripts/make_contact_sheet.py --eval-dir <run>/eval_output --every 15` | a labelled frame timeline per rollout |
| 7 | collect the corpus | `scripts/run_collect.py --scene X --target 2500` | `rlt_tokens_step_time/<scene>/*.npz` + manifest |
| 8 | train the encoders | `scripts/run_encoders.py --corpus <dir>` | `ae_<scene>.pt` x3 plus `ae_combined.pt` |
| 9 | run the matrix | `scripts/run_matrix.py` | 12 cells, in waves of `--gpus` |

**Step 1 exists because the original scene selection was not valid.** The three scenes
were inherited; the run that chose them is recorded as contaminated — several candidates
were written into the same house directory and their episodes counted together — and no
artefacts survive to recheck it. Their rates were also measured under chunk 1 +
`plan_time`. Selection is therefore redone by measurement, against these bands:

| band | meaning |
|---|---|
| ~0% | RL never sees a reward. Useful **only** if the failure is the grasp itself; if the policy reaches for the wrong object, a chunk-level correction cannot fix it. Video decides which. |
| 20–30% | both positive and negative examples to learn from |
| 40–60%+ | headroom, and a clear ceiling to beat |

Coarse rates from 4 rollouts cannot separate 1/4 from 2/4, so step 1 is only ever used to
bucket candidates. The quotable rate comes from step 3.

### Step 3 is not optional, and not done by eye

A scene at ~0% is only worth training on if the GRASP is what fails. If the policy reaches
for the wrong object, nothing applied to the action chunk can fix it: the model chose what
to look at before the chunk existed.

This was learned by getting it wrong. A candidate was accepted on a wrist frame showing a
gold object squarely between the fingers � and that object was not the target. The wine
bottle it was asked for was the dark green shape at the very edge of the same frame.

MolmoSpaces records the ground truth and always did:

```
obs/extra/object_image_points/pickup_obj/<camera>/points   (T, 10, 2), normalised
obs/extra/object_image_points/gripper/<camera>/points      the same for the hand
```

Two rules fell out of it:

- **A success proves on-target behaviour.** A scene with successes cannot be a systematic
  wrong-object scene, so only a 0% scene has to prove targeting � by the hand ending up
  where the target is.
- **Calibrate against outcomes, never guess a threshold.** The first version used a
  guessed gap of 0.35 and flagged successful grasps as wrong-object. Measured, contact is
  a gap of 0.37�0.52: the hand's keypoints sit below the object in the wrist view, so the
  gap never reaches zero.

When the flag says `NOT AT OBJ` on a scene with no successes, confirm before rejecting �
draw the recorded target points onto the wrist frame at the step the fingers close, and
look at what is actually there.

**Step 4 is done by looking, not by a rule.** One video frame is one env step — the
simulator drives the policy at `policy_dt_ms = 66` and the writer emits one frame per
policy step at 15.15 fps — so a contact sheet reads directly as a timeline, and the step
where the object first sits between the fingers is the gate. It is chosen per scene: when
the gripper arrives depends on where the object is.

---

## 7. Decisions: settled and open

**Settled 2026-08-20:**

1. **`chunk_size = 8`, `conversion = "step_time"`**, project-wide, from the val sweep in
   section 4. Guarded by `test_pi05_config.py::test_the_defaults_are_the_regime_the_val_sweep_settled_on`
   so a change has to be deliberate.
2. **`rl_action_space = "delta"`**, section 5.

**Still open:**

3. **`store_pre_gate`** — currently `False`, so only transitions the actor could have
   influenced enter the buffer. Storing them is valid off-policy data and carries the
   value of arriving at the gate in a good pose, but spends update budget on states the
   actor never acts in. The paper does not say. This does not block the RL matrix.

**What the settled regime changes for work already done.** Every pre-2026-08-20 number in
this project, including the per-scene chunk table and the 37.5% / 40.6% / 33.6% / 36.7%
comparison against MolmoAct2, was measured under chunk 1 + `plan_time`. Those numbers are
still valid measurements of *that* regime, but they are no longer the baseline: the
frozen VLA is a **62.5%** policy on the held-out slice, not a 40.6% one. Anything RL is
compared against has to be re-measured in the new regime, and the `gate_step = 40` choice
was made from rollouts recorded under the old one — worth re-watching before the matrix,
since the arm now moves differently.

---

## 8. Things that cost a day already

- **One worker per server � but that is NOT one job per GPU.** Two clients on one pi0.5
  server time out during the handshake and their episodes are skipped *silently*: one run
  reported 7 of 16 episodes and would have given a wrong rate. `EvalConfig.validate`
  refuses `workers != 1`.

  That rule is about sharing a *server*. It was wrongly generalised to "one job per card"
  and cost about half the machine. Measured 2026-08-21: the VLA forward pass is only
  **5�10% of an episode** (68.7 s/episode, of which 6.5 s is inference), so the server
  idles most of the time and a card holds 11 GB of 80. Two jobs on one card, each with its
  own server, cost **3�16% more seconds per env step** and nearly double throughput:

  | | s per env step |
  |---|---|
  | desk_mug, alone on a card | 0.199 |
  | desk_mug, sharing a card | 0.206 (+3.5%) |
  | sink_dispenser, alone | 0.141 |
  | sink_dispenser, sharing | 0.163 (+16%) |

  Utilisation swings from 0% to 86% within seconds as simulation, rendering and inference
  take turns, which is why co-tenants fill each other's gaps so cheaply.

- **Compare seconds per env step, never per episode.** A 0% scene runs the full horizon
  every time while a 60% scene ends early, so per-episode timings are not comparable
  across scenes -- and a scene that merely succeeds more often looks faster.
- **The token capture is not a hook.** openpi calls
  `paligemma.language_model.forward(...)` directly, and `register_forward_hook` only fires
  on `__call__`. `pi05/vla_server.py` wraps the bound method instead. A hook there fails
  silently and the RL state is empty.
- **Set env before importing.** MuJoCo picks its EGL device and MolmoSpaces resolves its
  asset paths at import time. `prepare_environment(cfg)` must run first; both entry points
  do it before touching `molmo_spaces`. EGL device order is not CUDA order.
- **Anything that imports JAX claims a GPU** (61 GB seen). For CPU work set
  `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu`.
- **`eval_main` writes videos and HDF5 in a batch when a house finishes**, not per
  episode. Missing files mid-run are normal.
- **Do not `pkill -f <pattern>`** where the pattern appears in your own command line — it
  kills the SSH session. Kill by explicit PID.

---

## 9. Tests

```bash
cd <rl_token>
CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu RLT_VLA_TOKEN_DIM=2048 \
  $SIM -m pytest tests/ -q --ignore=tests/test_pi05_accum.py --ignore=tests/test_pi05_convert.py
```

196 tests, no simulator and no GPU needed. The two ignored files need `lerobot` and run
under openpi's venv instead (one of them has a pre-existing failure asserting
`DROP_FIRST_STEPS == 1` when it is 2 — unrelated to this code).

They encode the facts above — the conversion, the two execution regimes, the gate, the
config guard rails — so a change that breaks one of them is changing a measured finding,
not a style choice. Worth knowing about two in particular:

- `test_pi05_model.py::test_plan_time_reproduces_the_old_server_side_conversion` is what
  keeps every previously measured success rate describing this code after the conversion
  moved out of the server.
- `test_pi05_train_wiring.py` drives the rollout recorder without a simulator. That path
  is silent when it breaks — the episode still runs and the loss still falls — so it pins
  the two alignments that matter: every plan extends `committed` by exactly `chunk_size`
  even when it records no decision, and a row's reward window covers the whole chunk
  rather than the single step that had executed when the decision fired.
