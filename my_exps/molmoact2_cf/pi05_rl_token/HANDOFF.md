# Handoff: pi0.5 + RL Token on MolmoSpaces

You are taking over an in-flight project. The previous agent ran out of context. This
document contains everything established by measurement so you do not re-derive it or
re-break it. Facts marked **measured** were verified on the machine; anything not marked
that way is untested.

---

## 0. The task you are being handed

The user asked for five things, in this order. **Nothing else. Do not add scope.**

1. **Restructure the code** into a clean, readable layout like popular RL libraries
   (CleanRL / SB3 style: separate files for config, model, env, algorithm, train, eval).
   Right now logic is scattered across one-off scripts and parameters are buried in
   command lines. The user cannot run anything without the agent, which is the problem
   being solved.
2. **An eval entry point plus a written guide**, where scene parameters are visible and
   editable in a file rather than passed as flags. The user says evaluation is currently
   a black box to them.
3. **A model file** that makes the architecture explicit: what `forward` does, what
   `chunk_size` is, how a chunk is unpacked into actions, how actions reach the
   environment, how many chunk actions are executed, and where RL Token plugs in.
4. **Documentation of how RL Token trains**: what goes in, what the stride is, which
   actions of the generated chunk are executed.
5. **How to launch training.**

Write code and documentation **in English**. The user is Russian-speaking but explicitly
asked for English in the code and docs; chat replies to them can be Russian.

Two partially-written files exist in the previous agent's scratchpad and were being
copied to the repo — treat them as drafts, not finished work: `pi05/config.py` (all
parameters in one dataclass-based place) and `pi05/model.py` (policy, chunk executor,
action conversion). Their content is described below so you can rewrite from scratch if
you prefer.

---

## 1. Working rules the user enforces

These come from the project brief and from corrections the user made repeatedly. Ignoring
them has already caused friction.

- **Ask, do not invent.** If a hyperparameter, threshold, or data format is not obvious,
  stop and ask. The previous agent invented explanations twice and was called out both
  times. A wrong guess costs a day here.
- **A question is a question.** When the user asks "can we do X?", answer it. Do not start
  doing X. This caused an explicit reprimand.
- **Do not delete anything** on the server without explicit permission, and when given
  permission, delete only inside
  `/home/jovyan/users/staroverov/pi05_molmospaces/`. Never touch anything else.
- **Keep the GPUs busy with useful work**, and never leave idle processes holding a card.
  The user is accountable for utilisation. Equally: do not burn GPUs on junk.
- **Disk is tight.** The shared volume runs at 95-99%. Keep new data inside ~100 GB.
- **Only one directory is writable for code:**
  `/home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token`
  Everything else is read-only; to change a file outside it, ask first.
- Long jobs must be detached: `setsid nohup ... < /dev/null &` with `PYTHONUNBUFFERED=1`.
  SSH drops under load.
- Never use `pkill -f <pattern>` or `pgrep -f <pattern>` where the pattern appears in your
  own command line — it kills your SSH session. Kill by explicit PID.

---

## 2. Machine and paths

SSH alias: `Cloud-mlspace-worker`. **8x H100 80GB.** `$HOME` is `/home/user`; work under
`/home/jovyan`. Never use `~`.

Note `/home/jovyan/users/staroverov/...` and `/workspace-SR008.nfs2/users/staroverov/...`
resolve to the same NFS volume.

Everything this project produced lives in one root:

```
/home/jovyan/users/staroverov/pi05_molmospaces/
  checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999/   14 GB  the frozen VLA
  pi05_runs/checkpoints/pi05_droid_finetune/pick_full_v3/39999/ 43 GB  JAX original
  pi05_runs/token_ae/ae_{house2,house10,house21,combined}.pt    4 phase-1 encoders
  rlt_tokens/{house2,house10,house21}/                          52 GB  4608 seqs/scene
  lerobot_pick/merged/                                          95 GB  SFT dataset
  mbdata_pick/                                                  66 GB  raw demos
  venvs/pi05_torch/                                             patched venv (see below)
  bench_collect_{house2,house10,house21}/                       jittered scene benchmarks
```

Three interpreters, not interchangeable:

- `venvs/pi05_torch/bin/python` — **serving the PyTorch pi0.5**. It is a copy of openpi's
  venv with `transformers_replace` applied and `json_numpy` added. openpi's own venv is
  NOT patched, and the model refuses to load without the patch.
- `submodules/molmospaces/.venv/bin/python` — the simulator and the `rlt` package.
- `submodules/openpi/.venv/bin/python` — JAX side, dataset tooling, eval driver.

---

## 3. What the VLA is and how it must be driven

**Model.** pi0.5, fine-tuned on MolmoSpaces Pick. PaliGemma (`gemma_2b`, width **2048**)
reads two camera images plus the instruction; a `gemma_300m` action expert denoises the
action chunk by flow matching. Inference is: `embed_prefix` -> PaliGemma prefix pass
(builds a KV cache) -> 10 denoising steps in the expert.

**Actions (measured).** The model emits `action_horizon=16` actions; openpi returns the
first 8 dims of each:

- `[0:7]` joint **deltas** (not absolute positions)
- `[7]` gripper in 0..1

**The environment executes ABSOLUTE joint targets** (`command_mode["arm"]="joint_position"`)
and gripper in 0..255. So every action must be converted:

```
absolute_arm = action[:7] + current_arm_qpos
gripper      = 255 if action[7] > 0.5 else 0     # demos only ever use 0 or 255
```

**This conversion is the single most important thing in the project.** Feeding raw deltas
to the simulator drives the arm toward a folded pose every step. The published pi0.5
baseline of 0/10 on this benchmark is that bug, not the model: with the conversion fixed,
the *unmodified* pretrained checkpoint scores **2/8** on the same benchmark (measured).

**State** fed to the model is 8 numbers: 7 arm joints, then gripper qpos divided by
`0.824033` and clipped to 0..1 (constant taken from `molmo_spaces/policy/learned_policy/pi_policy.py`).

---

## 4. Chunk size: the unresolved finding, and the current blocker

The model predicts 16 actions; the caller chooses how many to execute before re-planning.
**Measured, 36 episodes per cell, same checkpoint, nothing else varied:**

| chunk | house10 (ladle) | house21 (pot) |
|-------|-----------------|---------------|
| 1     | 36.1%           | **88.9%**     |
| 2     | 50.0%           | 86.1%         |
| 4     | 72.2%           | 77.8%         |
| 8     | **83.3%**       | 58.3%         |

Both trends significant (Fisher p=0.0001 for house10 1->8, p=0.0066 for house21 1->8),
and **they point in opposite directions**. There is no single optimal value.

Why `chunk_size=1` is the formally correct one: the fine-tuning data labels each frame's
action as a delta from *that frame's* state, so applying `action[k]` to the state from
step `t` is wrong by the distance travelled since — **0.027 rad per step, measured** on
demonstrations, reaching 0.32 rad by k=15. Longer chunks lag, which makes motion slower
and smoother; that apparently helps house10 and hurts house21.

**This is why the previous RL results were uninterpretable**: training ran the RL harness
at stride 8 while every benchmark number was produced at chunk 1, so "RL improved things"
could not be separated from "the chunk changed". The user's standing instruction is that
**evaluation must use the same execution regime as training**.

The user was asked, and has not yet answered, whether to fix one chunk size project-wide
(and which), or per scene. **Ask before running any RL experiment.**

---

## 5. RL Token integration: what exists and what is verified

The `rlt/` package (previous work, MolmoAct2-based) is reused. Its layout is documented in
`rl_token/README.md`. Phases: 1) token autoencoder, 2-3) online actor-critic refining the
VLA's chunk.

**Token capture (measured, working).** The RL state is PaliGemma's prefix hidden state:
**(968, 2048)** per step. Deterministic across identical calls (max |diff| = 0.00) and
sensitive to the image (mean |diff| = 0.0416).

**Trap:** openpi calls `paligemma.language_model.forward(...)` *directly*, so
`register_forward_hook` never fires — it only triggers on `__call__`. The working
implementation wraps the bound method instead (`pi05/vla_server.py`, class `FrozenPi05`).
A hook here fails silently and the RL state would be empty.

**Token width is 2048, not MolmoAct2's 2560.** `rlt/config.py` was changed so
`VLA_TOKEN_DIM` reads the env var `RLT_VLA_TOKEN_DIM`, defaulting to 2560 to keep the
MolmoAct2 path intact. **Every pi0.5 run must export `RLT_VLA_TOKEN_DIM=2048`**, or
`rlt/data.py` rejects the corpus. Encoders trained on MolmoAct2 tokens are unusable.

**Two protocols, do not mix them up:**
- `rlt/vla.py` (`VlaClient`) speaks **HTTP** `POST /act` via `requests`, payload
  `{external_cam, wrist_cam, instruction, state}`, response
  `{actions, token_features, token_attention_mask}`, serialised with `json_numpy`.
  Server: `scripts/serve_pi05_http.py` (stdlib `ThreadingHTTPServer`; FastAPI is not
  installed anywhere on this machine).
- The MolmoSpaces evaluator speaks **websocket** (openpi's `WebsocketPolicyServer`).
  Server: `scripts/serve_pi05.py`.

**One inference at a time per server.** pi0.5 inference blocks the event loop; two sim
workers against one server produce `timed out while waiting for handshake` and episodes
get silently skipped (a run reported 7 of 16 episodes and would have given a wrong SR).
Use **one worker per server, one server per GPU**. That is also faster: 8 shards x 1
worker did 128 episodes in ~25 min versus ~58 min for 4 workers on one server.

**Reward (verified, and it is correct).** `rlt/rl_policy.py` gives `+1` on the first
success step; `end_on_success=True` means the episode terminates there, so
`success_count > 0` and `success[-1]` agree. An earlier claim that the criterion was
wrong was itself wrong — do not repeat it.

---

## 6. The three scenes

Each is one val-benchmark episode repeated with ±2 cm horizontal jitter on the object
(`scripts/make_repeat_benchmark.py`). Benchmarks already built under
`pi05_molmospaces/bench_collect_<scene>/`.

| scene | val ep | house | task | note |
|-------|--------|-------|------|------|
| house2 | 117 | 2 | pick up the bottle. | soap dispenser on a toilet lid, elongated. **0/76 successes** at chunk 1. The user kept it deliberately: the policy reaches the right pose and pushes instead of grasping, so it tests whether RL can start from zero reward. |
| house10 | 2 | 10 | pick up the ladle. | 36% at chunk 1, 83% at chunk 8 |
| house21 | 127 | 21 | pick up the pot. | 89% at chunk 1, 58% at chunk 8 |

**Warning about an earlier table.** A scene-profiling run reported 28% for house10 and 58%
for house21. Those came from a sharded run where several candidates landed in the same
house directory and their episodes were counted together. Treat the chunk-sweep numbers
above as authoritative; the profiling numbers are contaminated.

**Gate.** The user asked that RL take over only once the gripper is near the object, and
after watching videos chose **step 40**, **latched** (once open, stays open for the rest
of the episode). Implemented in `pi05/proximity.py` as `step_gate_mask` / `GATE_STEP`,
with tests — **but it was never wired into the training runs.** Wiring it in is part of
the outstanding work.

A distance-based gate was tried and abandoned for a good reason: the evaluator records the
object's pose from the **benchmark spec**, not its live position (`obs_start` and
`obs_end` are constant across an episode, equal to `pickup_obj_start_pose` and
`pickup_obj_goal_pose`). Once the policy nudges the object the distance stops describing
reality. On an elongated object the centre also stays far while the gripper is at a
graspable end. If you want a real distance gate you must pull the live pose out of MuJoCo
during the rollout; do not resurrect the spec-pose version.

Also note `obs/extra/grasp_state_pickup_obj.touching` read 0 across 64 rollouts where the
user could see contact on video. Do not trust that flag.

---

## 7. Results so far

**SFT.** pi0.5 fine-tuned on 25,544 episodes / 1,782,564 frames (33 h of demonstrations,
2,461 houses). 40k steps, effective batch 240-256, ~4.5 h on 6-8 GPUs. Loss 2.79 -> 0.003.
Converged by ~step 20,000: SR at 10k/20k/30k/40k was 27.3% / 37.5% / 36.7% / 36.7% on 128
val episodes; steps beyond 20k added nothing (Fisher p=1.000).

**Benchmark comparison** (128 val episodes, chunk 1): fine-tuned pi0.5 **37.5%**,
PyTorch-converted **40.6%**, MolmoAct2 reference 33.6% and 36.7%. Statistically
indistinguishable from MolmoAct2 (p=0.60, p=1.00).

**JAX -> PyTorch conversion verified.** Same weights, same injected noise: max arm
difference **0.0107 rad**, 2.25% relative — consistent with bf16 activation precision, not
a mapping error. Norm stats had to be copied into the converted directory manually.

**Phase-1 encoders trained.** 4 checkpoints, 8000 steps each, reconstruction loss
1.03 -> 0.017-0.020, on 4608 sequences per scene.

**The 8-run RL matrix was discarded** (2 scenes x 4 encoders) because of the chunk
mismatch described in section 4, plus the missing gate. Its outputs were deleted at the
user's request. Do not cite its numbers.

---

## 8. Traps already paid for

- `np.array(list_of_arrays, dtype=object)` on uniform shapes builds a 3-D object array at
  8 bytes per element — 4x the size of float16. Use `np.stack(...).astype(np.float16)`.
- argparse: `--server-arg --record-tokens` is parsed as a missing value. Use
  `--server-arg=--record-tokens`.
- orbax async checkpointing **deadlocks** on a mid-run save (0 bytes for 10+ minutes,
  timeout is 2 h). `pi05/checkpointing.py` forces synchronous saves; a 43 GB checkpoint
  then takes ~60 s. Also set `keep_period=None` or every 5000th checkpoint is kept forever
  (~250 GB for a 30k-step run).
- A checkpoint saved under one device mesh **cannot be resumed under another**
  (`Sharding passed to pjit does not match`). 6-GPU and 8-GPU runs are not interchangeable.
- openpi's dataloader defaults to `num_workers=2`, which starves 8 GPUs: 15 samples/s
  versus **124.7** with 32 workers. Same fix matters everywhere.
- Importing anything that pulls in JAX inside a CPU job makes it claim a GPU (61 GB seen).
  Set `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu` for CPU work.
- The HF datasets cache must live at
  `/home/jovyan/users/staroverov/.cache/huggingface` (`HF_HOME`), never under `/home/user`.
- Simulator episode logs count lines with duplicates; do not use `grep -c` on them as a
  progress metric.
- `eval_main` writes videos and h5 **in a batch when a house finishes**, not per episode.
  Absence of files mid-run is normal.

---

## 9. Suggested structure for the refactor

The user compared what they want to popular RL libraries. A layout that satisfies the five
requests:

```
rl_token/pi05/
  config.py      dataclasses: paths, SCENES, EvalConfig, RLConfig. Every knob, no flags.
  model.py       Pi05Policy, ChunkExecutor, observation_from_env, to_env_action.
                 The only place actions become environment commands.
  vla_server.py  FrozenPi05: loads the checkpoint, captures prefix tokens. (exists)
  rl_token.py    gate + actor/critic wiring; where the correction enters the chunk
  eval.py        evaluation loop driven by EvalConfig
  train.py       RL training loop driven by RLConfig
  README.md      the guide: what each file is, how to run eval and training, what the
                 numbers mean
rl_token/scripts/
  run_eval.py    thin CLI over pi05/eval.py
  run_train.py   thin CLI over pi05/train.py
```

Existing files worth keeping and folding in: `pi05/vla_server.py`, `pi05/proximity.py`
(gate + tests), `pi05/checkpointing.py`, `pi05/accum.py`, `pi05/merge.py`,
`pi05/token_recorder.py`, `scripts/serve_pi05_http.py`, `scripts/serve_pi05.py`,
`scripts/eval_pi05.py`, `scripts/make_repeat_benchmark.py`.

There are ~100 tests under `rl_token/tests/` (52 from the previous MolmoAct2 work plus
~50 added for pi0.5: action space, converter, merge, serving, accumulation, proximity).
They pass. Keep them passing; they encode most of section 3 and 6.

---

## 10. First things to do

1. Do the refactor and the guide. That is what was asked; do not start experiments first.
2. **Ask the user the chunk-size question** from section 4 before any RL run — it decides
   what the numbers mean.
3. Wire the step-40 latched gate into the RL policy. It exists and is tested but unused.
4. Only then re-run the RL matrix, with training and evaluation in the same regime.

GPUs are currently idle and the user notices. If you need a long job while you write code,
the useful one is a chunk sweep on the **full 128-episode val benchmark** to pick a
project-wide regime — but ask first, since it presumes the answer to question 2.
