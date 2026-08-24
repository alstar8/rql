# RL Token on MolmoAct2 / MolmoSpaces

Implementation of RL Token (arXiv 2604.23073, Physical Intelligence): a frozen
VLA supplies the RL state (the *RL token*) and the reference action chunk, and a
small actor-critic refines that chunk online. The paper's LaTeX source is
vendored at `submodules/RLT/`; `sec/3.method.tex` is the authority on the
formulas and `sec/7.appendix.tex` on the hyperparameters.

## Layout

```
rlt/
  config.py          every hyperparameter, phase 1 and phases 2-3
  cli.py             dataclass -> argparse
  logging_utils.py   JSONL + TensorBoard
  mlspaces_env.py    asset paths MolmoSpaces reads at import time

  token_ae.py        phase 1: RL token encoder/decoder, L_ro   Eq. 1, 2
  train_token_ae.py  phase 1 training
  data.py            token-shard reader + memmap cache
  serve_vla.py       frozen-VLA HTTP server (actions + final-layer tokens)

  networks.py        actor (Gaussian, fixed sigma) and twin critic
  agent.py           critic TD backup and actor objective       Eq. 3, 5
  replay.py          chunk transitions + replay buffer
  vla.py             VLA client and the frozen encoder -> z_rl
  rl_policy.py       the MolmoSpaces policy: rolls out and records
  rollout.py         one benchmark episode at a time
  learner.py         Algorithm 1's inner loop (store, then G updates)
  train_online.py    phases 2-3 entry point, one collector
  train_parallel.py  the same, with K collectors feeding one learner
  shared.py          the two channels between them: rows out, weights back
  evaluate.py        success rate: one episode repeated, or the benchmark
tests/               31 tests, CPU only
scripts/
  profile_episodes.py     success rate per candidate episode, in parallel
  replay_learner.py       replay a saved buffer through the learner, no env
  prepare_token_cache.py  flatten the corpus into a shared memmap
  launch_ae_sweep.sh      phase 1 across GPUs
  watch_sweep.sh          GPU load + per-run progress
  compare_sweep.py        rank runs fairly
  validate_data.py        inspect the token shards, change nothing
  jsonl_to_tb.py          rebuild TensorBoard from JSONL
  summarize_eval.py       success rates from an eval run
  make_grid_video.py      tile episode videos 4x4
  egl_device_map.py       MUJOCO_EGL_DEVICE_ID -> physical GPU
  clean_asset_remnants.py clear half-extracted asset dirs
```

## Paths

`$HOME` is `/home/user`, but the work lives under `/home/jovyan`. Never use `~`.

```
code   /home/jovyan/users/staroverov/B1K/B1K_AIRI/ReseachOS/projects/behavior-openpi-oar-rl/method_cards/rl_token
data   /home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/rql/my_exps/molmoact2_cf/runs/rlt_pretrain_demo1k
python /home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python
server /home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmoact2/.venv/bin/python
assets /home/jovyan/users/staroverov/B1K/mlspaces/assets
```

The two venvs disagree about torch (molmoact2 2.5.1, molmospaces 2.7), so the
VLA lives in its own process and everything talks to it over HTTP.

## Running

Tests:

```bash
/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python -m pytest tests -q
```

One frozen-VLA server per GPU (13 GB each, ~420 ms per call):

```bash
CUDA_VISIBLE_DEVICES=0 /home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmoact2/.venv/bin/python -m rlt.serve_vla --port 8000 --device cuda:0
```

Pick a training episode (only `episode_idx >= 128`; 0-127 are the test set):

```bash
python scripts/profile_episodes.py --episodes 128-135 --rollouts 30 --shards 3 --parallel 12 --ports 8000,8001,8002,8003
```

Train, and measure:

```bash
python -m rlt.train_online --episode_idx <k> --episodes 400 --out_dir runs/online_<k>
python -m rlt.evaluate --episode_idx <k> --repeats 50 --actor runs/online_<k>/agent.pt
python -m rlt.evaluate --max_episodes 128 --num_workers 4 --actor runs/online_<k>/agent.pt
```

Four collectors instead of one, continuing an existing run:

```bash
python -m rlt.train_parallel --episode_idx <k> --episodes 400 --collectors 4 --ports 8000,8001,8002,8003 --encoder_devices cuda:0,cuda:1,cuda:2,cuda:3 --egl_devices 0,1,2,3 --out_dir runs/online_<k> --resume
```

`train_parallel` is a deliberate deviation: Algorithm 1 has one rollout stream.
The buffer is off-policy either way and UTD is counted per stored transition, so
K collectors change the wall clock and nothing else. Collectors are `spawn`
processes (torch requires it for CUDA in children; forked ones hang), they ship
transitions over a queue, and they reload the actor from a file the learner
replaces every few seconds -- staleness of a few seconds, matching the paper's
asynchronous learner.

## What the paper fixes, and what we chose

The paper leaves several values unstated. These were decided by the project
owner rather than picked by default, and live in `config.py`:

| | value | source |
| --- | --- | --- |
| chunk length C | 8 | ours -- what the MolmoSpaces client executes, so the baseline stays comparable (paper: 10) |
| subsampling stride | C, i.e. none | ours -- see below (paper: 2) |
| sigma | 0.02 | ours -- half of pi_vla's own spread of 0.044 |
| beta | 1.0 | ours -- a deviation the size of the VLA's own noise then costs about as much as a plausible Q gain |
| gamma | 0.99 per env step | ours |
| UTD | 5 per stored transition | ours (paper says 5, without naming the unit) |
| critic:actor | 2:1 | paper |
| N_warm | 40 episodes | ours |
| networks | 2-layer MLP, width 256, twin Q, min for targets | paper |
| reference dropout | 50% | paper |
| optimizer | Adam 3e-4, batch 256, tau 0.005 | ours -- TD3 defaults |
| reward | +1 on success, 0 otherwise | paper (the operator's sparse label; here the env's success flag) |
| RL scope | the whole episode | ours -- the paper's critical phase is delimited by a human operator, which simulation has none of |
| interventions | none | ours -- impossible in simulation |
| TD target | clipped to [0, 1] | ours -- forced, see below |

### One scene works, 872 scenes do not

Training on the whole Pick benchmark minus the test set -- episodes 128-999,
872 scenes, round-robin, three visits each, 2616 episodes over 10 hours --
produced no improvement at all. Compared within a scene, so scene difficulty
cancels:

```
visit 1 -> 2   25.7% -> 27.6%   106 improved, 89 regressed   McNemar p = 0.25
visit 1 -> 3   24.9% -> 26.6%   105 improved, 92 regressed   McNemar p = 0.39
visit 2 -> 3   27.5% -> 26.6%    81 improved, 88 regressed   McNemar p = 0.64
```

On the 128 scenes whose first visit was warmup, the comparison is directly
against the frozen VLA, and the direction is if anything negative:

```
VLA           41/128 = 32.0%
actor visit 2 37/128 = 28.9%   p = 0.60
actor visit 3 31/128 = 24.2%   p = 0.14
```

On the held-out test set -- the 128 episodes the actor never trained on -- it
lands exactly on the baseline:

```
RL Token actor  41/128 = 32.0%   95% CI [24.6%, 40.5%]
frozen VLA      43/128 = 33.6%   Fisher p = 0.89
frozen VLA      47/128 = 36.7%   Fisher p = 0.51   (a second, independent run)
```

No gain and no significant loss: after ten hours of online RL across 872
scenes, the actor is the base policy with noise on top.

The mechanism is in the critic, and it is the same one that killed the earlier
offline attempt:

```
                 Q(actor)   Q(actor) - Q(reference)   ratio   deviation
one scene         0.2726            0.0323            0.118     0.0204
872 scenes        0.0294            0.0003            0.009     0.0082
```

Across 872 scenes the critic is flat in the action -- the gap is nine
thousandths of Q's own scale, and negative at several points during the run.
The actor then barely deviates (beta pulls it to the reference and Q offers no
direction to trade against), and what deviation remains is approximation noise,
which is what costs those few points.

Why: counterfactual coverage comes from revisiting a state with a different
action. One scene repeated 300 times supplies that; 872 scenes visited three
times each never do -- a state is essentially never seen twice. More scenes at
a fixed episode budget therefore makes the critic worse, not better. This is a
property of the data, not of the implementation.

### Why the stride was dropped

The paper stores `<x_0, a_{0:C}>, <x_2, a_{2:C+2}>, ...`. Every component of the
intermediate row is real -- the robot was in that state, executed those commands,
got that reward -- but the action is spliced from two decisions and the stored
reference is the plan the VLA would have proposed at that step, which was never
executed. Measured on warmup data, where the VLA itself is acting, the two
differ by 0.020-0.055 per coordinate. Two consequences:

* The critic's only data point at such a state sits ~0.04 away from where beta
  anchors the actor, so the actor's anchor is a place the critic never measured.
  Three rows in four are built this way.
* `action_coverage` in warmup read 0.041, and that is *not* counterfactual
  coverage. No state was ever tried with two different actions; it is one
  behaviour described against a different reference. At stride C the same
  measurement reads exactly 0.000 until the actor starts to deviate.

At stride C every row is one decision: the action is the chunk that was chosen,
the reference is what the VLA proposed for that same state, nothing is stitched.
The cost is four times fewer rows per episode -- affordable here, because the
first run reached its plateau after ~50 actor episodes, so data volume was never
the binding constraint.

### The critic diverges without a bounded target

Eq. 3 bootstraps through the online actor, `a' ~ pi_theta`. Run literally, the
actor maximises Q, the target is then computed *through the action the actor
chose*, and the two chase each other. Measured during warmup, on data the frozen
VLA produced, so no rollout of ours is implicated:

```
ep3  q=0.140  actor_q=0.162   ep5  q=0.498  actor_q=0.795
ep4  q=0.173  actor_q=0.292   ep6  q=1.3e6  actor_q=3.7e9
```

`scripts/replay_learner.py` replays that run's saved buffer (1848 rows, 8
rewarded) through six variants, same seed and same data, 9000 iterations:

```
variant                       q_actor  verdict
paper_literal               1.028e+11  diverged at ~2700
target_actor                1.675e+11  diverged at ~3600
clip_target                    0.1524  stayed in [0,1]
target_actor+clip              0.1339  stayed in [0,1]
layer_norm                    0.05478  stayed in [0,1]
layer_norm+target_actor       0.07572  stayed in [0,1]
```

A target actor -- the obvious reading of "we follow TD3" -- does **not** prevent
it; it buys 900 iterations. What works is bounding the target, or normalising
the critic trunk. We clip: the reward is a single +1 on a terminal step and
gamma < 1, so the true return is inside [0, 1] by construction and the clip
removes nothing that is real. The architecture stays the one the paper
specifies. `clip_target` therefore defaults to true -- a default that reliably
diverges is a trap -- and `target_actor`, `target_smoothing` and
`critic_layer_norm` exist as flags but are off.

## Phase 1 result

Seven configurations, 20000 steps, compared at a common step with averaging.
Single log lines swing by ~0.18 and must not be compared directly - that is
what `compare_sweep.py` exists for.

```
run                      recon    spread      config
wide_deep               3.2086    0.1790      z=512   d512 L4
very_wide_deep          3.2117    0.1702      z=1024  d512 L4
deep_decoder            3.2145    0.1840      z=256   d512 L4
deeper                  3.2254    0.1776      z=256   d512 L6
wide_bottleneck         3.5048    0.1874      z=512   d384 L3
reference               3.8811    0.2126      z=256   d256 L2
tight_bottleneck        3.9007    0.2216      z=128   d256 L2
```

- **Token width does not matter.** At fixed capacity, z=256/512/1024 give
  3.2145/3.2086/3.2117 - a 4x range in the bottleneck changes nothing.
- **Capacity matters and saturates at d512 L4.** d256 L2 -> 3.88, d384 L3 ->
  3.50, d512 L4 -> 3.21; L6 gives nothing further.

Checkpoints in `runs/ae_sweep/`. `deep_decoder.pt` is the one to use downstream:
tied for best and the narrowest z, so the RL state stays 256+16 instead of
512+16.

## Milestone 1: one episode, one scene

Before the benchmark, RL has to improve a single fixed episode -- that separates
"the mechanics are broken" from "there is not enough data", and it is the only
setting where the actor tries different actions *in the same states*.

Candidates were profiled with 30 rollouts of the frozen VLA each
(`scripts/profile_episodes.py`, episodes 128-135, 44 min on 12 processes):

```
episode  128    129    130    131    132    133    134    135
SR       20.0%  53.3%  76.7%   0.0%   3.3%   0.0%  36.7%  23.3%
```

**Episode 134 was chosen** -- mid-range, and its rate matches the 33.6% the VLA
scores over the whole test set. Its baseline, over 60 rollouts:

```
frozen VLA on episode 134: 24/60 = 40.0%   95% CI [28.6%, 52.6%]
```

At n=60 against a 40% base, a difference under about 18 points is not
distinguishable from noise, so the actor has to reach roughly 58% to count.

### Result

The actor trained on episode 134 alone: 20 warmup episodes executing the VLA,
then 380 online. Both arms were then measured **at the same time, on the same
four GPUs**, 60 rollouts each, the actor at its deterministic mean with no
exploration noise:

```
                     rate            95% CI       steps to success (successes only)
frozen VLA      20/60 = 33.3%   [22.7%, 45.9%]    median 143  (n=20)
RL Token actor  40/60 = 66.7%   [54.1%, 77.3%]    median 104  (n=40)

success:  +33.3 points, Fisher exact one-sided p = 0.0002
speed:    Mann-Whitney one-sided p = 0.0028
```

Steps are compared **among successful rollouts only**. A failure always runs the
full 500-step horizon, so mean steps over all rollouts mostly restates the
success rate; `rlt.evaluate` therefore records `steps_per_rollout` and
`scripts/compare_rates.py` filters by outcome before testing.

The baseline was also measured before training, on a different day and with
different GPU load: 24/60 = 40.0%. The two agree (Fisher p = 0.57); pooled,
the frozen VLA is 44/120 = 36.7% on this episode, in line with the 33.6% it
scores across the whole test set.

**All of the learning happens in the first ~50 actor episodes.** Against the
rest of the run:

```
first  25 actor episodes: 0.320   rest: 0.585   p = 0.009
first  50:                0.420   rest: 0.590   p = 0.018
first 100:                0.520   rest: 0.585   p = 0.156
first half (191):         0.539   rest: 0.597   p = 0.151
```

so ~350 of the 400 episodes bought nothing. Splitting the run in half shows no
improvement at all -- not because there was none, but because it was over long
before the midpoint. Beware reading 25-episode windows: at p = 0.57 their 95%
band is 0.37 to 0.76, wide enough to manufacture any trend you like.

Throughout, the critic stays inside [0, 1] (Q ~ 0.08, close to gamma^150 * SR),
prefers the actor's chunk to the VLA's by ~0.03, and the buffer's action
coverage grows from 0.041 to 0.080 as the actor learns to deviate.

Note what this does and does not show. It is the same episode the actor trained
on, which is the point of the milestone -- it tests the mechanics, not
generalisation. Generalisation is untested.

Videos: `rlt.evaluate --video_dir <dir>` keeps one mp4 per rollout, named for
its outcome, and `scripts/make_grid_video.py` tiles 16 of them with a green
border on success and a red one on failure. Grids for both arms are under
`runs/final_actor/grid` and `runs/final_baseline/grid`.

## Measurements worth keeping

- **Frozen-VLA baseline on MS Pick-v1.1, first 128 episodes: 43/128 = 33.6%**
  (95% CI [26.0, 42.1]). A second independent run scored 47/128 = 36.7%, so
  run-to-run spread is ~3 points; anything smaller is noise. Output kept at
  `eval_output/rlt_reference_128ep/`.
- **Successes are early, failures are not.** Over 90 successful episodes the
  median success step is 126 and the 90th percentile is 395; every failure runs
  the full 500-step horizon. A 250-step horizon would keep 78% of the successes.
- **`pi_vla` is stochastic.** Five POSTs with an identical observation return
  actions differing by 0.044 std, against a per-step motion of 0.0121 and an
  action magnitude of 0.455. Action diversity in an on-policy buffer therefore
  exists without adding any exploration noise.
- **Cost, measured directly.** `/act` is 420 ms round-trip, of which 393 ms is
  the forward pass inside the server; the 481x2560 token payload costs 6.6 MB
  and ~27 ms. An episode is 89 s at stride 8 and 180 s at stride 2 (500 steps,
  63 or 250 calls, ~20 s of scene setup and HDF5 writing). One server sustains
  about three collectors; four servers fit on the four H100s at 13 GB each.
- **`chunk_replay_merged.npz` has `executed_actions == reference_actions` in all
  24,888 rows.** Purely on-policy, no counterfactual actions anywhere: unusable
  as an off-policy buffer, fine as a source of token sequences. `ChunkReplay`
  reports `action_coverage()` for exactly this reason -- if it is ever 0, the
  critic has nothing to learn from.
- **EGL device order is not CUDA order** on this box:
  `MUJOCO_EGL_DEVICE_ID` 0/1/2/3 -> GPU 3/2/0/1. Use `scripts/egl_device_map.py`.
- **`qvel` needs no patch.** The commented-out `qvel` sits in `RobotStateSensor`
  (uuid `robot_state`); the sensor set actually used adds
  `RobotJointVelocitySensor(uuid="qvel")`, so `obs["qvel"]["arm"]` is there.
- MolmoAct2-DROID is 5.44 B parameters (10.9 GB in bf16), 481 tokens x 2560 per
  call.
```
