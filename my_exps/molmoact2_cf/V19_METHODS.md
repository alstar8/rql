# V19: iterated CFGRL extractor on frozen MolmoAct2 (corrected)

**Status (2026-08-19).** 3 concurrent EGL/GPU aborted in `mjr_readPixels` after the first chunk (~8 steps, then SIGABRT ~10 min). Phase 0 stayed 0/24. Packing is now **4 workers/GPU**, **1 EGL/GPU** (3 wait), machine-wide cap `n_gpu×1` so **both GPUs still render**. Isolated CUDA-free children; parents retry on abort. `runs/rlt_cf_v19_kettle/`. Train split only.

**Intent (unchanged).** Same kettle task, same frozen VLA prior, RLT token + critic + a small actor, in an offline-fit / online-collect loop.

**What changed vs the original V19 sketch.** That sketch stacked CFGRL + CF \(\nabla Q\) guide + Q-max actor, moved the VLA prior (LoRA VLM / full AE), and froze \(Q\) while collecting. CFGRL replaces the *actor / BC extractor*, not the critic and not ConsensusFlow. Molmo stays frozen (no VLM LoRA, no AE full-tune, no joint denoising). One flow net, not actor+AE+guide stacked on the same ODE.

**What the 100-round run showed (do not ignore).**
- Phase SR oscillated 0–19% with **no trend**. A vs next B deltas were \(\pm 12\)pp — binomial noise, not learning. Phase B does not update the actor.
- **Four train poses** (ids 1, 2, 16, 19) sit at ~30–39% SR; most of the other 20 are ~0%. Rotating the extra 8/32 probes by phase made “SR went up” equal “this phase double-counted easy poses.”
- Frozen VLA on this 32-ep set is ~3–8%, **not** the 22% buffer SR (buffer sampling is not this pose set).
- Actor pooled SR **7.4%** \(\approx\) VLA collect SR **7.6%**. Extractor did not beat the prior.
- Phase A `bc` only drifted 0.075→0.051; `adv=False` every round; critic never healthy; `empirical_actor_episodes=0`; collect was **100% VLA** (`p=0`) for ~40 rounds. Gate cannot open: it wants 16 actor collect eps, but collect never uses the actor.
- Actor Phase A used `sample_natural` (~11% success rows) and trained **o=NEG on failures** (~80% of the batch) while **deploy \(w=1\)** threw away the uncond VLA clone. That is not CFGRL.

---

## 0. Why this is the correction

The original sketch stacked three improvement operators (CFGRL + CF \(\nabla Q\) guide + Q-max actor), moved the VLA prior (LoRA VLM / full AE), and froze \(Q\) while collecting. CFGRL’s product-policy theorem needs a **fixed** \(\hat\pi\) and a **fixed** \(A_{\hat\pi}\) during one extraction round. RLT’s critic is the thing that produces \(A\). ConsensusFlow’s BC term is the behavior reference its guide corrects — do not delete it *and* keep the guide.

V19 is therefore: **RLT critic + CFGRL policy extraction + frozen Molmo as \(\hat\pi\)**, iterated.

---

## 1. Task / data (same as V18)

| Item | Value |
| --- | --- |
| Task | house0 `pick up the kettle.` |
| Train poses | 24 specs in `runs/benchmarks/house0_kettle_v13/train` |
| Val holdout | **12 episodes in `.../house0_kettle_v13/val` — unused.** Not for training, not for online validation, not for snapshots, gate, or phase probes. |
| Offline buffer | 1200 reference-VLA trajs, ~22% SR |
| Horizon | 500 |
| Frozen VLA | MolmoAct2-DROID via HTTP (VLM **and** action expert frozen) |
| Init | kettle flow token AE + critic pretrain; **do not** start from collapsed V18 online actors |
| Collect budget | **12 Phase-B eps × \(N\) workers** (this host: \(N=8\) → 96; 8 GPU × 4/GPU: \(N=32\) → 384). Plus phase probes (below). |

**Do not train:** Molmo VLM, Molmo AE (LoRA or full), CF guide.  
**Do not touch the 12-episode val split.** Trainers hard-error if `--benchmark_dir` contains `val`.  
**Flow-only arm.** CFGRL needs score/velocity composition; residual Gaussian cannot do it.

---

## 2. Architecture

```
RGB, proprio, language
        │
   MolmoAct2 (frozen VLM + frozen AE)     ← π̂, reference chunk ã
        │ token embeddings z_{1:M}
   Token AE encoder g_φ  →  z_rl ∈ R^{256}   ← freeze after first Phase A
        │
   x = (z_rl, proprio)
        │
   EnsembleTimeCQL Q_ψ  (N=10)              ← RLT chunk TD, always on
   FlowVelocityActor v_θ(x, a_t, t, o)      ← o ∈ {∅, 0, 1}, shared net
        │
   sample: v = (1-w) v_θ(·, ∅) + w v_θ(·, o=1)
```

One network, optimality embedding, 10% cond-dropout. This *is* the CFGRL coupling: uncond ≈ VLA prior, cond ≈ high-advantage actions. Do not add a second velocity from the AE or from \(G_\phi\).

---

## 3. Losses

### Critic (RLT, unchanged role)

Chunk TD + light CQL/rank. \(Q\) is trained on executed chunks. Labels for CFGRL are stop-grad.

\[
\mathcal{L}_Q
=
\mathbb{E}\big[(\hat Q - Q_\psi(x,a))^2\big]
+ \lambda_{\mathrm{CQL}}\,\mathcal{L}_{\mathrm{CQL}}
+ \lambda_{\mathrm{rank}}\,\mathcal{L}_{\mathrm{rank}}
\]

Advantage vs the frozen VLA (matches RLT’s actor baseline):

\[
A(x,a)=Q_\psi(x,a)-Q_\psi(x,\tilde a),\qquad
o=\begin{cases}
1 & A\ge 0\\
0 & A< 0
\end{cases}
\]

Until critic health (rank loss down, LCB not nonsense), use **episode-success** as the *positive* condition on chunks from successful trajs (CFGRL-GCBC). Switch to \(A\ge 0\) after that. Never use peaked AWR weights. **Do not train a third “failure” velocity** that clones unsuccessful \(a\) — that dominated the last run (~80% of natural batches) and pulled the shared backbone off the VLA clone.

### Actor (CFGRL only — replaces V18 \(\beta\|a-\tilde a\|^2\) and CF BC)

Flow-matching with cond dropout (p=0.1 set \(o=\varnothing\)):

\[
\mathcal{L}_{\mathrm{CFGRL}}
=
\mathbb{E}\big[\|v_\theta(x_t,t,o)-(a^\star-a_0)\|_2^2\big]
\]

Target \(a^\star\):

| \(o\) | \(a^\star\) |
| --- | --- |
| \(\varnothing\) | VLA reference \(\tilde a\) (this *is* \(\hat\pi\)). Failures / \(A<0\) also go here. |
| \(1\) | executed \(a\) with success (or \(A\ge 0\)) |
| \(0\) | **unused.** Do not clone unsuccessful executed chunks. |

Actor batches are **stratified** (`pos_frac=0.5`), not `sample_natural`. That is the “even batch weights” this file already asked for and the last run did not do.

**No** \(\mathcal{L}_{\mathrm{actor}}=-Q\), **no** \(\mathcal{L}_G\), **no** residual \(\beta\) toward \(\tilde a\) (the uncond head already is that prior). Optional tiny endpoint clip \(\|a_{\mathrm{end}}-\tilde a\|\le 0.05\) as a safety ball only.

Sampling:

\[
v=(1-w)\,v_\theta(x_t,t,\varnothing)+w\,v_\theta(x_t,t,o{=}1)
\]

\(w\) is **not** trained. **Start at \(w=0\)** (pure uncond / VLA clone). After each Phase A, log uncond endpoint MSE to \(\tilde a\). Raise to **\(w=0.5\)** only if that MSE \(\le 0.02\). Do **not** deploy \(w=1\) (that discards the prior). Do not jump to 1.5. After uncond clone is good, a later \(w\) sweep \(\{0.5, 1.0\}\) is allowed only if the \(w=0.5\) probe already beats Phase 0.

---

## 4. Two-phase loop (CPI) + per-phase SR

Start from offline kettle replay. **12 CPI rounds.** Every worker uses the same quota so a phase’s SR is always over the **same 24 train poses**.

### Fleet barrier (phases are sequential)

Trainers are separate processes, so without a lock the fast shards race into round \(r{+}1\) while slow shards are still on \(r\). That is why `PHASE_SR.md` used to show several later phases as `pending` at once.

**Fix:** after a worker writes its probe row(s) for phase \(P\) (and on resume, after loading the last recorded \(P\)), it **blocks** until every `flow_cfgrl/shard_*/phase_probe.jsonl` has \(P\). Only then does it start the next step (Phase A fit, Phase B collect, or the next probe). The watchdog may restart a dead shard; the barrier re-reads jsonl and does not need extra marker files.

So the fleet-wide order is strict:

```
all workers × Phase 0 probe  →  all × Phase A + 1A probe  →  all × 1 collect + 1B probe  →  …
```

`PHASE_SR.md` should have at most **one** in-progress phase (plus empty future rows). A mid-run enablement resyncs at the current frontier (fast shards idle until the slowest catches up to their last recorded phase), then stays locked.

Disable with `V19_PHASE_BARRIER=0` (passes `--no_cfgrl_phase_barrier`). Default on.

### Phase SR probe (the number we plot)

After **every** phase, the fleet covers **all 24 train poses** with a **fixed** assignment (no rotation by phase). 2 workers: 12 poses each (even / odd). 8 workers: poses \(s, s{+}8, s{+}16\). 32 workers: pose \(s \bmod 24\) (workers 24–31 repeat 0–7). **Do not rotate extras by phase** — that made 0% vs 19% look like learning. **No weight update, not added to replay.** Never val. Compare phases only on this identical set.

| Phase | Policy on the probe | When |
| --- | --- | --- |
| **0** | frozen VLA reference | before any CFGRL fit |
| **\(r\)A** | CFGRL actor at current \(w\) | immediately after that round’s Phase A fit |
| **\(r\)B** | CFGRL actor at current \(w\) | immediately after that round’s 1 collect ep / worker |

Read `runs/rlt_cf_v19_kettle/PHASE_SR.md` (and `phase_sr.jsonl`). Example of the intended view:

```
Phase 0  → 30%
Phase 1A → 32%
Phase 1B → 35%
Phase 2A → 37%
Phase 2B → 39%
…
```

### Phase A — fit on **all** replay

Not 1 epoch. Critic needs bootstraps on a 500-step sparse task.

1. Freeze Molmo. Token AE: train only in **round 0**, then freeze (RLT).
2. Critic: **K_Q updates** (round 0: 2048; later rounds: 1024; batch 128).
3. Snapshot \(Q_{\psi'}\) (stop-grad teacher).
4. Relabel every replay chunk with \(o\) from \(Q_{\psi'}\) (or success bit if critic not healthy).
5. Actor: **K_π CFGRL updates** (round 0: 1024; later: 512) on those labels. Stratified batches (`pos_frac=0.5`). Log `bc`, `pos_frac`, uncond/cond endpoint MSE to \(\tilde a\).
6. Set \(w\) from uncond MSE (0 or 0.5). Run the **\(r\)A** probe on the **fixed** pose set.

### Phase B — collect **1** episode per worker (\(N\) pooled)

1. Molmo, token, **and the CFGRL weights used to act** are held fixed for the collect episode (clean one-step extract).
2. **Critic keeps doing TD** on incoming chunks (16 updates/ep). Do **not** freeze \(Q\) while collecting.
3. Each collect episode draws a **uniform random train pose** in \(\{0,\ldots,23\}\) (per-worker RNG). Do not walk `0,1,2,…` and do not give every worker the same `--start_episode`.
4. Collect **episode-level mixture**. Do **not** wait for the V18 empirical gate (16 actor collect eps): that gate can never open if collect is p=0, which is exactly what happened.
   - After the \(r\)A barrier: if latest **complete \(r\)A** probe SR \(\ge\) Phase 0 SR \(- 1/n\), **p=0.25** actor / 0.75 VLA (\(n=\) probe count, 24 on 2 GPU / 32 on 8 GPU). Ignore \(r\)B for this decision (\(r\)B is a noisy repeat of the same actor).
   - Otherwise **p=0** (VLA only). If the extractor is not at least tying the prior, do not collect from it.
5. Run the **\(r\)B** **actor** probe on the **same** poses as \(r\)A. Expect \(r\)B \(\approx\) \(r\)A unless that round’s collect mixed actor and the next Phase A has not run yet — \(r\)B is a noisy repeat, not a new extractor.
6. If \(r < 12\), go to Phase \((r+1)\)A.

Round 0 Phase A is the offline pretrain (1200 trajs only). Do not collect from a randomly initialized cond head. Phase 0 is the VLA baseline **before** that fit.

---

## 5. \(w\) and collect schedule

| Round | Collect eps (pooled) | \(w\) deploy | Notes |
| --- | --- | --- | --- |
| 0 | 0 | 0 | Phase 0 VLA probe, then Phase A; uncond must clone \(\tilde a\) |
| 1–12 | **1 / worker** (2 GPU: 8; 8 GPU: 32) | 0 until uncond MSE \(\le 0.02\), then **0.5** | Mixture p=0.25 only if latest complete **rA** probe \(\ge\) Phase 0 \(-1/n\) |

If uncond endpoint MSE to \(\tilde a\) is not small after round 0, **keep \(w=0\)**. A \(w=1\) “cond BC” policy with an undertrained o=1 head is worse than the VLA.

---

## 6. Explicit non-goals (do not reintroduce)

- LoRA / full-tune Molmo VLM or AE
- `v = v_{\mathrm{AE}} + v_{\mathrm{actor}}\) or CFG mix of AE and compact actor (different state, 15×8 vs \(C\times 8\), q01–q99 vs compact)
- CF guide \(G_\phi\) on this arm (ablate later **instead of** CFGRL, not with it)
- V18 Q-max actor + \(\beta\|a-\tilde a\|^2\) on the same net
- `--always_collect_actor`
- 1 epoch critic
- Residual-Gaussian CFGRL

---

## 7. Success criteria

1. **Phase 0** (VLA, same 24 poses every later phase) is the baseline. The 1200-traj buffer SR (~22%) is **not** this number.
2. Report the chain **0 → 1A → 1B → … → 12B** from `PHASE_SR.md`. CFGRL is working only if some later actor probe beats Phase 0 **on this fixed set** by more than ~1/n.
3. Uncond flow clones \(\tilde a\) (uncond endpoint MSE \(\le 0.02\)) before any \(w>0\).
4. Stretch goals (0.5 / 0.8 window SR) are **not** in scope for this run.

Ablation if it works: same loop with V18 actor loss instead of CFGRL (isolates extractor). Do not add CF until that comparison exists.

---

## 8. Code map

| Piece | Change |
| --- | --- |
| `FlowVelocityActor` | add optimality embedding \(o\in\{\varnothing,0,1\}\) |
| `actor_step` | binary CFGRL (uncond=\(\tilde a\), cond=success); stratified batches; no o=NEG failure BC |
| `guide_step` | **off** (`use_cf_guide=False`) |
| `v19_harness.py` | flow-only; fixed `phase_probe_episode_idx`; `cfgrl_collect_mixture_prob`; `wait_for_phase_barrier`; train split only |
| Online loop | Phase 0 probe → **barrier** → A fit → set \(w\) from uncond MSE → \(r\)A probe → **barrier** → B collect (mixture vs Phase 0) → \(r\)B probe → **barrier** |
| Eval | **phase probe = same 24 train poses every phase**; **never** the 12-ep val split |

---

## 9. Launch (auto GPU packing)

Packing: **1 Molmo HTTP server / GPU** + **4 trainers / GPU**. This machine: **2 servers** (ports **8760–8761**) + **8 trainers**. 8-GPU box: 8760–8767 + 32 trainers. EGL: `RLT_EGL_PER_GPU=1` (3 of 4 workers wait) and `RLT_EGL_MAX_CONCURRENT=n_gpu` so **both GPUs render**. Do not set per-GPU EGL above 1 on this host — 3 concurrent contexts abort after the first chunk. Each env episode runs in a **CUDA-free child** (`RLT_ISOLATED_ROLLOUT=1`) so parent trainers survive `mjr_readPixels` SIGABRT. Run dir `runs/rlt_cf_v19_kettle`. With 8 workers each phase probe is **24/24** (poses \(s,s{+}8,s{+}16\)). Collect pooled = \(8\times12\times1=96\). Changing shard count requires `FRESH=1`. Resume without wiping probes: `FRESH=0`.

```bash
GPU_IDS=0,1 V19_MODE=long FRESH=1 bash launch_v19_rlt_cfgrl.sh
```

Stop: `bash stop_run.sh runs/rlt_cf_v19_kettle`

Live table: `runs/rlt_cf_v19_kettle/PHASE_SR.md` (rewritten every watchdog poll). Per-shard rows: `flow_cfgrl/shard_*/phase_probe.jsonl`. Collect pose is random; **probe poses are fixed per worker**. After each probe the worker waits until all shards have that phase.

The 12-episode val directory is recorded in `MANIFEST.json` as holdout and is never passed to `--benchmark_dir`.
 