# V18: RLT + CF on house0 kettle (≤2k online episodes)

**Goal.** Reach ~**80%** sim success rate (SR) with residual/flow **CF** online RL within **≤2000** valid episodes per shard, without collapsing below frozen-VLA SR during warmup.

**Status (2026-08-17).** Online live under `runs/rlt_cf_v18_kettle/` (ports **8750–8757**). Early (~120–140 eps): residual cumSR **~0.09–0.14**, flow **~0.06–0.11**; gate **closed** (`actor_lcb_below_threshold`). Still in BC warmup (target 400). Not yet competitive with frozen VLA (~0.20–0.22).

Launch: `V18_MODE=long FRESH=1 bash launch_v18_rlt_cf.sh`

---

## 1. Task setup

| Item | Value |
| --- | --- |
| Simulator | MolmoSpaces / ProcTHOR house **0**, task **`pick up the kettle.`** |
| Benchmark | `runs/benchmarks/house0_kettle_v13/train` (24 pose specs, cycle 24) |
| Robot / cameras | Franka-style; wrist + exo RGB → MolmoAct2 VLA server |
| Horizon | 500 env steps / episode |
| Success | Binary episode success from eval pipeline |
| Frozen VLA | MolmoAct2-DROID (tokens / `rl_token` feature mode) |
| Offline demos | 8×150 = **1200** reference-VLA rollouts on the same kettle benchmark → `runs/rlt_pretrain_house0_kettle/` (merged chunk + token NPZs) |
| Offline SR | ~**0.22** episode SR (VLA baseline on this task) |

Online and offline share the **same** house0 kettle distribution (domain-matched vs older MolmoBot+DROID demo1k).

---

## 2. Architecture

Frozen VLA produces token embeddings; a small AE compresses them to an RL state; lightweight actor/critic/guide refine VLA action chunks.

```
RGB, proprio, language
        │
   MolmoAct2 VLA (frozen)
        │  token embeddings z_{1:M}
   Token AE encoder g_φ  →  z_rl ∈ R^{256}
        │
   x = (z_rl, proprio)
        │
   π_VLA → reference chunk ã ∈ R^{C×A}
        │
   ┌────┴────┐
   │ Actor   │  residual: ChunkGaussianActor
   │         │  flow:     FlowVelocityActor (T=10)
   └────┬────┘
        │ a (optional + CF guide Δ)
   Ensemble critic Q_ψ  (N=10)
   CF guide G_ω         (residual CFGradientGuide / flow FlowCFGuide)
```

| Module | Spec |
| --- | --- |
| RL token dim | `z_dim = 256` |
| Token transformer | `d_model = 512`, **4** layers, **4** heads |
| Critics | Ensemble **N = 10**; residual `EnsembleCQL`, flow `EnsembleTimeCQL` |
| Actor | Residual Gaussian chunk / flow velocity field (`flow_steps = 10`, `guidance_coef = 0.5`) |
| Guide | Consensus-flow style: distill stochastic critic gradient into additive Δ on actor/ref |
| Chunking | Non-overlapping action chunks; replay capacity 50k; `pos_frac = 0.5` |
| Init ckpts | `runs/rlt_pretrain_house0_kettle_v18_success_bc/` (residual success-BC; flow copied from V17 kettle pretrain) |

---

## 3. Method

### 3.1 Offline (once)

1. **Collect** reference VLA rollouts on kettle (+ export tokens).
2. **Token AE** warmup: reconstruct VLA embeddings from `z_rl` (Eq. readout loss below).
3. **Residual critic** offline TD/CQL on kettle chunks; then **success-only actor BC** (`q_coef=0`, β=5, 8k steps, `pos_frac=1` sampling).
4. **Flow** critic+actor: reuse V17 kettle flow ckpt (no extra success-BC).

### 3.2 Online (per episode)

1. **Gate** decides deploy policy (`deploy_policy=gated`): deploy actor only if Q-LCB and empirical mixture gates pass.
2. **Collect** episode-level mixture (not always-π):
   - Residual: **p=0.05** pre-gate, **p=0.25** post-gate.
   - Flow: **p=0** pre-gate (reference only), **p=0.25** post-gate.
3. Store chunks in replay; run ≤128 updates / episode (60s cap).
4. **Curriculum**
   - Episodes `[0, 400)`: actor **BC** (`q_coef=0`).
   - `[400, 600)`: linear **q_coef ramp** 0→1 over 200 eps.
   - After: full clipped-Q actor + CF guide.
5. Always-collect-after-BC is **off** (avoids V16 SR collapse).
6. Explore noise: residual std **0.01** on actor deltas.

### 3.3 Deploy / empirical gate

Actor opens only if all hold (after `g_start_episodes=40`):

- Critic healthy; paired **LCB**(Q(a)−Q(ã)) ≥ `0.003`; action sensitivity ≥ `0.003`.
- Empirical mixture: ≥16 actor and ≥16 ref episodes; Wilson LCB of (SR_actor − SR_ref) ≥ `0.0`.

---

## 4. Losses

Notation: `x=(z_rl, s^p)`, reference chunk `ã`, executed/actor chunk `a`, discount `γ=0.99`, chunk length `C`.

### Token AE (offline)

\[
\mathcal{L}_{\mathrm{ro}}
=
\mathbb{E}\Big[\sum_i
\big\| h_\phi\big(d_\phi([\mathbf{z}_{\mathrm{rl}},\bar{\mathbf{z}}_{1:i-1}])\big)_i
- \bar{\mathbf{z}}_i\big\|_2^2\Big]
\]

### Critic (offline + online)

TD on chunks + CQL / rank auxiliaries (defaults: `cql_coef=0.1`, `rank_coef=1`, `far_rank_coef=0.5`, `shuffle_rank_coef=0.5`, `mc_coef=0.1`, `ref_dropout=0.5`):

\[
\mathcal{L}_Q
=
\mathbb{E}\big[(\hat Q - Q_\psi(x,a))^2\big]
+ \lambda_{\mathrm{CQL}}\,\mathcal{L}_{\mathrm{CQL}}
+ \lambda_{\mathrm{rank}}\,\mathcal{L}_{\mathrm{rank}}
\]

\[
\hat Q
=
\sum_{t'=1}^{C}\gamma^{t'-1} r_{t'}
+ \gamma^C\,\mathbb{E}_{a'\sim\pi}\big[Q_{\psi'}(x',a')\big]
\]

### Actor (RLT soft BC + optional Q)

Implemented residual form (advantage vs reference, soft β, learned α trust):

\[
\mathcal{L}_\pi
=
-\,\alpha_Q\cdot\mathrm{clip}(Q_\psi(x,a)-Q_\psi(x,\tilde a),\,[-c_A,c_A])
+ \beta\,\|a-\tilde a\|_2^2
+ \alpha\,\|a-\tilde a\|_2^2
\]

with V18: `β=5`, `α_Q=q_coef∈[0,1]` (curriculum), `c_A=0.05`, hard residual ball `|Δ|≤0.02`, train ref-dropout `0.5`, optional `actor_cql_coef=0.1`.  
**BC phase:** `α_Q=0` (pure BC toward `ã`).

Flow actor uses the same BC/Q schedule on velocity / endpoint objectives (`actor_q_coef` analog).

### CF guide

Guide predicts raw field `W=G_ω(x,ã)`; distill stochastic target-critic gradient `z`:

\[
\mathcal{L}_G = \|W - z\|_2^2
\]

Advantage / magnitude of guided Δ are diagnostics only (not in the train loss). Deploy: `a ← a_actor + Δ_G` when guide gated on.

---

## 5. Experiment design

| Arm | GPUs | Ports | Init ckpt | Collect mixture |
| --- | --- | --- | --- | --- |
| `residual_rlt_cf` | 0–3 | 8750–8753 | success-BC residual | 0.05 → 0.25 |
| `flow_rlt_cf` | 4–7 | 8754–8757 | V17 kettle flow | 0 → 0.25 |

| Hyperparameter | Value |
| --- | --- |
| Max valid episodes / shard | **2000** |
| Target env steps | 1.2e6 |
| Updates / episode | 128 (≤60s) |
| Batch size | 128 |
| Actor BC / Q ramp | **400** / **200** |
| Actor β | **5.0** |
| Explore residual std | **0.01** |
| Replay `pos_frac` | **0.5** |
| Snapshots | 0,100,200,400,700,1000,1500,2000 |
| Seeds | per-shard offsets from 20260813 |

**Controls / success criteria**

1. Eps 0–400: cum/window SR stays near VLA (**≥~0.20**); no collapse to ~0.08.
2. Emp actor SR trends above ref before gate; gate opens by ~600–800 if healthy.
3. By ≤2k: window SR ≥0.5 first, stretch toward ~0.8.

---

## 6. Results

### Offline

| Stage | Artifact | Note |
| --- | --- | --- |
| Kettle collect | `runs/rlt_pretrain_house0_kettle/` | 1200 ref eps; ~0.22 SR |
| V17 kettle pretrain | `runs/rlt_pretrain_house0_kettle_z256_d512_l4/` | token / residual / flow |
| V18 success-BC | `runs/rlt_pretrain_house0_kettle_v18_success_bc/` | residual actor 8k success-only BC; flow copied |

### Online V18 (live, ~120–140 / 2000 eps)

| Arm | Shard | eps | cumSR | winSR | Gate | Emp actor / ref |
| --- | --- | --- | --- | --- | --- | --- |
| residual | 0 | 140 | 0.114 | 0.10 | closed | 0.00 / 0.12 |
| residual | 1 | 140 | 0.114 | 0.14 | closed | 0.20 / 0.11 |
| residual | 2 | 140 | 0.136 | 0.20 | closed | 0.30 / 0.12 |
| residual | 3 | 130 | 0.085 | 0.12 | closed | 0.20 / 0.08 |
| flow | 0 | 130 | 0.062 | 0.12 | closed | — / 0.06 |
| flow | 1 | 120 | 0.108 | 0.12 | closed | — / 0.11 |
| flow | 2 | 120 | 0.092 | 0.12 | closed | — / 0.09 |
| flow | 3 | 120 | 0.108 | 0.12 | closed | — / 0.11 |

- Phase: **`bc_warmup`** (`q_coef=0`); collect mostly **`reference`**.
- Critic health: OK (updates running); actor LCB ≈ 0 or negative → gate block `actor_lcb_below_threshold`.
- Emp mixture underpowered (few actor episodes at p=0.05 / flow p=0).
- **Verdict so far:** below VLA baseline; criterion (1) failing early. Continue through BC=400; if still ≪0.20 with gate closed, revise method (data / mixture / BC / gate).

### Prior online (V17, stopped; for context)

Same task/arch; shorter BC (150). Residual settled **~0.17** cumSR by ~950 eps; flow **~0.15** with emp actor SR **0.05–0.12** vs ref ~0.17. **Gate never opened.** Stopped as insufficient for ~80% @ 2k.

---

## 7. Code map

| Piece | Path |
| --- | --- |
| Harness / flags | `v18_harness.py` |
| Launch / watchdog | `launch_v18_rlt_cf.sh` |
| Success-only residual BC | `finetune_v18_success_actor_bc.sh` |
| Online trainer | `train_rlt_online.py` |
| Actor / guide losses | `train_rlt.py` (`actor_step`, `guide_step`) |
| Phase curriculum | `v17_helpers.py` (BC + q ramp; V18 lengths) |
