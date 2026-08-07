# Decoupled ConsensusFlow (BC-v / RL-G) — Work Summary

**Date:** 2026-07-30 → 2026-07-31  
**Plan:** Decoupled CF research (`decoupled_cf_research_902a6552`)  
**Codebase:** `B1K_AIRI/submodules/rql`

---

## 1. Goal

Replace the failed full-ODE residual objective (`dflrql10`) with a strict **BC flow `v` + RL improvement module `G`** design:

1. **Priority 1 — endpoint refiner `G_a`** (DeFlow-style): `a = clip(a_v + G_a(s, sg[a_v]))`
2. **Priority 2 — latent policy `G_z`** (Flow Latent Policy-style): Gaussian over initial flow noise + latent critic + KL
3. Modes: frozen-`v` and BC-only-online-`v`; optional consensus trust that scales residual **budget**, not direction
4. Restore from pure-BC `cfbcvg-fig6-*` @1M backbones (`actor_q_coef=0`)

Success gate (plan): beat base CF five-task 2M mean (~0.863) or online AUC with positive G-off gaps.

---

## 2. What was implemented

### 2.1 Endpoint agent — `agents/dflrql11.py`

- BC flow `v` / `target_actor`, endpoint refiner `G_a` / `target_refiner`, Q ensemble, dual Lagrange `log_alpha`
- `v` loss = flow matching only (no Q/advantage gradient into `v`)
- Zero-init `G_a(s, a_v)`; Q-normalized DPG + Lagrange targeting residual MSE `δ`
- Delayed actor updates, target-policy smoothing, ensemble pessimism, Polyak targets, n-step chunk returns
- Optional `consensus_trust` scales allowed residual magnitude from critic-gradient agreement
- Modes: `freeze_v=True` and `freeze_v=False` (BC-only online updates to `v`)
- `disable_rl_policy` for residual-off / role-gap eval

### 2.2 Latent agent — `agents/dflrql12.py`

- Latent Gaussian `G_z(s)` initialized exactly at `N(0,I)` (zero KL at init)
- Frozen / BC-only `v` decodes latents; reverse BC ODE maps replay actions → latents
- Latent critic `Q_z(s,z)`; policy loss `-Q_z + β KL(N(μ,σ)||N(0,I))`
- Best-of-4 candidates for TD targets and deployment
- Same freeze / BC-only-online-`v` modes

### 2.3 Backbone restore — `utils/flax_utils.py`

- `restore_agent_backbone(...)`: shape-checked copy of actor / target-actor
- Optional critic restore (maps legacy `value` → `critic` for endpoint arms)
- Fresh RL heads, optimizer state, RNG, and step counter

### 2.4 Training plumbing — `main.py`

- `--restore_backbone_only`, `--restore_backbone_critic`
- `--training_step_offset` for plot/checkpoint naming after BC restore
- Stratified offline/online replay (`online_replay_fraction_max`, ramp, online buffer)
- `--eval_residual_off` for paired G-on vs G-off success and role-gap logging

### 2.5 Registration & tests

| Artifact | Path |
|---|---|
| Registry | `agents/__init__.py` (`dflrql11`, `dflrql12`) |
| Endpoint tests | `tests/test_dflrql11_decoupled_endpoint.py` |
| Latent tests | `tests/test_dflrql12_decoupled_latent.py` |
| Restore tests | `tests/test_flax_utils_backbone_restore.py` |

**Unit tests:** 22 passed (frozen-`v` bitwise unchanged, BC-only grads into `v`, RL-only grads into `G`/`G_z`, Lagrange dual direction, zero-residual equivalence, reverse ODE, backbone isolation).

### 2.6 Launchers & evaluator

| Script | Role |
|---|---|
| `cloud_job/run_cf_decoupled_fig6.sh` | Five-arm seed-0 screen |
| `cloud_job/run_cf_decoupled_scale_fig6.sh` | Scale wrapper (top arms → queue) |
| `cloud_job/run_cf_decoupled_scale_queue.sh` | Shared GPU job queue for scale-up |
| `scripts/evaluate_decoupled_cf.py` | Success @1.2M/1.4M/2M, online AUC, steps-to-90%, residual-off gap, dual α / residual RMS, paired bootstrap |
| Method-card link | `method_cards/ConsensusFlow/scripts/evaluate_decoupled_cf.py` → rql script |

**Protocol:** restore `cfbcvg-fig6-*` @1M → 200k offline G adaptation + online fine-tune; stratified online share up to 0.5.

**Bugfix during screen:** puzzle failed because `--ogbench_dataset_dir` (100M shards) cannot be combined with `online_steps > 0` (`main.py` assert). Launcher now skips the 100M flag whenever online steps are nonzero and uses the standard 1M OGBench npz.

---

## 3. Seed-0 screening results

**Tag:** `seed0-o200k-n200k-v1`  
**Budget:** 200k offline + 200k online → report @ **1.4M**  
**Tasks:** antmaze-giant-t1, antsoccer-t4, cube-double-t2, puzzle-4x4-t4  
**Artifact:** `submodules/rql/my_exps/decoupled_cf_screen_seed0.json`

### 3.1 Success @1.4M (seed 0)

| Arm | antmaze | soccer | cube | puzzle | mean Δ vs base CF |
|---|---:|---:|---:|---:|---:|
| **endpoint_bc** | 0.55 | 0.35 | 1.00 | 1.00 | **−0.065** |
| **endpoint_trust** | 0.25 | 0.40 | 0.95 | 1.00 | −0.140 |
| endpoint_frozen | 0.20 | 0.50 | 1.00 | 0.00 | −0.365 |
| latent_bc | 0.05 | 0.00 | 0.95 | 0.00 | −0.540 |
| latent_frozen | 0.00 | 0.00 | 1.00 | 0.00 | −0.540 |
| base CF (seed0) | 0.62 | 0.58 | 0.96 | 1.00 | — |

### 3.2 Screening conclusions

- **Promoted top 2:** `endpoint_bc`, `endpoint_trust`
- **Stopped:** both latent arms (near-zero on locomotion) and `endpoint_frozen` (puzzle collapse; weaker overall)
- No arm beat base CF at the short 1.4M screen budget; `endpoint_bc` was closest
- Puzzle residual-off evals (retry runs) showed `endpoint_trust` can have large positive role-gap (G carries success)

---

## 4. Scale-up — completed

**Tag:** `scale-o200k-n800k-v1` · **30/30 jobs done, 0 fails**  
**Artifact:** `my_exps/decoupled_cf_scale_seeds012.json`

| Method | five-task 2M mean | Δ vs base (0.863) | mean role-gap | online AUC Δ |
|---|---:|---:|---:|---:|
| **endpoint_bc** | **0.747** | **−0.116** | +0.353 | −0.210 |
| endpoint_trust | 0.560 | −0.303 | +0.553 | −0.402 |
| base CF | 0.863 | — | — | — |

### 4.1 `endpoint_bc` @2M vs base CF (3-seed means)

| Task | method | base CF | Δ | role-gap |
|---|---:|---:|---:|---:|
| antmaze | 0.883 | 0.907 | −0.023 | **+0.817** |
| humanoid | 0.867 | 0.980 | −0.113 | **+0.767** |
| soccer | 0.333 | 0.440 | −0.107 | +0.167 |
| cube | 0.983 | 0.987 | −0.003 | ~0 |
| puzzle | 0.667 | 1.000 | −0.333 | ~0 (s1=0) |

### 4.2 `endpoint_trust` @2M (complete)

| Task | method | base | Δ | role-gap |
|---|---:|---:|---:|---:|
| antmaze | 0.400 | 0.907 | −0.507 | +0.40 |
| humanoid | 0.650 | 0.980 | −0.330 | +0.65 |
| soccer | 0.267 | 0.440 | −0.173 | +0.25 |
| cube | 0.917 | 0.987 | −0.070 | +0.90 |
| puzzle | 0.567 | 1.000 | −0.433 | +0.57 |

---

## 4A. What worked vs what did not

### Worked (keep)

1. **Endpoint `G_a` as the RL channel** — On antmaze/humanoid, residual-off success collapses (~0–0.15) while G-on reaches ~0.8–1.0. Role-gaps **+0.65 to +0.90** prove G carries real sample-time improvement (unlike joint CF where λ-gap often → 0).
2. **BC-only-online `v` (`endpoint_bc`)** — Best arm by a wide margin vs frozen-`v` / latent. Matches the residual-pilot diagnosis that plastic BC `v` for coverage matters more than freezing the expert.
3. **Pure-BC backbone restore + stratified replay + dual Lagrange residual** — Training is stable (finite losses; residual RMS ~0.03–0.05; dual α adapts). Unit tests pass (22/22).
4. **Screen → kill weak arms** — Correctly stopped latents and frozen endpoint before burning 3-seed scale budget.
5. **Saturated manip matching** — Cube ~matches CF (0.983 vs 0.987). Puzzle s0/s2 also match when they don’t collapse.

### Did not work (stop or redesign)

1. **Beating base CF on the success gate** — Five-task 2M mean **0.747 vs 0.863**. No arm clears the plan gate (success or online AUC).
2. **Latent `G_z` + KL** — Near-zero on locomotion in screen; killed. Mode selection in noise space is not enough without stronger decoder adaptation.
3. **Frozen-`v` endpoint** — Trails BC-online everywhere that isn’t saturated; puzzle screen collapse; scale trust arm ~half of base on antmaze.
4. **Consensus trust as residual budget gate** — Screen #2 and scale trust arm underperform plain endpoint; agreement scaling did not help success.
5. **Soccer** — Structural miss for endpoint_bc (−0.11) and worse for trust; G helps only modestly (gap ~+0.17).
6. **Humanoid sample efficiency** — Final success only −0.11, but online AUC collapses (0.64 vs 0.98) → learns much later than joint CF.
7. **Puzzle seed fragility** — s1 = 0 at 2M with G-off also 0 → not a G bug, a full policy collapse for that seed/run.
8. **Prior `dflrql10` / cfgres residual ODE** — Already closed: none beat CF; missing plastic `v` + tiny applied residual budget.

### Interpretation

Decoupled CF **validates the role split** (G matters; v can stay BC-only) but **does not yet replace joint CF** as a performance method. The gap is largest where joint CF jointly reshapes `v` online (soccer, early humanoid) or where a single bad seed nukes a saturated task (puzzle). Next research should either (a) hybridize: keep BC+endpoint G but allow limited Q-lookahead into `v`, or (b) keep strict split and fix soccer/humanoid with better exploration / larger residual trust / critic calibration — not resurrect latent-only or frozen-v.

---

## 4B. Mathematical claim on $v$–$G$ coupling

LaTeX section (provable budget/absorption + empirical role/λ-gaps):

`my_exps/decoupled_cf_vg_coupling_section.tex`

**Sharpened claim (supported):** $v$ should be trained under guided dynamics that include $\mathrm{sg}(G)$ (CF actor lookahead), and a live residual with $\Delta=J(v,G)-J(v,0)>0$ is necessary whenever the BC map is far from optimal. Full payoff absorption ($\Delta\to 0$) is a distinct limit that can hurt (joint humanoid) but is not universally worse (cube/puzzle).

**Not supported as stated:** “Any integration of $G$ into $v$ is harmful” — false; cube/puzzle absorbed solutions stay strong. “BC-only $v$ is enough if $G$ is good” — also false vs base CF.

## 4C. Frozen-$v$ RL adapters (VLA fine-tune figure)

Static figure (architecture + success bars):

- `my_exps/frozen_v_rl_adapter_figure.png`
- `my_exps/frozen_v_rl_adapter_figure.pdf`
- Data table: `my_exps/frozen_v_variants_table.json`

**Answer to “can a separate RL action expert improve a frozen VLA?”**  
Not stably on this suite. Every hard-frozen-$v$ recipe trails base CF’s five-task mean (~0.86). Best frozen aggregate is CF G-role freeze (~0.50), with a single bright spot (humanoid ≈ 0.99). Soft freeze / BC-plastic $v$ + endpoint $G$ (0.75) is better than hard freeze but still below joint CF. Latent-$G_z$ freeze fails locomotion.

## 4D. π₀.₅ + CF target-task fine-tune pipeline

Design for fine-tuning π₀.₅ on one task under sparse success + limited online budget (freeze VLM, LoRA/BC on action expert, CQL critic, endpoint $G$):

- Spec: [`pi05_cf_finetune_pipeline.md`](pi05_cf_finetune_pipeline.md)
- Figure: `pi05_cf_finetune_pipeline_figure.{png,pdf}`
- Scaffold agent: `agents/pi05_cf.py` (`pi05_cf`), tests in `tests/test_pi05_cf_scaffold.py`

## 5. How to finish evaluation when scale completes

```bash
cd B1K_AIRI/submodules/rql
python scripts/evaluate_decoupled_cf.py \
  --save-dir exp \
  --tag scale-o200k-n800k-v1 \
  --arms endpoint_bc endpoint_trust \
  --tasks antmaze_giant_task1 humanoidmaze_medium_task1 antsoccer_arena_task4 cube_double_task2 puzzle_4x4_task4 \
  --seeds 0 1 2 \
  --online-start 1200000 \
  --report-step 2000000 \
  --milestones 1200000 1400000 2000000 \
  --out my_exps/decoupled_cf_scale_seeds012.json
```

Then compare five-task 2M mean and online AUC vs base CF groups listed in the evaluator (`BASE_GROUPS`), plus mean role-gap.

---

## 6. Key design lessons from this effort

1. **Std-CF residual budget is too small** for endpoint success (`guidance_coef * t * unit_ball(G)` on the full chunk); `bigres` only changed damping, not the applied correction budget — motivating decoupled endpoint / latent `G`.
2. Prefer **pure-BC** (`cfbcvg`) restore over std-CF restore so `v` is not already RL-contaminated.
3. **Strict gradient separation** (BC→`v`, RL→`G`) is enforced in code and unit-tested.
4. Latent KL alone did **not** screen well on locomotion; endpoint + BC-online-`v` did.
5. Puzzle online must **not** use `--ogbench_dataset_dir` (100M rotating shards).

---

## 7. File index

```
submodules/rql/
  agents/dflrql11.py
  agents/dflrql12.py
  agents/__init__.py
  utils/flax_utils.py          # restore_agent_backbone
  main.py                      # backbone restore, stratified replay, residual-off eval
  scripts/evaluate_decoupled_cf.py
  tests/test_dflrql11_decoupled_endpoint.py
  tests/test_dflrql12_decoupled_latent.py
  tests/test_flax_utils_backbone_restore.py
  my_exps/decoupled_cf_screen_seed0.json
  my_exps/decoupled_cf_work_summary.md   # this file

cloud_job/
  run_cf_decoupled_fig6.sh
  run_cf_decoupled_scale_fig6.sh
  run_cf_decoupled_scale_queue.sh

ReseachOS/.../ConsensusFlow/scripts/evaluate_decoupled_cf.py  # symlink
```

---

## 8. Todo status vs plan

| Todo | Status |
|---|---|
| Endpoint agent + unit tests | **Done** |
| Latent agent + unit tests | **Done** |
| Backbone restore + stratified replay + five-arm launcher | **Done** |
| Seed-0 screen vs base CF | **Done** (promoted `endpoint_bc`, `endpoint_trust`) |
| Scale top-2 → seeds 0–2 / all Fig. 6 + role-gap/AUC analysis | **Done** — 30/30; neither beats CF (`endpoint_bc` 0.747 vs base 0.863, Δ=−0.116) |
