# π₀.₅ + ConsensusFlow Target-Task Fine-Tune Pipeline

**Date:** 2026-07-31  
**Code scaffold:** `agents/pi05_cf.py` (`agent_name=pi05_cf`)  
**Figure:** `my_exps/pi05_cf_finetune_pipeline_figure.{png,pdf}`  
**CF evidence:** [`decoupled_cf_work_summary.md`](decoupled_cf_work_summary.md), [`decoupled_cf_vg_coupling_section.tex`](decoupled_cf_vg_coupling_section.tex)

---

## 1. Problem setting

Fine-tune a pretrained **π₀.₅** VLA on **one** target task so success rises gradually under:

| Constraint | Implication |
|---|---|
| Strong diffusion/flow action expert | Prefer residual improvement, not full policy rewrite |
| No critic in the base VLA | Bootstrap \(Q\) from offline data |
| Sparse binary success only | Stratified replay + conservative Q; no dense shaping |
| Very limited online steps | Most learning offline (Phases 0–2); online only calibrates shift |
| Mixed success/fail buffer | Oversample \(\mathcal{D}^+\); success-conditioned BC |

**Design locks**

- Freeze the **VLM**; treat its outputs as features \(h=\mathrm{VLM}(o,\ell)\).
- Soft-plastic **action expert** via **LoRA + BC only** (no actor-Q into the expert by default).
- Train a new **endpoint residual** \(G_a\) with RL under a Lagrange trust region.
- Do **not** use latent \(G_z\), hard-frozen expert-only, or full joint CF into the VLM.

---

## 2. Role mapping (π₀.₅ ↔ CF)

| CF object | π₀.₅ / scaffold object |
|---|---|
| Behavior field \(v\) | Flow/diffusion **action expert** (+ LoRA adapters) |
| Residual \(G\) | Endpoint refiner \(G_a(h,a_v)\) |
| Critic \(Q\) | New ensemble + CQL (sparse success) |
| Guided actor (std CF) | Disabled by default; optional late hybrid only |

**Deployed policy**

\[
a=\mathrm{clip}\bigl(a_v+G_\phi(h,\mathrm{sg}[a_v])\bigr),
\qquad
\mathbb{E}\|G_\phi\|^2\approx\delta
\]

with \(a_v=T_{v_{\theta+\mathrm{LoRA}}}(h,z)\) and frozen \(h=\mathrm{VLM}(o,\ell)\).

**Role gap** (scientific metric)

\[
\Delta(v,G)=J(v,G)-J(v,0).
\]

Keep \(\Delta>0\) on the target task. Payoff absorption into \(v\) alone is optional and late.

**Residual budget (theory).** For trust radius \(\varepsilon\) (or dual target \(\delta\)), reachable actions lie in \(\mathcal{B}_\varepsilon(a_v)\). If good actions lie outside that ball, either LoRA/BC must move \(a_v\) or \(\delta\) must grow. This is why hard-frozen experts underperformed in decoupled CF screens.

---

## 3. Phases

### Phase 0 — Buffer hygiene (offline, no RL)

1. Filter trajectories to the target task (language / task id).
2. Build stratified replay \(\mathcal{D}=\mathcal{D}^+\cup\mathcal{D}^-\) with positive fraction \(p_+\in[0.3,0.5]\).
3. Sparse returns: \(r_T\in\{0,1\}\) at episode end (elsewhere \(0\)); optional success-as-absorbing.
4. Cache or re-infer expert chunk actions for BC / residual conditioning.

### Phase 1 — Critic bootstrap + LoRA BC (offline)

**Critic (Cal-QL / CQL-style)**

\[
\mathcal{L}_Q=\mathrm{TD}(Q;r_{\mathrm{sparse}},\gamma)
+\alpha_{\mathrm{CQL}}\,\mathbb{E}_{h\sim\mathcal{D}}
\Bigl[\log\sum_a e^{Q(h,a)}-\mathbb{E}_{a\sim\mathcal{D}(\cdot|h)}Q(h,a)\Bigr].
\]

Ensemble + pessimistic targets (min / mean−ρ·std). No gradients into the VLM.

**Expert LoRA + BC only**

\[
\mathcal{L}_v=\mathbb{E}_{(h,a)\sim\mathcal{D}}\;
\mathcal{L}_{\mathrm{flow}}\bigl(v_{\theta+\Delta\theta}^{\mathrm{LoRA}};h,a\bigr),
\]

with overweight on \(\mathcal{D}^+\) / near-success prefixes. Small LoRA rank; early-stop on held-out BC.

**Init \(G_a\equiv 0\)**, small trust target \(\delta\).

**Exit gates:** critic AUROC / \(Q\) gap on success vs fail; LoRA BC down; \(G\equiv 0\) eval ≈ base (no regression).

### Phase 2 — Residual RL offline

\[
\mathcal{L}_G=-\mathbb{E}\bigl[Q(h,\mathrm{clip}(a_v+G_\phi))\bigr]
+\lambda\bigl(\mathbb{E}\|G_\phi\|^2-\delta\bigr).
\]

Gradients: **only** \(G\) (+ dual). LoRA continues with **BC only** (strict BC-\(v\) / RL-\(G\)). Delayed \(G\) updates, Polyak targets, optional target-policy smoothing.

**Exit gates:** positive offline advantage proxy \(Q(h,a_v+G)-Q(h,a_v)\); residual RMS healthy.

### Phase 3 — Budgeted online fine-tune

Hard step budget \(N_{\mathrm{online}}\ll\) usual CF.

Each outer loop:

1. Roll out \(a_v+G\) (optional ε-greedy residual / best-of-\(k\)).
2. Append online buffer; stratify with \(\ge p_+\) positives; cap online share early.
3. Updates: critic TD (+ light CQL on offline support) → LoRA BC → \(G\) + dual.
4. Grow \(\delta\) only if role-gap stagnates **and** \(a_v\) is already near good modes.
5. Guards: monitor \(J(v,0)\); if \(J(v,G)<J(v,0)\), shrink/reset \(G\); stop on plateau.

**Optional late hybrid** (only if \(\Delta\approx 0\) but \(J\) still low): tiny guided-actor LoRA with \(\mathrm{sg}(G)\) lookahead, weight \(\ll\) BC.

---

## 4. Why this is coherent

1. **Expressivity** — small \(\|G\|\) cannot replace moving \(a_v\) (residual-budget proposition).
2. **Role separation** — sparse RL credit stays on \(G\); strong prior is protected; measure \(\Delta\).
3. **No base critic** — conservative offline \(Q\) before DPG.
4. **Limited online** — Phases 1–2 do the heavy lift; Phase 3 calibrates shift.
5. **Sparse reward** — stratification + CQL replace dense shaping; success-conditioned BC is the main gradual lift for \(v\).

---

## 5. Evaluation protocol

| Metric | Purpose |
|---|---|
| Success vs online step | Product metric |
| Role-gap \(\Delta=J(v,G)-J(v,0)\) | \(G\) carries improvement |
| \(J(v,0)\) alone | LoRA forgetting / prior damage |
| Critic AUROC on held-out success/fail | Phase 1 gate |
| Residual RMS / dual \(\alpha\) | Trust health |
| Ablations | (i) \(G\)-only frozen LoRA, (ii) LoRA-BC no \(G\), (iii) full pipeline |

Expectation from CF evidence: **(iii) > (i)**; **(ii)** may match base without improving.

---

## 6. Scaffold modules (`agents/pi05_cf.py`)

Research prototype treating observations as **frozen VLM features** \(h\):

| Module | Role |
|---|---|
| `actor` + `target_actor` | Frozen base flow expert (π₀.₅ stand-in) |
| `actor_lora` + `target_actor_lora` | Low-rank residual on flow velocity (BC-only) |
| `critic` / `target_critic` | Ensemble \(Q(h,a)\) + CQL |
| `refiner` / `target_refiner` | Zero-init endpoint \(G_a\) |
| `log_alpha` | Lagrange trust on \(\mathbb{E}\|G\|^2\) |

**Phase flag** `train_phase ∈ {1,2,3}`:

- `1`: critic + LoRA BC; refiner weight \(0\)
- `2`/`3`: add residual RL (same losses; Phase 3 expects online data in the trainer)

Config highlights: `cql_coef`, `success_bc_boost`, `lora_rank`, `freeze_base_actor=True`, `freeze_v` unused (base always frozen; plasticity = LoRA).

Reuse patterns from [`agents/dflrql11.py`](../agents/dflrql11.py). Unit tests: [`tests/test_pi05_cf_scaffold.py`](../tests/test_pi05_cf_scaffold.py).

---

## 7. Risk → mitigation

| Risk | Mitigation |
|---|---|
| Critic overfits sparse labels | CQL, ensemble pessimism, holdout AUROC |
| LoRA forgets general skill | Tiny rank, mixed \(\mathcal{D}^+\cup\mathcal{D}^-\), monitor \(J(v,0)\) |
| Harmful \(G\) | Small \(\delta\), zero-init, \(J(v,G)\ge J(v,0)\) |
| Offline–online shift | Short Phase 3 rollouts; stratified online share |
| Rare success | Oversample \(\mathcal{D}^+\); boost BC on successes |
| Mode switch needs large residual | Raise \(\delta\) only after LoRA moves \(a_v\) |

---

## 8. File index

```
submodules/rql/
  agents/pi05_cf.py
  agents/__init__.py              # registers pi05_cf
  tests/test_pi05_cf_scaffold.py
  my_exps/pi05_cf_finetune_pipeline.md
  my_exps/pi05_cf_finetune_pipeline_figure.png
  my_exps/pi05_cf_finetune_pipeline_figure.pdf
  my_exps/make_pi05_cf_pipeline_figure.py
```
