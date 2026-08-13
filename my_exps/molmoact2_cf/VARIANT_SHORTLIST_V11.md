# CF_VLA shortlist — V12 (being superseded by controlled V13)

- **V13 correction/audit:** `V13_THEORY_IMPLEMENTATION_AUDIT.md`
- **V13 launcher:** `launch_v13_controlled.sh`
- **V13 controlled benchmark:** `runs/benchmarks/house0_kettle_v13/`
- **V13 run:** `runs/rlt_cf_v13_controlled/`

V13 retains the eight-arm comparison but fixes raw ConsensusFlow target
distillation, AE-consistent critic/gate paths, stable adapter-disabled AE
references, atomic resume bundles, and held-out evaluation. V12 is preserved
as historical evidence and is stopped only after its final snapshot.

**Active run:** `runs/rlt_cf_v12_shortlist/`  
**Launch:** `launch_v12_shortlist.sh`  
**Stop:** `./stop_run.sh runs/rlt_cf_v12_shortlist`  
**Logs:** `$B1K_TMP/rlt_cf_v12_shortlist_logs/` (symlink `runs/rlt_cf_v12_shortlist/logs`)  
**Started:** 2026-08-09T13:56:30Z — fresh from offline pretrain (`--no_resume`)

| Init ckpt | Path |
| --- | --- |
| Residual / residual-serve | `runs/rlt_pretrain_demo1k/rlt_cf_pretrain_demo1k.pt` |
| Flow / AE | `runs/rlt_pretrain_demo1k/rlt_cf_flow_pretrain_demo1k.pt` |

**Live packing:** 1 variant / GPU × **4 shards** (24 HTTP servers + 32 trainers). AE uses in-process MolmoAct2 (no `serve.py` on GPUs 6–7).

V11 / V11.1 dual runs (`rlt_cf_v11_residual`, `rlt_cf_v11_flow_joint`, `rlt_cf_v11_1_flow_ae`) are **stopped** and superseded by this matrix.

---

## Naming map (paper → stack)

| Paper | Residual | Flow (RLT \(v_\theta\)) | Flow (Molmo AE) |
| --- | --- | --- | --- |
| \(V\) / \(v_\theta\) | RLT residual actor | RLT `FlowVelocityActor` | MolmoAct2 AE (AE-only LoRA) |
| \(G_\phi\) | `CFGradientGuide` | `FlowCFGuide` | `FlowCFGuide` |
| Critic | residual ensemble | time critic | time critic |
| RL token | frozen | frozen | frozen (forced) |
| VLM | frozen on `serve.py` | frozen on `serve.py` | frozen in-process |
| Reference \(\tilde a\) | HTTP `/act` | HTTP `/act` | AE unguided sample |

Deploy: residual \(a=\tilde a+\Delta_\pi(+\Delta_g)\); flow/AE \(\dot x=v+\mathrm{sg}(G)\) when gate ON.

---

## V12 knobs (all learned variants)

| Knob | Value |
| --- | --- |
| `--no_resume` | on (offline pretrain only) |
| `explore_residual_std` / `explore_deploy_std` | **0.02** / **0.02** |
| `explore_warmup_mult` | 1.0 (no gate-off boost) |
| `bc_ref_coef` | **1.0** (flow/AE BC → reference) |
| `g_min_advantage` | 0.003 (**required** for residual, joint, and AE) |
| `g_min_guide_advantage` | 0.001 (CF configs with guide) |
| `g_min_action_sensitivity` | 0.003 |
| `guide_beta` / `guide_target_delta_frac` | 0.05 / 1.0 (residual + flow magnitude distill) |
| `guidance_coef` / `flow_steps` | 0.5 / 10 |
| updates/ep | 8 (HTTP RLT), 4 (AE), 0 (baselines) |

---

## The 8 variants (GPU layout)

| GPU | Config dir | Geometry | Actor | Guide | Joint / AE |
| --- | --- | --- | --- | --- | --- |
| 0 | `residual_mean_pool_baseline` | residual | `vla_only` | off | — |
| 1 | `residual_rlt_actor_no_guide` | residual | RLT Δπ | off | — |
| 2 | `residual_rlt_cf_frozen_token` ★ | residual | RLT Δπ | on | — |
| 3 | `flow_mean_pool_baseline` | flow | `vla_only` | off | — |
| 4 | `flow_rlt_actor_no_guide` | flow | `FlowVelocityActor` | off | `--joint_cf` |
| 5 | `flow_rlt_cf_frozen_token` | flow | `FlowVelocityActor` | on | `--joint_cf` |
| 6 | `ae_no_guide` | flow | Molmo AE LoRA | off | `--ae_trainable` |
| 7 | `ae_cf_frozen_token` ★ | flow | Molmo AE LoRA | on | `--ae_trainable` |

Shard paths: `runs/rlt_cf_v12_shortlist/<config>/shard_{0..3}/`.

### Roles

1. **Baselines (0, 3)** — VLA-only ceiling / canary; no train updates.
2. **Res no-guide (1)** — ablation: residual \(V\) without \(G\).
3. **Res CF frozen (2)** — primary residual CF: \(a=\tilde a+\Delta_\pi+\Delta_g\).
4. **Flow no-guide (4)** — ablation: joint RLT \(v_\theta\) without \(G\).
5. **Flow CF frozen (5)** — RLT flow + `FlowCFGuide`.
6. **AE no-guide (6)** — paper \(V\)=Molmo AE, no guide.
7. **AE CF frozen (7)** — main paper bet: \(v_{\mathrm{AE}}+\mathrm{sg}(G)\).

---

## Why these 8 (V11 evidence, archived)

Shortlist chosen from V11 residual / flow-joint / V11.1 AE cuts. Online-token dropped (worst or tied SR, nonstationary \(z\)).

| Variant | V11/V11.1 SR (cut) | Note |
| --- | --- | ---: |
| Res / flow baseline | ~0.31 / ~0.32 | ceiling |
| Res no-guide / CF | ~0.16 / ~0.16 | CF ≈ no-guide; adaptation tax |
| Flow no-guide / CF | ~0.15 / ~0.18 | CF slight edge; weak `guide_adv` |
| AE no-guide / CF | ~0.23 / ~0.29 @~35 eps | early; critic not healthy yet |

V12 restarts all eight with the knobs above so those failure modes (negative-adv joint deploy, explore 0.05, BC-on-executed, weak flow guide) are not repeated.

---

## Dropped (do not relaunch)

| Variant | Why |
| --- | --- |
| `rlt_cf_online_token` (residual/flow) | Worst residual SR; no gain vs frozen token |
| Flow without `--joint_cf` | Superseded by joint CF actor |
| Extra AE token/guide combos | Premature until AE critic healthy and CF ≥ no-guide |

---

## Success criteria

- **AE (6–7):** `ae_grad_norm` > 0; no OOM at 4/GPU (else set `AE_INSTANCES_PER_GPU=1|2`); after critic healthy + gate ON, AE CF SR ≥ AE no-guide by ~100–200 eps.
- **Residual CF (2):** gate ON, healthy sens, `guide_adv` > 0; CF SR ≥ no-guide (1); chase baseline (0).
- **Flow joint (4–5):** `actor_adv` rises or clear SR gain vs no-guide; CF ≥ no-guide; deploy only with `g_pred_adv ≥ g_min_advantage`.
- **Baselines (0, 3):** stable SR ~0.3; serve/EGL healthy.

---

## Ops

```bash
cd B1K_AIRI/submodules/rql/my_exps/molmoact2_cf
# already running; relaunch only after stop:
# NO_SCREEN=1 DETACH_AFTER_START=1 bash launch_v12_shortlist.sh
./stop_run.sh runs/rlt_cf_v12_shortlist
```

Watch: `runs/rlt_cf_v12_shortlist/logs/train_<config>_s*_gpu*.log` and per-shard `metrics.jsonl`.
