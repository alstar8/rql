# ICRA 2027 figures for V23 (LoRA action-expert RL on π0.5)

Camera-ready schemes of the **architecture** and **training process** in
`V23_METHODS.md`, drawn for an IEEE ICRA reviewer: frozen vs trainable
color, real BEHAVIOR-1K cameras, and the same math as the loss.

Layout follows Physical Intelligence RLT (`papers/RLT/RLT.tex` Fig. 1–3):
full-width architecture, staged training recipe, task filmstrip.

## Use these in the paper

| File | Role | Caption (draft) |
| --- | --- | --- |
| `icra/fig_architecture.png` (+ `.pdf`) | **Fig. architecture** | LoRA on a frozen π0.5 action expert. Three cameras and a prompt enter frozen PaliGemma; an encoder–decoder bottleneck yields RL token \(z\) for the critic only. Dual Euler unrolls share noise: stop-grad \(\tilde a\) from frozen weights, executed \(a\) from live LoRA. A 10-head chunk critic \(Q(s,\mathrm{flatten}(a))\) bootstraps from EMA LoRA. Ice-blue is frozen; coral is trained. |
| `icra/fig_training.png` (+ `.pdf`) | **Fig. training** | Stage 0: 100 specialist rollouts (9650 chunks). Stage 1: token AE only (8000 steps). Stage 2: critic TD while LoRA stays a specialist copy via BC+anchor. After a 1-ep probe, Stage 3: 50 collect–train rounds on instance 308. \(\lambda_\pi\) gates only \(-Q\); anchor and BC stay on. |
| `icra/fig_task_filmstrip.png` (+ `.pdf`) | **Fig. task** (optional, like RLT Fig. 3) | `turning_on_radio` from the policy cameras: head, left wrist, right wrist at approach / reach / press. |
| `icra/fig_success.png` (+ `.pdf`) | **Fig. results** | Online LoRA RL on instance 308 (`cf_v23_ae2`). Last-10 and cumulative success vs episode, with the frozen pt12 specialist (8/10) as a dashed line. Strip: per-episode success vs timeout. Length panel: env steps, timeout 4300. Overall 37/50 = 74%; last-10 finishes 10/10. |

`illustrated/` holds RLT-style illustrated drafts (image-gen). Prefer
`icra/*.pdf` for the paper: labels match the method, photos are the real
chunk-cache frames.

Env stills (from `openpi_comet/outputs/pt12_cs32_radio_10x10_states/_cf_ae_chunks`):
`env/head_*.png`, `env/left_wrist_*.png`, `env/right_wrist_*.png`,
`env/cameras_{approach,reach,press}.png`.

## Regenerate

```bash
python3 figures/scripts/extract_env_frames.py
python3 figures/scripts/render_icra_figures.py
```

## PaperBanana (when an API key is available)

PaperBanana is the intended illustrator (`submodules/PaperBanana`). This
machine has no `configs/model_config.yaml` and no `GOOGLE_API_KEY` /
`OPENROUTER_API_KEY`, so the CLI could not be run here. Briefs are in
`paperbanana/`. After filling a key:

```bash
cd /home/staroverov/B1K_AIRI/submodules/PaperBanana
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
python skill/run.py \
  --content-file ../rql/my_exps/molmoact2_cf/figures/paperbanana/architecture_method.md \
  --caption "$(cat ../rql/my_exps/molmoact2_cf/figures/paperbanana/architecture_caption.txt)" \
  --aspect-ratio 16:9 --num-candidates 3 --exp-mode demo_full \
  --output /home/staroverov/B1K_AIRI/submodules/rql/my_exps/molmoact2_cf/figures/illustrated/architecture_banana.png
python skill/run.py \
  --content-file ../rql/my_exps/molmoact2_cf/figures/paperbanana/training_method.md \
  --caption "$(cat ../rql/my_exps/molmoact2_cf/figures/paperbanana/training_caption.txt)" \
  --aspect-ratio 16:9 --num-candidates 3 --exp-mode demo_full \
  --output /home/staroverov/B1K_AIRI/submodules/rql/my_exps/molmoact2_cf/figures/illustrated/training_banana.png
```

Then paste the real `env/` tiles over any generated camera placeholders.
The matplotlib `icra/` figures already do that.
