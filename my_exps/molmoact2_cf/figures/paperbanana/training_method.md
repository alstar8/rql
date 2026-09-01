# Method content for ICRA 2027 training-process figure

The training recipe is a four-stage pipeline plus an online collect–train loop. Draw it as a left-to-right IEEE ICRA figure (full-width, 16:9), Physical Intelligence / RLT aesthetic: pastel stage zones, numbered badges, real BEHAVIOR-1K camera thumbnails in the environment box, cylinders for replay, snowflake vs flame for freeze/train. No figure title inside the image.

## Stage 0 — Data (cream zone)

A cylinder **Replay \(\mathcal{B}\)** being filled from 100 `state_action.npz` rollouts of the frozen specialist `pi05-b1kpt12-cs32` on `turning_on_radio` (public_test instances 0–9 / 301–310).

Show a 1×3 filmstrip of env cameras (head / left wrist / right wrist) of the R1Pro at a living-room table with a radio. Label: **100 episodes → 9650 decision chunks** on a 32-step grid (timeout 4300 steps → 135 chunks). Stratified 50/50 success/fail **episodes**.

Each MDP decision is one 32-step chunk: \(s\) = obs at chunk start (3 RGB + proprio + prompt), \(a\in\mathbb{R}^{32\times 23}\) executed chunk, \(r=1\) only if the episode succeeds inside the chunk, \(done=1\) if the episode ends inside the chunk, \(s'\) = next chunk start.

## Stage 1 — AE pretrain (mint zone), 8000 steps

Badge **1**. Only the token autoencoder trains.

- `cf_ae_only=1`, `actor_coef=0`, `freeze_critic=1`, `freeze_pool=0`, `ae_coef=1`
- Loss: \(L_{\mathrm{ro}}\) only. **No Euler, no TD.**
- Encoder+decoder reconstruct stop-grad PaliGemma prefix tokens.
- VLM, action expert, LoRA remain unused for gradients.

Icon: encoder–decoder with reconstruction arrows. Small check: “RL state \(z\) is ready.”

## Stage 2 — Actor–critic pretrain (ice-blue zone), 8000 steps (to step 16 000)

Badge **2**. Critic learns values; LoRA stays a copy of the specialist.

- `actor_coef=0`, `freeze_pool=1`, `freeze_critic=0`, `ae_coef=0`
- Critic TD on \(\mathrm{sg}(\mathrm{target\_pool}(s))\)
- LoRA still trained with **BC + anchor** (\(\beta=100\)) so the expert remains a pt12 copy
- \(\gamma=0.99\) per chunk, critic coef 4, TD action noise 0.08

Icon: Q-heads lighting up; LoRA box with a tiny \(\Delta\approx 0\) residual.

## Probe (narrow grey chip between 2 and 3)

**1 episode** on instance 308, live expert, **not written to replay**. Smoke check before online RL.

## Stage 3 — Online loop (coral zone), 50 rounds

Badge **3**. Draw a **cycle**, not a line:

```
publish cf_live.npz (target LoRA + frozen encoder)
        ↓
serve + client  (instance 308, eval id 7)
        ↓
collect 1 episode  →  append npz to replay
        ↓
train  5 × n_chunks  SGD steps   (UTD = 5)
        ↓
(repeat × 50)
```

Online flags: `actor_coef=1`, `freeze_critic=0`, `freeze_pool=1`, `ae_coef=1`.

Inside the collect box, show three real camera thumbnails of the R1Pro pressing the radio, plus a success/fail chip (`q_score.final ≥ 1` vs timeout 4300).

Replay cylinder now: **100 pretrain trajs (all instances) + all online npz**, still stratified 50/50.

Serve payload: frozen encoder / critic trunk / **EMA target LoRA** (not live LoRA). Sample with `lora_scale=1`.

## Losses (bottom strip, shared across stages)

One compact equation bar so a reviewer can map boxes to math:

\[
L = 4\,L_{\mathrm{td}} + \underbrace{\lambda_{\pi}(-\min_k Q_k(s,a))}_{\text{online only}} + \beta\|a-\tilde{a}\|^2 + L_{\mathrm{bc}} + L_{\mathrm{ro}}
\]

\(\lambda_{\pi}=0\) in stages 1–2, \(\lambda_{\pi}=1\) online. Anchor and BC **always on**. Kill if \(|Q|>2\) or actor–ref RMSE \(>0.04\).

## Color / freeze legend (required)

- Mint = AE parameters
- Ice-blue = frozen VLA / frozen pool at online time
- Coral = live LoRA + critic online
- Cylinder = replay
- Camera tiles = real env images (leave three square slots on the left of Stage 0 and inside Stage 3 collect)

Do not invent extra phases (no human teleop, no reference-action dropout, no guide network). This is LoRA-on-expert RL, not a separate Gaussian actor.
