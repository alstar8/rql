# Method content for ICRA 2027 architecture figure

We fine-tune a frozen π0.5 vision-language-action (VLA) specialist on the BEHAVIOR-1K task `turning_on_radio` with a LoRA residual on the action expert and an RL-token critic. The figure must be readable in an IEEE ICRA two-column paper (full-width, ~7.1 in, 16:9). Style: Physical Intelligence RLT / IEEE robotics — soft pastels, frozen modules in ice-blue with a snowflake, trainable modules in warm orange/coral, serif-italic math, sans-serif labels, orthogonal arrows, real camera thumbnails (not generic icons).

## Scene (left column, ~22% width)

Show the actual OmniGibson / R1Pro living-room task, not a toy cartoon:

- Three labeled photo tiles stacked or in a 1×3 strip:
  - **Head** camera: egocentric view of a living-room table with a radio receiver.
  - **Left wrist** camera: left gripper holding / approaching the radio body.
  - **Right wrist** camera: right fingertip at the radio power button (the critical press).
- A language chip under the photos: “Turn on the radio receiver that’s on the table in the living room.”
- A small proprioception bar: 23-d BEHAVIOR state (base, torso, two 7-DoF arms, two grippers), padded to 32-d.

Caption this block **Observation \(s\)** (chunk start).

## Frozen VLA (center-left, ice-blue dashed enclosure, snowflake “frozen”)

**PaliGemma VLM** (`gemma_2b`, frozen). It reads the three RGB images + prompt and emits prefix tokens.

A row of small token squares labeled **prefix tokens**. A stop-gradient mark `sg` on the path that leaves the VLM.

The action expert is a **one-way sandwich**, not a bidirectional hub. Spell names exactly: `action_in_proj`, `time_mlp`, `action_out_proj`. These three are **frozen linear adapters around** `gemma_300m` (width **1024**), not layers inside it, and they do **not** connect with two-way arrows into the LoRA box.

```
x_t (noisy 32-d chunk) ──► action_in_proj (frozen) ──► action tokens
t  ──► sin/cos ──► time_mlp (frozen) ──► adaRMS cond (side input)
action tokens + prefix KV ──► gemma_300m (frozen weights) + LoRA on attn/FFN
hidden ──► action_out_proj (frozen) ──► velocity v_t
```

Arrows: **in_proj only into** the expert; **time_mlp only into** the expert (adaRMS); **out_proj only out of** the expert. No up-and-down arrows through all three.

## Trainable LoRA actor (center, warm coral enclosure, flame “trainable”)

LoRA rank 32 / α 32 on **attention and FFN of gemma_300m only** (width 1024, so adapters are 1024×32 / 32×1024). **B = 0 at init.** `lora_scale` 0 copies the specialist; 1 is the live policy. CFTrunk 1536 is the critic, **not** the LoRA width.

Draw LoRA as a badge **on** the expert, not as a separate network fed by the projections.

Two parallel **Euler flow-matching** unrolls **after** `action_out_proj` (10 steps, OpenPI time \(t=1\to 0\)), sharing the same noise:

- Top (ice-blue, stop-grad): \(\tilde{a} = \mathrm{Euler}(v_{\mathrm{frozen}}, \mathrm{noise})\) — reference chunk from the frozen expert.
- Bottom (coral, gradients flow): \(a = \mathrm{Euler}(v_{\mathrm{live}}, \mathrm{noise})\) — executed 32×23 chunk.

Draw 10 small denoising dots along each Euler path. Output tensors: \(\tilde{a}, a \in \mathbb{R}^{32\times 23}\).

A dashed **anchor** arrow \(\beta\|a-\tilde{a}\|^2\) between \(a\) and \(\tilde{a}\) (per-dimension, \(\beta=100\)). A dashed **BC** arrow \(L_{\mathrm{bc}}=\mathrm{MSE}(v_{\mathrm{live}}(x_t,t), \mathrm{sg}(v_{\mathrm{frozen}}(x_t,t)))\).

This is the actor. There is **no separate Gaussian residual network**. The policy *is* LoRA on the expert.

## RL token autoencoder (center-right, mint enclosure)

Encoder **CFTokenPool**: \(d=256\), 4 heads, 2 layers, 8 queries. Compresses prefix tokens to a pooled vector.

Decoder **CFTokenDecoder** (same width) reconstructs stop-grad prefix tokens: \(L_{\mathrm{ro}}=\mathrm{MSE}(\mathrm{decoder}(z_{\mathrm{live}}), \mathrm{sg}(\mathrm{prefix}))\).

The RL state used by Q is **not** the live encoding. Draw a stop-grad lock: \(\phi(s)=[\mathrm{sg}(\mathrm{target\_pool}(\mathrm{prefix})),\;\mathrm{normalize}(\mathrm{proprio})]\). Pooled tokens 1024-d + proprio padded to 32-d.

Label the compact vector **RL token \(z\)**. During online RL the encoder is frozen (`freeze_pool=1`); only \(L_{\mathrm{ro}}\) trains the live pool in AE pretrain.

## Chunk critic (right, pale lavender enclosure)

**CFTrunk** (hidden 1536) + **10 Q-heads**, value \(=\min_k Q_k\). Input concat \([z,\;a_{\mathrm{flat}}]\) with \(a_{\mathrm{flat}}\in\mathbb{R}^{736}\) (32×23). This is a **decision-chunk** critic, not per-step \((a_t,t)\).

A small TD target chip: \(y=\mathrm{clip}(r+\gamma(1-d)\min_k Q^-_k(s',a'(s')),0,1)\), \(\gamma=0.99\) **per chunk** (not \(\gamma^{32}\)). Bootstrap action \(a'(s')\) comes from **EMA LoRA** (`cf_target_lora`, ema 0.999), not live LoRA.

Actor uses **stop-grad critic weights** so \(-Q\) cannot inflate Q. TD sees \(a+\varepsilon\) (\(\sigma=0.08\)).

## Legend (bottom strip, mandatory)

| Color / icon | Meaning |
| --- | --- |
| Ice-blue + snowflake | Frozen (VLM, base expert, projections, target pool, target LoRA) |
| Coral + flame | Trainable (live LoRA, live pool during AE, critic) |
| Dashed grey | Stop-grad / auxiliary loss |
| Solid black | Forward data / action |

Do **not** draw `cf_guide_*` or OT compose — unused at train and serve.

Do **not** put a figure title inside the artwork. Leave a 3-photo slot on the left sized for real 224×224 env frames.
