# CF_VLA V13 theory and implementation audit

Audit date: 2026-08-11  
V12 code revision: `2395704b8a90a926aca89a1ed98614150e9cd122`  
MolmoSpaces revision: `e2474eb13769f5bd02271ecf8c11c5a75b50a77a`

This document defines the claims that must be verified before the controlled
V13 run is launched. A verdict applies to the V13 implementation only after
the referenced unit, integration, and rollout checks pass.

## V12 evidence snapshot

At 2026-08-11T11:26:06Z all 56 recorded V12 processes were alive. The eight
arms had completed 19,430 valid episodes in total.

| Arm | Episodes | Pooled SR | Window SR | Gate | V12 verdict |
| --- | ---: | ---: | ---: | --- | --- |
| Residual frozen VLA | 2,640 | 0.311 | 0.315 | off by design | valid canary |
| Residual actor | 2,480 | 0.204 | 0.203 | on, 4/4 shards | implementation runs; efficacy not verified |
| Residual actor + CF | 2,400 | 0.164 | 0.175 | on, 4/4 shards | theory target invalid; efficacy not verified |
| Flow frozen VLA | 2,630 | 0.313 | 0.323 | off by design | valid canary |
| Lightweight flow actor | 2,570 | 0.298 | 0.270 | off, 0/4 shards | learned policy was not evaluated by training SR |
| Lightweight flow + CF | 2,590 | 0.308 | 0.273 | off, 0/4 shards | learned policy was not evaluated by training SR |
| Molmo AE LoRA | 2,010 | 0.091 | 0.080 | off, 0/4 shards | critic/gate scored the wrong velocity model |
| Molmo AE LoRA + CF | 2,110 | 0.121 | 0.113 | off, 0/4 shards | critic/gate scored the wrong velocity model |

V12 SR is training-rollout SR over the full MiniBench and is not held-out
validation. It must not be used as evidence for generalization.

## Theory invariants

### Consensus teacher

For a sampled target-critic member \(J_i\) at each batch element,

\[
z_i = \frac{g_{J_i}}{\lVert g_{J_i}\rVert + c\,m_B + \epsilon},
\qquad
g_{J_i} = \nabla_a Q'_{J_i}.
\]

The raw guide student \(W_\phi\), not the bounded deployed field, minimizes
\(\lVert W_\phi-z\rVert^2\). The magnitude of \(z\) must not be re-normalized
away. This is required for the conditional-moment interpretation: opposed
critic directions cancel while aligned directions retain magnitude.

### Shared flow field

For lightweight-flow and Molmo-AE arms, reverse critic states, actor
lookahead, guide training, and deployment must use the same velocity provider
and the same bounded guidance map:

\[
\dot{x}=v(x,t)+G_\phi(x,t), \qquad
G_\phi=\lambda t\,\mathrm{safe}(W_\phi;v).
\]

Actor gradients pass through \(v\) while \(G_\phi\) is stop-gradient in actor
lookahead. Guide gradients pass through \(W_\phi\) only.

### Stable references

- Residual and lightweight-flow reference: frozen HTTP MolmoAct2 action chunk.
- Molmo-AE reference: pretrained Molmo action expert with LoRA adapters
  disabled.
- Gate off executes the stable reference; gate on executes the learned actor,
  optionally with the guide.
- Training exploration is data collection and is logged separately.
  Validation has zero exploration.

### Validation isolation

Validation may read a checkpoint and its stored gate decision. It must not add
replay rows, update an optimizer, save a training checkpoint, or change a
training metric file.

## Eight-arm contract

### 1. Residual frozen-VLA canary

- Trainable online modules: none.
- Action: \(a=\tilde a_{\mathrm{VLA}}\).
- Acceptance: no optimizer is constructed or stepped; action bytes equal the
  server reference when exploration is zero.

### 2. Residual actor

- Trainable: residual actor and residual critic ensemble.
- Action when enabled: \(a=\tilde a+\Delta_\pi\).
- Acceptance: actor gradients do not reach token encoder or guide; actor-only
  held-out SR is reported separately from the reference.

### 3. Residual actor + CF

- Trainable: residual actor, critic ensemble, and raw consensus student.
- Action when enabled: \(a=\tilde a+\Delta_\pi+\Delta_g\).
- Acceptance: stochastic target-critic gradients train raw \(W_\phi\);
  opposed-gradient tests cancel; the residual bound holds; actor+guide and
  actor-only validation are paired.

### 4. Flow frozen-VLA canary

- Trainable online modules: none.
- Action: \(a=\tilde a_{\mathrm{VLA}}\).
- Acceptance: matches arm 1 within paired rollout variation.

### 5. Lightweight flow actor

- Trainable: `FlowVelocityActor` and time critic ensemble.
- Action when enabled: endpoint of \(\dot x=v_\theta\).
- Acceptance: reverse-state TD, actor lookahead, and deployment all use
  `FlowVelocityActor`; forced actor validation proves the learned path runs.

### 6. Lightweight flow actor + CF

- Trainable: flow actor, time critic, and `FlowCFGuide`.
- Action when enabled: endpoint of \(\dot x=v_\theta+G_\phi\).
- Acceptance: raw \(W_\phi\) receives the common-scale target; all flow paths
  use the same safety transform; actor+guide is compared with actor-only.

### 7. Molmo AE LoRA actor

- Status: explicit extension, not the frozen-action-expert CF_VLA claim.
- Trainable: Molmo action-expert LoRA and AE-consistent time critic.
- Gate-off reference: the same action expert with adapters disabled.
- Acceptance: critic reverse states and TD bootstrap call the AE velocity
  provider and never the frozen `FlowVelocityActor`; LoRA save/load is exact.

### 8. Molmo AE LoRA actor + CF

- Status: explicit extension.
- Trainable: AE LoRA, AE-consistent time critic, and `FlowCFGuide`.
- Action when enabled: endpoint of \(\dot x=v_{\mathrm{AE-LoRA}}+G_\phi\).
- Acceptance: AE-backed gate scores the exact deployed action; missing RL
  state or guide is fatal; paired actor-only/actor+guide validation is stored.

## Active-path authenticity checks

The V13 launcher must record model class, checkpoint SHA-256, benchmark hash,
trainable parameter names/counts, and optimizer membership. The wiring probe
must demonstrate a finite parameter delta in every expected trainable module
after one synthetic update and zero delta in every frozen module.

The following V12 constructs are not accepted in an active V13 path:

- fixed-magnitude, re-unitized consensus targets;
- dummy `next_*` image transitions;
- zero-state guided AE fallback;
- silent guided-to-unguided fallback;
- synthesized terminal rewards after reward-cache mismatch;
- watchdog deletion of online checkpoints;
- AE LoRA save without load;
- dead `--joint_cf` semantics;
- validation writes to training replay or metrics.

Black-image server warmup and legacy offline placeholders are initialization or
inactive legacy code, respectively. They are not evidence of mock rollout
components; the V13 manifest and runtime assertions must prove they never
supply training observations, actions, features, or rewards.

## Verdict rule

Each claim is reported as `VERIFIED`, `NOT VERIFIED`, or `INCONCLUSIVE`.
Passing unit tests verifies only implementation invariants. Actor or CF
efficacy is verified only by held-out house-0 kettle validation with fixed
poses and seeds; a negative SR delta remains a valid negative result.
