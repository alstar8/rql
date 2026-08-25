# V23: cf_ae adapted to BEHAVIOR-1K `turning_on_radio` (openpi_comet / pi0.5)

V23 ports the proven V22 `cf_ae` recipe (MolmoSpaces `desk_mug`, PyTorch) to the
BEHAVIOR-1K `turning_on_radio` task in the `openpi_comet` JAX stack. The base VLA is
the Compet-pt12 pi0.5 specialist (`pi05-b1kpt12-cs32`, action horizon 32, LoRA-free
full checkpoint). The ConsensusFlow machinery lives in
`openpi_comet/src/openpi/models/pi0_cf.py` (V17 lineage); V23 replaces the V17 loss
with the V22 `cf_ae` objective while keeping the V17 architecture (CFTokenPool,
CFTrunk, ensemble critic heads, zero-init guide head).

## What V22 cf_ae is (reference)

- Frozen VLA flow `v_theta` (Euler, 10 steps) produces the reference chunk `x_ref`.
- Actor = CF composition `V = v_theta - G`, `G = guidance_coef * W(x, z_rl, t)`,
  `W` zero-init MLP on top of a shared trunk; state features `x = (z_rl, proprio)`.
- Actor loss (Eq. 5): `E[ ||x'_1 - x_ref||^2 - beta * min_k Q_k(s, x'_1) ]`,
  endpoint from a 10-step guided unroll, anchor to the VLA reference chunk.
  **Default `beta=1`** (same as V21/V22).
- Critic: twin Q (min over ensemble), TD with target networks, clipped targets,
  bootstrapped with `a'(s')` from the target actor.
- Pretrain AC on 100 frozen-VLA episodes (actor_coef=0), then probe, then online RL.

## V23 mapping (V22 -> openpi_comet)

| V22 (MolmoSpaces)                  | V23 (openpi_comet / BEHAVIOR-1K)                        |
|------------------------------------|---------------------------------------------------------|
| Flow VLA `v_theta` (PyTorch)       | pi0.5 pt12 checkpoint, frozen (no LoRA), Euler 10 steps |
| Token AE `z_rl` (online-finetuned) | CFTokenPool over VLM prefix tokens (trained by RL loss) |
| proprio concat                     | normalized 23-d state concatenated to pooled features   |
| `W(x,z,t)` zero-init guide MLP     | `cf_guide_head` (zero-init) on `CFTrunk`                |
| DoubleCritic twin-Q                | `cf_critic_heads` ensemble (10 heads), min over heads   |
| 100 VLA episodes buffer            | 100 collected `state_action.npz` (pt12 rollouts)        |
| online episodes                    | 500 online episodes, 2 workers, rounds of 20            |

## Decision-chunk MDP (the key data adaptation)

The env steps at 30 Hz control but the policy commits to a 32-step action chunk
(re-inference only when the queue is empty; queue cleared at episode reset, so chunk
boundaries align to 32-step grid from t=0). V23 therefore treats each 32-step chunk
as one MDP decision:

- `s`     = observation at chunk start (3 images + 256-d proprio -> 23-d state + prompt tokens)
- `a`     = the 32x23 action chunk executed (what the VLA/sampled policy produced)
- `r`     = chunk reward = 1 if the episode terminated with success inside the chunk else 0
- `done`  = 1 if the episode ended (terminated or truncated) inside the chunk
- `s'`    = observation at the start of the next chunk (or terminal obs)

This is exactly the granularity at which the actor acts, so the TD backup is
well-defined and matches V22's (s, a_chunk, r, s') transitions.

`state_action.npz` already stores everything needed per step: `state` (256-d),
`action` (N,1,23), `next_state`, `next_action`, `reward`, `done`, `truncated`,
`actor_obs__0..3` (proprio + head/left/right images), `next_actor_obs__*`,
`metadata` (`success`, `prompt`). The V23 replay dataset
(`openpi/training/cf_ae_replay.py`) chunks each episode on the 32-step grid and emits
one sample per decision chunk. Format verified suitable on the live collection
(2026-08-25); no changes to the collector were needed.

## Model / loss (pi0_cf.py, `cf_v23=True`)

State features: `phi(s) = [ CFTokenPool(prefix_tokens), normalize(state_23) ]`
(1024 + 32 padded). Trunk `h = CFTrunk([phi, a_t, t_emb])`.

Critic (twin-Q TD, V22-style):
- `y = r + gamma_chunk * (1 - done) * min_k Q_target_k(s', a'(s'))`, clipped to [0, 1].
- `a'(s')` = endpoint of a 10-step Euler unroll at `s'` with the *target* guide
  (frozen `v_theta`, stop-grad everywhere).
- `gamma_chunk = 0.99` per 32-step chunk.
- Loss: MSE over all 10 heads.

Actor (Eq. 5 anchor + Q-ascent):
- `x_ref` = 10-step Euler unroll with frozen `v_theta` only (stop-grad).
- `x_guided` = 10-step unroll with `v = sg(v_theta) - guidance_coef * W(phi, x_t, t)`,
  gradients flow through `W` (and the pool/trunk via `phi`).
- `L_actor = ||x_guided - x_ref||^2 / D  -  beta * min_k Q_k(s, x_guided)`,
  `D = 32*32 = 1024`, **`beta = cf_anchor_beta = 1`** (paper / V21–V22 default).
- `L = L_td + cf_actor_coef * L_actor + cf_w_l2 * ||W||^2`.
- No SPSA distill, no `safe()` shaping, no `t_max` gate (V17 leftovers removed from
  the V23 path). Guide acts on all 10 Euler steps, deployed the same way.

Target networks: explicit `cf_target_*` modules (pool/trunk/critic/guide), excluded
from training by the freeze filter, polyak-updated (`tau = 0.005`) inside the jitted
train step after each optimizer update.

## Training protocol

1. **Data**: 100 `state_action.npz` trajectories from the pt12 collector
   (`outputs/pt12_cs32_radio_10x10_states`), public_test instances 0-9.
2. **AC pretrain**: 8000 steps, batch 8, `cf_actor_coef=0` (critic + pool/trunk only;
   guide stays ~0 so the served policy is an exact VLA copy). LR 1e-4 on CF modules,
   VLA frozen.
3. **Probe**: 10 episodes with guide on (expect ~= VLA performance; sanity that the
   deploy path with guide is exact).
4. **Online**: 25 rounds x (collect 20 episodes with 2 workers, guide on, latest
   checkpoint) + (train 1000 steps, `cf_actor_coef=1`). 500 online episodes total.
   Replay = 100 pretrain + all online npz, stratified 50/50 success/fail over
   decision chunks.

## Files

- `openpi_comet/src/openpi/training/cf_ae_replay.py`  - decision-chunk replay dataset
- `openpi_comet/src/openpi/models/pi0_cf.py`          - V23 loss + target modules + guide deploy
- `openpi_comet/src/openpi/models/model.py`           - Observation carries next-obs/reward/done
- `openpi_comet/src/openpi/policies/b1k_policy.py`    - B1kInputs emits next_* fields
- `openpi_comet/src/openpi/transforms.py`             - TokenizeNextStatePrompt, next-state norm/pad
- `openpi_comet/src/openpi/training/config.py`        - `pi05_b1k-turning_on_radio_cf_v23` config
- `openpi_comet/src/openpi/training/data_loader.py`   - replay-only loader path
- `openpi_comet/scripts/train.py`                     - polyak target update in train_step
- `openpi_comet/scripts/run_cf_v23_radio.sh`          - orchestration + watchdog heartbeats

## Watchdog

The orchestrator (`openpi_comet/scripts/run_cf_v23_radio.sh`, inside `b1k-airi-dev`)
heartbeats to `outputs/cf_v23_radio_status.log`; its stdout goes to
`outputs/cf_v23_radio_driver.log`. The host-side watchdog
(`openpi_comet/scripts/cf_v23_watchdog.sh`, logs to
`outputs/cf_v23_radio_watchdog.log`) checks every 20 min: driver alive (restarts via
`docker exec` if dead), and progress advancing (mtime of driver/status logs, npz
files, checkpoints). If nothing progresses for 35 min it kills the stuck phase
processes (serve/eval/train) so the driver's retry loop re-runs the phase; if still
stuck at 80 min it restarts the driver. Completed rounds are skipped via checkpoint
steps, so restarts are idempotent. `outputs/cf_v23_radio_DONE` ends the watchdog.
