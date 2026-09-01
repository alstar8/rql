# V23 attempts that did not work

Current method and the finished `cf_v23_ae2` run: [`V23_METHODS.md`](V23_METHODS.md).
This file is the recipes, knobs, and jobs that failed or were abandoned.
Do not resume these checkpoint families; GraphDefs and/or hyperparameters
are incompatible with the current actor path.

## Injection (`v_θ − G`) — `cf_v23_radio`

Serve Euler was `x ← x + dt · (v_θ − G)` with a learned guide `G` subtracted
from the frozen expert velocity. Train used the leftover injection loss
(`compute_cf_loss` when `cf_v23=False`): SPSA distill, CQL on success
chunks, late-Euler gate, freeze-LoRA-after-warmup.

**10-instance rerun** (2026-08-26 23:39 → 2026-08-27 09:09 UTC): frozen
critic, `beta=100`, actor-coef ramp 0.1 → 1. Probe 6/10 (ckpt 7999).
Online collected 2 episodes then died. Kill at step 24000, NaN `actor_q`.
Rounds 2–8 collected 0 npz (all slices failed, 3 retries each). Status:
`outputs/cf_v23_radio_status.log`.

`G` never left a ~1% residual of the expert chunk (`q_gap≈0`). SR sat on
the frozen-specialist floor when it ran at all; it did not learn a new
radio policy. Abandoned: V23 `sample_actions` ignores `apply_guide` when
`cf_v23=True`.

## OT-compose MLP `G` — `cf_v23_ot`

Analytic OT base `V = (ã − noise) + G` with `G` an MLP on RL tokens
(`cf_guide_trunk` / `cf_guide_head`). Identity at `G=0`. Unit-ball
projection starved `G` to ~1% of `x_ref` RMS; turning unit-ball off
(`OPENPI_CF_UNIT_BALL=0`) still left `G / x_ref ≈ 1%` (`q_gap≈0.001`,
`actor_ref_rmse` well under the old 0.10 kill).

**AE + live TD, unbounded G** (`outputs/cf_v23_radio_online`, finished
2026-08-28): 50×1-ep on instance 308. Online **43/50 = 86%**, last-10
**6/10**, peak last-10 10/10 (R34–41), then a four-fail streak (R42–45)
pulled last-10 to 60%. Timeouts: 1, 13, 24, 42–45. Probe 1/1 in 1566
steps. Final ckpt 30229. No kill. Mean `G / x_ref` 0.96% (max 3.2% then
it shrank). Same frozen-specialist floor (8/10 on 308), not steering.

**No-AE live-TD cousin**: 15/18 = 83% then stopped. Same floor.

The MLP `G` path is unused by `compute_cf_v23_loss`. Helpers remain for
GraphDef / tests only.

## Frozen critic online

`OPENPI_CF_FREEZE_CRITIC=1` after AC pretrain. Combined with the
injection / ramp job above: actor had no live TD to follow, `actor_q`
NaN-killed. Current recipe keeps TD live (`freeze_critic=0`) and freezes
only the token-pool snapshot used for RL `z`.

## `β=1` anchor

Eq. 5 with `β=1` instead of `β=100`. The trust region is too weak for
23-d delta chunks: Q-ascent walks off the expert manifold before the
critic is meaningful. Current default is `OPENPI_CF_ANCHOR_BETA=100`
with `cf_anchor_normalize=1`.

## LoRA ×10 during AC / online

`OPENPI_CF_LORA_LR_MULT=10` was a workaround for `β=100` + BC winning
over default Adam. Through 10-step expert Euler the copy blew:

- AC ×10: pt12 copy destroyed in ~20 steps.
- Online ×10: see `cf_v23_ae` below.

`dq_da_rms` stayed ~3e-4 in every ×10 run, so the extra LoRA step was
transformer autodiff noise, not a real `∂Q/∂a`. Current default is ×1.
Q-ascent is allowed to move LoRA only after the critic actually depends
on the action.

## First action-expert job — `cf_v23_ae`

Same LoRA-on-`gemma_300m` actor as the current code, wrong MDP numbers
and a LoRA step the Euler graph cannot absorb. Finished 2026-08-30.
Frozen specialist on 308: **8/10 = 80%**. Online **18/50 = 36%**, last-10
**4/10**. Probe 0/1 timeout. Failures almost all 4300-step timeouts. No
`_KILLED` (`rmse` peak 0.063, kill was 0.10).

| Knob | `cf_v23_ae` | Effect |
| --- | --- | --- |
| Discount | `γ^32 ≈ 0.725` per chunk | Typical 308 success ~35 chunks. First-chunk return `0.725^35 ≈ 4e-5`. Q collapsed to `V(s)`. |
| LoRA Adam | ×10 online | Residual 2.4% → **22%** after round 1. Kill 0.10 never fired. |
| LR | cosine `1e-4 → 0` at 40k | Online starts at 16k; adapters frozen by ~round 40. |
| Replay | filtered to instance 308 (`OPENPI_CF_V23_INSTANCE`) | 10 trajs, **8 chunks with `r=1`**. Batch-8 TD never saw action ranking. |
| `dq_da_rms` | ~3e-4 the whole run | `−Q` through 10 expert Euler steps was noise. |

Rounds 1–4 still succeeded because serve loads **EMA** LoRA (`ema=0.999`).
After ~1500 online steps the blown adapters mixed in; SR dropped below
the specialist floor and never recovered.

Last-10 SR ended at 40%. Cumulative 36%. Timeouts: 5–8, 10–15, 17–18,
20, 22, 26, 28–35, 38–41, 43, 46–47, 49–50.

Do not resume `checkpoints/cf_v23_ae/`. Current recipe
(`cf_v23_ae2`) keeps the LoRA actor but uses per-chunk `γ=0.99`, LoRA
×1, constant LR after warmup, full 100-traj replay, `critic_coef=4`,
TD noise 0.08, kill 0.04.

## Cosine LR to zero

SFT schedule `CosineDecaySchedule` peak `1e-4`, decay to 0 at 40k
steps. RL online lives in 16k–33k; the last ~10 rounds had ~zero LR.
Replaced by `WarmupConstantSchedule` (warmup 200, then constant
`1e-4`).

## Instance-308-only replay

`data_loader.py` used to filter train chunks with
`OPENPI_CF_V23_INSTANCE`. On 308 that is 10 trajs / 8 success chunks.
TD on batch 8 cannot rank actions. Split: collect instance remains
`OPENPI_CF_V23_INSTANCE`; train filter is
`OPENPI_CF_REPLAY_INSTANCE` (empty = all 100 trajs).

## `γ^C` chunk discount

`chunk_discount = cf_gamma ** action_horizon` (`0.99^32 ≈ 0.725`).
Radio successes are ~35 chunks, so the first-chunk return is ~4e-5 and
the critic becomes a state baseline. Flag
`cf_discount_per_chunk=True` sets `chunk_discount = cf_gamma`. Legacy
path is still in `compute_cf_v23_loss` when the flag is off.

## Leftover `cf_v23_ae_301` watchdog

A host watchdog for an aborted instance-301 job kept `pkill`ing every
`train.py` in `b1k-airi-dev`. Host `cf_v23_watchdog.sh` was also briefly
0 bytes on disk (2026-08-30) and had to be restored from git. Current
watchdog still `pkill`s all `train.py` in the container — do not run two
exps against the same container.

## Checkpoint families (do not mix)

| Exp | Actor | Why dead |
| --- | --- | --- |
| `cf_v23_radio` | injection `v_θ − G` | NaN kill; `G` unused in current serve |
| `cf_v23_ot` | OT-compose MLP `G` | residual ~1%; GraphDef ≠ LoRA expert |
| `cf_v23_ae` | LoRA expert, ×10 / `γ^32` / cosine | copy blown, SR 36% |
| `cf_v23_ae2` | current | see `V23_METHODS.md` |
