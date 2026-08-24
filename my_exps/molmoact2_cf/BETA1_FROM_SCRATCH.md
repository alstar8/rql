# Beta=1 RL Token from a freshly collected frozen-pi0.5 corpus

Both tasks start from the **pretrained pi0.5 checkpoint only**. The phase-1 token
encoder is **not** the original `ae_desk_mug.pt`. Each task collects its own token
sequences, trains its own AE, then runs RL Token at **beta=1** (the value used in
the original desk_mug matrix cell that scored 63/64).

## What is collected

`scripts/run_collect.py --target 2500`, **two parallel collectors** per scene — the
PI05 recipe. One token sequence is one VLA call (chunk 8). A 500-step failure is
~63 sequences; a quick success is ~13. Tokens are recorded for the **whole
episode** (the AE has to represent states before the gate as well as after).

Success rate on this corpus is the frozen VLA’s train-split rate. It is **not** an
AE training label.

After collect finishes, counts land in:

- `runs/beta1_from_scratch/<scene>/corpus_stats.json`
- `runs/beta1_from_scratch/<scene>/corpus_stats.md`

## Mug (GPUs 0–3)

| Step | GPUs | What |
| --- | --- | --- |
| Collect | 0, 1 | train-split `desk_mug`, horizon 500, gate 56 |
| Frozen eval | 2 | held-out 16 specs → 64 rollouts |
| AE | 3 | 8000 steps, cap 6000 sequences |
| RL Token | 0 and 1 | seeds 0/1, beta=1, 300 eps, warmup 40 |
| Actor eval | 2 then 3 | `agent.pt` on held-out bench |

## Kettle (GPUs 4–7)

Same pipeline on the V21 pose-0 kettle benches, horizon **400**, gate **40**.

## Original PI05 desk_mug collect (for comparison)

The AE this project used before was trained on **127 train episodes, 61
successes, 48.0% SR**, plus other pick scenes for `ae_combined`. This run does
**not** reuse those shards.
