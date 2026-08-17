# V16 improved RLT+CF (z=256 / d=512 / 4 layers)

Status: **pretrain → launch** (`runs/rlt_cf_v16_rlt_improved`).  
Predecessor: V16 controlled long run (RLT actors/CF below baseline; actor SR collapse after BC warmup).

## Changes vs V16 defaults

| Knob | V16 controlled | This run |
| --- | --- | --- |
| RL-token `z` | 256 | **256** |
| RL-token `d_model` | 256 | **512** |
| RL-token layers | 2 | **4** |
| Chunk store | non-overlapping C=8 | **same** (explicitly **not** paper stride-2) |
| Soft β / collect | β=2, always collect actor | unchanged |
| Arms | 8-way matrix | **`residual_rlt_cf` GPUs 0–3**, **`flow_rlt_cf` GPUs 4–7** (4 shards each) |

Paper stride-2 would store overlapping `<x_t, a_{t:t+C}>` every 2 control steps. We keep normal non-overlapping batch order at chunk boundaries.

## Checkpoints

```bash
bash pretrain_rlt_d512_l4.sh
# → runs/rlt_pretrain_demo1k_z256_d512_l4/
#    rlt_token_demo1k_z256_d512_l4.pt
#    rlt_cf_pretrain_demo1k_z256_d512_l4.pt
#    rlt_cf_flow_pretrain_demo1k_z256_d512_l4.pt
```

Reuses demo1k token/chunk NPZs (re-encodes `z` with the new AE).

## Launch

```bash
# After pretrain completes:
V16_MODE=long FRESH=1 bash launch_v16_rlt_cf_improved.sh
```

Ports: `8720–8727` (avoids V16 controlled `8710–8716`).  
Logs: `tmp/rlt_cf_v16_rlt_improved_logs/`.
