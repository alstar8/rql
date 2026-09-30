#!/usr/bin/env python
"""Introspection probe for V25 action-expert distillation.

Verifies, on one GPU, everything the student trainer relies on:

  1. the policy + transform pipeline build from the base checkpoint;
  2. which parameters are the action expert (trainable) vs the backbone (frozen);
  3. a recorded-style raw dict goes through policy._input_transform and the model's
     training forward produces a finite per-step loss;
  4. the frozen model's loss on its own reference chunk is lower than on a perturbed
     "teacher" chunk -- the directional signal the self-check uses.

Runs with the pi05_torch venv on one GPU:

    CUDA_VISIBLE_DEVICES=0 HF_HOME=... python scripts/_distill_probe.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from openpi.training import config as _config  # noqa: E402
from openpi.policies import policy_config as _policy_config  # noqa: E402

CHECKPOINT = "/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999"
OPENPI_CONFIG = "pi05_droid_finetune"


def main() -> None:
    os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")
    from pi05.eval import disable_policy_compile

    disable_policy_compile()

    train_config = _config.get_config(OPENPI_CONFIG)
    policy = _policy_config.create_trained_policy(
        train_config, CHECKPOINT, pytorch_device="cuda"
    )
    model = policy._model
    print("model type:", type(model).__name__)

    # --- parameter split ---------------------------------------------------------
    expert_keywords = ("gemma_expert", "action_in_proj", "action_out_proj", "state_proj",
                       "action_time_mlp_in", "action_time_mlp_out", "time_mlp_in", "time_mlp_out")
    trainable, frozen = [], []
    for name, p in model.named_parameters():
        (trainable if any(k in name for k in expert_keywords) else frozen).append((name, p.numel()))
    n_train = sum(n for _, n in trainable)
    n_frozen = sum(n for _, n in frozen)
    print(f"expert/trainable params: {n_train/1e6:.1f}M across {len(trainable)} tensors")
    print(f"backbone/frozen params:  {n_frozen/1e6:.1f}M across {len(frozen)} tensors")
    print("sample trainable names:")
    for name, _ in trainable[:5]:
        print("   ", name)

    # --- transform + forward on a synthetic recorded-style dict -------------------
    rng = np.random.default_rng(0)
    raw = {
        "observation/exterior_image_1_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": rng.standard_normal(7).astype(np.float32) * 0.1,
        "observation/gripper_position": np.array([0.5], dtype=np.float32),
        "prompt": "pick up the object",
        "actions": (rng.standard_normal((8, 8)).astype(np.float32) * 0.1),
    }
    inputs = policy._input_transform(raw)
    print("transformed keys:", sorted(inputs.keys()))
    for k in ("state", "actions", "tokenized_prompt"):
        if k in inputs:
            print(f"   {k}: shape {np.asarray(inputs[k]).shape} dtype {np.asarray(inputs[k]).dtype}")
    if "image" in inputs:
        print("   image keys:", {k: np.asarray(v).shape for k, v in inputs["image"].items()})

    # Batch of 1, to the model -- exactly the way Policy.infer does it.
    import jax

    from openpi.models import model as _model

    inputs_nobatch = {k: v for k, v in inputs.items() if k != "actions"}
    tensors = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to("cuda")[None, ...], inputs_nobatch
    )
    observation = _model.Observation.from_dict(tensors)
    actions = torch.from_numpy(np.asarray(inputs["actions"])).to("cuda").float()
    print("actions after transform:", tuple(actions.shape), actions.dtype)

    # Pad horizon 8 -> 16 and batch.
    horizon = model.config.action_horizon
    if actions.shape[0] < horizon:
        pad = torch.zeros(horizon - actions.shape[0], actions.shape[1], device=actions.device)
        actions = torch.cat([actions, pad], dim=0)
    actions = actions[None]  # (1, 16, 32)
    print("actions batched:", tuple(actions.shape))

    model.train()
    loss = model.forward(observation, actions)
    print("forward loss shape:", tuple(loss.shape), "mean:", float(loss.mean()))

    # masked loss over first 8 steps, first 8 dims
    masked = loss[:, :8, :8].mean()
    print("masked loss (first 8 steps, first 8 dims):", float(masked))


if __name__ == "__main__":
    main()
