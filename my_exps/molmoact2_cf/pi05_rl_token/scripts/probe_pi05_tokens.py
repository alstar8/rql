"""Check that the RL-Token hook actually captures pi0.5's prefix tokens.

RL Token's whole state representation comes from these tokens, so a hook that silently
catches nothing -- or catches a denoising step instead of the prefix pass -- would train
the actor on garbage while everything still runs. This loads the fine-tuned checkpoint,
does one inference, and reports what the hook saw.

    python scripts/probe_pi05_tokens.py --checkpoint <pytorch ckpt>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STATE = np.array([-0.0017, -0.0774, 0.0463, -2.0859, -0.0220, 2.0248, 0.0884], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", default="pi05_droid_finetune")
    args = ap.parse_args()

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    from pi05.vla_server import FrozenPi05

    policy = _policy_config.create_trained_policy(
        _config.get_config(args.config), str(args.checkpoint), pytorch_device="cuda"
    )
    frozen = FrozenPi05(policy)
    print("policy loaded, hook attached")

    rng = np.random.default_rng(0)
    observation = {
        "observation/exterior_image_1_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": STATE,
        "observation/gripper_position": np.array([0.2], dtype=np.float32),
        "prompt": "pick up the bottle.",
    }

    out = frozen.predict(observation)
    actions, tokens, mask = out["actions"], out["token_features"], out["token_attention_mask"]

    np.set_printoptions(precision=4, suppress=True)
    print(f"\nactions       {actions.shape}   first step {actions[0][:8]}")
    print(f"token_features {tokens.shape}   dtype {tokens.dtype}")
    print(f"token mask     {mask.shape}   sum {mask.sum():.0f}")
    print(f"token stats    mean {tokens.mean():+.4f}  std {tokens.std():.4f}  "
          f"min {tokens.min():+.3f}  max {tokens.max():+.3f}")

    problems = []
    if tokens.shape[-1] != 2048:
        problems.append(f"token width {tokens.shape[-1]}, expected 2048")
    if tokens.shape[0] < 64:
        problems.append(f"only {tokens.shape[0]} tokens; the prefix should hold hundreds "
                        "(two images plus language), so a denoising step was captured")
    if not np.isfinite(tokens).all():
        problems.append("tokens contain non-finite values")
    if float(tokens.std()) < 1e-6:
        problems.append("tokens are constant, so they carry no scene information")

    # The same scene twice must give the same tokens: the prefix pass is deterministic
    # even though action sampling is not.
    again = frozen.predict(observation)["token_features"]
    drift = float(np.abs(again - tokens).max())
    print(f"repeat determinism: max |diff| {drift:.2e}")
    if drift > 1e-3:
        problems.append(f"tokens differ by {drift:.3e} between identical calls")

    # A different scene must give different tokens, or the hook is capturing something
    # that does not depend on the input.
    other = dict(observation)
    other["observation/exterior_image_1_left"] = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    other_tokens = frozen.predict(other)["token_features"]
    delta = float(np.abs(other_tokens - tokens).mean())
    print(f"different image: mean |diff| {delta:.4f}")
    if delta < 1e-4:
        problems.append("tokens barely change with a different image")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("VERDICT: hook captures the prefix tokens; shape, determinism and sensitivity all hold")


if __name__ == "__main__":
    main()
