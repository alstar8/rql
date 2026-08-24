"""What scale of actions does the pretrained pi05_droid checkpoint actually emit?

The MolmoSpaces eval bridge feeds the model's 7 arm outputs to the environment as
*absolute* joint position targets. The checkpoint's own norm stats put those outputs
at mean~0, std~0.1, which cannot be absolute Franka targets (joint 4 sits near -2 rad).
This runs one real inference and prints the raw numbers, so the question is settled by
observation rather than by reading stats.

Loads the checkpoint, does a handful of inferences, exits. Leaves nothing on the GPU.
"""

import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config

CKPT = "/home/jovyan/users/staroverov/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos"

# A plausible MolmoSpaces Pick state: the mean arm posture measured over 392 demos.
STATE = np.array([-0.0017, -0.0774, 0.0463, -2.0859, -0.0220, 2.0248, 0.0884], dtype=np.float32)

def main() -> None:
    cfg = _config.get_config("pi05_droid")
    policy = policy_config.create_trained_policy(cfg, CKPT)
    print("loaded policy:", type(policy).__name__)

    rng = np.random.default_rng(0)
    for trial in range(3):
        example = {
            "observation/exterior_image_1_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
            "observation/joint_position": STATE,
            "observation/gripper_position": np.array([0.2], dtype=np.float32),
            "prompt": "pick up the mug.",
        }
        out = policy.infer(example)
        act = np.asarray(out["actions"])
        print(f"\n--- trial {trial}: actions shape {act.shape}")
        np.set_printoptions(precision=4, suppress=True)
        print("  step 0 :", act[0])
        print("  step 1 :", act[1])
        print("  chunk mean:", act.mean(0))
        print("  chunk min :", act.min(0))
        print("  chunk max :", act.max(0))

    print("\nstate given to the model (arm):", STATE)
    print("If actions[:7] hover near 0 they are deltas/velocities, not absolute targets.")
    print("If they hover near the state above they are absolute joint positions.")


if __name__ == "__main__":
    main()
