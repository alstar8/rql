"""Run ``benchs/bench_server.py`` with the proprio gripper channel corrected.

The shipped ``SimlerWrapper._compute_eef_pos`` hands pi0 ``1 - qpos[-1]``, where
qpos[-1] is the WidowX finger joint in metres. Across its whole travel that is
0.962 (open) to 0.986 (closed): near-constant, and inverted. The Bridge proprio
the checkpoint was trained on spans [0.05, 1.01] with 0 = closed, 1 = open.

This launcher patches that one entry and then runs the unmodified bench server,
so nothing in the repository changes. Every argument is forwarded verbatim.

    python bench_server_fixed.py --seed 0 --scenes X --use_vlm 1 ...

``--no-fix`` runs the original (buggy) channel through the same launcher, so the
control arm differs from the fixed arm in nothing but this one number.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLA_DATA_DIR", "./")

import runpy
import sys
from pathlib import Path

REPO = Path("/home/jovyan/users/staroverov/GuideVLA")
BENCH_SERVER = REPO / "notebooks/benchs/bench_server.py"

# WidowX finger joint travel, from ManiSkill's WidowX250SSimpler gripper controller:
# lower = 0.015 - 0.001, upper = 0.037 + 0.001
GRIPPER_LOW, GRIPPER_HIGH = 0.014, 0.038


def patch_gripper_channel() -> None:
    import torch
    from simpler_env.env.simpler_wrapper import SimlerWrapper

    original = SimlerWrapper._compute_eef_pos

    def patched(self, obs):
        eef_pos = original(self, obs)                                  # [B, 8]
        qpos = self.env.unwrapped.agent.robot.get_qpos()[:, -1:]
        openness = ((qpos - GRIPPER_LOW) / (GRIPPER_HIGH - GRIPPER_LOW)).clamp(0.0, 1.0)
        return torch.cat([eef_pos[:, :7], openness.to(eef_pos.dtype)], dim=1)

    SimlerWrapper._compute_eef_pos = patched


if __name__ == "__main__":
    if "--no-fix" in sys.argv:
        sys.argv.remove("--no-fix")
        print("[gripper] control run -- original channel left in place", flush=True)
    else:
        patch_gripper_channel()
        print("[gripper] patched to normalised openness", flush=True)

    runpy.run_path(str(BENCH_SERVER), run_name="__main__")
