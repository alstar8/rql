"""Do the JAX and PyTorch pi0.5 models produce the same actions from the same weights?

Converting a checkpoint between frameworks is only trustworthy if the two
implementations agree. A wrong weight mapping does not raise -- it produces plausible
numbers that steer the arm somewhere else, which is invisible until a full evaluation.

The comparison is exact rather than statistical because Policy.infer accepts injected
noise for both backends, so the flow-matching sampler starts from identical values and
any difference in the output is a difference in the model.

Run once per backend, then compare:

    python scripts/compare_backends.py --checkpoint <jax dir>     --out /tmp/jax.npz
    python scripts/compare_backends.py --checkpoint <pytorch dir> --out /tmp/torch.npz --device cpu
    python scripts/compare_backends.py --compare /tmp/jax.npz /tmp/torch.npz

The two runs need different virtualenvs, which is why this is one script run twice
rather than a single process loading both.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A realistic Pick state: the mean arm posture measured over the demonstrations.
STATE = np.array([-0.0017, -0.0774, 0.0463, -2.0859, -0.0220, 2.0248, 0.0884], dtype=np.float32)
PROMPT = "pick up the mug."
ACTION_HORIZON, ACTION_DIM = 15, 32


def build_observation() -> dict:
    rng = np.random.default_rng(12345)
    return {
        "observation/exterior_image_1_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": rng.integers(0, 256, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position": STATE,
        "observation/gripper_position": np.array([0.2], dtype=np.float32),
        "prompt": PROMPT,
    }


def fixed_noise() -> np.ndarray:
    # Same values fed to both samplers; without this the outputs differ by sampling alone.
    return np.random.default_rng(0).standard_normal((ACTION_HORIZON, ACTION_DIM)).astype(np.float32)


def run(checkpoint: Path, out: Path, device: str | None, norm_stats_from: Path | None) -> None:
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config("pi05_droid")

    norm_stats = None
    if norm_stats_from is not None:
        from openpi.shared import normalize as _normalize

        norm_stats = _normalize.load(norm_stats_from)
        print(f"norm stats loaded from {norm_stats_from}")

    kwargs = {}
    if device:
        kwargs["pytorch_device"] = device
    if norm_stats is not None:
        kwargs["norm_stats"] = norm_stats

    policy = _policy_config.create_trained_policy(train_config, str(checkpoint), **kwargs)
    print(f"policy loaded from {checkpoint}")

    actions = np.asarray(policy.infer(build_observation(), noise=fixed_noise())["actions"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, actions=actions)

    np.set_printoptions(precision=5, suppress=True)
    print(f"actions shape {actions.shape}")
    print(f"step 0: {actions[0][:8]}")
    print(f"wrote {out}")


def compare(a_path: Path, b_path: Path) -> None:
    a = np.load(a_path)["actions"]
    b = np.load(b_path)["actions"]
    if a.shape != b.shape:
        print(f"FAIL: shapes differ, {a.shape} vs {b.shape}")
        sys.exit(1)

    diff = np.abs(a - b)
    scale = np.abs(a).max()
    np.set_printoptions(precision=5, suppress=True)
    print(f"shape {a.shape}")
    print(f"jax     step 0 [:8]: {a[0][:8]}")
    print(f"pytorch step 0 [:8]: {b[0][:8]}")
    print()
    print(f"max |difference|   {diff.max():.6f}")
    print(f"mean |difference|  {diff.mean():.6f}")
    print(f"max |action|       {scale:.6f}")
    print(f"relative error     {diff.max() / max(scale, 1e-9):.4%}")

    # The arm channels are what actually drives the robot; the rest is padding to 32 dims.
    arm = np.abs(a[..., :7] - b[..., :7]).max()
    grip = np.abs(a[..., 7] - b[..., 7]).max()
    print(f"max |diff| arm     {arm:.6f} rad")
    print(f"max |diff| gripper {grip:.6f}")
    print()
    # Demonstration steps move about 0.023 rad, so a disagreement well under that cannot
    # change behaviour; anything approaching it would.
    if arm < 1e-3:
        print("VERDICT: implementations agree; the conversion preserved the weights")
    elif arm < 0.02:
        print("VERDICT: small disagreement, below one demonstration step of motion.")
        print("         Usable, but worth understanding before trusting it for eval.")
    else:
        print("VERDICT: DISAGREEMENT large enough to change behaviour. Do not use.")
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default=None, help="pytorch device, e.g. cpu")
    ap.add_argument("--norm-stats-from", type=Path, default=None,
                    help="directory holding norm_stats.json, if the checkpoint lacks assets")
    ap.add_argument("--compare", nargs=2, type=Path, default=None)
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
    else:
        if not args.checkpoint or not args.out:
            ap.error("--checkpoint and --out are required unless --compare is used")
        run(args.checkpoint, args.out, args.device, args.norm_stats_from)


if __name__ == "__main__":
    main()
