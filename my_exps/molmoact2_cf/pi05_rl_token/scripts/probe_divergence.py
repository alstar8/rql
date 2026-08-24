"""Where do two machines start to disagree?

Runs one batch of benchmark episodes exactly as ``benchs/bench_server.py`` does
with ``--use_vlm 0``, and records a fingerprint at every policy call: the image
the policy sees, the proprioceptive state it sees, and the action chunk it
returns. Run it on both machines with the same ``--seed``, then compare the two
JSON files. The first call whose hashes disagree names the channel:

    call 0, image hash differs          -> rendering (Vulkan; GPU model / driver)
    call 0, image equal, action differs -> policy arithmetic (inductor / cuBLAS / SDPA)
    call 0 equal, a later call differs  -> physics (PhysX GPU solver)
    nothing differs                     -> the machines agree; the gap is elsewhere

``--no-compile`` replaces ``torch.compile`` with the identity before the model is
built. Running with and without it says how much of the divergence is the
TorchInductor version rather than the hardware.

Writes only the two files derived from ``--out``: the JSON of fingerprints, and a
``.npy`` holding the first action chunk in full so the numbers can be subtracted.
Changes nothing else on disk, and exits after one batch.

    python probe_divergence.py --repo /workspace --seed 0 --out probe_ml2_s0.json
"""

from __future__ import annotations

import os

# agent.configuration_pipeline reads this at import time, exactly as bench_server.py sets it
os.environ.setdefault("VLA_DATA_DIR", "./")

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

VLA_PATH = "juexzz/INTACT-pi0-finetune-bridge"


# ---------------------------------------------------------------------------
# fingerprints
# ---------------------------------------------------------------------------

def _array(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return np.ascontiguousarray(x.detach().cpu().numpy())
    return np.ascontiguousarray(np.asarray(x))


def sha(x) -> str:
    """Content hash of the exact bytes -- what the two machines must agree on."""
    return hashlib.sha256(_array(x).tobytes()).hexdigest()[:16]


def stats(x) -> dict:
    a = _array(x).astype(np.float64)
    return {
        "shape": list(a.shape),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "sum": float(a.sum()),
    }


# ---------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------

def import_env_class(repo: Path):
    """``archer`` is importable through PYTHONPATH on one machine and through an
    editable install on the other. Fall back to the repo path only if neither is."""
    try:
        from archer.pi0_ms3 import EfficientMultiScenePI0Env
    except ModuleNotFoundError:
        sys.path.insert(0, str(repo / "SimplerEnv"))
        sys.path.insert(0, str(repo))
        from archer.pi0_ms3 import EfficientMultiScenePI0Env
    return EfficientMultiScenePI0Env


# WidowX finger joint travel, from ManiSkill's WidowX250SSimpler gripper controller:
# lower = 0.015 - 0.001, upper = 0.037 + 0.001
GRIPPER_LOW, GRIPPER_HIGH = 0.014, 0.038


def patch_gripper_channel() -> None:
    """Replace the gripper entry of the proprio vector with a normalised openness.

    The shipped ``_compute_eef_pos`` returns ``1 - qpos[-1]``, where qpos[-1] is the
    finger joint in metres. Over its whole travel that is 0.962 (open) to 0.986
    (closed): near-constant, and inverted. Bridge proprio expects 0 = closed,
    1 = open. Everything else in the vector is left exactly as it was.
    """
    from simpler_env.env.simpler_wrapper import SimlerWrapper

    original = SimlerWrapper._compute_eef_pos

    def patched(self, obs):
        eef_pos = original(self, obs)                                  # [B, 8], last entry wrong
        qpos = self.env.unwrapped.agent.robot.get_qpos()[:, -1:]
        openness = ((qpos - GRIPPER_LOW) / (GRIPPER_HIGH - GRIPPER_LOW)).clamp(0.0, 1.0)
        return torch.cat([eef_pos[:, :7], openness.to(eef_pos.dtype)], dim=1)

    SimlerWrapper._compute_eef_pos = patched


def describe_machine() -> dict:
    out = {
        "host": platform.node(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": os.getcwd(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled"):
        try:
            out[name] = getattr(torch.backends.cuda, name)()
        except Exception:
            pass
    for mod in ("sapien", "mani_skill"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
            if hasattr(m, "ASSET_DIR"):
                out[mod + "_asset_dir"] = str(m.ASSET_DIR)
        except Exception:
            out[mod] = "not importable"
    return out


def weight_fingerprint(policy) -> dict:
    """Prove both machines put the same numbers in GPU memory, not just on disk."""
    try:
        model = policy.wrapper.model
        model = getattr(model, "_orig_mod", model)          # unwrap torch.compile
        named = list(model.named_parameters())
        middle = len(named) // 2
        picked = named[:1] + named[middle:middle + 1] + named[-1:]
        return {
            "n_parameters": len(named),
            "total_elements": int(sum(p.numel() for _, p in named)),
            "samples": {n: sha(p) for n, p in picked},
        }
    except Exception as exc:                                 # never fail the run over a diagnostic
        return {"error": repr(exc)}


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------

def run(args) -> dict:
    repo = Path(args.repo).resolve()
    EfficientMultiScenePI0Env = import_env_class(repo)

    if args.no_compile:
        torch.compile = lambda model, *a, **kw: model        # before the model is built

    if args.fix_gripper:
        patch_gripper_channel()

    # bench_server.main, verbatim
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    episode_ids = torch.from_numpy(np.random.randint(0, 1000000000, size=(args.num_envs,)))

    env = EfficientMultiScenePI0Env(num_envs=args.num_envs, scenes=[args.scene], vla_path=VLA_PATH)

    # the client's 'reset' command
    obs, instr, _ = env.get_obs(env_id=0, episode_id=episode_ids, obj_set=args.obj_set)
    obs = obs.cpu().numpy()

    calls: list[dict] = []
    first_chunk: dict = {}
    original_get_action = env.policy.get_action

    def recording_get_action(x, *a, **kw):
        i = len(calls)
        row = {
            "call": i,
            "image_sha": sha(x["image"]),
            "image": stats(x["image"]),
            "state_sha": sha(x["pi_0"]["eef_pos"]),
            "state": stats(x["pi_0"]["eef_pos"]),
            "gripper_channel": stats(x["pi_0"]["eef_pos"][:, 7]),
        }
        t0 = time.perf_counter()
        values, actions, logprobs = original_get_action(x, *a, **kw)
        row["seconds"] = round(time.perf_counter() - t0, 3)
        row["action_sha"] = sha(actions)
        row["action"] = stats(actions)
        row["action_env0"] = [float(v) for v in _array(actions)[0].ravel()[:24]]
        calls.append(row)
        if i == 0:
            first_chunk["actions"] = _array(actions)
        return values, actions, logprobs

    env.policy.get_action = recording_get_action

    # the client's 'step' command. use_vlm=0 makes the client echo the instruction
    # back, and extract_answer is the identity on it (verified in the saved YAMLs).
    t0 = time.perf_counter()
    reward, sr, cgr, gr = env.step_single_with_id(0, list(instr), args.obj_set, episode_ids)
    elapsed = time.perf_counter() - t0

    result = {
        "args": vars(args),
        "machine": describe_machine(),
        "weights": weight_fingerprint(env.policy),
        "episode_ids_sha": sha(episode_ids),
        "episode_ids": episode_ids.tolist(),
        "instruction_sha": hashlib.sha256("|".join(instr).encode()).hexdigest()[:16],
        "reset_image_sha": sha(obs),
        "calls": calls,
        "stats": {
            "reward": float(np.mean(_array(reward))),
            "success": float(np.mean(_array(sr))),
            "consecutive_grasp": float(np.mean(_array(cgr))),
            "is_src_obj_grasped": float(np.mean(_array(gr))),
        },
        "per_episode_success": [float(v) for v in _array(sr).ravel()],
        "seconds_total": round(elapsed, 1),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    if "actions" in first_chunk:
        np.save(out.with_suffix(".npy"), first_chunk["actions"])
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="directory holding archer/ and SimplerEnv/")
    p.add_argument("--scene", default="PutOnColorInSceneMulti-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--obj_set", default="test")
    p.add_argument("--out", required=True)
    p.add_argument("--no-compile", action="store_true", help="run the model eager, without TorchInductor")
    p.add_argument("--fix-gripper", action="store_true", help="feed a normalised gripper openness instead of 1 - qpos")
    return p.parse_args()


if __name__ == "__main__":
    res = run(parse_args())
    print("")
    print("success {:.4f}  grasp {:.4f}  {} policy calls  {}s".format(
        res["stats"]["success"], res["stats"]["is_src_obj_grasped"],
        len(res["calls"]), res["seconds_total"]))
    g = [r["gripper_channel"] for r in res["calls"]]
    print("gripper channel over the episode: min {:.4f}  max {:.4f}".format(
        min(x["min"] for x in g), max(x["max"] for x in g)))
    for row in res["calls"][:3]:
        print("  call {}: image {}  state {}  action {}".format(
            row["call"], row["image_sha"], row["state_sha"], row["action_sha"]))
