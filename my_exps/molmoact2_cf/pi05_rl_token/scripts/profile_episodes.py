"""Milestone 1, step 2: find a training episode the VLA solves sometimes.

Runs the frozen VLA on each candidate episode many times and reports the success
rate with a Wilson interval. What we want is an episode in roughly 20-60%: at 0%
RL never sees a positive example, at 100% there is nothing to improve.

    python scripts/profile_episodes.py --episodes 128-135 --rollouts 30 \
        --shards 3 --parallel 12 --ports 8000,8001,8002,8003

Each (episode, shard) pair is one subprocess, so a single episode's rollouts can
be spread over several processes too. One VLA server sustains about three
collectors: an episode spends ~30% of its time waiting on inference and the rest
in MuJoCo. Rendering follows MUJOCO_EGL_DEVICE_ID, which is not the CUDA order --
see scripts/egl_device_map.py.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlt.evaluate import wilson  # noqa: E402


def parse_episodes(text: str) -> list[int]:
    """'128-131,140' -> [128, 129, 130, 131, 140]"""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def launch(job: int, episode: int, shard: int, repeats: int, args) -> subprocess.Popen:
    port = [int(p) for p in args.ports.split(",")][job % len(args.ports.split(","))]
    egl = [int(d) for d in args.egl_devices.split(",")][job % len(args.egl_devices.split(","))]
    env = dict(os.environ, PYTHONUNBUFFERED="1", MUJOCO_EGL_DEVICE_ID=str(egl))
    tag = f"_s{shard}"
    cmd = [
        args.python, "-m", "rlt.evaluate",
        "--episode_idx", str(episode),
        "--repeats", str(repeats),
        "--tag", tag,
        "--token_ae", args.token_ae,
        "--out_dir", args.out_dir,
        "--server_port", str(port),
        "--horizon", str(args.horizon),
        "--actor", args.actor,
        "--benchmark_dir", args.benchmark_dir,
    ]
    if args.video_dir:
        # One directory per shard: rollout numbering restarts in each process,
        # so a shared directory would have them overwrite each other.
        cmd += ["--video_dir", f"{args.video_dir}/e{episode}_s{shard}"]
    log_path = Path(args.out_dir) / f"episode_{episode}{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  episode {episode:>4} shard {shard}  x{repeats}  port {port}  egl {egl}")
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT)


def collect(episode: int, args) -> dict | None:
    """Merge the shards of one episode."""
    outcomes: list[int] = []
    steps: list[float] = []
    for shard in range(args.shards):
        path = Path(args.out_dir) / f"episode_{episode}_s{shard}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        outcomes += data["successes_mask"]
        steps += [data["mean_steps"]] * data["rollouts"]
    if not outcomes:
        return None
    k, n = sum(outcomes), len(outcomes)
    lo, hi = wilson(k, n)
    return {
        "episode_idx": episode,
        "rollouts": n,
        "successes": k,
        "success_rate": k / n,
        "ci95": [lo, hi],
        "mean_steps": sum(steps) / len(steps),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", default="128-135", help="e.g. 128-143 or 128,140,155")
    ap.add_argument("--rollouts", type=int, default=30, help="per episode, split over --shards")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--ports", default="8000", help="comma-separated VLA server ports")
    ap.add_argument("--egl_devices", default="0,1,2,3")
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--actor", default="", help="checkpoint to evaluate instead of the plain VLA")
    ap.add_argument("--video_dir", default="", help="keep one mp4 per rollout, for the grids")
    ap.add_argument("--benchmark_dir", default="", help="a variant benchmark from make_scene_variants.py")
    ap.add_argument("--token_ae", default="runs/ae_sweep/deep_decoder.pt")
    ap.add_argument("--out_dir", default="runs/episode_profile")
    ap.add_argument(
        "--python",
        default="/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/.venv/bin/python",
    )
    args = ap.parse_args()

    episodes = parse_episodes(args.episodes)
    per_shard = -(-args.rollouts // args.shards)
    jobs = [(e, s) for e in episodes for s in range(args.shards)]
    print(f"{len(episodes)} candidates x {args.shards} shards x {per_shard} rollouts, "
          f"{args.parallel} processes at a time")

    running: list[tuple[int, int, subprocess.Popen]] = []
    pending = list(enumerate(jobs))
    start = time.time()
    while pending or running:
        while pending and len(running) < args.parallel:
            index, (episode, shard) = pending.pop(0)
            running.append((episode, shard, launch(index, episode, shard, per_shard, args)))
        time.sleep(5)
        for entry in list(running):
            episode, shard, proc = entry
            if proc.poll() is not None:
                running.remove(entry)
                print(f"  episode {episode} shard {shard} exited with {proc.returncode} "
                      f"({len(pending)} queued, {len(running)} running)")

    print(f"\nall done in {(time.time() - start) / 60:.1f} min\n")
    print(f"{'episode':>8} {'ok':>4} {'n':>4} {'SR':>7}  {'95% CI':>16}  {'mean steps':>10}")
    summary = {}
    for episode in episodes:
        row = collect(episode, args)
        if row is None:
            print(f"{episode:>8}   no result -- see the logs")
            continue
        summary[str(episode)] = row
        lo, hi = row["ci95"]
        usable = 0.2 <= row["success_rate"] <= 0.6
        print(
            f"{episode:>8} {row['successes']:>4} {row['rollouts']:>4} {row['success_rate']:>6.1%}  "
            f"[{lo:>5.1%}, {hi:>5.1%}]  {row['mean_steps']:>10.0f}" + ("   <- usable" if usable else "")
        )

    path = Path(args.out_dir) / "profile.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
