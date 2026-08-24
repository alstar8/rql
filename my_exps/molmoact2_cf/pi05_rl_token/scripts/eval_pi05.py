"""SUPERSEDED by scripts/run_eval.py -- kept to reproduce pre-refactor runs.

This drives the websocket server, which converts deltas to absolute targets itself.
The current path (scripts/run_eval.py, pi05/eval.py) converts in pi05/model.py
instead, so the chunk size and the conversion regime are visible in one config file.
Both produce identical actions at conversion="plan_time"; see
tests/test_pi05_model.py::test_plan_time_reproduces_the_old_server_side_conversion.

Evaluate a pi0.5 checkpoint in the MolmoSpaces simulator, end to end.

One command: bring up the policy server, run the benchmark, collect the success rate
with a confidence interval, build a grid video of the episodes, and shut the server down
again. The last part matters -- a policy server left running holds a GPU and about 12 GB
for nothing, so it is stopped even when the run fails.

    python scripts/eval_pi05.py --checkpoint <dir> --episodes 128

Defaults evaluate the first 128 episodes of the shipped val benchmark, which is the slice
MolmoAct2 was measured on (43/128 and 47/128 in two runs), so the numbers are comparable.

The eval runs with chunk_size=1 because the model speaks deltas; see pi05/eval_configs.py.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MOLMOSPACES = Path("/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces")
OPENPI = Path("/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/openpi")
ASSETS = Path("/home/jovyan/users/staroverov/B1K/mlspaces/assets")
CACHE = Path("/home/jovyan/users/staroverov/B1K/mlspaces/cache")
VAL_BENCHMARK = (
    CACHE / "benchmarks/molmospaces-bench-v1/20260408/procthor-10k/FrankaPickDroidMiniBench"
    / "FrankaPickDroidMiniBench_json_benchmark_20251231"
)


def wait_for_server(log: Path, timeout: float) -> bool:
    """The server is ready once it says it is listening; a crash shows up in the log too."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if log.exists():
            text = log.read_text(errors="replace")
            if "server listening" in text:
                return True
            if "Traceback" in text or "Error" in text:
                print(text[-2000:])
                return False
        time.sleep(3)
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--benchmark", type=Path, default=VAL_BENCHMARK)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True, help="output directory for this eval")
    ap.add_argument("--server-gpu", default="0", help="CUDA device for the policy server")
    ap.add_argument("--sim-gpu", default="1", help="CUDA device for the simulator")
    ap.add_argument("--egl-device", default="1", help="MUJOCO_EGL_DEVICE_ID (EGL order != CUDA order)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--task-horizon", type=int, default=500)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument(
        "--server-script",
        type=Path,
        default=None,
        help="policy server to launch; defaults to scripts/serve_pi05.py",
    )
    ap.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="extra argument passed through to the server, repeatable",
    )
    ap.add_argument(
        "--server-python",
        type=Path,
        default=OPENPI / ".venv/bin/python",
        help="interpreter for the policy server; PyTorch checkpoints need the venv "
             "carrying the transformers_replace patch, not openpi's own",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    server_log = args.out / "server.log"
    eval_log = args.out / "eval.log"

    server_env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": args.server_gpu,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "HF_HUB_OFFLINE": "1",
    }
    server_cmd = [
        str(args.server_python),
        str(args.server_script or ROOT / "scripts/serve_pi05.py"),
        "--checkpoint", str(args.checkpoint),
        "--port", str(args.port),
        *args.server_arg,
    ]

    print(f"starting policy server on GPU {args.server_gpu} (port {args.port})")
    server = subprocess.Popen(
        server_cmd, stdout=server_log.open("w"), stderr=subprocess.STDOUT,
        env=server_env, cwd=str(OPENPI), start_new_session=True,
    )

    result = {"checkpoint": str(args.checkpoint), "episodes_requested": args.episodes}
    try:
        if not wait_for_server(server_log, timeout=600):
            raise RuntimeError(f"policy server never became ready; see {server_log}")
        print("server ready")

        eval_env = {
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": str(ROOT),
            "MUJOCO_GL": "egl",
            "MUJOCO_EGL_DEVICE_ID": args.egl_device,
            "CUDA_VISIBLE_DEVICES": args.sim_gpu,
            "MLSPACES_ASSETS_DIR": str(ASSETS),
            "MLSPACES_CACHE_DIR": str(CACHE),
            # The eval config reads these so client and server agree on the endpoint.
            "PI05_POLICY_HOST": "127.0.0.1",
            "PI05_POLICY_PORT": str(args.port),
        }
        eval_cmd = [
            str(MOLMOSPACES / ".venv/bin/python"),
            "molmo_spaces/evaluation/eval_main.py",
            "pi05.eval_configs:Pi05PickEvalConfig",
            "--benchmark_dir", str(args.benchmark),
            "--task_horizon_steps", str(args.task_horizon),
            "--max_episodes", str(args.episodes),
            "--num_workers", str(args.workers),
            "--no_wandb",
            "--output_dir", str(args.out / "eval_output"),
        ]
        print(f"running {args.episodes} episodes on GPU {args.sim_gpu} (EGL {args.egl_device})")
        started = time.time()
        with eval_log.open("w") as handle:
            code = subprocess.run(
                eval_cmd, stdout=handle, stderr=subprocess.STDOUT, env=eval_env,
                cwd=str(MOLMOSPACES),
            ).returncode
        result["eval_seconds"] = round(time.time() - started, 1)
        result["eval_returncode"] = code

        # eval_main reports the rate in its log; it does not write results.json.
        successes = total = None
        for line in eval_log.read_text(errors="replace").splitlines():
            if "Evaluation complete:" in line:
                fragment = line.split("Evaluation complete:")[1].strip().split()[0]
                successes, total = (int(x) for x in fragment.split("/"))
        result["successes"], result["episodes_run"] = successes, total

        if total:
            rate = successes / total
            result["success_rate"] = round(rate, 4)
            lo, hi = wilson_interval(successes, total)
            result["wilson_95"] = [round(lo, 4), round(hi, 4)]
            print(f"\nSUCCESS RATE: {successes}/{total} = {100 * rate:.1f}%  "
                  f"(95% Wilson {100 * lo:.1f}-{100 * hi:.1f}%)")
        else:
            print(f"\nno episodes completed; see {eval_log}")

        if not args.no_video and total:
            make_grid_video(args.out)

    finally:
        # Always: an orphaned server holds a GPU until someone notices.
        print("stopping policy server")
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        server.wait(timeout=30)

    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out / 'result.json'}")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: sane at the 0% and 100% ends, unlike the normal one."""
    if total == 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def make_grid_video(out: Path) -> None:
    """Build the 4x4 grid, green-bordered on success, using the existing rl_token tool."""
    runs = sorted((out / "eval_output").rglob("*.mp4"))
    if not runs:
        print("no episode videos found, skipping grid")
        return
    # make_grid_video walks an eval output directory itself; hand it the run root, and
    # name the camera the demonstrations and benchmark actually record.
    eval_dirs = sorted((out / "eval_output").glob("*/*"))
    episode_root = eval_dirs[-1] if eval_dirs else runs[0].parent.parent
    cmd = [
        str(MOLMOSPACES / ".venv/bin/python"),
        str(ROOT / "scripts/make_grid_video.py"),
        "--eval_dir", str(episode_root),
        "--camera", "exo_camera_1",
        "--out", str(out / "grid.mp4"),
    ]
    print(f"building grid video from {len(runs)} clips")
    result = subprocess.run(cmd, capture_output=True, text=True)
    code = result.returncode
    if code != 0:
        print(f"grid video failed (exit {code}); episode clips are still under {episode_root}")
        print((result.stderr or result.stdout or "").strip()[-600:])
    else:
        print(f"grid video: {out / 'grid.mp4'}")


if __name__ == "__main__":
    main()
