"""Evaluate checkpoints as training produces them, and keep only the ones worth keeping.

Runs alongside training on the GPUs training is not using. For each new checkpoint it
runs the full benchmark slice, records the success rate in a ledger, and then keeps the
latest checkpoint and the best-scoring one, deleting others only when their Wilson
intervals are clearly separated -- see pi05/curator.py for why that matters.

    python scripts/curate_checkpoints.py \
        --run-dir <.../checkpoints/pi05_droid_finetune/pick_full_v3> \
        --episodes 128 --workers 4 --server-gpu 6 --sim-gpu 7

Deletes 48 GB directories, so it refuses to run without --allow-delete and always keeps
the newest checkpoint no matter what it scored.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.curator import Evaluated, describe, select_survivors  # noqa: E402

OPENPI_PYTHON = Path("/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/openpi/.venv/bin/python")


def complete_checkpoints(run_dir: Path) -> list[int]:
    """Step numbers of checkpoints that are fully written.

    A directory still being written has an `.orbax-checkpoint-tmp` suffix, and a
    finished one contains both params and train_state; anything else is skipped rather
    than evaluated half-formed.
    """
    steps = []
    for child in run_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        if (child / "params").exists() and (child / "train_state").exists():
            steps.append(int(child.name))
    return sorted(steps)


def load_ledger(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"evaluated": {}, "deleted": []}


def save_ledger(path: Path, ledger: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2))


def evaluate(checkpoint: Path, out: Path, args) -> dict | None:
    cmd = [
        str(OPENPI_PYTHON),
        str(ROOT / "scripts/eval_pi05.py"),
        "--checkpoint", str(checkpoint),
        "--episodes", str(args.episodes),
        "--workers", str(args.workers),
        "--out", str(out),
        "--server-gpu", args.server_gpu,
        "--sim-gpu", args.sim_gpu,
        "--egl-device", args.egl_device,
        "--port", str(args.port),
    ]
    print(f"  evaluating {checkpoint.name} on {args.episodes} episodes...", flush=True)
    started = time.time()
    subprocess.run(cmd, check=False)
    result_path = out / "result.json"
    if not result_path.exists():
        print(f"  no result.json produced; see {out}", flush=True)
        return None
    result = json.loads(result_path.read_text())
    print(f"  {checkpoint.name}: {result.get('successes')}/{result.get('episodes_run')} "
          f"in {time.time() - started:.0f}s", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True, help="dir holding <step>/ checkpoints")
    ap.add_argument("--eval-root", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--server-gpu", default="6")
    ap.add_argument("--sim-gpu", default="7")
    ap.add_argument("--egl-device", default="7")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--poll-seconds", type=int, default=300)
    ap.add_argument("--allow-delete", action="store_true",
                    help="actually remove checkpoints; without it, decisions are printed only")
    ap.add_argument("--once", action="store_true", help="single pass instead of watching")
    args = ap.parse_args()

    eval_root = args.eval_root or (args.run_dir.parent.parent / "curation" / args.run_dir.name)
    ledger_path = eval_root / "ledger.json"
    ledger = load_ledger(ledger_path)

    print(f"watching {args.run_dir}")
    print(f"ledger   {ledger_path}")
    print(f"deletion {'ENABLED' if args.allow_delete else 'disabled (dry run)'}\n", flush=True)

    while True:
        steps = complete_checkpoints(args.run_dir) if args.run_dir.exists() else []
        pending = [s for s in steps if str(s) not in ledger["evaluated"]]

        for step in pending:
            result = evaluate(args.run_dir / str(step), eval_root / f"step_{step}", args)
            if result and result.get("episodes_run"):
                ledger["evaluated"][str(step)] = {
                    "successes": result["successes"],
                    "episodes": result["episodes_run"],
                    "success_rate": result["success_rate"],
                    "wilson_95": result.get("wilson_95"),
                }
                save_ledger(ledger_path, ledger)

        evaluated = [
            Evaluated(step=int(k), successes=v["successes"], episodes=v["episodes"])
            for k, v in ledger["evaluated"].items()
        ]
        if evaluated and steps:
            survivors = select_survivors(evaluated, latest_step=max(steps))
            print("\n" + describe(evaluated, survivors), flush=True)

            for step in steps:
                if step in survivors:
                    continue
                target = args.run_dir / str(step)
                if not args.allow_delete:
                    print(f"  would delete {target} (dry run)", flush=True)
                    continue
                size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
                shutil.rmtree(target)
                ledger["deleted"].append({"step": step, "freed_gb": round(size / 1e9, 1)})
                save_ledger(ledger_path, ledger)
                print(f"  deleted {target} ({size / 1e9:.1f} GB)", flush=True)

        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
