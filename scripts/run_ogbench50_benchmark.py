#!/usr/bin/env python3
"""Launch OGBench 50-task RQL benchmark jobs across a GPU pool."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

SCRIPT_DIR = Path(__file__).resolve().parent
RQL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from ogbench50_config import (  # noqa: E402
    OGBench50Task,
    build_main_args,
    expand_tasks,
    exp_root_for_task,
    is_run_complete,
)


def setup_python_env(python: str) -> None:
    subprocess.run([python, "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True)
    subprocess.run([python, "-m", "pip", "install", "-q", "-r", str(RQL_DIR / "requirements.txt")], check=True)
    subprocess.run(
        [python, "-m", "pip", "install", "-q", "einops", "opencv-python-headless"],
        check=True,
    )


def run_job(
    python: str,
    task: OGBench50Task,
    seed: int,
    gpu_id: str,
    save_dir: str,
    ogbench_data_dir: str,
    log_dir: Path,
    offline_steps: int,
    batch_size: int,
) -> tuple[str, int, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    args = build_main_args(
        task,
        seed=seed,
        save_dir=save_dir,
        ogbench_data_dir=ogbench_data_dir,
        offline_steps=offline_steps,
        batch_size=batch_size,
    )
    log_path = log_dir / f"{task.log_stem}_seed{seed}_gpu{gpu_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            [python, *args],
            cwd=RQL_DIR,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return f"{task.task_id}/seed{seed}", proc.returncode, str(log_path)


def pending_jobs(
    tasks: list[OGBench50Task],
    seeds: list[int],
    save_dir: str,
    skip_complete: bool,
) -> list[tuple[OGBench50Task, int]]:
    jobs: list[tuple[OGBench50Task, int]] = []
    for task in tasks:
        for seed in seeds:
            if skip_complete and is_run_complete(exp_root_for_task(save_dir, task), seed):
                continue
            jobs.append((task, seed))
    return jobs


def gpu_worker(
    gpu_id: str,
    job_queue: Queue,
    python: str,
    save_dir: str,
    ogbench_data_dir: str,
    log_dir: Path,
    offline_steps: int,
    batch_size: int,
    failed: list[str],
    lock: threading.Lock,
) -> None:
    while True:
        try:
            task, seed = job_queue.get_nowait()
        except Empty:
            return

        job_id = f"{task.task_id}/seed{seed}"
        print(f"[start] {job_id} -> GPU {gpu_id}")
        job_id, code, log_path = run_job(
            python,
            task,
            seed,
            gpu_id,
            save_dir,
            ogbench_data_dir,
            log_dir,
            offline_steps,
            batch_size,
        )
        status = "OK" if code == 0 else f"FAIL({code})"
        print(f"[{status}] {job_id} gpu={gpu_id} log={log_path}")
        if code != 0:
            with lock:
                failed.append(job_id)
        job_queue.task_done()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--gpu-ids", default=None, help="Comma-separated GPU ids.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--save-dir", default=str(RQL_DIR / "exp"))
    parser.add_argument(
        "--ogbench-data-dir",
        default=os.environ.get("OGBENCH_DATA_DIR", ""),
    )
    parser.add_argument("--offline-steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--skip-complete", action="store_true", default=True)
    parser.add_argument("--no-skip-complete", dest="skip_complete", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default="")
    parser.add_argument("--log-dir", default=str(RQL_DIR / "exp" / "ogbench50_logs"))
    args = parser.parse_args()

    if not args.ogbench_data_dir:
        parser.error("--ogbench-data-dir or OGBENCH_DATA_DIR is required")

    gpu_ids = args.gpu_ids.split(",") if args.gpu_ids else [str(i) for i in range(args.num_gpus)]
    python = args.python or os.environ.get(
        "RQL_PYTHON",
        os.path.join(
            os.environ.get("RQL_ENV", "/workspace-SR008.nfs2/users/staroverov/envs/rql_baseline"),
            "bin",
            "python",
        ),
    )

    tasks = expand_tasks()
    if args.domains:
        tasks = [t for t in tasks if t.domain_id in args.domains]
    if args.tasks:
        tasks = [t for t in tasks if t.task_id in args.tasks]

    jobs = pending_jobs(tasks, args.seeds, args.save_dir, args.skip_complete)
    print(f"Pending jobs: {len(jobs)} (tasks={len(tasks)}, seeds={args.seeds}, gpus={gpu_ids})")

    if args.dry_run:
        for task, seed in jobs[:20]:
            print(f"  {task.env_name} seed={seed}")
        if len(jobs) > 20:
            print(f"  ... and {len(jobs) - 20} more")
        return

    if not Path(python).exists():
        parser.error(f"Python not found: {python}")

    setup_python_env(python)

    job_queue: Queue = Queue()
    for job in jobs:
        job_queue.put(job)

    failed: list[str] = []
    lock = threading.Lock()
    log_dir = Path(args.log_dir)
    threads = [
        threading.Thread(
            target=gpu_worker,
            args=(
                gpu_id,
                job_queue,
                python,
                args.save_dir,
                args.ogbench_data_dir,
                log_dir,
                args.offline_steps,
                args.batch_size,
                failed,
                lock,
            ),
            daemon=True,
        )
        for gpu_id in gpu_ids
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if failed:
        print(f"{len(failed)} jobs failed.", file=sys.stderr)
        sys.exit(1)
    print("All benchmark jobs finished.")


if __name__ == "__main__":
    main()
