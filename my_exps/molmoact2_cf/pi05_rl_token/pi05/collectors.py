"""Four rollout workers on one GPU, without multiprocessing.Queue.

The previous spawn+Queue path died in SemLock._rebuild: MolmoSpaces also
spawns a Worker-0, and pickling a mp.Queue into that nested spawn unlinks the
semaphore. These workers are ordinary subprocesses (fresh interpreters). They
talk to the parent through files:

    counter   fcntl-locked integer, so two workers never run the same episode
    inbox     one pickle per finished episode (rows + success + steps)
    weights   agent.pt the parent rewrites after each actor-critic burst
    egl_slots three lock files; the fourth worker blocks until a slot frees

The frozen pi0.5 server is shared; Pi05Client flocks /act.
"""

from __future__ import annotations

import fcntl
import logging
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("pi05.collectors")

WEIGHTS = "actor_live.pt"
COUNTER = "episode.counter"
INBOX = "inbox"
EGL = "egl_slots"


class FileCounter:
    """Hands out 0, 1, 2, ... under an exclusive flock."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("0")

    def claim(self) -> int:
        with open(self.path, "r+") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            raw = handle.read().strip()
            index = int(raw or "0")
            handle.seek(0)
            handle.truncate()
            handle.write(str(index + 1))
            handle.flush()
            os.fsync(handle.fileno())
            return index


class EglSlots:
    """At most `n` concurrent MuJoCo EGL contexts on this GPU."""

    def __init__(self, directory: str | Path, n: int) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._files = []
        for i in range(n):
            path = self.directory / f"slot_{i}"
            path.touch(exist_ok=True)
            self._files.append(open(path, "a+"))

    def acquire(self):
        while True:
            for handle in self._files:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return handle
                except BlockingIOError:
                    continue
            time.sleep(0.05)

    def release(self, handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def close(self) -> None:
        for handle in self._files:
            handle.close()


def load_live_weights(agent, path: Path, seen_version: int, attempts: int = 8) -> int:
    """Reload actor_live.pt, retrying if the learner is still writing it."""
    if not path.exists():
        return seen_version
    mtime = path.stat().st_mtime_ns
    if mtime == seen_version:
        return seen_version
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            agent.load(str(path))
            return path.stat().st_mtime_ns
        except (RuntimeError, OSError, EOFError) as exc:
            last_err = exc
            time.sleep(0.05 * (attempt + 1))
    assert last_err is not None
    raise last_err


def write_inbox(directory: Path, seq: int, rank: int, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f"{seq:08d}_r{rank}.pkl"
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(tmp, dest)


def read_inbox(directory: Path) -> list[dict]:
    if not directory.exists():
        return []
    items = []
    for path in sorted(directory.glob("*.pkl")):
        try:
            items.append(pickle.loads(path.read_bytes()))
        except (pickle.UnpicklingError, EOFError):
            continue
        path.unlink(missing_ok=True)
    return items


def collector_env(out_dir: Path) -> dict:
    """Env for one rollout worker. Actor is CPU; CUDA stays visible so NVIDIA EGL works."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["RLT_VLA_TOKEN_DIM"] = env.get("RLT_VLA_TOKEN_DIM", "2048")
    env["MUJOCO_GL"] = "egl"
    env["PI05_VLA_LOCK_DIR"] = str(out_dir / "vla_lock")
    env["RLT_COLLECTOR"] = "1"
    return env


def start_workers(cfg, out_dir: Path, n: int) -> list[subprocess.Popen]:
    """Fresh interpreters, not spawn children of this CUDA process."""
    cfg_path = out_dir / "config.json"
    worker = Path(__file__).resolve().parent.parent / "scripts" / "run_collector.py"
    python = sys.executable
    procs = []
    env = collector_env(out_dir)
    for rank in range(n):
        proc = subprocess.Popen(
            [python, str(worker), "--rank", str(rank), "--config", str(cfg_path), "--out", str(out_dir)],
            env=env,
            stdout=open(out_dir / f"collector_{rank}.log", "a"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        procs.append(proc)
        log.info("collector %d pid=%d", rank, proc.pid)
    return procs


def stop_workers(procs: list[subprocess.Popen], grace: float = 30.0) -> None:
    deadline = time.time() + grace
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    while time.time() < deadline and any(p.poll() is None for p in procs):
        time.sleep(0.2)
    for proc in procs:
        if proc.poll() is None:
            proc.kill()


# Names used by collector scripts that still import the previous spellings.
FileCounter = FileCounter
EglSlots = EglSlots
write_inbox = write_inbox
read_inbox = read_inbox
collector_env = collector_env
start_workers = start_workers
stop_workers = stop_workers
