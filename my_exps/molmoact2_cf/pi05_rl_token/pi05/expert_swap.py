"""Learner-side protocol for the parallel action-expert swap.

The student trainer is a separate process. The learner writes `command` and waits on
`status.json`; once a new generation is on disk it tells every frozen server to copy
those weights.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from .client import Pi05Client
from .expert_io import expert_safetensors_path

log = logging.getLogger(__name__)


def snapshot_checkpoint(src: str | Path, dest: str | Path) -> Path:
    """Copy model.safetensors (+ config/assets) so a later /reload can restore it."""
    src_dir = Path(src)
    dest_dir = Path(dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    weights = expert_safetensors_path(src_dir)
    shutil.copy2(weights, dest_dir / "model.safetensors")
    config = src_dir / "config.json"
    if config.exists():
        shutil.copy2(config, dest_dir / "config.json")
    assets = src_dir / "assets"
    if assets.exists():
        dest_assets = dest_dir / "assets"
        if dest_assets.exists():
            shutil.rmtree(dest_assets)
        shutil.copytree(assets, dest_assets)
    log.info("snapshot %s -> %s", src_dir, dest_dir)
    return dest_dir


def frozen_checkpoint(cfg) -> str:
    """Last accepted frozen expert, or the original pi0.5 checkpoint."""
    raw = str(getattr(cfg, "distill_frozen_dir", "") or "")
    if raw:
        frozen = Path(raw)
    elif getattr(cfg, "distill_student_dir", ""):
        frozen = Path(cfg.distill_student_dir).parent / "frozen_expert"
    else:
        frozen = None
    if frozen is not None and (frozen / "model.safetensors").exists():
        return str(frozen)
    return str(cfg.checkpoint)


def write_command(control_dir: str | Path, command: str) -> None:
    path = Path(control_dir)
    path.mkdir(parents=True, exist_ok=True)
    dest = path / "command"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(command)
    tmp.replace(dest)


def read_status(control_dir: str | Path) -> dict:
    path = Path(control_dir) / "status.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def request_student_save(
    control_dir: str | Path,
    *,
    previous_generation: int,
    timeout_sec: float = 1800.0,
    poll_sec: float = 1.0,
    is_alive=None,
    student_dir: str | Path | None = None,
    snapshot_dir: str | Path | None = None,
) -> dict:
    """Ask the student to write a checkpoint, then wait until generation advances.

    The student blocks on `command=save` until we flip it back to `run`. Snapshot the
    weights first so probes reload a frozen file while the student keeps training.
    """
    write_command(control_dir, "save")
    deadline = time.time() + timeout_sec
    last: dict = {}
    while time.time() < deadline:
        if is_alive is not None and not is_alive():
            raise RuntimeError("student trainer exited before saving a checkpoint")
        last = read_status(control_dir)
        if last.get("saved") and int(last.get("generation", 0)) > previous_generation:
            if student_dir is not None and snapshot_dir is not None:
                snapshot_checkpoint(student_dir, snapshot_dir)
            write_command(control_dir, "run")
            return last
        time.sleep(poll_sec)
    raise TimeoutError(
        f"student did not save within {timeout_sec:.0f}s "
        f"(last status={last}, waiting for generation>{previous_generation})"
    )


def reload_frozen_servers(ports: list[int], checkpoint: str) -> list[dict]:
    """Copy the student checkpoint into every frozen action expert."""
    replies = []
    for port in ports:
        client = Pi05Client(port=port)
        replies.append(client.reload(checkpoint))
        log.info("reloaded frozen expert on port %d from %s", port, checkpoint)
    return replies
