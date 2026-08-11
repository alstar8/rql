"""Atomically snapshot run PIDs, latest metrics, checkpoints, host, and GPUs."""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Sequence

from v13_harness import atomic_write_json, latest_jsonl_row, redact_command, sha256_file


_HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = _HERE / "runs" / "rlt_cf_v12_shortlist"
CHECKPOINT_NAMES = ("rlt_cf_latest.pt", "molmo_ae_lora_latest.pt")


def _read_process_file(path: Path, separator: bytes) -> list[str]:
    try:
        payload = path.read_bytes()
    except OSError:
        return []
    return [
        item.decode("utf-8", errors="replace")
        for item in payload.split(separator)
        if item
    ]


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def read_pid_record(pidfile: Path, run_dir: Path) -> dict[str, Any]:
    """Read the PID from the manifest's first field and inspect it safely."""

    record: dict[str, Any] = {
        "pidfile": str(pidfile),
        "relative_pidfile": str(pidfile.relative_to(run_dir)),
        "valid_pid": False,
        "alive": False,
        "belongs_to_run": False,
    }
    try:
        fields = pidfile.read_text(encoding="utf-8", errors="replace").split()
    except OSError as error:
        record["read_error"] = str(error)
        return record
    record["manifest_fields"] = fields
    if not fields:
        record["parse_error"] = "empty PID manifest"
        return record
    try:
        pid = int(fields[0])
    except ValueError:
        record["parse_error"] = f"first field is not an integer: {fields[0]!r}"
        return record
    if pid <= 1:
        record["parse_error"] = f"unsafe PID: {pid}"
        return record

    record["pid"] = pid
    record["valid_pid"] = True
    record["alive"] = _pid_alive(pid)
    if not record["alive"]:
        return record

    command = _read_process_file(Path(f"/proc/{pid}/cmdline"), b"\0")
    environment_rows = _read_process_file(Path(f"/proc/{pid}/environ"), b"\0")
    environment = {}
    for row in environment_rows:
        key, separator, value = row.partition("=")
        if separator:
            environment[key] = value
    marker = environment.get("RLT_CF_V4_RUN_DIR", "")
    try:
        marker_path = str(Path(marker).resolve()) if marker else ""
    except OSError:
        marker_path = marker
    record.update(
        {
            "command": redact_command(command),
            "belongs_to_run": marker_path == str(run_dir.resolve()),
            "run_marker": marker_path,
            "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
            "mujoco_egl_device_id": environment.get("MUJOCO_EGL_DEVICE_ID"),
        }
    )
    return record


def _jsonl_stats(path: Path) -> dict[str, Any]:
    valid_rows = 0
    malformed_rows = 0
    latest: dict[str, Any] | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    malformed_rows += 1
                    continue
                if isinstance(candidate, dict):
                    valid_rows += 1
                    latest = candidate
                else:
                    malformed_rows += 1
    except OSError as error:
        return {"read_error": str(error), "latest": None}
    return {
        "valid_rows": valid_rows,
        "malformed_rows": malformed_rows,
        "latest": latest,
    }


def collect_metrics(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(run_dir.rglob("metrics.jsonl")):
        try:
            relative = path.relative_to(run_dir)
        except ValueError:
            relative = path
        stat = path.stat()
        records.append(
            {
                "path": str(path),
                "relative_path": str(relative),
                "size_bytes": stat.st_size,
                "mtime": dt.datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=dt.timezone.utc,
                ).isoformat(),
                **_jsonl_stats(path),
            }
        )
    return records


def collect_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for name in CHECKPOINT_NAMES:
        paths.update(run_dir.rglob(name))
    records = []
    for path in sorted(paths):
        record: dict[str, Any] = {
            "path": str(path),
            "relative_path": str(path.relative_to(run_dir)),
        }
        try:
            stat = path.stat()
            record.update(
                {
                    "size_bytes": stat.st_size,
                    "mtime": dt.datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=dt.timezone.utc,
                    ).isoformat(),
                    "sha256": sha256_file(path),
                }
            )
        except OSError as error:
            record["hash_error"] = str(error)
        records.append(record)
    return records


def _git_state(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
        result.update({"sha": revision, "dirty": bool(status.strip())})
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        result["unavailable_reason"] = str(error)
    return result


def _gpu_info() -> dict[str, Any]:
    fields = (
        "index,uuid,name,memory.total,memory.used,utilization.gpu,"
        "temperature.gpu,driver_version"
    )
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"available": False, "unavailable_reason": str(error), "gpus": []}
    names = fields.split(",")
    rows = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        rows.append(dict(zip(names, values)))
    return {"available": True, "gpus": rows}


def build_status_snapshot(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    pid_dir = run_dir / "pids"
    pidfiles = sorted(pid_dir.glob("*.pid")) if pid_dir.is_dir() else []
    pid_records = [
        read_pid_record(pidfile, run_dir)
        for pidfile in pidfiles
    ]
    metrics = collect_metrics(run_dir)
    repositories = {
        "rql": _HERE.parents[1],
        "molmoact2": _HERE.parents[2] / "molmoact2",
        "molmospaces": _HERE.parents[2] / "molmospaces",
        "b1k_airi": _HERE.parents[3],
    }
    return {
        "schema_version": "run-status-1",
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "host": {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "user": getpass.getuser(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "gpu": _gpu_info(),
        "repositories": {
            name: _git_state(path)
            for name, path in repositories.items()
        },
        "pids": pid_records,
        "pid_summary": {
            "manifests": len(pid_records),
            "valid": sum(bool(record.get("valid_pid")) for record in pid_records),
            "alive": sum(bool(record.get("alive")) for record in pid_records),
            "alive_and_owned": sum(
                bool(record.get("alive")) and bool(record.get("belongs_to_run"))
                for record in pid_records
            ),
        },
        "metrics": metrics,
        "metrics_summary": {
            "files": len(metrics),
            "latest_valid_episodes_total": sum(
                int((record.get("latest") or {}).get("valid_episodes", 0) or 0)
                for record in metrics
            ),
            "latest_env_steps_total": sum(
                int((record.get("latest") or {}).get("env_steps", 0) or 0)
                for record in metrics
            ),
        },
        "checkpoints": collect_checkpoints(run_dir),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to RUN_DIR/status_snapshots/status_<UTC>.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    output = args.output
    if output is None:
        timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = run_dir / "status_snapshots" / f"status_{timestamp}.json"
    payload = build_status_snapshot(run_dir)
    atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "alive_and_owned": payload["pid_summary"]["alive_and_owned"],
                "metrics_files": payload["metrics_summary"]["files"],
                "checkpoints": len(payload["checkpoints"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
