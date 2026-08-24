"""Metric logging for openpi training: readable progress lines plus JSONL for TensorBoard.

openpi logs only to wandb, which is not configured here. Rather than edit openpi or
install anything into its venv, ``install_jsonl_logging`` replaces ``wandb.log`` for the
training process, so every metric openpi already reports is written out instead.

JSONL is the source of truth, matching the rest of this project: scripts/jsonl_to_tb.py
renders TensorBoard from it and is safe to re-run mid-run, so the board refreshes while
training continues. Every line is flushed immediately -- a run that dies at hour six must
not lose its history to a buffer.

The printed line answers the question someone actually has when they check on a long run:
is this progressing, is the loss moving, and when will it finish.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def install_jsonl_logging(
    path: Path,
    *,
    total_steps: int | None = None,
    batch_size: int | None = None,
    accumulate: int = 1,
) -> None:
    """Point ``wandb.log`` at ``path`` for this process and print readable progress."""
    import wandb

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", buffering=1)  # line buffered
    state = {"started": time.time(), "last_step": 0, "last_time": time.time()}

    def log(data: dict, step: int | None = None, **kwargs) -> None:
        now = time.time()
        step = int(step) if step is not None else 0
        elapsed = now - state["started"]

        row = {"step": step, "elapsed_s": round(elapsed, 1)}
        for key, value in (data or {}).items():
            # openpi hands back numpy/JAX scalars, and np.float32 is NOT a Python float,
            # so an isinstance check against (int, float) silently drops every metric and
            # leaves TensorBoard empty. Coerce instead, and skip only what will not
            # convert -- images and other rich media, which are wandb-only.
            if isinstance(value, bool):
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            row[key] = scalar

        # Throughput over the interval between log calls, not since the start, so a slow
        # patch shows up instead of being averaged away by a fast beginning.
        step_delta = step - state["last_step"]
        time_delta = now - state["last_time"]
        if step_delta > 0 and time_delta > 0:
            steps_per_s = step_delta / time_delta
            row["steps_per_s"] = round(steps_per_s, 3)
            if batch_size:
                row["samples_per_s"] = round(steps_per_s * batch_size, 1)
        state["last_step"], state["last_time"] = step, now

        handle.write(json.dumps(row) + "\n")
        handle.flush()

        loss = row.get("loss")
        parts = [f"step {step:>7,}"]
        if total_steps:
            parts.append(f"/{total_steps:,} ({100 * step / total_steps:5.1f}%)")
        if accumulate > 1:
            parts.append(f"| update {step // accumulate:>6,}")
        if loss is not None:
            parts.append(f"| loss {loss:8.4f}")
        if "learning_rate" in row:
            parts.append(f"| lr {row['learning_rate']:.2e}")
        if "grad_norm" in row:
            parts.append(f"| grad {row['grad_norm']:7.3f}")
        if "samples_per_s" in row:
            parts.append(f"| {row['samples_per_s']:6.1f} samp/s")
        parts.append(f"| {_format_duration(elapsed)} elapsed")
        if total_steps and step > 0:
            # From the recent rate, not the cumulative average: dataset init and XLA
            # compilation take about ten minutes before the first step, which would
            # otherwise inflate the estimate for hours.
            rate = row.get("steps_per_s") or (step / elapsed)
            if rate > 0:
                parts.append(f"| ETA {_format_duration((total_steps - step) / rate)}")
        print(" ".join(parts), flush=True)

    def noop(*args, **kwargs):
        return None

    wandb.log = log
    # openpi also calls init and finish; neither should reach the network.
    wandb.init = noop
    wandb.finish = noop
