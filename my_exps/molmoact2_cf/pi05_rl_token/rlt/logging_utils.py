"""Metrics go to two places: a JSONL file and TensorBoard.

The JSONL is the source of truth -- greppable, diffable, and readable without a
server. TensorBoard is for looking at several runs at once, which is the whole
point of the sweep. `scripts/jsonl_to_tb.py` can rebuild the TB side from JSONL,
so a run that was logged before TB existed is not lost.
"""

from __future__ import annotations

import json
from pathlib import Path


class RunLogger:
    """Writes one JSONL line and one set of TB scalars per call.

    TensorBoard is optional: if it is missing, JSONL still works and the run
    prints a warning rather than dying.
    """

    def __init__(
        self, jsonl_path: Path, tb_dir: Path | None, run_name: str = "", append: bool = False
    ) -> None:
        self.jsonl_path = jsonl_path
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = jsonl_path.open("a" if append else "w")
        self.writer = None

        if tb_dir is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            print("tensorboard not installed -- JSONL only")
            return
        tb_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"tensorboard: {tb_dir}" + (f"  (run {run_name})" if run_name else ""))

    def log(self, step: int, metrics: dict) -> None:
        self.fh.write(json.dumps({"step": step, **metrics}) + "\n")
        self.fh.flush()
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def add_text(self, tag: str, text: str) -> None:
        if self.writer is not None:
            self.writer.add_text(tag, f"```\n{text}\n```", 0)

    def close(self) -> None:
        self.fh.close()
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def default_tb_dir(out_path: Path, tb_dir: str) -> Path:
    """`runs/ae_sweep/reference.pt` -> `runs/ae_sweep/tb/reference`.

    Every run in a sweep lands under one `tb/` root, so a single
    `tensorboard --logdir runs/ae_sweep/tb` overlays all of them.
    """
    return Path(tb_dir) if tb_dir else out_path.parent / "tb" / out_path.stem
