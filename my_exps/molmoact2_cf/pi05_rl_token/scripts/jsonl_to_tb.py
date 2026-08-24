"""Rebuild TensorBoard event files from `.metrics.jsonl`.

Every trainer writes JSONL as the source of truth, so a run started before TB
logging existed -- or one whose TB directory was lost -- can be reconstructed
without retraining. Safe to re-run: it overwrites the TB dir for each run.

Also works on a run still in progress; re-run it to pick up new lines.

    python scripts/jsonl_to_tb.py --runs_dir <.../runs/ae_sweep>
    python scripts/jsonl_to_tb.py --metrics <.../agent.metrics.jsonl>

Then:  tensorboard --logdir <.../runs/ae_sweep/tb> --port 6006
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running as `python scripts/x.py` puts scripts/ on sys.path, not the package root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json


def convert(metrics_path: Path, tb_root: Path) -> int:
    from torch.utils.tensorboard import SummaryWriter

    # foo.metrics.jsonl -> run name "foo"
    name = metrics_path.name
    for suffix in (".metrics.jsonl", ".jsonl"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    rows = []
    for line in metrics_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A run still writing can leave a torn final line; ignore it.
            continue

    if not rows:
        print(f"  {metrics_path.name}: no rows yet, skipped")
        return 0

    out_dir = tb_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    for row in rows:
        step = int(row.get("step", 0))
        for key, value in row.items():
            if key == "step":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                writer.add_scalar(key, value, step)
    writer.close()
    print(f"  {name}: {len(rows)} rows -> {out_dir}")
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="", help="dir holding *.metrics.jsonl")
    ap.add_argument("--metrics", default="", help="a single .metrics.jsonl")
    ap.add_argument("--tb_root", default="", help="default: <runs_dir>/tb")
    args = ap.parse_args()

    if args.metrics:
        path = Path(args.metrics)
        tb_root = Path(args.tb_root) if args.tb_root else path.parent / "tb"
        convert(path, tb_root)
    elif args.runs_dir:
        runs = Path(args.runs_dir)
        tb_root = Path(args.tb_root) if args.tb_root else runs / "tb"
        found = sorted(runs.glob("*.metrics.jsonl"))
        if not found:
            raise SystemExit(f"no *.metrics.jsonl under {runs}")
        print(f"{len(found)} run(s) -> {tb_root}")
        for path in found:
            convert(path, tb_root)
    else:
        raise SystemExit("give --runs_dir or --metrics")

    print("\ntensorboard --logdir <tb_root> --port 6006 --bind_all")


if __name__ == "__main__":
    main()
