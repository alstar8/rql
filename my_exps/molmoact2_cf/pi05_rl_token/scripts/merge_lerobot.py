"""Join the LeRobot datasets built by parallel converter workers into one.

    python scripts/merge_lerobot.py --parts <dir>/part_* <existing_dataset> --out <dir>/merged

Reads the parts and writes a new dataset beside them; nothing is modified or removed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.merge import merge_datasets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parts", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=17, help="parts merged in parallel")
    ap.add_argument("--drop-leading-frames", type=int, default=1,
                    help="frames removed from the start of each episode (the recorded hold)")
    args = ap.parse_args()

    parts = sorted(args.parts)
    print(f"merging {len(parts)} parts into {args.out} with {args.workers} workers")
    stats = merge_datasets(parts, args.out, workers=args.workers,
                           drop_leading_frames=args.drop_leading_frames)
    print(
        f"\nmerged {stats['parts']} parts: {stats['episodes']} episodes, "
        f"{stats['frames']} frames, {stats['tasks']} distinct instructions"
    )


if __name__ == "__main__":
    main()
