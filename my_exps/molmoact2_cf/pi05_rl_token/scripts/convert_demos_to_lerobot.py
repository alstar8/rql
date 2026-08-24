"""Convert MolmoSpaces Pick demonstrations into a LeRobot dataset openpi can train on.

    # one house, for the wiring test
    python scripts/convert_demos_to_lerobot.py --houses house_0 --out /path/pick_house0

    # a whole shard set, appending to what is already converted
    python scripts/convert_demos_to_lerobot.py --out /path/pick_all

Runs on CPU. Re-running is safe: already-converted trajectories are skipped via the
manifest in the output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.convert import convert  # noqa: E402

DEFAULT_SOURCE = Path(
    "/workspace-SR008.nfs2/users/staroverov/pi05_molmospaces/mbdata_pick/FrankaPickOmniCamConfig/part0/train"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, required=True, help="LeRobot dataset directory")
    parser.add_argument("--repo-id", default="molmospaces/pick_pi05")
    parser.add_argument(
        "--houses", nargs="*", default=None, help="restrict to these house_N directories"
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="also convert trajectories whose final success flag is false",
    )
    parser.add_argument(
        "--keep-first-step",
        action="store_true",
        help="keep each trajectory's first transition (a ~1.2 rad jump from the reset pose)",
    )
    parser.add_argument(
        "--no-exterior2",
        action="store_true",
        help="omit exterior_image_2_left (only valid with a custom training config)",
    )
    args = parser.parse_args()

    stats = convert(
        args.source,
        args.out,
        args.repo_id,
        houses=args.houses,
        max_episodes=args.max_episodes,
        duplicate_exterior2=not args.no_exterior2,
        require_success=not args.keep_failures,
        drop_first_steps=0 if args.keep_first_step else 1,
    )
    print(
        f"\nwrote {stats['written']} episodes ({stats['frames']} frames), "
        f"skipped {stats['skipped']} already converted; "
        f"{stats['total']} episodes in {args.out}"
    )


if __name__ == "__main__":
    main()
