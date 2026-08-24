"""Convert the demonstration corpus with several workers at once.

One process is bound by video decoding and takes about sixteen hours for the whole
corpus while 192 cores sit idle. Each worker builds its own LeRobot dataset under
``<out>/part_NN``; scripts/merge_lerobot.py joins them afterwards.

    python scripts/convert_parallel.py --out <dir> --workers 16

Houses already converted into an existing dataset can be excluded with --skip-manifest,
so work done by an earlier run is not repeated.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = Path(
    "/workspace-SR008.nfs2/users/staroverov/pi05_molmospaces/mbdata_pick/FrankaPickOmniCamConfig/part0/train"
)
OPENPI_PYTHON = Path(
    "/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/openpi/.venv/bin/python"
)


def house_sort_key(name: str) -> int:
    return int(name.split("_")[1])


def covered_houses(manifest: Path) -> set[str]:
    if not manifest.exists():
        return set()
    episodes = json.loads(manifest.read_text())["episodes"]
    return {episode.split("/")[0] for episode in episodes}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, required=True, help="directory to hold part_NN datasets")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--skip-manifest",
        type=Path,
        default=None,
        help="converted_manifest.json whose houses should be left out",
    )
    args = ap.parse_args()

    houses = sorted((p.name for p in args.source.glob("house_*") if p.is_dir()), key=house_sort_key)
    skip = covered_houses(args.skip_manifest) if args.skip_manifest else set()
    remaining = [h for h in houses if h not in skip]
    print(f"{len(houses)} houses, {len(skip)} already covered, {len(remaining)} to convert")

    if not remaining:
        print("nothing to do")
        return

    # Round-robin rather than contiguous blocks: houses vary in episode count and
    # trajectory length, so interleaving keeps the workers finishing at similar times.
    groups: list[list[str]] = [[] for _ in range(args.workers)]
    for i, house in enumerate(remaining):
        groups[i % args.workers].append(house)

    args.out.mkdir(parents=True, exist_ok=True)
    procs = []
    for index, group in enumerate(groups):
        if not group:
            continue
        part = args.out / f"part_{index:02d}"
        log = args.out / f"part_{index:02d}.log"
        cmd = [
            str(OPENPI_PYTHON),
            str(ROOT / "scripts" / "convert_demos_to_lerobot.py"),
            "--source", str(args.source),
            "--out", str(part),
            "--repo-id", f"molmospaces/pick_part_{index:02d}",
            "--houses", *group,
        ]
        # A CPU job: hiding the GPUs stops any imported framework from taking a card.
        env = {
            "PATH": "/usr/bin:/bin",
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "HF_HUB_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
        handle = log.open("w")
        procs.append((index, subprocess.Popen(cmd, stdout=handle, stderr=handle, env=env), handle))
        print(f"  worker {index:02d}: {len(group)} houses -> {part}")

    print(f"\n{len(procs)} workers running; logs in {args.out}/part_NN.log")
    start = time.time()
    failed = []
    for index, proc, handle in procs:
        code = proc.wait()
        handle.close()
        status = "ok" if code == 0 else f"FAILED ({code})"
        print(f"  worker {index:02d} {status} at {time.time() - start:.0f}s")
        if code != 0:
            failed.append(index)

    print(f"\nall workers done in {time.time() - start:.0f}s")
    if failed:
        print(f"WORKERS FAILED: {failed} -- inspect their logs before merging")
        sys.exit(1)


if __name__ == "__main__":
    main()
