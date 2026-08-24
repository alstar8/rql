"""Remove half-extracted asset directories that block the MolmoSpaces downloader.

The resource manager refuses to start when a `<type>/<source>/<version>` directory
exists on disk but is absent from the cache manifest -- the signature of an
extraction that died partway. It names one offender at a time, so fixing them by
hand means one crash per remnant.

This lists every remnant at once. It is read-only unless you pass --yes, and it
only ever touches directories that are (a) under the cache root, (b) shaped like
`<type>/<source>/<version>`, and (c) missing from the manifest.

    python scripts/clean_asset_remnants.py --cache $MLSPACES_CACHE_DIR
    python scripts/clean_asset_remnants.py --cache $MLSPACES_CACHE_DIR --yes
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MANIFEST = "mjthor_data_type_to_source_to_versions.json"


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def find_remnants(cache: Path) -> list[tuple[Path, int, int]]:
    """-> [(path, n_files, bytes)] for version dirs missing from the manifest."""
    manifest_path = cache / MANIFEST
    if not manifest_path.exists():
        raise SystemExit(f"no manifest at {manifest_path} -- is {cache} really the cache dir?")
    manifest = json.loads(manifest_path.read_text())

    remnants = []
    for type_dir in sorted(p for p in cache.iterdir() if p.is_dir()):
        known_sources = manifest.get(type_dir.name, {})
        for source_dir in sorted(p for p in type_dir.iterdir() if p.is_dir()):
            known_versions = known_sources.get(source_dir.name, [])
            if isinstance(known_versions, str):
                known_versions = [known_versions]
            for version_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
                if version_dir.name not in known_versions:
                    n_files = sum(1 for _ in version_dir.rglob("*"))
                    remnants.append((version_dir, n_files, dir_size(version_dir)))
    return remnants


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="MLSPACES_CACHE_DIR")
    ap.add_argument("--yes", action="store_true", help="actually delete (default: report only)")
    args = ap.parse_args()

    cache = Path(args.cache)
    remnants = find_remnants(cache)

    if not remnants:
        print(f"no remnants under {cache} -- the manifest matches what is on disk")
        return

    print(f"{len(remnants)} remnant director{'y' if len(remnants) == 1 else 'ies'} under {cache}:")
    for path, n_files, size in remnants:
        print(f"  {path.relative_to(cache)}  {n_files} files  {size / 2**20:.1f} MiB")

    if not args.yes:
        print("\nreport only -- re-run with --yes to delete them, then restart the download")
        return

    for path, _, _ in remnants:
        shutil.rmtree(path)
        print(f"removed {path.relative_to(cache)}")
    print(f"\n{len(remnants)} removed; restart the asset download now")


if __name__ == "__main__":
    main()
