"""Environment variables MolmoSpaces reads at import time.

Kept free of MolmoSpaces imports so it can run before them.
"""

from __future__ import annotations

import os
from pathlib import Path


def configure_assets(assets_dir: str, cache_dir: str = "") -> None:
    """Point MolmoSpaces at the persistent asset copy.

    The stock symlinks under ~/.cache/molmospaces/assets point into the
    container's ephemeral overlay and dangle after a job restart; the copy under
    B1K/mlspaces is on real storage. `cache_dir` defaults to its sibling
    `cache/`, which is where the benchmark symlinks resolve to.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    if assets_dir:
        os.environ["MLSPACES_ASSETS_DIR"] = assets_dir
        os.environ["MLSPACES_CACHE_DIR"] = cache_dir or str(Path(assets_dir).parent / "cache")
