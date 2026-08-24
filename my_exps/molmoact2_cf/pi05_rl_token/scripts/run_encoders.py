#!/usr/bin/env python
"""Phase 1: train the RL-token autoencoders on a collected corpus.

    scripts/run_encoders.py --corpus <dir> --out <dir>

Four checkpoints: one per scene, plus one on all scenes together. Training them
separately is what makes the later ablation possible -- whether a scene-specific encoder
beats a general one is an empirical question per task, not something to assume.

The encoder is the RL state. It is fit to the distribution of token sequences the frozen
policy actually produces, so a corpus collected under a different chunk size or
conversion would define a state space for a policy we do not run. This script refuses a
corpus whose manifest disagrees with the regime in `config.py`, because that mismatch is
invisible afterwards: the reconstruction loss converges either way.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.config import ROOT as DATA_ROOT  # noqa: E402
from pi05.config import SCENES, VENV_SIM, VLA_TOKEN_DIM  # noqa: E402

log = logging.getLogger("run_encoders")


def corpus_scenes(corpus: Path) -> list[str]:
    """Which scenes have shards, in config order."""
    return [name for name in SCENES if list((corpus / name).glob("*.npz"))]


def describe_corpus(corpus: Path, scene: str) -> str:
    shards = sorted((corpus / scene).glob("*.npz"))
    size = sum(p.stat().st_size for p in shards) / 1e9
    return f"{len(shards)} shards, {size:.1f} GB"


def check_manifest(corpus: Path, scene: str) -> None:
    """A corpus without provenance is one nobody can re-check later."""
    manifests = list((corpus / scene).glob("manifest_*.json"))
    if not manifests:
        log.warning("%s: no manifest; the collection regime cannot be verified", scene)
        return
    meta = json.loads(manifests[0].read_text())
    if meta.get("converts_delta_to_absolute"):
        raise SystemExit(
            f"{scene}: the corpus was collected from a server that converted deltas "
            "server-side, i.e. the pre-refactor regime. Recollect it."
        )
    log.info("%s: collected %s", scene, meta.get("collected_utc", "?"))


def launch(command: list[str], log_path: Path, env_extra: dict) -> subprocess.Popen:
    import os

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "RLT_VLA_TOKEN_DIM": str(VLA_TOKEN_DIM), **env_extra}
    return subprocess.Popen(
        command, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
        env=env, cwd=str(ROOT), start_new_session=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--corpus", type=Path, default=DATA_ROOT / "rlt_tokens_step_time")
    ap.add_argument("--out", type=Path, default=DATA_ROOT / "pi05_runs/token_ae_step_time")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--max-sequences", type=int, default=6000, dest="max_sequences")
    ap.add_argument("--first-gpu", type=int, default=0, dest="first_gpu")
    ap.add_argument("--wait", action="store_true", help="block until all four finish")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scenes = corpus_scenes(args.corpus)
    if not scenes:
        raise SystemExit(f"no shards under {args.corpus}; collect the corpus first")

    args.out.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        log.info("%s: %s", scene, describe_corpus(args.corpus, scene))
        check_manifest(args.corpus, scene)

    jobs: list[tuple[str, subprocess.Popen]] = []
    for offset, scene in enumerate(scenes):
        gpu = args.first_gpu + offset
        command = [
            str(VENV_SIM), "-m", "rlt.train_token_ae",
            "--token_replay", str(args.corpus / scene / "*.npz"),
            "--out", str(args.out / f"ae_{scene}.pt"),
            "--device", f"cuda:{gpu}",
            "--steps", str(args.steps),
            "--max_sequences", str(args.max_sequences),
        ]
        log.info("ae_%s on cuda:%d", scene, gpu)
        jobs.append((scene, launch(command, args.out / f"ae_{scene}.log", {})))

    gpu = args.first_gpu + len(scenes)
    command = [
        str(VENV_SIM), "-m", "rlt.train_token_ae",
        "--token_replay", str(args.corpus / "*" / "*.npz"),
        "--out", str(args.out / "ae_combined.pt"),
        "--device", f"cuda:{gpu}",
        "--steps", str(args.steps),
        "--max_sequences", str(args.max_sequences),
    ]
    log.info("ae_combined on cuda:%d", gpu)
    jobs.append(("combined", launch(command, args.out / "ae_combined.log", {})))

    log.info("%d autoencoder runs launched", len(jobs))
    if not args.wait:
        return

    started = time.time()
    for name, job in jobs:
        code = job.wait()
        log.info("ae_%s exited %d after %.0f min", name, code, (time.time() - started) / 60)


if __name__ == "__main__":
    main()
