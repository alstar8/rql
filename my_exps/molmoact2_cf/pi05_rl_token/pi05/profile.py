"""Rank candidate scenes by how often the frozen VLA solves them.

A scene is one val-benchmark episode -- one house, one object -- that RL will later be
pointed at with a small positional jitter for variety. Which episodes are worth that
depends on where the frozen policy already sits:

    ~0%       the policy never earns a reward. Useful only if the failure is the grasp
              itself, not reaching for the wrong object -- video tells them apart.
    20-30%    RL has both positive and negative examples to learn from.
    40-60%+   comfortable headroom, and a clear ceiling to beat.

At 100% there is nothing to improve, and at 0% with a *wrong-object* failure RL cannot
fix the problem either, because the correction acts on the chunk, not on what the model
decided to look at.

Two stages, because measuring every candidate properly is wasteful:

    coarse    every candidate a few times, to rank them          (this module)
    proper    the survivors, many jittered repeats, per scene    (scripts/run_eval.py)

Coarse rates from a handful of rollouts are noisy by construction -- 1/4 and 2/4 are not
distinguishable -- so they are only ever used to bucket candidates, never quoted.

Runs in the regime `EvalConfig` currently specifies, which is the point: a ranking taken
under a different chunk size or conversion would select scenes for a policy we do not run.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path

from .config import EvalConfig
from .eval import stop_server, wilson

log = logging.getLogger("pi05.profile")


def parse_episode_spec(spec: str) -> list[int]:
    """"0-127" or "3,5,9" or a mix. Returns a sorted, de-duplicated list."""
    out: set[int] = set()
    for piece in str(spec).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, _, hi = piece.partition("-")
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(piece))
    return sorted(out)


def shard_of(items: list[int], shard: int, num_shards: int) -> list[int]:
    """Round-robin, so every shard gets a similar mix of easy and slow episodes.

    Contiguous blocks would be worse: episode difficulty is correlated with how long an
    episode runs (a failure runs the full horizon), so one shard could take twice as long
    as another and leave seven GPUs waiting on it.
    """
    return [item for index, item in enumerate(items) if index % num_shards == shard]


def profile(cfg: EvalConfig, episodes: list[int], repeats: int, out_path: Path) -> list[dict]:
    """Run each episode `repeats` times and record how often it succeeded.

    Results are written after every episode, not at the end: a shard that dies partway
    through still contributes what it measured.
    """
    from molmo_spaces.evaluation.eval_main import run_evaluation

    from .policy import Pi05EvalConfig

    scratch = cfg.run_dir() / "scratch"
    rows: list[dict] = []
    started = time.time()

    for position, episode in enumerate(episodes):
        successes = trials = 0
        for attempt in range(repeats):
            episode_dir = scratch / f"ep{episode:04d}_{attempt}"
            shutil.rmtree(episode_dir, ignore_errors=True)
            try:
                results = run_evaluation(
                    eval_config_cls=Pi05EvalConfig,
                    benchmark_dir=cfg.benchmark_dir(),
                    task_horizon_steps=cfg.horizon,
                    num_workers=1,
                    use_wandb=False,
                    episode_idx=episode,
                    output_dir=episode_dir,
                )
                if results.total_count > 0:
                    trials += 1
                    successes += int(results.success_count > 0)
            except Exception as error:  # noqa: BLE001
                log.warning("episode %d attempt %d failed: %s", episode, attempt, error)
            finally:
                # Every attempt writes videos and an HDF5 trajectory. Kept, a full sweep
                # would be tens of GB of footage nobody looks at; the chosen scenes get
                # their videos recorded deliberately later.
                shutil.rmtree(episode_dir, ignore_errors=True)

        low, high = wilson(successes, trials)
        row = {
            "episode": episode,
            "successes": successes,
            "trials": trials,
            "rate": successes / trials if trials else None,
            "ci95": [round(low, 4), round(high, 4)],
        }
        rows.append(row)
        out_path.write_text(json.dumps(rows, indent=2))

        done, total = position + 1, len(episodes)
        elapsed = time.time() - started
        log.info(
            "[%d/%d] episode %d: %d/%d  (%.0f min elapsed, ~%.0f min left)",
            done, total, episode, successes, trials,
            elapsed / 60,
            (elapsed / done) * (total - done) / 60,
        )

    shutil.rmtree(scratch, ignore_errors=True)
    return rows


def run_shard(cfg: EvalConfig, episodes: list[int], repeats: int) -> list[dict]:
    """Bring up a frozen VLA for this shard, profile, and always shut it down."""
    from .eval import start_server

    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    server = start_server(cfg, run_dir / "server.log")
    try:
        return profile(cfg, episodes, repeats, run_dir / "profile.json")
    finally:
        stop_server(server)


# --------------------------------------------------------------------------------------
# Reading the result
# --------------------------------------------------------------------------------------

#: What the project owner asked the three scenes to span.
BANDS = {
    "zero": (0.0, 0.001),
    "low": (0.20, 0.30),
    "good": (0.40, 1.01),
}

# CALIBRATION, measured 2026-08-20 and worth more than the bands themselves.
#
# "zero" out of four rollouts does NOT mean zero. Candidate episode 4 scored 0/4 here and
# 2/12 = 16.7% when re-run with twelve rollouts. That is not a contradiction: a scene with
# a true rate of 17% comes back 0/4 about 0.83^4 = 47% of the time.
#
# So a coarse zero is only a shortlist. Certifying a scene as a genuine zero -- the band
# that is interesting precisely because RL has to start with no reward at all -- needs
# tens of rollouts, which is what the 48-episode per-scene measurement provides. Until
# then, read "zero" as "under roughly 30%, possibly nowhere near it".
#
# The same logic bounds the other direction: 4/4 is consistent with any true rate above
# ~50%, so a coarse "good" is not evidence of headroom either.


def bucket(rate: float | None) -> str:
    """Which band a coarse rate falls in, or "" for the gaps between them."""
    if rate is None:
        return ""
    for name, (low, high) in BANDS.items():
        if low <= rate < high:
            return name
    return ""


def collect(paths: list[Path]) -> list[dict]:
    """Merge every shard's profile.json, newest wins on a repeated episode."""
    merged: dict[int, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            merged[int(row["episode"])] = row
    rows = sorted(merged.values(), key=lambda r: (-(r["rate"] or 0.0), r["episode"]))
    for row in rows:
        row["band"] = bucket(row["rate"])
    return rows
