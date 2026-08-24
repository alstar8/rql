"""Merge LeRobot datasets produced by parallel converter workers into one.

Converting the whole corpus in a single process is bound by video decoding, so workers
each build their own dataset and the results are joined here. Three things have to be
renumbered, and each is a silent failure if missed:

  episode_index  restarts at zero in every part, in the parquet and the metadata alike;
  index          is a global frame counter, so it has to continue across parts;
  task_index     is assigned per part, so the same sentence has different indices in
                 different parts and the mapping has to be rebuilt from the strings.

The merge parallelises because all three renumberings are decided *before* any data is
read: every part's episode and frame offsets follow from the episode counts in its
metadata, and the task mapping from its tasks.jsonl. Workers therefore never coordinate,
and the result is identical to a sequential merge, which the tests assert directly.

Nothing is deleted: the parts are read and a new dataset is written beside them.
"""

from __future__ import annotations

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

CHUNK_SIZE = 1000
base_fps = 15


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _episode_path(root: Path, episode_index: int) -> Path:
    chunk = episode_index // CHUNK_SIZE
    return root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


@dataclass
class PartPlan:
    """Everything a worker needs to merge one part without talking to the others."""

    part: Path
    episode_offset: int
    index_offset: int
    task_remap: dict[int, int]
    episodes: list[dict]


def plan_merge(
    parts: list[Path], *, drop_leading_frames: int = 0
) -> tuple[list[PartPlan], dict[str, int]]:
    """Decide every renumbering up front, reading only metadata."""
    tasks: dict[str, int] = {}
    plans: list[PartPlan] = []
    episode_offset = 0
    index_offset = 0

    for part in parts:
        part_tasks = {
            row["task_index"]: row["task"] for row in _read_jsonl(part / "meta" / "tasks.jsonl")
        }
        # The same sentence must land on one global index whichever part it came from.
        task_remap = {
            local: tasks.setdefault(text, len(tasks)) for local, text in part_tasks.items()
        }

        episodes = sorted(
            _read_jsonl(part / "meta" / "episodes.jsonl"), key=lambda r: r["episode_index"]
        )
        plans.append(PartPlan(part, episode_offset, index_offset, task_remap, episodes))

        episode_offset += len(episodes)
        # "length" is the frame count recorded per episode; summing it gives this part's
        # contribution to the global frame counter without opening a single parquet.
        index_offset += sum(
            max(0, int(row["length"]) - drop_leading_frames) for row in episodes
        )

    return plans, tasks


def _merge_one_part(args: tuple[PartPlan, Path, bool, int]) -> dict:
    plan, out, verbose, drop_leading = args
    stats_by_episode = {
        row["episode_index"]: row
        for row in _read_jsonl(plan.part / "meta" / "episodes_stats.jsonl")
    }

    episodes_out, stats_out = [], []
    next_episode = plan.episode_offset
    next_index = plan.index_offset
    missing = 0

    for episode in plan.episodes:
        old_index = episode["episode_index"]
        source = _episode_path(plan.part, old_index)
        if not source.exists():
            missing += 1
            continue

        table = pq.read_table(source)
        if drop_leading:
            # The demonstrations open with a spurious command the controller ignores and
            # then a hold, so the recorded arm does not move until t2. Converting already
            # dropped the spurious command; dropping the hold as well stops every episode
            # from teaching "at the reset pose, output zero" -- the one moment every
            # evaluation episode starts from.
            table = table.slice(drop_leading)
        frame_count = table.num_rows
        if frame_count == 0:
            missing += 1
            continue
        # frame_index and timestamp restart from zero for the shortened episode.
        frame_index = list(range(frame_count))
        table = table.set_column(
            table.schema.get_field_index("frame_index"), "frame_index", [frame_index]
        )
        fps = float(base_fps) if base_fps else 15.0
        table = table.set_column(
            table.schema.get_field_index("timestamp"),
            "timestamp",
            [[i / fps for i in frame_index]],
        )

        table = table.set_column(
            table.schema.get_field_index("episode_index"),
            "episode_index",
            [[next_episode] * frame_count],
        )
        table = table.set_column(
            table.schema.get_field_index("index"),
            "index",
            [[next_index + i for i in frame_index]],
        )
        table = table.set_column(
            table.schema.get_field_index("task_index"),
            "task_index",
            [[plan.task_remap[i] for i in table.column("task_index").to_pylist()]],
        )

        destination = _episode_path(out, next_episode)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination)

        episodes_out.append({**episode, "episode_index": next_episode, "length": frame_count})
        if old_index in stats_by_episode:
            stats_out.append({**stats_by_episode[old_index], "episode_index": next_episode})

        next_episode += 1
        next_index += frame_count

    if verbose:
        print(f"  {plan.part.name}: {len(episodes_out)} episodes written", flush=True)

    return {
        "episodes": episodes_out,
        "stats": stats_out,
        "missing": missing,
    }


def merge_datasets(
    parts: list[Path],
    out: Path,
    *,
    workers: int = 1,
    drop_leading_frames: int = 0,
    verbose: bool = True,
) -> dict:
    """Join ``parts`` into a single LeRobot dataset at ``out``.

    ``workers`` changes only how fast this runs, never what it produces: with the plan
    fixed in advance, each part's output depends on nothing but its own contents.
    """
    parts = [Path(p) for p in parts if (Path(p) / "meta" / "info.json").exists()]
    if not parts:
        raise ValueError("no parts with meta/info.json to merge")
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists; move it aside rather than overwrite")

    base_info = json.loads((parts[0] / "meta" / "info.json").read_text())
    plans, tasks = plan_merge(parts, drop_leading_frames=drop_leading_frames)

    payloads = [(plan, out, verbose, drop_leading_frames) for plan in plans]
    if workers > 1:
        with mp.Pool(min(workers, len(payloads))) as pool:
            results = pool.map(_merge_one_part, payloads)
    else:
        results = [_merge_one_part(p) for p in payloads]

    episodes = [row for r in results for row in r["episodes"]]
    episode_stats = [row for r in results for row in r["stats"]]
    missing = sum(r["missing"] for r in results)

    # An episode listed in metadata but absent on disk would shift every later index, so
    # the offsets planned up front would no longer describe what was actually written.
    if missing:
        raise RuntimeError(
            f"{missing} episodes listed in metadata had no parquet file; the merged "
            "dataset would be misnumbered. Re-run the affected parts before merging."
        )

    _write_jsonl(out / "meta" / "episodes.jsonl", episodes)
    _write_jsonl(out / "meta" / "episodes_stats.jsonl", episode_stats)
    _write_jsonl(
        out / "meta" / "tasks.jsonl",
        [{"task_index": i, "task": t} for t, i in sorted(tasks.items(), key=lambda kv: kv[1])],
    )

    total_episodes = len(episodes)
    total_frames = sum(int(row["length"]) for row in episodes)

    info = dict(base_info)
    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_tasks"] = len(tasks)
    info["total_chunks"] = max(1, (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE)
    info["splits"] = {"train": f"0:{total_episodes}"}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))

    return {
        "episodes": total_episodes,
        "frames": total_frames,
        "tasks": len(tasks),
        "parts": len(parts),
    }
