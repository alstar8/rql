"""Merging converter parts, tested on the renumbering that fails silently.

A merge that loses an episode is obvious. A merge that keeps every episode but points
half of them at the wrong instruction, or repeats a frame index, trains a worse model
with no visible symptom -- so those are what these tests pin down.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pi05.merge import merge_datasets


def _make_part(root: Path, episodes: list[tuple[str, int]]) -> Path:
    """Build a minimal LeRobot part: episodes given as (task text, frame count)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}
    ep_rows, stat_rows = [], []
    running = 0
    for episode_index, (task, length) in enumerate(episodes):
        task_index = tasks.setdefault(task, len(tasks))
        table = pa.table(
            {
                "value": pa.array([float(episode_index)] * length),
                "timestamp": pa.array([i / 15 for i in range(length)]),
                "frame_index": pa.array(list(range(length))),
                "episode_index": pa.array([episode_index] * length),
                "index": pa.array([running + i for i in range(length)]),
                "task_index": pa.array([task_index] * length),
            }
        )
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet")
        ep_rows.append({"episode_index": episode_index, "tasks": [task], "length": length})
        stat_rows.append({"episode_index": episode_index, "stats": {"value": {"count": [length]}}})
        running += length

    with (root / "meta" / "episodes.jsonl").open("w") as f:
        for row in ep_rows:
            f.write(json.dumps(row) + "\n")
    with (root / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in stat_rows:
            f.write(json.dumps(row) + "\n")
    with (root / "meta" / "tasks.jsonl").open("w") as f:
        for text, index in sorted(tasks.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": index, "task": text}) + "\n")
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v2.1",
                "robot_type": "panda",
                "fps": 15,
                "chunks_size": 1000,
                "total_episodes": len(episodes),
                "total_frames": running,
                "total_tasks": len(tasks),
                "total_chunks": 1,
                "splits": {"train": f"0:{len(episodes)}"},
                "features": {},
            }
        )
    )
    return root


@pytest.fixture
def parts(tmp_path):
    # "pick up the mug." appears in both parts, at a different local index in each.
    a = _make_part(tmp_path / "part_a", [("pick up the mug.", 3), ("pick up the cone.", 2)])
    b = _make_part(tmp_path / "part_b", [("pick up the pot.", 4), ("pick up the mug.", 2)])
    return [a, b]


def _read_all(out: Path):
    rows = []
    for path in sorted((out / "data").rglob("*.parquet")):
        rows.append(pq.read_table(path).to_pandas())
    return rows


def test_all_episodes_and_frames_survive(parts, tmp_path):
    stats = merge_datasets(parts, tmp_path / "out", verbose=False)
    assert stats["episodes"] == 4
    assert stats["frames"] == 3 + 2 + 4 + 2


def test_episode_indices_are_contiguous_and_match_the_parquet(parts, tmp_path):
    out = tmp_path / "out"
    merge_datasets(parts, out, verbose=False)
    meta = [json.loads(l) for l in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in meta] == [0, 1, 2, 3]
    for frame in _read_all(out):
        assert frame["episode_index"].nunique() == 1


def test_global_frame_index_is_continuous_with_no_repeats(parts, tmp_path):
    out = tmp_path / "out"
    merge_datasets(parts, out, verbose=False)
    indices = sorted(i for frame in _read_all(out) for i in frame["index"].tolist())
    # A part-local counter restarting would duplicate low indices; this catches it.
    assert indices == list(range(len(indices)))


def test_a_shared_instruction_collapses_to_one_task_index(parts, tmp_path):
    out = tmp_path / "out"
    stats = merge_datasets(parts, out, verbose=False)
    tasks = [json.loads(l) for l in (out / "meta" / "tasks.jsonl").read_text().splitlines()]
    assert stats["tasks"] == 3  # mug, cone, pot -- mug appears in both parts
    by_text = {row["task"]: row["task_index"] for row in tasks}
    assert len(set(by_text.values())) == 3


def test_every_episode_keeps_its_own_instruction(parts, tmp_path):
    out = tmp_path / "out"
    merge_datasets(parts, out, verbose=False)
    tasks = {
        json.loads(l)["task_index"]: json.loads(l)["task"]
        for l in (out / "meta" / "tasks.jsonl").read_text().splitlines()
    }
    meta = [json.loads(l) for l in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    for row, frame in zip(meta, _read_all(out), strict=True):
        # The remapped index in the parquet must still name the sentence recorded for it.
        assert tasks[frame["task_index"].iloc[0]] == row["tasks"][0]


def test_info_totals_match_the_merged_content(parts, tmp_path):
    out = tmp_path / "out"
    merge_datasets(parts, out, verbose=False)
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 4
    assert info["total_frames"] == 11
    assert info["splits"] == {"train": "0:4"}


def test_stats_follow_their_renumbered_episode(parts, tmp_path):
    out = tmp_path / "out"
    merge_datasets(parts, out, verbose=False)
    stats = [json.loads(l) for l in (out / "meta" / "episodes_stats.jsonl").read_text().splitlines()]
    assert [row["episode_index"] for row in stats] == [0, 1, 2, 3]
    assert [row["stats"]["value"]["count"][0] for row in stats] == [3, 2, 4, 2]


def test_refuses_to_overwrite_an_existing_dataset(parts, tmp_path):
    out = tmp_path / "out"
    merge_datasets(parts, out, verbose=False)
    with pytest.raises(FileExistsError):
        merge_datasets(parts, out, verbose=False)


def test_empty_input_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        merge_datasets([tmp_path / "nothing"], tmp_path / "out", verbose=False)


def test_parallel_merge_matches_sequential(parts, tmp_path):
    seq = tmp_path / "seq"
    par = tmp_path / "par"
    a = merge_datasets(parts, seq, workers=1, verbose=False)
    b = merge_datasets(parts, par, workers=4, verbose=False)
    assert a == b
    for name in ("episodes.jsonl", "tasks.jsonl", "info.json"):
        assert (seq / "meta" / name).read_text() == (par / "meta" / name).read_text()
    seq_rows = _read_all(seq)
    par_rows = _read_all(par)
    assert len(seq_rows) == len(par_rows)
    for x, y in zip(seq_rows, par_rows, strict=True):
        assert x.equals(y)


def test_missing_parquet_is_refused_not_silently_renumbered(parts, tmp_path):
    # Deleting one episode file would shift every later index; the planned offsets would
    # then describe something other than what was written, so this must raise.
    victim = sorted((parts[0] / "data").rglob("*.parquet"))[0]
    victim.unlink()
    with pytest.raises(RuntimeError, match="misnumbered"):
        merge_datasets(parts, tmp_path / "out", workers=2, verbose=False)


def test_drop_leading_frames_shortens_every_episode_and_renumbers(parts, tmp_path):
    out = tmp_path / "dropped"
    stats = merge_datasets(parts, out, workers=2, drop_leading_frames=1, verbose=False)
    # Four episodes of 3,2,4,2 frames lose one each.
    assert stats["frames"] == (3 - 1) + (2 - 1) + (4 - 1) + (2 - 1)
    meta = [json.loads(l) for l in (out / "meta" / "episodes.jsonl").read_text().splitlines()]
    assert [row["length"] for row in meta] == [2, 1, 3, 1]
    frames = _read_all(out)
    for frame in frames:
        # frame_index must restart at zero, not begin at one.
        assert frame["frame_index"].tolist() == list(range(len(frame)))
    # the global index stays contiguous after the shortening
    idx = sorted(i for f in frames for i in f["index"].tolist())
    assert idx == list(range(len(idx)))


def test_drop_leading_frames_removes_the_first_row_not_an_arbitrary_one(parts, tmp_path):
    kept = merge_datasets(parts, tmp_path / "kept", workers=1, verbose=False)
    dropped = merge_datasets(parts, tmp_path / "drop", workers=1, drop_leading_frames=1, verbose=False)
    assert dropped["frames"] == kept["frames"] - kept["episodes"]
    a = _read_all(tmp_path / "kept")[0]
    b = _read_all(tmp_path / "drop")[0]
    # value column encodes the source episode; the surviving rows are the tail of the original
    assert b["value"].tolist() == a["value"].tolist()[1:]
