"""Tile per-episode eval videos into 4x4 grids.

MolmoSpaces writes one mp4 per episode per camera. This stacks 16 of them into
a single frame so a 64- or 128-episode run is watchable in a few clips instead
of a hundred. Episodes have different lengths, so shorter ones hold their last
frame; a thin border marks success (green) or failure (red).

    python scripts/make_grid_video.py --eval_dir eval_output/rlt_reference \
        --camera exo_camera_1 --out eval_output/rlt_reference/grids

Repeated rollouts of *one* episode are labelled differently: they all share a
(house, episode) key, so results.json cannot tell them apart. `rlt.evaluate
--video_dir` names those files `rollout_007_ok_<camera>.mp4`, and the outcome is
read from the name.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import imageio.v3 as iio
import numpy as np

EPISODE_RE = re.compile(r"episode_(\d+)_")


ROLLOUT_RE = re.compile(r"rollout_(\d+)_(ok|fail)_")


def find_videos(eval_dir: Path, camera: str) -> list[Path]:
    """Sorted by (house, episode) so grids are reproducible across arms."""
    rollouts = list(eval_dir.glob(f"**/rollout_*_{camera}*.mp4"))
    if rollouts:  # repeats of one episode, written by rlt.evaluate --video_dir
        # Shards number their rollouts independently, so the directory has to
        # come first or two shards' clips interleave differently each run.
        return sorted(rollouts, key=lambda p: (p.parent.name, p.name))
    videos = sorted(eval_dir.glob(f"**/episode_*_{camera}*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no '{camera}' videos under {eval_dir}")
    return sorted(videos, key=lambda p: (p.parent.name, p.name))


def video_key(path: Path) -> tuple[str, int]:
    """(house, episode_idx) from .../house_101/episode_00000001_<cam>_....mp4

    A house can hold several episodes, so the house alone is not a key -- using
    it silently mislabels every house with more than one episode.
    """
    m = EPISODE_RE.search(path.name)
    if m is None:
        raise ValueError(f"cannot read an episode index from {path.name}")
    return path.parent.name, int(m.group(1))


def success_from_name(path: Path) -> bool | None:
    """`rollout_007_ok_exo_camera_1.mp4` -> True. None when it is not one of ours."""
    m = ROLLOUT_RE.search(path.name)
    return None if m is None else m.group(2) == "ok"


def load_success(eval_dir: Path) -> dict[tuple[str, int], bool]:
    """Map (house, episode_idx) -> success, from results.json if present."""
    results = next(eval_dir.glob("**/results.json"), None)
    if results is None:
        return {}
    try:
        data = json.loads(results.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        (str(ep["house_id"]), int(ep["episode_idx"])): bool(ep["success"])
        for ep in data.get("episodes", [])
        if "house_id" in ep and "episode_idx" in ep
    }


def border(frames: np.ndarray, success: bool | None, width: int = 4) -> np.ndarray:
    if success is None:
        return frames
    color = (60, 200, 60) if success else (200, 60, 60)
    out = frames.copy()
    out[:, :width, :] = color
    out[:, -width:, :] = color
    out[:, :, :width] = color
    out[:, :, -width:] = color
    return out


def read_clip(path: Path, max_frames: int) -> np.ndarray:
    # FFMPEG, not pyav: molmospaces installs imageio[ffmpeg], pyav is absent.
    frames = np.asarray(iio.imread(path, plugin="FFMPEG"))
    return frames[:max_frames]


def pad_to(frames: np.ndarray, n: int) -> np.ndarray:
    """Hold the last frame so every tile in a grid runs the same length."""
    if len(frames) >= n:
        return frames[:n]
    tail = np.repeat(frames[-1:], n - len(frames), axis=0)
    return np.concatenate([frames, tail], axis=0)


def build_grid(clips: list[np.ndarray], nrow: int, ncol: int) -> np.ndarray:
    h, w = clips[0].shape[1:3]
    n_frames = max(len(c) for c in clips)
    blank = np.zeros((n_frames, h, w, 3), dtype=np.uint8)

    tiles = [pad_to(c, n_frames) for c in clips]
    tiles += [blank] * (nrow * ncol - len(tiles))  # pad a short final grid

    rows = [np.concatenate(tiles[r * ncol : (r + 1) * ncol], axis=2) for r in range(nrow)]
    grid = np.concatenate(rows, axis=1)
    # encoders want even dimensions
    return grid[:, : grid.shape[1] - grid.shape[1] % 2, : grid.shape[2] - grid.shape[2] % 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--camera", default="exo_camera_1")
    ap.add_argument("--out", default="")
    ap.add_argument("--nrow", type=int, default=4)
    ap.add_argument("--ncol", type=int, default=4)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max_frames", type=int, default=600, help="cap per clip; 500-step episodes are long")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out) if args.out else eval_dir / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(eval_dir, args.camera)
    per_grid = args.nrow * args.ncol
    print(f"{len(videos)} episode videos -> {(len(videos) + per_grid - 1) // per_grid} grid(s)")

    # Outcome per clip: from the filename for repeats of one episode, otherwise
    # from results.json. Cross-check before spending minutes on encoding -- a
    # count mismatch means the borders would be labelling the wrong clips.
    by_name = {p: success_from_name(p) for p in videos}
    if all(v is not None for v in by_name.values()):
        outcome = by_name
        print(f"labels from filenames; {sum(by_name.values())}/{len(videos)} successful (green)")
    else:
        results = load_success(eval_dir)
        outcome = {p: results.get(video_key(p)) for p in videos}
        labelled = sum(v is not None for v in outcome.values())
        if results:
            n_ok = sum(bool(v) for v in outcome.values())
            print(f"labelled {labelled}/{len(videos)} videos; {n_ok} successful (green)")
            if labelled != len(videos):
                print("  WARNING: some videos have no entry in results.json and stay unbordered")
        else:
            print("note: no results.json found, so no success/failure borders are drawn")

    for g_i in range(0, len(videos), per_grid):
        batch = videos[g_i : g_i + per_grid]
        clips = [border(read_clip(p, args.max_frames), outcome[p]) for p in batch]
        grid = build_grid(clips, args.nrow, args.ncol)

        out_path = out_dir / f"{args.camera}_grid_{g_i // per_grid:02d}.mp4"
        iio.imwrite(out_path, grid, fps=args.fps, codec="libx264", plugin="FFMPEG")
        print(f"  {out_path}  {grid.shape[0]} frames  {grid.shape[2]}x{grid.shape[1]}")


if __name__ == "__main__":
    main()
