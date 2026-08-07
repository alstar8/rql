"""Minimal IPEC DROID episode loader (no full LeRobot install required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


class IPECDroidEpisodes:
    """Lazy per-episode access to IPEC-COMMUNITY/droid_lerobot."""

    REPO = "IPEC-COMMUNITY/droid_lerobot"
    EXT_KEY = "observation.images.exterior_image_1_left"
    WRIST_KEY = "observation.images.wrist_image_left"
    CHUNK_SIZE = 1000

    def __init__(self, episode_indices: list[int], cache_dir: str | None = None) -> None:
        self.episode_indices = [int(i) for i in episode_indices]
        self.cache_dir = cache_dir
        self._tasks = self._load_tasks()
        self._episode_meta = self._load_episode_meta()

    def _dl(self, path: str) -> str:
        return hf_hub_download(repo_id=self.REPO, filename=path, repo_type="dataset")

    def _load_tasks(self) -> dict[int, str]:
        path = self._dl("meta/tasks.jsonl")
        out: dict[int, str] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                idx = int(row.get("task_index", -1))
                text = str(row.get("task") or "").strip()
                if idx >= 0 and text:
                    out[idx] = text
        return out

    def _load_episode_meta(self) -> dict[int, dict[str, Any]]:
        path = self._dl("meta/episodes.jsonl")
        wanted = set(self.episode_indices)
        out: dict[int, dict[str, Any]] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                idx = int(row["episode_index"])
                if idx in wanted:
                    out[idx] = row
                if len(out) >= len(wanted):
                    break
        return out

    def __len__(self) -> int:
        return len(self.episode_indices)

    def _paths(self, episode_index: int) -> tuple[str, str, str]:
        chunk = episode_index // self.CHUNK_SIZE
        data = f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        ext = (
            f"videos/chunk-{chunk:03d}/{self.EXT_KEY}/"
            f"episode_{episode_index:06d}.mp4"
        )
        wrist = (
            f"videos/chunk-{chunk:03d}/{self.WRIST_KEY}/"
            f"episode_{episode_index:06d}.mp4"
        )
        return data, ext, wrist

    @staticmethod
    def _read_video(path: str) -> list[np.ndarray]:
        container = av.open(path)
        stream = container.streams.video[0]
        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()
        return frames

    def load_episode(self, local_i: int) -> dict[str, Any]:
        episode_index = self.episode_indices[local_i]
        data_rel, ext_rel, wrist_rel = self._paths(episode_index)
        table = pq.read_table(self._dl(data_rel))
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        task_idx = int(np.asarray(table["task_index"].to_pylist(), dtype=np.int64)[0])
        instruction = self._tasks.get(task_idx, "complete the manipulation task")
        meta = self._episode_meta.get(episode_index, {})
        # Prefer non-empty task strings from episode meta if present.
        tasks = meta.get("tasks") or []
        for text in tasks:
            if str(text).strip():
                instruction = str(text).strip()
                break
        ext_frames = self._read_video(self._dl(ext_rel))
        wrist_frames = self._read_video(self._dl(wrist_rel))
        n = min(len(states), len(actions), len(ext_frames), len(wrist_frames))
        success = False
        # IPEC does not always expose success; leave False unless file path hints exist.
        return {
            "episode_index": episode_index,
            "n": n,
            "states": states[:n],
            "actions": actions[:n],
            "ext_frames": ext_frames[:n],
            "wrist_frames": wrist_frames[:n],
            "instruction": instruction,
            "success": success,
            "task_index": task_idx,
        }
