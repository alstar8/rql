"""Encode offline MolmoBot + DROID demos into TokenReplay / ChunkReplay via MolmoAct2 /act."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import requests
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from buffer import (  # noqa: E402
    MLSPACES_GRIPPER_MAX_POS,
    GRIPPER_ACTION_SCALE,
    decode_json_bytes,
    episode_terminal_success,
    _action_from_joint,
    _state_from_qpos,
)
from chunk_replay import ChunkReplay, TokenReplay  # noqa: E402
from rlt_models import ACTION_DIM, CHUNK_SIZE, FEATURE_DIM, Z_DIM  # noqa: E402

log = logging.getLogger("molmoact2_cf.encode_offline_demo_tokens")

EXT_CAM_CANDIDATES = (
    "randomized_zed2_analogue_1",
    "droid_shoulder_light_randomization",
    "randomized_zed2_analogue_2",
)
WRIST_CAM = "wrist_camera_zed_mini"
IMG_HW = (180, 320)


def _wait_server(host: str, port: int, timeout: float) -> dict[str, Any]:
    start = time.time()
    while time.time() - start < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/act", timeout=3.0)
            response.raise_for_status()
            body = response.json()
            if isinstance(body, dict) and body.get("status") == "ok":
                return body
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2.0)
    raise RuntimeError(f"server {host}:{port} not ready within {timeout}s")


def _post_act(
    host: str,
    port: int,
    external: np.ndarray,
    wrist: np.ndarray,
    state: np.ndarray,
    instruction: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    import json_numpy

    payload = {
        "external_cam": np.asarray(external, dtype=np.uint8),
        "wrist_cam": np.asarray(wrist, dtype=np.uint8),
        "state": np.asarray(state, dtype=np.float32),
        "instruction": instruction,
    }
    response = requests.post(
        f"http://{host}:{port}/act",
        data=json_numpy.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    response.raise_for_status()
    body = json_numpy.loads(response.content)
    if not isinstance(body, dict):
        raise RuntimeError(f"bad /act response type {type(body)}")
    if "error" in body:
        raise RuntimeError(str(body["error"]))
    return body


def _resize_rgb(frame: np.ndarray, hw: tuple[int, int] = IMG_HW) -> np.ndarray:
    from PIL import Image

    h, w = hw
    if frame.shape[:2] == (h, w):
        return np.asarray(frame, dtype=np.uint8)
    return np.asarray(Image.fromarray(frame).resize((w, h), resample=Image.BICUBIC), dtype=np.uint8)


def _load_video_frame(h5_dir: Path, traj_group: Any, cam: str, index: int) -> np.ndarray:
    from decord import VideoReader

    raw = traj_group[f"obs/sensor_data/{cam}"][()]
    filename = (raw.tobytes() if isinstance(raw, np.ndarray) else raw).decode("utf-8").rstrip("\x00")
    vr = VideoReader(str(h5_dir / filename))
    idx = min(max(int(index), 0), len(vr) - 1)
    return _resize_rgb(vr[idx].asnumpy())


def _pick_ext_cam(traj_group: Any) -> str:
    keys = set(traj_group["obs/sensor_data"].keys())
    for name in EXT_CAM_CANDIDATES:
        if name in keys:
            return name
    raise KeyError(f"no exterior camera in {sorted(keys)}")


def _chunk_starts(n_steps: int, chunk: int, max_chunks: int) -> list[int]:
    if n_steps <= 0:
        return []
    starts = list(range(0, n_steps, chunk))
    if len(starts) > max_chunks:
        idxs = np.linspace(0, len(starts) - 1, num=max_chunks, dtype=int)
        starts = [starts[int(i)] for i in idxs]
    return starts


def _pack_chunk_actions(actions: list[np.ndarray], start: int, chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out = np.zeros((chunk, ACTION_DIM), dtype=np.float32)
    mask = np.zeros((chunk,), dtype=np.float32)
    rewards = np.zeros((chunk,), dtype=np.float32)
    for i in range(chunk):
        t = start + i
        if t >= len(actions):
            break
        out[i] = actions[t]
        mask[i] = 1.0
    return out, mask, rewards


def iter_molmobot_index(data_dir: Path) -> list[dict[str, Any]]:
    """Parse validate_trajectories index: {house: {rel_h5: {traj_key: length}}}."""
    index_path = data_dir / "valid_trajectory_index.json"
    rows: list[dict[str, Any]] = []
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if isinstance(raw, dict):
            for _house, files in raw.items():
                if not isinstance(files, dict):
                    continue
                for rel_h5, trajs in files.items():
                    if not isinstance(trajs, dict):
                        continue
                    for traj_key in trajs.keys():
                        rows.append({"h5": str(rel_h5), "traj_key": str(traj_key)})
            if rows:
                return rows
    import h5py

    for h5_path in sorted(data_dir.rglob("*.h5")):
        with h5py.File(h5_path, "r") as handle:
            for key in sorted(k for k in handle.keys() if k.startswith("traj_")):
                rows.append({"h5": str(h5_path.relative_to(data_dir)), "traj_key": key})
    return rows


def sample_rows(rows: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    if len(rows) <= n:
        return list(rows)
    idxs = rng.choice(len(rows), size=n, replace=False)
    return [rows[int(i)] for i in sorted(idxs.tolist())]


def select_droid_episodes(n: int, seed: int = 0) -> list[int]:
    """Pick n IPEC episodes that have a non-empty language task when possible."""
    from droid_ipec_loader import IPECDroidEpisodes

    path = IPECDroidEpisodes([0])._dl("meta/episodes.jsonl")
    good: list[int] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            tasks = row.get("tasks") or []
            if any(str(t).strip() for t in tasks):
                good.append(int(row["episode_index"]))
            if len(good) >= n * 5:
                break
    if len(good) < n:
        good = list(range(max(n, 1)))
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(good), size=min(n, len(good)), replace=False)
    return sorted(int(good[int(i)]) for i in idxs)


def encode_molmobot_episode(
    data_dir: Path,
    row: dict[str, Any],
    host: str,
    port: int,
    *,
    max_chunks: int,
    gamma: float,
    episode_id: int,
    token_replay: TokenReplay,
    chunk_replay: ChunkReplay,
    chunk_token_replay: TokenReplay,
) -> int:
    import h5py

    h5_path = data_dir / row["h5"]
    traj_key = row["traj_key"]
    with h5py.File(h5_path, "r") as handle:
        tg = handle[traj_key]
        n_act = int(tg["actions/joint_pos"].shape[0])
        n_obs = int(tg["obs/agent/qpos"].shape[0])
        effective = min(n_act, n_obs) - 2
        if effective < CHUNK_SIZE:
            return 0
        success = episode_terminal_success(tg)
        scene = decode_json_bytes(tg["obs_scene"][()])
        instruction = str(scene.get("task_description") or scene.get("instruction") or "pick up the object")
        ext_cam = _pick_ext_cam(tg)
        qpos = [decode_json_bytes(tg["obs/agent/qpos"][i]) for i in range(effective + 1)]
        acts = [_action_from_joint(decode_json_bytes(tg["actions/joint_pos"][i + 1])) for i in range(effective)]
        states = [_state_from_qpos(q) for q in qpos]
        starts = _chunk_starts(effective, CHUNK_SIZE, max_chunks)
        zs: list[np.ndarray] = []
        proprios: list[np.ndarray] = []
        refs: list[np.ndarray] = []
        executed: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for start in starts:
            ext = _load_video_frame(h5_path.parent, tg, ext_cam, start)
            wrist = _load_video_frame(h5_path.parent, tg, WRIST_CAM, start)
            body = _post_act(host, port, ext, wrist, states[start], instruction)
            tokens = np.asarray(body["token_features"], dtype=np.float32)
            mask = body.get("token_attention_mask")
            token_replay.add(tokens, None if mask is None else np.asarray(mask))
            chunk_token_replay.add(tokens, None if mask is None else np.asarray(mask))
            ref, m, r = _pack_chunk_actions(acts, start, CHUNK_SIZE)
            # Sparse terminal reward on final chunk last valid step.
            zs.append(np.zeros(Z_DIM, dtype=np.float32))
            proprios.append(states[start].astype(np.float32))
            refs.append(ref)
            executed.append(ref.copy())
            rewards.append(r)
            masks.append(m)
        if success and masks:
            last = int(masks[-1].sum()) - 1
            if last >= 0:
                rewards[-1][last] = 1.0
        return chunk_replay.add_episode_chunks(
            zs, proprios, refs, executed, rewards, masks, success, gamma, episode_id=episode_id
        )


def encode_droid_episode(
    episode: dict[str, Any],
    host: str,
    port: int,
    *,
    max_chunks: int,
    gamma: float,
    episode_id: int,
    token_replay: TokenReplay,
    chunk_replay: ChunkReplay,
    chunk_token_replay: TokenReplay,
) -> int:
    n = int(episode["n"])
    if n < CHUNK_SIZE:
        return 0
    states_raw = episode["states"]
    actions_raw = episode["actions"]
    ext_frames = episode["ext_frames"]
    wrist_frames = episode["wrist_frames"]
    instruction = str(episode["instruction"])
    success = bool(episode["success"])

    def _state_at(t: int) -> np.ndarray:
        st = np.asarray(states_raw[t], dtype=np.float32).reshape(-1)
        out = np.zeros(8, dtype=np.float32)
        out[: min(8, st.shape[0])] = st[:8]
        return out

    def _action_at(t: int) -> np.ndarray:
        act = np.asarray(actions_raw[t], dtype=np.float32).reshape(-1)
        out = np.zeros(8, dtype=np.float32)
        out[: min(8, act.shape[0])] = act[:8]
        return out

    actions = [_action_at(t) for t in range(n)]
    states = [_state_at(t) for t in range(n)]
    starts = _chunk_starts(n, CHUNK_SIZE, max_chunks)

    zs: list[np.ndarray] = []
    proprios: list[np.ndarray] = []
    refs: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for start in starts:
        ext = _resize_rgb(np.asarray(ext_frames[start]))
        wrist = _resize_rgb(np.asarray(wrist_frames[start]))
        body = _post_act(host, port, ext, wrist, states[start], instruction)
        tokens = np.asarray(body["token_features"], dtype=np.float32)
        mask = body.get("token_attention_mask")
        token_replay.add(tokens, None if mask is None else np.asarray(mask))
        chunk_token_replay.add(tokens, None if mask is None else np.asarray(mask))
        ref, m, r = _pack_chunk_actions(actions, start, CHUNK_SIZE)
        zs.append(np.zeros(Z_DIM, dtype=np.float32))
        proprios.append(states[start].astype(np.float32))
        refs.append(ref)
        executed.append(ref.copy())
        rewards.append(r)
        masks.append(m)
    if success and masks:
        last_i = int(masks[-1].sum()) - 1
        if last_i >= 0:
            rewards[-1][last_i] = 1.0
    return chunk_replay.add_episode_chunks(
        zs, proprios, refs, executed, rewards, masks, success, gamma, episode_id=episode_id
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["molmobot", "droid"], required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--server_host", type=str, default="127.0.0.1")
    parser.add_argument("--server_port", type=int, default=8700)
    parser.add_argument("--server_wait_sec", type=float, default=600.0)
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--max_chunks", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--molmobot_dir",
        type=str,
        default=str(
            _HERE
            / "../../../molmospaces/mbdata/FrankaPickOmniCamConfig/part0/train"
        ),
    )
    parser.add_argument("--droid_repo", type=str, default="IPEC-COMMUNITY/droid_lerobot")
    parser.add_argument("--droid_root", type=str, default="")
    parser.add_argument("--save_every", type=int, default=25)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import json_numpy

    json_numpy.patch()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    health = _wait_server(args.server_host, args.server_port, args.server_wait_sec)
    log.info("server ready: %s", {k: health.get(k) for k in ("feature_mode", "repo_id", "device")})
    if health.get("feature_mode") not in {"tokens", "both"}:
        raise RuntimeError("server must use --feature_mode tokens (or both)")

    token_replay = TokenReplay()
    chunk_token_replay = TokenReplay()
    chunk_replay = ChunkReplay(max_transitions=500_000)
    tag = f"{args.source}_s{args.shard_id}"

    if args.source == "molmobot":
        data_dir = Path(args.molmobot_dir).resolve()
        rows = iter_molmobot_index(data_dir)
        rows = sample_rows(rows, args.num_episodes, args.seed)
        rows = rows[args.shard_id :: args.num_shards]
        log.info("molmobot episodes for shard=%d: %d", args.shard_id, len(rows))
        for i, row in enumerate(rows):
            try:
                added = encode_molmobot_episode(
                    data_dir,
                    row,
                    args.server_host,
                    args.server_port,
                    max_chunks=args.max_chunks,
                    gamma=args.gamma,
                    episode_id=i,
                    token_replay=token_replay,
                    chunk_replay=chunk_replay,
                    chunk_token_replay=chunk_token_replay,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("molmobot encode failed row=%s: %s", row, exc)
                continue
            if (i + 1) % 10 == 0:
                log.info(
                    "molmobot %d/%d tokens=%d chunks=%d last_added=%d",
                    i + 1,
                    len(rows),
                    len(token_replay),
                    len(chunk_replay),
                    added,
                )
            if (i + 1) % args.save_every == 0:
                _save_all(out_dir, tag, token_replay, chunk_replay, chunk_token_replay)
    else:
        from droid_ipec_loader import IPECDroidEpisodes

        episodes = select_droid_episodes(args.num_episodes, seed=args.seed)
        episodes = episodes[args.shard_id :: args.num_shards]
        log.info("loading IPEC DROID episodes=%d root=%s", len(episodes), args.droid_root or "<hf cache>")
        dataset = IPECDroidEpisodes(episodes, cache_dir=args.droid_root or None)
        for i in range(len(dataset)):
            try:
                ep = dataset.load_episode(i)
                added = encode_droid_episode(
                    ep,
                    args.server_host,
                    args.server_port,
                    max_chunks=args.max_chunks,
                    gamma=args.gamma,
                    episode_id=i,
                    token_replay=token_replay,
                    chunk_replay=chunk_replay,
                    chunk_token_replay=chunk_token_replay,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("droid encode failed local_i=%s: %s", i, exc)
                continue
            if (i + 1) % 10 == 0:
                log.info(
                    "droid %d/%d tokens=%d chunks=%d last_added=%d",
                    i + 1,
                    len(episodes),
                    len(token_replay),
                    len(chunk_replay),
                    added,
                )
            if (i + 1) % args.save_every == 0:
                _save_all(out_dir, tag, token_replay, chunk_replay, chunk_token_replay)

    _save_all(out_dir, tag, token_replay, chunk_replay, chunk_token_replay)
    summary = {
        "source": args.source,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "token_sequences": len(token_replay),
        "chunk_transitions": len(chunk_replay),
        "chunk_token_sequences": len(chunk_token_replay),
        "n_episodes": chunk_replay.n_episodes,
    }
    (out_dir / f"encode_summary_{tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("done %s", summary)


def _save_all(
    out_dir: Path,
    tag: str,
    token_replay: TokenReplay,
    chunk_replay: ChunkReplay,
    chunk_token_replay: TokenReplay,
) -> None:
    if len(token_replay):
        token_replay.save_npz(str(out_dir / f"token_replay_{tag}.npz"))
    if len(chunk_token_replay):
        chunk_token_replay.save_npz(str(out_dir / f"chunk_token_replay_{tag}.npz"))
    if len(chunk_replay):
        chunk_replay.save_npz(str(out_dir / f"chunk_replay_{tag}.npz"))


if __name__ == "__main__":
    main()
