#!/usr/bin/env python
"""Turn the local Franka Pick demonstrations into a QUORUM replay and distill observations.

The pretrained pi0.5 stays frozen. Each decision is one 8-step chunk of a successful
demonstration. The replay action is the demonstrator's joint deltas. The reference is
the frozen model's own 8-step chunk at that same image, so the critic sees an action
the VLA did not take. The RL state is the shared Pick-18 token AE plus proprioception,
the same pair evaluation uses.

Demonstrations open with two frames that are not motion: a commanded jump the arm
never executes, then a hold. pi0.5 finetuning drops both. Decisions start at step 2
and then every 8 steps, on the frames the finetune actually trained on.

    CUDA_VISIBLE_DEVICES=0 python scripts/encode_pick_demos.py --out /path/v25_pick1051
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.action_space import (  # noqa: E402
    action_from_command,
    normalize_instruction,
    state_from_qpos,
)
from pi05.convert import IMAGE_SIZE, _resize  # noqa: E402
from pi05.demo_reader import iter_episodes, read_video, unpack_json_row  # noqa: E402

CHUNK = 8
HORIZON = 16
GAMMA = 0.99
DROP_FIRST = 2
ARM = 7
AE = (
    "/workspace-SR008.nfs2/users/staroverov/B1K/B1K_AIRI/submodules/rql/"
    "my_exps/molmoact2_cf/runs/pick18_v24_shared/ae/ae_pick18_shared.pt"
)
CKPT = (
    "/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/"
    "pi05_droid_finetune_pick_full_v3_39999"
)
SOURCE = Path(
    "/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/molmospaces/"
    "mbdata/FrankaPickOmniCamConfig/part0/train"
)


def proprio_from(qpos: dict, qvel: dict) -> np.ndarray:
    """Same 16-vector the eval policy stashes: positions, then velocities."""
    arm_pos = np.asarray(qpos["arm"][:ARM], dtype=np.float32)
    grip_pos = float(np.clip(qpos["gripper"][0] / 0.824033, 0.0, 1.0))
    arm_vel = np.asarray(qvel["arm"][:ARM], dtype=np.float32)
    grip_vel = float(qvel["gripper"][0]) / 0.824033
    return np.concatenate([arm_pos, [grip_pos], arm_vel, [grip_vel]]).astype(np.float32)


def load_episode_side(episode) -> tuple[np.ndarray, list[dict]]:
    with h5py.File(episode.h5_path, "r") as handle:
        traj = handle[episode.traj_key]
        success = np.asarray(traj["success"]).astype(bool)
        qvel = [unpack_json_row(traj["obs/agent/qvel"][i]) for i in range(episode.n_steps)]
    return success, qvel


def decision_times(n_steps: int, terminal: int) -> list[int]:
    times = []
    t = DROP_FIRST
    while t + CHUNK <= n_steps and t <= terminal:
        times.append(t)
        t += CHUNK
    return times


def chunk_reward(terminal: int, t: int) -> tuple[float, float]:
    """Sparse +1 on the first success step, discounted inside the chunk. done is 1 if it lands here."""
    if terminal < t or terminal >= t + CHUNK:
        return 0.0, 0.0
    return float(GAMMA ** (terminal - t)), 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--token-ae", default=AE)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args()

    out = args.out
    row_dir = out / "rows"
    row_dir.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in row_dir.glob("*.npz")}

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    from pi05.eval import disable_policy_compile
    from pi05.vla_server import FrozenPi05
    from rlt.vla import TokenEncoder

    disable_policy_compile()
    policy = _policy_config.create_trained_policy(
        _config.get_config("pi05_droid_finetune"), args.checkpoint, pytorch_device="cuda"
    )
    frozen = FrozenPi05(policy)
    encoder = TokenEncoder(args.token_ae, "cuda")

    def episode_key(episode_id: str) -> str:
        return episode_id.replace("/", "__")

    n_eps = 0
    for episode in iter_episodes(args.source, require_success=True):
        if args.max_episodes is not None and n_eps >= args.max_episodes:
            break
        key = episode_key(episode.episode_id)
        if key in done:
            continue
        success, qvel = load_episode_side(episode)
        if not success.any():
            continue
        terminal = int(np.argmax(success))
        if terminal >= episode.n_steps:
            terminal = episode.n_steps - 1
        times = decision_times(episode.n_steps, terminal)
        if not times:
            done.add(episode.episode_id)
            continue
        exo = read_video(episode.exo_video, episode.n_steps + 1)
        wrist = read_video(episode.wrist_video, episode.n_steps + 1)
        instruction = normalize_instruction(episode.instruction)
        decoded = []
        for t in times:
            state8 = state_from_qpos(episode.qpos[t])
            observation = {
                "observation/exterior_image_1_left": _resize(exo[t]),
                "observation/wrist_image_left": _resize(wrist[t]),
                "observation/joint_position": state8[:ARM],
                "observation/gripper_position": state8[ARM:8],
                "prompt": instruction,
            }
            out_vla = frozen.predict(observation, want_tokens=True)
            actions16 = np.asarray(out_vla["actions"], dtype=np.float32)
            if actions16.shape[0] < HORIZON:
                raise RuntimeError(f"VLA chunk {actions16.shape}, expected {HORIZON}")
            z = encoder.encode(out_vla["token_features"], out_vla["token_attention_mask"])
            prop = proprio_from(episode.qpos[t], qvel[t])
            demo = np.stack(
                [
                    action_from_command(episode.commands[t + k], episode.qpos[t + k])
                    for k in range(CHUNK)
                ]
            )
            reward, done_flag = chunk_reward(terminal, t)
            decoded.append(
                {
                    "state": np.concatenate([z, prop]).astype(np.float32),
                    "action": demo.reshape(-1),
                    "reference8": actions16[:CHUNK].reshape(-1),
                    "reference16": actions16[:HORIZON, :8].astype(np.float32),
                    "reward": reward,
                    "done": done_flag,
                    "external_cam": observation["observation/exterior_image_1_left"],
                    "wrist_cam": observation["observation/wrist_image_left"],
                    "state8": state8,
                    "instruction": instruction,
                }
            )
        linked_state = []
        linked_action = []
        linked_ref = []
        linked_next = []
        linked_next_ref = []
        linked_reward = []
        linked_done = []
        for i, row in enumerate(decoded):
            terminal_row = row["done"] == 1.0 or i + 1 == len(decoded)
            if terminal_row:
                nxt = np.zeros_like(row["state"])
                nxt_ref = np.zeros_like(row["reference8"])
                row["done"] = 1.0
            else:
                nxt = decoded[i + 1]["state"]
                nxt_ref = decoded[i + 1]["reference8"]
            linked_state.append(row["state"])
            linked_action.append(row["action"])
            linked_ref.append(row["reference8"])
            linked_next.append(nxt)
            linked_next_ref.append(nxt_ref)
            linked_reward.append(row["reward"])
            linked_done.append(row["done"])
        np.savez(
            row_dir / f"{key}.npz",
            state=np.stack(linked_state),
            action=np.stack(linked_action),
            reference=np.stack(linked_ref),
            next_state=np.stack(linked_next),
            next_reference=np.stack(linked_next_ref),
            reward=np.asarray(linked_reward, dtype=np.float32),
            done=np.asarray(linked_done, dtype=np.float32),
            external_cam=np.stack([r["external_cam"] for r in decoded]),
            wrist_cam=np.stack([r["wrist_cam"] for r in decoded]),
            state8=np.stack([r["state8"] for r in decoded]),
            instruction=np.asarray([r["instruction"] for r in decoded]),
            reference16=np.stack([r["reference16"] for r in decoded]),
        )
        n_eps += 1
        if n_eps % 20 == 0:
            print(f"{n_eps} new episodes", flush=True)

    paths = sorted(row_dir.glob("*.npz"))
    if args.max_episodes is not None:
        return
    if not paths:
        raise SystemExit("no decisions encoded")
    rl = {k: [] for k in ("state", "action", "reference", "next_state", "next_reference", "reward", "done")}
    obs_dir = out / "obs"
    obs_dir.mkdir(parents=True, exist_ok=True)
    flat_ext, flat_wrist, flat_state, flat_instr, flat_ref = [], [], [], [], []
    shard_i = 0

    def dump() -> None:
        nonlocal shard_i
        if not flat_ext:
            return
        np.savez(
            obs_dir / f"obs_{shard_i:05d}.npz",
            external_cam=np.stack(flat_ext),
            wrist_cam=np.stack(flat_wrist),
            state=np.stack(flat_state),
            instruction=np.asarray(flat_instr),
            reference=np.stack(flat_ref),
        )
        shard_i += 1
        flat_ext.clear()
        flat_wrist.clear()
        flat_state.clear()
        flat_instr.clear()
        flat_ref.clear()

    for path in paths:
        with np.load(path) as part:
            for key in rl:
                rl[key].append(part[key])
            for i in range(part["state"].shape[0]):
                flat_ext.append(part["external_cam"][i])
                flat_wrist.append(part["wrist_cam"][i])
                flat_state.append(part["state8"][i])
                flat_instr.append(str(part["instruction"][i]))
                flat_ref.append(part["reference16"][i])
                if len(flat_ext) >= args.shard_size:
                    dump()
    dump()
    n = int(sum(chunk.shape[0] for chunk in rl["state"]))
    np.savez(
        out / "buffer.npz",
        size=np.int64(n),
        state=np.concatenate(rl["state"]),
        action=np.concatenate(rl["action"]),
        reference=np.concatenate(rl["reference"]),
        next_state=np.concatenate(rl["next_state"]),
        next_reference=np.concatenate(rl["next_reference"]),
        reward=np.concatenate(rl["reward"]),
        done=np.concatenate(rl["done"]),
    )
    meta = {
        "episodes_encoded": len(paths),
        "decisions": n,
        "obs_shards": shard_i,
        "gamma": GAMMA,
        "chunk": CHUNK,
        "drop_first": DROP_FIRST,
        "checkpoint": args.checkpoint,
        "token_ae": args.token_ae,
    }
    (out / "encode_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
