#!/usr/bin/env python
"""One rollout worker for pi05.train_parallel. See pi05/collectors.py.

Single-scene runs behave exactly as before. A shared multi-task run
(`task_pool` set) cycles episode N through `task_list()[N % len]` and keeps one
EpisodeRunner per scene, all on the same policy; the payload carries the scene
so the learner can log per-task curves. With several VLA servers (`vla_ports`)
or EGL devices (`egl_devices`), rank r is pinned to entry r % n of each, and
publishes config_r{r}.json -- the file the policy worker reads its port from.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.collectors import (
    COUNTER,
    EGL,
    INBOX,
    WEIGHTS,
    EglSlots,
    FileCounter,
    load_live_weights,
    write_inbox,
)
from pi05.config import SCENES, load_run_config
from pi05.round_eval import (
    EVAL_COUNTER,
    apply_arm,
    read_actor_limit,
    read_cmd,
    write_heartbeat,
)


def round_eval_payload(eval_i, rank, scene, arm, seq, *, success=0.0, steps=0, sec=0.0, **extra):
    return {
        "episode": eval_i,
        "rank": rank,
        "scene": scene,
        "probe": False,
        "round_eval": True,
        "arm": arm,
        "seq": seq,
        "success": float(success),
        "steps": int(steps),
        "rows": [],
        "sec": sec,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] collector-%(message)s"
    )
    log = logging.getLogger("pi05.collector")

    cfg = load_run_config(args.config)
    os.environ["RLT_COLLECTOR"] = "1"

    # Pin this rank to its slice of the fleet before anything imports MuJoCo:
    # prepare_environment publishes MUJOCO_EGL_DEVICE_ID / CUDA_VISIBLE_DEVICES
    # and the run config (whose port the policy worker dials) from cfg.
    endpoints = cfg.server_endpoints()
    egl_devices = [int(d) for d in cfg.egl_devices.split(",") if d.strip()]
    sharded = len(endpoints) > 1 or egl_devices
    if sharded:
        gpu, port = endpoints[args.rank % len(endpoints)]
        egl_device = egl_devices[args.rank % len(egl_devices)] if egl_devices else cfg.egl_device
        cfg = dataclasses.replace(cfg, gpu=gpu, port=port, egl_device=egl_device)

    from pi05.train import build, prepare_environment

    if sharded:
        prepare_environment(cfg, config_path=args.out / f"config_r{args.rank}.json")
    else:
        prepare_environment(cfg)
    out_dir = args.out
    online, agent, _, learner, policy, runner = build(cfg, out_dir, phase="online", with_runner=True)
    learner.updates = False
    policy.use_actor = True

    from rlt.cli import parse_episode_spec
    from rlt.replay import close_transitions
    from rlt.rollout import EpisodeRunner

    from pi05.policy import Pi05EvalConfig

    pool = parse_episode_spec(cfg.episode_pool) or [0]
    tasks = cfg.task_list()
    runners: dict[str, EpisodeRunner] = {}

    def runner_for(scene: str, horizon: int) -> EpisodeRunner:
        found = runners.get(scene)
        if found is None:
            if scene == cfg.scene and horizon == cfg.horizon:
                found = runner
            else:
                found = EpisodeRunner(
                    policy,
                    str(SCENES[scene].benchmark_train),
                    horizon,
                    online.tmp_rollout_dir,
                    eval_config_cls=Pi05EvalConfig,
                )
            runners[scene] = found
        return found

    counter = FileCounter(out_dir / COUNTER)
    eval_counter = FileCounter(out_dir / EVAL_COUNTER)
    # EGL slots cap concurrent MuJoCo contexts per device; with several devices
    # each gets its own lock dir, or the cap would be shared across GPUs.
    slot_dir = out_dir / EGL / f"dev{cfg.egl_device}" if egl_devices else out_dir / EGL
    slots = EglSlots(slot_dir, max(1, int(cfg.egl_slots)))
    weights = out_dir / WEIGHTS
    inbox = out_dir / INBOX
    seen_version = -1
    probe_n = int(cfg.probe_episodes)
    total = probe_n + int(cfg.episodes)
    round_eval = int(getattr(cfg, "round_eval_per_task", 0) or 0) > 0
    log.info(
        "rank=%d probe=%d online=%d tasks=%d port=%d egl=%d round_eval=%s",
        args.rank, probe_n, cfg.episodes, len(tasks), cfg.port, cfg.egl_device, round_eval,
    )

    def run_one(scene: str, horizon: int, env_index: int):
        nonlocal seen_version
        seen_version = load_live_weights(agent, weights, seen_version)
        learner.start_episode()
        handle = slots.acquire()
        started = time.time()
        try:
            rollout = runner_for(scene, horizon).run(pool[env_index % len(pool)])
        finally:
            slots.release(handle)
        return rollout, time.time() - started

    try:
        while True:
            cmd = read_cmd(out_dir)
            phase = cmd.get("phase", "run")
            if phase == "stop":
                write_heartbeat(out_dir, args.rank, "stop")
                break
            if phase == "hold":
                write_heartbeat(out_dir, args.rank, "hold")
                time.sleep(0.3)
                continue
            if phase == "round_eval":
                arm = str(cmd.get("arm") or "base_off")
                n_eval = int(cmd.get("n") or 0)
                seq = int(cmd.get("seq") or 0)
                cmd_now = read_cmd(out_dir)
                if (
                    cmd_now.get("phase") != "round_eval"
                    or str(cmd_now.get("arm") or "") != arm
                    or int(cmd_now.get("seq") or 0) != seq
                ):
                    write_heartbeat(out_dir, args.rank, "hold")
                    time.sleep(0.3)
                    continue
                apply_arm(policy, arm)
                eval_i = eval_counter.claim_below(n_eval, seq=seq)
                if eval_i is None:
                    write_heartbeat(out_dir, args.rank, "hold")
                    time.sleep(0.3)
                    continue
                cmd_now = read_cmd(out_dir)
                if (
                    cmd_now.get("phase") != "round_eval"
                    or str(cmd_now.get("arm") or "") != arm
                    or int(cmd_now.get("seq") or 0) != seq
                ):
                    log.warning(
                        "rank %d stale %s claim #%d (cmd now phase=%s arm=%s seq=%s); "
                        "returning an empty payload so the slot is not lost",
                        args.rank, arm, eval_i, cmd_now.get("phase"),
                        cmd_now.get("arm"), cmd_now.get("seq"),
                    )
                    scene, _horizon = tasks[eval_i % len(tasks)]
                    write_inbox(
                        inbox,
                        seq * 1000 + eval_i,
                        args.rank,
                        round_eval_payload(
                            eval_i, args.rank, scene, arm, seq, dropped=True,
                        ),
                    )
                    write_heartbeat(out_dir, args.rank, "hold")
                    time.sleep(0.3)
                    continue
                write_heartbeat(out_dir, args.rank, "eval", arm=arm, eval_i=eval_i)
                scene, horizon = tasks[eval_i % len(tasks)]
                try:
                    rollout, sec = run_one(scene, horizon, eval_i)
                except Exception:
                    log.exception(
                        "rank %d round-eval %s #%d %s crashed", args.rank, arm, eval_i, scene
                    )
                    write_inbox(
                        inbox,
                        seq * 1000 + eval_i,
                        args.rank,
                        round_eval_payload(
                            eval_i, args.rank, scene, arm, seq, error=True,
                        ),
                    )
                    continue
                if rollout is None:
                    log.warning("rank %d round-eval %s #%d produced nothing", args.rank, arm, eval_i)
                    write_inbox(
                        inbox,
                        seq * 1000 + eval_i,
                        args.rank,
                        round_eval_payload(
                            eval_i, args.rank, scene, arm, seq, empty=True,
                        ),
                    )
                    continue
                write_inbox(
                    inbox,
                    seq * 1000 + eval_i,
                    args.rank,
                    round_eval_payload(
                        eval_i, args.rank, scene, arm, seq,
                        success=rollout.success, steps=rollout.steps, sec=sec,
                    ),
                )
                log.info(
                    "rank %d round-eval %s #%d %s success=%d steps=%d %.0fs",
                    args.rank, arm, eval_i, scene, int(rollout.success), rollout.steps, sec,
                )
                continue

            apply_arm(policy, "run")
            limit = read_actor_limit(out_dir, default=total)
            episode = counter.claim_below(limit)
            if episode is None:
                write_heartbeat(out_dir, args.rank, "hold")
                if not round_eval and limit >= total:
                    break
                time.sleep(0.3)
                continue
            if episode >= total:
                break
            write_heartbeat(out_dir, args.rank, "actor", episode=episode)
            probing = episode < probe_n
            scene, horizon = tasks[episode % len(tasks)]
            # Walk layouts per task. episode % len(pool) ties layout to task and
            # only visits 8 of 48 train repeats; episode // n_tasks covers the pool.
            rollout, sec = run_one(scene, horizon, episode // len(tasks))
            if rollout is None:
                log.warning("rank %d episode %d produced nothing", args.rank, episode)
                continue
            rows = [] if probing else close_transitions(rollout, cfg.chunk_size, cfg.gamma, 0)[0]
            write_inbox(
                inbox,
                episode,
                args.rank,
                {
                    "episode": episode,
                    "rank": args.rank,
                    "scene": scene,
                    "probe": probing,
                    "success": float(rollout.success),
                    "steps": int(rollout.steps),
                    "rows": rows,
                    "sec": sec,
                },
            )
            log.info(
                "rank %d ep %d %s %s success=%d steps=%d %.0fs",
                args.rank,
                episode,
                scene,
                "probe" if probing else "actor",
                int(rollout.success),
                rollout.steps,
                sec,
            )
    finally:
        slots.close()


if __name__ == "__main__":
    main()
