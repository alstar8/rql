"""Phases 2-3 with several collectors feeding one learner.

    python -m rlt.train_parallel --episode_idx 134 --episodes 400 --collectors 4 \
        --ports 8000,8001,8002,8003 --encoder_devices cuda:0,cuda:1,cuda:2,cuda:3 \
        --device cuda:0 --out_dir runs/online_134

Same algorithm and same hyperparameters as `train_online`; K episodes simply run
at once. Episode numbers come from one shared counter, so warmup ends after
`warmup_episodes` episodes in total no matter who ran them, and the learner
keeps `utd` updates per stored transition however fast they arrive.

Deviation from the paper, made deliberately: Algorithm 1 has a single rollout
stream. The buffer is off-policy either way, so K collectors change the wall
clock and not the ratio of updates to data.
"""

from __future__ import annotations

import logging
import os
import queue as queue_module
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from .cli import describe, parse_into, save_config
from .config import ACTION_DIM, CHUNK, PROPRIO_DIM, OnlineConfig
from .logging_utils import RunLogger
from .mlspaces_env import configure_assets
from .train_online import checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rlt.train_parallel")

WEIGHTS = "actor_live.pt"

# Seconds between the learner's periodic duties. Module constants rather than
# literals so a test can set them to zero and exercise branches that would
# otherwise first run ten minutes into a job -- which is where a stale variable
# name sat undetected through a full test suite.
PUBLISH_EVERY = 5.0
CONSOLE_EVERY = 60.0
CHECKPOINT_EVERY = 600.0


def split(text: str, count: int) -> list[str]:
    """'a,b' with count=4 -> [a, b, a, b]; one entry per collector."""
    items = [part.strip() for part in text.split(",") if part.strip()]
    return [items[i % len(items)] for i in range(count)]


def collector_loop(rank: int, cfg: OnlineConfig, rows_queue, counter_value, ports, encoders, egls) -> None:
    """One robot: claim an episode, run it, ship what closed. No updates here."""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = egls[rank]
    configure_assets(cfg.assets_dir, cfg.cache_dir)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from .agent import RLTokenAgent
    from .eval_config import RLTokenEvalConfig
    from .replay import close_transitions
    from .rl_policy import RLTokenPolicy
    from .rollout import EpisodeRunner
    from .shared import EpisodeCounter, WeightLink
    from .vla import TokenEncoder, VlaClient

    torch.manual_seed(cfg.seed + rank)
    np.random.seed(cfg.seed + rank)

    vla = VlaClient(cfg.server_host, int(ports[rank]), cfg.server_timeout_sec)
    vla.wait_until_ready()
    encoder = TokenEncoder(cfg.token_ae, encoders[rank])

    # The actor is small; a collector keeps its own copy on the CPU and reloads
    # it from the learner's file whenever that file changes.
    local = OnlineConfig(**{**vars(cfg), "device": "cpu"})
    agent = RLTokenAgent(local, encoder.z_dim + PROPRIO_DIM, CHUNK * ACTION_DIM, CHUNK)
    weights = WeightLink(Path(cfg.out_dir) / WEIGHTS)
    counter = EpisodeCounter(counter_value)

    state = {"first": 0}

    def on_decision(policy) -> None:
        rows, state["first"] = close_transitions(
            policy.rollout_view(), CHUNK, cfg.gamma, state["first"]
        )
        if rows:
            rows_queue.put(("rows", rows))
        weights.refresh(agent.actor)

    policy = RLTokenPolicy(
        RLTokenEvalConfig(),
        vla=vla,
        encoder=encoder,
        agent=agent,
        chunk=CHUNK,
        stride=cfg.stride,
        explore=True,
        on_decision=on_decision,
    )
    runner = EpisodeRunner(policy, cfg.benchmark_dir, cfg.horizon, f"{cfg.tmp_rollout_dir}/w{rank}")
    pool = cfg.training_episodes()  # round-robin, so every scene gets the same share
    log.info(
        "collector %d ready: port %s, encoder %s, egl %s, %d training episode(s)",
        rank, ports[rank], encoders[rank], egls[rank], len(pool),
    )
    # Per-episode records are written once, by the learner, into episodes.jsonl.

    while True:
        episode = counter.claim()
        if episode >= cfg.episodes:
            break
        benchmark_idx = pool[episode % len(pool)]
        policy.use_actor = episode >= cfg.warmup_episodes
        state["first"] = 0
        weights.refresh(agent.actor)

        started = time.time()
        rollout = runner.run(benchmark_idx)
        if rollout is None:
            log.warning("collector %d: episode %d produced nothing", rank, episode)
            continue

        rows, state["first"] = close_transitions(rollout, CHUNK, cfg.gamma, state["first"])
        if rows:
            rows_queue.put(("rows", rows))
        rows_queue.put(
            (
                "episode",
                {
                    "episode": episode,
                    "benchmark_idx": benchmark_idx,
                    "collector": rank,
                    "phase": 0 if episode < cfg.warmup_episodes else 1,
                    "success": float(rollout.success),
                    "steps": rollout.steps,
                    "sec_per_episode": time.time() - started,
                    "vla_ms_per_call": 1000.0 * vla.seconds / max(vla.calls, 1),
                },
            )
        )
        log.info(
            "collector %d ep %d %s success=%d steps=%d %.0fs",
            rank, episode, "warmup" if episode < cfg.warmup_episodes else "actor",
            int(rollout.success), rollout.steps, time.time() - started,
        )
    rows_queue.put(("done", rank))


def main() -> None:
    cfg = parse_into(OnlineConfig)
    print(describe(cfg))
    configure_assets(cfg.assets_dir, cfg.cache_dir)

    from .agent import RLTokenAgent
    from .replay import ChunkReplay
    from .shared import WeightLink
    from .vla import token_z_dim

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.json")

    chunk_dim = CHUNK * ACTION_DIM
    state_dim = token_z_dim(cfg.token_ae) + PROPRIO_DIM
    agent = RLTokenAgent(cfg, state_dim, chunk_dim, CHUNK)
    buffer = ChunkReplay(cfg.buffer_capacity, state_dim, chunk_dim, cfg.seed)

    first_episode = 0
    if cfg.resume and (out_dir / "agent.pt").exists():
        import json

        agent.load(str(out_dir / "agent.pt"))
        buffer.load(str(out_dir / "buffer.npz"))
        first_episode = int(json.loads((out_dir / "progress.json").read_text())["episodes_done"])
        log.info("resumed at episode %d with %d transitions", first_episode, len(buffer))

    weights = WeightLink(out_dir / WEIGHTS)
    weights.publish(agent.actor)

    ports = split(cfg.ports or str(cfg.server_port), cfg.collectors)
    encoders = split(cfg.encoder_devices, cfg.collectors)
    egls = split(cfg.egl_devices, cfg.collectors)

    # CUDA in a child needs spawn; nothing is shared through memory because of it.
    context = mp.get_context("spawn")
    rows_queue = context.Queue(maxsize=4096)
    counter_value = context.Value("i", first_episode)
    workers = [
        context.Process(
            target=collector_loop,
            args=(rank, cfg, rows_queue, counter_value, ports, encoders, egls),
        )
        for rank in range(cfg.collectors)
    ]
    for worker in workers:
        worker.start()

    resumed = first_episode > 0
    logger = RunLogger(out_dir / "metrics.jsonl", out_dir / "tb", out_dir.name, append=resumed)
    logger.add_text("config", describe(cfg))
    episodes_log = RunLogger(out_dir / "episodes.jsonl", None, append=resumed)
    learn(cfg, agent, buffer, rows_queue, weights, logger, episodes_log, out_dir, len(workers))

    shutdown(workers, rows_queue)
    checkpoint(agent, buffer, out_dir, cfg.episodes)
    logger.close()
    episodes_log.close()


def shutdown(workers, rows_queue, grace: float = 30.0) -> None:
    """Leave no collector behind, and no GPU memory with it.

    Two things keep a collector alive after its work is done. A process that has
    put items on a queue does not exit until they are flushed, and the learner
    has stopped reading by then; and SIGTERM is swallowed by the MolmoSpaces
    rollout runner, which treats it as a request for a graceful stop. So: drain
    first, then wait, then SIGKILL whatever is left. Without this, eight
    collectors sat holding 24 GB of GPU memory long after the run was over.
    """
    deadline = time.time() + grace
    while time.time() < deadline and any(worker.is_alive() for worker in workers):
        try:
            rows_queue.get(timeout=0.1)
        except queue_module.Empty:
            pass

    for worker in workers:
        worker.join(timeout=5)
        if worker.is_alive():
            log.warning("collector %s ignored the shutdown; killing it", worker.pid)
            worker.kill()
            worker.join(timeout=10)

    rows_queue.close()
    rows_queue.cancel_join_thread()


def learn(
    cfg: OnlineConfig, agent, buffer, rows_queue, weights, logger, episodes_log, out_dir: Path, n_workers: int
) -> None:
    """Drain what the collectors send, then keep `utd` updates per transition.

    Logging is split on purpose. TensorBoard gets one point per finished
    episode, on a step that increases by exactly one, carrying rates and
    aggregates only. Raw per-episode records -- success 0/1, which collector,
    how long it took -- go to their own JSONL. Mixing the two put 246 backward
    steps into a 590-record run and made every aggregate unreadable.
    """
    done = success = finished = 0
    rows_total = 0
    recent: deque[int] = deque(maxlen=10)
    stats: dict[str, float] = {}
    last_console = last_save = last_publish = time.time()
    started = time.time()

    while finished < n_workers:
        pending = 0
        episodes: list[dict] = []
        for _ in range(256):  # bounded: a fast collector must not starve learning
            try:
                kind, payload = rows_queue.get(timeout=0.05)
            except queue_module.Empty:
                break
            if kind == "rows":
                buffer.extend(payload)
                pending += len(payload)
            elif kind == "episode":
                episodes.append(payload)
            else:
                finished += 1
        rows_total += pending

        if pending and len(buffer) >= cfg.batch_size:
            for _ in range(cfg.utd * pending):
                for _ in range(cfg.critic_updates_per_actor):
                    stats.update(agent.critic_step(buffer.sample(cfg.batch_size)))
                stats.update(agent.actor_step(buffer.sample(cfg.batch_size)))

        # Logged after the updates, never before: an episode that arrives in the
        # same pass as its own rows would otherwise report the state from before
        # any of them were learned from, and write a zero that is not one.
        for payload in episodes:
            done += 1
            success += int(payload["success"])
            recent.append(int(payload["success"]))
            episodes_log.log(payload["episode"], payload)
            log.info(
                "ep %d/%d (collector %d) success=%d steps=%d buffer=%d %.0fs",
                payload["episode"], cfg.episodes, payload["collector"], int(payload["success"]),
                payload["steps"], len(buffer), payload["sec_per_episode"],
            )
            metrics = {
                "sr_last10": float(np.mean(recent)),
                "sr_total": success / done,
                "episodes_success": success,
                "buffer": len(buffer),
                "buffer/reward_rows": buffer.reward_rows(),
                "buffer/action_coverage": buffer.action_coverage(),
                "critic_steps": agent.critic_steps,
                "actor_steps": agent.actor_steps,
                "updates_per_sec": agent.critic_steps / max(time.time() - started, 1e-9),
                # The update-to-data ratio, live. It has to sit at --utd.
                "updates_per_row": agent.critic_steps
                / max(rows_total * cfg.critic_updates_per_actor, 1),
                "queue_backlog": rows_queue.qsize(),
                **stats,
            }
            if len(buffer) >= cfg.batch_size:
                metrics.update(agent.probe(buffer.sample(cfg.batch_size)))
            logger.log(done, metrics)  # step grows by exactly one, never repeats

        now = time.time()
        if now - last_publish >= PUBLISH_EVERY:
            weights.publish(agent.actor)
            last_publish = now

        if now - last_console >= CONSOLE_EVERY:
            log.info(
                "learner: episodes %d/%d success %d (last10 %.2f) buffer %d rewarded %d "
                "cover %.4f updates/row %.2f backlog %d",
                done, cfg.episodes, success, float(np.mean(recent)) if recent else float("nan"),
                len(buffer), buffer.reward_rows(), buffer.action_coverage(),
                agent.critic_steps / max(rows_total * cfg.critic_updates_per_actor, 1),
                rows_queue.qsize(),
            )
            last_console = now

        if now - last_save >= CHECKPOINT_EVERY:
            checkpoint(agent, buffer, out_dir, done)
            last_save = now


if __name__ == "__main__":
    main()
