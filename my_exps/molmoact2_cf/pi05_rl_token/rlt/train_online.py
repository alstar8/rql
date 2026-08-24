"""Phases 2-3 of RL Token: online actor-critic on one benchmark episode.

    python -m rlt.train_online --episode_idx 137 --episodes 400

The first `warmup_episodes` rollouts execute the VLA's own chunks (Algorithm 1's
N_warm); the actor takes over afterwards. Learning runs from the first full
batch onwards, between env steps, exactly where Algorithm 1 puts it.

Everything lands in --out_dir: metrics.jsonl, TensorBoard, actor/critic
checkpoints and the resolved config.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .cli import describe, parse_into, save_config
from .config import ACTION_DIM, CHUNK, PROPRIO_DIM, OnlineConfig
from .logging_utils import RunLogger
from .mlspaces_env import configure_assets

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rlt.train_online")


def checkpoint(agent, buffer, out_dir: Path, episodes_done: int) -> None:
    """Networks and buffer together, so --resume picks up where this left off."""
    agent.save(str(out_dir / "agent.pt"))
    buffer.save(str(out_dir / "buffer.npz"))
    (out_dir / "progress.json").write_text(json.dumps({"episodes_done": episodes_done}))


def main() -> None:
    cfg = parse_into(OnlineConfig)
    print(describe(cfg))
    configure_assets(cfg.assets_dir, cfg.cache_dir)

    # MolmoSpaces reads its asset paths at import time, so these come after the
    # call above and not at the top of the file.
    from .agent import RLTokenAgent
    from .eval_config import RLTokenEvalConfig
    from .learner import Learner
    from .replay import ChunkReplay
    from .rl_policy import RLTokenPolicy
    from .rollout import EpisodeRunner
    from .vla import TokenEncoder, VlaClient

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.json")

    vla = VlaClient(cfg.server_host, cfg.server_port, cfg.server_timeout_sec)
    log.info("VLA server: %s", vla.wait_until_ready())
    encoder = TokenEncoder(cfg.token_ae, cfg.device)

    chunk_dim = CHUNK * ACTION_DIM
    state_dim = encoder.z_dim + PROPRIO_DIM
    agent = RLTokenAgent(cfg, state_dim, chunk_dim, CHUNK)
    buffer = ChunkReplay(cfg.buffer_capacity, state_dim, chunk_dim, cfg.seed)
    learner = Learner(cfg, agent, buffer, CHUNK)
    log.info("state_dim=%d (z %d + proprio %d), chunk_dim=%d", state_dim, encoder.z_dim, PROPRIO_DIM, chunk_dim)

    first_episode = 0
    if cfg.resume and (out_dir / "agent.pt").exists():
        agent.load(str(out_dir / "agent.pt"))
        buffer.load(str(out_dir / "buffer.npz"))
        first_episode = int(json.loads((out_dir / "progress.json").read_text())["episodes_done"])
        log.info("resumed at episode %d with %d transitions", first_episode, len(buffer))

    policy = RLTokenPolicy(
        RLTokenEvalConfig(),
        vla=vla,
        encoder=encoder,
        agent=agent,
        chunk=CHUNK,
        stride=cfg.stride,
        explore=True,
        on_decision=learner.on_decision,
    )
    runner = EpisodeRunner(policy, cfg.benchmark_dir, cfg.horizon, cfg.tmp_rollout_dir)

    logger = RunLogger(out_dir / "metrics.jsonl", out_dir / "tb", out_dir.name, append=first_episode > 0)
    logger.add_text("config", describe(cfg))

    recent = deque(maxlen=10)
    warm_successes = 0
    actor_successes = 0
    actor_episodes = 0
    start = time.time()

    for episode in range(first_episode, cfg.episodes):
        warming = episode < cfg.warmup_episodes
        policy.use_actor = not warming
        learner.start_episode()

        pool = cfg.training_episodes()  # round-robin, so every scene gets the same share
        t0 = time.time()
        rollout = runner.run(pool[episode % len(pool)])
        if rollout is None:
            log.warning("episode %d produced nothing; retrying", episode)
            continue
        rows = learner.finish_episode(rollout)

        recent.append(float(rollout.success))
        if warming:
            warm_successes += int(rollout.success)
        else:
            actor_episodes += 1
            actor_successes += int(rollout.success)

        metrics = {
            "episode": episode,
            "phase": 0 if warming else 1,
            "success": float(rollout.success),
            "steps": rollout.steps,
            "rows_last_close": rows,
            "buffer": len(buffer),
            "buffer/reward_rows": buffer.reward_rows(),
            "buffer/action_coverage": buffer.action_coverage(),
            "sr_last10": float(np.mean(recent)),
            "critic_steps": agent.critic_steps,
            "actor_steps": agent.actor_steps,
            "sec_per_episode": time.time() - t0,
            "vla_ms_per_call": 1000.0 * vla.seconds / max(vla.calls, 1),
            **learner.stats,
        }
        if len(buffer) >= cfg.batch_size:
            metrics.update(agent.probe(buffer.sample(cfg.batch_size)))
        logger.log(episode, metrics)

        log.info(
            "ep %d/%d %s success=%d steps=%d buffer=%d rewarded=%d cover=%.4f "
            "sr_last10=%.2f q=%.3f gap=%.4f dev=%.4f %.0fs",
            episode,
            cfg.episodes,
            "warmup" if warming else "actor",
            int(rollout.success),
            rollout.steps,
            len(buffer),
            buffer.reward_rows(),
            buffer.action_coverage(),
            metrics["sr_last10"],
            metrics.get("probe/q_actor", float("nan")),
            metrics.get("probe/q_gap", float("nan")),
            metrics.get("probe/actor_deviation", float("nan")),
            metrics["sec_per_episode"],
        )

        if (episode + 1) % cfg.save_every_episodes == 0 or episode + 1 == cfg.episodes:
            checkpoint(agent, buffer, out_dir, episode + 1)

    checkpoint(agent, buffer, out_dir, cfg.episodes)
    warm = cfg.warmup_episodes
    log.info(
        "done in %.1f h | warmup %d/%d = %.1f%% | actor %d/%d = %.1f%%",
        (time.time() - start) / 3600,
        warm_successes,
        warm,
        100.0 * warm_successes / max(warm, 1),
        actor_successes,
        actor_episodes,
        100.0 * actor_successes / max(actor_episodes, 1),
    )
    logger.close()


if __name__ == "__main__":
    main()
