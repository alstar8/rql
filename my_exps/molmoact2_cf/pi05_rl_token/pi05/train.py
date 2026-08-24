"""RL-Token training (phases 2-3), driven entirely by an `RLConfig`.

    from pi05.config import RLConfig
    from pi05.train import train
    train(RLConfig(scene="house21", encoder="house21"))

or from the shell:

    scripts/run_train.py --scene house21 --encoder house21


WHAT GOES IN, AND WHAT COMES OUT
--------------------------------
The VLA is frozen throughout. Nothing in this file updates it; the only things that learn
are the actor and the two critic heads, which are small MLPs.

Per env step t, with C = `chunk_size`:

    t % C == 0     ask the frozen VLA. It returns
                     - a chunk of 16 actions, of which the first C will run
                     - its prefix tokens, (968, 2048), the model's reading of the scene
                   The phase-1 encoder compresses the tokens to z_rl; together with
                   proprioception (7 joint positions + gripper + their velocities) that
                   is the RL state x. The VLA's own chunk is the reference a_ref.

                   Before `gate_step` (if > 0), or during warmup, the VLA's chunk is
                   executed and the decision is merely recorded. After both, the actor's
                   sample is executed instead. The default `gate_step=0` means RL drives
                   from the first env step.

    always         execute the committed action for this step, converted from a joint
                   delta into the absolute target the simulator wants (pi05/model.py).

A transition closes C steps after the decision that opened it:

    <x, a, a_ref, r, x'>      r = sum of gamma^i over the window, which is +1 exactly
                              once, on the step the task first reports success

and `utd` learner iterations run per stored transition, each being
`critic_updates_per_actor` critic updates and one actor update (rlt/learner.py).

STRIDE
------
One decision per executed chunk. The paper subsamples with a stride of 2, which stores
overlapping windows spliced from two committed chunks and pairs them with a reference the
robot never executed. At stride == C every stored row is one decision: the action is the
chunk that was chosen, the reference is what the VLA proposed for that same state, and
nothing is stitched.

TRAINING AND EVALUATION MUST AGREE
----------------------------------
`chunk_size` and `conversion` are copied into the run directory and must match the
evaluation that judges the result. The first RL matrix trained at chunk 8 and was
evaluated at chunk 1, and since chunk size alone moves the success rate by 30-50 points
in either direction depending on the scene, "RL helped" could not be separated from "the
chunk changed". That run was discarded.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from .config import (
    ACTION_DIM,
    PROPRIO_DIM,
    TORCH_DEVICE,
    RLConfig,
    TMP_ROLLOUT_DIR,
    save_run_config,
)
from .eval import prepare_environment as _prepare_eval_environment
from .eval import start_server, stop_server

log = logging.getLogger("pi05.train")


def prepare_environment(cfg: RLConfig) -> Path:
    """Same as the evaluation side: set what MolmoSpaces reads at import time.

    Must run before anything imports molmo_spaces or torch.
    """
    return _prepare_eval_environment(cfg)


def online_config(cfg: RLConfig, out_dir: Path, *, phase: str = "online"):
    """Translate an RLConfig into the rlt hyperparameter record.

    The values the paper fixes (utd, critic updates per actor, hidden width, reference
    dropout, the clipped TD target) keep their rlt defaults; only what this project
    decides is overridden here. rlt/config.py documents each one and says whether it came
    from the paper or from the project owner.
    """
    from rlt.config import OnlineConfig

    train_token = cfg.train_token_offline if phase == "pretrain" else cfg.train_token_online
    cadence = 0 if phase == "pretrain" else int(cfg.update_every_steps)
    return OnlineConfig(
        benchmark_dir=str(cfg.benchmark_dir()),
        episode_pool=cfg.episode_pool,
        horizon=cfg.horizon,
        episodes=cfg.episodes,
        warmup_episodes=cfg.warmup_episodes,
        token_ae=str(cfg.token_ae_path()),
        out_dir=str(out_dir),
        stride=cfg.resolved_stride(),
        sigma=cfg.sigma,
        beta=cfg.beta,
        gamma=cfg.gamma,
        utd=cfg.utd,
        device=TORCH_DEVICE,
        seed=cfg.seed,
        server_host="127.0.0.1",
        server_port=cfg.port,
        tmp_rollout_dir=str(TMP_ROLLOUT_DIR / "rl"),
        algorithm=cfg.algorithm,
        train_token=train_token,
        ae_finetune=cfg.ae_finetune,
        store_decision_tokens=cfg.store_decision_tokens,
        collectors=cfg.collectors,
        update_every_steps=cadence,
        rlt_critic=cfg.rlt_critic,
        flow_actor_coef=cfg.flow_actor_coef,
        flow_bc_coef=cfg.flow_bc_coef,
        flow_freeze_critic_online=cfg.flow_freeze_critic_online,
        flow_compose=cfg.flow_compose,
    )


def build(cfg: RLConfig, out_dir: Path, *, phase: str = "online", with_runner: bool = True):
    """Encoder, agent, buffer, learner, and optionally a MuJoCo runner.

    ``with_runner=False`` is the parallel-collector parent: it owns the GPU
    learner but must not open a MuJoCo context, or it would steal an EGL slot.
    """
    import torch

    from rlt.consensusflow import make_agent
    from rlt.learner import Learner
    from rlt.replay import ChunkReplay
    from rlt.token_ae import RLTokenAE, write_random_ae
    from rlt.vla import TokenEncoder

    from .policy import Pi05EvalConfig, Pi05RLPolicy
    from .rl_token import RLTokenCorrector, StepGate

    if cfg.init_random_ae:
        ae_path = Path(cfg.token_ae) if cfg.token_ae else cfg.token_ae_path()
        if not Path(ae_path).exists():
            write_random_ae(str(ae_path), token_dim=2048)

    online = online_config(cfg, out_dir, phase=phase)
    device = "cpu" if os.environ.get("RLT_COLLECTOR") == "1" else (
        online.device if torch.cuda.is_available() else "cpu"
    )
    online.device = device

    encoder = TokenEncoder(online.token_ae, device)
    state_dim = encoder.z_dim + PROPRIO_DIM
    chunk_dim = cfg.chunk_size * ACTION_DIM

    token_ae = None
    if cfg.algorithm in ("consensusflow", "flow_rlt") or cfg.ae_finetune or online.train_token:
        token_ae = RLTokenAE.load(online.token_ae, map_location=device)
    agent = make_agent(online, state_dim, chunk_dim, cfg.chunk_size, token_ae=token_ae)
    if not online.train_token and hasattr(agent, "freeze_token"):
        agent.freeze_token()
    if cfg.ae_finetune and hasattr(agent, "set_ae_finetune"):
        agent.set_ae_finetune(True)
    if hasattr(agent, "set_critic_frozen"):
        # flow_rlt with a learned critic: TD trains offline, and online only when the
        # arm keeps the critic trainable. An external rlt_critic stays frozen either way.
        agent.set_critic_frozen(phase == "online" and cfg.flow_freeze_critic_online)

    buffer = ChunkReplay(online.buffer_capacity, state_dim, chunk_dim, cfg.seed)
    learner = Learner(online, agent, buffer, cfg.chunk_size)
    log.info(
        "state_dim=%d (z %d + proprio %d), chunk_dim=%d (C=%d x %d dims) algorithm=%s "
        "collectors=%d egl_slots=%d update_every_steps=%d",
        state_dim, encoder.z_dim, PROPRIO_DIM, chunk_dim, cfg.chunk_size, ACTION_DIM,
        cfg.algorithm, cfg.collectors, cfg.egl_slots, online.update_every_steps,
    )
    if not with_runner:
        return online, agent, buffer, learner, None, None

    from rlt.rollout import EpisodeRunner

    corrector = RLTokenCorrector(
        agent,
        encoder,
        cfg.chunk_size,
        StepGate(cfg.resolved_gate_step(), cfg.gate_latch),
        action_space=cfg.rl_action_space,
        explore=True,
    )
    policy = Pi05RLPolicy(
        Pi05EvalConfig(),
        corrector=corrector,
        on_decision=learner.on_decision,
    )
    runner = EpisodeRunner(
        policy,
        str(cfg.benchmark_dir()),
        cfg.horizon,
        online.tmp_rollout_dir,
        eval_config_cls=Pi05EvalConfig,
    )
    return online, agent, buffer, learner, policy, runner


def is_replay_file(cfg: RLConfig) -> bool:
    """True when `vla_traj` points at a saved replay npz, so pretrain can skip VLA collect."""
    return bool(cfg.vla_traj) and Path(cfg.vla_traj).is_file()


def _metrics_progress(path: Path) -> tuple[int, int, list[float]]:
    """Stored actor episodes, successes, and the success series from metrics.jsonl."""
    n = successes = 0
    series: list[float] = []
    if not path.exists():
        return 0, 0, series
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        n += 1
        value = float(row.get("success", 0.0))
        successes += int(value)
        series.append(value)
    return n, successes, series


def attach_token_replay(agent, pattern: str, max_sequences: int | None = 6000) -> int:
    """Load token shards onto a ConsensusFlow agent for L_ro when the replay has no tokens."""
    if not pattern:
        return 0
    from glob import glob

    from rlt.data import TokenSequences

    paths = sorted(Path(p) for p in glob(pattern))
    if not paths:
        raise FileNotFoundError(f"token_replay matched nothing: {pattern}")
    corpus = TokenSequences.load(paths, max_sequences=max_sequences)
    agent.token_corpus = corpus
    return len(corpus)


def scoreable(episodes_done: int, warmup_episodes: int, window: int) -> bool:
    """Can the rolling success rate name a best actor yet?

    Only once the whole window is the actor's own episodes. During warmup the gate never
    opens, so sr_last10 measures the frozen VLA while the actor is still untrained; a
    "best" picked there saves a barely-trained actor and freezes it. The multi-task run
    recorded 0.90 at episode 10 that way, and would have been evaluated on noise. The
    window straddling the handover is mixed and just as misleading, so it waits for a
    clean one.
    """
    return episodes_done - warmup_episodes >= window


def checkpoint(agent, buffer, out_dir: Path, episodes_done: int, recent: float | None = None) -> None:
    """Networks and buffer together, so a resume picks up where this left off.

    Also keeps a best-so-far copy. Overwriting one agent.pt is destructive whenever a run
    peaks and then degrades, which RL does: atomizer x ae_combined reached rolling SR 0.80
    at episode 239 and fell to 0.10 by 299, so the only checkpoint left was the collapsed
    one and the peak could not be evaluated at all.
    """
    agent.save(str(out_dir / "agent.pt"))
    buffer.save(str(out_dir / "buffer.npz"))
    progress = {"episodes_done": episodes_done}

    if recent is not None:
        best_path = out_dir / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else {"sr": -1.0}
        if recent > best["sr"]:
            agent.save(str(out_dir / "agent_best.pt"))
            best_path.write_text(json.dumps({"sr": recent, "episode": episodes_done}))
        progress["best"] = best_path.exists() and json.loads(best_path.read_text())
    (out_dir / "progress.json").write_text(json.dumps(progress))


def train_parallel(cfg: RLConfig) -> dict:
    """Parent learner for subprocess collectors. No MuJoCo in this process."""
    import torch

    from rlt.logging_utils import RunLogger

    from .collectors import COUNTER, INBOX, WEIGHTS, read_inbox, start_workers, stop_workers

    out_dir = cfg.run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, out_dir / "config.json")
    inbox = out_dir / INBOX
    inbox.mkdir(exist_ok=True)
    for stale in inbox.glob("*.pkl"):
        stale.unlink()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    server = start_server(cfg, out_dir / "server.log")
    summary: dict = {"scene": cfg.scene, "encoder": cfg.encoder, "chunk_size": cfg.chunk_size}
    started = time.time()
    workers = []
    try:
        online, agent, buffer, learner, _, _ = build(cfg, out_dir, phase="online", with_runner=False)
        live_path = out_dir / WEIGHTS
        saved_path = out_dir / "agent.pt"
        replay_path = out_dir / "buffer.npz"
        if cfg.resume:
            if live_path.exists():
                log.info("resume: loading live actor %s", live_path)
                agent.load(str(live_path))
            elif saved_path.exists():
                log.info("resume: loading checkpoint %s", saved_path)
                agent.load(str(saved_path))
            elif cfg.init_actor:
                log.info("loading pretrained actor-critic %s", cfg.init_actor)
                agent.load(cfg.init_actor)
            else:
                raise FileNotFoundError(f"resume requested but no {WEIGHTS} or agent.pt in {out_dir}")
            if replay_path.exists():
                log.info("resume: loading replay %s", replay_path)
                buffer.load(str(replay_path))
            elif cfg.init_buffer:
                log.info("loading pretrained replay %s", cfg.init_buffer)
                buffer.load(cfg.init_buffer)
        else:
            if cfg.init_actor:
                log.info("loading pretrained actor-critic %s", cfg.init_actor)
                agent.load(cfg.init_actor)
            if cfg.init_buffer:
                log.info("loading pretrained replay %s", cfg.init_buffer)
                buffer.load(cfg.init_buffer)
        if cfg.token_replay:
            n_tok = attach_token_replay(agent, cfg.token_replay)
            log.info("token_replay %s -> %d sequences", cfg.token_replay, n_tok)
        agent.save(str(out_dir / WEIGHTS))
        logger = RunLogger(
            out_dir / "metrics.jsonl", out_dir / "tb", out_dir.name, append=bool(cfg.resume)
        )
        window = 10
        recent: deque[float] = deque(maxlen=window)
        probe_n = int(cfg.probe_episodes)
        probe_done = actor_episodes = actor_successes = probe_successes = 0
        if cfg.resume:
            actor_episodes, actor_successes, past = _metrics_progress(out_dir / "metrics.jsonl")
            if actor_episodes <= 0:
                raise RuntimeError(f"resume requested but {out_dir / 'metrics.jsonl'} is empty")
            recent.extend(past[-window:])
            probe_done = probe_n
            log.info(
                "resume at actor episode %d/%d success %d sr_last%d=%.2f",
                actor_episodes, cfg.episodes, actor_successes, window,
                float(np.mean(recent)) if recent else float("nan"),
            )
        (out_dir / COUNTER).write_text(
            str(probe_n + actor_episodes if cfg.resume else 0)
        )
        workers = start_workers(cfg, out_dir, cfg.collectors)
        log.info(
            "online: %d collectors, %d EGL slots, AC every %d stored env steps, prefill %d rows",
            cfg.collectors, cfg.egl_slots, cfg.update_every_steps, len(buffer),
        )
        last_console = last_save = time.time()
        while actor_episodes < cfg.episodes:
            payloads = read_inbox(inbox)
            if not payloads:
                if all(p.poll() is not None for p in workers):
                    # Collectors exit before the last pickles are visible on NFS.
                    time.sleep(0.5)
                    payloads = read_inbox(inbox)
                    if not payloads:
                        raise RuntimeError("all collectors exited before finishing episodes")
                else:
                    time.sleep(0.2)
                    continue
            for payload in payloads:
                if payload.get("probe"):
                    probe_done += 1
                    probe_successes += int(payload["success"])
                    recent.append(float(payload["success"]))
                    log.info(
                        "probe %d/%d (collector %d) success=%d steps=%d %.0fs sr_last%d=%.2f",
                        probe_done, probe_n, payload["rank"], int(payload["success"]),
                        payload["steps"], payload["sec"], window,
                        float(np.mean(recent)) if recent else float("nan"),
                    )
                    continue
                learner.absorb(payload.get("rows") or [])
                actor_episodes += 1
                actor_successes += int(payload["success"])
                recent.append(float(payload["success"]))
                metrics = {
                    "episode": payload["episode"],
                    "phase": 1,
                    "success": float(payload["success"]),
                    "steps": payload["steps"],
                    "buffer": len(buffer),
                    "buffer/reward_rows": buffer.reward_rows(),
                    "sr_last10": float(np.mean(recent)) if recent else float("nan"),
                    "critic_steps": agent.critic_steps,
                    "actor_steps": agent.actor_steps,
                    "sec_per_episode": payload["sec"],
                    **learner.stats,
                }
                if len(buffer) >= online.batch_size:
                    metrics.update(agent.probe(buffer.sample(online.batch_size)))
                logger.log(actor_episodes, metrics)
                log.info(
                    "ep %d/%d (collector %d) success=%d steps=%d buffer=%d sr_last%d=%.2f %.0fs",
                    actor_episodes, cfg.episodes, payload["rank"], int(payload["success"]),
                    payload["steps"], len(buffer), window, metrics["sr_last10"],
                    payload["sec"],
                )
                agent.save(str(out_dir / WEIGHTS))
                if actor_episodes % 10 == 0 or actor_episodes == cfg.episodes:
                    scored = float(np.mean(recent)) if scoreable(actor_episodes, 0, recent.maxlen) else None
                    checkpoint(agent, buffer, out_dir, actor_episodes, scored)
            now = time.time()
            if now - last_console >= 60:
                log.info(
                    "learner: episodes %d/%d success %d (last%d %.2f) buffer %d",
                    actor_episodes, cfg.episodes, actor_successes, window,
                    float(np.mean(recent)) if recent else float("nan"), len(buffer),
                )
                last_console = now
            if now - last_save >= 600:
                checkpoint(agent, buffer, out_dir, actor_episodes)
                last_save = now
        checkpoint(agent, buffer, out_dir, cfg.episodes)
        logger.close()
        summary.update(
            warmup_episodes=0,
            warmup_successes=0,
            warmup_rate=0.0,
            probe_episodes=probe_done,
            probe_successes=probe_successes,
            actor_episodes=actor_episodes,
            actor_successes=actor_successes,
            actor_rate=actor_successes / max(actor_episodes, 1),
            sr_last10=float(np.mean(recent)) if recent else float("nan"),
            agent=str(out_dir / "agent.pt"),
        )
    finally:
        stop_workers(workers)
        stop_server(server)
        summary["hours"] = round((time.time() - started) / 3600, 2)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def train(cfg: RLConfig) -> dict:
    """Run one RL-Token training job and return its summary."""
    problem = cfg.validate()
    if problem:
        raise ValueError(f"this RLConfig cannot be run: {problem}")
    if cfg.resolved_stride() != cfg.chunk_size:
        raise ValueError(
            f"stride {cfg.resolved_stride()} != chunk_size {cfg.chunk_size}. This trainer "
            "records one decision per executed chunk; a shorter stride would splice two "
            "committed chunks into one stored row and pair it with a reference the robot "
            "never executed. See the module docstring."
        )
    if cfg.collectors > 1:
        return train_parallel(cfg)

    import torch

    from rlt.logging_utils import RunLogger

    out_dir = cfg.run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, out_dir / "config.json")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    server = start_server(cfg, out_dir / "server.log")
    summary: dict = {"scene": cfg.scene, "encoder": cfg.encoder, "chunk_size": cfg.chunk_size}
    started = time.time()
    try:
        online, agent, buffer, learner, policy, runner = build(cfg, out_dir, phase="online")
        if cfg.init_actor:
            log.info("loading pretrained actor-critic %s", cfg.init_actor)
            agent.load(cfg.init_actor)
        if cfg.init_buffer:
            log.info("loading pretrained replay %s", cfg.init_buffer)
            buffer.load(cfg.init_buffer)
        if cfg.token_replay:
            n_tok = attach_token_replay(agent, cfg.token_replay)
            log.info("token_replay %s -> %d sequences", cfg.token_replay, n_tok)
        logger = RunLogger(out_dir / "metrics.jsonl", out_dir / "tb", out_dir.name)

        from rlt.cli import parse_episode_spec

        pool = parse_episode_spec(cfg.episode_pool) or [0]
        recent: deque[float] = deque(maxlen=10)
        warm_successes = actor_successes = actor_episodes = 0

        for episode in range(cfg.episodes):
            warming = episode < cfg.warmup_episodes
            policy.use_actor = not warming
            learner.start_episode()

            episode_started = time.time()
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
                "corrections": policy.corrector.corrections if policy.corrector else 0,
                "sr_last10": float(np.mean(recent)),
                "critic_steps": agent.critic_steps,
                "actor_steps": agent.actor_steps,
                "sec_per_episode": time.time() - episode_started,
                "vla_ms_per_call": policy.client.ms_per_call,
                **learner.stats,
            }
            if len(buffer) >= online.batch_size:
                metrics.update(agent.probe(buffer.sample(online.batch_size)))
            logger.log(episode, metrics)
            log.info(
                "ep %d/%d %s success=%d steps=%d buffer=%d sr_last10=%.2f q=%.3f gap=%.4f %.0fs",
                episode, cfg.episodes, "warmup" if warming else "actor",
                int(rollout.success), rollout.steps, len(buffer), metrics["sr_last10"],
                metrics.get("probe/q_actor", float("nan")),
                metrics.get("probe/q_gap", float("nan")),
                metrics["sec_per_episode"],
            )

            if (episode + 1) % 10 == 0 or episode + 1 == cfg.episodes:
                # sr over the recent window, so the "best" copy tracks the peak rather
                # than a single lucky episode. See scoreable() for why warmup is excluded.
                clean = scoreable(episode + 1, cfg.warmup_episodes, recent.maxlen)
                scored = float(np.mean(recent)) if not warming and clean else None
                checkpoint(agent, buffer, out_dir, episode + 1, scored)

        checkpoint(agent, buffer, out_dir, cfg.episodes)
        logger.close()
        summary.update(
            warmup_episodes=cfg.warmup_episodes,
            warmup_successes=warm_successes,
            warmup_rate=warm_successes / max(cfg.warmup_episodes, 1),
            actor_episodes=actor_episodes,
            actor_successes=actor_successes,
            actor_rate=actor_successes / max(actor_episodes, 1),
            sr_last10=float(np.mean(recent)) if recent else float("nan"),
            agent=str(out_dir / "agent.pt"),
        )
    finally:
        stop_server(server)
        summary["hours"] = round((time.time() - started) / 3600, 2)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info(
        "done in %.1f h | warmup %d/%d = %.1f%% | actor %d/%d = %.1f%%",
        summary["hours"],
        summary.get("warmup_successes", 0), cfg.warmup_episodes,
        100 * summary.get("warmup_rate", 0.0),
        summary.get("actor_successes", 0), summary.get("actor_episodes", 0),
        100 * summary.get("actor_rate", 0.0),
    )
    log.info(
        "the warmup rate is the frozen VLA measured in place; compare the actor against "
        "it, then confirm with scripts/run_eval.py at the SAME chunk_size and conversion"
    )
    return summary


def pretrain_actor_critic(cfg: RLConfig, episodes: int, offline_steps: int) -> dict:
    """Fill replay with frozen-VLA rollouts, then BC the actor and TD the critic.

    Online training can then start at episode 0 with the actor in control (warmup 0).
    The collect is the scene's train bench, same poses the online pool will see.
    """
    problem = cfg.validate()
    if problem:
        raise ValueError(f"this RLConfig cannot be run: {problem}")

    import torch

    from rlt.cli import parse_episode_spec
    from rlt.logging_utils import RunLogger

    out_dir = cfg.run_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(cfg, out_dir / "config.json")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    replay_only = is_replay_file(cfg)
    server = None if replay_only else start_server(cfg, out_dir / "server.log")
    summary: dict = {
        "scene": cfg.scene,
        "encoder": cfg.encoder,
        "pretrain_episodes": episodes,
        "offline_steps": offline_steps,
        "vla_traj": cfg.vla_traj,
        "replay_only": replay_only,
    }
    started = time.time()
    try:
        online, agent, buffer, learner, policy, runner = build(
            cfg, out_dir, phase="pretrain", with_runner=not replay_only
        )
        if policy is not None:
            policy.use_actor = False
        learner.updates = False
        logger = RunLogger(out_dir / "metrics.jsonl", out_dir / "tb", out_dir.name)
        successes = 0

        if replay_only:
            log.info("loading frozen-VLA replay %s (skip collect)", cfg.vla_traj)
            buffer.load(cfg.vla_traj)
            successes = buffer.reward_rows()
            log.info(
                "replay loaded: %d rows, %d reward rows",
                len(buffer), buffer.reward_rows(),
            )
        else:
            pool = parse_episode_spec(cfg.episode_pool) or [0]
            for episode in range(episodes):
                learner.start_episode()
                rollout = runner.run(pool[episode % len(pool)])
                if rollout is None:
                    log.warning("pretrain episode %d produced nothing; skipping", episode)
                    continue
                rows = learner.finish_episode(rollout)
                successes += int(rollout.success)
                log.info(
                    "pretrain ep %d/%d success=%d steps=%d rows=%d buffer=%d reward_rows=%d",
                    episode, episodes, int(rollout.success), rollout.steps, rows,
                    len(buffer), buffer.reward_rows(),
                )

        if cfg.token_replay:
            n_tok = attach_token_replay(agent, cfg.token_replay)
            log.info("token_replay %s -> %d sequences", cfg.token_replay, n_tok)

        if len(buffer) < online.batch_size:
            raise RuntimeError(
                f"pretrain buffer has {len(buffer)} rows, need {online.batch_size} for a batch"
            )

        stats: dict[str, float] = {}
        for step in range(1, offline_steps + 1):
            batch = buffer.sample(online.batch_size)
            stats.update(agent.critic_step(batch))
            stats.update(agent.actor_bc_step(batch))
            if step == 1 or step % 100 == 0 or step == offline_steps:
                stats["buffer"] = float(len(buffer))
                logger.log(step, stats)
                log.info(
                    "offline %d/%d critic=%.4f q=%.3f bc=%.4f rmse=%.4f",
                    step, offline_steps,
                    stats.get("critic_loss", float("nan")),
                    stats.get("q_mean", float("nan")),
                    stats.get("actor_bc_loss", float("nan")),
                    stats.get("actor_bc_rmse", float("nan")),
                )

        agent.save(str(out_dir / "agent.pt"))
        buffer.save(str(out_dir / "buffer.npz"))
        summary.update(
            successes=successes,
            success_rate=successes / max(episodes, 1),
            buffer_rows=len(buffer),
            reward_rows=buffer.reward_rows(),
            agent=str(out_dir / "agent.pt"),
            buffer=str(out_dir / "buffer.npz"),
        )
        logger.close()
    finally:
        if server is not None:
            stop_server(server)
        summary["hours"] = round((time.time() - started) / 3600, 2)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    log.info(
        "pretrain done in %.1f h | frozen collect %d/%d = %.1f%% | buffer %d rows",
        summary["hours"], successes, episodes, 100 * summary["success_rate"],
        summary.get("buffer_rows", 0),
    )
    return summary
