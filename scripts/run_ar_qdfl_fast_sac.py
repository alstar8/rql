"""Dedicated three-phase runner for AR-QDFL FastSAC (ConsensusFlowRL).

Phases (exact counters, resumable):
  1) offline: ``offline_steps`` joint updates via ``agent.offline_update``
  2) warmup:  ``warmup_updates`` critic-only updates on offline-prefilled replay
              (actor/teacher frozen); diagnostics go to ``warmup.csv`` only
  3) online:  ``online_steps`` env interactions into a separate online replay,
              stratified offline/online batches, UTD critic + delayed actor

``eval.csv`` plot steps: offline ``0..offline_steps`` and
``offline_steps + online_env_step`` online. Warmup is absent from eval.csv;
an immediate post-warmup eval is written at ``offline_steps + 1`` (so it does
not overwrite the offline-end row at ``offline_steps``).

Does not modify generic ``main.py`` or agent mathematical losses.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.ar_qdfl_fast_sac_plumbing import (
    DEFAULT_OFFLINE_STEPS,
    DEFAULT_ONLINE_STEPS,
    DEFAULT_WARMUP_UPDATES,
    PHASE_OFFLINE,
    PHASE_ONLINE,
    PHASE_WARMUP,
    OnlineTransitionJournal,
    example_transition_from_dataset,
    load_resume_blobs,
    load_runner_state,
    plot_step,
    sample_stratified_batch,
    save_resume_blobs,
    save_runner_state,
)
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import (
    CsvLogger,
    TensorboardLogger,
    get_exp_name,
    get_flag_dict,
    prepare_eval_video,
    setup_experiment_logging,
)

FLAGS = flags.FLAGS

flags.DEFINE_string("run_group", "Debug", "Run group.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("env_name", "", "Environment (dataset) name.")
flags.DEFINE_string(
    "ogbench_data_dir",
    None,
    "Local OGBench dataset cache directory.",
)
flags.DEFINE_string("save_dir", "exp/", "Save directory root.")
flags.DEFINE_string(
    "resume_dir",
    None,
    "Existing experiment directory to resume. Default: fresh run under save_dir.",
)
flags.DEFINE_integer("offline_steps", DEFAULT_OFFLINE_STEPS, "Offline joint updates.")
flags.DEFINE_integer(
    "warmup_updates",
    DEFAULT_WARMUP_UPDATES,
    "Hidden critic-only warmup updates.",
)
flags.DEFINE_integer(
    "online_steps",
    DEFAULT_ONLINE_STEPS,
    "Online environment interactions.",
)
flags.DEFINE_integer("offline_buffer_size", 3_100_000, "Offline replay capacity.")
flags.DEFINE_integer("online_buffer_size", 1_100_000, "Online-only replay capacity.")
flags.DEFINE_integer("log_interval", 5_000, "Training log interval.")
flags.DEFINE_integer("eval_interval", 100_000, "Evaluation interval (plot steps).")
flags.DEFINE_integer("save_interval", 100_000, "Checkpoint interval (plot steps).")
flags.DEFINE_integer("eval_episodes", 50, "Evaluation episodes.")
flags.DEFINE_integer("video_episodes", 0, "Video episodes per eval.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_float("p_aug", None, "Image augmentation probability.")
flags.DEFINE_integer("frame_stack", None, "Frame stack.")
flags.DEFINE_integer(
    "utd",
    0,
    "Online UTD override; 0 = use agent.config['utd'].",
)
flags.DEFINE_integer(
    "policy_frequency",
    0,
    "Delayed actor frequency override; 0 = use agent.config['policy_frequency'].",
)
flags.DEFINE_integer("journal_shard_size", 10_000, "Online journal shard size.")
flags.DEFINE_bool("sparse", False, "Map non-zero rewards to -1.")

config_flags.DEFINE_config_file(
    "agent", "agents/ar_qdfl_fast_sac.py", lock_config=False
)


def _as_python_scalar(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _info_to_metrics(prefix, info):
    return {
        f"{prefix}/{k}": _as_python_scalar(v)
        for k, v in info.items()
        if np.ndim(_as_python_scalar(v)) == 0
    }


def _restore_prng_key(values, *, fallback_seed: int):
    """Restore a JAX PRNG key from a stored uint32 vector."""
    key = np.asarray(values, dtype=np.uint32).reshape(-1)
    if key.size == 0:
        return jax.random.PRNGKey(fallback_seed)
    return jax.numpy.asarray(key, dtype=jax.numpy.uint32)


class AppendCsvLogger(CsvLogger):
    """CsvLogger that appends on resume instead of truncating.

    Opens the path for each write (no long-lived fd) so NFS replace/rsync of
    ``eval.csv`` cannot leave the logger writing into a deleted ``.nfs*`` orphan.
    """

    def log(self, row, step):
        row = dict(row)
        row["step"] = step
        filtered_row = {
            k: v for k, v in row.items() if self._is_row_scalar(v)
        }
        path_exists = os.path.isfile(self.path) and os.path.getsize(self.path) > 0
        if path_exists:
            if self.header is None:
                with open(self.path, "r") as existing:
                    header_line = existing.readline().strip()
                self.header = (
                    header_line.split(",") if header_line else list(filtered_row.keys())
                )
            with open(self.path, "a") as handle:
                handle.write(
                    ",".join(str(filtered_row.get(k, "")) for k in self.header) + "\n"
                )
                handle.flush()
        else:
            self.header = list(filtered_row.keys())
            with open(self.path, "w") as handle:
                handle.write(",".join(self.header) + "\n")
                handle.write(
                    ",".join(str(filtered_row.get(k, "")) for k in self.header) + "\n"
                )
                handle.flush()
        self.file = None

    @staticmethod
    def _is_row_scalar(value):
        if isinstance(value, (bool, np.bool_)):
            return False
        if hasattr(value, "ndim") and getattr(value, "ndim", None) == 0:
            return True
        return isinstance(value, (int, float, np.integer, np.floating))


def _process_train_dataset(ds, config):
    ds = Dataset.create(**ds)
    if FLAGS.sparse:
        sparse_rewards = (ds["rewards"] != 0.0) * -1.0
        ds_dict = {k: v for k, v in ds.items()}
        ds_dict["rewards"] = sparse_rewards
        ds = Dataset.create(**ds_dict)
    return ds


def _configure_dataset(dataset, config):
    dataset.p_aug = FLAGS.p_aug
    dataset.frame_stack = FLAGS.frame_stack
    dataset.config = config
    return dataset


def _latest_resume_candidate(save_root: Path, run_group: str, seed: int):
    """Pick the newest incomplete run dir for this seed, if any."""
    root = save_root / "rql" / run_group
    if not root.is_dir():
        return None
    candidates = sorted(root.glob(f"sd{seed:03d}_*"))
    for path in reversed(candidates):
        state_path = path / "runner_state.json"
        if not state_path.is_file():
            continue
        state = load_runner_state(state_path)
        if not bool(state.get("finished", False)):
            return path
    return None


def _persist(
    *,
    exp_dir: Path,
    agent,
    phase: str,
    offline_update_count: int,
    warmup_update_count: int,
    online_env_step: int,
    online_rng,
    done: bool,
    observation,
    numpy_rng: np.random.RandomState,
    journal: OnlineTransitionJournal,
    plot_epoch: int | None,
    save_periodic: bool,
):
    journal.flush()
    save_agent(agent, str(exp_dir), "resume")
    if save_periodic and plot_epoch is not None and plot_epoch > 0:
        save_agent(agent, str(exp_dir), int(plot_epoch))
    state = {
        "phase": phase,
        "offline_update_count": int(offline_update_count),
        "warmup_update_count": int(warmup_update_count),
        "online_env_step": int(online_env_step),
        "agent_offline_update_count": int(agent.offline_update_count),
        "agent_critic_update_count": int(agent.critic_update_count),
        "agent_actor_update_count": int(agent.actor_update_count),
        "agent_online_env_step": int(agent.online_env_step),
        "done": bool(done),
        "finished": False,
        "seed": int(FLAGS.seed),
        "offline_steps": int(FLAGS.offline_steps),
        "warmup_updates": int(FLAGS.warmup_updates),
        "online_steps": int(FLAGS.online_steps),
        "online_rng": [int(x) for x in np.asarray(online_rng).tolist()],
    }
    save_runner_state(exp_dir / "runner_state.json", state)
    save_resume_blobs(
        exp_dir / "resume_blobs",
        numpy_rng=numpy_rng,
        observation=None if observation is None else np.asarray(observation),
        online_rng_key=np.asarray(online_rng, dtype=np.uint32),
    )


def _run_eval(
    *,
    agent,
    eval_env,
    config,
    tb_logger,
    eval_logger,
    step: int,
):
    renders = []
    eval_metrics = {}
    eval_info, trajs, cur_renders = evaluate(
        agent=agent,
        env=eval_env,
        env_name=FLAGS.env_name,
        config=config,
        num_eval_episodes=FLAGS.eval_episodes,
        num_video_episodes=FLAGS.video_episodes,
        video_frame_skip=FLAGS.video_frame_skip,
    )
    renders.extend(cur_renders)
    for k, v in eval_info.items():
        eval_metrics[f"evaluation/{k}"] = v
        print(k, v, flush=True)
    if FLAGS.video_episodes > 0:
        video = prepare_eval_video(renders=renders)
        tb_logger.log_video("evaluation/video", video, step=step)
    tb_logger.log(eval_metrics, step=step)
    eval_logger.log(eval_metrics, step=step)


def main(_):
    config = FLAGS.agent
    if config.get("agent_name") != "ar_qdfl_fast_sac":
        raise ValueError(
            "This runner is dedicated to ar_qdfl_fast_sac; "
            f"got agent_name={config.get('agent_name')!r}"
        )

    save_root = Path(FLAGS.save_dir)
    resume_dir = FLAGS.resume_dir
    if resume_dir is None:
        auto = _latest_resume_candidate(save_root, FLAGS.run_group, FLAGS.seed)
        if auto is not None:
            resume_dir = str(auto)

    if resume_dir:
        exp_dir = Path(resume_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        FLAGS.save_dir = str(exp_dir)
        tb_logger = TensorboardLogger(
            str(exp_dir / "tensorboard"), hparams=get_flag_dict()
        )
        print(f"Resuming experiment directory: {exp_dir}", flush=True)
    else:
        exp_name = get_exp_name(FLAGS.seed)
        tb_logger, exp_dir_str = setup_experiment_logging(
            FLAGS.save_dir,
            project="rql",
            run_group=FLAGS.run_group,
            exp_name=exp_name,
            hparams=get_flag_dict(),
        )
        exp_dir = Path(exp_dir_str)
        FLAGS.save_dir = str(exp_dir)

    flag_dict = get_flag_dict()
    with open(exp_dir / "flags.json", "w") as f:
        json.dump(flag_dict, f)

    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=FLAGS.frame_stack,
        agent_config=config,
        dataset_dir=FLAGS.ogbench_data_dir,
    )
    train_dataset = _process_train_dataset(train_dataset, config)
    val_dataset = _process_train_dataset(val_dataset, config)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    numpy_rng = np.random.RandomState(FLAGS.seed)

    offline_capacity = max(int(FLAGS.offline_buffer_size), train_dataset.size + 1)
    offline_replay = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=offline_capacity
    )
    _configure_dataset(offline_replay, config)
    _configure_dataset(val_dataset, config)

    example_transition = example_transition_from_dataset(offline_replay)
    journal = OnlineTransitionJournal(
        exp_dir / "online_journal",
        shard_size=int(FLAGS.journal_shard_size),
    )

    ex_batch = offline_replay.sample(1)
    agent_class = agents[config["agent_name"]]
    agent = agent_class.create(
        FLAGS.seed,
        ex_batch["observations"],
        ex_batch["actions"],
        config,
    )

    phase = PHASE_OFFLINE
    offline_update_count = 0
    warmup_update_count = 0
    online_env_step = 0
    done = True
    observation = None
    online_rng = jax.random.PRNGKey(FLAGS.seed)
    finished = False

    resume_ckpt = exp_dir / "params_resume.pkl"
    state_path = exp_dir / "runner_state.json"
    if resume_ckpt.is_file() and state_path.is_file():
        agent = restore_agent(agent, str(resume_ckpt))
        state = load_runner_state(state_path)
        phase = str(state["phase"])
        offline_update_count = int(state["offline_update_count"])
        warmup_update_count = int(state["warmup_update_count"])
        online_env_step = int(state["online_env_step"])
        done = bool(state.get("done", True))
        finished = bool(state.get("finished", False))
        blobs = load_resume_blobs(exp_dir / "resume_blobs")
        if "numpy_rng" in blobs:
            numpy_rng = blobs["numpy_rng"]
        if "observation" in blobs:
            observation = blobs["observation"]
        if "online_rng" in state:
            online_rng = _restore_prng_key(
                state["online_rng"], fallback_seed=FLAGS.seed + online_env_step + 1
            )
        elif "online_rng_key" in blobs:
            online_rng = _restore_prng_key(
                blobs["online_rng_key"],
                fallback_seed=FLAGS.seed + online_env_step + 1,
            )
        # Checkpoints freeze agent.config; allow online knobs to be retuned
        # without redoing the 1M offline phase (e.g. KL coef after a collapse).
        live_config = dict(agent.config)
        for key in (
            "offline_kl_coef",
            "st_temperature",
            "target_entropy_per_register",
            "alpha_init",
            "alpha_min",
            "alpha_max",
            "alpha_lr",
            "policy_frequency",
            "utd",
            "actor_ema",
            "online_replay_fraction_max",
            "online_replay_ramp_steps",
            "critic_lr",
            "critic_tau",
        ):
            if key in config:
                live_config[key] = config[key]
        agent = agent.replace(config=type(agent.config)(**live_config))
        print(
            f"Restored phase={phase} offline={offline_update_count} "
            f"warmup={warmup_update_count} online_env={online_env_step} "
            f"offline_kl_coef={live_config.get('offline_kl_coef')}",
            flush=True,
        )

    online_replay = journal.rebuild_replay(
        example_transition,
        max_size=int(FLAGS.online_buffer_size),
        config=config,
        p_aug=FLAGS.p_aug,
        frame_stack=FLAGS.frame_stack,
    )
    if online_replay.size != journal.count and journal.count > 0:
        print(
            f"Warning: journal count={journal.count} "
            f"replay size={online_replay.size}",
            flush=True,
        )

    if finished:
        print("Runner already finished; nothing to do.", flush=True)
        tb_logger.close()
        return

    train_logger = AppendCsvLogger(str(exp_dir / "train.csv"))
    warmup_logger = AppendCsvLogger(str(exp_dir / "warmup.csv"))
    # Separate file: AppendCsvLogger freezes the header on first write, so
    # online FastSAC keys would otherwise be dropped from train.csv.
    online_logger = AppendCsvLogger(str(exp_dir / "online_train.csv"))
    eval_logger = AppendCsvLogger(str(exp_dir / "eval.csv"))

    utd = int(FLAGS.utd) if FLAGS.utd > 0 else int(config.get("utd", 4))
    policy_frequency = (
        int(FLAGS.policy_frequency)
        if FLAGS.policy_frequency > 0
        else int(config.get("policy_frequency", 4))
    )
    ramp_steps = int(config.get("online_replay_ramp_steps", 100_000))
    fraction_max = float(config.get("online_replay_fraction_max", 0.5))
    batch_size = int(config["batch_size"])

    first_time = time.time()
    last_time = time.time()

    def maybe_log(logger, metrics, step, interval_counter):
        nonlocal last_time
        if interval_counter % FLAGS.log_interval != 0 and interval_counter != 0:
            return
        metrics = dict(metrics)
        metrics["time/epoch_time"] = (time.time() - last_time) / max(
            FLAGS.log_interval, 1
        )
        metrics["time/total_time"] = time.time() - first_time
        last_time = time.time()
        tb_logger.log(metrics, step=step)
        logger.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Phase 1: offline joint updates
    # ------------------------------------------------------------------
    if phase == PHASE_OFFLINE:
        if offline_update_count == 0:
            _run_eval(
                agent=agent,
                eval_env=eval_env,
                config=config,
                tb_logger=tb_logger,
                eval_logger=eval_logger,
                step=0,
            )
            _persist(
                exp_dir=exp_dir,
                agent=agent,
                phase=PHASE_OFFLINE,
                offline_update_count=0,
                warmup_update_count=0,
                online_env_step=0,
                online_rng=online_rng,
                done=True,
                observation=None,
                numpy_rng=numpy_rng,
                journal=journal,
                plot_epoch=None,
                save_periodic=False,
            )

        pbar = tqdm.tqdm(
            total=FLAGS.offline_steps,
            initial=offline_update_count,
            desc="offline",
            dynamic_ncols=True,
        )
        while offline_update_count < FLAGS.offline_steps:
            batch = offline_replay.sample(batch_size)
            agent, update_info = agent.offline_update(batch)
            offline_update_count = int(agent.offline_update_count)
            pbar.update(1)
            metrics = _info_to_metrics("training", update_info)
            metrics["phase/offline"] = 1.0
            maybe_log(train_logger, metrics, offline_update_count, offline_update_count)
            if (
                FLAGS.eval_interval > 0
                and offline_update_count > 0
                and (
                    offline_update_count % FLAGS.eval_interval == 0
                    or offline_update_count == FLAGS.offline_steps
                )
            ):
                _run_eval(
                    agent=agent,
                    eval_env=eval_env,
                    config=config,
                    tb_logger=tb_logger,
                    eval_logger=eval_logger,
                    step=offline_update_count,
                )
            if FLAGS.save_interval > 0 and (
                offline_update_count % FLAGS.save_interval == 0
                or offline_update_count == FLAGS.offline_steps
            ):
                _persist(
                    exp_dir=exp_dir,
                    agent=agent,
                    phase=PHASE_OFFLINE,
                    offline_update_count=offline_update_count,
                    warmup_update_count=warmup_update_count,
                    online_env_step=online_env_step,
                    online_rng=online_rng,
                    done=True,
                    observation=None,
                    numpy_rng=numpy_rng,
                    journal=journal,
                    plot_epoch=offline_update_count,
                    save_periodic=True,
                )
            elif offline_update_count % FLAGS.log_interval == 0:
                _persist(
                    exp_dir=exp_dir,
                    agent=agent,
                    phase=PHASE_OFFLINE,
                    offline_update_count=offline_update_count,
                    warmup_update_count=warmup_update_count,
                    online_env_step=online_env_step,
                    online_rng=online_rng,
                    done=True,
                    observation=None,
                    numpy_rng=numpy_rng,
                    journal=journal,
                    plot_epoch=None,
                    save_periodic=False,
                )
        pbar.close()
        agent = agent.with_offline_reference()
        phase = PHASE_WARMUP
        _persist(
            exp_dir=exp_dir,
            agent=agent,
            phase=PHASE_WARMUP,
            offline_update_count=offline_update_count,
            warmup_update_count=warmup_update_count,
            online_env_step=online_env_step,
            online_rng=online_rng,
            done=True,
            observation=None,
            numpy_rng=numpy_rng,
            journal=journal,
            plot_epoch=FLAGS.offline_steps,
            save_periodic=True,
        )

    # Ensure reference snapshot exists when resuming into warmup/online.
    if phase in (PHASE_WARMUP, PHASE_ONLINE):
        if int(agent.offline_update_count) < FLAGS.offline_steps:
            raise ValueError(
                "Cannot enter warmup/online before completing offline updates "
                f"({int(agent.offline_update_count)} < {FLAGS.offline_steps})"
            )

    # ------------------------------------------------------------------
    # Phase 2: hidden critic warmup
    # ------------------------------------------------------------------
    if phase == PHASE_WARMUP:
        # Warmup counter is tracked separately from agent.critic_update_count so
        # resumes remain exact even if critic_update_count already advanced.
        pbar = tqdm.tqdm(
            total=FLAGS.warmup_updates,
            initial=warmup_update_count,
            desc="warmup",
            dynamic_ncols=True,
        )
        while warmup_update_count < FLAGS.warmup_updates:
            batch = offline_replay.sample(batch_size)
            agent, update_info = agent.critic_update(batch)
            warmup_update_count += 1
            pbar.update(1)
            metrics = _info_to_metrics("warmup", update_info)
            metrics["phase/warmup"] = 1.0
            metrics["warmup_update_count"] = warmup_update_count
            maybe_log(
                warmup_logger, metrics, warmup_update_count, warmup_update_count
            )
            if warmup_update_count % FLAGS.log_interval == 0 or (
                warmup_update_count == FLAGS.warmup_updates
            ):
                _persist(
                    exp_dir=exp_dir,
                    agent=agent,
                    phase=PHASE_WARMUP,
                    offline_update_count=offline_update_count,
                    warmup_update_count=warmup_update_count,
                    online_env_step=online_env_step,
                    online_rng=online_rng,
                    done=True,
                    observation=None,
                    numpy_rng=numpy_rng,
                    journal=journal,
                    plot_epoch=None,
                    save_periodic=False,
                )
        pbar.close()
        # Immediate eval at offline_steps+1 (warmup hidden; avoid clobbering
        # the offline-end row already logged at offline_steps).
        post_warmup_step = int(FLAGS.offline_steps) + 1
        _run_eval(
            agent=agent,
            eval_env=eval_env,
            config=config,
            tb_logger=tb_logger,
            eval_logger=eval_logger,
            step=post_warmup_step,
        )
        phase = PHASE_ONLINE
        _persist(
            exp_dir=exp_dir,
            agent=agent,
            phase=PHASE_ONLINE,
            offline_update_count=offline_update_count,
            warmup_update_count=warmup_update_count,
            online_env_step=online_env_step,
            online_rng=online_rng,
            done=True,
            observation=None,
            numpy_rng=numpy_rng,
            journal=journal,
            plot_epoch=post_warmup_step,
            save_periodic=True,
        )

    # ------------------------------------------------------------------
    # Phase 3: online interactions + stratified FastSAC updates
    # ------------------------------------------------------------------
    if phase == PHASE_ONLINE:
        pbar = tqdm.tqdm(
            total=FLAGS.online_steps,
            initial=online_env_step,
            desc="online",
            dynamic_ncols=True,
        )
        while online_env_step < FLAGS.online_steps:
            online_rng, key = jax.random.split(online_rng)
            if done:
                observation, info = env.reset()
                done = False

            action = agent.sample_actions(obs=observation, temperature=1, seed=key)
            action = np.asarray(action)
            next_ob, reward, terminated, truncated, info = env.step(
                action.copy()
            )

            if "action" in info:
                action = np.asarray(info["action"])
            if "intervene_action" in info:
                action = np.asarray(info["intervene_action"])

            if isinstance(terminated, np.ndarray):
                terminal = np.logical_or(terminated, truncated)
                done = bool(terminal.any())
                terminal = terminal.astype(float)
                stored_ob = np.concatenate([observation[-1:], next_ob[:-1]])
                stored_next = next_ob
                mask = 1.0 - terminated.astype(float)
            else:
                terminal = float(terminated or truncated)
                done = bool(terminated or truncated)
                stored_ob = np.asarray(observation, copy=True)
                stored_next = np.asarray(next_ob, copy=True)
                mask = float(1.0 - float(terminated))

            if FLAGS.sparse:
                assert reward <= 0.0
                reward = (reward != 0.0) * -1.0

            # Store actions with dataset geometry (Da,) for h=1.
            stored_action = np.asarray(action, copy=True)
            if stored_action.ndim == 2 and stored_action.shape[0] == 1:
                stored_action = stored_action[0]
            transition = {
                "observations": stored_ob,
                "actions": stored_action,
                "rewards": np.asarray(reward, copy=True),
                "terminals": np.asarray(terminal, copy=True),
                "masks": np.asarray(mask, copy=True),
                "next_observations": stored_next,
            }
            online_replay.add_transition(transition)
            journal.append(transition)
            observation = next_ob
            agent = agent.record_online_env_step(1)
            online_env_step = int(agent.online_env_step)
            pbar.update(1)

            update_info = {}
            mix_info = {}
            for _ in range(utd):
                batch, mix_info = sample_stratified_batch(
                    offline_replay,
                    online_replay,
                    batch_size,
                    online_env_step=online_env_step,
                    ramp_steps=ramp_steps,
                    fraction_max=fraction_max,
                )
                next_critic_count = int(agent.critic_update_count) + 1
                update_actor = (next_critic_count % policy_frequency) == 0
                agent, update_info = agent.online_update(
                    batch, update_actor=update_actor
                )

            pstep = plot_step(
                PHASE_ONLINE,
                offline_update_count=offline_update_count,
                online_env_step=online_env_step,
                offline_steps=FLAGS.offline_steps,
            )
            metrics = _info_to_metrics("training", update_info)
            metrics.update(_info_to_metrics("replay", mix_info))
            metrics["phase/online"] = 1.0
            metrics["online_env_step"] = online_env_step
            maybe_log(online_logger, metrics, int(pstep), online_env_step)

            if FLAGS.eval_interval > 0 and (
                online_env_step % FLAGS.eval_interval == 0
                or online_env_step == FLAGS.online_steps
            ):
                _run_eval(
                    agent=agent,
                    eval_env=eval_env,
                    config=config,
                    tb_logger=tb_logger,
                    eval_logger=eval_logger,
                    step=int(pstep),
                )
            if FLAGS.save_interval > 0 and (
                online_env_step % FLAGS.save_interval == 0
                or online_env_step == FLAGS.online_steps
            ):
                _persist(
                    exp_dir=exp_dir,
                    agent=agent,
                    phase=PHASE_ONLINE,
                    offline_update_count=offline_update_count,
                    warmup_update_count=warmup_update_count,
                    online_env_step=online_env_step,
                    online_rng=online_rng,
                    done=done,
                    observation=observation,
                    numpy_rng=numpy_rng,
                    journal=journal,
                    plot_epoch=int(pstep),
                    save_periodic=True,
                )
            elif online_env_step % FLAGS.log_interval == 0:
                _persist(
                    exp_dir=exp_dir,
                    agent=agent,
                    phase=PHASE_ONLINE,
                    offline_update_count=offline_update_count,
                    warmup_update_count=warmup_update_count,
                    online_env_step=online_env_step,
                    online_rng=online_rng,
                    done=done,
                    observation=observation,
                    numpy_rng=numpy_rng,
                    journal=journal,
                    plot_epoch=None,
                    save_periodic=False,
                )
        pbar.close()

    journal.flush()
    journal.save_snapshot(exp_dir / "online_replay_snapshot.npz")
    final_step = FLAGS.offline_steps + FLAGS.online_steps
    save_agent(agent, str(exp_dir), final_step)
    save_agent(agent, str(exp_dir), "resume")
    state = {
        "phase": PHASE_ONLINE,
        "offline_update_count": int(FLAGS.offline_steps),
        "warmup_update_count": int(FLAGS.warmup_updates),
        "online_env_step": int(FLAGS.online_steps),
        "agent_offline_update_count": int(agent.offline_update_count),
        "agent_critic_update_count": int(agent.critic_update_count),
        "agent_actor_update_count": int(agent.actor_update_count),
        "agent_online_env_step": int(agent.online_env_step),
        "done": True,
        "finished": True,
        "seed": int(FLAGS.seed),
        "offline_steps": int(FLAGS.offline_steps),
        "warmup_updates": int(FLAGS.warmup_updates),
        "online_steps": int(FLAGS.online_steps),
        "online_rng": [int(x) for x in np.asarray(online_rng).tolist()],
    }
    save_runner_state(exp_dir / "runner_state.json", state)
    save_resume_blobs(
        exp_dir / "resume_blobs",
        numpy_rng=numpy_rng,
        observation=None,
        online_rng_key=np.asarray(online_rng, dtype=np.uint32),
    )

    train_logger.close()
    warmup_logger.close()
    online_logger.close()
    eval_logger.close()
    tb_logger.close()
    print(
        f"Finished three-phase run at plot_step={final_step} dir={exp_dir}",
        flush=True,
    )


if __name__ == "__main__":
    app.run(main)
