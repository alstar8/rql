import os
import platform
import pathlib

import json
import random
import time
import copy

import glob, tqdm, os, json, random, time, jax
import numpy as np
from absl import app, flags
from ml_collections import config_flags
from collections import defaultdict

from agents import agents
from envs.env_utils import ObsPadWrapper, make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets

from utils.datasets import (
    Dataset,
    ReplayBuffer,
    compute_discounted_mc_returns,
    compute_episode_success_flags,
)
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, restore_agent_backbone, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, prepare_eval_video, setup_experiment_logging
from utils.ar_qdfl_fast_sac_plumbing import (
    example_transition_from_dataset,
    sample_stratified_batch,
)

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', '', 'Environment (dataset) name.')
flags.DEFINE_string('ogbench_dataset_dir', None, 'Dataset path.')
flags.DEFINE_string(
    'ogbench_data_dir',
    None,
    'Local OGBench dataset cache directory. Uses existing .npz files and skips download.',
)
flags.DEFINE_integer('dataset_replace_interval', 1000, 'Dataset replace interval, used for large datasets because of memory constraints')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_integer('offline_steps', 1000000, 'Number of offline steps.')
flags.DEFINE_integer('online_steps', 0, 'Number of online steps.')
flags.DEFINE_integer('buffer_size', 100000000, 'Replay buffer size.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')
flags.DEFINE_integer('eval_episodes', 50, 'Number of evaluation episodes.')
flags.DEFINE_bool(
    'eval_residual_off',
    False,
    'Also evaluate Decoupled CF with its RL policy-improvement module disabled.',
)
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('utd', 1, 'UTD.')
flags.DEFINE_integer(
    'training_step_offset',
    0,
    'Offset used for logs/checkpoint names after a reset-state backbone restore.',
)
flags.DEFINE_float(
    'online_replay_fraction_max',
    0.0,
    'Maximum online share in stratified batches; <=0 keeps legacy mixed replay.',
)
flags.DEFINE_integer(
    'online_replay_ramp_steps',
    100000,
    'Online interactions over which the stratified online share ramps up.',
)
flags.DEFINE_integer(
    'online_buffer_size',
    2000000,
    'Capacity of the online-only replay partition when stratification is enabled.',
)
# Optional resume: restore weights/opt/RNG after create(), then continue the
# training loop from agent.network.step (TrainState starts at 1 and increments
# once per update, so after a save at loop i the stored step is i+1 and the
# next iteration is i+1). Does not restore dataset shuffle / CSV / TB state;
# a new exp dir is still created. Default None = fresh run.
flags.DEFINE_string(
    'restore_path',
    None,
    'Checkpoint directory (with --restore_epoch) or path to params_*.pkl.',
)
flags.DEFINE_integer(
    'restore_epoch',
    None,
    'Checkpoint step when --restore_path is a directory. Ignored for .pkl files.',
)
flags.DEFINE_bool(
    'restore_backbone_only',
    False,
    'Copy only shape-compatible BC actor modules and reset optimizer/counter.',
)
flags.DEFINE_bool(
    'restore_backbone_critic',
    False,
    'With --restore_backbone_only, also copy compatible critic modules.',
)

config_flags.DEFINE_config_file('agent', 'agents/rql.py', lock_config=False)

def main(_):
    # Set up logger.
    exp_name = get_exp_name(FLAGS.seed)
    tb_logger, exp_dir = setup_experiment_logging(
        FLAGS.save_dir,
        project='rql',
        run_group=FLAGS.run_group,
        exp_name=exp_name,
        hparams=get_flag_dict(),
    )
    FLAGS.save_dir = exp_dir
    print(f'TensorBoard logs: {os.path.join(FLAGS.save_dir, "tensorboard")}')
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    # Make environment and datasets.
    config = FLAGS.agent
    if FLAGS.ogbench_dataset_dir is not None:
        assert FLAGS.dataset_replace_interval != 0 and FLAGS.online_steps == 0
        dataset_idx = 0
        dataset_paths = [
            file for file in sorted(glob.glob(f"{FLAGS.ogbench_dataset_dir}/*.npz")) if '-val.npz' not in file
        ]
        _, _, train_dataset, val_dataset = make_ogbench_env_and_datasets(
            FLAGS.env_name,
            dataset_path=dataset_paths[dataset_idx],
            compact_dataset=False,
        )
        env, eval_env, _, _ = make_env_and_datasets(
            FLAGS.env_name,
            frame_stack=FLAGS.frame_stack,
            agent_config=config,
            dataset_dir=FLAGS.ogbench_data_dir,
        )
    else:
        env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
            FLAGS.env_name,
            frame_stack=FLAGS.frame_stack,
            agent_config=config,
            dataset_dir=FLAGS.ogbench_data_dir,
        )

    # DFL-RQL guidance jax.grad hits an XLA reshape failure when both the
    # observation and action dimensions are even (e.g. antsoccer 42×8). Pad
    # observations by 1 so the obs dim is odd; RQL and odd×even domains are
    # unchanged.
    agent_name = str(config.get("agent_name", ""))
    obs_dim = int(np.asarray(train_dataset["observations"]).shape[-1])
    act_dim = int(np.asarray(train_dataset["actions"]).shape[-1])
    obs_pad = 1 if (agent_name.startswith("dflrql") and obs_dim % 2 == 0 and act_dim % 2 == 0) else 0
    if obs_pad:
        print(
            f"ObsPadWrapper(pad={obs_pad}): agent={agent_name} "
            f"obs_dim={obs_dim} act_dim={act_dim} (even×even XLA workaround)",
            flush=True,
        )
        env = ObsPadWrapper(env, pad=obs_pad)
        eval_env = ObsPadWrapper(eval_env, pad=obs_pad)

        def _pad_obs_dataset(ds):
            d = {k: v for k, v in ds.items()}
            zeros = np.zeros((d["observations"].shape[0], obs_pad), dtype=np.float32)
            d["observations"] = np.concatenate(
                [np.asarray(d["observations"], dtype=np.float32), zeros], axis=-1
            )
            d["next_observations"] = np.concatenate(
                [np.asarray(d["next_observations"], dtype=np.float32), zeros], axis=-1
            )
            return d

        train_dataset = _pad_obs_dataset(train_dataset)
        val_dataset = _pad_obs_dataset(val_dataset)

    def process_train_dataset(ds):
        ds = Dataset.create(**ds)
        if FLAGS.sparse:
            # Create a new dataset with modified rewards instead of trying to modify the frozen one
            sparse_rewards = (ds["rewards"] != 0.0) * -1.0
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["rewards"] = sparse_rewards
            ds = Dataset.create(**ds_dict)
        # Optional MC return-to-go (DiscreteARIQL mc_return mode). Off by default
        # so other agents / IQL runs are unchanged.
        if bool(config.get("use_mc_returns", False)):
            discount = float(config.get("discount", 0.99))
            mc_returns = compute_discounted_mc_returns(
                ds["rewards"], ds["terminals"], discount
            )
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["mc_returns"] = mc_returns
            ds = Dataset.create(**ds_dict)
        # Optional episode-success flags (DiscreteARIQL trajectory_success mode).
        # OGBench convention in this pipeline: masks <= 0 mark goal/success
        # transitions (humanoidmaze rewards in {-1,0}; masks=0 on goal). The
        # generic helper does not hardcode rewards — only the success signal.
        # Launcher is task-specific; other agents leave use_trajectory_success
        # False and are unaffected.
        if bool(config.get("use_trajectory_success", False)):
            step_success = np.asarray(ds["masks"]) <= 0
            traj_flags = compute_episode_success_flags(
                step_success, ds["terminals"]
            )
            if traj_flags.shape != np.asarray(ds["rewards"]).shape:
                raise ValueError(
                    f"trajectory_success shape {traj_flags.shape} != "
                    f"rewards shape {np.asarray(ds['rewards']).shape}"
                )
            ds_dict = {k: v for k, v in ds.items()}
            ds_dict["trajectory_success"] = traj_flags
            ds = Dataset.create(**ds_dict)
        return ds

    train_dataset = process_train_dataset(train_dataset)
    val_dataset = process_train_dataset(val_dataset)

    # Initialize agent.
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    
    # Set up datasets.
    train_dataset = Dataset.create(**train_dataset)
    use_stratified_replay = FLAGS.online_replay_fraction_max > 0.0
    offline_capacity = (
        train_dataset.size + 1
        if use_stratified_replay
        else max(FLAGS.buffer_size, train_dataset.size + 1)
    )
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=offline_capacity
    )
    replay_buffer = train_dataset
    online_replay = None
    if use_stratified_replay:
        if FLAGS.online_buffer_size <= 0:
            raise ValueError(
                f'online_buffer_size must be > 0, got {FLAGS.online_buffer_size}'
            )
        online_replay = ReplayBuffer.create(
            example_transition_from_dataset(train_dataset),
            size=FLAGS.online_buffer_size,
        )

    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer, online_replay]:
        if dataset is not None:
            dataset.p_aug = FLAGS.p_aug
            dataset.frame_stack = FLAGS.frame_stack
            dataset.config = config
    ex_batch = train_dataset.sample(1)

    # Create agent.
    agent_class = agents[config['agent_name']]
    agent = agent_class.create(
        FLAGS.seed,
        ex_batch['observations'],
        ex_batch['actions'],
        config,
    )

    # Optional resume: load pytree state (params, opt_state, network.step, rng).
    # Loop continues from agent.network.step so checkpointed steps are not redone.
    if FLAGS.restore_path:
        if FLAGS.restore_backbone_only:
            agent = restore_agent_backbone(
                agent,
                FLAGS.restore_path,
                FLAGS.restore_epoch,
                restore_critic=FLAGS.restore_backbone_critic,
            )
            print(
                'Backbone initialization: fresh RL heads/optimizer, '
                f'local network.step={int(agent.network.step)}, '
                f'plot offset={FLAGS.training_step_offset}'
            )
        else:
            agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
            print(
                f'Resume: restored step counter network.step={int(agent.network.step)} '
                f'(next train iteration will start at this index)'
            )
    elif FLAGS.restore_backbone_only:
        raise ValueError('--restore_backbone_only requires --restore_path')

    print("replay buffer size:", replay_buffer.size)
    if online_replay is not None:
        print(
            'stratified replay enabled: '
            f'offline_size={train_dataset.size} '
            f'online_capacity={online_replay.max_size} '
            f'online_fraction_max={FLAGS.online_replay_fraction_max}'
        )

    # Train agent.
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    done = True
    expl_metrics = dict()
    online_rng = jax.random.PRNGKey(FLAGS.seed)

    eps_dataset, eps = defaultdict(list), []

    # Fresh create() → step=1; after save at loop i → step=i+1. Start there.
    start_step = int(agent.network.step)
    total_steps = FLAGS.offline_steps + FLAGS.online_steps
    if start_step > total_steps:
        raise ValueError(
            f'Restored network.step={start_step} exceeds '
            f'offline_steps+online_steps={total_steps}; nothing to train.'
        )

    for i in tqdm.tqdm(
        range(start_step, total_steps + 1), smoothing=0.1, dynamic_ncols=True
    ):
        plot_step = FLAGS.training_step_offset + i
        if i <= FLAGS.offline_steps:
            if FLAGS.ogbench_dataset_dir is not None and FLAGS.dataset_replace_interval != 0 and i % FLAGS.dataset_replace_interval == 0:
                dataset_idx = (dataset_idx + 1) % len(dataset_paths)
                print(f"Using new dataset: {dataset_paths[dataset_idx]}", flush=True)
                train_dataset, val_dataset = make_ogbench_env_and_datasets(
                    FLAGS.env_name,
                    dataset_path=dataset_paths[dataset_idx],
                    compact_dataset=False,
                    dataset_only=True,
                    cur_env=env,
                )
                train_dataset = process_train_dataset(train_dataset)
                train_dataset.p_aug = FLAGS.p_aug # TODO: cleanup
                train_dataset.frame_stack = FLAGS.frame_stack
                train_dataset.config = config
            batch = train_dataset.sample(config['batch_size'])
            agent, update_info = agent.update(batch)
        else:
            online_rng, key = jax.random.split(online_rng)
            if done:
                step = 0
                ob, info = env.reset()
                 
            action = agent.sample_actions(obs=ob, temperature=1, seed=key)
            action = np.array(action)
            next_ob, reward, terminated, truncated, info = env.step(action.copy())

            if 'action' in info:
                action = info['action']

            if 'intervene_action' in info:
                action = info['intervene_action']

            if isinstance(terminated, np.ndarray): # chunk
                terminal = np.logical_or(terminated, truncated)
                done = terminal.any() # [-1]
                terminal = terminal.astype(float)
                ob = np.concatenate([ob[-1:], next_ob[:-1]])
            else:
                terminal = terminated or truncated
                done = terminal
                terminal = float(terminal)

            if FLAGS.sparse:
                assert reward <= 0.0
                reward = (reward != 0.0) * -1.0

            tstn = {
                'observations': np.array(ob, copy=True),
                'actions': np.array(action, copy=True),
                'rewards': np.array(reward, copy=True),
                'terminals': np.array(terminal, copy=True),
                'masks': np.array(1.0 - terminated, copy=True),
                'next_observations': np.array(next_ob, copy=True),
            }
            if online_replay is not None:
                online_replay.add_transition(tstn)
            else:
                replay_buffer.add_transition(tstn)

            ob = next_ob

            for _ in range(FLAGS.utd):
                if online_replay is not None:
                    batch, replay_info = sample_stratified_batch(
                        train_dataset,
                        online_replay,
                        config['batch_size'],
                        online_env_step=i - FLAGS.offline_steps,
                        ramp_steps=FLAGS.online_replay_ramp_steps,
                        fraction_max=FLAGS.online_replay_fraction_max,
                    )
                else:
                    # Legacy replay: the million-transition offline seed and
                    # online appends share one uniformly sampled partition.
                    batch = replay_buffer.sample(config['batch_size'])
                    replay_info = {}
                agent, update_info = agent.update(batch)
                update_info = {**update_info, **replay_info}

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            tb_logger.log(train_metrics, step=plot_step)
            train_logger.log(train_metrics, step=plot_step)

        # Evaluate agent. Also evaluate on the first resumed step so
        # offline→online joins get an immediate eval.csv point near restore.
        if FLAGS.eval_interval != 0 and (
            i == 1 or i == start_step or i % FLAGS.eval_interval == 0
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
                eval_metrics[f'evaluation/{k}'] = v
                print(k, v)

            if FLAGS.eval_residual_off and agent_name in (
                'dflrql11',
                'dflrql12',
            ):
                residual_off_config = agent.config.copy(
                    add_or_replace={'disable_rl_policy': True}
                )
                residual_off_agent = agent.replace(config=residual_off_config)
                residual_off_info, _, _ = evaluate(
                    agent=residual_off_agent,
                    env=eval_env,
                    env_name=FLAGS.env_name,
                    config=residual_off_config,
                    num_eval_episodes=FLAGS.eval_episodes,
                    num_video_episodes=0,
                    video_frame_skip=FLAGS.video_frame_skip,
                )
                for k, v in residual_off_info.items():
                    eval_metrics[f'evaluation/residual_off_{k}'] = v
                if (
                    'success' in eval_info
                    and 'success' in residual_off_info
                ):
                    role_gap = (
                        eval_info['success']
                        - residual_off_info['success']
                    )
                    eval_metrics['evaluation/role_gap_success'] = role_gap
                    print(
                        'residual_off_success',
                        residual_off_info['success'],
                        'role_gap_success',
                        role_gap,
                    )

            if FLAGS.video_episodes > 0:
                video = prepare_eval_video(renders=renders)
                tb_logger.log_video('evaluation/video', video, step=i)

            tb_logger.log(eval_metrics, step=plot_step)
            eval_logger.log(eval_metrics, step=plot_step)

        # Save agent (offline and online). Also save the final step once.
        if FLAGS.save_interval > 0 and (
            i % FLAGS.save_interval == 0 or i == total_steps
        ):
            save_agent(agent, FLAGS.save_dir, plot_step)

    train_logger.close()
    eval_logger.close()
    tb_logger.close()

if __name__ == '__main__':
    app.run(main)
