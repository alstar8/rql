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
from envs.env_utils import make_env_and_datasets
from envs.ogbench_utils import make_ogbench_env_and_datasets

from utils.datasets import (
    Dataset,
    ReplayBuffer,
    compute_discounted_mc_returns,
    compute_episode_success_flags,
)
from utils.evaluation import evaluate, flatten
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, prepare_eval_video, setup_experiment_logging

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
flags.DEFINE_integer('video_episodes', 0, 'Number of video episodes for each task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for videos.')
flags.DEFINE_bool('sparse', False, "make the task sparse reward")
flags.DEFINE_float('p_aug', None, 'Probability of applying image augmentation.')
flags.DEFINE_integer('frame_stack', None, 'Number of frames to stack.')
flags.DEFINE_integer('utd', 1, 'UTD.')
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
    train_dataset = ReplayBuffer.create_from_initial_dataset(
        dict(train_dataset), size=max(FLAGS.buffer_size, train_dataset.size + 1)
    )
    replay_buffer = train_dataset

    # Set p_aug and frame_stack.
    for dataset in [train_dataset, val_dataset, replay_buffer]:
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
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)
        print(
            f'Resume: restored step counter network.step={int(agent.network.step)} '
            f'(next train iteration will start at this index)'
        )

    print("replay buffer size:", replay_buffer.size)

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
            replay_buffer.add_transition(tstn)

            ob = next_ob

            for _ in range(FLAGS.utd):
                # Sample from the growing replay (offline seed + online appends).
                # ``replay_buffer`` aliases the offline-seeded buffer created above.
                batch = replay_buffer.sample(config['batch_size'])
                agent, update_info = agent.update(batch)

        # Log metrics.
        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            tb_logger.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent.
        if FLAGS.eval_interval != 0 and (i == 1 or i % FLAGS.eval_interval == 0):
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

            if FLAGS.video_episodes > 0:
                video = prepare_eval_video(renders=renders)
                tb_logger.log_video('evaluation/video', video, step=i)

            tb_logger.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent (offline and online). Also save the final step once.
        if FLAGS.save_interval > 0 and (
            i % FLAGS.save_interval == 0 or i == total_steps
        ):
            save_agent(agent, FLAGS.save_dir, i)

    train_logger.close()
    eval_logger.close()
    tb_logger.close()

if __name__ == '__main__':
    app.run(main)
