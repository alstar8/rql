import collections
import os
import re
import time

import gymnasium
import numpy as np
import ogbench
from gymnasium import spaces
from gymnasium.spaces import Box
import jax
import random
from utils.datasets import Dataset

class ObsPadWrapper(gymnasium.ObservationWrapper):
    """Pad flat observations with trailing zeros.

    Used as a workaround for an XLA compile failure in DFL-RQL guidance
    ``jax.grad`` when both observation and action dimensions are even
    (e.g. antsoccer 42×8). Padding obs by 1 makes the obs dim odd and
    restores successful compilation without changing semantics.
    """

    def __init__(self, env, pad: int = 1):
        super().__init__(env)
        if pad < 0:
            raise ValueError(f"pad must be >= 0, got {pad}")
        self.pad = int(pad)
        if self.pad == 0:
            return
        if not isinstance(env.observation_space, Box) or env.observation_space.shape is None:
            raise TypeError("ObsPadWrapper requires a flat Box observation space")
        low = np.concatenate(
            [env.observation_space.low, np.full((self.pad,), -np.inf, dtype=np.float32)]
        )
        high = np.concatenate(
            [env.observation_space.high, np.full((self.pad,), np.inf, dtype=np.float32)]
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        if self.pad == 0:
            return observation
        return np.concatenate(
            [np.asarray(observation, dtype=np.float32), np.zeros(self.pad, dtype=np.float32)],
            axis=-1,
        )


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Remove keys that are not needed for logging.
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['final_reward'] = reward
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return self.get_observation(), reward, terminated, truncated, info

class ActionChunkingWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env):
        super().__init__(env)
        self.action_queue = []

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        self.action_queue.clear()
        return ob, info

    def step(self, chunk_action):
        if not self.action_queue:
            self.action_queue.extend(chunk_action)
        action = self.action_queue.pop(0)
        ob, rew, term, trunc, info = self.env.step(action)
        info["action"] = action
        return ob, rew, term, trunc, info

def make_env_and_datasets(
    env_name,
    frame_stack=None,
    action_clip_eps=1e-5,
    dataset_path=None,
    agent_config=None,
    dataset_dir=None,
):
    if 'singletask' in env_name:
        # OGBench.
        if dataset_dir is None:
            dataset_dir = os.environ.get('OGBENCH_DATA_DIR')
        ogbench_kwargs = {}
        if dataset_dir is not None:
            ogbench_kwargs['dataset_dir'] = dataset_dir
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(env_name, **ogbench_kwargs)
        eval_env = ogbench.make_env_and_datasets(env_name, env_only=True, **ogbench_kwargs)
        env = EpisodeMonitor(env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        eval_env = EpisodeMonitor(eval_env, filter_regexes=['.*privileged.*', '.*proprio.*'])
        if dataset_path is not None:
            file = np.load(dataset_path)
            dataset = dict()
            for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals', 'masks']:
                dataset[k] = file[k][...].astype(np.float32, copy=False)
            train_dataset = Dataset.create(**dataset)
            val_dataset = Dataset.create(**dataset)
        else:
            train_dataset = Dataset.create(**train_dataset)
            val_dataset = Dataset.create(**val_dataset)

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
        eval_env = FrameStackWrapper(eval_env, frame_stack)
    if agent_config and "h" in agent_config: # chunk horizon
        env = ActionChunkingWrapper(env)
        eval_env = ActionChunkingWrapper(eval_env)

    env.reset()
    eval_env.reset()

    # Clip dataset actions.
    if action_clip_eps is not None and train_dataset is not None:
        train_dataset = train_dataset.copy(
            add_or_replace=dict(actions=np.clip(train_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
        )
        if val_dataset is not None:
            val_dataset = val_dataset.copy(
                add_or_replace=dict(actions=np.clip(val_dataset['actions'], -1 + action_clip_eps, 1 - action_clip_eps))
            )

    return env, eval_env, train_dataset, val_dataset
