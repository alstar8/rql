from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


def compute_discounted_mc_returns(rewards, terminals, discount):
    """Discounted Monte Carlo return-to-go per transition (episode-safe).

    For each timestep ``t`` within an episode::

        G_t = r_t + gamma * (1 - terminal_t) * G_{t+1}

    with ``G_{T+1} = 0`` after the final transition. When ``terminal_t = 1``,
    bootstrap is cut so ``G_t = r_t`` and the next earlier episode is isolated.

    Args:
        rewards: 1-D array of shape ``(N,)``.
        terminals: 1-D array of shape ``(N,)`` (0/1 or bool).
        discount: Scalar discount ``gamma`` in ``[0, 1]``.

    Returns:
        ``np.ndarray`` of shape ``(N,)`` with the same dtype as ``rewards``
        when it is floating, else ``float32``. Efficient single reverse pass
        suitable for ~2M rows.
    """
    rewards = np.asarray(rewards)
    terminals = np.asarray(terminals)
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {rewards.shape}")
    if terminals.shape != rewards.shape:
        raise ValueError(
            f"terminals shape {terminals.shape} != rewards shape {rewards.shape}"
        )
    n = int(rewards.shape[0])
    if n == 0:
        out_dtype = (
            rewards.dtype
            if np.issubdtype(rewards.dtype, np.floating)
            else np.float32
        )
        return np.zeros((0,), dtype=out_dtype)

    gamma = float(discount)
    # Accumulate in float64 for numerical stability on long horizons.
    rew = rewards.astype(np.float64, copy=False)
    term = terminals.astype(np.float64, copy=False)
    out = np.empty(n, dtype=np.float64)
    g = 0.0
    for t in range(n - 1, -1, -1):
        g = rew[t] + gamma * (1.0 - term[t]) * g
        out[t] = g

    if np.issubdtype(rewards.dtype, np.floating):
        return out.astype(rewards.dtype, copy=False)
    return out.astype(np.float32)


def compute_episode_success_flags(successes, terminals):
    """Per-transition flags: 1 for all steps in any episode with a success.

    Generic episode-boundary helper: does **not** interpret rewards. The caller
    supplies a per-transition success signal (bool / 0-1). Any transition in an
    episode that contains at least one True success is marked ``1.0``, including
    transitions *before* the success. Episodes are cut at ``terminals > 0``;
    a final unterminated suffix is treated as its own episode. No cross-episode
    leakage.

    Args:
        successes: 1-D array of shape ``(N,)`` (bool or 0/1).
        terminals: 1-D array of shape ``(N,)`` (0/1 or bool).

    Returns:
        ``np.ndarray`` float32 of shape ``(N,)`` with values in ``{0.0, 1.0}``.
    """
    successes = np.asarray(successes)
    terminals = np.asarray(terminals)
    if successes.ndim != 1:
        raise ValueError(f"successes must be 1-D, got shape {successes.shape}")
    if terminals.shape != successes.shape:
        raise ValueError(
            f"terminals shape {terminals.shape} != successes shape {successes.shape}"
        )
    n = int(successes.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.float32)

    succ = successes.astype(bool, copy=False)
    term = terminals > 0
    out = np.zeros(n, dtype=np.float32)
    start = 0
    for t in range(n):
        if term[t] or t == n - 1:
            end = t  # inclusive
            if np.any(succ[start : end + 1]):
                out[start : end + 1] = 1.0
            start = t + 1
    return out


@partial(jax.jit, static_argnames=('padding',))
def random_crop(img, crop_from, padding):
    """Randomly crop an image.

    Args:
        img: Image to crop.
        crop_from: Coordinates to crop from.
        padding: Padding size.
    """
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


@partial(jax.jit, static_argnames=('padding',))
def batched_random_crop(imgs, crop_froms, padding):
    """Batched version of random_crop."""
    return jax.vmap(random_crop, (0, 0, None))(imgs, crop_froms, padding)


class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields
        assert 'observations' in data

        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

        self.frame_stack = None  # Number of frames to stack; set outside the class.
        self.p_aug = None  # Image augmentation probability; set outside the class.
        
        # self.traj_len = None
        self.return_next_actions = False  # Whether to additionally return next actions; set outside the class.
        self.config = None

        # Compute terminal and initial locations.
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs):
        """Return `num_idxs` random indices."""
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None):
        """Sample a batch of transitions."""
        if self.config and 'h' in self.config:
            return self.sample_traj(batch_size)
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        batch = self.get_subset(idxs)
        if self.frame_stack is not None:
            # Stack frames.
            initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
            obs = []  # Will be [ob[t - frame_stack + 1], ..., ob[t]].
            next_obs = []  # Will be [ob[t - frame_stack + 2], ..., ob[t], next_ob[t]].
            for i in reversed(range(self.frame_stack)):
                # Use the initial state if the index is out of bounds.
                cur_idxs = np.maximum(idxs - i, initial_state_idxs)
                obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
                if i != self.frame_stack - 1:
                    next_obs.append(jax.tree_util.tree_map(lambda arr: arr[cur_idxs], self['observations']))
            next_obs.append(jax.tree_util.tree_map(lambda arr: arr[idxs], self['next_observations']))

            batch['observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *obs)
            batch['next_observations'] = jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *next_obs)
        if self.p_aug is not None:
            # Apply random-crop image augmentation.
            if np.random.rand() < self.p_aug:
                self.augment(batch, ['observations', 'next_observations'])
        return batch

    def sample_traj(self, batch_size: int, h=1):

        idxs = np.random.choice(np.flatnonzero(self.terminal_locs - self.initial_locs), batch_size, replace=batch_size >= len(self.initial_locs))
        bt_initial_locs = self.initial_locs[idxs] # [self.initial_locs[idx] for idx in idxs]
        bt_terminal_locs = self.terminal_locs[idxs]

        n = (self.config and self.config['h'] + 1) or n + 1
        start_locs = np.random.randint(bt_initial_locs, bt_terminal_locs)

        batch = {}

        obs = []
        actions = []
        rews = []
        terminals = []
        masks = []
        has_mc = "mc_returns" in self._dict
        mc_rets = [] if has_mc else None
        has_traj_succ = "trajectory_success" in self._dict
        traj_succ = [] if has_traj_succ else None

        for i in range(n):
            cur_idxs = np.minimum(i + start_locs, bt_terminal_locs)
            obs.append(jax.tree_util.tree_map(lambda arr: np.expand_dims(arr[cur_idxs], 0), self['observations']))
            actions.append(jax.tree_util.tree_map(lambda arr: np.expand_dims(arr[cur_idxs], 0), self['actions']))
            rews.append(jax.tree_util.tree_map(lambda arr: np.expand_dims(arr[cur_idxs], 0), self['rewards']))
            terminals.append(jax.tree_util.tree_map(lambda arr: np.expand_dims(arr[cur_idxs], 0), self['terminals']))
            masks.append(jax.tree_util.tree_map(lambda arr: np.expand_dims(arr[cur_idxs], 0), self['masks']))
            if has_mc:
                mc_rets.append(
                    jax.tree_util.tree_map(
                        lambda arr: np.expand_dims(arr[cur_idxs], 0),
                        self["mc_returns"],
                    )
                )
            if has_traj_succ:
                traj_succ.append(
                    jax.tree_util.tree_map(
                        lambda arr: np.expand_dims(arr[cur_idxs], 0),
                        self["trajectory_success"],
                    )
                )

        batch['observations'] = np.concatenate(obs, 0)
        batch['actions'] = np.concatenate(actions, 0)
        batch['rewards'] = np.concatenate(rews, 0)
        batch['terminals'] = np.concatenate(terminals, 0)
        batch['masks'] = np.concatenate(masks, 0)
        if has_mc:
            # (H+1, B) — same layout as rewards; actor uses index 0 (G at s_t).
            batch["mc_returns"] = np.concatenate(mc_rets, 0)
        if has_traj_succ:
            # (H+1, B) — same layout as rewards; actor uses index 0 (flag at s_t).
            batch["trajectory_success"] = np.concatenate(traj_succ, 0)

        return batch

    def get_subset(self, idxs):
        """Return a subset of the dataset given the indices."""
        result = jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)
        if self.return_next_actions:
            # WARNING: This is incorrect at the end of the trajectory. Use with caution.
            result['next_actions'] = self._dict['actions'][np.minimum(idxs + 1, self.size - 1)]
        return result

    def augment(self, batch, keys):
        """Apply image augmentation to the given keys."""
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding)) if len(arr.shape) == 4 else arr,
                batch[key],
            )


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding transitions.
    """

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)
        buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
        return cls(buffer_dict)

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        dataset = cls(buffer_dict)
        dataset.size = dataset.pointer = get_size(init_dataset)
        return dataset

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0
    
    def add_transition(self, transition):
        self._add_transition(transition)

    def _add_transition(self, transition):
        """Add a transition to the replay buffer."""

        def set_idx(buffer, new_element):
            buffer[self.pointer] = new_element
        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)
        if transition["terminals"]:
            self.initial_locs = np.concatenate([[0], self.terminal_locs + 1])
            self.terminal_locs = np.concatenate([self.terminal_locs, [self.pointer - 1]])

    def clear(self):
        """Clear the replay buffer."""
        self.pointer = 0
        ex_tstn = dict(observations=self['observations'][0], actions=self['actions'][0], rewards=self['rewards'][0], terminals=self['terminals'][0], masks=1.0 - self['terminals'][0], next_observations=self['observations'][1])
        new = self.__class__.create(ex_tstn, self.max_size)
        new.config = self.config
        new.p_aug = self.p_aug
        new.frame_stack = self.frame_stack
        new.size = self.size
        self.__dict__.update(new.__dict__)
