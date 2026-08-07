import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        grad_max = jax.tree_util.tree_map(jnp.max, grads)
        grad_min = jax.tree_util.tree_map(jnp.min, grads)
        grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grads)

        grad_max_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_max)], axis=0)
        grad_min_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_min)], axis=0)
        grad_norm_flat = jnp.concatenate([jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grad_norm)], axis=0)

        final_grad_max = jnp.max(grad_max_flat)
        final_grad_min = jnp.min(grad_min_flat)
        final_grad_norm = jnp.linalg.norm(grad_norm_flat, ord=1)

        info.update(
            {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')
    # print(agent.network.opt_state[0][1]['modules_critic']['value_net']['Dense_0']['kernel'])


def _resolve_restore_file(restore_path, restore_epoch=None):
    """Resolve a directory or glob to exactly one checkpoint file."""
    candidates = glob.glob(restore_path)
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
    candidate = candidates[0]

    if candidate.endswith('.pkl') or (
        os.path.isfile(candidate) and candidate.endswith('.pkl')
    ):
        restore_file = candidate
    else:
        if restore_epoch is None:
            raise ValueError(
                'restore_epoch is required when restore_path is a directory; '
                'got restore_path=%r' % (restore_path,)
            )
        restore_file = os.path.join(candidate, f'params_{restore_epoch}.pkl')
    if not os.path.isfile(restore_file):
        raise FileNotFoundError(f'checkpoint does not exist: {restore_file}')
    return restore_file


def restore_agent(agent, restore_path, restore_epoch=None):
    """Restore the complete agent pytree from a checkpoint file.

    Args:
        agent: Agent (template from ``create()``; non-pytree fields kept).
        restore_path: Directory (glob ok) containing ``params_*.pkl``, or a
            path/glob to a ``params_{epoch}.pkl`` file itself.
        restore_epoch: Step/epoch number. Required when ``restore_path`` is a
            directory; ignored when ``restore_path`` points at a ``.pkl`` file.

    Returns:
        Agent with pytree state loaded from the checkpoint.
    """
    restore_file = _resolve_restore_file(restore_path, restore_epoch)

    with open(restore_file, 'rb') as f:
        load_dict = pickle.load(f)

    agent = flax.serialization.from_state_dict(agent, load_dict['agent'])

    print(f'Restored from {restore_file}')
    return agent


def _shape_checked_module_copy(source, destination, module_name):
    """Copy a parameter subtree after exact structure and shape validation."""
    source_with_paths, source_tree = jax.tree_util.tree_flatten_with_path(source)
    destination_with_paths, destination_tree = (
        jax.tree_util.tree_flatten_with_path(destination)
    )
    if source_tree != destination_tree:
        raise ValueError(
            f'parameter structure mismatch for {module_name}: '
            f'source={source_tree} destination={destination_tree}'
        )
    copied_leaves = []
    for (source_path, source_leaf), (destination_path, destination_leaf) in zip(
        source_with_paths,
        destination_with_paths,
    ):
        if source_path != destination_path:
            raise ValueError(
                f'parameter path mismatch for {module_name}: '
                f'source={jax.tree_util.keystr(source_path)} '
                f'destination={jax.tree_util.keystr(destination_path)}'
            )
        source_shape = tuple(jnp.shape(source_leaf))
        destination_shape = tuple(jnp.shape(destination_leaf))
        if source_shape != destination_shape:
            raise ValueError(
                f'parameter shape mismatch for {module_name}'
                f'{jax.tree_util.keystr(source_path)}: '
                f'source={source_shape} destination={destination_shape}'
            )
        copied_leaves.append(
            jnp.asarray(source_leaf, dtype=jnp.asarray(destination_leaf).dtype)
        )
    return jax.tree_util.tree_unflatten(destination_tree, copied_leaves)


def _first_available_module(source_params, candidates):
    for candidate in candidates:
        if candidate in source_params:
            return candidate
    return None


def restore_agent_backbone(
    agent,
    restore_path,
    restore_epoch=None,
    *,
    restore_critic=False,
):
    """Initialize only behavior-policy modules from a different agent family.

    The destination's RL modules, optimizer state, RNG, and update counter stay
    freshly initialized. Actor and target-actor copies are mandatory and fully
    shape checked. Endpoint critics may optionally be initialized from either a
    ``critic`` or legacy time-conditioned ``value`` ensemble when shapes match.
    """
    restore_file = _resolve_restore_file(restore_path, restore_epoch)
    with open(restore_file, 'rb') as f:
        load_dict = pickle.load(f)

    try:
        source_params = load_dict['agent']['network']['params']
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f'checkpoint has no agent/network/params tree: {restore_file}'
        ) from exc

    destination_params = agent.network.params.copy()
    copied_modules = []
    actor_pairs = (
        ('modules_actor', ('modules_actor',)),
        (
            'modules_target_actor',
            ('modules_target_actor', 'modules_actor'),
        ),
    )
    for destination_name, source_candidates in actor_pairs:
        if destination_name not in destination_params:
            raise ValueError(
                f'destination agent has no required {destination_name}'
            )
        source_name = _first_available_module(
            source_params,
            source_candidates,
        )
        if source_name is None:
            raise ValueError(
                f'checkpoint has none of required modules {source_candidates}'
            )
        destination_params[destination_name] = _shape_checked_module_copy(
            source_params[source_name],
            destination_params[destination_name],
            destination_name,
        )
        copied_modules.append(f'{source_name}->{destination_name}')

    if restore_critic:
        critic_pairs = (
            (
                'modules_critic',
                ('modules_critic', 'modules_value'),
            ),
            (
                'modules_target_critic',
                (
                    'modules_target_critic',
                    'modules_target_value',
                    'modules_critic',
                    'modules_value',
                ),
            ),
            (
                'modules_latent_critic',
                ('modules_latent_critic',),
            ),
            (
                'modules_target_latent_critic',
                (
                    'modules_target_latent_critic',
                    'modules_latent_critic',
                ),
            ),
        )
        restored_any_critic = False
        for destination_name, source_candidates in critic_pairs:
            if destination_name not in destination_params:
                continue
            source_name = _first_available_module(
                source_params,
                source_candidates,
            )
            if source_name is None:
                continue
            destination_params[destination_name] = _shape_checked_module_copy(
                source_params[source_name],
                destination_params[destination_name],
                destination_name,
            )
            copied_modules.append(f'{source_name}->{destination_name}')
            restored_any_critic = True
        if not restored_any_critic:
            raise ValueError(
                'restore_critic=True, but no compatible critic module names '
                'exist in both checkpoint and destination'
            )

    # Deliberately reset optimizer state and counters for the new RL problem.
    destination_network = agent.network.replace(
        params=destination_params,
        opt_state=agent.network.tx.init(destination_params),
        step=1,
    )
    restored_agent = agent.replace(network=destination_network)
    print(
        f'Restored BC backbone from {restore_file}; '
        f'copied={copied_modules}; reset optimizer/counter'
    )
    return restored_agent
