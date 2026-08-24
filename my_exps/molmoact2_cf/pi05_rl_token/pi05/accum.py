"""Gradient accumulation for openpi training, without editing openpi.

pi0.5 predicts actions with a flow-matching head, so every sample draws its own random
flow timestep and per-sample gradients are noisy. openpi's own serious configs train at
batch 256 for that reason; only the small custom-dataset fine-tune uses 32. Two H100s
fit a batch of about 16 (measured: 62 GB per card, ~2.6 GB of activations per sample),
so 256 is reachable only by accumulating.

Accumulation is close to free in wall-clock: throughput is the same either way, so the
same number of samples becomes fewer, cleaner optimizer steps instead of many noisy ones.

The implementation wraps the optimizer in optax.MultiSteps rather than rewriting
openpi's train step, which keeps the change to one well-tested library primitive.

The subtlety is the learning-rate schedule. It is indexed by *optimizer* steps, and with
accumulation those are k times rarer than the training loop's steps. Left alone, a run of
30 000 loop steps with k=16 performs 1 875 optimizer steps, so a 1 000-step warmup would
cover more than half the run and the cosine decay would barely begin. schedule_steps()
converts the intended schedule into optimizer-step units.
"""

from __future__ import annotations

import dataclasses

import optax


def wrap_optimizer_factory(create_optimizer, accumulate: int):
    """Return a drop-in create_optimizer that accumulates over ``accumulate`` steps."""
    if accumulate < 1:
        raise ValueError(f"accumulate must be >= 1, got {accumulate}")

    def create(*args, **kwargs):
        tx = create_optimizer(*args, **kwargs)
        if accumulate == 1:
            return tx
        wrapped = optax.MultiSteps(tx, every_k_schedule=accumulate)
        # optax.MultiSteps is a plain object, but openpi's TrainState is beartype-checked
        # and insists on a literal GradientTransformation NamedTuple. Rebuilding one from
        # the wrapper's methods keeps the accumulation and satisfies the annotation.
        return optax.GradientTransformation(wrapped.init, wrapped.update)

    return create


def optimizer_steps(loop_steps: int, accumulate: int) -> int:
    """How many real parameter updates a run of ``loop_steps`` will perform."""
    return loop_steps // accumulate


def rescale_schedule(lr_schedule, loop_steps: int, accumulate: int):
    """Re-express a schedule written in loop steps as one in optimizer steps."""
    if accumulate == 1:
        return lr_schedule
    updates = optimizer_steps(loop_steps, accumulate)
    changes = {}
    if hasattr(lr_schedule, "decay_steps"):
        # Decay should finish as the run finishes, not 16x later.
        changes["decay_steps"] = max(1, updates)
    if hasattr(lr_schedule, "warmup_steps"):
        # Keep warmup at the same fraction of the run it was before.
        fraction = lr_schedule.warmup_steps / max(1, lr_schedule.decay_steps)
        changes["warmup_steps"] = max(1, int(round(fraction * updates)))
    return dataclasses.replace(lr_schedule, **changes) if changes else lr_schedule


def effective_batch(batch_size: int, accumulate: int) -> int:
    return batch_size * accumulate
