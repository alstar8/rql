"""Gradient accumulation, checked on the two things that quietly ruin a long run.

An optimizer that never applies an update, or a schedule still indexed in loop steps,
both produce a training curve that looks plausible for hours.
"""

import dataclasses

import jax.numpy as jnp
import optax
import pytest

from pi05.accum import (
    effective_batch,
    optimizer_steps,
    rescale_schedule,
    wrap_optimizer_factory,
)


@dataclasses.dataclass(frozen=True)
class FakeSchedule:
    warmup_steps: int = 1000
    decay_steps: int = 30000
    peak_lr: float = 2.5e-5


def _sgd(*args, **kwargs):
    return optax.sgd(0.1)


def test_effective_batch_multiplies():
    assert effective_batch(16, 16) == 256


def test_optimizer_steps_counts_real_updates():
    assert optimizer_steps(30000, 16) == 1875
    assert optimizer_steps(30000, 1) == 30000


def test_accumulate_one_returns_the_untouched_optimizer():
    tx = wrap_optimizer_factory(_sgd, 1)()
    assert not isinstance(tx, optax.MultiSteps)


def test_updates_are_withheld_until_the_last_micro_step():
    tx = wrap_optimizer_factory(_sgd, 4)()
    params = {"w": jnp.array([1.0])}
    state = tx.init(params)
    grads = {"w": jnp.array([1.0])}

    applied = []
    for _ in range(4):
        updates, state = tx.update(grads, state, params)
        applied.append(float(updates["w"][0]))

    # The first three steps must move nothing; the fourth carries the whole update.
    assert applied[:3] == [0.0, 0.0, 0.0]
    assert applied[3] != 0.0


def test_accumulated_update_matches_one_big_batch():
    # Averaging four identical gradients must equal a single step on that gradient.
    accumulated = wrap_optimizer_factory(_sgd, 4)()
    plain = _sgd()
    params = {"w": jnp.array([1.0])}
    grads = {"w": jnp.array([0.5])}

    state = accumulated.init(params)
    for _ in range(4):
        updates, state = accumulated.update(grads, state, params)

    plain_state = plain.init(params)
    expected, _ = plain.update(grads, plain_state, params)
    assert float(updates["w"][0]) == pytest.approx(float(expected["w"][0]), rel=1e-5)


def test_schedule_is_reexpressed_in_optimizer_steps():
    rescaled = rescale_schedule(FakeSchedule(), loop_steps=30000, accumulate=16)
    assert rescaled.decay_steps == 1875
    # Warmup keeps its share of the run: 1000/30000 of 1875.
    assert rescaled.warmup_steps == pytest.approx(62, abs=1)


def test_schedule_untouched_without_accumulation():
    original = FakeSchedule()
    assert rescale_schedule(original, loop_steps=30000, accumulate=1) is original


def test_rescaled_warmup_stays_a_minority_of_the_run():
    rescaled = rescale_schedule(FakeSchedule(), loop_steps=30000, accumulate=16)
    # The bug this guards: an unscaled 1000-step warmup over 1875 updates would spend
    # more than half the run climbing to peak learning rate.
    assert rescaled.warmup_steps < optimizer_steps(30000, 16) / 4


def test_bad_accumulation_is_rejected():
    with pytest.raises(ValueError):
        wrap_optimizer_factory(_sgd, 0)


def test_wrapped_optimizer_is_a_real_gradient_transformation():
    # openpi's TrainState is beartype-checked and rejects anything that is not literally
    # an optax.GradientTransformation, so returning a MultiSteps object crashes at init.
    tx = wrap_optimizer_factory(_sgd, 4)()
    assert isinstance(tx, optax.GradientTransformation)
    assert not isinstance(tx, optax.MultiSteps)
