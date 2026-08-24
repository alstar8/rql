"""Checkpoint retention decisions, tested before anything deletes 48 GB.

The dangerous failure is not keeping too much -- it is deleting a good checkpoint because
one evaluation sample went badly, which is unrecoverable without retraining.
"""

import pytest

from pi05.curator import Evaluated, clearly_worse, select_survivors, wilson_interval


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(30, 100)
    assert low < 0.30 < high


def test_wilson_interval_is_sane_at_zero_and_one():
    assert wilson_interval(0, 50)[0] == 0.0
    assert wilson_interval(50, 50)[1] == 1.0


def test_wilson_interval_narrows_with_more_episodes():
    small = wilson_interval(8, 32)
    large = wilson_interval(32, 128)
    assert (large[1] - large[0]) < (small[1] - small[0])


def test_no_evaluations_still_keeps_the_latest():
    assert select_survivors([], latest_step=5000) == {5000}


def test_single_checkpoint_is_never_deleted():
    entries = [Evaluated(step=5000, successes=1, episodes=128)]
    assert select_survivors(entries, latest_step=5000) == {5000}


def test_latest_survives_even_when_it_scores_worst():
    # The newest checkpoint is the resume point and the final model, so it stays
    # regardless of how it measured.
    entries = [
        Evaluated(step=5000, successes=60, episodes=128),
        Evaluated(step=10000, successes=5, episodes=128),
    ]
    survivors = select_survivors(entries, latest_step=10000)
    assert 10000 in survivors
    assert 5000 in survivors  # it is the best


def test_clearly_worse_checkpoint_is_dropped():
    best = Evaluated(step=10000, successes=64, episodes=128)  # 50%
    poor = Evaluated(step=5000, successes=6, episodes=128)  # 4.7%
    assert clearly_worse(poor, best)
    survivors = select_survivors([best, poor], latest_step=15000)
    assert 5000 not in survivors


def test_overlapping_intervals_keep_both():
    # 40% and 46% over 128 episodes overlap heavily; deleting either would be a guess.
    a = Evaluated(step=5000, successes=51, episodes=128)
    b = Evaluated(step=10000, successes=59, episodes=128)
    assert not clearly_worse(a, b)
    survivors = select_survivors([a, b], latest_step=10000)
    assert survivors == {5000, 10000}


def test_small_samples_almost_never_separate():
    # The exact case observed on the base checkpoint: 2/8 then 1/8. Nothing may be
    # deleted on evidence this thin.
    a = Evaluated(step=5000, successes=2, episodes=8)
    b = Evaluated(step=10000, successes=1, episodes=8)
    assert not clearly_worse(b, a)
    assert select_survivors([a, b], latest_step=10000) == {5000, 10000}


def test_best_is_kept_even_if_it_is_not_the_latest():
    entries = [
        Evaluated(step=5000, successes=64, episodes=128),
        Evaluated(step=10000, successes=8, episodes=128),
        Evaluated(step=15000, successes=10, episodes=128),
    ]
    survivors = select_survivors(entries, latest_step=15000)
    assert 5000 in survivors
    assert 15000 in survivors
    assert 10000 not in survivors


def test_ties_prefer_the_later_step_as_best_but_keep_both():
    a = Evaluated(step=5000, successes=40, episodes=128)
    b = Evaluated(step=10000, successes=40, episodes=128)
    survivors = select_survivors([a, b], latest_step=10000)
    assert survivors == {5000, 10000}


@pytest.mark.parametrize("episodes", [8, 32, 128])
def test_a_checkpoint_is_never_deleted_when_it_is_the_best(episodes):
    entries = [
        Evaluated(step=5000, successes=episodes // 2, episodes=episodes),
        Evaluated(step=10000, successes=0, episodes=episodes),
    ]
    assert 5000 in select_survivors(entries, latest_step=10000)
