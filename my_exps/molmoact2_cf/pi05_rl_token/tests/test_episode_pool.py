"""Which benchmark episodes a run trains on.

The held-out test set is episodes 0-127. Training on any of them would make
every later benchmark number meaningless, so the guard is a test, not a comment.
"""

from __future__ import annotations

from rlt.cli import parse_episode_spec
from rlt.config import OnlineConfig


def test_spec_parsing():
    assert parse_episode_spec("128-131,140") == [128, 129, 130, 131, 140]
    assert parse_episode_spec("134") == [134]
    assert parse_episode_spec("") == []


def test_a_pool_wins_over_the_single_episode():
    cfg = OnlineConfig(episode_idx=134, episode_pool="200-203")
    assert cfg.training_episodes() == [200, 201, 202, 203]


def test_single_episode_when_no_pool():
    assert OnlineConfig(episode_idx=134).training_episodes() == [134]


def test_test_set_episodes_are_refused():
    assert "held-out test set" in OnlineConfig(episode_idx=127).validate()
    assert "held-out test set" in OnlineConfig(episode_idx=134, episode_pool="120-140").validate()
    assert OnlineConfig(episode_idx=134, episode_pool="128-159").validate() == ""


def test_a_variant_benchmark_has_its_own_numbering():
    """The 0-127 guard protects the default benchmark. A variant benchmark holds
    variants of one episode, numbered from zero, with its own held-out tail."""
    cfg = OnlineConfig(episode_idx=-1, episode_pool="0-999", benchmark_dir="runs/variants/ep134")
    assert cfg.validate() == ""
    assert OnlineConfig(episode_idx=-1, episode_pool="0-999").validate() != ""


def test_round_robin_visits_every_episode_equally():
    pool = OnlineConfig(episode_idx=-1, episode_pool="128-131").training_episodes()
    visited = [pool[i % len(pool)] for i in range(40)]
    assert {e: visited.count(e) for e in pool} == {128: 10, 129: 10, 130: 10, 131: 10}
