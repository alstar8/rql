"""The learner <-> collector channels: weights out, episode numbers apart.

Both fail quietly if they break -- collectors would keep acting with the initial
actor forever, or two of them would run the same episode.
"""

from __future__ import annotations

import multiprocessing as mp

import torch

from rlt.shared import EpisodeCounter, WeightLink


def _claim(value, out):
    counter = EpisodeCounter(value)
    out.extend([counter.claim() for _ in range(5)])  # a proxy pickles its argument


def test_episode_numbers_are_handed_out_once_each():
    ctx = mp.get_context("spawn")
    value = ctx.Value("i", 0)
    manager = mp.Manager()
    lists = [manager.list() for _ in range(3)]
    workers = [ctx.Process(target=_claim, args=(value, lst)) for lst in lists]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(60)

    claimed = sorted(n for lst in lists for n in lst)
    assert claimed == list(range(15))
    assert value.value == 15


def test_weights_move_from_writer_to_reader(tmp_path):
    writer = torch.nn.Linear(4, 3)
    reader = torch.nn.Linear(4, 3)
    with torch.no_grad():
        reader.weight.fill_(0.0)

    link_out = WeightLink(tmp_path / "actor.pt")
    link_in = WeightLink(tmp_path / "actor.pt")
    link_out.publish(writer)

    assert link_in.refresh(reader) is True
    assert torch.allclose(reader.weight, writer.weight)
    assert link_in.refresh(reader) is False, "unchanged weights must not be reloaded"

    with torch.no_grad():
        writer.weight.add_(1.0)
    link_out.publish(writer)
    assert link_in.refresh(reader) is True
    assert torch.allclose(reader.weight, writer.weight)


def test_back_to_back_publishes_are_not_collapsed(tmp_path):
    """Two publishes inside one filesystem timestamp tick must still both land:
    a collector that misses one keeps acting on weights the learner replaced."""
    writer = torch.nn.Linear(4, 3)
    reader = torch.nn.Linear(4, 3)
    link_out = WeightLink(tmp_path / "actor.pt")
    link_in = WeightLink(tmp_path / "actor.pt")

    for _ in range(5):  # no sleeps: this is the case coarse mtimes would swallow
        with torch.no_grad():
            writer.weight.add_(1.0)
        link_out.publish(writer)
        assert link_in.refresh(reader) is True
        assert torch.allclose(reader.weight, writer.weight)


def test_missing_weight_file_is_not_an_error(tmp_path):
    """A collector may start before the learner has published anything."""
    reader = torch.nn.Linear(4, 3)
    assert WeightLink(tmp_path / "absent.pt").refresh(reader) is False


def test_atomic_torch_save_leaves_no_tmp_and_roundtrips(tmp_path):
    from rlt.shared import atomic_torch_save

    path = tmp_path / "actor_live.pt"
    atomic_torch_save({"k": torch.tensor([1.0, 2.0])}, path)
    loaded = torch.load(path, weights_only=False)
    assert torch.equal(loaded["k"], torch.tensor([1.0, 2.0]))
    assert not (tmp_path / "actor_live.pt.tmp").exists()
