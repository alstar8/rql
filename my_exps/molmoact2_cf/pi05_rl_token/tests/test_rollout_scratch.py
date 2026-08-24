"""Two runners must not share a scratch directory.

Episode output goes to `<tmp>/ep_000001`, numbered from one per runner and
deleted after the episode. Two eval shards pointed at the same root therefore
delete each other's output; the success counts survive (they come from memory)
but the videos quietly vanish, which is how a 16-rollout grid came out with 14
clips.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _import_rollout_without_molmospaces(monkeypatch):
    """rlt.rollout imports MolmoSpaces at module level; stub it for a CPU test."""
    for name in ("molmo_spaces", "molmo_spaces.evaluation", "molmo_spaces.evaluation.eval_main"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["molmo_spaces.evaluation.eval_main"].run_evaluation = lambda **kw: None

    stub = types.ModuleType("rlt.eval_config")
    stub.RLTokenEvalConfig = object
    stub.default_benchmark_dir = lambda: Path(".")
    monkeypatch.setitem(sys.modules, "rlt.eval_config", stub)

    monkeypatch.delitem(sys.modules, "rlt.rollout", raising=False)
    import rlt.rollout as rollout

    return rollout


def test_each_runner_gets_its_own_scratch_directory(tmp_path, monkeypatch):
    rollout = _import_rollout_without_molmospaces(monkeypatch)

    first = rollout.EpisodeRunner(policy=None, benchmark_dir=".", horizon=1, tmp_dir=str(tmp_path))
    second = rollout.EpisodeRunner(policy=None, benchmark_dir=".", horizon=1, tmp_dir=str(tmp_path))

    assert first.tmp_dir != second.tmp_dir
    assert first.tmp_dir.is_dir() and second.tmp_dir.is_dir()
    assert first.tmp_dir.parent == tmp_path
