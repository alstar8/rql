"""Plumbing between the learner and its collector processes.

Collectors need CUDA (the token encoder), and torch requires the `spawn` start
method for that -- forked children hang or refuse to re-initialise CUDA. So
nothing is shared through memory. Two channels instead, both boring:

    rows      collector -> learner, over a queue: a few hundred KB per episode
    weights   learner -> collector, through one file replaced atomically

A collector refreshes its actor at every chunk boundary, so it acts with weights
at most a few seconds old -- the same staleness the paper's asynchronous learner
has, arrived at differently.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


class EpisodeCounter:
    """Hands out episode numbers so K collectors never run the same one twice."""

    def __init__(self, value) -> None:
        self._value = value  # multiprocessing.Value("i"), passed to each collector

    def claim(self) -> int:
        with self._value.get_lock():
            index = self._value.value
            self._value.value += 1
            return index


def atomic_torch_save(obj, path: str | Path) -> None:
    """Write a checkpoint to `path` without leaving a half-written file.

    Collectors load `actor_live.pt` while the learner rewrites it. `torch.save`
    truncates first, so a reader can see a torn zip (`PytorchStreamReader
    failed reading file data/259`). Same-directory `os.replace` is atomic: the
    collector either gets the previous file or the new one.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, dest)


class WeightLink:
    """One file the learner writes and the collectors read.

    `os.replace` is atomic, so a collector either sees the previous weights or
    the new ones, never a half-written file. A counter in a sidecar file says
    which: modification times are not reliable here, because the filesystem's
    timestamp granularity can be coarser than the gap between two publishes and
    a collector would then keep acting on stale weights.

    The counter is written after the weights, so seeing a new version means the
    weights behind it are already in place.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.version_path = self.path.with_suffix(".version")
        self.version = -1

    def publish(self, module: torch.nn.Module) -> None:
        self.version += 1
        tmp = self.path.with_suffix(".tmp")
        torch.save({k: v.detach().cpu() for k, v in module.state_dict().items()}, tmp)
        os.replace(tmp, self.path)
        tmp_version = self.version_path.with_suffix(".tmp")
        tmp_version.write_text(str(self.version))
        os.replace(tmp_version, self.version_path)

    def refresh(self, module: torch.nn.Module, device: str = "cpu") -> bool:
        """Reload if the writer has published since last time. True if weights moved."""
        try:
            version = int(self.version_path.read_text())
        except (FileNotFoundError, ValueError):
            return False
        if version == self.version:
            return False
        state = torch.load(self.path, map_location=device, weights_only=True)
        module.load_state_dict(state)
        self.version = version
        return True
