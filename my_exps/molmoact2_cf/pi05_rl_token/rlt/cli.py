"""Turn a dataclass into a CLI, so every hyperparameter has exactly one home.

Keeps configs greppable: if a knob is not in config.py, it does not exist.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")


def parse_episode_spec(text: str) -> list[int]:
    """'128-131,140' -> [128, 129, 130, 131, 140]. Empty string -> []."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def parse_into(cls: type[T], argv: list[str] | None = None) -> T:
    """Build `cls` from command-line flags named after its fields."""
    parser = argparse.ArgumentParser(description=cls.__doc__)
    for f in dataclasses.fields(cls):
        if f.type is bool or f.type == "bool":
            # --flag / --no-flag, so booleans cannot be set by accident.
            parser.add_argument(f"--{f.name}", action="store_true", default=None)
            parser.add_argument(f"--no_{f.name}", dest=f.name, action="store_false")
        else:
            ftype = {"int": int, "float": float, "str": str}.get(str(f.type), f.type)
            parser.add_argument(f"--{f.name}", type=ftype, default=None)

    args = parser.parse_args(argv)
    given = {k: v for k, v in vars(args).items() if v is not None}
    cfg = cls(**given)

    validate = getattr(cfg, "validate", None)
    if callable(validate):
        problem = validate()
        if problem:
            parser.error(problem)
    return cfg


def save_config(cfg, path: Path) -> None:
    """Dump the resolved config next to the checkpoint. Every run is reproducible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclasses.asdict(cfg), indent=2, default=str))


def describe(cfg) -> str:
    rows = [f"  {f.name:20s} {getattr(cfg, f.name)}" for f in dataclasses.fields(cfg)]
    return f"{type(cfg).__name__}:\n" + "\n".join(rows)
