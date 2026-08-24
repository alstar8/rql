"""Decide which training checkpoints to keep, based on measured success rate.

Each pi0.5 checkpoint costs 48 GB, so a 40k-step run saving every 5000 steps would leave
about 380 GB behind. This keeps two: the latest, so training can always be resumed and
the final model is available, and the best measured so far.

The delicate part is deletion. Success rate is measured on a finite sample and the
flow-matching sampler is stochastic -- the same base checkpoint scored 2/8 and then 1/8
on repeat runs. Ranking two checkpoints by point estimate alone would therefore throw
away good models on noise. A checkpoint is only deleted when its Wilson interval does not
overlap the best one's: when the intervals touch, both are kept until the evidence
separates them.

Nothing here removes files; it returns the decision, and the caller does the deleting.
"""

from __future__ import annotations

from dataclasses import dataclass


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: behaves sensibly at 0% and 100%, unlike the normal one."""
    if total <= 0:
        return 0.0, 1.0
    phat = successes / total
    denom = 1 + z**2 / total
    centre = (phat + z**2 / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) / total + z**2 / (4 * total**2)) ** 0.5) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


@dataclass(frozen=True)
class Evaluated:
    """One checkpoint and what it scored."""

    step: int
    successes: int
    episodes: int

    @property
    def rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.episodes)


def clearly_worse(candidate: Evaluated, best: Evaluated) -> bool:
    """True only when the two intervals do not overlap.

    Requiring separation rather than a lower point estimate is what stops the curator
    from deleting a checkpoint that merely got unlucky on its evaluation sample.
    """
    _, candidate_high = candidate.interval
    best_low, _ = best.interval
    return candidate_high < best_low


def select_survivors(evaluated: list[Evaluated], *, latest_step: int | None = None) -> set[int]:
    """Return the steps worth keeping.

    Always keeps the latest checkpoint -- it is the resume point and the final model --
    and the best scoring one. Everything else survives unless it is clearly worse than
    the best.
    """
    if not evaluated:
        return set() if latest_step is None else {latest_step}

    best = max(evaluated, key=lambda e: (e.rate, e.step))
    survivors = {best.step}
    if latest_step is not None:
        survivors.add(latest_step)

    for entry in evaluated:
        if entry.step in survivors:
            continue
        if not clearly_worse(entry, best):
            survivors.add(entry.step)

    return survivors


def describe(evaluated: list[Evaluated], survivors: set[int]) -> str:
    """A human-readable account of the decision, for the log and the ledger."""
    if not evaluated:
        return "nothing evaluated yet"
    best = max(evaluated, key=lambda e: (e.rate, e.step))
    lines = []
    for entry in sorted(evaluated, key=lambda e: e.step):
        low, high = entry.interval
        mark = "keep" if entry.step in survivors else "DELETE"
        note = ""
        if entry.step == best.step:
            note = "  <- best"
        elif entry.step not in survivors:
            note = f"  (upper {high:.1%} < best lower {best.interval[0]:.1%})"
        lines.append(
            f"  step {entry.step:>7,}  {entry.successes:>3}/{entry.episodes:<4} "
            f"= {entry.rate:6.1%}  [{low:5.1%}, {high:5.1%}]  {mark}{note}"
        )
    return "\n".join(lines)
