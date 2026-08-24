"""Is one success rate actually higher than the other, or is it n?

Milestone 1 compares the trained actor against the frozen VLA on the same
episode. Both are measured as independent rollouts, so this is a two-proportion
comparison -- Fisher's exact test, no normal approximation, because 60 rollouts
at 40% has a confidence interval nearly 24 points wide.

    python scripts/compare_rates.py \
        --baseline "runs/episode_profile*/episode_134_s*.json" \
        --actor    "runs/eval_actor_134/episode_134_s*.json"

Each argument is a glob over the JSON files `rlt.evaluate` writes; shards of the
same episode are merged.
"""

from __future__ import annotations

import argparse
import json
import sys
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlt.evaluate import wilson  # noqa: E402


def merge(pattern: str) -> tuple[int, int, list[str], list[int]]:
    """Successes, rollouts, files, and the step counts of the successful rollouts.

    Steps are collected only for successes: a failure always runs the full
    horizon, so mean steps over all rollouts mostly measures the success rate
    again rather than how quickly the task got done.
    """
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"no result files match {pattern!r}")
    successes = rollouts = 0
    steps_ok: list[int] = []
    for path in files:
        data = json.loads(Path(path).read_text())
        successes += int(data["successes"])
        rollouts += int(data["rollouts"])
        per_rollout = data.get("steps_per_rollout") or []
        mask = data.get("successes_mask") or []
        steps_ok += [s for s, ok in zip(per_rollout, mask) if ok]
    return successes, rollouts, files, steps_ok


def report(name: str, k: int, n: int, files: list[str]) -> None:
    lo, hi = wilson(k, n)
    print(f"{name:<10} {k:>3}/{n:<4} = {k / n:>6.1%}   95% CI [{lo:.1%}, {hi:.1%}]   ({len(files)} files)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="glob of frozen-VLA result files")
    ap.add_argument("--actor", required=True, help="glob of trained-actor result files")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    base_k, base_n, base_files, base_steps = merge(args.baseline)
    actor_k, actor_n, actor_files, actor_steps = merge(args.actor)

    print()
    report("baseline", base_k, base_n, base_files)
    report("actor", actor_k, actor_n, actor_files)
    print(f"\ndifference: {actor_k / actor_n - base_k / base_n:+.1%}")

    from scipy.stats import fisher_exact, mannwhitneyu

    table = [[actor_k, actor_n - actor_k], [base_k, base_n - base_k]]
    _, p_two = fisher_exact(table)
    _, p_greater = fisher_exact(table, alternative="greater")
    print(f"Fisher exact: two-sided p = {p_two:.4f}, one-sided (actor > baseline) p = {p_greater:.4f}")
    print(
        "verdict: "
        + (
            "the actor is above the baseline"
            if p_greater <= args.alpha
            else f"not distinguishable from the baseline at alpha={args.alpha}"
        )
    )

    if base_steps and actor_steps:
        import statistics

        print(
            f"\nsteps to success (successful rollouts only): "
            f"baseline median {statistics.median(base_steps):.0f} (n={len(base_steps)}), "
            f"actor median {statistics.median(actor_steps):.0f} (n={len(actor_steps)})"
        )
        p_faster = mannwhitneyu(actor_steps, base_steps, alternative="less")[1]
        print(f"Mann-Whitney (actor faster) p = {p_faster:.4f}")
    else:
        print("\nno per-rollout step counts in these files, so time-to-success is not compared")


if __name__ == "__main__":
    main()
