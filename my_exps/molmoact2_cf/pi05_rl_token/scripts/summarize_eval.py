"""Readable success rates from an eval run: overall, per task, and paired.

MolmoSpaces never writes a results file -- `run_evaluation` collects episode
results in memory and returns them, so a command-line run leaves only HDF5
trajectories behind. This reads those directly (via the same
`collect_episode_results` the library uses) and writes `results.json` next to
them, which `make_grid_video.py` then uses for its success borders.

    python scripts/summarize_eval.py --eval_dir eval_output/rlt_reference_128ep

Two runs paired per episode (same episodes in both):

    python scripts/summarize_eval.py \
        --eval_dir eval_output/rlt_reference_128ep \
        --compare  eval_output/rlt_actor_128ep

Add --tb_dir to also emit the headline numbers as TensorBoard scalars.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running as `python scripts/x.py` puts scripts/ on sys.path, not the package root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
from collections import defaultdict


def find_run_dir(eval_dir: Path) -> Path:
    """The directory that actually holds house_* -- eval_main nests it under
    <output_dir>/<ConfigName>/<timestamp>/."""
    if any(eval_dir.glob("house_*")):
        return eval_dir
    candidates = sorted({p.parent for p in eval_dir.glob("**/house_*")})
    if not candidates:
        raise FileNotFoundError(f"no house_* directories under {eval_dir}")
    return candidates[-1]  # newest timestamp sorts last


def load(eval_dir: Path, rebuild: bool = False) -> dict:
    """Read results.json, or rebuild it from the HDF5 trajectories and cache it."""
    run_dir = find_run_dir(eval_dir)
    results_path = run_dir / "results.json"

    if results_path.exists() and not rebuild:
        return json.loads(results_path.read_text())

    from molmo_spaces.utils.eval_utils import collect_episode_results

    print(f"building {results_path} from HDF5 ...")
    episodes = collect_episode_results(run_dir)
    if not episodes:
        raise FileNotFoundError(f"no episode data under {run_dir}")

    rows = [
        {
            "house_id": str(e.house_id),
            "episode_idx": int(e.episode_idx),
            "success": bool(e.success),
            "oracle_done": bool(e.oracle_done) if e.oracle_done is not None else bool(e.success),
            "num_steps": int(e.num_steps),
            "task_description": e.task_description,
            "object_name": e.object_name,
        }
        for e in episodes
    ]
    data = {
        "run_dir": str(run_dir),
        "num_episodes": len(rows),
        "success_count": sum(r["success"] for r in rows),
        "oracle_success_count": sum(r["oracle_done"] for r in rows),
        "mean_steps": sum(r["num_steps"] for r in rows) / len(rows),
        "episodes": rows,
    }
    data["success_rate"] = data["success_count"] / data["num_episodes"]
    data["oracle_success_rate"] = data["oracle_success_count"] / data["num_episodes"]
    results_path.write_text(json.dumps(data, indent=2))
    print(f"wrote {results_path}")
    return data


def task_of(episode: dict) -> str:
    """'pick up the kettle.' -> 'kettle'."""
    text = str(episode.get("task_description") or "").strip().rstrip(".").lower()
    return text[len("pick up the "):] if text.startswith("pick up the ") else (text or "unknown")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a rate. n is small here, so no normal approximation."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(data: dict, name: str) -> dict:
    eps = data["episodes"]
    n = len(eps)
    k = sum(bool(e["success"]) for e in eps)
    ko = sum(bool(e.get("oracle_done", e["success"])) for e in eps)
    lo, hi = wilson(k, n)

    print(f"\n=== {name}")
    print(f"  at-end SR   {k:3d}/{n:<4d} = {k / n:6.1%}   95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  oracle SR   {ko:3d}/{n:<4d} = {ko / n:6.1%}")
    print(f"  mean steps  {data.get('mean_steps', float('nan')):.0f}")

    by_task: dict[str, list[bool]] = defaultdict(list)
    for e in eps:
        by_task[task_of(e)].append(bool(e["success"]))

    print(f"\n  by task ({len(by_task)} distinct, sorted by SR):")
    print(f"    {'task':<26} {'n':>4} {'ok':>4} {'SR':>7}")
    rows = sorted(by_task.items(), key=lambda kv: (-sum(kv[1]) / len(kv[1]), -len(kv[1])))
    for task, outcomes in rows:
        ok, tot = sum(outcomes), len(outcomes)
        print(f"    {task[:26]:<26} {tot:>4} {ok:>4} {ok / tot:>6.0%}")

    solved = [t for t, o in by_task.items() if any(o)]
    never = [t for t, o in by_task.items() if not any(o)]
    print(f"\n  tasks with at least one success: {len(solved)}/{len(by_task)}")
    if never:
        shown = ", ".join(sorted(never)[:10])
        print(f"  never solved ({len(never)}): {shown}" + (" ..." if len(never) > 10 else ""))

    return {"n": n, "success": k, "oracle": ko, "sr": k / n}


def compare(a: dict, b: dict, name_a: str, name_b: str) -> None:
    """Paired on (house, episode_idx): only episodes present in both count."""
    key = lambda e: (e["house_id"], e["episode_idx"])  # noqa: E731
    ma = {key(e): bool(e["success"]) for e in a["episodes"]}
    mb = {key(e): bool(e["success"]) for e in b["episodes"]}
    shared = sorted(set(ma) & set(mb))

    print(f"\n=== paired comparison on {len(shared)} shared episodes")
    if len(shared) < len(ma) or len(shared) < len(mb):
        print(f"  note: {name_a} has {len(ma)}, {name_b} has {len(mb)} -- comparing the overlap")

    sa = sum(ma[k] for k in shared)
    sb = sum(mb[k] for k in shared)
    both = sum(ma[k] and mb[k] for k in shared)
    only_a = sum(ma[k] and not mb[k] for k in shared)
    only_b = sum(mb[k] and not ma[k] for k in shared)
    neither = len(shared) - both - only_a - only_b

    print(f"  {name_a:<26} {sa:3d}/{len(shared)} = {sa / max(1, len(shared)):.1%}")
    print(f"  {name_b:<26} {sb:3d}/{len(shared)} = {sb / max(1, len(shared)):.1%}")
    print(f"\n  both solved  {both:3d}")
    print(f"  only first   {only_a:3d}")
    print(f"  only second  {only_b:3d}")
    print(f"  neither      {neither:3d}")
    print(f"\n  delta (second - first): {(sb - sa) / max(1, len(shared)):+.1%}")

    disagree = only_a + only_b
    if disagree == 0:
        print("  the arms never disagree -- no evidence of any difference")
        return
    # Exact two-sided McNemar: under H0 each disagreement is a coin flip.
    tail = sum(math.comb(disagree, i) for i in range(0, min(only_a, only_b) + 1))
    p = min(1.0, 2 * tail / (2**disagree))
    print(f"  McNemar exact p = {p:.4f} on {disagree} disagreeing episodes")
    print("  -> " + ("significant at 0.05" if p <= 0.05 else "within noise at this n"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_dir", required=True)
    ap.add_argument("--compare", default="", help="second eval dir to pair against")
    ap.add_argument("--rebuild", action="store_true", help="ignore a cached results.json")
    ap.add_argument("--tb_dir", default="", help="also write headline SR as TB scalars")
    args = ap.parse_args()

    a_dir = Path(args.eval_dir)
    a = load(a_dir, args.rebuild)
    stats_a = summarize(a, a_dir.name)
    stats = {a_dir.name: stats_a}

    if args.compare:
        b_dir = Path(args.compare)
        b = load(b_dir, args.rebuild)
        stats[b_dir.name] = summarize(b, b_dir.name)
        compare(a, b, a_dir.name, b_dir.name)

    if args.tb_dir:
        from torch.utils.tensorboard import SummaryWriter

        for name, s in stats.items():
            writer = SummaryWriter(log_dir=str(Path(args.tb_dir) / name))
            writer.add_scalar("eval/success_rate", s["sr"], 0)
            writer.add_scalar("eval/oracle_rate", s["oracle"] / s["n"], 0)
            writer.add_scalar("eval/episodes", s["n"], 0)
            writer.close()
        print(f"\ntensorboard scalars -> {args.tb_dir}")


if __name__ == "__main__":
    main()
