"""CLI: MolmoBot FrankaPick H5 → stratified CF buffer (.npz)."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffer import build_stratified_arrays, save_buffer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Root with extracted FrankaPickOmniCamConfig H5 packages",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("runs/pick_buffer.npz"),
    )
    p.add_argument("--max_episodes", type=int, default=200)
    p.add_argument("--pos_frac", type=float, default=0.4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    arrays = build_stratified_arrays(
        args.data_dir,
        max_episodes=args.max_episodes,
        pos_frac=args.pos_frac,
        gamma=args.gamma,
        seed=args.seed,
    )
    save_buffer(arrays, args.out)
    n = int(arrays["n_transitions"][0])
    print(
        f"Wrote {args.out}: transitions={n} "
        f"pos_eps={int(arrays['n_pos_eps'][0])} "
        f"neg_eps={int(arrays['n_neg_eps'][0])} "
        f"success_frac={float(arrays['successes'].mean()):.3f}"
    )


if __name__ == "__main__":
    main()
