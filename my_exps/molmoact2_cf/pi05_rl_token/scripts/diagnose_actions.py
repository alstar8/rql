"""Why does the actor's motion look jerky? Read it off the buffer, no rollouts.

Each stored row holds C consecutive *executed* actions, so a saved buffer
already contains the executed trajectory in overlapping windows. Three things
are worth separating:

  profile     step-to-step change at each position inside the window. The
              reference chunk is always one VLA plan and is internally smooth;
              the executed window is spliced from two committed chunks, so a
              seam shows up as a bump at one position.
  lag         how much the deviation (executed - reference) at one decision
              predicts the next. Near zero means the correction is redrawn
              every time -- approximation noise, not a learned edit.
  per-dim     which coordinates the deviation lives in, and whether the
              gripper leaves [0, 1], where the client maps it to 0..255.

    python scripts/diagnose_actions.py --buffer runs/online_134/buffer.npz --warmup_rows 3784
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rlt.config import ACTION_DIM, CHUNK  # noqa: E402


def chunks(flat: np.ndarray) -> np.ndarray:
    """(N, C*A) -> (N, C, A)"""
    return flat.reshape(len(flat), CHUNK, ACTION_DIM)


def step_profile(flat: np.ndarray, dims: slice) -> np.ndarray:
    """Mean |a_{i+1} - a_i| at each of the C-1 positions inside the window."""
    seq = chunks(flat)[:, :, dims]
    return np.abs(np.diff(seq, axis=1)).mean(axis=(0, 2))


def report_profile(name: str, action: np.ndarray, reference: np.ndarray) -> None:
    for label, dims in (("joints", slice(0, 7)), ("gripper", slice(7, 8))):
        executed = step_profile(action, dims)
        planned = step_profile(reference, dims)
        print(f"  {name:<8} {label:<8} executed " + " ".join(f"{v:.4f}" for v in executed))
        print(f"  {'':<8} {'':<8} planned  " + " ".join(f"{v:.4f}" for v in planned))
        ratio = executed / np.maximum(planned, 1e-9)
        print(f"  {'':<8} {'':<8} ratio    " + " ".join(f"{v:.2f}" for v in ratio))


def deviation_lag(action: np.ndarray, reference: np.ndarray, max_lag: int = 8) -> list[float]:
    """Correlation between the deviation of row i and row i+lag.

    Rows are stored in decision order, `stride` env steps apart, so lag 4 is one
    chunk at C=8, stride=2. Rows spanning an episode boundary add noise, which
    only pushes the correlation down.
    """
    dev = (action - reference).astype(np.float64)
    dev -= dev.mean(axis=0, keepdims=True)
    out = []
    for lag in range(1, max_lag + 1):
        a, b = dev[:-lag], dev[lag:]
        num = (a * b).sum()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        out.append(float(num / den) if den else 0.0)
    return out


def describe_stream(path: str) -> None:
    """Step-to-step change of what was actually commanded, from rlt.evaluate.

    Split by position relative to a replan: index 0 of each plan is where a new
    plan takes over, so a seam shows up there and nowhere else.
    """
    data = np.load(path)
    actions, lengths = data["actions"], data["lengths"]
    chunk = int(data["chunk"])
    successes = data["successes"]

    seam, inside = {"joints": [], "gripper": []}, {"joints": [], "gripper": []}
    start = 0
    for length in lengths:
        episode = actions[start : start + length]
        start += length
        delta = np.abs(np.diff(episode, axis=0))  # (T-1, A); row i is the step i -> i+1
        for label, dims in (("joints", slice(0, 7)), ("gripper", slice(7, 8))):
            values = delta[:, dims].mean(axis=1)
            at_seam = (np.arange(1, len(episode)) % chunk) == 0
            seam[label].append(values[at_seam])
            inside[label].append(values[~at_seam])

    print(f"{Path(path).name}: {len(lengths)} rollouts, chunk={chunk}, "
          f"success {int(successes.sum())}/{len(successes)}, {int(lengths.sum())} steps")
    for label in ("joints", "gripper"):
        s, i = np.concatenate(seam[label]), np.concatenate(inside[label])
        s_mean = s.mean() if s.size else float("nan")
        print(
            f"  {label:<8} within a plan {i.mean():.4f}   at a replan {s_mean:.4f}"
            f"   ratio {s_mean / max(i.mean(), 1e-9):.2f}   (n={i.size} / {s.size})"
        )
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--streams", nargs="*", default=[], help="actions_*.npz from rlt.evaluate")
    ap.add_argument("--buffer", default="")
    ap.add_argument(
        "--warmup_rows",
        type=int,
        default=0,
        help="rows collected before the actor took over; the log's buffer size at that episode",
    )
    args = ap.parse_args()

    for path in args.streams:
        describe_stream(path)
    if not args.buffer:
        return

    data = np.load(args.buffer)
    size = int(data["size"])
    action, reference = data["action"][:size], data["reference"][:size]
    split = min(args.warmup_rows, size)
    arms = {"warmup": (action[:split], reference[:split]), "actor": (action[split:], reference[split:])}
    print(f"{size} rows: {split} warmup (the VLA acting), {size - split} actor\n")

    print("step-to-step change by position inside the C-step window")
    print("  (executed windows are spliced from two committed chunks; planned ones are not)")
    for name, (a, r) in arms.items():
        report_profile(name, a, r)
    print()

    print("deviation autocorrelation by lag (lag 4 = one chunk apart)")
    for name, (a, r) in arms.items():
        print(f"  {name:<8} " + " ".join(f"{v:+.3f}" for v in deviation_lag(a, r)))
    print()

    print("deviation RMS per action dimension (q1..q7, gripper)")
    for name, (a, r) in arms.items():
        dev = (chunks(a) - chunks(r)).reshape(-1, ACTION_DIM)
        print(f"  {name:<8} " + " ".join(f"{v:.4f}" for v in np.sqrt((dev**2).mean(axis=0))))
    print()

    print("gripper command range (the client maps this to 0..255 without clipping)")
    for name, (a, r) in arms.items():
        g_exec, g_ref = chunks(a)[:, :, 7].ravel(), chunks(r)[:, :, 7].ravel()
        outside = float(((g_exec < 0) | (g_exec > 1)).mean())
        print(
            f"  {name:<8} executed [{g_exec.min():+.3f}, {g_exec.max():+.3f}] "
            f"outside [0,1]: {outside:.1%}   reference [{g_ref.min():+.3f}, {g_ref.max():+.3f}]"
        )


if __name__ == "__main__":
    main()
