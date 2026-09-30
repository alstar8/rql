#!/usr/bin/env python
"""V25: distill the deployed teacher (frozen expert + V + G) into the pi0.5 action expert.

The teacher's corrected chunks were recorded by `Pi05RLPolicy` (`distill_out`) during an
online round. This script trains a *student* copy of the pi0.5 action expert so that
`student.sample_actions(obs) ≈ teacher`, leaving the PaliGemma backbone bit-identical so
the token AE and the RL state survive the swap.

    CUDA_VISIBLE_DEVICES=0 python scripts/train_expert_distill.py \
        --data runs/pick18_v25_distill/round1/distill \
        --init /path/to/base/or/previous/expert \
        --out runs/pick18_v25_distill/expert_e1 --steps 12000

WHAT IS TRAINED AND WHAT IS FROZEN
----------------------------------
Frozen: `paligemma_with_expert.paligemma.*` (vision tower, language model, projector).
Trained: `gemma_expert.*` (the 300M action expert) plus `action_in/out_proj`,
`state_proj`, and the adaRMS time MLPs. The backbone never moves, so prefix tokens are
unchanged and the shared AE stays valid across the swap.

THE TARGET
----------
The teacher commits 8 steps at the current observation. A recorded shard may also
store a stitched tail (gOn at the next observation) for analysis. The training
target keeps only the committed 8 and fills steps [8, 16) with the frozen expert's
unused tail. The loss is flow-matching MSE on steps [0, 8) and the 8 real action
dims. Eval replans every 8, so the tail is never executed and is not supervised.

SELF-CHECK (runs before any training, gates it)
-----------------------------------------------
With the student still equal to its init, the masked flow-matching loss on the recorded
*reference* chunks must be low (they are genuine samples of the frozen model -- this
validates the entire preprocessing + action-normalisation chain), and the loss on
*teacher* chunks must be higher (the correction moved the action). If the reference loss
is not low, the pipeline is wrong and the run aborts instead of training on garbage.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pi05.expert_io import freeze_backbone  # noqa: E402

log = logging.getLogger("train_expert_distill")

BASE_CHECKPOINT = "/home/jovyan/users/staroverov/pi05_molmospaces/checkpoints_pytorch/pi05_droid_finetune_pick_full_v3_39999"
OPENPI_CONFIG = "pi05_droid_finetune"
CHUNK = 8          # teacher commits this many of the 16-step horizon
ACTION_DIM = 8     # real action dims; the model pads to 32
HORIZON = 16


def build_policy(checkpoint: str):
    """Load the model and the transform pipeline exactly as the server does.

    openpi and pi05.eval are imported here rather than at module top for the same reason
    serve_pi05_http.py does it: importing them is what triggers HF-hub resolution and
    CUDA context setup, which must not happen until the environment (HF_HOME, single
    visible GPU) is in place.
    """
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    from pi05.eval import disable_policy_compile

    disable_policy_compile()
    train_config = _config.get_config(OPENPI_CONFIG)
    policy = _policy_config.create_trained_policy(
        train_config, checkpoint, pytorch_device="cuda"
    )
    return policy, policy._model


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------


def shard_paths(data_dir: str) -> list[Path]:
    paths = sorted(Path(data_dir).glob("distill_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no distill_*.npz shards in {data_dir}")
    return paths


def load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sample_count(paths: list[Path]) -> int:
    total = 0
    for p in paths:
        with np.load(p) as z:
            total += int(z["teacher"].shape[0])
    return total


class ShardStreamer:
    """Yields (raw_obs, teacher, reference) one decision at a time, shard-shuffled.

    One shard (~256 decisions) is resident at a time, so an arbitrarily large corpus
    streams from NFS without filling RAM. Rows within a shard are shuffled too.
    """

    def __init__(self, paths: list[Path], seed: int = 0) -> None:
        self.paths = list(paths)
        self.rng = np.random.default_rng(seed)

    def iter_epoch(self):
        order = self.rng.permutation(len(self.paths))
        for shard_idx in order:
            shard = load_shard(self.paths[shard_idx])
            n = shard["teacher"].shape[0]
            for row in self.rng.permutation(n):
                raw_obs = {
                    "observation/exterior_image_1_left": shard["external_cam"][row],
                    "observation/wrist_image_left": shard["wrist_cam"][row],
                    "observation/joint_position": shard["state"][row][:7],
                    "observation/gripper_position": shard["state"][row][7:8],
                    "prompt": str(shard["instruction"][row]),
                }
                yield raw_obs, shard["teacher"][row], shard["reference"][row]


def make_target(teacher: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Committed 8-step teacher plus the frozen expert's unused tail -> (16, 8).

    Stitched shards store gOn(obs_{t+8}) in steps [8, 16) for analysis. That tail is
    not a function of the observation the student sees, so it is not the training
    target. Steps [8, 16) keep the reference chunk recorded at obs_t.
    """
    teacher = np.asarray(teacher, dtype=np.float32).reshape(-1, ACTION_DIM)
    if teacher.shape[0] < CHUNK:
        raise ValueError(f"teacher needs {CHUNK} steps, got {teacher.shape}")
    reference = np.asarray(reference, dtype=np.float32).reshape(HORIZON, ACTION_DIM)
    if reference.shape[0] < HORIZON:
        raise ValueError(f"reference needs {HORIZON} steps, got {reference.shape}")
    return np.concatenate([teacher[:CHUNK], reference[CHUNK:HORIZON]], axis=0)


def split_train_holdout(
    paths: list[Path], holdout_frac: float, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Deterministic shard split. The holdout is never empty when there are two shards."""
    paths = list(paths)
    if holdout_frac <= 0 or len(paths) < 2:
        return paths, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n_hold = max(1, int(round(len(paths) * holdout_frac)))
    n_hold = min(n_hold, len(paths) - 1)
    hold_idx = {int(i) for i in order[:n_hold]}
    train, hold = [], []
    for index, path in enumerate(paths):
        (hold if index in hold_idx else train).append(path)
    return train, hold


def prefix_flow_loss(per_step: torch.Tensor) -> torch.Tensor:
    """Mean flow-matching loss on the committed prefix and the real action dims."""
    return slice_flow_loss(per_step, CHUNK)


def slice_flow_loss(per_step: torch.Tensor, n_steps: int) -> torch.Tensor:
    """Mean flow-matching loss on the first `n_steps` and the real action dims."""
    steps = min(int(n_steps), int(per_step.shape[1]))
    return per_step[:, :steps, :ACTION_DIM].mean()


def to_model_inputs(policy, raw_obs: dict, actions_phys: np.ndarray):
    """One raw sample -> (transformed obs dict, normalised (16,32) actions).

    Both go through the policy's own `_input_transform`, so preprocessing is the serving
    code, not a reimplementation. The action path is quantile-normalised then padded to
    32 by the same transforms the model was trained with.
    """
    raw = dict(raw_obs)
    raw["actions"] = np.asarray(actions_phys, dtype=np.float32)
    inputs = policy._input_transform(raw)
    obs = {k: v for k, v in inputs.items() if k != "actions"}
    actions = np.asarray(inputs["actions"], dtype=np.float32)  # (16, 32)
    return obs, actions


def _stack_obs(obs_list: list[dict]) -> dict:
    """Stack a list of transformed observation dicts into a batched dict.

    jax.tree.map would do this in one line, but importing jax initialises its CUDA
    backend and it would fight torch for the card, so the two-level dict is stacked by
    hand instead.
    """
    out: dict = {}
    for key, value in obs_list[0].items():
        if isinstance(value, dict):
            out[key] = {k: np.stack([o[key][k] for o in obs_list]) for k in value}
        else:
            out[key] = np.stack([o[key] for o in obs_list])
    return out


def _to_torch(node, device):
    if isinstance(node, dict):
        return {k: _to_torch(v, device) for k, v in node.items()}
    return torch.from_numpy(np.array(node)).to(device)


def collate(batch, device):
    """Stack a list of (obs, actions) into an Observation and an action tensor."""
    from openpi.models import model as _model

    obs_list, act_list = zip(*batch)
    tensors = _to_torch(_stack_obs(obs_list), device)
    observation = _model.Observation.from_dict(tensors)
    actions = torch.from_numpy(np.stack(act_list)).to(device).float()
    return observation, actions


def masked_flow_loss(
    model, observation, actions, n_steps: int = HORIZON, endpoint_coef: float = 0.0,
) -> torch.Tensor:
    """Flow-matching MSE on the first `n_steps` and the 8 real action dims.

    The training target is gOn on steps [0, 8) and the frozen expert's own tail
    after that. Supervising all 16 steps anchors the tail to the base policy.
    `n_steps=8` drops that anchor and fits only the committed prefix.

    `endpoint_coef` adds MSE between the decoded clean action and the teacher.
    The guidance shift is ~0.002, which the velocity loss near pure noise ignores.
    """
    out = model.forward(observation, actions)
    if isinstance(out, tuple):
        per_step, x0 = out
    else:
        per_step, x0 = out, None
    flow = slice_flow_loss(per_step, n_steps)
    if endpoint_coef <= 0.0 or x0 is None:
        return flow
    steps = min(int(n_steps), int(x0.shape[1]))
    endpoint = (x0[:, :steps, :ACTION_DIM] - actions[:, :steps, :ACTION_DIM]).pow(2).mean()
    return flow + endpoint_coef * endpoint


# --------------------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------------------


@torch.no_grad()
def _mean_loss(
    model, policy, streamer, device, which: str, n_batches: int, batch_size: int, n_steps: int,
) -> float:
    model.eval()
    losses, seen = [], 0
    batch = []
    for raw_obs, teacher, reference in streamer.iter_epoch():
        target = reference if which == "reference" else make_target(teacher, reference)
        batch.append(to_model_inputs(policy, raw_obs, target))
        if len(batch) == batch_size:
            observation, actions = collate(batch, device)
            losses.append(float(masked_flow_loss(model, observation, actions, n_steps)))
            seen += len(batch)
            batch = []
            if seen >= n_batches * batch_size:
                break
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def paired_self_check(
    model, policy, streamer, device, n_batches: int, batch_size: int, n_steps: int,
) -> tuple[float, float]:
    """Reference and teacher loss on the same observations.

    Independent shuffles hide a 0.002 action correction inside the flow-matching
    noise. Pairing the two targets removes that.
    """
    model.eval()
    ref_losses, teacher_losses = [], []
    ref_batch, teacher_batch = [], []
    seen = 0
    for raw_obs, teacher, reference in streamer.iter_epoch():
        reference = np.asarray(reference, dtype=np.float32).reshape(HORIZON, ACTION_DIM)
        ref_batch.append(to_model_inputs(policy, raw_obs, reference))
        teacher_batch.append(to_model_inputs(policy, raw_obs, make_target(teacher, reference)))
        if len(ref_batch) == batch_size:
            ref_obs, ref_actions = collate(ref_batch, device)
            teacher_obs, teacher_actions = collate(teacher_batch, device)
            ref_losses.append(float(masked_flow_loss(model, ref_obs, ref_actions, n_steps)))
            teacher_losses.append(float(masked_flow_loss(model, teacher_obs, teacher_actions, n_steps)))
            seen += len(ref_batch)
            ref_batch, teacher_batch = [], []
            if seen >= n_batches * batch_size:
                break
    model.train()
    if not ref_losses:
        return float("nan"), float("nan")
    return float(np.mean(ref_losses)), float(np.mean(teacher_losses))


# --------------------------------------------------------------------------------------
# checkpoint save
# --------------------------------------------------------------------------------------


def copy_checkpoint_sidecars(init_checkpoint: str, out_dir: str) -> None:
    """Copy config.json / assets from the init checkpoint. No-op if src is dest."""
    src = Path(init_checkpoint).resolve()
    out = Path(out_dir).resolve()
    if src == out:
        return
    for extra in ("config.json",):
        if (src / extra).exists():
            shutil.copy2(src / extra, out / extra)
    if (src / "assets").exists():
        dest_assets = out / "assets"
        if dest_assets.exists():
            shutil.rmtree(dest_assets)
        shutil.copytree(src / "assets", dest_assets)


def save_checkpoint(model, init_checkpoint: str, out_dir: str) -> None:
    """Full pi0.5 checkpoint dir: model.safetensors + config.json + assets/.

    The server and the pure-VLA eval load this with zero code changes.
    After the first save, `--init` and `--out` may be the same directory (resume
    loads the student in place); skip sidecar copies in that case.
    """
    import safetensors.torch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_model(model, str(out / "model.safetensors"))
    copy_checkpoint_sidecars(init_checkpoint, out_dir)
    log.info("saved student checkpoint to %s", out)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True, help="dir with distill_*.npz shards")
    ap.add_argument("--init", default=BASE_CHECKPOINT,
                    help="checkpoint to start from (base pi0.5, or the previous round's student)")
    ap.add_argument("--out", required=True, help="output checkpoint dir for the student")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--self-check-batches", type=int, default=8)
    ap.add_argument("--max-loss-ratio", type=float, default=20.0,
                    help="abort if frozen teacher/reference loss ratio exceeds this")
    ap.add_argument("--max-reference-loss", type=float, default=0.5,
                    help="abort if the frozen model cannot reproduce the recorded reference")
    ap.add_argument("--allow-teacher-below-reference", action="store_true",
                    help="keep training when init is already closer to the teacher than to the reference")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="shard fraction held out for early stopping; 0 trains on every shard")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batches", type=int, default=16)
    ap.add_argument("--patience", type=int, default=3,
                    help="held-out evals without improvement before stopping")
    ap.add_argument("--min-steps", type=int, default=2000,
                    help="do not early-stop before this many optimiser steps")
    ap.add_argument("--min-delta", type=float, default=1e-4,
                    help="held-out loss must drop by this much to reset patience")
    ap.add_argument("--supervise-steps", type=int, default=HORIZON,
                    help="flow-matching steps to supervise; 16 anchors the reference tail, 8 fits the prefix only")
    ap.add_argument("--endpoint-coef", type=float, default=0.0,
                    help="weight of clean-action MSE beside the velocity loss; 0 keeps pure flow matching")
    ap.add_argument("--time-max", type=float, default=1.0,
                    help="sample flow time in (0, time-max]; below 1 trains near the clean action")
    ap.add_argument("--time-low-frac", type=float, default=1.0,
                    help="when time-max < 1, fraction of times drawn in (0.001, time-max]; the rest use Beta(1.5, 1)")
    args = ap.parse_args()
    if not 1 <= args.supervise_steps <= HORIZON:
        raise SystemExit(f"--supervise-steps must be in 1..{HORIZON}, got {args.supervise_steps}")
    if not 0.0 < args.time_max <= 1.0:
        raise SystemExit(f"--time-max must be in (0, 1], got {args.time_max}")
    if not 0.0 < args.time_low_frac <= 1.0:
        raise SystemExit(f"--time-low-frac must be in (0, 1], got {args.time_low_frac}")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.environ.setdefault("TORCHINDUCTOR_CUDAGRAPHS", "0")

    device = "cuda"
    paths = shard_paths(args.data)
    n = sample_count(paths)
    log.info("data: %d decisions across %d shards from %s", n, len(paths), args.data)

    policy, model = build_policy(args.init)
    n_train, n_frozen = freeze_backbone(model)
    log.info("trainable %dM (expert+proj), frozen %dM (backbone)", n_train // 10**6, n_frozen // 10**6)
    if args.endpoint_coef > 0.0:
        model.return_x0 = True
        log.info("endpoint loss coef %.3f", args.endpoint_coef)
    if args.time_max < 1.0:
        hi = float(args.time_max)
        frac = float(args.time_low_frac)
        original_sample_time = model.sample_time

        def sample_time_mixed(bsize, device, _hi=hi, _frac=frac, _orig=original_sample_time):
            low = 0.001 + torch.rand(bsize, device=device) * (_hi - 0.001)
            if _frac >= 1.0:
                return low.to(dtype=torch.float32)
            full = _orig(bsize, device)
            take_low = torch.rand(bsize, device=device) < _frac
            return torch.where(take_low, low, full).to(dtype=torch.float32)

        model.sample_time = sample_time_mixed
        log.info(
            "flow time: %.0f%% in (0.001, %.3f], %.0f%% Beta(1.5, 1)",
            100 * frac, hi, 100 * (1.0 - frac),
        )

    train_paths, hold_paths = split_train_holdout(paths, args.holdout_frac, seed=args.seed)
    log.info("split: %d train shards, %d holdout shards", len(train_paths), len(hold_paths))
    streamer = ShardStreamer(train_paths, seed=args.seed)
    hold_streamer = ShardStreamer(hold_paths, seed=args.seed + 1) if hold_paths else None

    # --- self-check: frozen reproduction of the reference, teacher above it -----------
    log.info("self-check: frozen-student loss on reference vs teacher chunks")
    t0 = time.time()
    ref_loss, teacher_loss = paired_self_check(
        model, policy, streamer, device,
        args.self_check_batches, args.batch_size, args.supervise_steps,
    )
    ratio = teacher_loss / max(ref_loss, 1e-9)
    log.info(
        "self-check (%.0fs): reference=%.4f teacher=%.4f ratio=%.2f",
        time.time() - t0, ref_loss, teacher_loss, ratio,
    )
    if not np.isfinite(ref_loss) or not np.isfinite(teacher_loss):
        raise RuntimeError("self-check produced a non-finite loss; preprocessing is broken")
    if ref_loss > args.max_reference_loss:
        raise RuntimeError(
            f"reference loss {ref_loss:.4f} exceeds {args.max_reference_loss}; "
            "the frozen expert does not reproduce the recorded reference chunks"
        )
    if teacher_loss <= ref_loss and not args.allow_teacher_below_reference:
        raise RuntimeError(
            f"teacher loss {teacher_loss:.4f} is not above reference loss {ref_loss:.4f}; "
            "the committed prefix carries no correction beyond the frozen expert"
        )
    if teacher_loss <= ref_loss:
        log.warning(
            "teacher loss %.4f is not above reference %.4f; continuing",
            teacher_loss, ref_loss,
        )
    if ratio > args.max_loss_ratio:
        raise RuntimeError(
            f"teacher/reference loss ratio {ratio:.1f} exceeds {args.max_loss_ratio}; "
            "the recorded teacher is too far from the frozen model to be a distillation "
            "target -- check that the recorder stored the corrected chunk, not noise"
        )

    # --- optimisation ------------------------------------------------------------------
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay, eps=1e-8)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * step / max(1, args.warmup)
        # cosine to 10% of peak
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t)))

    model.train()
    stream = iter_forever(streamer)
    batch: list = []
    running: list[float] = []
    started = time.time()
    best_hold = float("inf")
    best_step = 0
    stall = 0
    stopped_early = False
    hold_history: list[dict] = []
    loss = float("nan")
    for step in range(1, args.steps + 1):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        while len(batch) < args.batch_size:
            raw_obs, teacher, reference = next(stream)
            batch.append(to_model_inputs(policy, raw_obs, make_target(teacher, reference)))
        observation, actions = collate(batch, device)
        batch = []
        loss = masked_flow_loss(
            model, observation, actions, args.supervise_steps, args.endpoint_coef,
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        running.append(float(loss))
        if step % args.log_every == 0:
            log.info(
                "step %d/%d loss=%.4f lr=%.2e (%.1f it/s, %.1f h eta)",
                step, args.steps, float(np.mean(running)), lr_at(step),
                step / max(time.time() - started, 1e-9),
                (args.steps - step) / max(step / max(time.time() - started, 1e-9), 1e-9) / 3600,
            )
            running = []
        if hold_streamer is not None and step % args.eval_every == 0:
            hold_loss = _mean_loss(
                model, policy, hold_streamer, device, "teacher",
                args.eval_batches, args.batch_size, args.supervise_steps,
            )
            hold_history.append({"step": step, "teacher_loss": hold_loss})
            if hold_loss < best_hold - args.min_delta:
                best_hold = hold_loss
                best_step = step
                stall = 0
                save_checkpoint(model, args.init, args.out)
            else:
                stall += 1
            log.info(
                "holdout step %d teacher_loss=%.4f (best %.4f at step %d, stall %d/%d)",
                step, hold_loss, best_hold, best_step, stall, args.patience,
            )
            if stall >= args.patience and step >= args.min_steps and hold_loss >= best_hold - args.min_delta:
                log.info(
                    "early stop at step %d; kept holdout %.4f from step %d",
                    step, best_hold, best_step,
                )
                stopped_early = True
                break
        elif hold_streamer is None and step % args.save_every == 0:
            save_checkpoint(model, args.init, args.out)
            snap = Path(args.out).parent / f"{Path(args.out).name}_step{step}"
            save_checkpoint(model, args.init, str(snap))

    if hold_streamer is None or best_step == 0:
        save_checkpoint(model, args.init, args.out)
        best_step = step
    summary = {
        "data": args.data, "init": args.init, "out": args.out,
        "steps_ran": step, "steps_budget": args.steps,
        "train_shards": len(train_paths), "holdout_shards": len(hold_paths),
        "decisions": n, "self_check": {"reference": ref_loss, "teacher": teacher_loss},
        "train_loss_at_stop": float(loss),
        "best_holdout_loss": None if best_hold == float("inf") else best_hold,
        "best_step": best_step,
        "stopped_early": stopped_early,
        "holdout": hold_history,
        "hours": round((time.time() - started) / 3600, 2),
        "loss_mask": [0, args.supervise_steps],
        "endpoint_coef": args.endpoint_coef,
        "time_max": args.time_max,
        "time_low_frac": args.time_low_frac,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "distill_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("done in %.2f h -> %s", summary["hours"], args.out)


def iter_forever(streamer: ShardStreamer):
    while True:
        yield from streamer.iter_epoch()


if __name__ == "__main__":
    main()
