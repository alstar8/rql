"""Fine-tune pi0.5 on MolmoSpaces Pick with gradient accumulation.

openpi's training loop is used as-is; the only change is that the optimizer accumulates
gradients so the effective batch reaches the 256 its own large configs use, which two
H100s cannot hold directly. See pi05/accum.py for why that matters and what it costs.

    python scripts/train_pi05.py \
        --exp-name pick_full --repo-id molmospaces/pick_all \
        --batch-size 16 --accumulate 16 --num-train-steps 30000

That is an effective batch of 256 and 1875 real optimizer updates. Checkpoints are
written once at the end unless --save-interval says otherwise; each is about 42 GB, of
which 12 GB is the model and the rest optimizer and EMA state needed only to resume.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OPENPI = Path("/home/jovyan/users/staroverov/B1K/B1K_AIRI/submodules/openpi")
sys.path.insert(0, str(OPENPI / "scripts"))

from openpi.training import config as _config  # noqa: E402
from openpi.training import optimizer as _optimizer  # noqa: E402

from pi05.checkpointing import force_synchronous_checkpointing  # noqa: E402
from pi05.metrics import install_jsonl_logging  # noqa: E402
from pi05.accum import (  # noqa: E402
    effective_batch,
    optimizer_steps,
    rescale_schedule,
    wrap_optimizer_factory,
)

CHECKPOINT = Path(
    "/home/jovyan/users/staroverov/.cache/openpi/openpi-assets/checkpoints/pi05_droid_jointpos"
)
RUNS = Path("/workspace-SR008.nfs2/users/staroverov/pi05_molmospaces/pi05_runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--repo-id", required=True, help="LeRobot dataset under HF_LEROBOT_HOME")
    ap.add_argument("--config", default="pi05_droid_finetune")
    ap.add_argument("--batch-size", type=int, default=16, help="micro-batch that fits on the GPUs")
    ap.add_argument("--accumulate", type=int, default=16, help="micro-batches per optimizer step")
    ap.add_argument("--num-train-steps", type=int, default=30000, help="loop steps, not updates")
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--save-interval", type=int, default=30000)
    ap.add_argument("--fsdp-devices", type=int, default=2,
                    help="model-sharding width; the mesh is (devices/fsdp, fsdp), so 2 "
                         "keeps all 8 GPUs busy with 4-way data parallelism")
    ap.add_argument("--num-workers", type=int, default=24,
                    help="dataloader processes; openpi defaults to 2, which starves 8 GPUs")
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--keep-period", type=int, default=None,
                    help="retain every Nth checkpoint permanently; needed to keep "
                         "candidates for best-by-success-rate selection")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    config = _config.get_config(args.config)

    data = dataclasses.replace(
        config.data,
        repo_id=args.repo_id,
        assets=dataclasses.replace(config.data.assets, assets_dir=str(args.checkpoint / "assets")),
    )
    lr_schedule = rescale_schedule(config.lr_schedule, args.num_train_steps, args.accumulate)

    config = dataclasses.replace(
        config,
        exp_name=args.exp_name,
        data=data,
        lr_schedule=lr_schedule,
        weight_loader=_optimizer_weight_loader(args.checkpoint),
        batch_size=args.batch_size,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        # Nothing pinned: without this every 5000th checkpoint is kept forever, and at
        # 42 GB each a long run would leave hundreds of gigabytes behind.
        keep_period=args.keep_period,
        fsdp_devices=args.fsdp_devices,
        num_workers=args.num_workers,
        checkpoint_base_dir=str(RUNS / "checkpoints"),
        assets_base_dir=str(RUNS / "assets"),
        wandb_enabled=False,
    )

    # A mid-run async save deadlocked the first full run with all eight GPUs idle;
    # synchronous saves cost about ninety seconds each instead.
    force_synchronous_checkpointing()

    # Patched before openpi builds the optimizer, and only for this process.
    _optimizer.create_optimizer = wrap_optimizer_factory(
        _optimizer.create_optimizer, args.accumulate
    )

    # Metrics as JSONL; scripts/jsonl_to_tb.py renders TensorBoard from it, live.
    metrics_path = RUNS / "metrics" / f"{args.exp_name}.metrics.jsonl"
    install_jsonl_logging(
        metrics_path,
        total_steps=args.num_train_steps,
        batch_size=args.batch_size,
        accumulate=args.accumulate,
    )

    updates = optimizer_steps(args.num_train_steps, args.accumulate)
    print(f"dataset          {args.repo_id}")
    import jax
    devices = jax.device_count()
    print(f"devices          {devices} GPUs, mesh {devices // args.fsdp_devices} data x "
          f"{args.fsdp_devices} fsdp")
    print(f"micro-batch      {args.batch_size} ({args.batch_size // max(1, devices // args.fsdp_devices)} per data group)")
    print(f"dataloader       {args.num_workers} workers")
    print(f"accumulate       {args.accumulate}  ->  effective batch "
          f"{effective_batch(args.batch_size, args.accumulate)}")
    print(f"loop steps       {args.num_train_steps}  ->  {updates} optimizer updates")
    print(f"samples seen     {args.num_train_steps * args.batch_size}")
    print(f"lr schedule      warmup {getattr(lr_schedule, 'warmup_steps', '?')}, "
          f"decay over {getattr(lr_schedule, 'decay_steps', '?')} updates, "
          f"peak {getattr(lr_schedule, 'peak_lr', '?')}")
    print(f"checkpoints      {RUNS / 'checkpoints' / config.name / args.exp_name}\n")

    import train as openpi_train  # noqa: PLC0415  (needs the patched optimizer in place)

    openpi_train.main(dataclasses.replace(config, overwrite=args.overwrite, resume=args.resume))


def _optimizer_weight_loader(checkpoint: Path):
    from openpi.training import weight_loaders

    return weight_loaders.CheckpointWeightLoader(str(checkpoint / "params"))


if __name__ == "__main__":
    main()
