"""Force openpi's checkpoint saves to be synchronous.

A mid-run save deadlocked the first full training run. Orbax had written 3.6 GB of the
params and then stopped: zero bytes for over ten minutes, 32 threads in uninterruptible
I/O wait, while the filesystem itself was healthy (495 MB/s on a concurrent write test)
and all eight GPUs sat idle. Orbax would not have given up for two hours -- openpi builds
its CheckpointManagerOptions with `async_options=AsyncOptions(timeout_secs=7200)`.

Both short probe runs saved fine, and the difference is instructive: they saved on their
final step, where openpi waits for the write and exits. Only a save that runs *while the
training loop keeps going* hung, which points at the async checkpointer racing the loop
rather than at orbax's serialisation or at gradient accumulation.

Making saves synchronous costs a pause of roughly ninety seconds per checkpoint at the
measured throughput, which is a good trade against losing a whole run to a deadlock.

openpi is not modified: this replaces the options class for the training process only,
the same approach used for the optimizer in accum.py.
"""

from __future__ import annotations


def force_synchronous_checkpointing() -> None:
    """Make every CheckpointManagerOptions built in this process synchronous."""
    import orbax.checkpoint as ocp

    original = ocp.CheckpointManagerOptions
    if getattr(original, "_pi05_sync_patched", False):
        return

    def options(*args, **kwargs):
        kwargs["enable_async_checkpointing"] = False
        return original(*args, **kwargs)

    options._pi05_sync_patched = True
    ocp.CheckpointManagerOptions = options
