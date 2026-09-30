"""Action-expert parameter filter and in-place weight copy.

The PaliGemma backbone is the RL token source and must stay bit-identical across every
swap. Only the action expert (gemma_300m + action/time projections) is copied from the
learning expert into the frozen server.
"""

from __future__ import annotations

from pathlib import Path

TRAIN_PREFIXES = (
    "action_in_proj",
    "action_out_proj",
    "state_proj",
    "time_mlp_in",
    "time_mlp_out",
)
TRAIN_SUBSTR = ("gemma_expert",)
BACKBONE_ROOT = "paligemma_with_expert"


def is_expert_param(name: str) -> bool:
    """True for tensors the student trains and the frozen server must receive on a swap."""
    return name.startswith(TRAIN_PREFIXES) or any(s in name for s in TRAIN_SUBSTR)


def freeze_backbone(model) -> tuple[int, int]:
    """Train the action expert + projections; freeze the PaliGemma backbone."""
    n_train = n_frozen = 0
    missed = []
    for name, p in model.named_parameters():
        trainable = is_expert_param(name)
        p.requires_grad = trainable
        if trainable:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
            if not name.startswith(BACKBONE_ROOT):
                missed.append(name)
    if missed:
        raise RuntimeError(
            "these action-side parameters are outside the backbone but matched no entry "
            f"in TRAIN_PREFIXES, so they would train nothing: {missed}"
        )
    if n_train == 0:
        raise RuntimeError("no trainable parameters; TRAIN_PREFIXES/TRAIN_SUBSTR match nothing")
    return n_train, n_frozen


def expert_safetensors_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"no model.safetensors at {checkpoint}")
    return path


def copy_expert_weights(model, checkpoint: str | Path) -> int:
    """Overwrite the action-expert tensors on `model` from a saved checkpoint.

    Backbone parameters are not written, even if they are present in the file. Returns
    the number of tensors copied; zero is a hard error so a path mix-up cannot look like
    a successful swap.
    """
    import safetensors.torch
    import torch

    weights = expert_safetensors_path(checkpoint)
    tensors = safetensors.torch.load_file(str(weights), device="cpu")
    name_to_param = dict(model.named_parameters())
    copied = 0
    with torch.no_grad():
        for name, tensor in tensors.items():
            if not is_expert_param(name):
                continue
            param = name_to_param.get(name)
            if param is None:
                continue
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
            copied += 1
    if copied == 0:
        raise RuntimeError(f"no expert tensors copied from {weights}")
    return copied


def swap_due(episode: int, every: int) -> bool:
    """True on episode `every`, `2*every`, ... Never on episode 0."""
    return every > 0 and episode > 0 and episode % every == 0
