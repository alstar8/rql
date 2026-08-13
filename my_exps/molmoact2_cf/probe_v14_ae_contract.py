"""Real-checkpoint GPU probe for the V14 Molmo action-expert contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from molmo_ae_backend import MolmoAEBackend
from rlt_models import ACTION_DIM, CHUNK_SIZE, MolmoAct2RLTCF


def _synthetic_observation() -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    state = np.asarray(
        [0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.7, 1.0],
        dtype=np.float32,
    )
    return image, state


def _perturb_lora_b(backend: MolmoAEBackend, scale: float = 2e-4) -> None:
    with torch.no_grad():
        for name, parameter in backend.model.named_parameters():
            if parameter.requires_grad and "lora_B" in name:
                index = torch.arange(
                    parameter.numel(),
                    device=parameter.device,
                    dtype=torch.float32,
                ).reshape(parameter.shape)
                parameter.copy_(
                    (scale * torch.sin(index + 1.0)).to(dtype=parameter.dtype)
                )
    backend.invalidate_modulation_cache()


def _finite_gradient_report(backend: MolmoAEBackend) -> dict[str, Any]:
    image, state = _synthetic_observation()
    backend.train(True)
    backend.model.zero_grad(set_to_none=True)
    context, _features = backend.encode_context(
        image,
        image,
        "pick up the kettle.",
        state,
    )
    source = backend.sample_native_source(context, seed=991)
    target = torch.zeros_like(source)
    flow_time = torch.full(
        (source.shape[0],),
        0.5,
        device=source.device,
        dtype=torch.float32,
    )
    flow_state = 0.5 * source + 0.5 * target
    velocity = backend.velocity(context, flow_state, flow_time)
    loss = velocity.float().square().mean()
    loss.backward()

    trainable = {
        name: parameter
        for name, parameter in backend.model.named_parameters()
        if parameter.requires_grad
    }
    finite = [
        name
        for name, parameter in trainable.items()
        if parameter.grad is not None and torch.isfinite(parameter.grad).all()
    ]
    nonzero = [
        name
        for name, parameter in trainable.items()
        if parameter.grad is not None
        and float(parameter.grad.detach().float().abs().sum()) > 0.0
    ]
    context_k = [
        name for name in nonzero if "context_k_proj" in name
    ]
    context_v = [
        name for name in nonzero if "context_v_proj" in name
    ]
    return {
        "loss": float(loss.detach()),
        "trainable_tensors": len(trainable),
        "finite_gradient_tensors": len(finite),
        "nonzero_gradient_tensors": len(nonzero),
        "context_k_nonzero": len(context_k),
        "context_v_nonzero": len(context_v),
        "all_finite": len(finite) == len(trainable),
        "all_nonzero": len(nonzero) == len(trainable),
    }


def run_probe(device: str, rank: int) -> dict[str, Any]:
    rlt = MolmoAct2RLTCF(
        cf_mode="flow",
        use_cf_guide=True,
        tune_token_online=False,
        token_layers=1,
        token_d_model=64,
    ).to(device)
    with torch.no_grad():
        rlt.action_mean.zero_()
        rlt.action_std.fill_(1.0)
    backend = MolmoAEBackend(
        device=device,
        dtype=torch.bfloat16,
        enable_lora=True,
        lora_rank=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        rlt=rlt,
        feature_mode="features",
    )
    image, state = _synthetic_observation()

    native = backend.predict_native_reference(
        image,
        image,
        "pick up the kettle.",
        state,
        source_seed=123,
    )
    custom = backend.predict_reference(
        image,
        image,
        "pick up the kettle.",
        state,
        source_seed=123,
    )
    parity_delta = np.abs(
        np.asarray(native["actions"]) - np.asarray(custom["actions_raw_full"])
    )

    raw = np.tile(state[None, :], (15, 1))
    native_roundtrip = backend.normalize_actions(raw)
    raw_roundtrip = backend.unnormalize_actions(native_roundtrip)
    roundtrip_delta = np.abs(raw - raw_roundtrip)

    _perturb_lora_b(backend)
    base_first = backend.predict_reference(
        image,
        image,
        "pick up the kettle.",
        state,
        source_seed=456,
    )
    current_first = backend.predict(
        image,
        image,
        "pick up the kettle.",
        state,
        source_seed=456,
    )
    current_second = backend.predict(
        image,
        image,
        "pick up the kettle.",
        state,
        source_seed=456,
    )
    base_second = backend.predict_reference(
        image,
        image,
        "pick up the kettle.",
        state,
        source_seed=456,
    )
    base_order_delta = np.abs(
        base_first["actions_raw_full"] - base_second["actions_raw_full"]
    )
    current_order_delta = np.abs(
        current_first["actions_raw_full"] - current_second["actions_raw_full"]
    )
    adapter_delta = np.abs(
        current_first["actions_raw_full"] - base_first["actions_raw_full"]
    )

    rlt_state = torch.zeros(
        1,
        rlt.state_dim,
        device=device,
        dtype=torch.float32,
    )
    guided = backend.predict(
        image,
        image,
        "pick up the kettle.",
        state,
        apply_guide=True,
        rlt_state=rlt_state,
        source_seed=789,
    )
    gradient_report = _finite_gradient_report(backend)
    trainable_names = [
        name
        for name, parameter in backend.model.named_parameters()
        if parameter.requires_grad
    ]
    contract = backend.action_contract()

    checks = {
        "native_custom_parity": float(parity_delta.max()) <= 1e-6,
        "native_roundtrip": float(roundtrip_delta.max()) <= 2e-2,
        "base_call_order_invariant": float(base_order_delta.max()) <= 1e-6,
        "current_call_order_invariant": float(current_order_delta.max()) <= 1e-6,
        "adapter_is_active": float(adapter_delta.max()) > 0.0,
        "guided_raw_finite": bool(np.isfinite(guided["actions_raw_full"]).all()),
        "guided_deploy_shape": tuple(guided["actions"].shape)
        == (CHUNK_SIZE, ACTION_DIM),
        "lora_scope": bool(trainable_names)
        and all(
            "action_expert" in name and "lora_" in name.lower()
            for name in trainable_names
        ),
        "all_trainable_gradients_finite": gradient_report["all_finite"],
        "context_projection_gradients_nonzero": (
            gradient_report["context_k_nonzero"] > 0
            and gradient_report["context_v_nonzero"] > 0
        ),
        "contract_shape": contract
        == {
            "action_dim": 8,
            "max_action_dim": 32,
            "action_horizon": 15,
            "n_action_steps": 15,
            "n_obs_steps": 1,
        },
    }
    return {
        "verdict": "VERIFIED" if all(checks.values()) else "NOT VERIFIED",
        "checks": checks,
        "contract": contract,
        "native_custom_max_abs_delta": float(parity_delta.max()),
        "native_roundtrip_max_abs_delta": float(roundtrip_delta.max()),
        "base_call_order_max_abs_delta": float(base_order_delta.max()),
        "current_call_order_max_abs_delta": float(current_order_delta.max()),
        "adapter_base_max_abs_delta": float(adapter_delta.max()),
        "gradient_report": gradient_report,
        "trainable_parameter_count": int(
            sum(parameter.numel() for parameter in backend.trainable_parameters())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lora-rank", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_probe(args.device, args.lora_rank)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if report["verdict"] != "VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
