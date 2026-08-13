"""CPU/checkpoint wiring probe for all eight controlled V13 variants."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from models import EnsembleCQL
from molmo_ae_backend import MolmoAEBackend, _AE_LORA_TARGET_REGEX
from rlt_models import (
    ACTION_DIM,
    CHUNK_SIZE,
    STATE_DIM,
    CFGradientGuide,
    EnsembleTimeCQL,
    FlowCFGuide,
    FlowVelocityActor,
    ChunkGaussianActor,
    MolmoAct2RLTCF,
)
from v13_harness import (
    N_CRITICS,
    VARIANTS,
    VariantSpec,
    atomic_write_json,
    sha256_file,
)


_HERE = Path(__file__).resolve().parent
DEFAULT_RESIDUAL_CHECKPOINT = (
    _HERE / "runs" / "rlt_pretrain_demo1k" / "rlt_cf_pretrain_demo1k.pt"
)
DEFAULT_FLOW_CHECKPOINT = (
    _HERE / "runs" / "rlt_pretrain_demo1k" / "rlt_cf_flow_pretrain_demo1k.pt"
)
DEFAULT_OUTPUT = _HERE / "runs" / "rlt_cf_v13_controlled" / "wiring_probe.json"


class FakeLoRAActionExpert(nn.Module):
    """Tiny differentiable stand-in with AE-like LoRA parameter names."""

    def __init__(self, flat_action: int, rank: int = 4) -> None:
        super().__init__()
        self.lora_A = nn.Linear(flat_action, rank, bias=False)
        self.lora_B = nn.Linear(rank, flat_action, bias=False)
        nn.init.normal_(self.lora_A.weight, std=0.05)
        nn.init.normal_(self.lora_B.weight, std=0.05)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        flat = actions.flatten(1)
        update = self.lora_B(torch.tanh(self.lora_A(flat)))
        return update.reshape_as(actions)


class FakeMolmoAEBackend(nn.Module):
    """No-weight fake used to prove AE adapter optimizer/gradient wiring."""

    def __init__(self) -> None:
        super().__init__()
        self.action_expert = FakeLoRAActionExpert(CHUNK_SIZE * ACTION_DIM)

    def velocity(
        self,
        actions: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        scale = 0.5 + time.reshape(-1, 1, 1)
        return self.action_expert(actions) * scale

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def _small_model(variant: VariantSpec) -> MolmoAct2RLTCF:
    model = MolmoAct2RLTCF(
        hidden=32,
        n_critics=2,
        token_d_model=32,
        token_layers=1,
        token_heads=4,
        cf_mode=variant.cf_mode,
        flow_steps=2,
        use_cf_guide=variant.use_guide,
        tune_token_online=False,
    )
    model.freeze_token_encoder()
    if variant.is_baseline:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    elif variant.ae_trainable:
        for parameter in model.actor.parameters():
            parameter.requires_grad_(False)
        model.v_source = "molmo_ae"
    return model


def _checkpoint_metadata(path: Path, expected_mode: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload is not a dictionary: {path}")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no non-empty state_dict: {path}")
    mode = str(payload.get("cf_mode", "residual"))
    if mode != expected_mode:
        raise ValueError(
            f"Checkpoint {path} has cf_mode={mode!r}, expected {expected_mode!r}"
        )
    n_critics = int(payload.get("n_critics", N_CRITICS))
    if n_critics != N_CRITICS:
        raise ValueError(
            f"Checkpoint {path} has n_critics={n_critics}, expected {N_CRITICS}"
        )
    prefixes = sorted(
        {
            str(key).split(".", 1)[0]
            for key in state
        }
    )
    required_prefixes = {"token_ae", "critic", "target_critic", "actor"}
    missing = sorted(required_prefixes - set(prefixes))
    if missing:
        raise ValueError(f"Checkpoint {path} lacks module prefixes: {missing}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "cf_mode": mode,
        "n_critics": n_critics,
        "bounded_critic": bool(payload.get("bounded_critic", True)),
        "use_cf_guide": bool(payload.get("use_cf_guide", True)),
        "state_tensor_count": len(state),
        "state_prefixes": prefixes,
    }


def _named_parameters(
    model: MolmoAct2RLTCF,
    fake_ae: FakeMolmoAEBackend | None,
) -> dict[str, nn.Parameter]:
    output = {
        f"rlt.{name}": parameter
        for name, parameter in model.named_parameters()
    }
    if fake_ae is not None:
        output.update(
            {
                f"ae_backend.{name}": parameter
                for name, parameter in fake_ae.named_parameters()
            }
        )
    return output


def _optimizer_parameter_ids(
    optimizers: dict[str, torch.optim.Optimizer],
) -> dict[str, list[int]]:
    return {
        name: [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        for name, optimizer in optimizers.items()
    }


def _build_probe_optimizers(
    variant: VariantSpec,
    model: MolmoAct2RLTCF,
    fake_ae: FakeMolmoAEBackend | None,
) -> dict[str, torch.optim.Optimizer]:
    if variant.is_baseline:
        return {}
    actor_parameters: Iterable[nn.Parameter]
    if variant.ae_trainable:
        if fake_ae is None:
            raise RuntimeError("AE variant requires the fake AE backend")
        actor_parameters = fake_ae.trainable_parameters()
    else:
        actor_parameters = model.actor.parameters()
    optimizers = {
        "critic": torch.optim.SGD(model.critic.parameters(), lr=1e-2),
        "actor": torch.optim.SGD(actor_parameters, lr=1e-2),
        "alpha": torch.optim.SGD(model.log_alpha.parameters(), lr=1e-2),
    }
    if variant.use_guide:
        if model.guide is None:
            raise RuntimeError("CF variant has no guide")
        optimizers["guide"] = torch.optim.SGD(model.guide.parameters(), lr=1e-2)
    return optimizers


def _snapshot_parameters(parameters: dict[str, nn.Parameter]) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in parameters.items()
    }


def _max_delta(
    before: dict[str, torch.Tensor],
    parameters: dict[str, nn.Parameter],
    names: Iterable[str],
) -> float:
    deltas = []
    for name in names:
        previous = before[name]
        current = parameters[name].detach()
        deltas.append(float((current - previous).abs().max()))
    return max(deltas, default=0.0)


def _module_parameter_names(
    parameters: dict[str, nn.Parameter],
    prefix: str,
    *,
    trainable_only: bool = False,
) -> list[str]:
    return sorted(
        name
        for name, parameter in parameters.items()
        if name.startswith(prefix)
        and (not trainable_only or parameter.requires_grad)
    )


def _synthetic_loss(
    variant: VariantSpec,
    model: MolmoAct2RLTCF,
    fake_ae: FakeMolmoAEBackend | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    torch.manual_seed(1000 + variant.gpu)
    batch_size = 3
    state = torch.randn(batch_size, STATE_DIM)
    reference = torch.randn(batch_size, CHUNK_SIZE, ACTION_DIM) * 0.1
    actions_for_critic = torch.randn_like(reference) * 0.1
    if model.is_flow:
        time = torch.full((batch_size, 1), 0.65)
        critic_values = model.q_chunk(state, actions_for_critic, t=time)
    else:
        time = None
        critic_values = model.q_chunk(state, actions_for_critic)
    finite_outputs = [critic_values]
    losses = [F.mse_loss(critic_values, torch.full_like(critic_values, 0.4))]
    diagnostics: dict[str, Any] = {
        "critic_output_shape": list(critic_values.shape),
    }

    if variant.is_baseline:
        deployed = reference.detach().clone()
        finite_outputs.append(deployed)
        diagnostics["reference_bytes_equal"] = (
            deployed.numpy().tobytes() == reference.detach().numpy().tobytes()
        )
        total = sum(loss.detach() for loss in losses)
        diagnostics["all_outputs_finite"] = all(
            bool(torch.isfinite(output).all()) for output in finite_outputs
        )
        return total, diagnostics

    if variant.ae_trainable:
        if fake_ae is None or time is None:
            raise RuntimeError("AE synthetic loss requires fake backend and flow time")
        actor_output = fake_ae.velocity(reference, time)
        actor_target = torch.full_like(actor_output, 0.15)
        diagnostics["deployed_actor_class"] = MolmoAEBackend.__name__
    elif model.is_flow:
        if time is None:
            raise RuntimeError("Flow actor requires time")
        actor_output = model.actor(state, reference, time, reference)
        actor_target = torch.full_like(actor_output, 0.2)
        diagnostics["deployed_actor_class"] = type(model.actor).__name__
    else:
        actor_output, _ = model.actor_chunk(
            state,
            reference,
            deterministic=True,
            apply_guide=False,
        )
        actor_target = reference + 0.03
        diagnostics["deployed_actor_class"] = type(model.actor).__name__
    losses.append(F.mse_loss(actor_output, actor_target))
    finite_outputs.append(actor_output)

    if variant.use_guide:
        if model.guide is None:
            raise RuntimeError("Expected guide is absent")
        if model.is_flow:
            if time is None:
                raise RuntimeError("Flow guide requires time")
            raw_guide = model.guide.raw_w(state, reference, time)
        else:
            raw_guide = model.guide.raw_w(state, reference)
        guide_target = torch.linspace(
            -0.2,
            0.2,
            raw_guide.numel(),
        ).reshape_as(raw_guide)
        losses.append(F.mse_loss(raw_guide, guide_target))
        finite_outputs.append(raw_guide)

    alpha_value = model.log_alpha()
    losses.append((alpha_value - 0.5).square())
    finite_outputs.append(alpha_value)
    total = sum(losses)
    diagnostics.update(
        {
            "loss": float(total.detach()),
            "all_outputs_finite": all(
                bool(torch.isfinite(output).all()) for output in finite_outputs
            ),
        }
    )
    return total, diagnostics


def probe_variant(
    variant: VariantSpec,
    checkpoint_info: dict[str, Any] | None,
) -> dict[str, Any]:
    model = _small_model(variant)
    fake_ae = FakeMolmoAEBackend() if variant.ae_trainable else None
    parameters = _named_parameters(model, fake_ae)
    trainable_names = sorted(
        name
        for name, parameter in parameters.items()
        if parameter.requires_grad
    )
    frozen_names = sorted(set(parameters) - set(trainable_names))
    optimizers = _build_probe_optimizers(variant, model, fake_ae)
    memberships = _optimizer_parameter_ids(optimizers)
    membership_counts = Counter(
        parameter_id
        for ids in memberships.values()
        for parameter_id in ids
    )
    trainable_ids = {
        id(parameter)
        for parameter in parameters.values()
        if parameter.requires_grad
    }
    optimizer_ids = set(membership_counts)
    duplicate_optimizer_parameters = sorted(
        parameter_id
        for parameter_id, count in membership_counts.items()
        if count != 1
    )
    missing_trainable = trainable_ids - optimizer_ids
    frozen_in_optimizer = optimizer_ids - trainable_ids

    expected_actor_type: type[nn.Module]
    if variant.cf_mode == "flow":
        expected_actor_type = FlowVelocityActor
        expected_critic_type = EnsembleTimeCQL
    else:
        expected_actor_type = ChunkGaussianActor
        expected_critic_type = EnsembleCQL
    expected_guide_type: type[nn.Module] | None
    if not variant.use_guide:
        expected_guide_type = None
    elif variant.cf_mode == "flow":
        expected_guide_type = FlowCFGuide
    else:
        expected_guide_type = CFGradientGuide

    class_checks = {
        "rlt_actor": isinstance(model.actor, expected_actor_type),
        "critic": isinstance(model.critic, expected_critic_type),
        "guide": (
            model.guide is None
            if expected_guide_type is None
            else isinstance(model.guide, expected_guide_type)
        ),
        "real_ae_backend_class_available": (
            MolmoAEBackend.__name__ == "MolmoAEBackend"
            if variant.ae_trainable
            else True
        ),
        "ae_lora_targets_action_expert": (
            "action_expert" in _AE_LORA_TARGET_REGEX
            if variant.ae_trainable
            else True
        ),
    }

    before = _snapshot_parameters(parameters)
    loss, forward = _synthetic_loss(variant, model, fake_ae)
    gradient_finite = True
    if optimizers:
        for optimizer in optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name in trainable_names:
            gradient = parameters[name].grad
            if gradient is not None and not bool(torch.isfinite(gradient).all()):
                gradient_finite = False
        for optimizer in optimizers.values():
            optimizer.step()

    module_prefixes = {
        "critic": "rlt.critic.",
        "actor": (
            "ae_backend.action_expert."
            if variant.ae_trainable
            else "rlt.actor."
        ),
        "guide": "rlt.guide.",
        "alpha": "rlt.log_alpha.",
        "token": "rlt.token_ae.",
        "target_critic": "rlt.target_critic.",
        "inactive_rlt_actor": "rlt.actor.",
    }
    module_deltas = {
        name: _max_delta(
            before,
            parameters,
            _module_parameter_names(parameters, prefix),
        )
        for name, prefix in module_prefixes.items()
    }
    frozen_delta = _max_delta(before, parameters, frozen_names)

    expected_changed = []
    if not variant.is_baseline:
        expected_changed.extend(["critic", "actor", "alpha"])
        if variant.use_guide:
            expected_changed.append("guide")
    changed_checks = {
        name: module_deltas[name] > 0.0
        for name in expected_changed
    }
    forbidden_checks = {
        "all_frozen_unchanged": frozen_delta == 0.0,
        "token_unchanged": module_deltas["token"] == 0.0,
        "target_critic_unchanged": module_deltas["target_critic"] == 0.0,
        "ae_inactive_rlt_actor_unchanged": (
            module_deltas["inactive_rlt_actor"] == 0.0
            if variant.ae_trainable
            else True
        ),
        "baseline_has_no_trainable_parameters": (
            not trainable_names if variant.is_baseline else True
        ),
        "baseline_has_no_optimizer": (
            not optimizers if variant.is_baseline else True
        ),
    }
    optimizer_checks = {
        "each_trainable_exactly_once": (
            not missing_trainable and not duplicate_optimizer_parameters
        ),
        "no_frozen_members": not frozen_in_optimizer,
        "optimizer_count": len(optimizers),
    }
    checks = [
        *class_checks.values(),
        bool(forward["all_outputs_finite"]),
        math.isfinite(float(loss.detach())),
        gradient_finite,
        *changed_checks.values(),
        *forbidden_checks.values(),
        bool(optimizer_checks["each_trainable_exactly_once"]),
        bool(optimizer_checks["no_frozen_members"]),
    ]
    return {
        "variant": variant.name,
        "gpu": variant.gpu,
        "checkpoint": checkpoint_info,
        "expected": {
            "actor_class": variant.actor_class,
            "rlt_actor_class": expected_actor_type.__name__,
            "critic_class": expected_critic_type.__name__,
            "guide_class": (
                expected_guide_type.__name__
                if expected_guide_type is not None
                else None
            ),
            "updates_per_episode": variant.updates_per_episode,
            "ae_backend": variant.ae_trainable,
        },
        "actual": {
            "rlt_actor_class": type(model.actor).__name__,
            "critic_class": type(model.critic).__name__,
            "guide_class": (
                type(model.guide).__name__
                if model.guide is not None
                else None
            ),
            "deployed_actor_class": forward.get("deployed_actor_class", "reference"),
        },
        "class_checks": class_checks,
        "parameters": {
            "trainable_names": trainable_names,
            "frozen_names": frozen_names,
            "trainable_count": sum(
                parameter.numel()
                for parameter in parameters.values()
                if parameter.requires_grad
            ),
            "frozen_count": sum(
                parameter.numel()
                for parameter in parameters.values()
                if not parameter.requires_grad
            ),
        },
        "optimizer_membership": {
            "optimizers": {
                name: len(ids)
                for name, ids in memberships.items()
            },
            "missing_trainable_count": len(missing_trainable),
            "duplicate_parameter_count": len(duplicate_optimizer_parameters),
            "frozen_member_count": len(frozen_in_optimizer),
            **optimizer_checks,
        },
        "synthetic_update": {
            **forward,
            "gradient_finite": gradient_finite,
            "module_max_abs_delta": module_deltas,
            "frozen_max_abs_delta": frozen_delta,
            "expected_changed": changed_checks,
            "forbidden_unchanged": forbidden_checks,
        },
        "passed": all(checks),
    }


def run_probe(
    residual_checkpoint: Path,
    flow_checkpoint: Path,
    *,
    skip_checkpoints: bool = False,
) -> dict[str, Any]:
    checkpoint_records: dict[str, dict[str, Any]] = {}
    checkpoint_failures = []
    if skip_checkpoints:
        checkpoint_records = {
            "residual": {"skipped": True, "expected_cf_mode": "residual"},
            "flow": {"skipped": True, "expected_cf_mode": "flow"},
        }
    else:
        for kind, path, mode in (
            ("residual", residual_checkpoint, "residual"),
            ("flow", flow_checkpoint, "flow"),
        ):
            try:
                checkpoint_records[kind] = _checkpoint_metadata(path, mode)
            except Exception as error:
                checkpoint_records[kind] = {
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
                checkpoint_failures.append(kind)

    records = []
    for variant in VARIANTS:
        try:
            checkpoint = checkpoint_records.get(variant.checkpoint_kind)
            record = probe_variant(variant, checkpoint)
            if variant.checkpoint_kind in checkpoint_failures:
                record["passed"] = False
                record["checkpoint_failure"] = True
        except Exception as error:
            record = {
                "variant": variant.name,
                "gpu": variant.gpu,
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
    passed = not checkpoint_failures and all(
        bool(record.get("passed"))
        for record in records
    )
    if skip_checkpoints:
        passed = all(bool(record.get("passed")) for record in records)
    return {
        "schema_version": "v13-wiring-probe-1",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "verdict": "VERIFIED" if passed else "NOT VERIFIED",
        "passed": passed,
        "cpu_only": True,
        "full_molmo_ae_weights_loaded": False,
        "real_ae_check_deferred_to_smoke": True,
        "checkpoint_records": checkpoint_records,
        "variants": records,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--residual-checkpoint",
        type=Path,
        default=DEFAULT_RESIDUAL_CHECKPOINT,
    )
    parser.add_argument(
        "--flow-checkpoint",
        type=Path,
        default=DEFAULT_FLOW_CHECKPOINT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Unit-test mode; still probes all eight real RLT module wirings.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_probe(
        args.residual_checkpoint,
        args.flow_checkpoint,
        skip_checkpoints=args.skip_checkpoints,
    )
    atomic_write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "verdict": payload["verdict"],
                "passed_variants": sum(
                    bool(record.get("passed"))
                    for record in payload["variants"]
                ),
                "total_variants": len(payload["variants"]),
            },
            indent=2,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
