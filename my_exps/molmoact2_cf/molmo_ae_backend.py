"""In-process MolmoAct2 Action Expert backend for V11_1 ConsensusFlow.

Loads ``allenai/MolmoAct2-DROID`` like ``serve.py``, freezes the VLM, and
optionally attaches AE-only LoRA (``peft``).  Provides:

* rollout ``predict`` (actions + VLM tokens/features)
* detached KV ``encode_context`` for knowledge-insulated AE training
* continuous-time AE ``velocity`` for paper CF: ``ẋ = v_AE + sg(G)``
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from models import FEATURE_DIM  # noqa: E402
from rlt_models import ACTION_DIM, CHUNK_SIZE, MolmoAct2RLTCF  # noqa: E402

log = logging.getLogger("molmoact2.cf.ae_backend")

REPO_ID = "allenai/MolmoAct2-DROID"
NORM_TAG = "franka_droid"
DEFAULT_NUM_STEPS = 10

# Linear leaves inside HF ActionExpert (matches LeRobot AE LoRA regex intent).
_AE_LORA_TARGET_REGEX = (
    r"action_expert\.(?:"
    r"time_embed\.(1|3)|"
    r"action_embed|context_k_proj|context_v_proj|"
    r"blocks\.\d+\.self_attn\.(qkv|out_proj)|"
    r"blocks\.\d+\.cross_attn\.(q_proj|out_proj)|"
    r"blocks\.\d+\.mlp\.(up_proj|gate_proj|down_proj)|"
    r"blocks\.\d+\.modulation\.linear|"
    r"final_layer\.(modulation\.linear|linear)"
    r")$"
)


def _patch_modeling_for_bf16(local_dir: str) -> None:
    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        applied: list[str] = []
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            log.info("Applied patches %s in %s", applied, path)


def _to_pil(arr: Any) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


def _mean_pool_hidden(
    last_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    h = last_hidden.float()
    if attention_mask is None:
        return h.mean(dim=1)
    mask = attention_mask.to(dtype=h.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (h * mask).sum(dim=1) / denom


@dataclass
class AEFlowContext:
    """Detached VLM KV context for AE velocity / guided ODE."""

    encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]]
    encoder_attention_mask: torch.Tensor
    action_dim: int = ACTION_DIM
    max_action_dim: int = 32
    action_horizon: int = 15
    n_action_steps: int = 15
    n_obs_steps: int = 1
    action_dim_is_pad: torch.Tensor | None = None


class MolmoAEBackend:
    """Trainable MolmoAct2 Action Expert + frozen VLM for in-process CF."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.bfloat16,
        repo_id: str = REPO_ID,
        enable_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        num_steps: int = DEFAULT_NUM_STEPS,
        rlt: MolmoAct2RLTCF | None = None,
        feature_mode: str = "tokens",
    ) -> None:
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_steps = int(num_steps)
        self.feature_mode = str(feature_mode)
        self.rlt = rlt
        self._lock = threading.Lock()

        local_dir = snapshot_download(repo_id=repo_id)
        log.info("Resolved MolmoAct2 snapshot: %s", local_dir)
        _patch_modeling_for_bf16(local_dir)

        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir, trust_remote_code=True, torch_dtype=dtype
            )
            .to(self.device)
        )
        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(inputs: Any, dev: Any, _target: torch.dtype = target_dtype) -> dict:
            out: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        self.model._move_inputs_to_device = _move_and_cast

        # Knowledge insulation: freeze everything, then AE LoRA / AE params.
        for p in self.model.parameters():
            p.requires_grad_(False)

        if enable_lora:
            self._apply_ae_lora(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        else:
            self._unfreeze_action_expert()
        stats = self.model._get_robot_stats()
        if stats.norm_mode != "q01_q99":
            raise RuntimeError(
                "V14 Molmo AE requires q01_q99 coordinates, got "
                f"{stats.norm_mode!r}"
            )
        contract = self.action_contract()
        if contract["action_dim"] != ACTION_DIM:
            raise RuntimeError(
                f"Molmo robot action_dim={contract['action_dim']} != {ACTION_DIM}"
            )
        if contract["action_horizon"] < CHUNK_SIZE:
            raise RuntimeError(
                "Molmo action horizon is shorter than the RLT deployment chunk"
            )
        self.invalidate_modulation_cache()

        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.model.parameters())
        log.info(
            "MolmoAEBackend ready trainable=%s / total=%s (%.3f%%) lora=%s",
            f"{n_train:,}",
            f"{n_total:,}",
            100.0 * n_train / max(n_total, 1),
            enable_lora,
        )

    def _backbone(self) -> Any:
        # HF wrappers nest as model.model (PeftModel) → .model (MolmoAct2Model).
        root = self.model
        if hasattr(root, "get_base_model"):
            root = root.get_base_model()
        inner = getattr(root, "model", root)
        return inner

    def _action_expert(self) -> Any:
        bb = self._backbone()
        require = getattr(bb, "_require_action_expert", None)
        if callable(require):
            return require()
        ae = getattr(bb, "action_expert", None)
        if ae is None:
            raise RuntimeError("MolmoAct2 checkpoint has no action_expert")
        return ae

    def _apply_ae_lora(self, *, rank: int, alpha: int, dropout: float) -> None:
        # Prefer explicit Linear module names under action_expert (more reliable
        # than regex across Peft/HF wrapping variants).
        linear_names: list[str] = []
        for name, module in self.model.named_modules():
            if "action_expert" not in name:
                continue
            # Keep time/action embeds on native bf16 Linear (no Peft dtype mix).
            if ".time_embed." in name or name.endswith(".time_embed") or ".action_embed" in name:
                continue
            if isinstance(module, torch.nn.Linear):
                linear_names.append(name)
        if not linear_names:
            raise RuntimeError("No action_expert Linear modules found for LoRA")
        # peft matches against the *suffix* / leaf; use unique full-path-safe leaves
        # by passing the full names when supported, else leaf set.
        cfg = LoraConfig(
            r=int(rank),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias="none",
            target_modules=linear_names,
        )
        # Full AE-scoped names are deliberate.  A generic leaf-name fallback
        # (for example ``out_proj``) can attach adapters to the frozen VLM.
        self.model = get_peft_model(self.model, cfg)
        # Peft adapters default to float32; keep them on the backbone dtype
        # (bf16) so time_embed / AE Linear matmuls do not mix Float/BFloat16.
        self.model.to(device=self.device, dtype=self.dtype)
        for name, p in self.model.named_parameters():
            if "lora_" not in name.lower() and "action_expert" not in name:
                p.requires_grad_(False)
            elif "lora_" not in name.lower() and "action_expert" in name:
                # Keep base AE frozen; only LoRA adapters train.
                p.requires_grad_(False)
            elif p.dtype != self.dtype:
                p.data = p.data.to(dtype=self.dtype)
        trainable_names = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable_names or any(
            "action_expert" not in name or "lora_" not in name.lower()
            for name in trainable_names
        ):
            raise RuntimeError(
                "AE LoRA scope violation: every trainable tensor must be an "
                "action_expert LoRA parameter"
            )
        for required in ("context_k_proj", "context_v_proj"):
            if not any(required in name for name in trainable_names):
                raise RuntimeError(
                    f"AE LoRA is missing required trainable {required} tensors"
                )

    def _unfreeze_action_expert(self) -> None:
        found = False
        for name, p in self.model.named_parameters():
            if "action_expert" in name:
                p.requires_grad_(True)
                found = True
        if not found:
            raise RuntimeError("No action_expert parameters found to unfreeze")

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def train(self, mode: bool = True) -> None:
        # Keep VLM eval; only AE / LoRA in train mode.
        self.model.eval()
        ae = self._action_expert()
        ae.train(mode)

    def eval(self) -> None:
        self.model.eval()

    @contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        """Run against the frozen base AE using PEFT's adapter-disable context."""
        disable_adapter = getattr(self.model, "disable_adapter", None)
        if disable_adapter is None:
            if any(parameter.requires_grad for parameter in self.model.parameters()):
                raise RuntimeError(
                    "Trainable AE has no PEFT disable_adapter context; "
                    "a stable frozen reference cannot be produced"
                )
            yield
            return
        if not callable(disable_adapter):
            raise RuntimeError("PEFT disable_adapter attribute is not callable")
        self.invalidate_modulation_cache()
        try:
            with disable_adapter():
                yield
        finally:
            self.invalidate_modulation_cache()

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pad native robot-width actions to Molmo's expert width."""
        contract = self.action_contract()
        max_dim = contract["max_action_dim"]
        if actions.shape[-1] == max_dim:
            return self._mask_padded_dims(
                actions,
                self._action_dim_is_pad(actions.shape[0]),
            )
        if actions.shape[-1] != contract["action_dim"]:
            raise ValueError(
                f"AE action width {actions.shape[-1]} != robot width "
                f"{contract['action_dim']} or expert width {max_dim}"
            )
        pad = actions.new_zeros(*actions.shape[:-1], max_dim - actions.shape[-1])
        return torch.cat([actions, pad], dim=-1)

    def _trim_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions[..., : self.action_contract()["action_dim"]]

    def pad_native_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self._pad_actions(actions)

    def compact_native_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected batched native actions, got {actions.shape}")
        return actions[:, :CHUNK_SIZE, :ACTION_DIM]

    def action_contract(self) -> dict[str, int]:
        stats = self.model._get_robot_stats()
        norm_tag = stats.validate_tag(NORM_TAG)
        action_dim = stats.get_action_dim(norm_tag)
        if action_dim is None:
            raise RuntimeError(f"Missing action dimension for norm tag {norm_tag!r}")
        action_horizon = stats.get_action_horizon(norm_tag)
        if action_horizon is None:
            action_horizon = int(self.model.config.max_action_horizon)
        n_action_steps = stats.get_n_action_steps(norm_tag)
        if n_action_steps is None:
            n_action_steps = int(action_horizon)
        return {
            "action_dim": int(action_dim),
            "max_action_dim": int(self.model.config.max_action_dim),
            "action_horizon": int(action_horizon),
            "n_action_steps": int(n_action_steps),
            "n_obs_steps": int(self.model.config.n_obs_steps),
        }

    def normalize_actions(self, actions_raw: np.ndarray | torch.Tensor) -> torch.Tensor:
        stats = self.model._get_robot_stats()
        norm_tag = stats.validate_tag(NORM_TAG)
        normalizer = stats.action_normalizers.get(norm_tag)
        normalized = actions_raw if normalizer is None else normalizer.normalize(actions_raw)
        out = torch.as_tensor(normalized, device=self.device, dtype=self.dtype)
        if out.shape[-1] != self.action_contract()["action_dim"]:
            raise ValueError(
                f"Normalized AE action has invalid width {out.shape[-1]}"
            )
        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite normalized AE action")
        return out

    def unnormalize_actions(self, actions_native: torch.Tensor) -> np.ndarray:
        stats = self.model._get_robot_stats()
        norm_tag = stats.validate_tag(NORM_TAG)
        # Preserve the expert dtype through unnormalization.  Molmo's native
        # path unnormalizes bf16 and only then casts raw actions to float32.
        native = self._trim_actions(actions_native).detach()
        raw = stats.unnormalize_action(native, norm_tag)
        raw_t = torch.as_tensor(raw, device=self.device, dtype=torch.float32)
        if not torch.isfinite(raw_t).all():
            raise RuntimeError("Non-finite unnormalized AE action")
        return raw_t.cpu().numpy().astype(np.float32, copy=False)

    def _action_dim_is_pad(self, batch_size: int) -> torch.Tensor:
        contract = self.action_contract()
        mask = torch.ones(
            batch_size,
            contract["max_action_dim"],
            device=self.device,
            dtype=torch.bool,
        )
        mask[:, : contract["action_dim"]] = False
        return mask

    @staticmethod
    def _mask_padded_dims(
        tensor: torch.Tensor,
        action_dim_is_pad: torch.Tensor,
    ) -> torch.Tensor:
        return tensor.masked_fill(action_dim_is_pad[:, None, :], 0.0)

    def invalidate_modulation_cache(self) -> None:
        action_expert = self._action_expert()
        for name in (
            "_modulation_cache",
            "modulation_cache",
            "_cached_modulation",
            "cached_modulation",
            "_modulation_cache_key",
            "_modulation_cache_value",
        ):
            if hasattr(action_expert, name):
                setattr(action_expert, name, None)

    def _build_robot_inputs(
        self,
        images: list[Image.Image],
        instruction: str,
        state_f32: np.ndarray,
    ) -> dict[str, Any]:
        stats = self.model._get_robot_stats()
        norm_tag = stats.validate_tag(NORM_TAG)
        metadata = stats.get_metadata(norm_tag)
        normalized_state = np.asarray(
            stats.normalize_state(state_f32, norm_tag), dtype=np.float32
        )
        num_state_tokens = int(self.model.config.num_state_tokens or 0)
        modeling = sys.modules[self.model.__class__.__module__]
        # Peft may wrap the class module; fall back to backbone module.
        if not hasattr(modeling, "_build_robot_text"):
            modeling = sys.modules[self._backbone().__class__.__module__]
        discrete_state_string = modeling._build_discrete_state_string(
            normalized_state, num_state_tokens
        )
        task_text = modeling._normalize_question_text(str(instruction or ""))
        text = modeling._build_robot_text(
            task=task_text,
            style="robot_action",
            discrete_state_string=discrete_state_string,
            setup_type=str(metadata.get("setup_type", "") or ""),
            control_mode=str(metadata.get("control_mode", "") or ""),
            add_setup_tokens=bool(self.model.config.add_setup_tokens),
            add_control_tokens=bool(self.model.config.add_control_tokens),
            num_images=self.model._count_images(images)
            if hasattr(self.model, "_count_images")
            else len(images),
        )
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs = self.model._move_inputs_to_device(inputs, self.device)
        if hasattr(self.model, "_drop_trivial_attention_mask"):
            inputs = self.model._drop_trivial_attention_mask(inputs)
        return inputs

    def encode_context(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        action_horizon: int | None = None,
    ) -> tuple[AEFlowContext, dict[str, np.ndarray]]:
        """Frozen VLM forward → detached KV + pooled/token features."""
        ext_pil = _to_pil(external_cam)
        wri_pil = _to_pil(wrist_cam)
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (8,):
            raise ValueError(f"state must be shape (8,), got {state_f32.shape}")

        inputs = self._build_robot_inputs([ext_pil, wri_pil], instruction, state_f32)
        bb = self._backbone()
        with torch.no_grad():
            # Native continuous inference enters the inner MolmoAct2 model here.
            # Calling the outer conditional-generation wrapper produces slightly
            # different KV tensors in bf16 and breaks same-source parity.
            outputs = bb(
                **inputs,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            encoder_kv = bb._extract_kv_states(outputs.past_key_values)
            encoder_kv = tuple(
                (k.detach().clone(), v.detach().clone()) for k, v in encoder_kv
            )
            enc_mask = bb._get_encoder_attention_mask(
                inputs.get("input_ids"), inputs.get("attention_mask")
            ).detach().clone()
            depth_gate, depth_mask = bb._depth_gate_from_condition(
                input_ids=inputs.get("input_ids"),
                encoder_attention_mask=enc_mask,
                layer_kv_states=encoder_kv,
            )
            encoder_kv = bb._apply_depth_gate_to_layer_kv_states(
                encoder_kv, depth_mask, depth_gate
            )
            encoder_kv = tuple((k.detach(), v.detach()) for k, v in encoder_kv)

        contract = self.action_contract()
        resolved_horizon = int(action_horizon or contract["action_horizon"])
        if resolved_horizon != contract["action_horizon"]:
            raise ValueError(
                "Custom AE paths must preserve Molmo's native action horizon: "
                f"{resolved_horizon} != {contract['action_horizon']}"
            )
        batch_size = encoder_kv[0][0].shape[0]
        action_dim_is_pad = self._action_dim_is_pad(batch_size)
        ctx = AEFlowContext(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=enc_mask,
            action_dim=contract["action_dim"],
            max_action_dim=contract["max_action_dim"],
            action_horizon=contract["action_horizon"],
            n_action_steps=contract["n_action_steps"],
            n_obs_steps=contract["n_obs_steps"],
            action_dim_is_pad=action_dim_is_pad,
        )

        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and getattr(outputs, "hidden_states", None) is not None:
            hidden = outputs.hidden_states[-1]
        feats: dict[str, np.ndarray] = {}
        if hidden is not None:
            mask_t = inputs.get("attention_mask")
            pooled = _mean_pool_hidden(hidden, mask_t)
            features = pooled[0].detach().float().cpu().numpy().astype(np.float32)
            if features.shape[0] != FEATURE_DIM:
                out_f = np.zeros(FEATURE_DIM, dtype=np.float32)
                n = min(FEATURE_DIM, int(features.shape[0]))
                out_f[:n] = features[:n]
                features = out_f
            feats["features"] = features
            if self.feature_mode in ("tokens", "both", "rl_token"):
                feats["token_features"] = (
                    hidden[0].detach().to(dtype=torch.float16).cpu().numpy()
                )
                if mask_t is not None:
                    feats["token_attention_mask"] = (
                        mask_t[0].detach().to(dtype=torch.uint8).cpu().numpy()
                    )
                else:
                    feats["token_attention_mask"] = np.ones(
                        hidden.shape[1], dtype=np.uint8
                    )
            if (
                self.feature_mode in ("rl_token", "both")
                and self.rlt is not None
                and "token_features" in feats
            ):
                tok = hidden.to(device=self.device, dtype=torch.float32)
                msk = (
                    mask_t.to(device=self.device)
                    if mask_t is not None
                    else torch.ones(hidden.shape[:2], device=self.device, dtype=torch.long)
                )
                z = self.rlt.encode_z(tok, msk, use_target=False, detach=True)
                feats["z_rl"] = z[0].float().cpu().numpy().astype(np.float32)
        return ctx, feats

    def velocity(
        self,
        ctx: AEFlowContext,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        action_context: Any | None = None,
        modulation: Any | None = None,
    ) -> torch.Tensor:
        """AE velocity in Molmo's padded normalized coordinate system."""
        ae = self._action_expert()
        if x_t.ndim != 3 or x_t.shape[1:] != (
            ctx.action_horizon,
            ctx.max_action_dim,
        ):
            raise ValueError(
                "AE velocity requires native padded shape "
                f"(B,{ctx.action_horizon},{ctx.max_action_dim}), got {tuple(x_t.shape)}"
            )
        x_pad = self._mask_padded_dims(
            x_t.to(dtype=ae.action_embed.weight.dtype),
            ctx.action_dim_is_pad,
        )
        if t.ndim == 2:
            t_flat = t.reshape(-1)
        else:
            t_flat = t
        t_flat = t_flat.to(device=x_pad.device, dtype=torch.float32)
        if action_context is None:
            action_context = ae.prepare_context(
                encoder_kv_states=ctx.encoder_kv_states,
                encoder_attention_mask=ctx.encoder_attention_mask,
                state_embeddings=None,
                batch_size=x_pad.shape[0],
                seq_len=ctx.action_horizon,
                device=self.device,
                dtype=x_pad.dtype,
            )
        v_pad = ae.forward_with_context(
            x_pad,
            t_flat,
            context=action_context,
            modulation=modulation,
        )
        return self._mask_padded_dims(v_pad, ctx.action_dim_is_pad)

    def sample_native_source(
        self,
        ctx: AEFlowContext,
        *,
        seed: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if seed is not None and generator is not None:
            raise ValueError("Pass either seed or generator, not both")
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
        ae = self._action_expert()
        source = torch.randn(
            (
                ctx.encoder_kv_states[0][0].shape[0],
                ctx.action_horizon,
                ctx.max_action_dim,
            ),
            device=self.device,
            dtype=ae.action_embed.weight.dtype,
            generator=generator,
        )
        return self._mask_padded_dims(source, ctx.action_dim_is_pad)

    def integrate_native(
        self,
        ctx: AEFlowContext,
        source_native: torch.Tensor,
        *,
        steps: int,
        apply_guide: bool = False,
        rlt_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler integration over the full native Molmo action horizon."""
        if steps < 1:
            raise ValueError(f"steps must be positive, got {steps}")
        if source_native.shape[1:] != (
            ctx.action_horizon,
            ctx.max_action_dim,
        ):
            raise ValueError(
                "Native source shape mismatch: "
                f"{tuple(source_native.shape)} vs "
                f"(B,{ctx.action_horizon},{ctx.max_action_dim})"
            )
        if apply_guide:
            if self.rlt is None or self.rlt.guide is None or rlt_state is None:
                raise RuntimeError(
                    "Guided AE integration requires an RLT guide and state"
                )
            if ctx.action_horizon < CHUNK_SIZE or ctx.action_dim < ACTION_DIM:
                raise RuntimeError("Native AE contract is smaller than the RLT chunk")

        ae = self._action_expert()
        trajectory = self._mask_padded_dims(
            source_native.to(device=self.device, dtype=ae.action_embed.weight.dtype),
            ctx.action_dim_is_pad,
        )
        action_context = ae.prepare_context(
            encoder_kv_states=ctx.encoder_kv_states,
            encoder_attention_mask=ctx.encoder_attention_mask,
            state_embeddings=None,
            batch_size=trajectory.shape[0],
            seq_len=ctx.action_horizon,
            device=self.device,
            dtype=trajectory.dtype,
        )
        timesteps = [
            torch.full(
                (trajectory.shape[0],),
                index / float(steps),
                device=self.device,
                dtype=torch.float32,
            )
            for index in range(steps)
        ]
        # Always prepare from current trainable weights.  Reusing Molmo's eval
        # cache across adapter/base switches silently serves stale modulation.
        modulations = ae.prepare_modulation_cache(timesteps)
        dt = 1.0 / float(steps)
        for index, timestep in enumerate(timesteps):
            velocity = self.velocity(
                ctx,
                trajectory,
                timestep,
                action_context=action_context,
                modulation=modulations[index],
            )
            if apply_guide:
                compact_x = trajectory[:, :CHUNK_SIZE, :ACTION_DIM].float()
                compact_v = velocity[:, :CHUNK_SIZE, :ACTION_DIM].float()
                mean = self.rlt.action_mean.to(self.device)
                std = self.rlt.action_std.to(self.device).clamp_min(1e-6)
                compact_x_n = (compact_x - mean) / std
                compact_v_n = compact_v / std
                guide_t = timestep[:, None]
                guide_n, _magnitude, _diagnostics = self.rlt.guide.guidance(
                    rlt_state.to(self.device).float(),
                    compact_x_n,
                    guide_t,
                    compact_v_n.detach(),
                )
                velocity = velocity.clone()
                velocity[:, :CHUNK_SIZE, :ACTION_DIM] += (
                    guide_n * std
                ).to(dtype=velocity.dtype)
            trajectory = self._mask_padded_dims(
                trajectory + dt * velocity,
                ctx.action_dim_is_pad,
            )
            if not torch.isfinite(trajectory).all():
                raise RuntimeError(
                    f"Non-finite native AE trajectory at flow step {index}"
                )
        return trajectory

    def _decode_native_actions(
        self,
        actions_native_padded: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        native_full = self._trim_actions(actions_native_padded)
        if not torch.isfinite(native_full).all():
            raise RuntimeError("Non-finite native AE endpoint")
        raw_full = self.unnormalize_actions(native_full)
        if raw_full.ndim != 3:
            raise ValueError(f"Unexpected decoded action shape {raw_full.shape}")
        start = max(0, self.action_contract()["n_obs_steps"] - 1)
        stop = start + self.action_contract()["n_action_steps"]
        raw_deploy = raw_full[:, start:stop]
        raw_chunk = raw_deploy[:, :CHUNK_SIZE]
        if raw_chunk.shape[1:] != (CHUNK_SIZE, ACTION_DIM):
            raise ValueError(f"Unexpected deploy chunk shape {raw_chunk.shape}")
        if not np.isfinite(raw_full).all():
            raise RuntimeError("Non-finite raw AE endpoint")
        return (
            raw_chunk[0].astype(np.float32, copy=False),
            raw_full[0].astype(np.float32, copy=False),
            native_full[0].detach().float().cpu().numpy().astype(np.float32),
        )

    def _predict_custom(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        steps: int,
        apply_guide: bool,
        rlt_state: torch.Tensor | None,
        source_seed: int | None,
        source_native: torch.Tensor | np.ndarray | None,
    ) -> dict[str, Any]:
        if apply_guide:
            if self.rlt is None:
                raise RuntimeError(
                    "Guided AE prediction requires an RLT model"
                )
            if self.rlt.guide is None:
                raise RuntimeError(
                    "Guided AE prediction requires an RLT guide"
                )
            if rlt_state is None:
                raise RuntimeError(
                    "Guided AE prediction requires an encoded RLT state"
                )
        ctx, feats = self.encode_context(
            external_cam,
            wrist_cam,
            instruction,
            state,
        )
        if source_native is None:
            source = self.sample_native_source(ctx, seed=source_seed)
        else:
            source = torch.as_tensor(
                source_native,
                device=self.device,
                dtype=self._action_expert().action_embed.weight.dtype,
            )
            if source.ndim == 2:
                source = source.unsqueeze(0)
            source = self._mask_padded_dims(source, ctx.action_dim_is_pad)
        endpoint = self.integrate_native(
            ctx,
            source,
            steps=steps,
            apply_guide=apply_guide,
            rlt_state=rlt_state,
        )
        actions, raw_full, native_full = self._decode_native_actions(endpoint)
        return {
            "actions": actions,
            "actions_raw_full": raw_full,
            "actions_native_full": native_full,
            "source_native": source[0].detach().float().cpu().numpy().astype(np.float32),
            "source_seed": source_seed,
            **feats,
        }

    @torch.inference_mode()
    def predict(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        num_steps: int | None = None,
        apply_guide: bool = False,
        rlt_state: torch.Tensor | None = None,
        action_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
        source_seed: int | None = None,
        source_native: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Roll out trainable AE in Molmo-native coordinates."""
        if action_stats is not None:
            raise ValueError(
                "External raw-action statistics are invalid for Molmo AE flow"
            )
        steps = int(num_steps or self.num_steps)
        with self._lock:
            return self._predict_custom(
                external_cam,
                wrist_cam,
                instruction,
                state,
                steps=steps,
                apply_guide=apply_guide,
                rlt_state=rlt_state,
                source_seed=source_seed,
                source_native=source_native,
            )

    @torch.inference_mode()
    def predict_reference(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        num_steps: int | None = None,
        source_seed: int | None = None,
        source_native: torch.Tensor | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Predict with the frozen base AE, independent of trainable adapters."""
        steps = int(num_steps or self.num_steps)
        with self._lock:
            self.model.eval()
            with self.adapter_disabled():
                out = self._predict_custom(
                    external_cam,
                    wrist_cam,
                    instruction,
                    state,
                    steps=steps,
                    apply_guide=False,
                    rlt_state=None,
                    source_seed=source_seed,
                    source_native=source_native,
                )
        return out

    @torch.inference_mode()
    def predict_native_reference(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        num_steps: int | None = None,
        source_seed: int | None = None,
    ) -> dict[str, Any]:
        """Call Molmo's untouched native inference path for parity probes."""
        steps = int(num_steps or self.num_steps)
        with self._lock:
            self.model.eval()
            with self.adapter_disabled():
                actions, feats = self._predict_unguided(
                    external_cam,
                    wrist_cam,
                    instruction,
                    state,
                    steps=steps,
                    source_seed=source_seed,
                )
        return {"actions": actions, "source_seed": source_seed, **feats}

    def _predict_unguided(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        steps: int,
        source_seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        ext_pil = _to_pil(external_cam)
        wri_pil = _to_pil(wrist_cam)
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)

        captured: dict[str, torch.Tensor] = {}

        def _capture_hidden(module: Any, args: Any, kwargs: Any, out: Any) -> None:  # noqa: ARG001
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None and isinstance(out, (tuple, list)) and len(out) > 0:
                hidden = out[0] if torch.is_tensor(out[0]) else None
            if hidden is None or not torch.is_tensor(hidden):
                return
            prev = captured.get("hidden")
            if prev is None or hidden.shape[1] >= prev.shape[1]:
                captured["hidden"] = hidden
                mask = kwargs.get("attention_mask") if isinstance(kwargs, dict) else None
                if mask is not None and torch.is_tensor(mask):
                    captured["attention_mask"] = mask
                else:
                    captured["attention_mask"] = torch.ones(
                        hidden.shape[:2], device=hidden.device, dtype=torch.long
                    )

        hook_handles = []
        # After Peft wrapping: PeftModel → get_base_model() → .model (HF body).
        hook_root: Any = self.model
        if hasattr(hook_root, "get_base_model"):
            try:
                hook_root = hook_root.get_base_model()
            except Exception:  # noqa: BLE001
                pass
        target = getattr(hook_root, "model", hook_root)
        if target is not None:
            hook_handles.append(target.register_forward_hook(_capture_hidden, with_kwargs=True))
        generator = None
        if source_seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(source_seed))
        try:
            result = self.model.predict_action(
                processor=self.processor,
                images=[ext_pil, wri_pil],
                task=instruction,
                state=state_f32,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=steps,
                generator=generator,
                normalize_language=True,
                enable_cuda_graph=False,
            )
        finally:
            for h in hook_handles:
                h.remove()

        raw = result.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"unexpected action shape {actions.shape}")

        feats: dict[str, np.ndarray] = {}
        if "hidden" in captured:
            hidden = captured["hidden"].float()
            mask_t = captured.get("attention_mask")
            if mask_t is None:
                mask_t = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)
            pooled = _mean_pool_hidden(hidden, mask_t)
            features = pooled[0].detach().float().cpu().numpy().astype(np.float32)
            if features.shape[0] != FEATURE_DIM:
                out_f = np.zeros(FEATURE_DIM, dtype=np.float32)
                n = min(FEATURE_DIM, int(features.shape[0]))
                out_f[:n] = features[:n]
                features = out_f
            feats["features"] = features
            if self.feature_mode in ("tokens", "both", "rl_token"):
                feats["token_features"] = (
                    hidden[0].detach().to(dtype=torch.float16).cpu().numpy()
                )
                feats["token_attention_mask"] = (
                    mask_t[0].detach().to(dtype=torch.uint8).cpu().numpy()
                )
            if (
                self.feature_mode in ("rl_token", "both")
                and self.rlt is not None
            ):
                tok = hidden.to(device=self.device, dtype=torch.float32)
                msk = mask_t.to(device=self.device)
                z = self.rlt.encode_z(tok, msk, use_target=False, detach=True)
                feats["z_rl"] = z[0].float().cpu().numpy().astype(np.float32)
        if self.feature_mode in ("tokens", "both", "rl_token") and "token_features" not in feats:
            # Peft wrapping can miss the VLM hook; fall back to an explicit encode.
            _ctx, feats2 = self.encode_context(
                external_cam, wrist_cam, instruction, state_f32
            )
            del _ctx
            feats.update(feats2)
        if "features" not in feats:
            feats["features"] = np.zeros(FEATURE_DIM, dtype=np.float32)
        return actions, feats

    def _predict_guided(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        steps: int,
        rlt_state: torch.Tensor | None,
        action_stats: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if action_stats is not None:
            raise ValueError(
                "External raw-action statistics are invalid for Molmo AE flow"
            )
        out = self._predict_custom(
            external_cam,
            wrist_cam,
            instruction,
            state,
            steps=steps,
            apply_guide=True,
            rlt_state=rlt_state,
            source_seed=None,
            source_native=None,
        )
        actions = np.asarray(out.pop("actions"), dtype=np.float32)
        feats = {
            key: value
            for key, value in out.items()
            if isinstance(value, np.ndarray)
        }
        return actions, feats

    def save_trainable(
        self,
        path: str | Path,
        *,
        meta: dict[str, Any] | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        torch.save({"ae_trainable": payload, "meta": meta or {}}, path)

    def load_trainable(self, path: str | Path) -> dict[str, Any]:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob.get("ae_trainable", blob)
        expected = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if not isinstance(state, dict):
            raise TypeError(f"AE trainable checkpoint must contain a state dict, got {type(state)}")
        missing_trainable = sorted(set(expected) - set(state))
        unexpected_trainable = sorted(set(state) - set(expected))
        bad_shapes = sorted(
            name
            for name in set(expected) & set(state)
            if tuple(expected[name].shape) != tuple(state[name].shape)
        )
        if missing_trainable or unexpected_trainable or bad_shapes:
            raise RuntimeError(
                "AE trainable checkpoint mismatch: "
                f"missing={missing_trainable[:8]} "
                f"unexpected={unexpected_trainable[:8]} "
                f"bad_shapes={bad_shapes[:8]}"
            )
        missing = self.model.load_state_dict(state, strict=False)
        if missing.unexpected_keys:
            raise RuntimeError(
                "AE trainable checkpoint had unmatched keys: "
                f"{missing.unexpected_keys[:8]}"
            )
        meta = dict(blob.get("meta") or {}) if isinstance(blob, dict) else {}
        self.invalidate_modulation_cache()
        self.loaded_trainable_meta = meta
        log.info("Loaded AE trainable weights from %s missing=%s", path, missing.missing_keys[:8])
        return meta


GuideFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
]
