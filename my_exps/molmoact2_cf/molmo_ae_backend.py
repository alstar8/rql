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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

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
    action_context: Any
    action_dim: int = ACTION_DIM
    max_action_dim: int = 32


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
        try:
            self.model = get_peft_model(self.model, cfg)
        except Exception:
            # Fallback: leaf-name targets that only appear under AE.
            leaves = sorted({n.split(".")[-1] for n in linear_names})
            cfg = LoraConfig(
                r=int(rank),
                lora_alpha=int(alpha),
                lora_dropout=float(dropout),
                bias="none",
                target_modules=leaves,
            )
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

    def _pad_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pad (B,H,8) → (B,H,max_action_dim)."""
        ae = self._action_expert()
        max_dim = int(getattr(ae.config, "action_dim", None) or getattr(self.model.config, "max_action_dim", 32))
        if actions.shape[-1] == max_dim:
            return actions
        if actions.shape[-1] > max_dim:
            return actions[..., :max_dim]
        pad = actions.new_zeros(*actions.shape[:-1], max_dim - actions.shape[-1])
        return torch.cat([actions, pad], dim=-1)

    def _trim_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions[..., :ACTION_DIM]

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

    @torch.no_grad()
    def encode_context(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        action_horizon: int = CHUNK_SIZE,
    ) -> tuple[AEFlowContext, dict[str, np.ndarray]]:
        """Frozen VLM forward → detached KV + pooled/token features."""
        ext_pil = _to_pil(external_cam)
        wri_pil = _to_pil(wrist_cam)
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (8,):
            raise ValueError(f"state must be shape (8,), got {state_f32.shape}")

        inputs = self._build_robot_inputs([ext_pil, wri_pil], instruction, state_f32)
        bb = self._backbone()
        outputs = self.model(
            **inputs,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        encoder_kv = bb._extract_kv_states(outputs.past_key_values)
        # Detach + clone so AE training cannot push grads into VLM.
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
        encoder_kv = tuple(
            (k.detach(), v.detach()) for k, v in encoder_kv
        )

        ae = self._action_expert()
        batch_size = encoder_kv[0][0].shape[0]
        traj_dtype = ae.action_embed.weight.dtype
        action_context = ae.prepare_context(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=enc_mask,
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=int(action_horizon),
            device=self.device,
            dtype=traj_dtype,
        )
        ctx = AEFlowContext(
            encoder_kv_states=encoder_kv,
            encoder_attention_mask=enc_mask,
            action_context=action_context,
            action_dim=ACTION_DIM,
            max_action_dim=int(getattr(self.model.config, "max_action_dim", 32)),
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
    ) -> torch.Tensor:
        """AE flow velocity at (x_t, t). ``x_t`` is (B,H,8) in raw action units."""
        ae = self._action_expert()
        x_pad = self._pad_actions(x_t.to(dtype=ae.action_embed.weight.dtype))
        if t.ndim == 2:
            t_flat = t.reshape(-1)
        else:
            t_flat = t
        t_flat = t_flat.to(device=x_pad.device, dtype=torch.float32)
        v_pad = ae.forward_with_context(
            x_pad,
            t_flat,
            context=ctx.action_context,
            modulation=None,
        )
        return self._trim_actions(v_pad)

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
    ) -> dict[str, Any]:
        """Rollout act: AE ODE, optionally ``v_AE + G`` with G from RLT guide."""
        steps = int(num_steps or self.num_steps)
        with self._lock:
            if apply_guide and self.rlt is not None and self.rlt.guide is not None:
                actions, feats = self._predict_guided(
                    external_cam,
                    wrist_cam,
                    instruction,
                    state,
                    steps=steps,
                    rlt_state=rlt_state,
                    action_stats=action_stats,
                )
            else:
                actions, feats = self._predict_unguided(
                    external_cam, wrist_cam, instruction, state, steps=steps
                )
        out: dict[str, Any] = {"actions": actions, **feats}
        return out

    def _predict_unguided(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        *,
        steps: int,
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
                external_cam, wrist_cam, instruction, state_f32, action_horizon=CHUNK_SIZE
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
        """Euler ``x ← x + (v_AE + G) dt`` in raw action space with G from RLT."""
        if self.rlt is None or self.rlt.guide is None:
            return self._predict_unguided(
                external_cam, wrist_cam, instruction, state, steps=steps
            )
        ctx, feats = self.encode_context(
            external_cam, wrist_cam, instruction, state, action_horizon=CHUNK_SIZE
        )
        ae = self._action_expert()
        b = 1
        x = torch.randn(
            (b, CHUNK_SIZE, ACTION_DIM),
            device=self.device,
            dtype=ae.action_embed.weight.dtype,
        )
        dt = 1.0 / float(steps)
        if action_stats is None:
            mean = self.rlt.action_mean.to(self.device)
            std = self.rlt.action_std.to(self.device)
        else:
            mean, std = action_stats
            mean = mean.to(self.device)
            std = std.to(self.device)
        if rlt_state is None:
            # Fall back to zeros-state guide (should be rare).
            s = torch.zeros(
                (1, self.rlt.state_dim), device=self.device, dtype=torch.float32
            )
        else:
            s = rlt_state.to(self.device)

        for i in range(steps):
            t = torch.full((b, 1), i / float(steps), device=self.device, dtype=torch.float32)
            v = self.velocity(ctx, x, t)
            x_n = (x.float() - mean) / std.clamp_min(1e-6)
            v_n = v.float() / std.clamp_min(1e-6)
            g_n, _, _ = self.rlt.guide.guidance(s.float(), x_n, t, v_n.detach())
            g = g_n * std
            x = x + (v + g.to(dtype=x.dtype)) * dt

        actions = x[0].detach().float().cpu().numpy().astype(np.float32)
        return actions, feats

    def save_trainable(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            name: param.detach().cpu()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        torch.save({"ae_trainable": payload}, path)

    def load_trainable(self, path: str | Path) -> None:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        state = blob.get("ae_trainable", blob)
        missing = self.model.load_state_dict(state, strict=False)
        log.info("Loaded AE trainable weights from %s missing=%s", path, missing.missing_keys[:8])


GuideFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor, dict[str, Any]],
]
