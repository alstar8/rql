"""CF-aware MolmoAct2-DROID FastAPI server: frozen expert + VLA features/tokens."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from models import FEATURE_DIM, MolmoAct2CF  # noqa: E402
from rlt_models import MolmoAct2RLTCF  # noqa: E402

json_numpy.patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("molmoact2.cf.server")

REPO_ID = "allenai/MolmoAct2-DROID"
NORM_TAG = "franka_droid"
DEFAULT_NUM_STEPS = 10


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
    """Mean-pool (B, S, D) → (B, D) over non-pad tokens."""
    h = last_hidden.float()
    if attention_mask is None:
        return h.mean(dim=1)
    mask = attention_mask.to(dtype=h.dtype).unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (h * mask).sum(dim=1) / denom


class CFPolicy:
    def __init__(
        self,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        cf_ckpt: str | None,
        enable_g: bool,
        action_clip: float,
        return_features: bool = True,
        feature_mode: str = "mean_pool",
        rlt_ckpt: str | None = None,
    ) -> None:
        self.enable_g = enable_g and cf_ckpt is not None
        self.delta_clip = action_clip
        self.return_features = return_features
        self.feature_mode = str(feature_mode)
        local_dir = snapshot_download(repo_id=repo_id)
        log.info("Resolved snapshot dir: %s", local_dir)
        _patch_modeling_for_bf16(local_dir)

        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir, trust_remote_code=True, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        self.device = device
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
        self._lock = threading.Lock()

        self.cf: MolmoAct2CF | None = None
        self.rlt: MolmoAct2RLTCF | None = None
        self.last_residual_rms = 0.0
        self.last_features = np.zeros(FEATURE_DIM, dtype=np.float32)
        self.last_token_features: np.ndarray | None = None
        self.last_token_mask: np.ndarray | None = None
        self.last_z_rl: np.ndarray | None = None
        if cf_ckpt:
            self.cf = MolmoAct2CF.load(cf_ckpt, map_location=device).to(device).eval()
            log.info("Loaded CF ckpt %s enable_g=%s", cf_ckpt, self.enable_g)
        if rlt_ckpt:
            self.rlt = MolmoAct2RLTCF.load(rlt_ckpt, map_location=device).to(device).eval()
            log.info("Loaded RLT ckpt %s feature_mode=%s", rlt_ckpt, self.feature_mode)

    def _extract_pooled_features(
        self,
        images: list[Image.Image],
        instruction: str,
        state_f32: np.ndarray,
    ) -> np.ndarray:
        """Second forward: mean-pool VLM last_hidden_state → (FEATURE_DIM,)."""
        # Mirror predict_action's discrete-state prompting so features match the
        # representation that produced the action chunk.
        out = self.model.predict_action  # keep attribute access for typing silence
        _ = out
        stats = self.model._get_robot_stats()
        norm_tag = stats.validate_tag(NORM_TAG)
        metadata = stats.get_metadata(norm_tag)
        normalized_state = np.asarray(
            stats.normalize_state(state_f32, norm_tag), dtype=np.float32
        )
        num_state_tokens = int(self.model.config.num_state_tokens or 0)
        # Import helpers from the loaded modeling module.
        modeling = sys.modules[self.model.__class__.__module__]
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
            num_images=self.model._count_images(images),
        )
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs = self.model._move_inputs_to_device(inputs, self.device)
        if hasattr(self.model, "_drop_trivial_attention_mask"):
            inputs = self.model._drop_trivial_attention_mask(inputs)

        outputs = self.model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and getattr(outputs, "hidden_states", None) is not None:
            hidden = outputs.hidden_states[-1]
        if hidden is None:
            raise RuntimeError("MolmoAct2 forward returned no hidden states")
        pooled = _mean_pool_hidden(hidden, inputs.get("attention_mask"))
        feat = pooled[0].detach().float().cpu().numpy().astype(np.float32)
        if feat.shape != (FEATURE_DIM,):
            # Tolerate dim mismatch by pad/truncate to FEATURE_DIM.
            out_f = np.zeros(FEATURE_DIM, dtype=np.float32)
            n = min(FEATURE_DIM, int(feat.shape[0]))
            out_f[:n] = feat[:n]
            log.warning("feature dim %s != %d; padded/truncated", feat.shape, FEATURE_DIM)
            return out_f
        return feat

    @torch.inference_mode()
    def predict(
        self,
        external_cam: np.ndarray,
        wrist_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = DEFAULT_NUM_STEPS,
    ) -> tuple[np.ndarray, np.ndarray]:
        ext_pil = _to_pil(external_cam)
        wri_pil = _to_pil(wrist_cam)
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (8,):
            raise ValueError(f"state must be shape (8,), got {state_f32.shape}")

        captured: dict[str, torch.Tensor] = {}

        def _capture_hidden(module: Any, args: Any, kwargs: Any, out: Any) -> None:  # noqa: ARG001
            # Prefer the longest sequence (prefill), not a decode step.
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None and isinstance(out, (tuple, list)) and len(out) > 0:
                hidden = out[0] if torch.is_tensor(out[0]) else None
            if hidden is None or not torch.is_tensor(hidden):
                return
            prev = captured.get("hidden")
            if prev is None or hidden.shape[1] >= prev.shape[1]:
                captured["hidden"] = hidden
                mask = kwargs.get("attention_mask") if isinstance(kwargs, dict) else None
                if mask is None and isinstance(args, tuple) and len(args) > 1:
                    # Best-effort: ignore positional unknowns.
                    mask = None
                if mask is not None and torch.is_tensor(mask):
                    captured["attention_mask"] = mask
                else:
                    captured["attention_mask"] = torch.ones(
                        hidden.shape[:2], device=hidden.device, dtype=torch.long
                    )

        # Prefer hooking the inner VLM so we reuse the action-generation prefill
        # instead of a second full forward (4×/GPU was OOMing with dual forward).
        hook_handles = []
        target = getattr(self.model, "model", None)
        if self.return_features and target is not None:
            hook_handles.append(target.register_forward_hook(_capture_hidden, with_kwargs=True))

        with self._lock:
            try:
                out = self.model.predict_action(
                    processor=self.processor,
                    images=[ext_pil, wri_pil],
                    task=instruction,
                    state=state_f32,
                    norm_tag=NORM_TAG,
                    inference_action_mode="continuous",
                    enable_depth_reasoning=False,
                    num_steps=num_steps,
                    normalize_language=True,
                    enable_cuda_graph=False,
                )
            finally:
                for h in hook_handles:
                    h.remove()

            token_features: np.ndarray | None = None
            token_mask: np.ndarray | None = None
            z_rl: np.ndarray | None = None
            if self.return_features:
                if "hidden" in captured:
                    hidden = captured["hidden"].float()
                    mask_t = captured.get("attention_mask")
                    if mask_t is None:
                        mask_t = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.long)
                    # Mean-pool for backward-compatible `features`.
                    pooled = _mean_pool_hidden(hidden, mask_t)
                    features = pooled[0].detach().float().cpu().numpy().astype(np.float32)
                    if features.shape[0] != FEATURE_DIM:
                        out_f = np.zeros(FEATURE_DIM, dtype=np.float32)
                        n = min(FEATURE_DIM, int(features.shape[0]))
                        out_f[:n] = features[:n]
                        features = out_f
                    if self.feature_mode in ("tokens", "both", "rl_token"):
                        token_features = (
                            hidden[0].detach().to(dtype=torch.float16).cpu().numpy()
                        )
                        token_mask = mask_t[0].detach().to(dtype=torch.uint8).cpu().numpy()
                    if self.feature_mode in ("rl_token", "both") and self.rlt is not None:
                        with torch.no_grad():
                            tok = hidden.to(device=self.device, dtype=torch.float32)
                            msk = mask_t.to(device=self.device)
                            z = self.rlt.encode_z(tok, msk, use_target=False, detach=True)
                            z_rl = z[0].float().cpu().numpy().astype(np.float32)
                else:
                    # Fallback: dedicated feature forward (slower / more VRAM).
                    features = self._extract_pooled_features(
                        [ext_pil, wri_pil], instruction, state_f32
                    )
            else:
                features = np.zeros(FEATURE_DIM, dtype=np.float32)

        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[-1] != 8:
            raise ValueError(f"unexpected action shape {actions.shape}")

        self.last_residual_rms = 0.0
        self.last_features = features
        self.last_token_features = token_features
        self.last_token_mask = token_mask
        self.last_z_rl = z_rl
        if self.enable_g and self.cf is not None:
            s = torch.from_numpy(state_f32).unsqueeze(0).to(self.device)
            h = torch.from_numpy(features).unsqueeze(0).to(self.device)
            refined_rows = []
            deltas = []
            for i in range(actions.shape[0]):
                a = torch.from_numpy(actions[i]).unsqueeze(0).to(self.device)
                r, d = self.cf.refine_raw(s, a, features=h, delta_clip=self.delta_clip)
                refined_rows.append(r.squeeze(0).cpu().numpy())
                deltas.append(d.squeeze(0).cpu().numpy())
            actions = np.stack(refined_rows, axis=0).astype(np.float32)
            delta_arr = np.stack(deltas, axis=0)
            self.last_residual_rms = float(np.sqrt(np.mean(delta_arr**2)))
        return actions, features


def build_app(policy: CFPolicy) -> FastAPI:
    app = FastAPI(title="MolmoAct2-DROID + CF server", version="0.2.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": REPO_ID,
                "norm_tag": NORM_TAG,
                "device": policy.device,
                "enable_g": policy.enable_g,
                "cf_loaded": policy.cf is not None,
                "rlt_loaded": policy.rlt is not None,
                "feature_dim": FEATURE_DIM,
                "feature_mode": policy.feature_mode,
                "return_features": policy.return_features,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "enable_g": policy.enable_g,
                "feature_dim": FEATURE_DIM,
            }
        )

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            body = json_numpy.dumps({"error": f"failed to decode: {e}"})
            return Response(content=body, status_code=400, media_type="application/json")

        try:
            external_cam = payload["external_cam"]
            wrist_cam = payload["wrist_cam"]
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            body = json_numpy.dumps({"error": f"missing field: {e}"})
            return Response(content=body, status_code=400, media_type="application/json")

        num_steps = int(payload.get("num_steps", DEFAULT_NUM_STEPS))
        t0 = time.perf_counter()
        try:
            actions, features = policy.predict(
                external_cam=external_cam,
                wrist_cam=wrist_cam,
                instruction=instruction,
                state=state,
                num_steps=num_steps,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            body = json_numpy.dumps({"error": f"inference failed: {e}"})
            return Response(content=body, status_code=500, media_type="application/json")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        kind = {
            "mean_pool": "llm_mean_pool",
            "tokens": "llm_last_hidden_sequence",
            "rl_token": "rlt_z_rl",
            "both": "llm_tokens_and_mean_pool",
        }.get(policy.feature_mode, "llm_mean_pool")
        payload_out: dict[str, Any] = {
            "actions": actions,
            "features": features,
            "feature_dim": int(features.shape[0]),
            "feature_kind": kind,
            "feature_schema_version": 2 if policy.feature_mode != "mean_pool" else 1,
            "dt_ms": dt_ms,
            "residual_rms": policy.last_residual_rms,
            "enable_g": policy.enable_g,
        }
        if policy.last_token_features is not None:
            payload_out["token_features"] = policy.last_token_features
            payload_out["token_attention_mask"] = policy.last_token_mask
            payload_out["token_count"] = int(policy.last_token_features.shape[0])
            payload_out["token_dim"] = int(policy.last_token_features.shape[1])
        if policy.last_z_rl is not None:
            payload_out["z_rl"] = policy.last_z_rl
            payload_out["z_dim"] = int(policy.last_z_rl.shape[0])
        body = json_numpy.dumps(payload_out)
        return Response(content=body, media_type="application/json")

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--repo_id", type=str, default=REPO_ID)
    p.add_argument("--cf_ckpt", type=str, default=None)
    p.add_argument("--enable_g", action="store_true", default=False)
    p.add_argument("--disable_g", action="store_true", default=False)
    p.add_argument(
        "--action_clip",
        type=float,
        default=0.05,
        help="Normalized residual cap (EndpointG is also intrinsically capped at 0.05)",
    )
    p.add_argument("--no_features", action="store_true", default=False)
    p.add_argument(
        "--feature_mode",
        type=str,
        default="mean_pool",
        choices=["mean_pool", "tokens", "rl_token", "both"],
        help="mean_pool (v3), tokens (raw sequence), rl_token (requires --rlt_ckpt), both",
    )
    p.add_argument("--rlt_ckpt", type=str, default=None)
    args = p.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        args.dtype
    ]
    enable_g = bool(args.enable_g) and not bool(args.disable_g)
    policy = CFPolicy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        cf_ckpt=args.cf_ckpt,
        enable_g=enable_g,
        action_clip=args.action_clip,
        return_features=not bool(args.no_features),
        feature_mode=args.feature_mode,
        rlt_ckpt=args.rlt_ckpt,
    )
    dummy = np.zeros((180, 320, 3), dtype=np.uint8)
    try:
        acts, feats = policy.predict(dummy, dummy, "warmup", np.zeros(8, dtype=np.float32))
        log.info("Warmup ok actions=%s features=%s", acts.shape, feats.shape)
    except Exception:  # noqa: BLE001
        log.exception("warmup failed (continuing)")

    import uvicorn

    app = build_app(policy)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
