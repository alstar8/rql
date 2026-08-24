"""Frozen MolmoAct2 server: returns an action chunk AND the final-layer tokens.

Runs in the molmoact2 venv (torch 2.5.1); everything else runs in the
molmospaces venv (torch 2.7). HTTP is the seam between them.

    GET  /act   -> {"status": "ok", ...}
    POST /act   -> {"actions": (15,8), "token_features": (S,2560),
                    "token_attention_mask": (S,), "dt_ms": float}

The token sequence is captured by a forward hook on the inner VLM, reusing the
prefill that action generation already runs. A second forward pass would double
the cost and, at several server instances per GPU, run out of memory.

    python -m rlt.serve_vla --port 8000 --device cuda:0
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .cli import describe, parse_into
from .config import ServeConfig

json_numpy.patch()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("rlt.serve")

NORM_TAG = "franka_droid"


def to_pil(arr: Any) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


class FrozenVLA:
    """Loads MolmoAct2 once and serializes inference behind a lock."""

    def __init__(self, cfg: ServeConfig) -> None:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.dtype]

        # The model code reads norm_stats.json from config._name_or_path, so it
        # must be loaded from a real directory, not a repo id.
        local_dir = snapshot_download(repo_id=cfg.repo_id)
        log.info("snapshot: %s", local_dir)

        # tokenizer_config.json ships extra_special_tokens as a list; transformers
        # >=4.46 wants a dict. The model only resolves these via
        # convert_tokens_to_ids, so an empty dict is safe.
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(local_dir, trust_remote_code=True, torch_dtype=dtype)
            .to(cfg.device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.device = cfg.device
        self.num_steps = cfg.num_steps

        # _move_inputs_to_device moves tensors but does not cast them, so the
        # processor's float32 pixel_values hit bf16 weights and blow up with
        # "mat1 and mat2 must have the same dtype". Cast on the instance.
        target = next(self.model.parameters()).dtype

        def move_and_cast(inputs: Any, dev: Any, _t: torch.dtype = target) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    v = v.to(dev)
                    if v.is_floating_point() and v.dtype != _t:
                        v = v.to(_t)
                out[k] = v
            return out

        self.model._move_inputs_to_device = move_and_cast
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(self, external_cam, wrist_cam, instruction: str, state: np.ndarray) -> dict:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (8,):
            raise ValueError(f"state must be (8,), got {state.shape}")

        captured: dict[str, torch.Tensor] = {}

        def capture(module, args, kwargs, out) -> None:
            h = getattr(out, "last_hidden_state", None)
            if h is None and isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
                h = out[0]
            if not torch.is_tensor(h):
                return
            # Keep the longest sequence seen: that is the prefill, not a decode step.
            prev = captured.get("hidden")
            if prev is None or h.shape[1] >= prev.shape[1]:
                captured["hidden"] = h
                m = kwargs.get("attention_mask") if isinstance(kwargs, dict) else None
                captured["mask"] = (
                    m if torch.is_tensor(m) else torch.ones(h.shape[:2], device=h.device, dtype=torch.long)
                )

        handle = self.model.model.register_forward_hook(capture, with_kwargs=True)
        try:
            with self._lock:
                out = self.model.predict_action(
                    processor=self.processor,
                    images=[to_pil(external_cam), to_pil(wrist_cam)],
                    task=instruction,
                    state=state,
                    norm_tag=NORM_TAG,
                    inference_action_mode="continuous",
                    enable_depth_reasoning=False,
                    num_steps=self.num_steps,
                    normalize_language=True,
                    enable_cuda_graph=False,
                )
        finally:
            handle.remove()

        actions = out.actions
        if torch.is_tensor(actions):
            actions = actions.detach().to(torch.float32).cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]

        if "hidden" not in captured:
            raise RuntimeError("forward hook captured no hidden states")
        hidden = captured["hidden"][0].detach().to(torch.float32).cpu().numpy()
        mask = captured["mask"][0].detach().to(torch.float32).cpu().numpy()

        return {"actions": actions, "token_features": hidden, "token_attention_mask": mask}


def build_app(vla: FrozenVLA, cfg: ServeConfig) -> FastAPI:
    app = FastAPI(title="RLT frozen-VLA server")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {"status": "ok", "repo_id": cfg.repo_id, "device": cfg.device, "dtype": str(vla.model.dtype)}
        )

    @app.post("/act")
    async def act(request: Request) -> Response:
        payload = json_numpy.loads((await request.body()).decode("utf-8"))
        t0 = time.perf_counter()
        try:
            out = vla.predict(
                payload["external_cam"], payload["wrist_cam"], str(payload["instruction"]), payload["state"]
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return Response(
                content=json_numpy.dumps({"error": str(e)}), status_code=500, media_type="application/json"
            )
        out["dt_ms"] = (time.perf_counter() - t0) * 1000.0
        return Response(content=json_numpy.dumps(out), media_type="application/json")

    return app


def main() -> None:
    cfg = parse_into(ServeConfig)
    print(describe(cfg))

    vla = FrozenVLA(cfg)
    if cfg.warmup:
        log.info("warmup ...")
        t0 = time.perf_counter()
        vla.predict(np.zeros((224, 224, 3), np.uint8), np.zeros((224, 224, 3), np.uint8), "warmup", np.zeros(8, np.float32))
        log.info("warmup done in %.0f ms", (time.perf_counter() - t0) * 1000)

    import uvicorn

    uvicorn.run(build_app(vla, cfg), host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
