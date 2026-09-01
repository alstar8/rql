#!/usr/bin/env python3
"""Find local radio rollouts or download one HF episode, then dump camera stills."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FIG_ROOT = Path(__file__).resolve().parent.parent
ENV_OUT = FIG_ROOT / "env"

LOCAL_CHUNK_CANDIDATES = [
    Path("/workspace/openpi_comet/outputs/pt12_cs32_radio_10x10_states/_cf_ae_chunks"),
    Path("/home/staroverov/B1K_AIRI/openpi_comet/outputs/pt12_cs32_radio_10x10_states/_cf_ae_chunks"),
    Path("/data/openpi_comet/outputs/pt12_cs32_radio_10x10_states/_cf_ae_chunks"),
]

LOCAL_VIDEO_CANDIDATES = [
    Path("/data/DATASETS/behavior/comet-1.5k/videos/task-0000"),
    Path("/data/DATASETS/behavior/2026-challenge-demos/videos"),
]


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def labeled_tile(img: Image.Image, label: str, size: int = 256) -> Image.Image:
    tile = img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 28), (255, 255, 255))
    canvas.paste(tile, (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size, 28), fill=(23, 32, 51))
    draw.text((8, 6), label, fill=(255, 255, 255), font=_font(14))
    return canvas


def strip(tiles: list[Image.Image], pad: int = 8) -> Image.Image:
    w = sum(t.width for t in tiles) + pad * (len(tiles) + 1)
    h = max(t.height for t in tiles) + 2 * pad
    canvas = Image.new("RGB", (w, h), (245, 247, 250))
    x = pad
    for tile in tiles:
        canvas.paste(tile, (x, pad))
        x += tile.width + pad
    return canvas


def save_triplet(head, left, right, stem: str, extra=None) -> None:
    ENV_OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(head).save(ENV_OUT / f"head_{stem}.png")
    Image.fromarray(left).save(ENV_OUT / f"left_wrist_{stem}.png")
    Image.fromarray(right).save(ENV_OUT / f"right_wrist_{stem}.png")
    tiles = [
        labeled_tile(Image.fromarray(head), "Head"),
        labeled_tile(Image.fromarray(left), "Left wrist"),
        labeled_tile(Image.fromarray(right), "Right wrist"),
    ]
    if extra is not None:
        tiles.append(labeled_tile(Image.fromarray(extra), "External"))
        Image.fromarray(extra).save(ENV_OUT / f"external_{stem}.png")
    strip(tiles).save(ENV_OUT / f"cameras_{stem}.png")
    print(ENV_OUT / f"cameras_{stem}.png")


def from_chunk_cache(cache: Path) -> bool:
    index_path = cache / "index.json"
    if not index_path.is_file():
        return False
    index = json.loads(index_path.read_text())
    n = len(index)
    head = np.memmap(cache / "head.bin", dtype=np.uint8, mode="r")
    left = np.memmap(cache / "left.bin", dtype=np.uint8, mode="r")
    right = np.memmap(cache / "right.bin", dtype=np.uint8, mode="r")
    row = 224 * 224 * 3
    if head.size < n * row:
        n = head.size // row
    head = np.asarray(head[: n * row]).reshape(n, 224, 224, 3)
    left = np.asarray(left[: n * row]).reshape(n, 224, 224, 3)
    right = np.asarray(right[: n * row]).reshape(n, 224, 224, 3)
    # Prefer a successful press-like chunk if labels exist.
    picks = {"approach": 0, "reach": n // 2, "press": n - 1}
    for i, rec in enumerate(index):
        if rec.get("success") and rec.get("reward", 0) == 1:
            picks["press"] = i
            picks["reach"] = max(i - 8, 0)
            picks["approach"] = max(i - 24, 0)
            break
    for name, idx in picks.items():
        save_triplet(head[idx], left[idx], right[idx], name)
    print(f"extracted from chunk cache {cache} n={n}")
    return True


def from_huggingface() -> bool:
    try:
        import cv2
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(f"hf/cv2 missing: {exc}")
        return False

    repo = "Hoshipu/behavior-1k-mp-collected-turning-on-radio"
    episode = "00003020"
    cameras = ("head", "left_wrist", "right_wrist", "external")
    paths = {}
    for camera in cameras:
        rel = (
            "success/2025-challenge-demos/videos/task-0000/"
            f"observation.images.rgb.{camera}/episode_{episode}.mp4"
        )
        try:
            paths[camera] = Path(
                hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=rel,
                    local_dir=str(ENV_OUT / "videos"),
                )
            )
        except Exception as exc:
            print(f"skip {camera}: {exc}")
    if not {"head", "left_wrist", "right_wrist"} <= paths.keys():
        return False

    def grab(path: Path, frac: float) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(n * frac), n - 1))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"read fail {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for name, frac in (("approach", 0.15), ("reach", 0.55), ("press", 0.88)):
        extra = grab(paths["external"], frac) if "external" in paths else None
        save_triplet(
            grab(paths["head"], frac),
            grab(paths["left_wrist"], frac),
            grab(paths["right_wrist"], frac),
            name,
            extra=extra,
        )
    print("extracted from Hugging Face videos")
    return True


def main() -> int:
    ENV_OUT.mkdir(parents=True, exist_ok=True)
    for cache in LOCAL_CHUNK_CANDIDATES:
        if from_chunk_cache(cache):
            return 0
    if from_huggingface():
        return 0
    print("no env images found", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
