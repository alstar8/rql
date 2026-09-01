#!/usr/bin/env python3
"""Paste real env cameras onto the illustrated architecture draft."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


FIG = Path(__file__).resolve().parent.parent
ENV = FIG / "env"
SRC = FIG / "illustrated" / "architecture_illustrated.png"
OUT = FIG / "illustrated" / "architecture_illustrated_photos.png"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def tile(path: Path, size: int, label: str) -> Image.Image:
    img = ImageOps.fit(Image.open(path).convert("RGB"), (size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 26), (255, 255, 255))
    canvas.paste(img, (0, 26))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, size, 26), fill=(23, 32, 51))
    draw.text((8, 5), label, fill=(255, 255, 255), font=font(max(11, size // 18)))
    return canvas


def main() -> None:
    base = Image.open(SRC).convert("RGB")
    w, h = base.size
    col_w = int(w * 0.168)
    pad = 10
    inner = col_w - 2 * pad
    photos = [
        tile(ENV / "head_reach.png", inner, "Head"),
        tile(ENV / "left_wrist_reach.png", inner, "Left wrist"),
        tile(ENV / "right_wrist_press.png", inner, "Right wrist"),
    ]
    gap = 8
    stack_h = sum(p.height for p in photos) + gap * (len(photos) - 1)
    y0 = max(int(h * 0.13), (h - stack_h) // 2)
    overlay = Image.new("RGB", (col_w, h), (252, 248, 236))
    y = y0
    for photo in photos:
        overlay.paste(photo, (pad, y))
        y += photo.height + gap
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((0, 0, col_w - 1, h - 1), outline=(196, 163, 90), width=2)
    draw.text((pad, 16), "Observation  s", fill=(23, 32, 51), font=font(18))
    out = base.copy()
    out.paste(overlay, (0, 0))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
