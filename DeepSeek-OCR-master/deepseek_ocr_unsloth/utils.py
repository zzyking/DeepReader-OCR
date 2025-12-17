"""Small helpers for the Unsloth runner."""
from __future__ import annotations

import argparse
import io
import os
from pathlib import Path

import fitz
from PIL import Image


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unexpected boolean value: {value}")


def ensure_dir(path: str | Path) -> str:
    os.makedirs(path, exist_ok=True)
    return str(path)


def render_pdf(pdf_path: str, dpi: int) -> list[Image.Image]:
    images: list[Image.Image] = []
    pdf_document = fitz.open(pdf_path)
    zoom = max(dpi, 72) / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page in pdf_document:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None
        img_data = pixmap.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        images.append(img)

    pdf_document.close()
    return images
