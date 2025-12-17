"""Inference helpers for Unsloth FastVisionModel."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .utils import ensure_dir, render_pdf


def run_image(
    model,
    tokenizer,
    image_path: str,
    output_dir: str,
    prompt: str,
    runtime: dict,
):
    res = model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=runtime["base_size"],
        image_size=runtime["image_size"],
        crop_mode=runtime["crop_mode"],
        save_results=True,
        test_compress=False,
    )
    print("Inference result:")
    print(res)
    return res


def run_pdf(
    model,
    tokenizer,
    pdf_path: str,
    output_dir: str,
    prompt: str,
    runtime: dict,
    render_dpi: int,
):
    images = render_pdf(pdf_path, render_dpi)
    if not images:
        raise RuntimeError("No pages rendered from PDF.")

    base_name = Path(pdf_path).stem
    pages_dir = ensure_dir(Path(output_dir) / "pages")

    page_outputs: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, page_img in enumerate(images):
            tmp_img_path = os.path.join(tmpdir, f"{base_name}_page_{idx+1}.png")
            page_img.save(tmp_img_path)
            page_output_dir = ensure_dir(Path(pages_dir) / f"page_{idx+1:04d}")

            res = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=tmp_img_path,
                output_path=page_output_dir,
                base_size=runtime["base_size"],
                image_size=runtime["image_size"],
                crop_mode=runtime["crop_mode"],
                save_results=True,
                test_compress=False,
            )
            page_outputs.append(str(res))
            print(f"[PDF] Page {idx+1}/{len(images)} done.")

    combined_md = os.path.join(output_dir, f"{base_name}.md")
    with open(combined_md, "w", encoding="utf-8") as f:
        for idx, text in enumerate(page_outputs, start=1):
            f.write(f"## Page {idx}\n\n{text}\n\n")
    print(f"PDF inference finished. Combined markdown: {combined_md}")

    return combined_md
