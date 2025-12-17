"""Unified runner for Unsloth DeepSeek-OCR (images + PDFs)."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

# Apply CUDA mask as early as possible in this module.
_cuda_env = os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES")
if _cuda_env:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_env

from . import config, infer
from .concurrency import run_tasks
from .engine_cache import get_engine, unload_engine


def _clone(spec: dict) -> dict:
    return deepcopy(spec)


def _run_images(model, tokenizer, requests: List[dict], prompt: str, runtime: dict) -> List[dict]:
    results: List[dict] = []
    for spec in requests:
        image_path = spec["input"]
        output_dir = spec.get("output")
        if not output_dir:
            stem = Path(image_path).stem
            output_dir = Path("outputs") / f"unsloth_image_{stem}"
        res = infer.run_image(
            model=model,
            tokenizer=tokenizer,
            image_path=image_path,
            output_dir=str(output_dir),
            prompt=spec.get("prompt", prompt),
            runtime={
                "base_size": spec.get("base_size", runtime["base_size"]),
                "image_size": spec.get("image_size", runtime["image_size"]),
                "crop_mode": spec.get("crop_mode", runtime["crop_mode"]),
            },
        )
        results.append({"input": image_path, "output_dir": str(output_dir), "text": str(res)})
    return results


def _run_pdfs(model, tokenizer, requests: List[dict], prompt: str, runtime: dict, render_dpi: int) -> List[dict]:
    results: List[dict] = []
    for spec in requests:
        pdf_path = spec["input"]
        output_dir = spec.get("output")
        if not output_dir:
            stem = Path(pdf_path).stem
            output_dir = Path("outputs") / f"unsloth_pdf_{stem}"
        combined_md = infer.run_pdf(
            model=model,
            tokenizer=tokenizer,
            pdf_path=pdf_path,
            output_dir=str(output_dir),
            prompt=spec.get("prompt", prompt),
            runtime={
                "base_size": spec.get("base_size", runtime["base_size"]),
                "image_size": spec.get("image_size", runtime["image_size"]),
                "crop_mode": spec.get("crop_mode", runtime["crop_mode"]),
            },
            render_dpi=spec.get("pdf_render_dpi", spec.get("render_dpi", render_dpi)),
        )
        results.append(
            {
                "input": pdf_path,
                "output_dir": str(output_dir),
                "combined_md": combined_md,
            }
        )
    return results


def run_mixed_image_pdf(
    image_requests: Optional[List[dict]] = None,
    pdf_requests: Optional[List[dict]] = None,
    *,
    model_path: Optional[str] = None,
    image_crop_mode: Optional[bool] = None,
    pdf_crop_mode: Optional[bool] = None,
    num_workers: Optional[int] = None,  # parity only; unused
    max_concurrency: Optional[int] = None,
    cuda_visible_devices: Optional[str] = None,
    keep_model_loaded: Optional[bool] = None,
    load_in_4bit: bool = False,
    compile: bool = True,
    gradient_checkpointing: str = "unsloth",
    revision: Optional[str] = None,
    local_dir: Optional[str] = None,
    prompt: Optional[str] = None,
    render_dpi: Optional[int] = None,
    base_size: Optional[int] = None,
    image_size: Optional[int] = None,
    crop_mode: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run mixed image/PDF OCR with cached Unsloth model (vLLM-style API)."""

    image_requests = image_requests or []
    pdf_requests = pdf_requests or []
    if not image_requests and not pdf_requests:
        return {"image_results": [], "pdf_results": []}

    model_path = model_path or config.DEFAULT_MODEL_PATH
    prompt = prompt or config.DEFAULT_PROMPT
    render_dpi = render_dpi or config.DEFAULT_RENDER_DPI

    vision_settings = {
        "base_size": base_size or 1024,
        "image_size": image_size or 640,
        "crop_mode": True if crop_mode is None else crop_mode,
    }
    default_image_crop = image_crop_mode if image_crop_mode is not None else vision_settings["crop_mode"]
    default_pdf_crop = pdf_crop_mode if pdf_crop_mode is not None else vision_settings["crop_mode"]

    if cuda_visible_devices is None:
        cuda_visible_devices = os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "0"))

    model, tokenizer = get_engine(
        model_path=model_path,
        local_dir=local_dir,
        revision=revision,
        load_in_4bit=load_in_4bit,
        compile=compile,
        gradient_checkpointing=gradient_checkpointing,
        cuda_visible_devices=cuda_visible_devices,
    )

    image_specs = [_clone(spec) for spec in image_requests]
    pdf_specs = [_clone(spec) for spec in pdf_requests]

    tasks = []
    if image_specs:
        tasks.append(
            lambda: _run_images(
                model=model,
                tokenizer=tokenizer,
                requests=image_specs,
                prompt=prompt,
                runtime={
                    "base_size": vision_settings["base_size"],
                    "image_size": vision_settings["image_size"],
                    "crop_mode": default_image_crop,
                },
            )
        )
    if pdf_specs:
        tasks.append(
            lambda: _run_pdfs(
                model=model,
                tokenizer=tokenizer,
                requests=pdf_specs,
                prompt=prompt,
                runtime={
                    "base_size": vision_settings["base_size"],
                    "image_size": vision_settings["image_size"],
                    "crop_mode": default_pdf_crop,
                },
                render_dpi=render_dpi,
            )
        )

    results = run_tasks(tasks, max_concurrency=max_concurrency or 1)

    image_results: List[dict] = []
    pdf_results: List[dict] = []
    for item in results:
        if isinstance(item, list) and item and "combined_md" in item[0]:
            pdf_results.extend(item)
        elif isinstance(item, list):
            image_results.extend(item)

    if keep_model_loaded is False:
        unload_engine()

    return {"image_results": image_results, "pdf_results": pdf_results}
