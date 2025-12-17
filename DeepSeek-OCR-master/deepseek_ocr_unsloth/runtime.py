"""Runtime resolution for prompts and vision modes."""
from __future__ import annotations

import os

from . import config


def resolve_prompt(args) -> str:
    if config.VLLM_CONFIG:
        if getattr(args, "prompt_template", None):
            config.VLLM_CONFIG.set_prompt_template(args.prompt_template)
        if args.prompt is not None:
            return args.prompt
        return getattr(config.VLLM_CONFIG, "PROMPT", config.DEFAULT_PROMPT)
    return args.prompt or config.DEFAULT_PROMPT


def resolve_runtime_settings(args) -> dict:
    base_size = args.base_size
    image_size = args.image_size
    crop_mode = args.crop_mode

    if config.VLLM_CONFIG:
        if args.mode:
            config.VLLM_CONFIG.set_mode(args.mode)
        if base_size is None:
            base_size = getattr(config.VLLM_CONFIG, "BASE_SIZE", None)
        if image_size is None:
            image_size = getattr(config.VLLM_CONFIG, "IMAGE_SIZE", None)
        if crop_mode is None:
            crop_mode = getattr(config.VLLM_CONFIG, "CROP_MODE", None)

    return {
        "base_size": base_size or 1024,
        "image_size": image_size or 640,
        "crop_mode": True if crop_mode is None else crop_mode,
    }


def apply_cuda_env(cuda_visible_devices: str) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    os.environ.setdefault("UNSLOTH_WARN_UNINITIALIZED", "0")
