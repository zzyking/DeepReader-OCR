"""Cache the Unsloth model to avoid reloads between requests."""
from __future__ import annotations

import os
import threading
from typing import Optional, Tuple

import torch

from . import model as model_loader

_engine_lock = threading.Lock()
_cached: Optional[Tuple[object, object, Tuple]] = None  # (model, tokenizer, key)


def _make_key(
    model_path: str,
    local_dir: Optional[str],
    revision: Optional[str],
    load_in_4bit: bool,
    compile: bool,
    gradient_checkpointing: str,
    cuda_visible_devices: Optional[str],
) -> Tuple:
    return (
        model_path,
        local_dir,
        revision,
        bool(load_in_4bit),
        bool(compile),
        gradient_checkpointing,
        cuda_visible_devices or "",
    )


def get_engine(
    model_path: str,
    local_dir: Optional[str],
    revision: Optional[str],
    load_in_4bit: bool,
    compile: bool,
    gradient_checkpointing: str,
    cuda_visible_devices: Optional[str] = None,
):
    key = _make_key(model_path, local_dir, revision, load_in_4bit, compile, gradient_checkpointing, cuda_visible_devices)
    mask = cuda_visible_devices or os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES")
    if mask:
        os.environ["CUDA_VISIBLE_DEVICES"] = mask
    with _engine_lock:
        global _cached
        if _cached and _cached[2] == key:
            return _cached[0], _cached[1]
        model, tokenizer = model_loader.load_unsloth_model(
            model_path=model_path,
            local_dir=local_dir,
            revision=revision,
            load_in_4bit=load_in_4bit,
            compile=compile,
            gradient_checkpointing=gradient_checkpointing,
        )
        _cached = (model, tokenizer, key)
        return model, tokenizer


def unload_engine() -> None:
    with _engine_lock:
        global _cached
        _cached = None
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
