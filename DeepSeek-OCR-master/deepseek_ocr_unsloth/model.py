"""Model loading helpers for Unsloth FastVisionModel."""
from __future__ import annotations

from typing import Optional

from huggingface_hub import snapshot_download
from transformers import AutoModel

import os


def download_model_if_needed(model_path: str, local_dir: Optional[str], revision: Optional[str]) -> str:
    if not local_dir:
        return model_path
    snapshot_download(
        model_path,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        revision=revision,
    )
    return local_dir


def resolve_gc_arg(arg: str) -> bool | str:
    if arg == "off":
        return False
    if arg == "true":
        return True
    return "unsloth"

def load_unsloth_model(
    model_path: str,
    local_dir: Optional[str],
    revision: Optional[str],
    load_in_4bit: bool,
    compile: bool,
    gradient_checkpointing: str,
):

    from unsloth import FastVisionModel  # defer until after CUDA_VISIBLE_DEVICES is set

    model_dir = download_model_if_needed(model_path, local_dir, revision)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_dir,
        load_in_4bit=load_in_4bit,
        auto_model=AutoModel,
        trust_remote_code=True,
        unsloth_force_compile=compile,
        use_gradient_checkpointing=resolve_gc_arg(gradient_checkpointing),
    )
    return model, tokenizer
