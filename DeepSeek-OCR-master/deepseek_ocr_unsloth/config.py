"""Helpers to reuse vLLM config defaults for the Unsloth runner."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def load_vllm_config():
    """Load DeepSeek-OCR-vllm/config.py if available."""
    repo_root = Path(__file__).resolve().parents[1]
    vllm_dir = repo_root / "DeepSeek-OCR-vllm"
    if not vllm_dir.exists():
        return None
    sys.path.append(str(vllm_dir))
    try:
        import config as vllm_config  # type: ignore
    except Exception:
        return None
    return vllm_config


VLLM_CONFIG = load_vllm_config()

# Expose defaults that mirror vLLM when present
MODES = getattr(VLLM_CONFIG, "MODES", {})
PROMPT_TEMPLATES = getattr(VLLM_CONFIG, "PROMPT_TEMPLATES", {})

DEFAULT_MODEL_PATH = os.getenv("DEEPREADER_MODEL_PATH", "unsloth/DeepSeek-OCR")
DEFAULT_PROMPT = os.getenv(
    "DEEPREADER_PROMPT",
    getattr(VLLM_CONFIG, "PROMPT", "<image>\nFree OCR.") if VLLM_CONFIG else "<image>\nFree OCR.",
)
DEFAULT_OUTPUT = os.getenv("DEEPREADER_OUTPUT_PATH", "outputs/unsloth")
DEFAULT_RENDER_DPI = getattr(VLLM_CONFIG, "PDF_RENDER_DPI", 144) if VLLM_CONFIG else 144
DEFAULT_MODE = getattr(VLLM_CONFIG, "ACTIVE_MODE", None)
