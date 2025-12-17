"""CLI parsing for the Unsloth runner."""
from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

from . import config
from .utils import str2bool


def _choice_list(values: Iterable[str] | None):
    return sorted(values) if values else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR with Unsloth FastVisionModel")
    parser.add_argument("--model-path", default=config.DEFAULT_MODEL_PATH, help="Model identifier or local path")
    parser.add_argument("--local-dir", default=None, help="Optional local dir to place downloaded weights")
    parser.add_argument("--revision", default=None, help="Optional model revision/hash to pin")
    parser.add_argument("--mode", choices=_choice_list(config.MODES.keys()), default=None, help="Predefined vision mode to apply")
    parser.add_argument("--input", "-i", required=True, help="Path to an image or PDF file to OCR")
    parser.add_argument("--output-dir", "-o", default=config.DEFAULT_OUTPUT, help="Directory to save outputs")
    parser.add_argument("--prompt", "-p", default=None, help="Prompt text; overrides template/default")
    parser.add_argument("--prompt-template", choices=_choice_list(config.PROMPT_TEMPLATES.keys()), default=None,
                        help="Select a predefined prompt template")
    parser.add_argument("--base-size", type=int, default=None, help="Base size fed to the vision tower")
    parser.add_argument("--image-size", type=int, default=None, help="Image size used for cropping")
    parser.add_argument("--crop-mode", type=str2bool, default=None, help="Enable cropping during preprocessing")
    parser.add_argument("--load-in-4bit", type=str2bool, default=False, help="Use 4bit loading to reduce VRAM")
    parser.add_argument("--compile", type=str2bool, default=True, help="Enable unsloth_force_compile")
    parser.add_argument(
        "--gradient-checkpointing",
        choices=["off", "true", "unsloth"],
        default="unsloth",
        help="Enable gradient checkpointing; 'unsloth' keeps long-context optimizations",
    )
    parser.add_argument(
        "--pdf-render-dpi",
        type=int,
        default=config.DEFAULT_RENDER_DPI,
        help="DPI used to rasterize PDF pages for model input",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
        help="Comma-separated GPU device ids to expose to the runner",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
