"""Main entrypoint wiring Unsloth inference components together."""
from __future__ import annotations

import os

from . import cli, runtime, utils
from .runner import run_mixed_image_pdf


def main() -> None:
    args = cli.parse_args()

    runtime.apply_cuda_env(args.cuda_visible_devices)
    # Import unsloth after CUDA env is set so GPU masking is respected.
    import unsloth  # noqa: F401

    prompt = runtime.resolve_prompt(args)
    vision_settings = runtime.resolve_runtime_settings(args)
    output_dir = utils.ensure_dir(os.path.abspath(args.output_dir))

    input_lower = args.input.lower()
    image_requests = []
    pdf_requests = []
    if input_lower.endswith(".pdf"):
        pdf_requests.append(
            {
                "input": args.input,
                "output": output_dir,
                "prompt": prompt,
                "crop_mode": vision_settings["crop_mode"],
                "base_size": vision_settings["base_size"],
                "image_size": vision_settings["image_size"],
                "pdf_render_dpi": args.pdf_render_dpi,
            }
        )
    else:
        image_requests.append(
            {
                "input": args.input,
                "output": output_dir,
                "prompt": prompt,
                "crop_mode": vision_settings["crop_mode"],
                "base_size": vision_settings["base_size"],
                "image_size": vision_settings["image_size"],
            }
        )

    run_mixed_image_pdf(
        image_requests=image_requests,
        pdf_requests=pdf_requests,
        model_path=args.model_path,
        image_crop_mode=vision_settings["crop_mode"],
        pdf_crop_mode=vision_settings["crop_mode"],
        cuda_visible_devices=args.cuda_visible_devices,
        keep_model_loaded=True,
        load_in_4bit=args.load_in_4bit,
        compile=args.compile,
        gradient_checkpointing=args.gradient_checkpointing,
        revision=args.revision,
        local_dir=args.local_dir,
        prompt=prompt,
        render_dpi=args.pdf_render_dpi,
        base_size=vision_settings["base_size"],
        image_size=vision_settings["image_size"],
        crop_mode=vision_settings["crop_mode"],
    )

if __name__ == "__main__":
    main()
