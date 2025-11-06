import argparse
import io
import os
import shutil
import sys
import time
import uuid
import zipfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Optional, Tuple, Union

APP_ROOT = Path(__file__).resolve().parent
VLLM_DIR = APP_ROOT / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"

sys.path.insert(0, str(VLLM_DIR))

os.environ['GRADIO_TEMP_DIR'] = './tmp'

import gradio as gr  # noqa: E402
import config  # noqa: E402
from run_dpsk_ocr_image import unload_image_engines  # noqa: E402
from run_dpsk_ocr_pdf import unload_pdf_models  # noqa: E402
from mixed_runner import run_mixed_image_pdf  # noqa: E402

SESSION_ROOT = (APP_ROOT / "outputs" / "gradio_sessions").resolve()
SESSION_ROOT.mkdir(parents=True, exist_ok=True)


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def cleanup_sessions(max_sessions: int = 20, max_age_hours: int = 24) -> None:
    entries = []
    cutoff = time.time() - max_age_hours * 3600

    for item in SESSION_ROOT.iterdir():
        if item.is_dir():
            try:
                mtime = item.stat().st_mtime
            except OSError:
                continue
            entries.append((mtime, item))

    # Remove directories beyond retention count (newest first)
    entries.sort(reverse=True)
    for _, path in entries[max_sessions:]:
        shutil.rmtree(path, ignore_errors=True)

    # Remove directories older than cutoff
    for mtime, path in entries[:max_sessions]:
        if mtime < cutoff:
            shutil.rmtree(path, ignore_errors=True)


def _env_bool(var_name: str, default: bool) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _prepare_session(input_file_path: str) -> Tuple[Path, Path, Path]:
    input_suffix = Path(input_file_path).suffix.lower()
    session_id = uuid.uuid4().hex
    session_dir = SESSION_ROOT / session_id
    output_dir = session_dir / "output"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_target = session_dir / f"input{input_suffix}"
    shutil.copy(input_file_path, input_target)
    return session_dir, output_dir, input_target


def run_deepreader(
    uploaded_file: Union[str, None],
    mode: str,
    template: str,
    prompt: str,
):
    if not uploaded_file:
        return None, "No file submitted. Please upload an image or PDF."

    input_path = Path(uploaded_file)
    suffix = input_path.suffix.lower()

    if suffix not in SUPPORTED_IMAGE_SUFFIXES and suffix != ".pdf":
        return None, f"Unsupported file type: {suffix}. Please upload an image or PDF."

    cleanup_sessions()

    session_dir, output_dir, staged_input = _prepare_session(str(input_path))

    cuda_visible_devices = os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    cuda_visible_devices = cuda_visible_devices.strip() or None
    keep_models_loaded = _env_bool("DEEPREADER_KEEP_MODELS_LOADED", True)

    mode_choice = (mode or ("gundam (hi-res)" if config.ACTIVE_MODE == "gundam" else "base")).lower()
    mode_map = {
        "base": "base",
        "gundam (hi-res)": "gundam",
        "gundam": "gundam",
    }
    mode_key = mode_map.get(mode_choice, "base")
    config.set_mode(mode_key)

    template_key = (template or config.DEFAULT_TEMPLATE).lower()
    if template_key in config.PROMPT_TEMPLATES:
        config.set_prompt_template(template_key)

    runtime_prompt = prompt or config.PROMPT
    effective_crop_mode = config.CROP_MODE
    is_pdf = suffix == ".pdf"

    gpu_mem_util = config.GPU_MEMORY_UTILIZATION
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            image_specs = []
            pdf_specs = []

            if is_pdf:
                pdf_specs.append(
                    {
                        "input": str(staged_input),
                        "output": str(output_dir),
                        "prompt": runtime_prompt,
                        "crop_mode": effective_crop_mode,
                    }
                )
            else:
                image_specs.append(
                    {
                        "input": str(staged_input),
                        "output": str(output_dir),
                        "prompt": runtime_prompt,
                        "crop_mode": effective_crop_mode,
                        "save_results": True,
                    }
                )

            result_bundle = run_mixed_image_pdf(
                image_requests=image_specs,
                pdf_requests=pdf_specs,
                cuda_visible_devices=cuda_visible_devices,
                gpu_memory_utilization=gpu_mem_util,
                keep_model_loaded=keep_models_loaded,
            )

            if image_specs and result_bundle["image_results"]:
                print("[gradio] image transcription completed.")
            if pdf_specs and result_bundle["pdf_results"]:
                primary = result_bundle["pdf_results"][0]
                print(
                    "[gradio] pdf artifacts:",
                    primary["mmd_path"],
                    primary["mmd_det_path"],
                    primary["layouts_pdf_path"],
                )
    except Exception:
        unload_models()
        error_trace = traceback.format_exc()
        log_text = buffer.getvalue() + f"\n[error]\n{error_trace}"
        (output_dir / "gradio_run.log").write_text(log_text, encoding="utf-8")
        return None, log_text.strip()
    log_text = buffer.getvalue()

    (output_dir / "gradio_run.log").write_text(log_text, encoding="utf-8")

    zip_path = session_dir / "deepreader_output.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for item in output_dir.rglob("*"):
            if item.is_file():
                zip_file.write(item, arcname=item.relative_to(output_dir))

    log_text = log_text.strip() or "Run completed with no console output."
    return str(zip_path), log_text


def unload_models() -> tuple[Optional[str], str]:
    unload_image_engines()
    unload_pdf_models()
    return None, "Models unloaded and GPU cache cleared."


def build_interface() -> gr.Blocks:
    section_css = """
    .section-box {
        border: 1px solid var(--border-color-primary);
        border-radius: 12px;
        padding: 18px;
        background: var(--panel-background-fill);
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.08);
    }
    """

    with gr.Blocks(title="DeepReader Gradio Interface", css=section_css) as demo:
        gr.Markdown(
            """
            # DeepReader
            Upload an image or PDF to generate Markdown, figure crops, and annotated layouts.
            The pipeline runs the appropriate DeepSeek-OCR flow and returns a zipped bundle of results.
            ## Usage Notes
            - Allocate ≥10 GB free VRAM for smooth inference.
            - Advanced runtime options (devices, batching, memory) follow backend defaults.
            """
        )
        with gr.Column(elem_classes="section-box"):
            gr.Markdown("### Inputs")
            with gr.Row(equal_height=True):
                with gr.Column():
                    file_input = gr.File(label="Document (image or PDF)", file_types=None, type="filepath")
                with gr.Column():
                    current_mode_display = "gundam (hi-res)" if config.ACTIVE_MODE == "gundam" else "base"
                    with gr.Row():
                        mode_dropdown = gr.Dropdown(
                            label="Vision mode",
                            choices=["base", "gundam (hi-res)"],
                            value=current_mode_display,
                        )
                        template_dropdown = gr.Dropdown(
                            label="Prompt template",
                            choices=sorted(config.PROMPT_TEMPLATES.keys()),
                            value=config.DEFAULT_TEMPLATE,
                        )
                    prompt_input = gr.Textbox(
                        label="Prompt",
                        value=config.PROMPT,
                        lines=4,
                        placeholder="<image>\n<|grounding|>Convert the document to markdown.",
                    )

        with gr.Column(elem_classes="section-box"):
            gr.Markdown("### Outputs")
            with gr.Row(equal_height=True):
                with gr.Column():
                    run_button = gr.Button("Run DeepReader", variant="primary")
                    zip_output = gr.File(label="Zipped results", interactive=False)
                with gr.Column():
                    unload_button = gr.Button("Unload Models", variant="secondary")
                    log_output = gr.Textbox(label="Inference log", lines=20)

        run_button.click(
            run_deepreader,
            inputs=[
                file_input,
                mode_dropdown,
                template_dropdown,
                prompt_input,
            ],
            outputs=[zip_output, log_output],
        )

        def _template_to_prompt(selected: str) -> str:
            try:
                return config.get_prompt_template(selected)
            except Exception:
                return config.PROMPT

        template_dropdown.change(
            _template_to_prompt,
            inputs=template_dropdown,
            outputs=prompt_input,
        )

        unload_button.click(
            unload_models,
            inputs=None,
            outputs=[zip_output, log_output],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the DeepReader Gradio interface")
    parser.add_argument("--host", default=os.getenv("DEEPREADER_GRADIO_HOST", "127.0.0.1"), help="Host/IP to bind the Gradio server")
    parser.add_argument("--port", type=int, default=int(os.getenv("DEEPREADER_GRADIO_PORT", "7860")), help="Port for the Gradio server")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--queue", action="store_true", help="Enable Gradio queue")
    parser.add_argument(
        "--allow-path",
        action="append",
        dest="allowed_paths",
        help="Additional directories Gradio may serve files from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    interface = build_interface()

    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "inbrowser": not args.no_browser,
        "show_error": True,
    }

    if args.queue:
        interface = interface.queue()

    allowed_paths = {str(APP_ROOT), str(SESSION_ROOT), str(VLLM_DIR)}
    if args.allowed_paths:
        allowed_paths.update(os.path.abspath(path) for path in args.allowed_paths)

    launch_kwargs["allowed_paths"] = list(allowed_paths)

    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
