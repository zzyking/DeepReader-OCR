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

import gradio as gr  # noqa: E402
import config  # noqa: E402
from run_dpsk_ocr_image import run_image_pipeline, unload_image_engines  # noqa: E402
from run_dpsk_ocr_pdf import run_pdf_pipeline, unload_pdf_models  # noqa: E402


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


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "" or value is False:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prepare_session(input_file_path: str) -> Tuple[Path, Path, Path, Path]:
    input_suffix = Path(input_file_path).suffix.lower()
    session_id = uuid.uuid4().hex
    session_dir = SESSION_ROOT / session_id
    output_dir = session_dir / "output"
    tmp_dir = session_dir / "tmp"
    session_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    input_target = session_dir / f"input{input_suffix}"
    shutil.copy(input_file_path, input_target)
    return session_dir, output_dir, tmp_dir, input_target


def run_deepreader(
    uploaded_file: Union[str, None],
    mode: str,
    template: str,
    prompt: str,
    gpu_mem_util: float,
    cuda_visible_devices: str,
    max_concurrency: str,
    num_workers: str,
    skip_repeat: bool,
    keep_models_loaded: bool,
):
    if not uploaded_file:
        return None, "No file submitted. Please upload an image or PDF."

    input_path = Path(uploaded_file)
    suffix = input_path.suffix.lower()

    if suffix not in SUPPORTED_IMAGE_SUFFIXES and suffix != ".pdf":
        return None, f"Unsupported file type: {suffix}. Please upload an image or PDF."

    cleanup_sessions()

    session_dir, output_dir, tmp_dir, staged_input = _prepare_session(str(input_path))

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

    previous_tmp = os.environ.get("TMPDIR")
    os.environ["TMPDIR"] = str(tmp_dir)

    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            if is_pdf:
                mc_value = _safe_int(max_concurrency)
                nw_value = _safe_int(num_workers)

                kwargs = {}
                if mc_value:
                    kwargs["max_concurrency"] = mc_value
                if nw_value:
                    kwargs["num_workers"] = nw_value

                run_pdf_pipeline(
                    input_path=str(staged_input),
                    output_path=str(output_dir),
                    prompt_text=runtime_prompt,
                    crop_mode=effective_crop_mode,
                    skip_repeat=skip_repeat,
                    cuda_visible_devices=cuda_visible_devices or None,
                    gpu_memory_utilization=gpu_mem_util,
                    keep_model_loaded=keep_models_loaded,
                    **kwargs,
                )
            else:
                run_image_pipeline(
                    input_path=str(staged_input),
                    output_path=str(output_dir),
                    prompt=runtime_prompt,
                    crop_mode=effective_crop_mode,
                    cuda_visible_devices=cuda_visible_devices or None,
                    gpu_memory_utilization=gpu_mem_util,
                    keep_model_loaded=keep_models_loaded,
                )
    except Exception:
        unload_image_engines()
        unload_pdf_models()
        error_trace = traceback.format_exc()
        log_text = buffer.getvalue() + f"\n[error]\n{error_trace}"
        (output_dir / "gradio_run.log").write_text(log_text, encoding="utf-8")
        return None, log_text.strip()
    finally:
        if previous_tmp is not None:
            os.environ["TMPDIR"] = previous_tmp
        else:
            os.environ.pop("TMPDIR", None)

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
    default_cuda = os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    with gr.Blocks(title="DeepReader Gradio Interface") as demo:
        gr.Markdown(
            """
            # DeepReader
            Upload an image or PDF to generate Markdown, figure crops, and annotated layouts.
            The pipeline runs the appropriate DeepSeek-OCR flow and returns a zipped bundle of results.
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=3, min_width=360):
                gr.Markdown("### Document Input & Prompt")
                file_input = gr.File(label="Document (image or PDF)", file_types=None, type="filepath")
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
                    lines=6,
                    placeholder="<image>\n<|grounding|>Convert the document to markdown.",
                )

            with gr.Column(scale=2, min_width=260):
                gr.Markdown("### Model Settings")
                gpu_mem_slider = gr.Slider(
                    label="GPU memory utilisation",
                    minimum=0.1,
                    maximum=1.0,
                    value=config.GPU_MEMORY_UTILIZATION,
                    step=0.05,
                )
                cuda_input = gr.Textbox(label="CUDA visible devices", value=default_cuda, placeholder="0")
                keep_models_checkbox = gr.Checkbox(label="Keep models loaded", value=True)
                gr.Markdown("Allocate ≥10 GB free VRAM for smooth inference.")

                gr.Markdown("### Advanced Settings (No need to change)")
                skip_repeat_checkbox = gr.Checkbox(label="Skip repeat pages (PDF only)", value=config.SKIP_REPEAT)
                with gr.Row():
                    max_concurrency_input = gr.Textbox(label="Max concurrency", value=str(config.MAX_CONCURRENCY))
                    num_workers_input = gr.Textbox(label="Preprocess workers", value=str(config.NUM_WORKERS))
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=3, min_width=360):
                run_button = gr.Button("Run DeepReader", variant="primary")
                zip_output = gr.File(label="Zipped results", interactive=False)
            
            with gr.Column(scale=2, min_width=260):
                unload_button = gr.Button("Unload Models", variant="secondary")
                log_output = gr.Textbox(label="Inference log", lines=20)
            

        run_button.click(
            run_deepreader,
            inputs=[
                file_input,
                mode_dropdown,
                template_dropdown,
                prompt_input,
                gpu_mem_slider,
                cuda_input,
                max_concurrency_input,
                num_workers_input,
                skip_repeat_checkbox,
                keep_models_checkbox,
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
