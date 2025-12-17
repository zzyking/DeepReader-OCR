import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

import config
from concurrency import run_streaming_generation
from engine_cache import get_engine, unload_all_engines
from run_dpsk_ocr_image import make_sampling_params as make_image_sampling_params
from run_dpsk_ocr_image import stream_image_jobs
from run_dpsk_ocr_pdf import finalize_pdf_outputs, stream_pdf_jobs


def _clone_request(spec: dict) -> dict:
    cloned = deepcopy(spec)
    return cloned


def run_mixed_image_pdf(
    image_requests: Optional[List[dict]] = None,
    pdf_requests: Optional[List[dict]] = None,
    *,
    model_path: Optional[str] = None,
    image_crop_mode: Optional[bool] = None,
    pdf_crop_mode: Optional[bool] = None,
    skip_repeat: Optional[bool] = None,
    num_workers: Optional[int] = None,
    max_concurrency: Optional[int] = None,
    cuda_visible_devices: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
    keep_model_loaded: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Execute mixed image and PDF OCR requests concurrently on a single GPU.

    Args:
        image_requests: List of dicts mirroring ``run_image_batch`` request payloads.
        pdf_requests: List of dicts mirroring CLI arguments for ``run_pdf_pipeline``.
        model_path: Optional override for the model identifier/path.
        image_crop_mode: Default crop mode for image requests (overridden per spec via ``crop_mode``).
        pdf_crop_mode: Default crop mode for PDF requests (overridden per spec via ``crop_mode``).
        skip_repeat: Default repeat-skipping flag for PDF requests (per spec override via ``skip_repeat``).
        num_workers: Default preprocessing workers for PDF requests (per spec override via ``num_workers``).
        max_concurrency: Upper bound for concurrent vLLM requests.
        cuda_visible_devices: CUDA device string to expose.
        gpu_memory_utilization: GPU memory utilisation fraction.
        keep_model_loaded: Retain the cached engine after completion.

    Returns:
        Dictionary containing image textual outputs and PDF artifact paths.
    """

    model_path = model_path or config.MODEL_PATH
    default_image_crop = image_crop_mode if image_crop_mode is not None else config.CROP_MODE
    default_pdf_crop = pdf_crop_mode if pdf_crop_mode is not None else config.CROP_MODE
    default_skip_repeat = skip_repeat if skip_repeat is not None else config.SKIP_REPEAT
    default_num_workers = num_workers if num_workers is not None else config.NUM_WORKERS
    default_pdf_render_dpi = config.PDF_RENDER_DPI
    default_pdf_annot_dpi = config.PDF_ANNOT_DPI
    effective_concurrency = max_concurrency if max_concurrency is not None else config.MAX_CONCURRENCY
    keep_model_loaded = keep_model_loaded if keep_model_loaded is not None else config.KEEP_MODELS_LOADED
    if cuda_visible_devices is None:
        cuda_visible_devices = os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    gpu_memory_utilization = gpu_memory_utilization if gpu_memory_utilization is not None else config.GPU_MEMORY_UTILIZATION

    image_requests = image_requests or []
    pdf_requests = pdf_requests or []

    image_jobs: List[dict] = []
    pdf_contexts: List[dict] = []
    sampling_params = make_image_sampling_params()

    if not image_requests and not pdf_requests:
        return {"image_results": [], "pdf_results": []}

    engine, _ = get_engine(model_path, cuda_visible_devices, gpu_memory_utilization)

    def _request_iter():
        if image_requests:
            for spec in image_requests:
                spec_data = _clone_request(spec)
                crop_choice = spec_data.pop("crop_mode", default_image_crop)
                for job in stream_image_jobs(
                    [spec_data],
                    crop_mode=crop_choice,
                    sampling_params=sampling_params,
                ):
                    image_jobs.append(job)
                    yield (job["payload"], job["sampling_params"], job["handle_result"])

        if pdf_requests:
            for spec in pdf_requests:
                spec_data = _clone_request(spec)
                pdf_input = spec_data["input"]
                pdf_output = spec_data.get("output", config.OUTPUT_PATH)
                pdf_prompt = spec_data.get("prompt", config.PROMPT)
                pdf_crop = spec_data.get("crop_mode", default_pdf_crop)
                pdf_skip = spec_data.get("skip_repeat", default_skip_repeat)
                pdf_workers = spec_data.get("num_workers", default_num_workers)
                pdf_render_dpi = spec_data.get("pdf_render_dpi", spec_data.get("render_dpi", default_pdf_render_dpi))
                pdf_annot_dpi = spec_data.get("pdf_annot_dpi", spec_data.get("annot_dpi", default_pdf_annot_dpi))

                # pdf_workers is unused in streaming path (render is sequential), kept for API compatibility.
                _ = pdf_workers

                job_iter_fn, context, _pdf_sampling = stream_pdf_jobs(
                    input_path=pdf_input,
                    output_path=pdf_output,
                    prompt_text=pdf_prompt,
                    crop_mode=pdf_crop,
                    skip_repeat=pdf_skip,
                    num_workers=pdf_workers,
                    render_dpi=pdf_render_dpi,
                    annot_dpi=pdf_annot_dpi,
                )
                pdf_contexts.append(context)
                for job in job_iter_fn():
                    yield (job["payload"], job["sampling_params"], job["handle_result"])

    run_streaming_generation(
        engine=engine,
        request_iterable=_request_iter(),
        max_concurrency=effective_concurrency,
    )

    image_results = [job["result"] for job in image_jobs]

    pdf_results = []
    for context in pdf_contexts:
        mmd_path = finalize_pdf_outputs(context)
        base_name = context["base_name"]
        output_dir = context["output_path"]
        pdf_results.append(
            {
                "mmd_path": mmd_path,
                "mmd_det_path": os.path.join(output_dir, f"{base_name}_det.mmd"),
                "layouts_pdf_path": os.path.join(output_dir, f"{base_name}_layouts.pdf"),
            }
        )

    if not keep_model_loaded:
        unload_all_engines()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    return {
        "image_results": image_results,
        "pdf_results": pdf_results,
    }
