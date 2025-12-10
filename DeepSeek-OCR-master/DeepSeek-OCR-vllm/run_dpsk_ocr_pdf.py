import argparse
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path
import fitz
import torch
from tqdm import tqdm

if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["VLLM_LOG_LEVEL"] = "debug"

import config

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from deepseek_ocr import DeepseekOCRForCausalLM

from vllm.model_executor.models.registry import ModelRegistry

from vllm import SamplingParams
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from engine_cache import get_engine, unload_all_engines
from concurrency import run_concurrent_generation, run_streaming_generation

# Prevent duplicate registration when modules are imported in multiple processes.
_registry_map = getattr(ModelRegistry, "_model_name_to_cls", None)
if not (isinstance(_registry_map, dict) and "DeepseekOCRForCausalLM" in _registry_map):
    ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def resolve_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unexpected boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR PDF pipeline without editing config.py")
    parser.add_argument("--mode", choices=sorted(config.MODES.keys()), default=None, help="Predefined vision mode to apply")
    parser.add_argument("--input", "-i", default=config.INPUT_PATH, help="Path to a PDF file to OCR")
    parser.add_argument("--output", "-o", default=config.OUTPUT_PATH, help="Directory where results will be saved")
    parser.add_argument("--prompt", "-p", default=None, help="Prompt sent to the model")
    parser.add_argument("--prompt-template", choices=sorted(config.PROMPT_TEMPLATES.keys()), default=None,
                        help="Select a predefined prompt template")
    parser.add_argument("--model-path", default=config.MODEL_PATH, help="Model identifier or local path")
    parser.add_argument("--crop-mode", type=str2bool, default=None, help="Enable cropping during preprocessing")
    parser.add_argument("--skip-repeat", type=str2bool, default=config.SKIP_REPEAT, help="Skip pages if EOS token missing")
    parser.add_argument("--max-concurrency", type=int, default=config.MAX_CONCURRENCY, help="Maximum concurrent sequences for vLLM")
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS, help="Thread pool workers for preprocessing")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0")),
        help="Comma-separated GPU device ids to expose to the runner",
    )
    parser.add_argument("--gpu-memory-util", type=float, default=None,
                        help="Fraction of GPU memory to reserve (default 0.8, requires ≈10GB free VRAM)")
    parser.add_argument("--keep-model-loaded", type=str2bool, default=False, help="Keep the PDF model in memory after the run")
    parser.add_argument("--pdf-render-dpi", type=int, default=config.PDF_RENDER_DPI,
                        help="DPI used to rasterize each PDF page for model input (affects VRAM usage)")
    parser.add_argument("--pdf-annot-dpi", type=int, default=config.PDF_ANNOT_DPI,
                        help="DPI used to rasterize PDF pages for bounding boxes and crops (<=0 reuses render DPI)")
    return parser.parse_args()


def unload_pdf_models() -> None:
    unload_all_engines()
    torch.cuda.empty_cache()


def make_pdf_sampling_params() -> SamplingParams:
    logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids={128821, 128822})]
    return SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
    )


class Colors:
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    RESET = '\033[0m' 

def pdf_to_images_high_quality(pdf_path, dpi=144, image_format="PNG"):
    """
    pdf2images
    """
    images = []
    
    pdf_document = fitz.open(pdf_path)
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]

        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        Image.MAX_IMAGE_PIXELS = None

        if image_format.upper() == "PNG":
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
        else:
            img_data = pixmap.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
        
        images.append(img)
    
    pdf_document.close()
    return images


def _render_pdf_page(pdf_document: fitz.Document, page_index: int, dpi: int) -> Image.Image:
    """Render a single PDF page to a PIL image."""
    page = pdf_document[page_index]
    zoom = max(dpi, 72) / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    Image.MAX_IMAGE_PIXELS = None
    img_data = pixmap.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    return img

def _color_from_indices(page_index: int, box_index: int) -> tuple[float, float, float]:
    seed = (page_index + 1) * 7919 + (box_index + 1) * 104729
    rng = np.random.default_rng(seed)
    base = rng.random(3) * 0.6 + 0.2
    return tuple(float(val) for val in base)


def annotate_pdf_with_boxes(
    pdf_path: str,
    output_path: str,
    page_annotations: list[list[tuple[str, tuple[float, float, float, float]]]],
    *,
    font_name: str = "helv",
    font_size: float = 5.0,
    label_padding: float = 2.0,
):
    if not page_annotations:
        return

    pdf_document = fitz.open(pdf_path)

    for page_index in range(pdf_document.page_count):
        page = pdf_document.load_page(page_index)
        annotations = page_annotations[page_index] if page_index < len(page_annotations) else []
        if not annotations:
            continue

        page_width = page.rect.width
        page_height = page.rect.height

        for box_index, (label_type, norm_rect) in enumerate(annotations):
            x1_norm, y1_norm, x2_norm, y2_norm = norm_rect
            rect = fitz.Rect(
                (x1_norm / 999.0) * page_width,
                (y1_norm / 999.0) * page_height,
                (x2_norm / 999.0) * page_width,
                (y2_norm / 999.0) * page_height,
            )

            color = _color_from_indices(page_index, box_index)
            page.draw_rect(rect, color=color, width=1.2, fill=color, fill_opacity=0.1, overlay=True)

            if not label_type:
                continue

            try:
                text_width = page.get_text_length(label_type, fontname=font_name, fontsize=font_size)
            except Exception:
                text_width = len(label_type) * font_size * 0.6

            text_height = font_size
            label_top = max(page.rect.y0, rect.y0 - text_height - label_padding)
            text_position = (
                rect.x0 + label_padding,
                label_top + text_height,
            )
            page.insert_text(
                text_position,
                label_type,
                fontsize=font_size,
                color=color,
                fontname=font_name,
                overlay=True,
            )

    pdf_document.save(output_path, garbage=4, deflate=True)
    pdf_document.close()



def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):


    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, jdx, output_dir):

    image_width, image_height = image.size
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    overlay = Image.new('RGBA', img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)
    
    #     except IOError:
    font = ImageFont.load_default()

    img_idx = 0
    vector_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1_norm, y1_norm, x2_norm, y2_norm = points
                    vector_boxes.append((label_type, (x1_norm, y1_norm, x2_norm, y2_norm)))

                    x1 = int(x1_norm / 999 * image_width)
                    y1 = int(y1_norm / 999 * image_height)

                    x2 = int(x2_norm / 999 * image_width)
                    y2 = int(y2_norm / 999 * image_height)

                    if label_type == 'image':
                        try:
                            cropped = image.crop((x1, y1, x2, y2))
                            cropped.save(os.path.join(output_dir, "images", f"{jdx}_{img_idx}.jpg"))
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1
                        
                    try:
                        if label_type == 'title':
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)
                        else:
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle([x1, y1, x2, y2], fill=color_a, outline=(0, 0, 0, 0), width=1)

                        text_x = x1
                        text_y = max(0, y1 - 15)
                            
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], 
                                    fill=(255, 255, 255, 30))
                        
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw, vector_boxes


def process_image_with_refs(image, ref_texts, jdx, output_dir):
    result_image, vector_boxes = draw_bounding_boxes(image, ref_texts, jdx, output_dir)
    return result_image, vector_boxes


def process_single_image(image, prompt_text, crop_mode):
    """single image"""
    cache_item = {"prompt": prompt_text}

    if '<image>' in prompt_text:
        cache_item["multi_modal_data"] = {
            "image": DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=crop_mode, prompt=prompt_text
            )
        }
    return cache_item


def _make_pdf_result_handler(context: dict, page_index: int):
    def _handle(text: str) -> None:
        context["page_outputs"][page_index] = text

    return _handle


def prepare_pdf_jobs(
    input_path: str,
    output_path: str,
    prompt_text: str,
    crop_mode: bool,
    skip_repeat: bool,
    num_workers: int,
    render_dpi: int,
    annot_dpi: int | None,
) -> tuple[list[dict], dict]:
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

    print(f'{Colors.RED}PDF loading .....{Colors.RESET}')

    images = pdf_to_images_high_quality(input_path, dpi=max(render_dpi, 72))
    if annot_dpi is None or annot_dpi <= 0 or annot_dpi == render_dpi:
        annot_images = images
    else:
        annot_images = pdf_to_images_high_quality(input_path, dpi=max(annot_dpi, 72))
    if len(annot_images) != len(images):
        raise RuntimeError("Annotation images and render images differ in length; ensure PDF pages render consistently.")
    sampling_params = make_pdf_sampling_params()

    process_fn = partial(process_single_image, prompt_text=prompt_text, crop_mode=crop_mode)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        batch_inputs = list(
            tqdm(
                executor.map(process_fn, images),
                total=len(images),
                desc="Pre-processed images"
            )
        )

    for item in batch_inputs:
        prompt_value = item.get("prompt", "")
        if "<image>" not in prompt_value:
            item.pop("multi_modal_data", None)

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    context = {
        "input_path": input_path,
        "output_path": output_path,
        "skip_repeat": skip_repeat,
        "annot_images": annot_images,
        "base_name": base_name,
        "page_outputs": [None] * len(batch_inputs),
    }

    jobs: list[dict] = []
    for page_index, payload in enumerate(batch_inputs):
        job = {
            "type": "pdf",
            "page_index": page_index,
            "payload": payload,
            "sampling_params": sampling_params,
            "handle_result": _make_pdf_result_handler(context, page_index),
        }
        jobs.append(job)

    return jobs, context


def finalize_pdf_outputs(context: dict) -> str:
    output_path = context["output_path"]
    base_name = context["base_name"]
    skip_repeat = context["skip_repeat"]
    page_outputs = context["page_outputs"]
    images = context["annot_images"]
    mmd_det_path = os.path.join(output_path, f'{base_name}_det.mmd')
    mmd_path = os.path.join(output_path, f'{base_name}.mmd')
    pdf_out_path = os.path.join(output_path, f'{base_name}_layouts.pdf')

    contents_det = ''
    contents = ''
    page_annotations: list[list[tuple[str, tuple[float, float, float, float]]]] = [list() for _ in images]

    def _process_page(args: tuple[int, str, Image.Image]) -> tuple[int, str, str, list[tuple[str, tuple[float, float, float, float]]]]:
        page_idx, raw_content, img = args
        content = raw_content

        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        else:
            if skip_repeat:
                return page_idx, '', '', []

        page_num = '\n<--- Page Split --->'
        image_draw = img.copy()

        matches_ref, matches_images, mathes_other = re_match(content)
        _, vector_boxes = process_image_with_refs(image_draw, matches_ref, page_idx, output_path)

        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/{page_idx}_{idx}.jpg)\n')

        for a_match_other in mathes_other:
            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

        return page_idx, raw_content + f'\n{page_num}\n', content + f'\n{page_num}\n', vector_boxes

    work_items = [(idx, content, img) for idx, (content, img) in enumerate(zip(page_outputs, images)) if content is not None]
    if work_items:
        max_workers = max(1, min(len(work_items), os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for page_idx, det_fragment, content_fragment, vectors in executor.map(_process_page, work_items):
                if det_fragment:
                    contents_det += det_fragment
                if content_fragment:
                    contents += content_fragment
                page_annotations[page_idx] = vectors

    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    annotate_pdf_with_boxes(context["input_path"], pdf_out_path, page_annotations)
    return mmd_path

def stream_pdf_jobs(
    input_path: str,
    output_path: str,
    prompt_text: str,
    crop_mode: bool,
    skip_repeat: bool,
    num_workers: int,
    render_dpi: int,
    annot_dpi: int | None,
) -> tuple[callable, dict, SamplingParams]:
    """
    Generator-friendly variant of ``prepare_pdf_jobs`` that yields jobs one by one
    so vLLM can start generating while preprocessing continues.
    """
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

    pdf_bytes = Path(input_path).read_bytes()
    with fitz.open(stream=pdf_bytes, filetype="pdf") as _doc_probe:
        page_count = _doc_probe.page_count
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    context = {
        "input_path": input_path,
        "output_path": output_path,
        "skip_repeat": skip_repeat,
        "annot_images": [None] * page_count,
        "base_name": base_name,
        "page_outputs": [None] * page_count,
    }

    sampling_params = make_pdf_sampling_params()

    def _render_and_prepare(page_index: int) -> tuple[int, dict, Image.Image]:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            render_img = _render_pdf_page(doc, page_index, render_dpi)
            if annot_dpi is None or annot_dpi <= 0 or annot_dpi == render_dpi:
                annot_img = render_img
            else:
                annot_img = _render_pdf_page(doc, page_index, annot_dpi)

        payload = process_single_image(render_img, prompt_text=prompt_text, crop_mode=crop_mode)
        if "<image>" not in payload.get("prompt", ""):
            payload.pop("multi_modal_data", None)

        return page_index, payload, annot_img

    def _job_iter():
        max_workers = max(1, min(num_workers, page_count))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_render_and_prepare, idx) for idx in range(page_count)]
            for fut in as_completed(futures):
                page_index, payload, annot_img = fut.result()
                context["annot_images"][page_index] = annot_img
                yield {
                    "type": "pdf",
                    "page_index": page_index,
                    "payload": payload,
                    "sampling_params": sampling_params,
                    "handle_result": _make_pdf_result_handler(context, page_index),
                }

    return _job_iter, context, sampling_params


def run_pdf_pipeline(
    input_path: str,
    output_path: str,
    prompt_text: str,
    model_path: str | None = None,
    crop_mode: bool | None = None,
    skip_repeat: bool | None = None,
    max_concurrency: int | None = None,
    num_workers: int | None = None,
    cuda_visible_devices: str | None = None,
    gpu_memory_utilization: float | None = None,
    keep_model_loaded: bool = True,
    pdf_render_dpi: int | None = None,
    pdf_annot_dpi: int | None = None,
) -> str:
    model_path = model_path or config.MODEL_PATH
    if crop_mode is None:
        crop_mode = config.CROP_MODE
    if skip_repeat is None:
        skip_repeat = config.SKIP_REPEAT
    if max_concurrency is None:
        max_concurrency = config.MAX_CONCURRENCY
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    if pdf_render_dpi is None:
        pdf_render_dpi = config.PDF_RENDER_DPI
    if pdf_annot_dpi is None:
        pdf_annot_dpi = config.PDF_ANNOT_DPI

    engine, _ = get_engine(model_path, cuda_visible_devices, gpu_memory_utilization)
    job_iter_fn, context, sampling_params = stream_pdf_jobs(
        input_path=input_path,
        output_path=output_path,
        prompt_text=prompt_text,
        crop_mode=crop_mode,
        skip_repeat=skip_repeat,
        num_workers=num_workers,
        render_dpi=pdf_render_dpi,
        annot_dpi=pdf_annot_dpi,
    )

    def _request_iter():
        for job in job_iter_fn():
            yield (job["payload"], job["sampling_params"], job["handle_result"])

    run_streaming_generation(
        engine=engine,
        request_iterable=_request_iter(),
        max_concurrency=max_concurrency,
    )

    mmd_path = finalize_pdf_outputs(context)

    if not keep_model_loaded:
        unload_pdf_models()

    return mmd_path

def main():
    args = parse_args()

    if args.mode:
        config.set_mode(args.mode)

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    if args.prompt_template:
        config.set_prompt_template(args.prompt_template)

    prompt_text = args.prompt if args.prompt is not None else config.PROMPT # use prompt from args if provided, otherwise use prompt template
    model_path = args.model_path
    crop_mode = config.CROP_MODE if args.crop_mode is None else args.crop_mode
    skip_repeat = args.skip_repeat
    max_concurrency = args.max_concurrency
    num_workers = args.num_workers
    cuda_visible_devices = args.cuda_visible_devices
    gpu_memory_util = args.gpu_memory_util if args.gpu_memory_util is not None else config.GPU_MEMORY_UTILIZATION
    keep_model_loaded = args.keep_model_loaded
    pdf_render_dpi = args.pdf_render_dpi if args.pdf_render_dpi is not None else config.PDF_RENDER_DPI
    pdf_annot_dpi = args.pdf_annot_dpi if args.pdf_annot_dpi is not None else config.PDF_ANNOT_DPI

    run_pdf_pipeline(
        input_path=input_path,
        output_path=output_path,
        prompt_text=prompt_text,
        model_path=model_path,
        crop_mode=crop_mode,
        skip_repeat=skip_repeat,
        max_concurrency=max_concurrency,
        num_workers=num_workers,
        cuda_visible_devices=cuda_visible_devices,
        gpu_memory_utilization=gpu_memory_util,
        keep_model_loaded=keep_model_loaded,
        pdf_render_dpi=pdf_render_dpi,
        pdf_annot_dpi=pdf_annot_dpi,
    )


if __name__ == "__main__":
    main()
