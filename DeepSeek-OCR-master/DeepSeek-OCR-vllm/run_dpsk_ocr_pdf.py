import argparse
import io
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List

import fitz
import img2pdf
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
from concurrency import run_concurrent_generation

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

def pil_to_pdf_img2pdf(pil_images, output_path):

    if not pil_images:
        return
    
    image_bytes_list = []
    
    for img in pil_images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=95)
        img_bytes = img_buffer.getvalue()
        image_bytes_list.append(img_bytes)
    
    try:
        pdf_bytes = img2pdf.convert(image_bytes_list)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    except Exception as e:
        print(f"error: {e}")



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
    
    for i, ref in enumerate(refs):
        try:
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result
                
                color = (np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 255))

                color_a = color + (20, )
                for points in points_list:
                    x1, y1, x2, y2 = points

                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)

                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

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
    return img_draw


def process_image_with_refs(image, ref_texts, jdx, output_dir):
    result_image = draw_bounding_boxes(image, ref_texts, jdx, output_dir)
    return result_image


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
) -> tuple[list[dict], dict]:
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

    print(f'{Colors.RED}PDF loading .....{Colors.RESET}')

    images = pdf_to_images_high_quality(input_path)
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
        "images": images,
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
    images = context["images"]
    page_outputs = context["page_outputs"]

    mmd_det_path = os.path.join(output_path, f'{base_name}_det.mmd')
    mmd_path = os.path.join(output_path, f'{base_name}.mmd')
    pdf_out_path = os.path.join(output_path, f'{base_name}_layouts.pdf')

    contents_det = ''
    contents = ''
    draw_images = []

    for jdx, (content, img) in enumerate(zip(page_outputs, images)):
        if content is None:
            continue

        if '<｜end▁of▁sentence｜>' in content:
            content = content.replace('<｜end▁of▁sentence｜>', '')
        else:
            if skip_repeat:
                continue

        page_num = '\n<--- Page Split --->'

        contents_det += content + f'\n{page_num}\n'

        image_draw = img.copy()

        matches_ref, matches_images, mathes_other = re_match(content)
        result_image = process_image_with_refs(image_draw, matches_ref, jdx, output_path)

        draw_images.append(result_image)

        for idx, a_match_image in enumerate(matches_images):
            content = content.replace(a_match_image, f'![](images/{jdx}_{idx}.jpg)\n')

        for a_match_other in mathes_other:
            content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')

        contents += content + f'\n{page_num}\n'

    with open(mmd_det_path, 'w', encoding='utf-8') as afile:
        afile.write(contents_det)

    with open(mmd_path, 'w', encoding='utf-8') as afile:
        afile.write(contents)

    pil_to_pdf_img2pdf(draw_images, pdf_out_path)
    return mmd_path


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

    jobs, context = prepare_pdf_jobs(
        input_path=input_path,
        output_path=output_path,
        prompt_text=prompt_text,
        crop_mode=crop_mode,
        skip_repeat=skip_repeat,
        num_workers=num_workers,
    )

    engine, _ = get_engine(model_path, cuda_visible_devices, gpu_memory_utilization)
    request_queue = [(job["payload"], job["sampling_params"]) for job in jobs]
    outputs_list = run_concurrent_generation(
        engine=engine,
        requests=request_queue,
        max_concurrency=max_concurrency,
    )

    for job, text in zip(jobs, outputs_list, strict=False):
        job["handle_result"](text)

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
    )


if __name__ == "__main__":
    main()
