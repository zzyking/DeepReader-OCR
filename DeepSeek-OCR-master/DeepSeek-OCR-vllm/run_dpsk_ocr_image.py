import argparse
import asyncio
import os
import re

import torch
if torch.version.cuda == '11.8':
    os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"

os.environ['VLLM_USE_V1'] = '0'

from vllm import SamplingParams
from vllm.model_executor.models.registry import ModelRegistry
import time
from deepseek_ocr import DeepseekOCRForCausalLM
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from tqdm import tqdm
from process.ngram_norepeat import NoRepeatNGramLogitsProcessor
from process.image_process import DeepseekOCRProcessor
from engine_cache import get_engine, unload_all_engines
from concurrency import run_concurrent_generation
import config
from vllm import AsyncLLMEngine



ModelRegistry.register_model("DeepseekOCRForCausalLM", DeepseekOCRForCausalLM)


def unload_image_engines() -> None:
    unload_all_engines()
    torch.cuda.empty_cache()


def make_sampling_params() -> SamplingParams:
    logits_processors = [
        NoRepeatNGramLogitsProcessor(ngram_size=30, window_size=90, whitelist_token_ids={128821, 128822})
    ]
    return SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        logits_processors=logits_processors,
        skip_special_tokens=False,
    )

def load_image(image_path):

    try:
        image = Image.open(image_path)
        
        corrected_image = ImageOps.exif_transpose(image)

        return corrected_image
        
    except Exception as e:
        print(f"error: {e}")
        try:
            return Image.open(image_path)
        except:
            return None


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
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR image pipeline without editing config.py")
    parser.add_argument("--mode", choices=sorted(config.MODES.keys()), default=None, help="Predefined vision mode to apply")
    parser.add_argument("--input", "-i", default=config.INPUT_PATH, help="Path to an image file to OCR")
    parser.add_argument("--output", "-o", default=config.OUTPUT_PATH, help="Directory where results will be saved")
    parser.add_argument("--prompt", "-p", default=None, help="Prompt sent to the model")
    parser.add_argument("--prompt-template", choices=sorted(config.PROMPT_TEMPLATES.keys()), default=None,
                        help="Select a predefined prompt template")
    parser.add_argument("--model-path", default=config.MODEL_PATH, help="Model identifier or local path")
    parser.add_argument("--crop-mode", type=str2bool, default=None, help="Enable cropping during preprocessing")
    parser.add_argument("--save-results", type=str2bool, default=True, help="Persist markdown and visualizations to disk")
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.getenv("DEEPREADER_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "4")),
        help="Comma-separated GPU device ids to expose to the runner",
    )
    parser.add_argument("--gpu-memory-util", type=float, default=None,
                        help="Fraction of GPU memory to reserve (default 0.8, requires ≈10GB free VRAM)")
    parser.add_argument("--keep-model-loaded", type=str2bool, default=False, help="Keep the image model in memory after the run")
    return parser.parse_args()


def draw_bounding_boxes(image, refs, output_dir):

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
                            cropped.save(os.path.join(output_dir, "images", f"{img_idx}.jpg"))
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


def process_image_with_refs(image, ref_texts, output_dir):
    result_image = draw_bounding_boxes(image, ref_texts, output_dir)
    return result_image


def _make_image_result_handler(job: dict):
    def _handle(text: str) -> None:
        job["result"] = text
        if job["save_results"] and "<image>" in job["prompt"]:
            print('=' * 15 + 'save results:' + '=' * 15)
            _write_results(text, job["image"], job["output_path"])

    return _handle


def prepare_image_jobs(
    request_specs: list[dict],
    crop_mode: bool,
    sampling_params: SamplingParams | None = None,
) -> list[dict]:
    sampling_params = sampling_params or make_sampling_params()
    jobs: list[dict] = []
    for spec in request_specs:
        input_path = resolve_path(spec["input"])
        output_path = resolve_path(spec.get("output", config.OUTPUT_PATH))
        prompt_text = spec.get("prompt") or config.PROMPT
        save_results = bool(spec.get("save_results", True))

        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "images"), exist_ok=True)

        image = load_image(input_path)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {input_path}")
        image = image.convert("RGB")

        payload = {"prompt": prompt_text}
        if "<image>" in prompt_text:
            image_features = DeepseekOCRProcessor().tokenize_with_images(
                images=[image], bos=True, eos=True, cropping=crop_mode, prompt=prompt_text
            )
            payload["multi_modal_data"] = {"image": image_features}

        job: dict = {
            "type": "image",
            "input_path": input_path,
            "output_path": output_path,
            "prompt": prompt_text,
            "save_results": save_results,
            "image": image,
            "payload": payload,
            "sampling_params": sampling_params,
            "result": None,
            "handle_result": None,
        }
        job["handle_result"] = _make_image_result_handler(job)
        jobs.append(job)
    return jobs


def run_image_batch(
    batch_requests: list[dict],
    model_path: str | None = None,
    crop_mode: bool | None = None,
    cuda_visible_devices: str | None = None,
    gpu_memory_utilization: float | None = None,
    max_concurrency: int | None = None,
    keep_model_loaded: bool = True,
) -> list[str]:
    if not batch_requests:
        return []

    model_path = model_path or config.MODEL_PATH
    if crop_mode is None:
        crop_mode = config.CROP_MODE

    sampling_params = make_sampling_params()
    jobs = prepare_image_jobs(batch_requests, crop_mode, sampling_params=sampling_params)

    engine, _ = get_engine(model_path, cuda_visible_devices, gpu_memory_utilization)
    request_queue = [(job["payload"], job["sampling_params"]) for job in jobs]
    outputs = run_concurrent_generation(
        engine=engine,
        requests=request_queue,
        max_concurrency=max_concurrency,
    )

    for job, text in zip(jobs, outputs, strict=False):
        job["handle_result"](text)

    if not keep_model_loaded:
        unload_image_engines()

    return [job["result"] for job in jobs]


async def stream_generate(engine: AsyncLLMEngine, sampling_params: SamplingParams, image=None, prompt=''):
    request_id = f"request-{int(time.time())}"

    printed_length = 0
    final_output = ""

    if image and '<image>' in prompt:
        request = {
            "prompt": prompt,
            "multi_modal_data": {"image": image}
        }
    elif prompt:
        request = {"prompt": prompt}
    else:
        raise ValueError('prompt is none!!!')

    async for request_output in engine.generate(request, sampling_params, request_id):
        if request_output.outputs:
            full_text = request_output.outputs[0].text
            new_text = full_text[printed_length:]
            print(new_text, end='', flush=True)
            printed_length = len(full_text)
            final_output = full_text
    print('\n')

    return final_output


def _write_results(outputs: str, image, output_path: str) -> None:
    image_draw = image.copy()

    with open(os.path.join(output_path, 'result_ori.mmd'), 'w', encoding='utf-8') as afile:
        afile.write(outputs)

    matches_ref, matches_images, mathes_other = re_match(outputs)
    result = process_image_with_refs(image_draw, matches_ref, output_path)

    for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
        outputs = outputs.replace(a_match_image, f'![](images/{idx}.jpg)\n')

    for idx, a_match_other in enumerate(tqdm(mathes_other, desc="other")):
        outputs = outputs.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:')

    with open(os.path.join(output_path, 'result.mmd'), 'w', encoding='utf-8') as afile:
        afile.write(outputs)

    if 'line_type' in outputs:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        lines = eval(outputs)['Line']['line']
        line_type = eval(outputs)['Line']['line_type']
        endpoints = eval(outputs)['Line']['line_endpoint']

        fig, ax = plt.subplots(figsize=(3, 3), dpi=200)
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)

        for idx, line in enumerate(lines):
            try:
                p0 = eval(line.split(' -- ')[0])
                p1 = eval(line.split(' -- ')[-1])

                ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linewidth=0.8, color='k')
                ax.scatter(p0[0], p0[1], s=5, color='k')
                ax.scatter(p1[0], p1[1], s=5, color='k')
            except Exception:
                pass

        for endpoint in endpoints:
            try:
                label = endpoint.split(': ')[0]
                (x, y) = eval(endpoint.split(': ')[1])
                ax.annotate(label, (x, y), xytext=(1, 1), textcoords='offset points', fontsize=5, fontweight='light')
            except Exception:
                pass

        try:
            if 'Circle' in eval(outputs).keys():
                circle_centers = eval(outputs)['Circle']['circle_center']
                radius = eval(outputs)['Circle']['radius']

                for center, r in zip(circle_centers, radius):
                    center = eval(center.split(': ')[1])
                    circle = Circle(center, radius=r, fill=False, edgecolor='black', linewidth=0.8)
                    ax.add_patch(circle)
        except Exception:
            pass

        plt.savefig(os.path.join(output_path, 'geo.jpg'))
        plt.close()

    result.save(os.path.join(output_path, 'result_with_boxes.jpg'))


def run_image_pipeline(
    input_path: str,
    output_path: str,
    prompt: str,
    model_path: str | None = None,
    crop_mode: bool | None = None,
    save_results: bool = True,
    cuda_visible_devices: str | None = None,
    gpu_memory_utilization: float | None = None,
    keep_model_loaded: bool = True,
) -> str:
    model_path = model_path or config.MODEL_PATH
    if crop_mode is None:
        crop_mode = config.CROP_MODE

    sampling_params = make_sampling_params()
    job = prepare_image_jobs(
        [
            {
                "input": input_path,
                "output": output_path,
                "prompt": prompt,
                "save_results": save_results,
            }
        ],
        crop_mode=crop_mode,
        sampling_params=sampling_params,
    )[0]

    engine, _ = get_engine(model_path, cuda_visible_devices, gpu_memory_utilization)
    image_features = job["payload"].get("multi_modal_data", {}).get("image")
    result_out = asyncio.run(stream_generate(engine, sampling_params, image_features, job["prompt"]))

    job["handle_result"](result_out)

    if not keep_model_loaded:
        unload_image_engines()

    return job["result"]


def main():
    args = parse_args()

    if args.mode:
        config.set_mode(args.mode)

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)

    if args.prompt_template:
        config.set_prompt_template(args.prompt_template)

    prompt = args.prompt if args.prompt is not None else config.PROMPT # use prompt from args if provided, otherwise use prompt template
    model_path = args.model_path
    crop_mode = config.CROP_MODE if args.crop_mode is None else args.crop_mode
    save_results = args.save_results
    cuda_visible_devices = args.cuda_visible_devices
    gpu_memory_util = args.gpu_memory_util if args.gpu_memory_util is not None else config.GPU_MEMORY_UTILIZATION
    keep_model_loaded = args.keep_model_loaded

    run_image_pipeline(
        input_path=input_path,
        output_path=output_path,
        prompt=prompt,
        model_path=model_path,
        crop_mode=crop_mode,
        save_results=save_results,
        cuda_visible_devices=cuda_visible_devices,
        gpu_memory_utilization=gpu_memory_util,
        keep_model_loaded=keep_model_loaded,
    )


if __name__ == "__main__":
    main()
