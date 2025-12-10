import os


MODES = {
    "tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

PROMPT_TEMPLATES = {
    "document": '<image>\n<|grounding|>Convert the document to markdown.',
    "other_image": '<image>\n<|grounding|>OCR this image.',
    "without_layouts": '<image>\nFree OCR.',
    "figures": '<image>\nParse the figure.',
    "general": '<image>\nDescribe this image in detail.',
    "rec": '<image>\nLocate <|ref|>xxxx<|/ref|> in the image.',
}


def _parse_bool_env(var_name: str, default: bool) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return default
    if value.lower() in {"1", "true", "yes", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int_env(var_name: str, default: int) -> int:
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


DEFAULT_MODE = os.getenv("DEEPREADER_MODE", "gundam").lower()
if DEFAULT_MODE not in MODES:
    DEFAULT_MODE = "gundam"

_mode_defaults = MODES[DEFAULT_MODE]

BASE_SIZE = _parse_int_env("DEEPREADER_BASE_SIZE", _mode_defaults["base_size"])
IMAGE_SIZE = _parse_int_env("DEEPREADER_IMAGE_SIZE", _mode_defaults["image_size"])
CROP_MODE = _parse_bool_env("DEEPREADER_CROP_MODE", _mode_defaults["crop_mode"])
MIN_CROPS = _parse_int_env("DEEPREADER_MIN_CROPS", 2)
MAX_CROPS = _parse_int_env("DEEPREADER_MAX_CROPS", 6)  # max:9; reduce to 6 if GPU memory is small.
MAX_CONCURRENCY = _parse_int_env("DEEPREADER_MAX_CONCURRENCY", 100)  # Lower if GPU memory is limited.
NUM_WORKERS = _parse_int_env("DEEPREADER_NUM_WORKERS", 4)  # image pre-process (resize/padding) workers
PRINT_NUM_VIS_TOKENS = _parse_bool_env("DEEPREADER_PRINT_NUM_VIS_TOKENS", False)
SKIP_REPEAT = _parse_bool_env("DEEPREADER_SKIP_REPEAT", True)
MODEL_PATH = os.getenv("DEEPREADER_MODEL_PATH", "deepseek-ai/DeepSeek-OCR")  # change to your model path
GPU_MEMORY_UTILIZATION = float(os.getenv("DEEPREADER_GPU_MEM_UTIL", "0.8"))
KEEP_MODELS_LOADED = _parse_bool_env("DEEPREADER_KEEP_MODELS_LOADED", True)
PDF_RENDER_DPI = _parse_int_env("DEEPREADER_PDF_RENDER_DPI", 144)
PDF_ANNOT_DPI = _parse_int_env("DEEPREADER_PDF_ANNOT_DPI", max(PDF_RENDER_DPI, 432))
ACTIVE_MODE = DEFAULT_MODE

# TODO: change INPUT_PATH
# .pdf: run_dpsk_ocr_pdf.py;
# .jpg, .png, .jpeg: run_dpsk_ocr_image.py;
# Omnidocbench images path: run_dpsk_ocr_eval_batch.py

INPUT_PATH = os.getenv("DEEPREADER_INPUT_PATH", "/home/zangzeyuan/DeepReader/docs/qwen2vl.pdf")
OUTPUT_PATH = os.getenv("DEEPREADER_OUTPUT_PATH", "/home/zangzeyuan/DeepReader/outputs/qwen2vl.pdf")

DEFAULT_TEMPLATE = os.getenv("DEEPREADER_PROMPT_TEMPLATE", "document").lower()
if DEFAULT_TEMPLATE not in PROMPT_TEMPLATES:
    DEFAULT_TEMPLATE = "document"

PROMPT = os.getenv("DEEPREADER_PROMPT", PROMPT_TEMPLATES[DEFAULT_TEMPLATE])


def set_mode(mode: str) -> None:
    """Update runtime mode settings."""
    mode_key = mode.lower()
    if mode_key not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Available modes: {', '.join(sorted(MODES))}")

    values = MODES[mode_key]

    global BASE_SIZE, IMAGE_SIZE, CROP_MODE, ACTIVE_MODE
    BASE_SIZE = values["base_size"]
    IMAGE_SIZE = values["image_size"]
    CROP_MODE = values["crop_mode"]
    ACTIVE_MODE = mode_key

    os.environ["DEEPREADER_MODE"] = mode_key


def get_mode_settings(mode: str | None = None) -> dict:
    mode_key = (mode or ACTIVE_MODE).lower()
    if mode_key not in MODES:
        raise ValueError(f"Unknown mode '{mode}'. Available modes: {', '.join(sorted(MODES))}")
    return MODES[mode_key].copy()


def set_prompt_template(template: str) -> None:
    template_key = template.lower()
    if template_key not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template '{template}'. Available templates: {', '.join(sorted(PROMPT_TEMPLATES))}")

    prompt_value = PROMPT_TEMPLATES[template_key]

    global PROMPT, DEFAULT_TEMPLATE
    PROMPT = prompt_value
    DEFAULT_TEMPLATE = template_key

    os.environ["DEEPREADER_PROMPT_TEMPLATE"] = template_key


def get_prompt_template(template: str | None = None) -> str:
    template_key = (template or DEFAULT_TEMPLATE).lower()
    if template_key not in PROMPT_TEMPLATES:
        raise ValueError(f"Unknown prompt template '{template}'. Available templates: {', '.join(sorted(PROMPT_TEMPLATES))}")
    return PROMPT_TEMPLATES[template_key]


from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
