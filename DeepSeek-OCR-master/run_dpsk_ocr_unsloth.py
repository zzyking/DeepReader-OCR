"""CLI entrypoint for Unsloth-based DeepSeek-OCR (image/PDF)."""
import os
import sys
from pathlib import Path

# # Ensure env is set before importing unsloth-heavy modules
# if "DEEPREADER_CUDA_VISIBLE_DEVICES" in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["DEEPREADER_CUDA_VISIBLE_DEVICES"]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepseek_ocr_unsloth.main import main


if __name__ == "__main__":
    main()
