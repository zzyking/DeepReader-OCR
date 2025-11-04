# DeepReader

DeepReader is an agentic reading toolkit that couples DeepSeek-OCR with opinionated defaults for running single-document or batch OCR. It streamlines image/PDF ingestion, produces Markdown accompanied by figure crops and layout previews, and exposes knobs for both CLI and Gradio workflows.

***
## Project Layout

- `images/`: Sample page images for quick smoke-tests.
- `docs/`: Input PDFs for full-length papers.
- `outputs/`: Generated Markdown, annotated images, and layout PDFs.
- `DeepSeek-OCR-master/DeepSeek-OCR-vllm/`: vLLM-powered runtime (default entry points).

***
## Environment Setup

```bash
conda create -n deepreader python=3.12.9 -y
conda activate deepreader
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu118
pip install vllm-0.8.5+cu118-cp38-abi3-manylinux1_x86_64.whl
pip install -r requirements.txt
```

Optional extras:

- `pip install flash-attn==2.7.3 --no-build-isolation` (faster attention if supported).
- `export TMPDIR=$PWD/outputs/tmp` when `/tmp` isn’t writable.

***
## Configuration Strategy

`DeepSeek-OCR-vllm/config.py` reads all defaults from environment variables, making it easy to swap inputs, outputs, prompts, or GPU settings without editing code.

```bash
export DEEPREADER_INPUT_PATH="$PWD/docs/paper.pdf"
export DEEPREADER_OUTPUT_PATH="$PWD/outputs/paper_run"
export DEEPREADER_PROMPT='<image>
<|grounding|>Convert the document to markdown.'
export DEEPREADER_PROMPT_TEMPLATE=document
export DEEPREADER_MODE=gundam
export DEEPREADER_CUDA_VISIBLE_DEVICES=0
export DEEPREADER_GPU_MEM_UTIL=0.8
```

>**GPU tip**: the default vLLM config assumes ≈10 GB of free VRAM. Tune `DEEPREADER_GPU_MEM_UTIL` down if you’re memory-constrained.

***

## Image OCR Pipeline (CLI)

```bash
python DeepSeek-OCR-master/DeepSeek-OCR-vllm/run_dpsk_ocr_image.py \
  --input ./images/sample.png \
  --output ./outputs/sample_image_run \
  --mode gundam \
  --prompt-template figures \
  --prompt '<image>\n<|grounding|>Convert the document to markdown.' \
  --gpu-memory-util 0.8 \
  --cuda-visible-devices 0 \
  --keep-model-loaded true
```

Outputs:

- `result.mmd`: Markdown transcription with resolved image crops.
- `result_with_boxes.jpg`: Annotated source page.
- `images/*.jpg`: Extracted figure crops.
- `geo.jpg` (conditional): Geometry reconstruction if the model returns structured line data.

***
## PDF OCR Pipeline (CLI)

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm
python run_dpsk_ocr_pdf.py \
  --input ../../docs/paper.pdf \
  --output ../../outputs/paper_pdf_run \
  --mode gundam \
  --prompt-template document \
  --gpu-memory-util 0.8 \
  --cuda-visible-devices 0 \
  --keep-model-loaded true
```

You'll get:

- `<paper>_det.mmd`: Raw model output with detection tags.
- `<paper>.mmd`: Cleaned Markdown with inline image references.
- `<paper>_layouts.pdf`: Stitch of annotated page previews.
- `images/`: page-level crops.

Disable repeat filtering with `--skip-repeat false` if you need every page’s raw output.

****
## Gradio Interface

Launch an interactive UI that accepts image or PDF uploads and returns a zipped bundle (Markdown, annotated layouts, figure crops):

```bash
python run_deepreader_gradio.py --host 0.0.0.0 --port 10086 --no-browser
```

Features:

- Upload image/PDF from the left column, pick a vision mode (`base` or `gundam (hi-res)`), and select a prompt template.
- Adjust GPU memory utilisation (default 0.8 ≈ 80% usage) and device ID.
- Toggle “Keep models loaded” to reuse the engine between runs; click “Unload Models” to free VRAM.
- Gradio keeps only the latest 20 sessions and drops anything older than 24 hours, so `outputs/gradio_sessions/` stays tidy.
- Each run produces a ZIP bundle (`result.mmd`, figures, annotated layouts, logs) ready for download.

Controls let you override the prompt, crop mode, CUDA device list, GPU memory utilisation, and PDF-specific knobs (`max concurrency`, `num workers`, `skip repeat`). The “Keep models loaded” toggle reuses the same vLLM weights across submissions; click “Unload Models” to free GPU memory manually. Each run creates a session under `outputs/gradio_sessions/`, and the downloaded archive also includes a `gradio_run.log` with console output. Use the prompt-template dropdown to populate the textbox with a preset, adjust the GPU memory slider (default 0.8 ≈ 80%), and feel free to tweak the textbox before running. Use `--port` to pick a different port, `--share` for public links, `--queue` to enable Gradio’s request queue, and `--allow-path <dir>` to expose extra directories for downloads if needed.

>**Note**: plan for ≈10 GB of free VRAM for the default gundam (hi-res) mode. Lower the slider/flag if your GPU has less headroom.

***
### Vision Modes


| Mode | Base Size | Image Size |	Crop Mode |	Notes |
|-|-|-|-|-|
| `base` |	1024	| 1024 |	False	| Standard quality |
| `gundam (hi-res)` |	1024 |	640	| True |	Dynamic high-res crops (default) |

Switch via `--mode` (CLI), the Gradio dropdown, or `DEEPREADER_MODE`.

***
### Prompt Templates

Named templates keep prompts consistent across runs:

- `document`: `<image>\n<|grounding|>Convert the document to markdown.`
- `other_image`: `<image>\n<|grounding|>OCR this image.`
- `without_layouts`: `<image>\nFree OCR.`
- `figures`: `<image>\nParse the figure.`
- `general`: `<image>\nDescribe this image in detail.`
- `rec`: `<image>\nLocate <|ref|>xxxx<|/ref|> in the image.`

Select them via the template dropdown/CLI flag/env var, or exceed with a custom `--prompt`.

***
## Changelog

| Version | Date       | Highlights |
|---------|------------|------------|
| 0.1.0   | 2025-11-04 | Initial public drop: shared vLLM engine cache, mode/prompt presets, GPU memory control, refreshed Gradio UI, session auto-cleanup |

***
## Troubleshooting

- **GPU capability errors**: ensure `CUDA_DEVICE_ORDER=PCI_BUS_ID` (automatically set) and specify a single `--cuda-visible-devices` index.
- **Low VRAM**: reduce `--gpu-memory-util` (e.g., 0.6) or use `base` mode.
- **Missing `/tmp` write access**: `export TMPDIR=$PWD/outputs/tmp`.
- **NVML InvalidArgument**: usually indicates pointing to a non-existent GPU index; double-check the selected CUDA device.

***
## License & Contributions
This repo mirrors [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) OSS code plus integration glue. Follow upstream licensing for model usage. Contributions welcome—please use Conventional Commit messages (`feat: ...`, `fix: ...`) and include CLI examples/VRAM notes in PR descriptions.
