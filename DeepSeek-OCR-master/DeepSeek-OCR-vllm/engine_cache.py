import os
from typing import Dict, Optional, Tuple

from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs

import config


_ENGINE_CACHE: Dict[Tuple[str, str, float], AsyncLLMEngine] = {}

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _normalize_cuda(cuda_visible_devices: Optional[str]) -> str:
    if cuda_visible_devices and cuda_visible_devices.strip():
        first_device = cuda_visible_devices.split(",")[0].strip()
        return first_device
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing:
        return existing.split(",")[0].strip()
    return "0"


def get_engine(
    model_path: str,
    cuda_visible_devices: Optional[str] = None,
    gpu_memory_utilization: Optional[float] = None,
) -> Tuple[AsyncLLMEngine, str]:
    cuda_devices = _normalize_cuda(cuda_visible_devices)
    mem_util = gpu_memory_utilization if gpu_memory_utilization is not None else config.GPU_MEMORY_UTILIZATION
    key = (model_path, cuda_devices, mem_util)
    if key in _ENGINE_CACHE:
        return _ENGINE_CACHE[key], cuda_devices

    prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    engine_args = AsyncEngineArgs(
        model=model_path,
        hf_overrides={"architectures": ["DeepseekOCRForCausalLM"]},
        block_size=256,
        max_model_len=8192,
        enforce_eager=False,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=mem_util,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    _ENGINE_CACHE[key] = engine

    if prev_cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", cuda_devices)

    return engine, cuda_devices


def unload_all_engines() -> None:
    while _ENGINE_CACHE:
        _, engine = _ENGINE_CACHE.popitem()
        try:
            engine.shutdown()
        except Exception:
            pass
