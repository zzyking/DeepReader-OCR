import asyncio
from typing import Callable, Iterable, List, Sequence, Tuple
from uuid import uuid4

from vllm import AsyncLLMEngine, SamplingParams


async def _collect_single_request(
    engine: AsyncLLMEngine,
    payload: dict,
    sampling_params: SamplingParams,
    request_id: str,
) -> str:
    """Run a single vLLM request and return the final text output."""
    final_text = ""
    async for output in engine.generate(payload, sampling_params, request_id):
        if output.outputs:
            final_text = output.outputs[0].text
    return final_text


async def generate_requests_concurrently(
    engine: AsyncLLMEngine,
    requests: Sequence[Tuple[dict, SamplingParams]],
    max_concurrency: int,
) -> List[str]:
    """Execute multiple vLLM requests concurrently with a bounded semaphore."""
    if not requests:
        return []

    max_concurrency = max(1, min(max_concurrency, len(requests)))
    semaphore = asyncio.Semaphore(max_concurrency)
    results: List[str] = [""] * len(requests)

    batch_token = uuid4().hex

    async def _runner(index: int, request: Tuple[dict, SamplingParams]) -> None:
        request_id = f"{batch_token}-{index}"
        payload, sampling_params = request
        async with semaphore:
            results[index] = await _collect_single_request(
                engine=engine,
                payload=payload,
                sampling_params=sampling_params,
                request_id=request_id,
            )

    await asyncio.gather(*(_runner(idx, request) for idx, request in enumerate(requests)))
    return results


def run_concurrent_generation(
    engine: AsyncLLMEngine,
    requests: Sequence[Tuple[dict, SamplingParams]],
    max_concurrency: int | None,
) -> List[str]:
    """Sync wrapper around ``generate_requests_concurrently``."""
    effective_concurrency = max_concurrency if max_concurrency and max_concurrency > 0 else len(requests) or 1
    return asyncio.run(
        generate_requests_concurrently(
            engine=engine,
            requests=requests,
            max_concurrency=effective_concurrency,
        )
    )


async def generate_requests_streaming(
    engine: AsyncLLMEngine,
    request_iterable: Iterable[Tuple[dict, SamplingParams] | Tuple[dict, SamplingParams, Callable[[str], None]]],
    max_concurrency: int,
) -> List[str]:
    """Execute vLLM requests as they are produced, allowing preprocessing to overlap."""
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be positive for streaming generation")

    request_count = len(request_iterable) if hasattr(request_iterable, "__len__") else None
    semaphore = asyncio.Semaphore(min(max_concurrency, request_count) if request_count else max_concurrency)
    results: List[str] = []
    tasks = []

    batch_token = uuid4().hex
    request_index = 0

    async def _runner(
        payload: dict,
        sampling_params: SamplingParams,
        handler: Callable[[str], None] | None,
        request_id: str,
    ):
        async with semaphore:
            text = await _collect_single_request(
                engine=engine,
                payload=payload,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        if handler:
            handler(text)
        return text

    for item in request_iterable:
        if len(item) == 2:
            payload, sampling_params = item  # type: ignore[misc]
            handler = None
        else:
            payload, sampling_params, handler = item  # type: ignore[misc]
        request_id = f"{batch_token}-{request_index}"
        request_index += 1
        tasks.append(asyncio.create_task(_runner(payload, sampling_params, handler, request_id)))
        # Give control back to the loop so generation can start while we continue producing requests.
        await asyncio.sleep(0)

    if tasks:
        results = await asyncio.gather(*tasks)

    return results


def run_streaming_generation(
    engine: AsyncLLMEngine,
    request_iterable: Iterable[Tuple[dict, SamplingParams] | Tuple[dict, SamplingParams, Callable[[str], None]]],
    max_concurrency: int,
) -> List[str]:
    """Sync wrapper around ``generate_requests_streaming``."""
    return asyncio.run(
        generate_requests_streaming(
            engine=engine,
            request_iterable=request_iterable,
            max_concurrency=max_concurrency,
        )
    )
