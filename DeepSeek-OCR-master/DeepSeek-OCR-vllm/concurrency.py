import asyncio
from typing import List, Sequence, Tuple
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
