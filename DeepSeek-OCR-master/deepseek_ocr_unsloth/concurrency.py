"""Lightweight concurrency helpers (sequential by default)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Optional


def run_tasks(tasks: Iterable[Callable[[], object]], max_concurrency: int = 1) -> List[object]:
    """Run callables with optional thread-level concurrency."""
    task_list = list(tasks)
    if not task_list:
        return []
    max_workers = max(1, int(max_concurrency) if max_concurrency else 1)
    if max_workers == 1:
        return [task() for task in task_list]
    results: List[object] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(task): idx for idx, task in enumerate(task_list)}
        ordered: List[object] = [None] * len(task_list)  # type: ignore
        for future in as_completed(future_map):
            idx = future_map[future]
            ordered[idx] = future.result()
        results = ordered
    return results
