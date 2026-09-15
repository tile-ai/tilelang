"""Shared result aggregation and synchronized wall-clock measurements."""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Literal

import torch

BenchMethod = Literal["event", "cupti", "cudagraph", "wall"]
ReturnMode = Literal["min", "max", "mean", "median"]


def aggregate_times(times: torch.Tensor, quantiles: list[float] | None, return_mode: str) -> float | list[float]:
    if quantiles is not None:
        values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float, device=times.device)).tolist()
        return values[0] if len(values) == 1 else values
    return getattr(torch, return_mode)(times).item()


def bench_wall(
    fn: Callable,
    *,
    synchronize: Callable[[], None],
    warmup: float,
    rep: float,
    _n_warmup: int,
    _n_repeat: int,
    quantiles: list[float] | None,
    return_mode: ReturnMode,
    early_stop_baseline: float | None,
    **_gpu_options,
) -> float | list[float]:
    """Measure ten synchronized batches, reporting per-call wall time in milliseconds.

    Statistics describe batch averages, including host launch overhead. Cache
    flushing is intentionally not part of this measurement method.
    """
    fn()
    synchronize()
    start = perf_counter()
    for _ in range(5):
        fn()
    synchronize()
    estimate_ms = max((perf_counter() - start) * 1e3 / 5, 1e-9)
    if early_stop_baseline is not None and estimate_ms > early_stop_baseline:
        return [estimate_ms] * len(quantiles) if quantiles is not None else estimate_ms

    n_samples = 10
    n_warmup = _n_warmup if _n_warmup > 0 else max(1, int(warmup / estimate_ms))
    n_repeat = _n_repeat if _n_repeat > 0 else max(1, int(rep / estimate_ms / n_samples))
    for _ in range(n_warmup):
        fn()
    synchronize()

    times = []
    for _ in range(n_samples):
        start = perf_counter()
        for _ in range(n_repeat):
            fn()
        synchronize()
        times.append((perf_counter() - start) * 1e3 / n_repeat)
    return aggregate_times(torch.tensor(times, dtype=torch.float, device="cpu"), quantiles, return_mode)
