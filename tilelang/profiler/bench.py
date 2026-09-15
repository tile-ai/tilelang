"""Common entry point for GPU and wall-clock benchmarking."""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from math import isfinite
from typing import Literal

import torch

from .torch_bench import (
    _CACHE_FLUSH_ID as _CACHE_FLUSH_ID,
    _cuda_synchronize as _cuda_synchronize,
    bench_with_cuda_events as _bench_with_cuda_events,
    bench_with_cudagraph as _bench_with_cudagraph,
    bench_with_cupti as _bench_with_cupti,
    suppress_stdout_stderr as suppress_stdout_stderr,
)
from .wall import bench_with_wall

logger = logging.getLogger(__name__)

IS_CUDA = torch.cuda.is_available()
device = "cuda:0" if IS_CUDA else "mps:0"


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    quantiles: list[float] | None = None,
    fast_flush: bool = True,
    backend: Literal["event", "cupti", "cudagraph", "wall"] = "event",
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
    device: int | torch.device | None = None,
    cache_size: int = 256,
    early_stop_baseline: float | None = None,
) -> float | list[float]:
    """Benchmark a callable with GPU timing or host wall-clock timing.

    The existing GPU timing methods provide accurate kernel timing by:
    - Clearing L2 cache between runs for consistent measurements
    - Auto-calculating warmup and repeat counts based on kernel runtime
    - Supporting multiple profiling backends (CUDA events, CUPTI, or CUDA graph replay)
    - Offering flexible result aggregation (mean/median/min/max/quantiles)

    Wall timing measures host elapsed time without cache flushing. With no
    device or a CPU device, the callable is assumed to be synchronous. An
    explicit CUDA/HIP or MPS device enables synchronization before and after
    each wall-clock sample, including launch and synchronization overhead.

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds (default: 25)
        rep: Target total benchmark time in milliseconds (default: 100)
        _n_warmup: Manual override for warmup iterations (default: 0 = auto)
        _n_repeat: Manual override for benchmark iterations (default: 0 = auto)
        quantiles: Performance percentiles to compute (e.g., [0.5, 0.95])
        fast_flush: Use faster GPU L2 cache flush with int32 vs int8 (default: True); ignored by "wall"
        backend: Timing method - "event", "cupti", "cudagraph", or "wall" (default: "event")
        return_mode: Result aggregation method - "mean", "median", "min", or "max"
        device: Optional device to benchmark on. GPU methods require a CUDA/HIP
            device and scope events, streams, cache buffers, and synchronization
            to it. Wall timing also accepts CPU and MPS devices.
        cache_size: GPU L2 cache flush buffer size in MB (default: 256); ignored by "wall"

    Returns:
        Runtime in milliseconds (float) or list of quantile values if quantiles specified
    """
    if backend == "wall":
        if return_mode not in ("min", "max", "mean", "median"):
            raise ValueError(f"Invalid return_mode: {return_mode}")
        if any(not isfinite(duration) or duration < 0 for duration in (warmup, rep)):
            raise ValueError("warmup and rep must be finite, non-negative millisecond budgets")
        if _n_warmup < 0 or _n_repeat < 0:
            raise ValueError("Iteration overrides must be non-negative")
        if quantiles is not None and any(not 0 <= quantile <= 1 for quantile in quantiles):
            raise ValueError("quantiles must be between 0 and 1")

        resolved_device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device) if device is not None else None
        device_context = nullcontext()
        synchronize = None
        if resolved_device is not None:
            if resolved_device.type == "cuda":
                device_context = torch.cuda.device(resolved_device)
                synchronize = partial(torch.cuda.synchronize, resolved_device)
            elif resolved_device.type == "mps":
                synchronize = torch.mps.synchronize
            elif resolved_device.type != "cpu":
                raise ValueError(f"Wall timing supports CPU, CUDA/HIP, or MPS devices, got {resolved_device}")

        with device_context:
            fn()
            if synchronize is not None:
                synchronize()

            estimate_ms = bench_with_wall(fn, n_repeat=5, synchronize=synchronize)
            if early_stop_baseline is not None and estimate_ms > early_stop_baseline:
                if quantiles is not None:
                    return estimate_ms if len(quantiles) == 1 else [estimate_ms] * len(quantiles)
                return estimate_ms

            estimate_ms = max(estimate_ms, 1e-6)
            n_warmup = _n_warmup if _n_warmup > 0 else int(warmup / estimate_ms)
            n_repeat = _n_repeat if _n_repeat > 0 else max(1, int(rep / estimate_ms))

            for _ in range(n_warmup):
                fn()
            if synchronize is not None:
                synchronize()

            return bench_with_wall(fn, n_repeat=n_repeat, quantiles=quantiles, return_mode=return_mode, synchronize=synchronize)

    assert return_mode in ["min", "max", "mean", "median"], f"Invalid return_mode: {return_mode}"

    device_idx = _normalize_cuda_device(device)
    if device_idx is not None:
        with torch.cuda.device(device_idx):
            return _do_bench_impl(
                fn,
                warmup=warmup,
                rep=rep,
                _n_warmup=_n_warmup,
                _n_repeat=_n_repeat,
                quantiles=quantiles,
                fast_flush=fast_flush,
                backend=backend,
                return_mode=return_mode,
                device_idx=device_idx,
                cache_size=cache_size,
                early_stop_baseline=early_stop_baseline,
            )

    return _do_bench_impl(
        fn,
        warmup=warmup,
        rep=rep,
        _n_warmup=_n_warmup,
        _n_repeat=_n_repeat,
        quantiles=quantiles,
        fast_flush=fast_flush,
        backend=backend,
        return_mode=return_mode,
        device_idx=None,
        cache_size=cache_size,
        early_stop_baseline=early_stop_baseline,
    )


def _normalize_cuda_device(benchmark_device: int | torch.device | None) -> int | None:
    """Return a concrete CUDA device index, preserving implicit mode for None."""
    if benchmark_device is None:
        return None
    if isinstance(benchmark_device, int):
        return benchmark_device

    torch_device = torch.device(benchmark_device)
    if torch_device.type != "cuda":
        raise ValueError(f"do_bench device must be a CUDA device, got {torch_device}")
    if torch_device.index is None:
        return torch.cuda.current_device()
    return torch_device.index


def _cache_device(device_idx: int | None) -> str | torch.device:
    if device_idx is None:
        return device
    return torch.device("cuda", device_idx)


def _do_bench_impl(
    fn: Callable,
    warmup: float,
    rep: float,
    _n_warmup: int,
    _n_repeat: int,
    quantiles: list[float] | None,
    fast_flush: bool,
    backend: Literal["event", "cupti", "cudagraph"],
    return_mode: Literal["min", "max", "mean", "median"],
    device_idx: int | None,
    cache_size: int,
    early_stop_baseline: float | None = None,
) -> float | list[float]:
    # Initial function call and synchronization
    fn()
    _cuda_synchronize(device_idx)

    # Create L2 cache flush buffer (`cache_size` MB)
    # Fast flush uses int32 (4 bytes), regular uses int8 (1 byte)
    cache_bytes = cache_size * 1024 * 1024
    cache_numel = cache_bytes // 4 if fast_flush else cache_bytes
    cache_dtype = torch.int if fast_flush else torch.int8
    cache = torch.empty(cache_numel, dtype=cache_dtype, device=_cache_device(device_idx))

    # Estimate kernel runtime with 5 iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    start_event.synchronize()
    end_event.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # Early stop: skip full benchmark if estimate exceeds baseline
    if early_stop_baseline is not None and estimate_ms > early_stop_baseline:
        logger.debug(
            "Early stop: estimate_ms=%.3fms exceeds baseline=%.3fms, skipping full benchmark.",
            estimate_ms,
            early_stop_baseline,
        )
        if quantiles is not None:
            return [estimate_ms] * len(quantiles)
        return estimate_ms

    # Calculate warmup and repeat counts (minimum 1 iteration each)
    n_warmup = _n_warmup if _n_warmup > 0 else max(1, int(warmup / estimate_ms))
    n_repeat = _n_repeat if _n_repeat > 0 else max(1, int(rep / estimate_ms))

    # Warmup phase
    for _ in range(n_warmup):
        fn()

    # Benchmarking phase
    if backend == "event":
        return _bench_with_cuda_events(fn, cache, n_repeat, quantiles, return_mode, device_idx)
    elif backend == "cupti":
        return _bench_with_cupti(fn, cache, n_repeat)
    elif backend == "cudagraph":
        return _bench_with_cudagraph(fn, cache, n_repeat, quantiles, return_mode, device_idx)
    else:
        raise ValueError(f"Unknown profiler backend: {backend}")
