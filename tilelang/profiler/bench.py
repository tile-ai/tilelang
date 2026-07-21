"""Profiler and benchmarking utilities for PyTorch functions."""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


class suppress_stdout_stderr:
    """Context manager to suppress stdout and stderr output.

    Source: https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/testing/bench.py
    """

    def __enter__(self):
        # Open null device files
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        # Save original file descriptors
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        # Save original stdout/stderr objects
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        # Redirect file descriptors and streams to null device
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file

        return self

    def __exit__(self, *_):
        # Restore original stdout/stderr objects
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        # Restore original file descriptors
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        # Close duplicated file descriptors
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        # Close null device files
        self.outnull_file.close()
        self.errnull_file.close()


IS_CUDA = torch.cuda.is_available()
device = "cuda:0" if IS_CUDA else "mps:0"
_CACHE_FLUSH_ID = "tilelang::cache_flush"


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    quantiles: list[float] | None = None,
    fast_flush: bool = True,
    backend: Literal["event", "cupti", "cudagraph"] = "event",
    return_mode: Literal["min", "max", "mean", "median"] = "mean",
    device: int | torch.device | None = None,
    cache_size: int = 256,
    early_stop_baseline: float | None = None,
) -> float | list[float]:
    """Benchmark the runtime of a PyTorch function with L2 cache management.

    This function provides accurate GPU kernel timing by:
    - Clearing L2 cache between runs for consistent measurements
    - Auto-calculating warmup and repeat counts based on kernel runtime
    - Supporting multiple profiling backends (CUDA events, CUPTI, or CUDA graph replay)
    - Offering flexible result aggregation (mean/median/min/max/quantiles)

    Args:
        fn: Function to benchmark
        warmup: Target warmup time in milliseconds (default: 25)
        rep: Target total benchmark time in milliseconds (default: 100)
        _n_warmup: Manual override for warmup iterations (default: 0 = auto)
        _n_repeat: Manual override for benchmark iterations (default: 0 = auto)
        quantiles: Performance percentiles to compute (e.g., [0.5, 0.95])
        fast_flush: Use faster L2 cache flush with int32 vs int8 (default: True)
        backend: Profiler backend - "event" (CUDA events), "cupti", or "cudagraph" (default: "event")
        return_mode: Result aggregation method - "mean", "median", "min", or "max"
        device: Optional CUDA device to benchmark on. When provided, CUDA
            events, streams, cache buffers, and synchronizations are scoped to
            that device.
        cache_size: L2 cache flush buffer size in MB (default: 256)

    Returns:
        Runtime in milliseconds (float) or list of quantile values if quantiles specified
    """
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


def _cuda_synchronize(device_idx: int | None = None) -> None:
    if device_idx is None:
        torch.cuda.synchronize()
    else:
        torch.cuda.synchronize(device_idx)


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
        logger.warning(
            "Early stop: estimate_ms=%.3fms exceeds baseline=%.3fms, skipping full benchmark.",
            estimate_ms,
            early_stop_baseline,
        )
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


def _bench_with_cuda_events(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
    device_idx: int | None,
) -> float | list[float]:
    """Benchmark using CUDA events for timing."""
    # Create timing events
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_repeat)]

    # Run benchmark iterations
    for i in range(n_repeat):
        cache.zero_()  # Clear L2 cache
        start_events[i].record()
        fn()
        end_events[i].record()

    # Synchronize and collect timings
    _cuda_synchronize(device_idx)
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_events, end_events)],
        dtype=torch.float,
    )

    # Return quantiles if requested
    if quantiles is not None:
        quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        return quantile_values[0] if len(quantile_values) == 1 else quantile_values

    # Return aggregated result
    return getattr(torch, return_mode)(times).item()


def _bench_with_cupti(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
) -> float:
    """Benchmark using CUPTI profiler for detailed kernel timing."""
    with suppress_stdout_stderr():
        schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
        )

        with profiler:
            for _ in range(2):
                for _ in range(n_repeat):
                    with torch.profiler.record_function(_CACHE_FLUSH_ID):
                        cache.zero_()
                    fn()
                profiler.step()

    # `cache.zero_()` and user code such as `torch.zeros` can share the same
    # generated kernel name, so exclude only the annotated cache flush range.
    def is_cuda_event(event):
        return getattr(getattr(event, "device_type", None), "name", "") == "CUDA"

    total_cuda_time = 0.0
    excluded_time = 0.0

    for event in profiler.events():
        if not is_cuda_event(event):
            continue

        if not event.is_user_annotation:
            total_cuda_time += event.self_device_time_total
        elif event.key == _CACHE_FLUSH_ID:
            excluded_time += event.self_device_time_total

    kernel_time_us = (total_cuda_time - excluded_time) / n_repeat
    return kernel_time_us * 1e-3  # Convert microseconds to milliseconds


def _bench_with_cudagraph(
    fn: Callable,
    cache: torch.Tensor,
    n_repeat: int,
    quantiles: list[float] | None,
    return_mode: str,
    device_idx: int | None,
) -> float | list[float]:
    """Benchmark using CUDA graph for minimal launch overhead.

    This implementation follows triton.testing.do_bench_cudagraph.
    It captures the kernel execution in a CUDA graph and replays it multiple
    times to minimize host overhead and provide accurate timing measurements.

    Note: Cache flushing is done before graph replay, not within the graph,
    since CUDA graphs require fixed execution patterns.
    """
    n_retries = 10
    stream = torch.cuda.Stream(device=device_idx) if device_idx is not None else torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Construct a CUDA graph with `n_repeat` unrolled function calls to minimize host overhead.
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n_repeat):
                fn()

        _cuda_synchronize(device_idx)

        # Measure time by replaying the graph multiple times.
        # Clear cache before each replay for consistent measurements.
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_retries)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_retries)]
        for i in range(n_retries):
            cache.zero_()  # Clear L2 cache before replay
            start_events[i].record()
            g.replay()
            end_events[i].record()

        _cuda_synchronize(device_idx)
        times = torch.tensor(
            [s.elapsed_time(e) / n_repeat for s, e in zip(start_events, end_events)],
            dtype=torch.float,
        )

        # Return quantiles if requested
        if quantiles is not None:
            quantile_values = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
            return quantile_values[0] if len(quantile_values) == 1 else quantile_values

        # Return aggregated result
        return getattr(torch, return_mode)(times).item()
