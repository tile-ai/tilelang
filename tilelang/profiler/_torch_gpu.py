"""Timing implementations shared by PyTorch's CUDA and ROCm runtimes."""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal
from collections.abc import Callable

import torch

from ._common import aggregate_times

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


_CACHE_FLUSH_ID = "tilelang::cache_flush"


def device_scope(device: torch.device):
    return torch.cuda.device(device)


def synchronize(device: torch.device | int | None) -> None:
    torch.cuda.synchronize(device)


def benchmark(
    fn: Callable,
    warmup: float,
    rep: float,
    _n_warmup: int,
    _n_repeat: int,
    quantiles: list[float] | None,
    fast_flush: bool,
    backend: Literal["event", "cupti", "cudagraph"],
    return_mode: Literal["min", "max", "mean", "median"],
    device: torch.device,
    cache_size: int,
    early_stop_baseline: float | None = None,
) -> float | list[float]:
    device_idx = device.index
    # Initial function call and synchronization
    fn()
    synchronize(device_idx)

    # Create L2 cache flush buffer (`cache_size` MB)
    # Fast flush uses int32 (4 bytes), regular uses int8 (1 byte)
    cache_bytes = cache_size * 1024 * 1024
    cache_numel = cache_bytes // 4 if fast_flush else cache_bytes
    cache_dtype = torch.int if fast_flush else torch.int8
    cache = torch.empty(cache_numel, dtype=cache_dtype, device=device)

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
    for repeat in range(n_repeat):
        cache.zero_()  # Clear L2 cache
        start_events[repeat].record()
        fn()
        end_events[repeat].record()

    # Synchronize and collect timings
    synchronize(device_idx)
    times = torch.tensor(
        [start.elapsed_time(end) for start, end in zip(start_events, end_events)],
        dtype=torch.float,
        device="cpu",
    )

    return aggregate_times(times, quantiles, return_mode)


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
    stream = torch.cuda.Stream(device=device_idx)
    with torch.cuda.stream(stream):
        # Construct a CUDA graph with `n_repeat` unrolled function calls to minimize host overhead.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(n_repeat):
                fn()

        synchronize(device_idx)

        # Measure time by replaying the graph multiple times.
        # Clear cache before each replay for consistent measurements.
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_retries)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_retries)]
        for retry in range(n_retries):
            cache.zero_()  # Clear L2 cache before replay
            start_events[retry].record()
            graph.replay()
            end_events[retry].record()

        synchronize(device_idx)
        times = torch.tensor(
            [start.elapsed_time(end) / n_repeat for start, end in zip(start_events, end_events)],
            dtype=torch.float,
            device="cpu",
        )

        return aggregate_times(times, quantiles, return_mode)
