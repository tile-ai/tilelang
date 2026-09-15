"""Device-independent entry point for benchmarking PyTorch callables."""

from __future__ import annotations

from collections.abc import Callable

import torch

from ._common import BenchMethod, ReturnMode
from .device import get_backend, resolve_device


def do_bench(
    fn: Callable,
    warmup: float = 25,
    rep: float = 100,
    _n_warmup: int = 0,
    _n_repeat: int = 0,
    quantiles: list[float] | None = None,
    fast_flush: bool = True,
    backend: BenchMethod = "event",
    return_mode: ReturnMode = "mean",
    device: int | str | torch.device | None = None,
    cache_size: int = 256,
    early_stop_baseline: float | None = None,
) -> float | list[float]:
    """Benchmark a callable, returning milliseconds or requested quantiles.

    ``backend`` selects a timing method, not a compilation or execution backend.
    CUDA/ROCm support ``event``, ``cupti`` (Torch profiler kernel time), and
    ``cudagraph``. CPU and Metal support ``wall`` (synchronized batch wall time).
    Unsupported methods raise rather than silently changing the measurement.

    ``warmup`` and ``rep`` are time budgets in milliseconds. Positive
    ``_n_warmup`` and ``_n_repeat`` override iteration counts. ``return_mode``
    selects min/max/mean/median; a single quantile returns a scalar and multiple
    quantiles return a list. As before, ``cupti`` returns mean kernel time only.

    ``device`` scopes allocations, synchronization, events, and streams. Integer
    devices are CUDA/ROCm ordinals. Without an explicit device, the current GPU,
    MPS, or CPU is used; a callable's closure is not inspected. ``fast_flush``
    and ``cache_size`` (MB) apply only to GPU timing methods. ``wall`` does not
    flush caches. ``early_stop_baseline`` may skip full timing after estimation.
    """
    benchmark_device = resolve_device(device)
    implementation = get_backend(benchmark_device)
    if backend not in implementation.SUPPORTED_METHODS:
        supported = ", ".join(sorted(implementation.SUPPORTED_METHODS))
        raise ValueError(f"Profiling method {backend!r} is not supported on {benchmark_device}; supported methods: {supported}")
    if return_mode not in ("min", "max", "mean", "median"):
        raise ValueError(f"Unknown benchmark return mode: {return_mode!r}")
    if quantiles is not None and (not quantiles or any(not 0 <= quantile <= 1 for quantile in quantiles)):
        raise ValueError("Benchmark quantiles must be a non-empty list of values between 0 and 1")

    with implementation.device_scope(benchmark_device):
        return implementation.benchmark(
            fn,
            warmup=warmup,
            rep=rep,
            _n_warmup=_n_warmup,
            _n_repeat=_n_repeat,
            quantiles=quantiles,
            fast_flush=fast_flush,
            backend=backend,
            return_mode=return_mode,
            device=benchmark_device,
            cache_size=cache_size,
            early_stop_baseline=early_stop_baseline,
        )
