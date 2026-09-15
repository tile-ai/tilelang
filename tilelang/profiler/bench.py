"""Compatibility imports for the shared CUDA and ROCm benchmark implementation."""

from ._torch_gpu import (
    IS_CUDA as IS_CUDA,
    _CACHE_FLUSH_ID as _CACHE_FLUSH_ID,
    _bench_with_cuda_events as _bench_with_cuda_events,
    _bench_with_cudagraph as _bench_with_cudagraph,
    _bench_with_cupti as _bench_with_cupti,
    _cache_device as _cache_device,
    _cuda_synchronize as _cuda_synchronize,
    _do_bench_impl as _do_bench_impl,
    _normalize_cuda_device as _normalize_cuda_device,
    device as device,
    do_bench as do_bench,
    logger as logger,
    suppress_stdout_stderr as suppress_stdout_stderr,
)
