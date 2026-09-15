"""Profiling backend descriptors and shared PyTorch GPU timing policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from tvm.target import Target


@dataclass(frozen=True, slots=True)
class ProfilerBackendSpec:
    name: str
    do_bench: Callable[..., float | list[float]]
    supports_target: Callable[[Target], bool] | None = None

    def matches(self, target: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target)


def _torch_gpu_do_bench(*args, **kwargs) -> float | list[float]:
    from tilelang.profiler._torch_gpu import do_bench

    return do_bench(*args, **kwargs)


EVENT_PROFILER_BACKEND = ProfilerBackendSpec("event", partial(_torch_gpu_do_bench, backend="event"))
CUPTI_PROFILER_BACKEND = ProfilerBackendSpec("cupti", partial(_torch_gpu_do_bench, backend="cupti"))
CUDAGRAPH_PROFILER_BACKEND = ProfilerBackendSpec("cudagraph", partial(_torch_gpu_do_bench, backend="cudagraph"))

TORCH_GPU_PROFILER_BACKENDS = (EVENT_PROFILER_BACKEND, CUPTI_PROFILER_BACKEND, CUDAGRAPH_PROFILER_BACKEND)
