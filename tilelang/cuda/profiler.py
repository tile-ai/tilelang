"""CUDA profiling methods and device operations."""

from tilelang.profiler._torch_gpu import benchmark as benchmark
from tilelang.profiler._torch_gpu import device_scope as device_scope
from tilelang.profiler._torch_gpu import synchronize as synchronize

SUPPORTED_METHODS = frozenset({"event", "cupti", "cudagraph"})
