from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

# Plain Ascend prefers tvm_ffi -- it is listed first, so execution_backend="auto"
# resolves to it -- and also accepts cython.
ASCEND_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
    ExecutionBackendSpec("cython"),
]
