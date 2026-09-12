from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec


EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
        native_multi_launch=True,
        native_argument_binding=True,
    ),
    ExecutionBackendSpec("cython"),
]
