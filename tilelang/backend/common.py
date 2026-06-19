from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend

# Importing the package registers its pass pipeline.
import tilelang.webgpu  # noqa: F401


register_execution_backend("webgpu", ExecutionBackendSpec("cython"), override=True)
register_execution_backend(
    "webgpu",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
