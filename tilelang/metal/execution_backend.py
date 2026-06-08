from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


register_execution_backend(
    "metal",
    ExecutionBackendSpec("tvm_ffi", priority=100, enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
register_execution_backend("metal", ExecutionBackendSpec("torch", priority=10), override=True)
