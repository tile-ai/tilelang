from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


register_execution_backend("c", ExecutionBackendSpec("cython"), override=True)
register_execution_backend(
    "c",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)

register_execution_backend(
    "llvm",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
