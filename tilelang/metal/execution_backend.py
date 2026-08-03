from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


# torch is registered first so that `execution_backend="auto"` resolves to the
# Metal adapter (torch.mps.compile_shader), which is the supported execution path
# for Metal. tvm_ffi remains available as an explicit opt-in.
register_execution_backend("metal", ExecutionBackendSpec("torch"), override=True)
register_execution_backend(
    "metal",
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    override=True,
)
