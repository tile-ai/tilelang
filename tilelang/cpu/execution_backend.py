from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


for _target_kind in ("c", "llvm"):
    register_execution_backend(_target_kind, ExecutionBackendSpec("cython"), override=True)
    register_execution_backend(
        _target_kind,
        ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
        override=True,
    )
