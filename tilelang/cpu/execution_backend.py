from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec


# C source modules are materialized together with the host module by TVM-FFI.
EXECUTION_BACKENDS = [
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True),
]
