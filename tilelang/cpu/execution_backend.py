from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec


def _is_c_target(target: Target) -> bool:
    return target.kind.name == "c"


EXECUTION_BACKENDS = [
    ExecutionBackendSpec("cython", supports_target=_is_c_target),
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
]
