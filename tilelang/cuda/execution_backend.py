from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec


def _is_cutedsl_available() -> bool:
    try:
        from tilelang.cuda.cutedsl_backend import check_cutedsl_available

        check_cutedsl_available()
    except ImportError:
        return False
    return True


CUDA_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
        supports_callee_allocated_outputs=True,
    ),
    ExecutionBackendSpec("cython"),
]

CUTEDSL_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        is_available=_is_cutedsl_available,
        enable_host_codegen=True,
        enable_device_compile=True,
        supports_callee_allocated_outputs=True,
    ),
]
