from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec


def _is_nvrtc_available() -> bool:
    try:
        from tilelang.jit.adapter.nvrtc import is_nvrtc_available
    except ImportError:
        return False
    return bool(is_nvrtc_available)


def _is_cutedsl_available() -> bool:
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
    except ImportError:
        return False
    return True


CUDA_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
    ExecutionBackendSpec("nvrtc", is_available=_is_nvrtc_available),
    ExecutionBackendSpec("cython"),
]

CUTEDSL_EXECUTION_BACKENDS = [
    ExecutionBackendSpec("cutedsl", is_available=_is_cutedsl_available),
]
