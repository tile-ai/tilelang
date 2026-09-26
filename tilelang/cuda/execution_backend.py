from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec


def _is_plain_cuda_execution_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "tileir" not in target.keys


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


def _is_tileir_keyed_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "tileir" in target.keys


def _is_tileir_available_for_target(target: Target) -> bool:
    if _is_tileir_keyed_target(target):
        return True
    try:
        from tilelang.tileir.checks import is_tileir_available
    except ImportError:
        return False
    return bool(is_tileir_available())


CUDA_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        supports_target=_is_plain_cuda_execution_target,
        enable_host_codegen=True,
        enable_device_compile=True,
        supports_callee_allocated_outputs=True,
    ),
    ExecutionBackendSpec(
        "nvrtc",
        is_available=_is_nvrtc_available,
        supports_target=_is_plain_cuda_execution_target,
    ),
    ExecutionBackendSpec("cython", supports_target=_is_plain_cuda_execution_target),
    ExecutionBackendSpec(
        "tileir",
        is_available=lambda: True,
        supports_target=_is_tileir_available_for_target,
    ),
]

CUTEDSL_EXECUTION_BACKENDS = [
    ExecutionBackendSpec("cutedsl", is_available=_is_cutedsl_available),
]
