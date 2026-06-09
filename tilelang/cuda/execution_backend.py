from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


def _is_cutedsl_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" in target.keys


def _is_plain_cuda_target(target: Target) -> bool:
    return target.kind.name == "cuda" and "cutedsl" not in target.keys


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


register_execution_backend(
    "cuda",
    ExecutionBackendSpec(
        "tvm_ffi",
        supports_target=_is_plain_cuda_target,
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
    override=True,
)
register_execution_backend(
    "cuda",
    ExecutionBackendSpec("nvrtc", is_available=_is_nvrtc_available, supports_target=_is_plain_cuda_target),
    override=True,
)
register_execution_backend(
    "cuda",
    ExecutionBackendSpec("cython", supports_target=_is_plain_cuda_target),
    override=True,
)
register_execution_backend(
    "cuda",
    ExecutionBackendSpec("cutedsl", is_available=_is_cutedsl_available, supports_target=_is_cutedsl_target),
    override=True,
)
