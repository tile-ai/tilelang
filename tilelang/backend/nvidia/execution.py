from __future__ import annotations


CUDA_EXECUTION_BACKENDS = ("tvm_ffi", "nvrtc", "cython")
CUDA_DEFAULT_EXECUTION_BACKEND = "tvm_ffi"
CUTEDSL_EXECUTION_BACKENDS = ("cutedsl",)
CUTEDSL_DEFAULT_EXECUTION_BACKEND = "cutedsl"


def unavailable_cuda_execution_backends() -> tuple[str, ...]:
    try:
        from tilelang.jit.adapter.nvrtc import is_nvrtc_available
    except Exception:
        return ()

    return () if is_nvrtc_available else ("nvrtc",)

