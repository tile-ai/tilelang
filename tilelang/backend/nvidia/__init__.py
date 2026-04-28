from __future__ import annotations

from tilelang.backend.base import DeviceBackend

from .execution import (
    CUDA_DEFAULT_EXECUTION_BACKEND,
    CUDA_EXECUTION_BACKENDS,
    CUDA_EXECUTION_SPECS,
    CUTEDSL_DEFAULT_EXECUTION_BACKEND,
    CUTEDSL_EXECUTION_BACKENDS,
    CUTEDSL_EXECUTION_SPECS,
    unavailable_cuda_execution_backends,
)
from .ffi import CUDA_COMPILED_BUILDER, CUDA_SOURCE_BUILDER, CUTEDSL_SOURCE_BUILDER
from .passes import NvidiaPassHooks
from .target import is_cutedsl_target, is_tilelang_cuda_target


def get_backends() -> tuple[DeviceBackend, ...]:
    hooks = NvidiaPassHooks()
    return (
        DeviceBackend(
            name="cutedsl",
            family="nvidia",
            match_target=is_cutedsl_target,
            source_builder=CUTEDSL_SOURCE_BUILDER,
            compiled_builder=None,
            execution_backends=CUTEDSL_EXECUTION_BACKENDS,
            default_execution_backend=CUTEDSL_DEFAULT_EXECUTION_BACKEND,
            source_kind="cutedsl_py",
            execution_specs=CUTEDSL_EXECUTION_SPECS,
            pass_hooks=hooks,
            metadata={"dialect": "cutedsl"},
        ),
        DeviceBackend(
            name="nvidia",
            family="nvidia",
            match_target=is_tilelang_cuda_target,
            source_builder=CUDA_SOURCE_BUILDER,
            compiled_builder=CUDA_COMPILED_BUILDER,
            execution_backends=CUDA_EXECUTION_BACKENDS,
            default_execution_backend=CUDA_DEFAULT_EXECUTION_BACKEND,
            source_kind="cuda",
            execution_specs=CUDA_EXECUTION_SPECS,
            pass_hooks=hooks,
            unavailable_execution_backends=unavailable_cuda_execution_backends,
            metadata={"dialect": "cuda"},
        ),
    )
