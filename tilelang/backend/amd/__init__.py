from __future__ import annotations

from tilelang.backend.base import DeviceBackend

from .execution import HIP_DEFAULT_EXECUTION_BACKEND, HIP_EXECUTION_BACKENDS
from .ffi import HIP_COMPILED_BUILDER, HIP_SOURCE_BUILDER
from .passes import AmdPassHooks
from .target import is_hip_target


def get_backends() -> tuple[DeviceBackend, ...]:
    return (
        DeviceBackend(
            name="amd",
            family="amd",
            match_target=is_hip_target,
            source_builder=HIP_SOURCE_BUILDER,
            compiled_builder=HIP_COMPILED_BUILDER,
            execution_backends=HIP_EXECUTION_BACKENDS,
            default_execution_backend=HIP_DEFAULT_EXECUTION_BACKEND,
            source_kind="hip",
            pass_hooks=AmdPassHooks(),
            metadata={"dialect": "hip"},
        ),
    )

