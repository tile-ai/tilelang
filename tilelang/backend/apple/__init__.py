from __future__ import annotations

from tilelang.backend.base import DeviceBackend

from .execution import METAL_DEFAULT_EXECUTION_BACKEND, METAL_EXECUTION_BACKENDS, METAL_EXECUTION_SPECS
from .ffi import METAL_COMPILED_BUILDER, METAL_SOURCE_BUILDER
from .passes import ApplePassHooks
from .target import is_metal_target


def get_backends() -> tuple[DeviceBackend, ...]:
    return (
        DeviceBackend(
            name="apple",
            family="apple",
            match_target=is_metal_target,
            source_builder=METAL_SOURCE_BUILDER,
            compiled_builder=METAL_COMPILED_BUILDER,
            execution_backends=METAL_EXECUTION_BACKENDS,
            default_execution_backend=METAL_DEFAULT_EXECUTION_BACKEND,
            source_kind="metal",
            execution_specs=METAL_EXECUTION_SPECS,
            pass_hooks=ApplePassHooks(),
            metadata={"dialect": "metal"},
        ),
    )
