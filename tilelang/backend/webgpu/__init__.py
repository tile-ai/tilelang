from __future__ import annotations

from tilelang.backend.base import DeviceBackend

from .execution import WEBGPU_DEFAULT_EXECUTION_BACKEND, WEBGPU_EXECUTION_BACKENDS, WEBGPU_EXECUTION_SPECS
from .ffi import WEBGPU_SOURCE_BUILDER
from .passes import WebGPUPassHooks
from .target import is_webgpu_target


def get_backends() -> tuple[DeviceBackend, ...]:
    return (
        DeviceBackend(
            name="webgpu",
            family="webgpu",
            match_target=is_webgpu_target,
            source_builder=WEBGPU_SOURCE_BUILDER,
            compiled_builder=None,
            execution_backends=WEBGPU_EXECUTION_BACKENDS,
            default_execution_backend=WEBGPU_DEFAULT_EXECUTION_BACKEND,
            source_kind="webgpu",
            execution_specs=WEBGPU_EXECUTION_SPECS,
            pass_hooks=WebGPUPassHooks(),
            metadata={"dialect": "webgpu"},
        ),
    )
