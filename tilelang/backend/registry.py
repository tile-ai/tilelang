from __future__ import annotations

from collections.abc import Iterable

from tvm.target import Target

from tilelang.backend import amd, apple, cpu, nvidia, webgpu
from tilelang.backend.base import DeviceBackend
from tilelang.backend.common.target import target_kind
from tilelang.backend.errors import BackendResolutionError


_CANONICAL_EXECUTION_BACKEND_MAP = {
    "dlpack": "tvm_ffi",
}

_DEVICE_BACKENDS: tuple[DeviceBackend, ...] = (
    *nvidia.get_backends(),
    *amd.get_backends(),
    *apple.get_backends(),
    *cpu.get_backends(),
    *webgpu.get_backends(),
)


def _canon_execution_backend(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).lower()
    return _CANONICAL_EXECUTION_BACKEND_MAP.get(key, key)


def _format_options(options: Iterable[str]) -> str:
    return ", ".join(sorted(options))


def registered_device_backends() -> tuple[DeviceBackend, ...]:
    return _DEVICE_BACKENDS


def get_device_backend(target: Target) -> DeviceBackend:
    for backend in _DEVICE_BACKENDS:
        if backend.match_target(target):
            return backend

    supported = _format_options(backend.name for backend in _DEVICE_BACKENDS)
    raise BackendResolutionError(
        f"No TileLang device backend is registered for target '{target_kind(target)}'. "
        f"Registered backends: {supported}."
    )


def build_device_module(device_mod, target: Target, *, compile: bool):
    backend = get_device_backend(target)
    builder = backend.builder_for(compile=compile)
    return builder.build(device_mod, target)


def allowed_execution_backends_for_target(target: Target, *, include_unavailable: bool = True) -> list[str]:
    return get_device_backend(target).allowed_execution_backends(include_unavailable=include_unavailable)


def resolve_execution_backend(requested: str | None, target: Target) -> str:
    req = _canon_execution_backend(requested)
    backend = get_device_backend(target)
    allowed_all = backend.allowed_execution_backends(include_unavailable=True)
    allowed_available = backend.allowed_execution_backends(include_unavailable=False)

    if req in (None, "auto"):
        choice = backend.default_execution_backend
        if choice not in allowed_available and allowed_available:
            choice = allowed_available[0]
        return choice

    if req not in allowed_all:
        raise ValueError(
            f"Invalid execution backend '{requested}' for target '{target_kind(target)}'. "
            f"Allowed: {_format_options(allowed_all)}. Tip: use execution_backend='auto'."
        )

    if req not in allowed_available:
        raise ValueError(
            f"Execution backend '{requested}' requires extra dependencies and is not available now. "
            f"Try one of: {_format_options(allowed_available)}."
        )

    return req

