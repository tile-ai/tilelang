from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from tilelang import tvm
from tvm.target import Target


BuilderMode = Literal["source", "compiled"]


@dataclass(frozen=True)
class FFIBuilderRef:
    """Lazy reference to a Python-accessible C++ FFI builder."""

    ffi_symbol: str
    mode: BuilderMode

    def build(self, mod: tvm.IRModule, target: Target) -> tvm.runtime.Module:
        return tvm.ffi.get_global_func(self.ffi_symbol)(mod, target)


class BackendPassHooks:
    """No-op backend pass hooks for shared pipeline integration."""

    def pre_layout(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def post_layout(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def post_tile_lowering(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def pre_storage_rewrite(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def post_storage_rewrite(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def before_split_host_device(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def after_split_host_device(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def before_device_codegen(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod


UnavailableExecutionBackends = Callable[[], tuple[str, ...]]


@dataclass(frozen=True)
class DeviceBackend:
    name: str
    family: str
    match_target: Callable[[Target], bool]
    source_builder: FFIBuilderRef | None
    compiled_builder: FFIBuilderRef | None
    execution_backends: tuple[str, ...]
    default_execution_backend: str
    source_kind: str
    pass_hooks: BackendPassHooks = field(default_factory=BackendPassHooks)
    unavailable_execution_backends: UnavailableExecutionBackends | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def builder_for(self, *, compile: bool) -> FFIBuilderRef:
        builder = self.compiled_builder if compile else self.source_builder
        if builder is None:
            mode = "compiled" if compile else "source"
            raise ValueError(f"Backend '{self.name}' does not support {mode} device codegen")
        return builder

    def allowed_execution_backends(self, *, include_unavailable: bool = True) -> list[str]:
        allowed = list(self.execution_backends)
        if include_unavailable or self.unavailable_execution_backends is None:
            return allowed

        unavailable = set(self.unavailable_execution_backends())
        return [backend for backend in allowed if backend not in unavailable]

