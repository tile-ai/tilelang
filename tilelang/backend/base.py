from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from tilelang import tvm
from tvm.target import Target


BuilderMode = Literal["source", "compiled"]
AdapterFactory = Callable[..., Any]
CacheFactory = Callable[[], Any]
SourceWrapperFactory = Callable[..., Any]
LibraryCommandFactory = Callable[[Target, str, str, dict[str, Any]], list[str]]


@dataclass(frozen=True)
class FFIBuilderRef:
    """Lazy reference to a Python-accessible C++ FFI builder."""

    ffi_symbol: str
    mode: BuilderMode

    def build(self, mod: tvm.IRModule, target: Target) -> tvm.runtime.Module:
        return tvm.ffi.get_global_func(self.ffi_symbol)(mod, target)


class BackendPassHooks:
    """No-op backend pass hooks for shared pipeline integration."""

    def adjust_aggressive_shared_memory_merge(self, enabled: bool, target: Target) -> bool:
        return enabled

    def pre_layout(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def post_layout(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def post_tile_lowering(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def optimize_entry(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def lower_shared_barrier(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def after_shared_barrier_lowering(self, mod: tvm.IRModule, target: Target, *, has_tma: bool) -> tvm.IRModule:
        return mod

    def pre_storage_rewrite(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def post_storage_rewrite(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def before_split_host_device(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def after_split_host_device(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def after_shared_memory_planning(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def after_shared_sync(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod

    def before_device_codegen(self, mod: tvm.IRModule, target: Target) -> tvm.IRModule:
        return mod


UnavailableExecutionBackends = Callable[[], tuple[str, ...]]


@dataclass(frozen=True)
class CacheArtifactSpec:
    kernel_lib_path: str = "kernel_lib.so"
    device_kernel_path: str = "device_kernel.cu"
    host_kernel_path: str = "host_kernel.cu"
    extra_required_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class LibraryCompileSpec:
    source_suffix: str
    library_suffix: str
    command_factory: LibraryCommandFactory


@dataclass(frozen=True)
class ExecutionBackendSpec:
    name: str
    adapter_factory: AdapterFactory
    database_adapter_factory: AdapterFactory | None
    cache_factory: CacheFactory
    cache_artifact: CacheArtifactSpec = field(default_factory=CacheArtifactSpec)
    c_source_wrapper_factory: SourceWrapperFactory | None = None
    python_source_wrapper_factory: SourceWrapperFactory | None = None
    library_compile_spec: LibraryCompileSpec | None = None
    requires_host_codegen: bool = False
    requires_device_compile: bool = False
    kernel_source_from_adapter: bool = True
    host_source_from_adapter: bool = True
    requires_cxx_compiler: bool = False


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
    execution_specs: tuple[ExecutionBackendSpec, ...] = ()
    pass_hooks: BackendPassHooks = field(default_factory=BackendPassHooks)
    unavailable_execution_backends: UnavailableExecutionBackends | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.execution_specs:
            spec_names = tuple(spec.name for spec in self.execution_specs)
            if self.execution_backends and self.execution_backends != spec_names:
                raise ValueError(
                    f"Backend '{self.name}' execution_backends must match execution_specs order: "
                    f"{self.execution_backends} != {spec_names}"
                )
            if self.default_execution_backend not in spec_names:
                raise ValueError(
                    f"Backend '{self.name}' default execution backend "
                    f"'{self.default_execution_backend}' is not registered"
                )

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

    def execution_spec(self, name: str) -> ExecutionBackendSpec:
        for spec in self.execution_specs:
            if spec.name == name:
                return spec
        raise ValueError(f"Execution backend '{name}' is not registered for device backend '{self.name}'")
