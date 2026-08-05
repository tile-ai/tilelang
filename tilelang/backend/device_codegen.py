"""Device codegen registry shared by backend packages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tvm import IRModule
from tvm.target import Target

from tilelang import tvm

DeviceCodegenFunc = Callable[[IRModule, Target], IRModule]
TargetPredicate = Callable[[Target], bool]


def global_func_device_codegen(global_func_name: str) -> DeviceCodegenFunc:
    """Create a device codegen callback backed by a TVM global function."""

    def build(mod: IRModule, target: Target) -> IRModule:
        return tvm.ffi.get_global_func(global_func_name)(mod, target)

    return build


@dataclass(frozen=True, slots=True)
class DeviceCodegen:
    """Device codegen entry points for one backend target variant."""

    name: str
    build: DeviceCodegenFunc | None = None
    build_without_compile: DeviceCodegenFunc | None = None
    supports_target: TargetPredicate | None = None

    def matches(self, target: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target)

    def lower(self, mod: IRModule, target: Target, *, compile_device: bool) -> IRModule:
        build_func = self.build if compile_device else self.build_without_compile
        if build_func is None:
            mode = "with compilation" if compile_device else "without compilation"
            raise ValueError(f"Device codegen '{self.name}' for target '{target.kind.name}' does not support lowering {mode}")
        return build_func(mod, target)


def _matching_device_codegens(target: Target) -> list[DeviceCodegen]:
    from tilelang.backend.spec import resolve_backend

    backend = resolve_backend(target)
    return [codegen for codegen in backend.device_codegens[target.kind.name] if codegen.matches(target)]


def allowed_device_codegens_for_target(target: Target) -> list[str]:
    """Return matching device codegen names for a target."""

    return [codegen.name for codegen in _matching_device_codegens(target)]


def resolve_device_codegen(target: Target) -> DeviceCodegen:
    """Compatibility lookup; core compilation uses BackendSpec directly."""
    from tilelang.backend.spec import resolve_backend

    backend = resolve_backend(target)
    return backend.get_device_codegen(target)
