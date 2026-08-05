"""Host codegen registry shared by backend packages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from tvm import IRModule
from tvm.target import Target

from tilelang import tvm

HostCodegenFunc = Callable[[IRModule, Target], IRModule]
HostCodegenHookFunc = Callable[[IRModule, Target, Target], IRModule]
TargetPredicate = Callable[[Target], bool]


def global_func_host_codegen(global_func_name: str) -> HostCodegenFunc:
    """Create a host codegen callback backed by a TVM global function."""

    def build(mod: IRModule, target_host: Target) -> IRModule:
        return tvm.ffi.get_global_func(global_func_name)(mod, target_host)

    return build


@dataclass(frozen=True, slots=True)
class HostCodegen:
    """Host codegen entry point for one host target variant."""

    name: str
    build: HostCodegenFunc
    supports_target: TargetPredicate | None = None

    def matches(self, target_host: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target_host)

    def lower(self, mod: IRModule, target_host: Target) -> IRModule:
        return self.build(mod, target_host)


@dataclass(frozen=True, slots=True)
class HostCodegenHook:
    """Device-backend hook applied before host codegen build."""

    name: str
    apply: HostCodegenHookFunc
    supports_target: TargetPredicate | None = None

    def matches(self, target: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target)

    def lower(self, mod: IRModule, target_host: Target, target: Target) -> IRModule:
        return self.apply(mod, target_host, target)


STANDARD_HOST_CODEGENS: Mapping[str, tuple[HostCodegen, ...]] = MappingProxyType(
    {
        "c": (HostCodegen("c", build=global_func_host_codegen("target.build.tilelang_c_host")),),
        "llvm": (HostCodegen("llvm", build=global_func_host_codegen("target.build.llvm")),),
    }
)


def _matching_host_codegens(target_host: Target) -> list[HostCodegen]:
    return [codegen for codegen in STANDARD_HOST_CODEGENS.get(target_host.kind.name, ()) if codegen.matches(target_host)]


def _matching_host_codegen_hooks(target: Target) -> list[HostCodegenHook]:
    from tilelang.backend.module import resolve_backend

    backend = resolve_backend(target)
    return [hook for hook in backend.host_codegen_hooks.get(target.kind.name, ()) if hook.matches(target)]


def allowed_host_codegens_for_target(target_host: Target) -> list[str]:
    """Return matching host codegen names for a host target."""

    return [codegen.name for codegen in _matching_host_codegens(target_host)]


def apply_host_codegen_hooks(mod: IRModule, target_host: Target, target: Target | None) -> IRModule:
    """Compatibility helper; core compilation uses BackendModule directly."""

    if target is None:
        return mod
    from tilelang.backend.module import resolve_backend

    backend = resolve_backend(target)
    return backend.preprocess_host_codegen(mod, target_host, target)


def resolve_host_codegen(target_host: Target) -> HostCodegen:
    """Compatibility lookup for the standard c/llvm host codegen."""
    matches = _matching_host_codegens(target_host)
    if not matches:
        raise ValueError(f"No standard host codegen matches target {target_host}")
    return matches[0]
