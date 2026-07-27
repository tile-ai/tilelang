"""Host codegen registry shared by backend packages."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

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


_HOST_CODEGENS: dict[str, list[HostCodegen]] = {}
_LAZY_HOST_CODEGENS: dict[str, str] = {}
_LOADED_HOST_CODEGENS: set[str] = set()

_HOST_CODEGEN_HOOKS: dict[str, list[HostCodegenHook]] = {}
_LAZY_HOST_CODEGEN_HOOKS: dict[str, str] = {}
_LOADED_HOST_CODEGEN_HOOKS: set[str] = set()


def register_host_codegen(
    target_host_kind: str,
    codegen: HostCodegen,
    *,
    override: bool = False,
) -> HostCodegen:
    """Register a host codegen entry for a host target kind."""

    codegens = _HOST_CODEGENS.setdefault(target_host_kind, [])
    if override:
        codegens[:] = [item for item in codegens if item.name != codegen.name]
    elif any(item.name == codegen.name for item in codegens):
        raise ValueError(f"Host codegen {codegen.name!r} is already registered for target kind {target_host_kind!r}")
    codegens.append(codegen)
    return codegen


def register_lazy_host_codegen(target_host_kind: str, import_path: str) -> None:
    """Register a backend module to import when a host target kind is first used."""

    _LAZY_HOST_CODEGENS[target_host_kind] = import_path
    _LOADED_HOST_CODEGENS.discard(target_host_kind)


def register_host_codegen_hook(
    target_kind: str,
    hook: HostCodegenHook,
    *,
    override: bool = False,
) -> HostCodegenHook:
    """Register a device-backend hook for host codegen preparation."""

    hooks = _HOST_CODEGEN_HOOKS.setdefault(target_kind, [])
    if override:
        hooks[:] = [item for item in hooks if item.name != hook.name]
    elif any(item.name == hook.name for item in hooks):
        raise ValueError(f"Host codegen hook {hook.name!r} is already registered for target kind {target_kind!r}")
    hooks.append(hook)
    return hook


def register_lazy_host_codegen_hooks(target_kind: str, import_path: str) -> None:
    """Register a backend module to import before applying host codegen hooks."""

    _LAZY_HOST_CODEGEN_HOOKS[target_kind] = import_path
    _LOADED_HOST_CODEGEN_HOOKS.discard(target_kind)


def _ensure_host_codegens_loaded(target_host_kind: str) -> None:
    if target_host_kind in _LOADED_HOST_CODEGENS:
        return
    import_path = _LAZY_HOST_CODEGENS.get(target_host_kind)
    if import_path is not None:
        import_module(import_path)
    _LOADED_HOST_CODEGENS.add(target_host_kind)


def _ensure_host_codegen_hooks_loaded(target_kind: str) -> None:
    if target_kind in _LOADED_HOST_CODEGEN_HOOKS:
        return
    import_path = _LAZY_HOST_CODEGEN_HOOKS.get(target_kind)
    if import_path is not None:
        import_module(import_path)
    _LOADED_HOST_CODEGEN_HOOKS.add(target_kind)


def _matching_host_codegens(target_host: Target) -> list[HostCodegen]:
    target_host_kind = target_host.kind.name
    _ensure_host_codegens_loaded(target_host_kind)
    return [codegen for codegen in _HOST_CODEGENS.get(target_host_kind, ()) if codegen.matches(target_host)]


def _matching_host_codegen_hooks(target: Target) -> list[HostCodegenHook]:
    target_kind = target.kind.name
    _ensure_host_codegen_hooks_loaded(target_kind)
    return [hook for hook in _HOST_CODEGEN_HOOKS.get(target_kind, ()) if hook.matches(target)]


def allowed_host_codegens_for_target(target_host: Target) -> list[str]:
    """Return matching host codegen names for a host target."""

    return [codegen.name for codegen in _matching_host_codegens(target_host)]


def _format_host_codegen_names(codegens: list[HostCodegen]) -> str:
    names = [codegen.name for codegen in codegens]
    return ", ".join(names) if names else "<none>"


def apply_host_codegen_hooks(mod: IRModule, target_host: Target, target: Target | None) -> IRModule:
    """Apply device-backend host codegen hooks."""

    if target is None:
        return mod
    for hook in _matching_host_codegen_hooks(target):
        mod = hook.lower(mod, target_host, target)
    return mod


def resolve_host_codegen(target_host: Target) -> HostCodegen:
    """Resolve a host codegen entry from a TVM host target."""

    matches = _matching_host_codegens(target_host)
    if not matches:
        target_host_kind = target_host.kind.name
        options = _format_host_codegen_names(_HOST_CODEGENS.get(target_host_kind, []))
        raise ValueError(f"No host codegen registered for target host '{target_host_kind}'. Available: {options}.")
    return matches[0]
