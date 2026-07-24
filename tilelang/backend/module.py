from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .device_codegen import DeviceCodegen
    from .execution_backend import ExecutionBackendSpec
    from .host_codegen import HostCodegen, HostCodegenHook


TargetKinds = str | Iterable[str] | None
_BACKEND_MODULES: list[BackendModule] = []


def _normalize_target_kinds(target_kinds: TargetKinds, default: str) -> tuple[str, ...]:
    if target_kinds is None:
        kinds = (default,)
    elif isinstance(target_kinds, str):
        kinds = (target_kinds,)
    else:
        kinds = tuple(target_kinds)

    if not kinds:
        raise ValueError("BackendModule must register at least one target kind")
    if any(not kind for kind in kinds):
        raise ValueError("target kind must not be empty")
    return kinds


@dataclass(slots=True, init=False)
class BackendModule:
    """Small manifest for lazy backend registration."""

    name: str
    target_kinds: tuple[str, ...]
    package: str
    execution_backend_module: str | None
    device_codegen_module: str | None
    host_codegen_module: str | None
    host_codegen_hooks_module: str | None
    _loaded_execution_backend_target_kinds: set[str]
    _loaded_device_codegen_target_kinds: set[str]
    _loaded_host_codegen_target_kinds: set[str]
    _loaded_host_codegen_hooks_target_kinds: set[str]

    def __init__(self, name: str, target_kinds: TargetKinds = None, *, package: str | None = None) -> None:
        if not name:
            raise ValueError("BackendModule name must not be empty")
        self.name = name
        self.target_kinds = _normalize_target_kinds(target_kinds, name)
        self.package = package or f"tilelang.{name}"
        self.execution_backend_module = None
        self.device_codegen_module = None
        self.host_codegen_module = None
        self.host_codegen_hooks_module = None
        self._loaded_execution_backend_target_kinds = set()
        self._loaded_device_codegen_target_kinds = set()
        self._loaded_host_codegen_target_kinds = set()
        self._loaded_host_codegen_hooks_target_kinds = set()

    def register_execution_backend(self, module: str | None = None) -> BackendModule:
        self.execution_backend_module = module or f"{self.package}.execution_backend"
        self._loaded_execution_backend_target_kinds.clear()
        _register_backend_module(self)
        return self

    def register_device_codegen(self, module: str | None = None) -> BackendModule:
        self.device_codegen_module = module or f"{self.package}.codegen"
        self._loaded_device_codegen_target_kinds.clear()
        _register_backend_module(self)
        return self

    def register_host_module(self, module: str | None = None) -> BackendModule:
        self.host_codegen_module = module or f"{self.package}.codegen"
        self._loaded_host_codegen_target_kinds.clear()
        _register_backend_module(self)
        return self

    def register_host_hooks_module(self, module: str | None = None) -> BackendModule:
        self.host_codegen_hooks_module = module or f"{self.package}.codegen"
        self._loaded_host_codegen_hooks_target_kinds.clear()
        _register_backend_module(self)
        return self

    def add_execution_backend(
        self,
        target_kind: str,
        spec: ExecutionBackendSpec,
        *,
        override: bool = False,
    ) -> ExecutionBackendSpec:
        from .execution_backend import _EXECUTION_BACKENDS

        specs = _EXECUTION_BACKENDS.setdefault(target_kind, [])
        if override:
            specs[:] = [item for item in specs if item.name != spec.name]
        elif any(item.name == spec.name for item in specs):
            raise ValueError(f"Execution backend {spec.name!r} is already registered for target kind {target_kind!r}")
        specs.append(spec)
        return spec

    def add_device_codegen(self, target_kind: str, codegen: DeviceCodegen, *, override: bool = False) -> DeviceCodegen:
        from .device_codegen import _DEVICE_CODEGENS

        codegens = _DEVICE_CODEGENS.setdefault(target_kind, [])
        if override:
            codegens[:] = [item for item in codegens if item.name != codegen.name]
        elif any(item.name == codegen.name for item in codegens):
            raise ValueError(f"Device codegen {codegen.name!r} is already registered for target kind {target_kind!r}")
        codegens.append(codegen)
        return codegen

    def add_host_codegen(self, target_host_kind: str, codegen: HostCodegen, *, override: bool = False) -> HostCodegen:
        from .host_codegen import _HOST_CODEGENS

        codegens = _HOST_CODEGENS.setdefault(target_host_kind, [])
        if override:
            codegens[:] = [item for item in codegens if item.name != codegen.name]
        elif any(item.name == codegen.name for item in codegens):
            raise ValueError(f"Host codegen {codegen.name!r} is already registered for target kind {target_host_kind!r}")
        codegens.append(codegen)
        return codegen

    def add_host_codegen_hook(self, target_kind: str, hook: HostCodegenHook, *, override: bool = False) -> HostCodegenHook:
        from .host_codegen import _HOST_CODEGEN_HOOKS

        hooks = _HOST_CODEGEN_HOOKS.setdefault(target_kind, [])
        if override:
            hooks[:] = [item for item in hooks if item.name != hook.name]
        elif any(item.name == hook.name for item in hooks):
            raise ValueError(f"Host codegen hook {hook.name!r} is already registered for target kind {target_kind!r}")
        hooks.append(hook)
        return hook


def _register_backend_module(backend_module: BackendModule) -> None:
    if not any(registered is backend_module for registered in _BACKEND_MODULES):
        _BACKEND_MODULES.append(backend_module)


def ensure_execution_backend_loaded(target_kind: str) -> None:
    _ensure_loaded(
        target_kind,
        module_attr="execution_backend_module",
        loaded_attr="_loaded_execution_backend_target_kinds",
    )


def ensure_device_codegen_loaded(target_kind: str) -> None:
    _ensure_loaded(
        target_kind,
        module_attr="device_codegen_module",
        loaded_attr="_loaded_device_codegen_target_kinds",
    )


def ensure_host_codegen_loaded(target_host_kind: str) -> None:
    _ensure_loaded(
        target_host_kind,
        module_attr="host_codegen_module",
        loaded_attr="_loaded_host_codegen_target_kinds",
    )


def ensure_host_codegen_hooks_loaded(target_kind: str) -> None:
    _ensure_loaded(
        target_kind,
        module_attr="host_codegen_hooks_module",
        loaded_attr="_loaded_host_codegen_hooks_target_kinds",
    )


def _ensure_loaded(target_kind: str, *, module_attr: str, loaded_attr: str) -> None:
    backend_module = _find_backend_module(target_kind, module_attr)
    if backend_module is None:
        return

    loaded_target_kinds = getattr(backend_module, loaded_attr)
    if target_kind in loaded_target_kinds:
        return

    import_module(getattr(backend_module, module_attr))
    loaded_target_kinds.add(target_kind)


def _find_backend_module(target_kind: str, module_attr: str) -> BackendModule | None:
    for backend_module in reversed(_BACKEND_MODULES):
        if target_kind in backend_module.target_kinds and getattr(backend_module, module_attr) is not None:
            return backend_module
    return None
