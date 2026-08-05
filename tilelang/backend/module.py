from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeVar

from tvm import IRModule
from tvm.target import Target

if TYPE_CHECKING:
    from tilelang.backend.device_codegen import DeviceCodegen
    from tilelang.backend.execution_backend import ExecutionBackendSpec
    from tilelang.backend.host_codegen import HostCodegen, HostCodegenHook
    from tilelang.backend.pass_pipeline import PassPipeline

BackendCallback = Callable[..., object]
TargetPredicate = Callable[[Target], bool]
_T = TypeVar("_T")


def _freeze_components(components: Mapping[str, tuple[_T, ...]]) -> Mapping[str, tuple[_T, ...]]:
    return MappingProxyType({target_kind: tuple(values) for target_kind, values in components.items()})


@dataclass(frozen=True, slots=True)
class BackendModule:
    """Complete Python registration manifest for one TileLang backend."""

    name: str
    target_kinds: tuple[str, ...]
    pipelines: Mapping[str, PassPipeline]
    device_codegens: Mapping[str, tuple[DeviceCodegen, ...]]
    execution_backends: tuple[ExecutionBackendSpec, ...]
    supports_target: TargetPredicate | None = None
    host_codegens: Mapping[str, tuple[HostCodegen, ...]] = field(default_factory=dict)
    host_codegen_hooks: Mapping[str, tuple[HostCodegenHook, ...]] = field(default_factory=dict)
    callbacks: Mapping[str, BackendCallback] = field(default_factory=dict)

    def __post_init__(self) -> None:
        target_kinds = tuple(self.target_kinds)
        if not self.name:
            raise ValueError("BackendModule.name must not be empty")
        if not target_kinds or any(not kind for kind in target_kinds):
            raise ValueError(f"BackendModule {self.name!r} must own at least one non-empty target kind")
        if len(set(target_kinds)) != len(target_kinds):
            raise ValueError(f"BackendModule {self.name!r} target kinds must be unique")

        target_kind_set = set(target_kinds)
        pipelines = MappingProxyType(dict(self.pipelines))
        if set(pipelines) != target_kind_set:
            raise ValueError(f"BackendModule {self.name!r} must define exactly one pipeline for every target kind")
        for target_kind, pipeline in pipelines.items():
            if pipeline.name != target_kind:
                raise ValueError(f"BackendModule {self.name!r} pipeline {pipeline.name!r} does not match target kind {target_kind!r}")

        device_codegens = _freeze_components(self.device_codegens)
        if set(device_codegens) != target_kind_set:
            raise ValueError(f"BackendModule {self.name!r} must define device codegen for every target kind")
        if any(not codegens for codegens in device_codegens.values()):
            raise ValueError(f"BackendModule {self.name!r} device codegen lists must not be empty")

        host_codegens = _freeze_components(self.host_codegens)
        host_codegen_hooks = _freeze_components(self.host_codegen_hooks)
        if any(not values for values in host_codegens.values()):
            raise ValueError(f"BackendModule {self.name!r} host codegen lists must not be empty")
        unknown_hook_targets = set(host_codegen_hooks) - target_kind_set
        if unknown_hook_targets:
            raise ValueError(
                f"BackendModule {self.name!r} host codegen hook targets are not owned by this backend: {sorted(unknown_hook_targets)}"
            )
        if any(not values for values in host_codegen_hooks.values()):
            raise ValueError(f"BackendModule {self.name!r} host codegen hook lists must not be empty")

        execution_backends = tuple(self.execution_backends)
        execution_names = [spec.name for spec in execution_backends]
        if not execution_backends:
            raise ValueError(f"BackendModule {self.name!r} must define at least one execution backend")
        if len(set(execution_names)) != len(execution_names):
            raise ValueError(f"BackendModule {self.name!r} execution backend names must be unique: {execution_names}")
        if any(spec.enable_host_codegen for spec in execution_backends) and not host_codegens:
            raise ValueError(f"BackendModule {self.name!r} enables host codegen but defines no host codegen targets")

        callbacks = MappingProxyType(dict(self.callbacks))
        if any(not name for name in callbacks):
            raise ValueError(f"BackendModule {self.name!r} callback names must not be empty")

        object.__setattr__(self, "target_kinds", target_kinds)
        object.__setattr__(self, "pipelines", pipelines)
        object.__setattr__(self, "device_codegens", device_codegens)
        object.__setattr__(self, "execution_backends", execution_backends)
        object.__setattr__(self, "host_codegens", host_codegens)
        object.__setattr__(self, "host_codegen_hooks", host_codegen_hooks)
        object.__setattr__(self, "callbacks", callbacks)

    def matches(self, target: Target) -> bool:
        if target.kind.name not in self.target_kinds:
            return False
        return self.supports_target(target) if self.supports_target is not None else True

    def _require_target(self, target: Target) -> str:
        target_kind = target.kind.name
        if not self.matches(target):
            raise ValueError(f"Backend {self.name!r} does not match target {target}")
        return target_kind

    def get_pipeline(self, target: Target) -> PassPipeline:
        return self.pipelines[self._require_target(target)]

    def lower(self, mod: IRModule, target: Target) -> IRModule:
        return self.get_pipeline(target).lower(mod, target)

    def get_device_codegen(self, target: Target) -> DeviceCodegen:
        target_kind = self._require_target(target)
        matches = [codegen for codegen in self.device_codegens[target_kind] if codegen.matches(target)]
        if not matches:
            raise ValueError(f"Backend {self.name!r} has no device codegen matching target {target}")
        return matches[0]

    def codegen_device(self, mod: IRModule, target: Target, *, compile_device: bool) -> IRModule:
        return self.get_device_codegen(target).lower(mod, target, compile_device=compile_device)

    def get_host_codegen(self, target_host: Target) -> HostCodegen:
        target_kind = target_host.kind.name
        matches = [codegen for codegen in self.host_codegens.get(target_kind, ()) if codegen.matches(target_host)]
        if not matches:
            raise ValueError(f"Backend {self.name!r} has no host codegen matching target {target_host}")
        return matches[0]

    def codegen_host(self, mod: IRModule, target_host: Target) -> IRModule:
        return self.get_host_codegen(target_host).lower(mod, target_host)

    def preprocess_host_codegen(self, mod: IRModule, target_host: Target, target: Target) -> IRModule:
        target_kind = self._require_target(target)
        for hook in self.host_codegen_hooks.get(target_kind, ()):
            if hook.matches(target):
                mod = hook.lower(mod, target_host, target)
        return mod

    def allowed_execution_backends(self, target: Target, *, include_unavailable: bool = True) -> tuple[str, ...]:
        self._require_target(target)
        specs = [spec for spec in self.execution_backends if spec.matches(target)]
        if not include_unavailable:
            specs = [spec for spec in specs if spec.is_available()]
        return tuple(spec.name for spec in specs)

    def resolve_execution_backend(self, requested: str | None, target: Target) -> ExecutionBackendSpec:
        from tilelang.backend.execution_backend import canonicalize_execution_backend

        self._require_target(target)
        requested_name = canonicalize_execution_backend(requested)
        all_specs = [spec for spec in self.execution_backends if spec.matches(target)]
        available_specs = [spec for spec in all_specs if spec.is_available()]

        if requested_name in (None, "auto"):
            if not available_specs:
                allowed = ", ".join(spec.name for spec in all_specs) or "<none>"
                raise ValueError(f"No available execution backend for target {target.kind.name!r}. Allowed: {allowed}.")
            return available_specs[0]

        spec = next((spec for spec in all_specs if spec.name == requested_name), None)
        if spec is None:
            allowed = ", ".join(spec.name for spec in all_specs) or "<none>"
            raise ValueError(
                f"Invalid execution backend {requested!r} for target {target.kind.name!r}. "
                f"Allowed: {allowed}. Tip: use execution_backend='auto'."
            )
        if not spec.is_available():
            available = ", ".join(spec.name for spec in available_specs) or "<none>"
            raise ValueError(
                f"Execution backend {requested!r} requires extra dependencies and is not available now. Try one of: {available}."
            )
        return spec


_BACKENDS: dict[str, BackendModule] = {}
_TARGET_KIND_INDEX: dict[str, list[str]] = {}


def register_backend(backend: BackendModule) -> BackendModule:
    """Validate and register every component declared by a backend manifest."""

    old = _BACKENDS.get(backend.name)
    if old is not None:
        if old == backend:
            return old
        raise ValueError(f"Backend {backend.name!r} is already registered with a different declaration")

    for target_kind in backend.target_kinds:
        for candidate_name in _TARGET_KIND_INDEX.get(target_kind, ()):
            candidate = _BACKENDS[candidate_name]
            if candidate.supports_target is None or backend.supports_target is None:
                raise ValueError(f"Backends sharing target kind {target_kind!r} must define supports_target predicates")

    _BACKENDS[backend.name] = backend
    for target_kind in backend.target_kinds:
        _TARGET_KIND_INDEX.setdefault(target_kind, []).append(backend.name)

    try:
        import tvm_ffi

        for name, callback in backend.callbacks.items():
            tvm_ffi.register_global_func(name, f=callback, override=True)
    except Exception:
        _BACKENDS.pop(backend.name, None)
        for target_kind in backend.target_kinds:
            names = _TARGET_KIND_INDEX.get(target_kind, [])
            if backend.name in names:
                names.remove(backend.name)
            if not names:
                _TARGET_KIND_INDEX.pop(target_kind, None)
        raise

    return backend


def get_backend(name: str) -> BackendModule:
    try:
        return _BACKENDS[name]
    except KeyError as err:
        available = ", ".join(sorted(_BACKENDS)) or "<none>"
        raise ValueError(f"Unknown backend {name!r}. Available: {available}") from err


def list_backends() -> dict[str, BackendModule]:
    return dict(_BACKENDS)


def list_backends_for_target_kind(target_kind: str) -> tuple[BackendModule, ...]:
    try:
        names = _TARGET_KIND_INDEX[target_kind]
    except KeyError as err:
        available = ", ".join(sorted(_TARGET_KIND_INDEX)) or "<none>"
        raise ValueError(f"No backend registered for target kind {target_kind!r}. Available: {available}") from err
    return tuple(_BACKENDS[name] for name in names)


def resolve_backend(target: Target) -> BackendModule:
    candidates = [backend for backend in list_backends_for_target_kind(target.kind.name) if backend.matches(target)]
    if not candidates:
        raise ValueError(f"No backend matches target {target}")
    if len(candidates) > 1:
        names = ", ".join(backend.name for backend in candidates)
        raise ValueError(f"Multiple backends match target {target}: {names}")
    return candidates[0]
