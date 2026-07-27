from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from tvm.target import Target

TargetPredicate = Callable[[Target], bool]
AvailabilityCheck = Callable[[], bool]

_CANONICAL_EXECUTION_BACKENDS = {
    "dlpack": "tvm_ffi",
}


def _always_available() -> bool:
    return True


def canonicalize_execution_backend(name: str | None) -> str | None:
    if name is None:
        return None
    key = str(name).lower()
    return _CANONICAL_EXECUTION_BACKENDS.get(key, key)


@dataclass(frozen=True, slots=True)
class ExecutionBackendSpec:
    name: str
    is_available: AvailabilityCheck = _always_available
    supports_target: TargetPredicate | None = None
    enable_host_codegen: bool = False
    enable_device_compile: bool = False

    def matches(self, target: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target)


_EXECUTION_BACKENDS: dict[str, list[ExecutionBackendSpec]] = {}
_LAZY_EXECUTION_BACKENDS: dict[str, str] = {}
_LOADED_EXECUTION_BACKENDS: set[str] = set()


def register_execution_backend(
    target_kind: str,
    spec: ExecutionBackendSpec,
    *,
    override: bool = False,
) -> ExecutionBackendSpec:
    specs = _EXECUTION_BACKENDS.setdefault(target_kind, [])
    if override:
        specs[:] = [item for item in specs if item.name != spec.name]
    elif any(item.name == spec.name for item in specs):
        raise ValueError(f"Execution backend {spec.name!r} is already registered for target kind {target_kind!r}")
    specs.append(spec)
    return spec


def register_lazy_execution_backends(target_kind: str, import_path: str) -> None:
    _LAZY_EXECUTION_BACKENDS[target_kind] = import_path


def _ensure_execution_backends_loaded(target_kind: str) -> None:
    if target_kind in _LOADED_EXECUTION_BACKENDS:
        return
    import_path = _LAZY_EXECUTION_BACKENDS.get(target_kind)
    if import_path is not None:
        import_module(import_path)
    _LOADED_EXECUTION_BACKENDS.add(target_kind)


def _matching_specs(target: Target, *, include_unavailable: bool) -> list[ExecutionBackendSpec]:
    target_kind = target.kind.name
    _ensure_execution_backends_loaded(target_kind)
    specs = [spec for spec in _EXECUTION_BACKENDS.get(target_kind, ()) if spec.matches(target)]
    if not include_unavailable:
        specs = [spec for spec in specs if spec.is_available()]
    return specs


def allowed_backends_for_target(target: Target, *, include_unavailable: bool = True) -> list[str]:
    return [spec.name for spec in _matching_specs(target, include_unavailable=include_unavailable)]


def _format_options(options: list[str]) -> str:
    return ", ".join(options) if options else "<none>"


def resolve_execution_backend(requested: str | None, target: Target) -> str:
    return resolve_execution_backend_spec(requested, target).name


def resolve_execution_backend_spec(requested: str | None, target: Target) -> ExecutionBackendSpec:
    requested_name = canonicalize_execution_backend(requested)
    allowed_all_specs = _matching_specs(target, include_unavailable=True)
    allowed_available_specs = _matching_specs(target, include_unavailable=False)
    allowed_all = [spec.name for spec in allowed_all_specs]
    allowed_available = [spec.name for spec in allowed_available_specs]

    if requested_name in (None, "auto"):
        if not allowed_available_specs:
            raise ValueError(f"No available execution backend for target '{target.kind.name}'. Allowed: {_format_options(allowed_all)}.")
        return allowed_available_specs[0]

    if requested_name not in allowed_all:
        raise ValueError(
            f"Invalid execution backend '{requested}' for target '{target.kind.name}'. "
            f"Allowed: {_format_options(allowed_all)}. Tip: use execution_backend='auto'."
        )
    if requested_name not in allowed_available:
        raise ValueError(
            f"Execution backend '{requested}' requires extra dependencies and is not available now. "
            f"Try one of: {_format_options(allowed_available)}."
        )
    return next(spec for spec in allowed_available_specs if spec.name == requested_name)
