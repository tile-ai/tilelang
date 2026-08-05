from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tvm.target import Target

TargetPredicate = Callable[[Target], bool]
AvailabilityCheck = Callable[[], bool]


def _always_available() -> bool:
    return True


def canonicalize_execution_backend(name: str | None) -> str | None:
    if name is None:
        return None
    return str(name).lower()


@dataclass(frozen=True, slots=True)
class ExecutionBackendSpec:
    name: str
    is_available: AvailabilityCheck = _always_available
    supports_target: TargetPredicate | None = None
    enable_host_codegen: bool = False
    enable_device_compile: bool = False

    def matches(self, target: Target) -> bool:
        return True if self.supports_target is None else self.supports_target(target)


def allowed_backends_for_target(target: Target, *, include_unavailable: bool = True) -> list[str]:
    """Compatibility lookup; core compilation uses BackendModule directly."""
    from tilelang.backend.module import resolve_backend

    backend = resolve_backend(target)
    return list(backend.allowed_execution_backends(target, include_unavailable=include_unavailable))


def resolve_execution_backend(requested: str | None, target: Target) -> str:
    return resolve_execution_backend_spec(requested, target).name


def resolve_execution_backend_spec(requested: str | None, target: Target) -> ExecutionBackendSpec:
    """Compatibility lookup; core compilation uses BackendModule directly."""
    from tilelang.backend.module import resolve_backend

    backend = resolve_backend(target)
    return backend.resolve_execution_backend(requested, target)
