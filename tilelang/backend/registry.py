from __future__ import annotations

from importlib import import_module

from tvm.target import Target

from .backend import Backend

_BACKENDS: dict[str, Backend] = {}
_TARGET_INDEX: dict[str, list[str]] = {}
_LAZY_IMPORTS: dict[str, str] = {}


def register_backend(backend: Backend, *, override: bool = False) -> Backend:
    """Register a TileLang backend descriptor."""

    if backend.name in _BACKENDS and not override:
        raise ValueError(f"Backend {backend.name!r} is already registered")

    if backend.name in _BACKENDS:
        old = _BACKENDS[backend.name]
        for kind in old.target_kinds:
            names = _TARGET_INDEX.get(kind, [])
            if old.name in names:
                names.remove(old.name)

    _BACKENDS[backend.name] = backend
    for kind in backend.target_kinds:
        _TARGET_INDEX.setdefault(kind, []).append(backend.name)
    return backend


def register_lazy_backend(target_kind: str, import_path: str) -> None:
    """Register an import path that can populate a target kind on demand."""

    _LAZY_IMPORTS[target_kind] = import_path


def get_backend(name: str) -> Backend:
    if name not in _BACKENDS:
        raise ValueError(f"No backend registered with name {name!r}")
    return _BACKENDS[name]


def list_backends() -> dict[str, Backend]:
    return dict(_BACKENDS)


def _ensure_loaded_for_kind(target_kind: str) -> None:
    if target_kind in _TARGET_INDEX:
        return
    import_path = _LAZY_IMPORTS.get(target_kind)
    if import_path is not None:
        import_module(import_path)


def resolve_backend(target: Target) -> Backend:
    """Resolve exactly one backend for a TVM target."""

    kind = target.kind.name
    _ensure_loaded_for_kind(kind)

    candidates = [_BACKENDS[name] for name in _TARGET_INDEX.get(kind, []) if _BACKENDS[name].matches(target)]
    if not candidates:
        available = ", ".join(sorted(_BACKENDS)) or "<none>"
        raise ValueError(f"No TileLang backend registered for target {target}. Registered backends: {available}")

    candidates.sort(key=lambda backend: backend.priority, reverse=True)
    if len(candidates) > 1 and candidates[0].priority == candidates[1].priority:
        names = ", ".join(backend.name for backend in candidates)
        raise ValueError(f"Multiple TileLang backends match target {target}: {names}")

    backend = candidates[0]
    if not backend.is_available():
        raise ValueError(f"TileLang backend {backend.name!r} is registered but unavailable")
    backend.ensure_callbacks_registered()
    return backend
