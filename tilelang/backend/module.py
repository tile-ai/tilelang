from __future__ import annotations

from dataclasses import dataclass

from tvm.target import Target


@dataclass(frozen=True, slots=True)
class BackendModule:
    """Identity and target ownership for one backend registration module."""

    name: str
    target_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        target_kinds = tuple(self.target_kinds)
        if not self.name:
            raise ValueError("BackendModule.name must not be empty")
        if not target_kinds or any(not kind for kind in target_kinds):
            raise ValueError(f"BackendModule {self.name!r} must own at least one non-empty target kind")
        if len(set(target_kinds)) != len(target_kinds):
            raise ValueError(f"BackendModule {self.name!r} target kinds must be unique")
        object.__setattr__(self, "target_kinds", target_kinds)

    def matches(self, target: Target) -> bool:
        return target.kind.name in self.target_kinds


_BACKEND_MODULES: dict[str, BackendModule] = {}
_TARGET_KIND_INDEX: dict[str, str] = {}


def register_backend_module(module: BackendModule) -> BackendModule:
    """Register the backend module that owns each target kind."""

    old = _BACKEND_MODULES.get(module.name)
    if old is not None:
        if old == module:
            return old
        raise ValueError(f"Backend module {module.name!r} is already registered with a different declaration")

    for kind in module.target_kinds:
        owner = _TARGET_KIND_INDEX.get(kind)
        if owner is not None and owner != module.name:
            raise ValueError(f"Target kind {kind!r} is already owned by backend module {owner!r}")

    _BACKEND_MODULES[module.name] = module
    for kind in module.target_kinds:
        _TARGET_KIND_INDEX[kind] = module.name
    return module


def get_backend_module(name: str) -> BackendModule:
    try:
        return _BACKEND_MODULES[name]
    except KeyError as err:
        available = ", ".join(sorted(_BACKEND_MODULES)) or "<none>"
        raise ValueError(f"Unknown backend module {name!r}. Available: {available}") from err


def list_backend_modules() -> dict[str, BackendModule]:
    return dict(_BACKEND_MODULES)


def get_backend_module_for_target_kind(target_kind: str) -> BackendModule:
    try:
        name = _TARGET_KIND_INDEX[target_kind]
    except KeyError as err:
        available = ", ".join(sorted(_TARGET_KIND_INDEX)) or "<none>"
        raise ValueError(f"No backend module registered for target kind {target_kind!r}. Available: {available}") from err
    return _BACKEND_MODULES[name]


def resolve_backend_module(target: Target) -> BackendModule:
    return get_backend_module_for_target_kind(target.kind.name)
