"""WEBGPU namespace; compiler registration is loaded by the backend pipeline."""

from __future__ import annotations

import importlib

_MODULES = ("backend", "language")
_ATTRIBUTES = {}
__all__ = (*_MODULES, *_ATTRIBUTES)


def __getattr__(name):
    if name in _MODULES:
        value = importlib.import_module(f".{name}", __name__)
    elif name in _ATTRIBUTES:
        value = getattr(importlib.import_module(f".{_ATTRIBUTES[name]}", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


def register_backend():
    """Load the implementation registries before target lowering or dispatch."""
    for name in _MODULES:
        if name != "language":
            __getattr__(name)
