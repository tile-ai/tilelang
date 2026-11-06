"""Default CUDA-flavoured TileLang language facade over native IR.

The public manifest remains complete. Backend helper classes are loaded when
looked up, so constructing ordinary DSL code does not import unrelated emitters.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING


def __getattr__(name):
    # Imports of DSL implementation modules must resolve
    # the real child module without initializing the default backend first.
    # The same rule permits helper modules to be imported directly.
    if name.isidentifier() and name != "__all__" and importlib.util.find_spec(f"{__name__}.{name}") is not None:
        return importlib.import_module(f".{name}", __name__)
    module = sys.modules.get("tilelang.cuda.language")
    common = sys.modules.get("tilelang.language.common")
    if any(candidate is not None and getattr(candidate.__spec__, "_initializing", False) for candidate in (module, common)):
        # Backend modules import language.common while initializing. Macro
        # annotations may look up T.Tensor in that interval. Do not eagerly
        # finish the backend, read its incomplete __all__, or cache a common
        # spelling that the completed CUDA dialect will override (e.g. Kernel).
        for candidate in (module, common):
            if candidate is not None and name in vars(candidate):
                return vars(candidate)[name]
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = importlib.import_module("tilelang.cuda.language")
    try:
        value = getattr(module, name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    globals()[name] = value
    return value


__tilelang_dialect__ = "cuda"


# The public callable shares a spelling with an implementation submodule.
# Importing that helper here creates a cycle through _typing when a layout or
# type consumer imports language first. Keep the helper lazy and reserve the
# public spelling when importlib later publishes the real child module.
from tilelang.language._allocation import alloc_buffer as allocate  # noqa: E402,F401


class _LanguageNamespace(ModuleType):
    def __setattr__(self, name, value):
        if name == "allocate" and isinstance(value, ModuleType) and value is sys.modules.get(f"{__name__}.allocate"):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _LanguageNamespace


def __dir__():
    return sorted(set(globals()) | set(__getattr__("__all__")))


if TYPE_CHECKING:
    from tilelang.cuda.language import *  # noqa: F401,F403
