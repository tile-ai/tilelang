from .plot_layout import plot_layout  # noqa: F401
from .Analyzer import *


def __getattr__(name):
    """Lazily import heavy tool submodules on first attribute access.

    ``tilelang.tools.lower_trace`` monkey-patches TVM/TileLang lowering on
    import, so it is imported lazily (only when accessed) to keep
    ``import tilelang`` free of tracing side effects when the feature is off.
    Other attributes raise ``AttributeError`` as usual.
    """
    if name == "lower_trace":
        import importlib

        mod = importlib.import_module(".lower_trace", __name__)
        globals()["lower_trace"] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
