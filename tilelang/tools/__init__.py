from .plot_layout import plot_layout  # noqa: F401
from .Analyzer import *


def __getattr__(name):
    if name == "lower_trace":
        import importlib

        mod = importlib.import_module(".lower_trace", __name__)
        globals()["lower_trace"] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
