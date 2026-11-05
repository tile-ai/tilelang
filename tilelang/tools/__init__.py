from .plot_layout import plot_layout  # noqa: F401

__all__ = ("plot_layout", "Analyzer", "AnalysisResult", "ARCH_CONFIGS")


def __getattr__(name):
    """Lazily import heavy tool submodules on first attribute access.

    ``tilelang.tools.lower_trace`` is a large debug-only module, so it is
    imported lazily (only when accessed) to keep ``import tilelang`` cheap
    when the feature is off. Other attributes raise ``AttributeError`` as usual.

    ``importlib.import_module`` is used rather than ``from . import lower_trace``:
    the latter re-enters this ``__getattr__`` while resolving the fromlist and
    recurses infinitely.
    """
    if name in {"Analyzer", "AnalysisResult", "ARCH_CONFIGS"}:
        import importlib

        module = importlib.import_module(".Analyzer", __name__)
        for exported in ("Analyzer", "AnalysisResult", "ARCH_CONFIGS"):
            globals()[exported] = getattr(module, exported)
        return globals()[name]
    if name == "lower_trace":
        import importlib

        return importlib.import_module(".lower_trace", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
