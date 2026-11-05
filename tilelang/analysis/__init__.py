"""Tilelang IR analysis & visitors."""

from .ast_printer import ASTPrinter  # noqa: F401
from .nested_loop_checker import NestedLoopChecker  # noqa: F401
from .fragment_loop_checker import FragmentLoopChecker  # noqa: F401
from .parallel_local_index_checker import ParallelLocalIndexChecker  # noqa: F401

__all__ = ("ASTPrinter", "NestedLoopChecker", "FragmentLoopChecker", "ParallelLocalIndexChecker", "LayoutVisual")


def __getattr__(name):
    if name == "LayoutVisual":
        from .layout_visual import LayoutVisual

        globals()[name] = LayoutVisual
        return LayoutVisual
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
