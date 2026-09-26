from __future__ import annotations

from typing import IO

from tilelang import tvm
from tilelang.ir import get_stmt_span
from tvm.tirx.stmt_functor import post_order_visit

__all__ = ["dump_tirx_with_span", "render_tirx_with_span"]

_STMT = tvm.tirx.Stmt
_PRIM_FUNC = tvm.tirx.PrimFunc


_SKIP_STMT_TYPES = frozenset({"SBlockRealize"})


def _location(span) -> str | None:
    """Canonical ``file:line:column`` for a defined span, else ``None``."""
    if span is None or span.source_name is None:
        return None
    return f"{span.source_name.name}:{span.line}:{max(span.column, 1)}"


def _collect_annotations(func) -> dict:
    """Map every real statement with a span to its ``file:line:column`` location."""
    annotate: dict = {}

    def visit(node):
        """Record the source location of each statement that carries a span."""
        if isinstance(node, _STMT) and type(node).__name__ not in _SKIP_STMT_TYPES:
            loc = _location(get_stmt_span(node))
            if loc is not None:
                annotate[node] = loc

    post_order_visit(func.body, visit)
    return annotate


def render_tirx_with_span(func_or_mod) -> str:
    """Return the official TVMScript of ``func_or_mod`` with spans appended.

    Parameters
    ----------
    func_or_mod : tvm.tirx.PrimFunc or tvm.IRModule
        The freshly-parsed PrimFunc, or an IRModule at any later lowering stage.
    """
    if isinstance(func_or_mod, tvm.IRModule):
        annotate: dict = {}
        for _gvar, func in func_or_mod.functions.items():
            if isinstance(func, _PRIM_FUNC):
                annotate.update(_collect_annotations(func))
        return func_or_mod.script(obj_to_annotate=annotate)
    return func_or_mod.script(obj_to_annotate=_collect_annotations(func_or_mod))


def dump_tirx_with_span(func_or_mod, file: IO[str] | None = None) -> None:
    """Print the span-annotated TVMScript of ``func_or_mod``."""
    print(render_tirx_with_span(func_or_mod), file=file)
