"""Preserve TileLang's source-span API using native TIRx locations."""

from __future__ import annotations

from functools import cache

import tvm_ffi
from tvm_tirx import ir, tirx

from .stmt_functor import post_order_visit
from .tensor import TensorView


@cache
def _span_ffi(name):
    return tvm_ffi.get_global_func(f"tl.ir.{name}")


def make_span(file: str, line: int) -> ir.SourceLoc:
    return ir.SourceLoc(ir.SourceName(file), line, 1, line, 1 << 20)


def span_to_location(span: ir.Location | None) -> tuple[str, int] | None:
    if isinstance(span, ir.SourceLoc):
        return span.source_name.name, span.start_line
    if isinstance(span, ir.CallSiteLoc):
        return span_to_location(span.callee) or span_to_location(span.caller)
    return None


def _known_location(span):
    return span if span_to_location(span) is not None else None


def set_stmt_span(stmt, span: ir.Location | None) -> None:
    _span_ffi("SetStmtSpan")(stmt, span)


def get_stmt_span(stmt) -> ir.Location | None:
    return _known_location(_span_ffi("GetStmtSpan")(stmt))


def set_buffer_span(buffer, span: ir.Location | None) -> None:
    _span_ffi("SetBufferSpan")(buffer.tensor if isinstance(buffer, TensorView) else buffer, span)


def get_buffer_span(buffer) -> ir.Location | None:
    return _known_location(_span_ffi("GetBufferSpan")(buffer.tensor if isinstance(buffer, TensorView) else buffer))


def set_prim_func_span(func, span: ir.Location | None) -> None:
    _span_ffi("SetPrimFuncSpan")(func, span)


def get_prim_func_span(func) -> ir.Location | None:
    return _known_location(_span_ffi("GetPrimFuncSpan")(func))


def stamp_stmt_spans(stmt, span: ir.Location | None) -> None:
    _span_ffi("StampSubtreeSpans")(stmt, span)


def span_coverage(func: tirx.Function) -> dict:
    statements = [0, 0]
    tensors = [0, 0]
    seen_tensors = set()

    def count_tensor(tensor):
        identity = tensor.__chandle__()
        if identity not in seen_tensors:
            seen_tensors.add(identity)
            tensors[1] += 1
            tensors[0] += get_buffer_span(tensor) is not None

    def visit(node):
        if isinstance(node, ir.Stmt):
            statements[1] += 1
            statements[0] += get_stmt_span(node) is not None
        elif isinstance(node, ir.Var) and tirx.is_tensor_var(node):
            count_tensor(node)

    if func.body is not None:
        post_order_visit(func.body, visit)
    for parameter in func.params:
        if tirx.is_tensor_var(parameter):
            count_tensor(parameter)
    return {"stmts": statements, "buffers": tensors}
