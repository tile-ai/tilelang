"""Python statement/expression functors using TIRx's structural hooks."""

from __future__ import annotations

import re

import tvm_ffi
from tvm_tirx import ir


def _method_name(node_type: type) -> str:
    name = node_type.__tvm_ffi_type_info__.type_key.rsplit(".", 1)[-1]
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    return f"visit_{name.lower()}_"


class PyStmtExprVisitor:
    """Override ``visit_for_``, ``visit_tensor_load_``, and other node hooks.

    A hook owns traversal of its node. Calling ``super().visit_for_(node)``
    visits its children through the same visitor and definition-region state.
    """

    def __init__(self):
        self._engines = []

    def _dispatch(self, node, engine):
        self._engines.append(engine)
        try:
            method = getattr(self, _method_name(type(node)), self._visit_default)
            return method(node)
        finally:
            self._engines.pop()

    def _visit_default(self, node):
        return self._engines[-1].default_visit(node)

    def visit(self, node):
        if self._engines:
            return self._engines[-1].visit(node)
        return tvm_ffi.structural_visit(node, ((ir.Expr, ir.Stmt), self._dispatch))

    visit_stmt = visit
    visit_expr = visit


class PyStmtExprMutator:
    """Functional node-hook mutator with native binding and COW semantics."""

    def __init__(self):
        self._engines = []

    def _dispatch(self, node, engine):
        self._engines.append(engine)
        try:
            method = getattr(self, _method_name(type(node)), self._visit_default)
            result = method(node)
            expected = ir.Stmt if isinstance(node, ir.Stmt) else ir.Expr
            if not isinstance(result, expected):
                raise TypeError(f"{_method_name(type(node))} must return {expected.__name__}")
            return result
        finally:
            self._engines.pop()

    def _visit_default(self, node):
        return self._engines[-1].default_mutate(node)

    def visit(self, node):
        if self._engines:
            return self._engines[-1].mutate(node)
        return tvm_ffi.structural_mutate(node, ((ir.Expr, ir.Stmt), self._dispatch))

    visit_stmt = visit
    visit_expr = visit


def visitor(cls):
    if not issubclass(cls, PyStmtExprVisitor):
        raise TypeError("visitor classes must inherit PyStmtExprVisitor")
    return cls


def mutator(cls):
    if not issubclass(cls, PyStmtExprMutator):
        raise TypeError("mutator classes must inherit PyStmtExprMutator")
    return cls


for _namespace in (ir, ir.prim):
    for _node_type in vars(_namespace).values():
        if isinstance(_node_type, type) and issubclass(_node_type, (ir.Expr, ir.Stmt)):
            _name = _method_name(_node_type)
            setattr(PyStmtExprVisitor, _name, PyStmtExprVisitor._visit_default)
            setattr(PyStmtExprMutator, _name, PyStmtExprMutator._visit_default)

del _namespace, _node_type, _name
