"""Read-only lexical traversal of the native IR used by semantic checks."""

from __future__ import annotations

from contextlib import contextmanager

import tvm_ffi
from tvm_tirx import ir, tirx


def tensor_scope(tensor):
    """Read storage scope from the actual tensor type, including tensor expressions."""
    return str(tensor.ty.scope) if isinstance(tensor.ty, tirx.TensorType) else None


def tilelang_metadata(node):
    """Read lexical metadata while retaining native TensorVar/Layout identities."""
    if isinstance(node, ir.RegionStmt) and node.op.name == "tl.attr":
        return tvm_ffi.get_global_func("tl.DecodeAnnotations")(node.args[0], node.attrs["annotation_schema"])
    if (
        isinstance(node, ir.Call)
        and isinstance(node.op, ir.Op)
        and (node.op.same_as(ir.Op.get("tl.layout_map")) or node.op.same_as(ir.Op.get("tl.annotation")))
    ):
        return tvm_ffi.get_global_func("tl.ReadAnnotations")(node)
    return None


def single_loop(body):
    """Find the sole loop behind normalized SeqStmt and loop-metadata wrappers.

    A Bind, an executable statement or a general RegionStmt breaks a strict
    consecutive nest. Only the metadata wrapper introduced by LoopFrame is
    transparent, corresponding to annotations on the original For node.
    """
    while isinstance(body, ir.SeqStmt) and len(body.seq) == 1:
        body = body.seq[0]
        if isinstance(body, ir.RegionStmt):
            metadata = tilelang_metadata(body)
            if metadata is None or metadata.get("key") != "tl.loop.annotations":
                return None
            body = body.body
    return body if isinstance(body, ir.For) else None


class NativeStmtVisitor:
    """Traverse actual shared IR with explicit lexical Bind and loop scopes.

    The FFI structural visitor handles ordinary expression/type descent. The
    small set of lexical statement callbacks here preserves definition scope
    and expression evaluation order. It does not mutate the input program.
    """

    def __init__(self):
        self.bindings = {}
        self.loop_stack = []

    def visit_stmt(self, statement):
        if statement is None:
            return None
        return tvm_ffi.structural_visit(
            statement,
            [
                (ir.SeqStmt, self.visit_seq),
                (ir.For, self.visit_for),
                (ir.If, self.visit_if),
                (ir.While, self.visit_while),
                (ir.RegionStmt, self.visit_region),
                (ir.Bind, self.visit_bind),
                (ir.TensorStore, self.visit_tensor_store),
                (ir.TensorLoad, self.visit_tensor_load),
                (ir.Call, self.visit_call),
            ],
        )

    @contextmanager
    def scope(self):
        enclosing = self.bindings
        self.bindings = enclosing.copy()
        try:
            yield
        finally:
            self.bindings = enclosing

    def uses_var(self, expression, target):
        """Follow scalar Bind aliases when checking an index's loop dependency."""
        pending = []
        tvm_ffi.structural_walk(expression, (ir.Var, pending.append), order="pre")
        seen = set()
        while pending:
            variable = pending.pop()
            if variable.same_as(target):
                return True
            if variable in seen:
                continue
            seen.add(variable)
            bound_value = self.bindings.get(variable)
            if bound_value is not None:
                tvm_ffi.structural_walk(bound_value, (ir.Var, pending.append), order="pre")
        return False

    def visit_seq(self, node, visitor):
        with self.scope():
            for statement in node.seq:
                visitor.visit(statement)

    def visit_for(self, node, visitor):
        for value in (node.begin, node.end, node.step, node.annotations):
            visitor.visit(value)
        self.loop_stack.append(node)
        try:
            with self.scope():
                self.bindings.pop(node.var, None)
                visitor.visit(node.body)
        finally:
            self.loop_stack.pop()

    def visit_if(self, node, visitor):
        visitor.visit(node.condition)
        visitor.visit(node.then_case)
        if node.else_case is not None:
            visitor.visit(node.else_case)

    def visit_while(self, node, visitor):
        visitor.visit(node.condition)
        with self.scope():
            visitor.visit(node.body)

    def visit_region(self, node, visitor):
        visitor.visit(node.args)
        visitor.visit(node.attrs)
        with self.scope():
            for parameter in node.body_params:
                self.bindings.pop(parameter, None)
            visitor.visit(node.body)
        for result in node.result_vars:
            self.bindings.pop(result, None)

    def visit_bind(self, node, visitor):
        visitor.visit(node.value)
        self.bindings[node.var] = node.value

    def visit_tensor_load(self, node, visitor):
        visitor.default_visit(node)

    def visit_tensor_store(self, node, visitor):
        visitor.default_visit(node)

    def visit_call(self, node, visitor):
        visitor.default_visit(node)
