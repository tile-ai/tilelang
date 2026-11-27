from tvm import tirx
from tvm.tirx import (
    BufferStore,
    Broadcast,
    Var,
    PrimFunc,
    PyStmtExprMutator,
    SeqStmt,
)
from tvm.tirx.stmt import Bind
from tvm.tirx.transform import prim_func_pass

from tilelang.ir import get_stmt_span, stamp_stmt_spans


@tirx.functor.mutator
class HoistBroadcastValuesMutator(PyStmtExprMutator):
    def __init__(self):
        super().__init__()
        self.pending_defs = []
        self.hoist_enabled = False
        self.var_remap = {}

    def visit_var_(self, op):
        return self.var_remap.get(op, op)

    def visit_let_(self, op):
        # Hoisting can give repeated occurrences of a shared Let different
        # values. Freshen its binder to keep those bindings independent.
        value = self.visit_expr(op.value)
        if isinstance(op.var, tirx.SizeVar):
            var = tirx.SizeVar(op.var.name, op.var.dtype, op.var.span)
        else:
            var = Var(op.var.name, op.var.type_annotation, op.var.span)
        previous = self.var_remap.get(op.var)
        self.var_remap[op.var] = var
        try:
            body = self.visit_expr(op.body)
        finally:
            if previous is None:
                del self.var_remap[op.var]
            else:
                self.var_remap[op.var] = previous
        return tirx.Let(var, value, body, op.span)

    def visit_broadcast_(self, op):
        if self.hoist_enabled and isinstance(op.value, (tirx.IntImm, tirx.FloatImm)):
            val = self.visit_expr(op.value)
            new_var = Var("broadcast_var", dtype=val.dtype)
            self.pending_defs.append((new_var, val))
            return Broadcast(new_var, op.lanes)
        return Broadcast(self.visit_expr(op.value), self.visit_expr(op.lanes))

    def visit_buffer_store_(self, op: BufferStore):
        saved_hoist_enabled = self.hoist_enabled
        saved_pending_defs = self.pending_defs

        self.hoist_enabled = True
        self.pending_defs = []

        new_indices = [self.visit_expr(idx) for idx in op.indices]
        new_stmt = BufferStore(op.buffer, self.visit_expr(op.value), new_indices)

        if self.pending_defs:
            stmts = [Bind(var, val) for var, val in self.pending_defs]
            stmts.append(new_stmt)
            new_stmt = SeqStmt(stmts)

        self.hoist_enabled = saved_hoist_enabled
        self.pending_defs = saved_pending_defs

        stamp_stmt_spans(new_stmt, get_stmt_span(op))
        return new_stmt

    def visit_bind_(self, op: Bind):
        saved_hoist_enabled = self.hoist_enabled
        saved_pending_defs = self.pending_defs

        self.hoist_enabled = True
        self.pending_defs = []

        new_value = self.visit_expr(op.value)

        new_stmt = Bind(op.var, new_value)

        if self.pending_defs:
            stmts = [Bind(var, val) for var, val in self.pending_defs]
            stmts.append(new_stmt)
            new_stmt = SeqStmt(stmts)

        self.hoist_enabled = saved_hoist_enabled
        self.pending_defs = saved_pending_defs

        stamp_stmt_spans(new_stmt, get_stmt_span(op))
        return new_stmt


def HoistBroadcastValues():
    """
    TVM Pass: HoistBroadcastValues.

    This pass scans the TIR for Broadcast operations involving immediate constants (IntImm, FloatImm).
    It extracts these constants into variables defined via Bind immediately preceding
    the statement where the broadcast occurs.

    Example Transformation:
    -----------------------
    Before:
        A[i] = B[i] + T.Broadcast(3.14, 4) + T.Broadcast(3.14, 4)

    After:
        bv_3_14 = 3.14
        bv_3_14_1 = 3.14
        A[i] = B[i] + T.Broadcast(bv_3_14, 4) + T.Broadcast(bv_3_14_1, 4)
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        mutator = HoistBroadcastValuesMutator()
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body, span=func.span)

    return prim_func_pass(pass_fn, opt_level=0)
