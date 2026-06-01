"""Mark host-side Metal kernel calls for MPS synchronization."""

from tvm import tirx as tir
from tvm.ir import Op
from tvm.tirx import AttrStmt, Evaluate, PyStmtExprMutator, functor
from tvm.tirx.transform import prim_func_pass


_tvm_call_packed_lowered = Op.get("tirx.tvm_call_packed_lowered")


@functor.mutator
class _MarkHostMetalContextMutator(PyStmtExprMutator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_in_compute_scope = False

    def visit_attr_stmt_(self, stmt):
        switch = stmt.attr_key == "compute_scope"
        old_value = False
        if switch:
            assert not self.is_in_compute_scope
            old_value, self.is_in_compute_scope = self.is_in_compute_scope, True
        s = self.visit_stmt(stmt.body)
        if switch:
            self.is_in_compute_scope = old_value
        return s

    def visit_evaluate_(self, op: Evaluate):
        if self.is_in_compute_scope and isinstance(op.value, tir.Call) and op.value.op.same_as(_tvm_call_packed_lowered):
            return AttrStmt(0, "metal_context", "", op)
        return op


def MarkHostMetalContext():
    def pass_fn(func, mod, ctx):
        mutator = _MarkHostMetalContextMutator()
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)


__all__ = ["MarkHostMetalContext"]
