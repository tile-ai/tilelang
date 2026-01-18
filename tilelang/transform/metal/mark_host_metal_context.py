from tvm.ir import Op
from tvm.tir import (
    PyStmtExprMutator,
    functor,
    Evaluate,
    AttrStmt,
)
from tvm.tir.transform import prim_func_pass

tvm_call_packed_lowered = Op.get("tir.tvm_call_packed_lowered")


@functor.mutator
class MarkHostMetalContextMutator(PyStmtExprMutator):
    is_in_compute_scope = False

    def visit_attr_stmt_(self, stmt):
        switch = stmt.attr_key == "compute_scope"
        old_value = False
        if switch:
            old_value, self.is_in_compute_scope = self.is_in_compute_scope, True
        s = self.visit_stmt(stmt.body)
        if switch:
            self.is_in_compute_scope = old_value
        return s

    def visit_evaluate_(self, op: Evaluate):
        if self.is_in_compute_scope and op.value.op.same_as(tvm_call_packed_lowered):
            return AttrStmt(0, "metal_context", "", op)
        return op


def MarkHostMetalContext():
    def pass_fn(func, mod, ctx):
        mutator = MarkHostMetalContextMutator()
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
