from tvm.tir import (
    PyStmtExprMutator,
    functor,
    Evaluate,
    AttrStmt,
)
from tvm.tir.transform import prim_func_pass


@functor.mutator
class MarkHostMetalContextMutator(PyStmtExprMutator):
    is_in_compute_scope = False

    def visit_attr_stmt_(self, stmt):
        switch = stmt.attr_key == "compute_scope"
        old_value = False
        if switch:
            old_value, self.is_in_compute_scope = self.is_in_compute_scope, True
        super().visit_attr_stmt_(stmt)
        if switch:
            self.is_in_compute_scope = old_value
        return stmt

    def visit_evaluate_(self, op: Evaluate):
        if self.is_in_compute_scope and op.value.op.name == "tir.tvm_call_packed_lowered":
            return AttrStmt(0, "metal_context", None, op)
        return op


def MarkHostMetalContext():
    def pass_fn(func, mod, ctx):
        mutator = MarkHostMetalContextMutator()
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
