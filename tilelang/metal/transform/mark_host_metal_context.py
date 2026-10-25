"""Mark native host launch calls with the Metal command-buffer context."""

from tvm_tirx import ir
from tvm_tirx.tirx.transform import function_pass

from tilelang._compat.functor import PyStmtExprMutator


class _MarkHostMetalContextMutator(PyStmtExprMutator):
    def __init__(self):
        super().__init__()
        self.in_compute_scope = False

    def visit_region_stmt_(self, statement):
        previous = self.in_compute_scope
        self.in_compute_scope |= statement.op.name == "tirx.compute_scope"
        try:
            return super().visit_region_stmt_(statement)
        finally:
            self.in_compute_scope = previous

    def visit_evaluate_(self, statement):
        value = statement.value
        if (
            self.in_compute_scope
            and isinstance(value, ir.Call)
            and isinstance(value.op, ir.Op)
            and value.op.name == "tirx.call_packed_lowered"
        ):
            return ir.RegionStmt(
                "tirx.scope",
                [],
                [],
                {"metal_context": True},
                ir.SeqStmt([statement]),
                loc=statement.loc,
            )
        return statement


def MarkHostMetalContext():
    def transform(function, module, context):
        return function.with_body(_MarkHostMetalContextMutator().visit(function.body))

    return function_pass(transform, opt_level=0, name="tl.MarkHostMetalContext")


__all__ = ["MarkHostMetalContext"]
