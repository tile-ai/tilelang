"""Mark host-side Metal kernel calls for MPS synchronization."""

from tvm import tirx as tir
from tvm.ir import Op
from tvm.tirx import AttrStmt, Evaluate, PyStmtExprMutator, functor
from tvm.tirx.transform import prim_func_pass


"""
Transformation pass to mark host-side kernel calls for Metal/MPS synchronization.

To execute TVM-generated Metal kernels within a PyTorch environment, the TVM runtime
must utilize PyTorch's active Metal command buffer (MPS). This ensures correct
execution ordering and memory consistency between PyTorch operators and TVM kernels.

This pass identifies calls to `tir.tvm_call_packed_lowered` occurring within a
`compute_scope` and wraps them with a `metal_context` attribute. This attribute
signals the host implementation to provide runtime logic that:
1. Retrieves the current command buffer from `torch::mps`.
2. Passes this stream to the TVM runtime before the kernel executes.
"""


_tvm_call_packed_lowered = Op.get("tirx.tvm_call_packed_lowered")


@functor.mutator
class _MarkHostMetalContextMutator(PyStmtExprMutator):
    def __init__(self, packed_call=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_in_compute_scope = False
        self.packed_call = packed_call

    def visit_attr_stmt_(self, stmt):
        switch = stmt.attr_key == "compute_scope"
        old_value = False
        if switch:
            assert not self.is_in_compute_scope
            old_value, self.is_in_compute_scope = self.is_in_compute_scope, True
        s = self.visit_stmt(stmt.body)
        if switch:
            self.is_in_compute_scope = old_value
        return AttrStmt(stmt.node, stmt.attr_key, stmt.value, s)

    def visit_evaluate_(self, op: Evaluate):
        if self.is_in_compute_scope and isinstance(op.value, tir.Call) and op.value.op.same_as(_tvm_call_packed_lowered):
            if self.packed_call is not None:
                return AttrStmt(0, "tl.packed_call", self.packed_call, op)
            return AttrStmt(0, "metal_context", "", op)
        return op


def MarkHostMetalContext(packed_call=None):
    def pass_fn(func, mod, ctx):
        mutator = _MarkHostMetalContextMutator(packed_call)
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body, span=func.span)

    return prim_func_pass(pass_fn, opt_level=0)


__all__ = ["MarkHostMetalContext"]
