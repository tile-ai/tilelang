"""Check local tensor indexing against enclosing native parallel-loop identities."""

from tvm_tirx import ir, tirx

from ._visitor import NativeStmtVisitor, tensor_scope


class _ParallelLocalIndexCheckVisitor(NativeStmtVisitor):
    def _check_indices(self, tensor, indices):
        if tensor_scope(tensor) != "local":
            return
        for loop in self.loop_stack:
            if loop.kind == ir.ForKind.PARALLEL and any(self.uses_var(index, loop.var) for index in indices):
                raise ValueError(
                    "[Tilelang Semantic Check] "
                    f"Local buffer `{getattr(tensor, 'name', tensor)}` is indexed by T.Parallel loop variable `{loop.var}`. "
                    "Local buffers are thread-private and do not participate in parallel layout inference. "
                    "Use T.serial/T.vectorized/T.unroll for per-thread local indexing, or T.alloc_fragment "
                    "when the indexed dimension should be distributed across threads."
                )

    def visit_tensor_load(self, op, visitor):
        self._check_indices(op.source, op.indices)
        super().visit_tensor_load(op, visitor)

    def visit_tensor_store(self, op, visitor):
        self._check_indices(op.dest, op.indices)
        super().visit_tensor_store(op, visitor)


def ParallelLocalIndexChecker():
    """Allow replicated local values and reject distributed local-array indices.

    Native flat Bind aliases are followed within their lexical scope so naming
    an expression does not hide its parallel-loop dependency. Variable names
    are diagnostic text only; dependency checks use actual Var identity.
    """

    def pass_fn(func, _mod, _ctx):
        _ParallelLocalIndexCheckVisitor().visit_stmt(func.body)
        return func

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.ParallelLocalIndexChecker")
