from __future__ import annotations

from tilelang._compat.tvm import tirx
from tilelang._compat.tirx import Buffer, BufferLoad, BufferStore, For, ForKind, PrimFunc, PyStmtExprVisitor, Var
from tvm_tirx.tirx.transform import function_pass as prim_func_pass

from tilelang.utils.language import is_local


@tirx.functor.visitor
class _LoopVarUseAnalyzer(PyStmtExprVisitor):
    """Check whether an expression refers to a particular loop variable."""

    def __init__(self, var: Var) -> None:
        super().__init__()
        self.var = var
        self.used = False

    def visit_var_(self, op: Var) -> None:
        if op.same_as(self.var):
            self.used = True


@tirx.functor.visitor
class _ParallelLocalIndexCheckVisitor(PyStmtExprVisitor):
    """Reject local-buffer indices that depend on an enclosing parallel loop."""

    def __init__(self) -> None:
        super().__init__()
        self.parallel_loop_stack: list[For] = []

    def visit_for_(self, op: For) -> None:
        is_parallel = op.kind == ForKind.PARALLEL
        if is_parallel:
            self.parallel_loop_stack.append(op)
        try:
            super().visit_for_(op)
        finally:
            if is_parallel:
                self.parallel_loop_stack.pop()

    def _check_indices(self, buffer: Buffer, indices) -> None:
        if not self.parallel_loop_stack or not is_local(buffer):
            return

        for loop in self.parallel_loop_stack:
            analyzer = _LoopVarUseAnalyzer(loop.var)
            for index in indices:
                analyzer.visit_expr(index)
            if analyzer.used:
                raise ValueError(
                    "[Tilelang Semantic Check] "
                    f"Local buffer `{buffer.name}` is indexed by T.Parallel loop variable `{loop.var}`. "
                    "Local buffers are thread-private and do not participate in parallel layout inference. "
                    "Use T.serial/T.vectorized/T.unroll for per-thread local indexing, or T.alloc_fragment "
                    "when the indexed dimension should be distributed across threads."
                )

    def visit_tensor_load_(self, op: BufferLoad) -> None:
        self._check_indices(op.source, op.indices)
        super().visit_tensor_load_(op)

    def visit_tensor_store_(self, op: BufferStore) -> None:
        self._check_indices(op.dest, op.indices)
        super().visit_tensor_store_(op)


def ParallelLocalIndexChecker():
    """Reject indexing a thread-private local buffer with a T.Parallel loop variable.

    Local buffers have no cross-thread ownership layout. A local access may
    appear inside a parallel loop when its index is independent of the parallel
    loop variables, for example a replicated scalar ``scale_local[0]``. For an
    indexed per-thread loop, use ``T.serial``, ``T.vectorized``, or ``T.unroll``;
    for distributed ownership, allocate a fragment instead.
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _ParallelLocalIndexCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
