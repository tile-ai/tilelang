from __future__ import annotations
from tvm import tir
from tvm.tir import PyStmtExprVisitor, BufferStore, For, Var, PrimFunc, BufferLoad, IntImm
from tvm.tir.transform import prim_func_pass
from tvm.tir.stmt_functor import post_order_visit


@tir.functor.visitor
class _LoopVarUseAnalyzer(PyStmtExprVisitor):
    """Analyze whether a loop variable is used in the given expr."""

    def __init__(self, var: Var) -> None:
        super().__init__()
        self.var = var
        self.used = False

    def visit_var_(self, op: Var) -> None:
        if op == self.var:
            self.used = True
        # Don't recursively visit children to avoid infinite recursion


def collect_local_buffer_accesses(statement) -> list[BufferLoad | BufferStore]:
    """
    Collect local buffer accesses in the loop body.

    Args:
        statement: The TIR statement to analyze

    Returns:
        Tuple of buffer accesses in the loop body.
    """

    buffer_accesses = []

    def visit_buffer_access(node):
        if isinstance(node, (BufferLoad, BufferStore)) and node.buffer.scope().startswith("local"):
            buffer_accesses.append(node)

    post_order_visit(statement, visit_buffer_access)

    return buffer_accesses


@tir.functor.visitor
class _FragmentLoopCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()

    def visit_for_(self, op: For) -> None:
        if op.kind == tir.ForKind.PARALLEL:
            # Fuse consecutive parallel loops
            # Other nested cases are all invalid in TileLang.
            loops = [op]
            child = op.body
            while isinstance(child, For) and child.kind == tir.ForKind.PARALLEL:
                loops.append(child)
                child = child.body

            loops_with_symbolic_ranges = []
            for loop in loops:
                if not (isinstance(loop.min, IntImm) and isinstance(loop.extent, IntImm)):
                    loops_with_symbolic_ranges.append(loop)

            if len(loops_with_symbolic_ranges) > 0:
                buffer_accesses = collect_local_buffer_accesses(child)
            for loop in loops_with_symbolic_ranges:
                for buffer_access in buffer_accesses:
                    indices = buffer_access.indices
                    analyzer = _LoopVarUseAnalyzer(loop.loop_var)
                    for index in indices:
                        analyzer.visit_expr(index)
                    if analyzer.used:
                        raise ValueError(
                            "[Tilelang Semantic Check] "
                            f"Loop variable {loop.loop_var} in a T.Parallel loop with symbolic range (min={loop.min}, extent={loop.extent}) is used to index "
                            "a local/fragment buffer, which is not allowed in Tilelang."
                        )

            return

        self.visit_stmt(op.body)


def FragmentLoopChecker():
    """
    When using T.Parallel over a local/fragment buffer, there are several restrictions:
    to ensure that the parallelization is valid.

    1. The range of loop can not be symbolic.

    Returns:
        A prim_func_pass that applies the transformation
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _FragmentLoopCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
