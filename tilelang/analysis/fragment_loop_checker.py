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


def collect_fragment_accesses(statement) -> list[BufferLoad | BufferStore]:
    """
    Collect fragment accesses in the loop body.

    Args:
        statement: The TIR statement to analyze

    Returns:
        Tuple of buffer accesses in the loop body.
    """

    buffer_accesses = []

    def visit_buffer_access(node):
        if isinstance(node, (BufferLoad, BufferStore)) and node.buffer.scope() == "local.fragment":
            buffer_accesses.append(node)

    post_order_visit(statement, visit_buffer_access)

    return buffer_accesses


@tir.functor.visitor
class _FragmentLoopCheckVisitor(PyStmtExprVisitor):
    """
    Check whether the fragment accesses are valid.

    This checker will recursively visit all the for loops until it reaches certain "inner most loop".
    Then it will start to check the validity of fragment access in the loop body. We need to maintain a stack of
    loops during the traversal since this is the context/scope of the fragment access.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loop_stack = []

    def visit_for_(self, op: For) -> None:
        self.loop_stack.append(op)
        child = op.body

        # Reach the the innermost loop
        # This may cause repeated check for cases like: For1{Stmt1; For2{}; For3{};};
        # But it's OK since the check is idempotent.
        if not isinstance(child, For):
            buffer_accesses = collect_fragment_accesses(child)

            loops_with_symbolic_ranges = []
            non_parallel_loops = []

            for loop in self.loop_stack:
                # Vectorized is also allowed
                if loop.kind != tir.ForKind.PARALLEL and loop.kind != tir.ForKind.VECTORIZED:
                    non_parallel_loops.append(loop)
                # symbolic
                elif not (isinstance(loop.min, IntImm) and isinstance(loop.extent, IntImm)):
                    loops_with_symbolic_ranges.append(loop)

            for buffer_access in buffer_accesses:
                indices = buffer_access.indices
                # Check 1
                for loop in loops_with_symbolic_ranges:
                    analyzer = _LoopVarUseAnalyzer(loop.loop_var)
                    for index in indices:
                        analyzer.visit_expr(index)
                    if analyzer.used:
                        raise ValueError(
                            "[Tilelang Semantic Check] "
                            f"Loop variable {loop.loop_var} in a T.Parallel loop with symbolic range (min={loop.min}, extent={loop.extent}) is used to index "
                            "a fragment buffer, which is not allowed in Tilelang."
                        )
                # Check 2
                for loop in non_parallel_loops:
                    analyzer = _LoopVarUseAnalyzer(loop.loop_var)
                    for index in indices:
                        analyzer.visit_expr(index)
                    if analyzer.used:
                        raise ValueError(
                            "[Tilelang Semantic Check] "
                            f"A non-parallel loop iterator {loop.loop_var} is used to index "
                            "a fragment buffer, which is not allowed in Tilelang."
                        )

        self.visit_stmt(op.body)
        self.loop_stack.pop()


def FragmentLoopChecker():
    """
    When using T.Parallel over a local/fragment buffer, there are several restrictions:
    to ensure that the parallelization is valid.

    1. The range of loop can not be symbolic.
    2. Any access/indexing of the fragment buffer should not contain other types of iterators (like loop variables from T.Serial).

    Returns:
        A prim_func_pass that applies the transformation
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _FragmentLoopCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
