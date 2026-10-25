from __future__ import annotations
from tilelang._compat.tvm import tirx
from tilelang._compat.tirx import PyStmtExprVisitor, BufferStore, For, Var, PrimFunc, BufferLoad, IntImm, ForKind
from tvm_tirx.tirx.transform import function_pass as prim_func_pass
from tilelang._compat.stmt_functor import post_order_visit

from tilelang.utils.language import is_fragment


@tirx.functor.visitor
class _LoopVarUseAnalyzer(PyStmtExprVisitor):
    """Analyze whether a loop variable is used in the given expr."""

    def __init__(self, var: Var) -> None:
        super().__init__()
        self.var = var
        self.used = False

    def visit_var_(self, op: Var) -> None:
        if op.same_as(self.var):
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
        buffer = node.source if isinstance(node, BufferLoad) else node.dest if isinstance(node, BufferStore) else None
        if buffer is not None and is_fragment(buffer):
            buffer_accesses.append(node)

    post_order_visit(statement, visit_buffer_access)

    return buffer_accesses


@tirx.functor.visitor
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

        # Reach the innermost loop -- This may cause repeated checks for cases like:
        #   For1{
        #       Stmt1;
        #       For2{};
        #       For3{};
        #   };
        # But it's OK since the check is idempotent.
        if not isinstance(child, For):
            buffer_accesses = collect_fragment_accesses(child)

            loops_with_symbolic_ranges = []

            for loop in self.loop_stack:
                # symbolic
                if loop.kind == ForKind.PARALLEL and not all(isinstance(bound, IntImm) for bound in (loop.begin, loop.end, loop.step)):
                    loops_with_symbolic_ranges.append(loop)

            for buffer_access in buffer_accesses:
                indices = buffer_access.indices
                # Check 1
                for loop in loops_with_symbolic_ranges:
                    analyzer = _LoopVarUseAnalyzer(loop.var)
                    for index in indices:
                        analyzer.visit_expr(index)
                    if analyzer.used:
                        raise ValueError(
                            "[Tilelang Semantic Check] "
                            f"Loop variable {loop.var} in a T.Parallel loop with symbolic range ({loop.begin}, {loop.end}, {loop.step}) is used to index "
                            "a fragment buffer, which is not allowed in Tilelang."
                        )

        self.visit_stmt(op.body)
        self.loop_stack.pop()


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
