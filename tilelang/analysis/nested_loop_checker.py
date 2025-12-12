from tvm import tir
from tvm.tir import (
    For,
    Call,
    PrimFunc,
    PyStmtExprVisitor,
)
from tvm.tir.transform import prim_func_pass


def is_pipelined_for(op: For) -> bool:
    """Check if a for loop is pipelined."""

    anno_keys = ["num_stages", "tl_pipeline_order", "tl_pipeline_stage", "tl_pipeline_sync", "tl_pipeline_group"]
    return any(key in op.annotations for key in anno_keys)


def is_tile_op(op: Call) -> bool:
    """Check if a call is a tile-op"""

    return op.op.get_attr("TLOpBuilder") is not None


@tir.functor.visitor
class _NestedLoopCheckVisitor(PyStmtExprVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.in_parallel_context = False

    def visit_for_(self, op: For) -> None:
        if op.kind == tir.ForKind.PARALLEL:
            child = op.body

            # Special case: continuous nested parallel loop is allowed.
            if isinstance(child, tir.For) and child.kind == tir.ForKind.PARALLEL:
                self.visit_stmt(child)
                return

            # Otherwise
            if self.in_parallel_context:
                raise ValueError("[Tilelang Semantic Check] Nested parallel loops are not allowed. Please check your loop structure.")
            self.in_parallel_context = True
            super().visit_for_(op)
            self.in_parallel_context = False
            return
        elif is_pipelined_for(op):
            if self.in_parallel_context:
                raise ValueError(
                    "[Tilelang Semantic Check] Pipelined loop cannot be nested inside a parallel loop. Please check your loop structure."
                )

        super().visit_for_(op)

    def visit_call_(self, op: Call) -> None:
        if self.in_parallel_context and is_tile_op(op):
            raise ValueError(
                f'[Tilelang Semantic Check] Only elementwise operations are allowed inside a parallel loop. Got a tile-op "{op.op}".'
            )


def NestedLoopChecker():
    """
    User-friendly pass which identifies any invalid any nested-loop pattern.

    Nested loops is an annoying problem in tilelang or other polyhedral-style compilers.
    It contains many corner cases and undefined behaviours.

    In tilelang, there are four loops:
        T.serial
        T.Parallel (T.vectorized)
        T.Pipelined
        T.Persistent

    T.Persistent is a new feature which we do not consider here.

    We define the following rules:
    - (Rule 1) T.serial can be nested inside any other loop type without restriction.
    - (Rule 2) Consecutive T.Parallel nested loops are not allowed. Including any TileOp (T.copy, etc.) which has
        "parallel" behaviours is also forbidden.

        Examples:
        for i in T.Parallel(M):
            stmt
            for j in T.Parallel(N):
                ...

        for i in T.Parallel(M):
            T.copy(A, B) # forbidden!

        **Only a special case is allowed: strict continuous Parallel loops.** Since we can fuse them into a single T.Parallel loop.
        Example:

        for i in T.Parallel(M):
                for j in T.Parallel(N):
                    ... # allowed
    - (Rule 3) T.Pipelined inside a T.Parallel is forbidden.

        Examples:
            for i in T.Parallel(M):
                for j in T.Pipelined(K): # forbidden!
                    ...

            for i in T.Pipelined(K):
                for j in T.Parallel(N): # allowed, ok
                    ...

    In summary, the problem mainly lies in the "T.Parallel". We highly recommend to use
    T.Parallel to implement a tiled operator inside a kernel (e.g. T.gemm level) instead of other usages.
    This guideline can help you avoid most of the issues.

    Returns:
        A prim_func_pass that applies the transformation
    """

    def pass_fn(func: PrimFunc, mod, ctx):
        _NestedLoopCheckVisitor().visit_stmt(func.body)
        return func

    return prim_func_pass(pass_fn, opt_level=0)
