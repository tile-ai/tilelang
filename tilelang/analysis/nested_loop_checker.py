"""Reject nonconsecutive parallel nests, nested pipelines and collective tile ops."""

from tvm_tirx import ir, tirx

from ._visitor import NativeStmtVisitor, single_loop, tilelang_metadata


def _has_pipeline_annotation(annotations):
    return any(key in annotations for key in ("num_stages", "tl_pipeline_order", "tl_pipeline_stage"))


def is_pipelined_for(op: ir.For) -> bool:
    return _has_pipeline_annotation(op.annotations)


def is_tile_op(op: ir.Call) -> bool:
    # The actual native registry is authoritative. Scalar reducer updates and
    # other per-iteration intrinsics do not register a TLOpBuilder.
    return isinstance(op.op, ir.Op) and op.op.get_attr("TLOpBuilder") is not None


class _NestedLoopCheckVisitor(NativeStmtVisitor):
    def __init__(self):
        super().__init__()
        self.parallel_depth = 0
        self.consecutive_child = None

    def visit_for(self, op, visitor):
        if is_pipelined_for(op) and self.parallel_depth:
            self.reject_pipeline()
        if op.kind != ir.ForKind.PARALLEL:
            return super().visit_for(op, visitor)
        if self.parallel_depth and (self.consecutive_child is None or not op.same_as(self.consecutive_child)):
            raise ValueError("[Tilelang Semantic Check] Nested parallel loops are not allowed. Please check your loop structure.")
        previous_child = self.consecutive_child
        self.consecutive_child = single_loop(op.body)
        self.parallel_depth += 1
        try:
            super().visit_for(op, visitor)
        finally:
            self.parallel_depth -= 1
            self.consecutive_child = previous_child

    def visit_region(self, op, visitor):
        metadata = tilelang_metadata(op)
        if (
            self.parallel_depth
            and metadata is not None
            and metadata.get("key") == "tl.loop.annotations"
            and _has_pipeline_annotation(metadata["value"])
        ):
            self.reject_pipeline()
        super().visit_region(op, visitor)

    @staticmethod
    def reject_pipeline():
        raise ValueError(
            "[Tilelang Semantic Check] Pipelined loop cannot be nested inside a parallel loop. Please check your loop structure."
        )

    def visit_call(self, op, visitor):
        if self.parallel_depth and is_tile_op(op):
            raise ValueError(
                f'[Tilelang Semantic Check] Only elementwise operations are allowed inside a parallel loop. Got a tile-op "{op.op}".'
            )
        super().visit_call(op, visitor)


def NestedLoopChecker():
    """Keep TileLang's loop composition rules on native Function/SeqStmt IR.

    Serial loops may enclose or appear within a parallel loop. Parallel loops
    may nest only as a strict consecutive chain, including its native singleton
    SeqStmt wrappers. A pipeline or collective tile operation cannot appear
    inside that chain. Independent sequential parallel nests remain valid.
    """

    def pass_fn(func, _mod, _ctx):
        _NestedLoopCheckVisitor().visit_stmt(func.body)
        return func

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.NestedLoopChecker")
