"""Disable software pipelining for loops with irregular exits.

A ``T.Pipelined`` loop whose body contains ``continue`` / ``break`` cannot be
software-pipelined: ``InjectSoftwarePipeline`` emits the stage-1 statements
of the last trip as an epilogue after the loop, and the loop exit goes with
them, producing CUDA that nvcc rejects (``a continue statement may only be
used within a loop``). Moving the exit back into the loop would not be
enough either: with ``num_stages=1`` a ``continue`` also skips the copy
issued for the next trip, and after an early ``break`` the epilogue would
still run the last trips.

This pass therefore strips the pipeline annotations (``num_stages``,
``tl_pipeline_order``, ``tl_pipeline_stage``) from any loop whose body
contains a loop exit targeting that loop, so ``PipelinePlanning`` leaves it
as a plain sequential loop and it compiles exactly like ``T.serial``. This
is the same rule TVM applies in ``AnnotateIrregularLoop``, which rejects
``software_pipeline_stage`` on a loop containing ``continue_loop`` /
``break_loop``, and mirrors the fallback already used by
``PipelinePlanning`` for targets without async-copy support.

A loop exit inside a *nested* loop targets that nested loop, so nested
``For`` / ``While`` bodies are not attributed to the loop being checked
(same as ``TargetedLoopBreakDetector`` in ``src/transform/unroll_loop.cc``,
the check added for fully unrolled loops in #3078).

This pass must run before ``PipelinePlanning`` in every backend pipeline,
and before warp specialization in the CUDA pipeline: warp specialization
rewrites a ``T.Pipelined`` loop with a TMA copy into a producer/consumer
structure with barriers, and a taken exit would skip the consumer arrive
for that iteration, deadlocking the producer on a barrier that is never
released. Stripping the annotations first makes
``ProducerConsumerWarpSpecialized`` skip such loops entirely (it only
rewrites loops carrying a ``num_stages`` annotation); the opt-in
``AutoWarpSpecialization`` pass does not require ``num_stages`` (its
``ScopeStages`` falls back to 1), so it separately declines kernels
containing ``tl.loop_break``, ``tirx.continue_loop``, or
``tirx.break_loop`` in ``PreprocessIR`` (``CheckCalls`` in
``src/cuda/transform/auto_schedule/preprocess_ir.cc``). Together, both
warp-specialization paths leave irregular loops sequential.
"""

from __future__ import annotations

from tvm import tirx
from tvm.tirx import (
    Call,
    For,
    PrimFunc,
    PyStmtExprVisitor,
    Stmt,
    While,
)
from tvm.tirx.transform import prim_func_pass

from tilelang.ir import get_stmt_span, stamp_stmt_spans

# Loop annotations consumed by PipelinePlanning to enable software
# pipelining. They are set by the frontend for T.Pipelined loops
# (see PipelinedFor in src/ir.cc).
_PIPELINE_ANNOTATIONS = ("num_stages", "tl_pipeline_order", "tl_pipeline_stage")

# Op name suffixes of the loop-exit intrinsics. The prefix differs between
# TVM versions (tir. vs tirx.), hence the suffix match.
_LOOP_EXIT_OP_SUFFIXES = ("continue_loop", "break_loop", "loop_break")


def _is_loop_exit(op: Call) -> bool:
    return any(op.op.name.endswith(suffix) for suffix in _LOOP_EXIT_OP_SUFFIXES)


@tirx.functor.visitor
class _LoopExitDetector(PyStmtExprVisitor):
    """Detect a continue/break that targets the loop being checked."""

    def __init__(self) -> None:
        super().__init__()
        self.found = False

    def visit_call_(self, op: Call) -> None:
        if _is_loop_exit(op):
            self.found = True

    def visit_for_(self, op: For) -> None:
        # A loop exit inside a nested loop targets that loop, not the loop
        # being checked. Do not recurse into nested loops.
        return

    def visit_while_(self, op: While) -> None:
        # As above, do not attribute a nested while-loop's exit to its parent.
        return


def _body_has_loop_exit(body: Stmt) -> bool:
    detector = _LoopExitDetector()
    detector.visit_stmt(body)
    return detector.found


@tirx.functor.mutator
class _DisablePipelineForIrregularLoopMutator(tirx.PyStmtExprMutator):
    """Strip pipeline annotations from loops with irregular exits."""

    def _make_for(self, original: For, new_body: Stmt, annotations=None) -> For:
        new_for = For(
            original.loop_var,
            original.min,
            original.extent,
            original.kind,
            new_body,
            original.thread_binding,
            original.annotations if annotations is None else annotations,
            original.step,
        )
        stamp_stmt_spans(new_for, get_stmt_span(original))
        return new_for

    def visit_for_(self, op: For) -> Stmt:
        # Recurse first so nested pipelined loops are handled as well.
        new_body = self.visit_stmt(op.body)
        if not _body_has_loop_exit(new_body):
            return self._make_for(op, new_body) if new_body is not op.body else op
        if not any(key in op.annotations for key in _PIPELINE_ANNOTATIONS):
            return self._make_for(op, new_body) if new_body is not op.body else op
        annotations = {key: value for key, value in op.annotations.items() if key not in _PIPELINE_ANNOTATIONS}
        return self._make_for(op, new_body, annotations)


def DisablePipelineForIrregularLoop():
    """Create a TVM pass that disables software pipelining for irregular loops.

    Loops whose body contains ``continue`` / ``break`` targeting the loop
    itself lose their pipeline annotations and are lowered as plain
    sequential loops, instead of producing invalid CUDA with a loop exit in
    the software-pipeline epilogue.

    Note:
        This pass must be applied before PipelinePlanning, while the
        ``T.Pipelined`` loop still carries its ``num_stages`` /
        ``tl_pipeline_order`` / ``tl_pipeline_stage`` annotations, and
        before warp specialization in the CUDA pipeline (see the module
        docstring).

    Returns:
        A TVM PrimFunc pass.
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        mutator = _DisablePipelineForIrregularLoopMutator()
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body, span=func.span)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.DisablePipelineForIrregularLoop")
