"""
Lower Async Copy Pass for TileLang.

This module implements the LowerAsyncCopy pass that transforms vectorized
copy loops from global to shared memory into ptx_cp_async intrinsics.

Design Philosophy:
==================
This pass uses the `pipeline_stage` attribute added by InjectPipeline to
determine synchronization points. This is much more robust than analyzing
buffer indices because:

1. InjectPipeline explicitly marks each statement with its pipeline stage
2. No pattern matching or guessing required
3. Clear separation of concerns between pipeline structure and lowering

Pipeline Synchronization Strategy:
==================================
For a software pipeline with num_stages buffers:

1. Prologue: Fill (num_stages - 1) buffers before starting computation
   - Each stage gets its own commit_group()

2. Steady State: Overlap copy and compute
   - Producer writes to buffer[(iter + offset) % num_stages] where offset > 0
   - Consumer reads from buffer[iter % num_stages]
   - wait_group(num_stages - 1) ensures the oldest copy completes

3. Epilogue: Drain remaining computations
   - wait_group decreases from (num_stages - 2) to 0

Example for num_stages=3:
    Prologue:
        copy -> buffer[0]; commit_group()  # Group 0 pending
        copy -> buffer[1]; commit_group()  # Groups 0,1 pending

    Steady State (for ko in range(N - num_stages + 1)):
        copy -> buffer[(ko+2) % 3]; commit_group()  # 3 groups pending
        wait_group(2)  # Wait until 2 groups pending (oldest done)
        compute(buffer[ko % 3])

    Epilogue:
        wait_group(1); compute(buffer[(N-2) % 3])
        wait_group(0); compute(buffer[(N-1) % 3])
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from tilelang import tvm as tvm
from tilelang.language.tir import op as tl_op
from tilelang.utils import is_global, is_shared
from tvm import tir
from tvm.tir import (
    AttrStmt,
    Block,
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    For,
    ForKind,
    IntImm,
    PrimExpr,
    PrimFunc,
    PyStmtExprMutator,
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tir.stmt_functor import post_order_visit
from tvm.tir.transform import prim_func_pass

# Op constants for type-safe comparison
_OP_PTX_CP_ASYNC = tir.op.Op.get("tl.ptx_cp_async")
_OP_ADDRESS_OF = tir.op.Op.get("tir.address_of")
_OP_IF_THEN_ELSE = tir.op.Op.get("tir.if_then_else")


def _compute_linear_offset(indices: list, buf: Buffer) -> PrimExpr:
    """Compute linear offset from multi-dimensional indices."""
    if len(indices) == 0:
        return IntImm("int32", 0)
    if len(indices) == 1:
        return indices[0]

    strides = []
    stride = IntImm("int32", 1)
    for i in range(len(buf.shape) - 1, -1, -1):
        strides.insert(0, stride)
        if i > 0:
            stride = stride * buf.shape[i]

    offset = IntImm("int32", 0)
    for idx, s in zip(indices, strides):
        offset = offset + idx * s

    return offset


def _substitute_var(expr: PrimExpr, var: Var, value: PrimExpr) -> PrimExpr:
    """Substitute a variable with a value in an expression."""
    var_map = {var: value}
    result = tir.stmt_functor.substitute(expr, var_map)
    analyzer = tvm.arith.Analyzer()
    return analyzer.simplify(result)


# =============================================================================
# Pipeline Stage Information
# =============================================================================


@dataclass
class PipelineStageAttr:
    """Information extracted from pipeline_stage attribute."""

    stage_expr: PrimExpr  # The full stage expression
    is_constant: bool  # True if stage is a constant (prologue/epilogue)
    constant_value: int | None  # The constant stage value if is_constant
    loop_var: Var | None  # The loop variable if not constant
    offset: int | None  # Offset from loop var, e.g., 2 for (ko + 2) % 3
    num_stages: int | None  # Total number of stages (modulo value)


def _extract_pipeline_stage_attr(attr: AttrStmt) -> PipelineStageAttr | None:
    """Extract pipeline stage information from AttrStmt."""
    if attr.attr_key != "pipeline_stage":
        return None

    stage_expr = attr.value

    # Case 1: Constant stage (prologue/epilogue)
    if isinstance(stage_expr, IntImm):
        return PipelineStageAttr(
            stage_expr=stage_expr,
            is_constant=True,
            constant_value=stage_expr.value,
            loop_var=None,
            offset=None,
            num_stages=None,
        )

    # Case 2: FloorMod expression (steady state)
    # Pattern: (loop_var + offset) % num_stages
    if isinstance(stage_expr, tir.FloorMod):
        num_stages = None
        if isinstance(stage_expr.b, IntImm):
            num_stages = stage_expr.b.value

        base = stage_expr.a
        loop_var = None
        offset = 0

        # Try to extract loop_var and offset from base
        if isinstance(base, Var):
            loop_var = base
            offset = 0
        elif isinstance(base, tir.Add):
            if isinstance(base.a, Var) and isinstance(base.b, IntImm):
                loop_var = base.a
                offset = base.b.value
            elif isinstance(base.b, Var) and isinstance(base.a, IntImm):
                loop_var = base.b
                offset = base.a.value

        if loop_var is not None:
            return PipelineStageAttr(
                stage_expr=stage_expr,
                is_constant=False,
                constant_value=None,
                loop_var=loop_var,
                offset=offset,
                num_stages=num_stages,
            )

    return None


def _is_pipeline_stage_attr(stmt: Stmt) -> bool:
    """Check if statement is an AttrStmt with pipeline_stage."""
    return isinstance(stmt, AttrStmt) and stmt.attr_key == "pipeline_stage"


def _get_pipeline_stage_attr(stmt: Stmt) -> PipelineStageAttr | None:
    """Get pipeline stage attr from statement if present."""
    if isinstance(stmt, AttrStmt) and stmt.attr_key == "pipeline_stage":
        return _extract_pipeline_stage_attr(stmt)
    return None


# =============================================================================
# Statement Classification
# =============================================================================


class StmtKind(Enum):
    """Classification of statements for async copy synchronization."""

    ASYNC_COPY = auto()  # Contains ptx_cp_async (needs commit_group after)
    SHARED_CONSUMER = auto()  # Reads from shared memory (needs wait_group before)
    MIXED = auto()  # Contains both (steady state loop body)
    OTHER = auto()  # Neither


class StatementClassifier:
    """Classify statements as producers, consumers, mixed, or other."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def classify(self, stmt: Stmt) -> StmtKind:
        """Classify a statement for async copy synchronization."""
        has_async_copy = self._contains_async_copy(stmt)
        has_shared_read = self._reads_shared_memory(stmt)

        if has_async_copy and has_shared_read:
            return StmtKind.MIXED
        elif has_async_copy:
            return StmtKind.ASYNC_COPY
        elif has_shared_read:
            return StmtKind.SHARED_CONSUMER
        else:
            return StmtKind.OTHER

    def _contains_async_copy(self, stmt: Stmt) -> bool:
        """Check if statement contains ptx_cp_async."""
        result = False

        def visitor(node):
            nonlocal result
            if isinstance(node, Call) and node.op.same_as(_OP_PTX_CP_ASYNC):
                result = True

        post_order_visit(stmt, visitor)
        return result

    def _reads_shared_memory(self, stmt: Stmt) -> bool:
        """Check if statement reads from shared memory."""
        result = False

        def visitor(node):
            nonlocal result
            if (
                isinstance(node, BufferLoad)
                and is_shared(node.buffer)
                or (
                    isinstance(node, Call)
                    and node.op.same_as(_OP_ADDRESS_OF)
                    and len(node.args) > 0
                    and isinstance(node.args[0], BufferLoad)
                    and is_shared(node.args[0].buffer)
                )
            ):
                result = True

        post_order_visit(stmt, visitor)
        return result


# =============================================================================
# Async Copy Pattern Matching
# =============================================================================


@dataclass
class AsyncCopyMatch:
    """Result of matching a vectorized copy pattern."""

    store_buffer: Buffer
    load_buffer: Buffer
    vec_len: int
    store_offset: PrimExpr
    load_offset: PrimExpr
    loop_var: Var


def _match_vectorized_copy(loop: For) -> AsyncCopyMatch | None:
    """Check if a For loop is a vectorized copy from global to shared memory."""
    if loop.kind != ForKind.VECTORIZED:
        return None

    if not isinstance(loop.min, IntImm) or loop.min.value != 0:
        return None

    if not isinstance(loop.extent, IntImm):
        return None

    vec_len = loop.extent.value
    loop_var = loop.loop_var

    body = loop.body
    if isinstance(body, SeqStmt) and len(body.seq) == 1:
        body = body.seq[0]

    if not isinstance(body, BufferStore):
        return None

    store = body
    store_buffer = store.buffer

    if not is_shared(store_buffer):
        return None

    load = None
    if isinstance(store.value, BufferLoad):
        load = store.value
    elif isinstance(store.value, Call):
        call = store.value
        if call.op.same_as(_OP_IF_THEN_ELSE) and len(call.args) >= 2 and isinstance(call.args[1], BufferLoad):
            load = call.args[1]

    if load is None:
        return None

    load_buffer = load.buffer

    if not is_global(load_buffer):
        return None

    def compute_offset_at_zero(indices, buf):
        substituted = [_substitute_var(idx, loop_var, IntImm("int32", 0)) for idx in indices]
        return _compute_linear_offset(substituted, buf)

    store_offset = compute_offset_at_zero(list(store.indices), store_buffer)
    load_offset = compute_offset_at_zero(list(load.indices), load_buffer)

    return AsyncCopyMatch(
        store_buffer=store_buffer,
        load_buffer=load_buffer,
        vec_len=vec_len,
        store_offset=store_offset,
        load_offset=load_offset,
        loop_var=loop_var,
    )


# =============================================================================
# Async Copy Lowering
# =============================================================================


@tir.functor.mutator
class AsyncCopyLowerer(PyStmtExprMutator):
    """Lower vectorized global->shared copy to ptx_cp_async."""

    def __init__(self, buffer_data_to_buffer: dict[Var, Buffer], verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.buffer_data_to_buffer = buffer_data_to_buffer
        self.has_async_copy: bool = False

    def visit_block_(self, op: Block) -> Block:
        for buffer in op.alloc_buffers:
            self.buffer_data_to_buffer[buffer.data] = buffer

        new_body = self.visit_stmt(op.body)
        if new_body.same_as(op.body):
            return op

        return Block(
            op.iter_vars,
            op.reads,
            op.writes,
            op.name_hint,
            new_body,
            op.init,
            op.alloc_buffers,
            op.match_buffers,
            op.annotations,
        )

    def visit_for_(self, loop: For) -> Stmt:
        match = _match_vectorized_copy(loop)
        if match is not None:
            result = self._create_async_copy(match)
            if result is not None:
                return result

        new_body = self.visit_stmt(loop.body)

        if new_body.same_as(loop.body):
            return loop

        return For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            new_body,
            loop.thread_binding,
            loop.annotations,
        )

    def _create_async_copy(self, match: AsyncCopyMatch) -> Stmt | None:
        """Create ptx_cp_async call from matched pattern."""
        dtype_bytes = match.load_buffer.dtype.bits // 8
        copy_bytes = match.vec_len * dtype_bytes

        if copy_bytes not in (4, 8, 16):
            return None

        if self.verbose:
            print(f"[AsyncCopy] {match.load_buffer.name} -> {match.store_buffer.name}, {copy_bytes} bytes")

        self.has_async_copy = True

        # Create access_ptr for dst (shared memory, write)
        dst_access_ptr = tir.tvm_access_ptr(
            tir.type_annotation(match.store_buffer.dtype),
            match.store_buffer.data,
            match.store_offset,
            IntImm("int32", match.vec_len),
            IntImm("int32", 2),  # write access
        )

        # Create access_ptr for src (global memory, read)
        src_access_ptr = tir.tvm_access_ptr(
            tir.type_annotation(match.load_buffer.dtype),
            match.load_buffer.data,
            match.load_offset,
            IntImm("int32", match.vec_len),
            IntImm("int32", 1),  # read access
        )

        # New signature: ptx_cp_async(dst_access_ptr, src_access_ptr, bytes, predicate=None)
        async_copy_call = tl_op.ptx_cp_async(
            dst_access_ptr,
            src_access_ptr,
            IntImm("int32", copy_bytes),
        )

        return tir.Evaluate(async_copy_call)


# =============================================================================
# Pipeline Synchronization Insertion
# =============================================================================


@tir.functor.mutator
class PipelineSyncInserter(PyStmtExprMutator):
    """
    Insert proper synchronization for software pipelining.

    Uses the pipeline_stage attribute from InjectPipeline to determine
    synchronization points.
    """

    def __init__(self, num_stages: int, verbose: bool = False):
        super().__init__()
        self.num_stages = num_stages
        self.verbose = verbose
        self.classifier = StatementClassifier(verbose)

    def visit_seq_stmt_(self, seq: SeqStmt) -> Stmt:
        """Process a SeqStmt with proper pipeline synchronization using pipeline_stage attr."""
        new_stmts = []
        pending_producers: list[tuple[Stmt, int | None]] = []  # (stmt, constant_stage)
        current_stage: int | None = None
        epilogue_consumer_idx = 0
        has_seen_steady_state = False

        for s in seq.seq:
            new_s = self.visit_stmt(s)

            # Check for pipeline_stage attr
            stage_attr = _get_pipeline_stage_attr(new_s)
            classification = self.classifier.classify(new_s)

            if self.verbose and stage_attr:
                if stage_attr.is_constant:
                    print(f"  [Stage] constant={stage_attr.constant_value}, kind={classification}")
                else:
                    print(f"  [Stage] offset={stage_attr.offset}, kind={classification}")

            if classification == StmtKind.ASYNC_COPY:
                if stage_attr and stage_attr.is_constant:
                    # Prologue producer with constant stage
                    producer_stage = stage_attr.constant_value

                    # Flush on stage change
                    if pending_producers and current_stage is not None and producer_stage != current_stage:
                        for p, _ in pending_producers:
                            new_stmts.append(p)
                        new_stmts.append(tir.Evaluate(tir.ptx_commit_group()))
                        pending_producers = []

                    pending_producers.append((new_s, producer_stage))
                    current_stage = producer_stage
                else:
                    # Non-pipelined or steady state producer (handled in loop body)
                    pending_producers.append((new_s, None))

            elif classification == StmtKind.MIXED:
                # Flush prologue producers before steady state
                if pending_producers:
                    for p, _ in pending_producers:
                        new_stmts.append(p)
                    new_stmts.append(tir.Evaluate(tir.ptx_commit_group()))
                    pending_producers = []

                new_stmts.append(new_s)
                has_seen_steady_state = True
                epilogue_consumer_idx = 0

            elif classification == StmtKind.SHARED_CONSUMER:
                # Flush pending producers
                has_pending = len(pending_producers) > 0
                if pending_producers:
                    for p, _ in pending_producers:
                        new_stmts.append(p)
                    new_stmts.append(tir.Evaluate(tir.ptx_commit_group()))
                    pending_producers = []

                # Determine wait_count based on context
                if has_seen_steady_state:
                    # Epilogue consumer - drain pipeline
                    wait_count = max(0, self.num_stages - 2 - epilogue_consumer_idx)
                    new_stmts.append(tir.Evaluate(tir.ptx_wait_group(wait_count)))
                    epilogue_consumer_idx += 1
                elif has_pending:
                    # Simple case - wait for all
                    new_stmts.append(tir.Evaluate(tir.ptx_wait_group(0)))

                new_stmts.append(new_s)

            else:  # StmtKind.OTHER
                if pending_producers:
                    for p, _ in pending_producers:
                        new_stmts.append(p)
                    new_stmts.append(tir.Evaluate(tir.ptx_commit_group()))
                    pending_producers = []
                new_stmts.append(new_s)

        # Handle remaining producers
        if pending_producers:
            for p, _ in pending_producers:
                new_stmts.append(p)
            new_stmts.append(tir.Evaluate(tir.ptx_commit_group()))
            new_stmts.append(tir.Evaluate(tir.ptx_wait_group(0)))

        if len(new_stmts) == 1:
            return new_stmts[0]
        return SeqStmt(new_stmts)

    def visit_block_(self, op: Block) -> Block:
        if op.name_hint and op.name_hint.startswith("_"):
            return op
        new_body = self.visit_stmt(op.body)
        if new_body.same_as(op.body):
            return op
        return Block(
            op.iter_vars,
            op.reads,
            op.writes,
            op.name_hint,
            new_body,
            op.init,
            op.alloc_buffers,
            op.match_buffers,
            op.annotations,
        )

    def visit_for_(self, loop: For) -> Stmt:
        classification = self.classifier.classify(loop)
        if classification == StmtKind.MIXED:
            # Process steady state loop body
            processed_body = self._process_steady_state_body(loop.body)
            return For(
                loop.loop_var,
                loop.min,
                loop.extent,
                loop.kind,
                processed_body,
                loop.thread_binding,
                loop.annotations,
            )
        elif classification in (StmtKind.ASYNC_COPY, StmtKind.SHARED_CONSUMER):
            return loop

        new_body = self.visit_stmt(loop.body)
        if new_body.same_as(loop.body):
            return loop
        return For(
            loop.loop_var,
            loop.min,
            loop.extent,
            loop.kind,
            new_body,
            loop.thread_binding,
            loop.annotations,
        )

    def _process_steady_state_body(self, body: Stmt) -> Stmt:
        """
        Process steady state loop body.

        Uses pipeline_stage attr to determine producer offset vs consumer offset.
        Producers have offset > 0, consumers have offset = 0.
        """
        if not isinstance(body, SeqStmt):
            return body

        new_stmts = []
        pending_producers = []

        for s in body.seq:
            classification = self.classifier.classify(s)

            if classification == StmtKind.ASYNC_COPY:
                pending_producers.append(s)
            else:
                if pending_producers:
                    new_stmts.extend(pending_producers)
                    new_stmts.append(tir.Evaluate(tir.ptx_commit_group()))
                    pending_producers = []

                if classification == StmtKind.SHARED_CONSUMER:
                    # Wait for oldest copy to complete
                    wait_count = self.num_stages - 1
                    new_stmts.append(tir.Evaluate(tir.ptx_wait_group(wait_count)))

                new_stmts.append(s)

        if pending_producers:
            new_stmts.extend(pending_producers)
            new_stmts.append(tir.Evaluate(tir.ptx_commit_group()))

        if len(new_stmts) == 1:
            return new_stmts[0]
        return SeqStmt(new_stmts)


# =============================================================================
# Pipeline Attribute Cleanup
# =============================================================================


@tir.functor.mutator
class PipelineAttrRemover(PyStmtExprMutator):
    """Remove pipeline_stage attributes after synchronization insertion."""

    def __init__(self):
        super().__init__()

    def visit_attr_stmt_(self, op: AttrStmt) -> Stmt:
        if op.attr_key == "pipeline_stage":
            # Remove this attr, just return the body
            return self.visit_stmt(op.body)
        new_body = self.visit_stmt(op.body)
        if new_body.same_as(op.body):
            return op
        return AttrStmt(op.node, op.attr_key, op.value, new_body)


# =============================================================================
# Main Pass
# =============================================================================


def _detect_num_stages(stmt: Stmt) -> int:
    """Detect the number of pipeline stages from buffer shapes."""
    result = 2

    def visitor(node):
        nonlocal result
        if isinstance(node, Block):
            for buf in node.alloc_buffers:
                if is_shared(buf) and len(buf.shape) >= 2:
                    first_dim = buf.shape[0]
                    if isinstance(first_dim, IntImm):
                        n = first_dim.value
                        if 2 <= n <= 8:
                            result = max(result, n)

    post_order_visit(stmt, visitor)
    return result


def LowerAsyncCopy(verbose: bool = False):
    """
    Create a lower async copy pass.

    This pass transforms vectorized copy loops from global to shared memory
    into ptx_cp_async intrinsics with proper pipeline synchronization.

    The transformation:
    1. Uses pipeline_stage attr from InjectPipeline to determine stage info
    2. Replaces vectorized copies with ptx_cp_async calls
    3. Inserts commit_group/wait_group based on stage information

    Parameters
    ----------
    verbose : bool
        Whether to print debug information.

    Returns
    -------
    pass : tvm.transform.Pass
        The lower async copy pass.
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        # Auto-detect num_stages
        stages = _detect_num_stages(func.body)

        if verbose:
            print(f"[LowerAsyncCopy] num_stages={stages}")

        # Collect buffer mappings
        buffer_data_to_buffer: dict[Var, Buffer] = {}
        for _, buffer in func.buffer_map.items():
            buffer_data_to_buffer[buffer.data] = buffer

        # Step 1: Lower vectorized copies to ptx_cp_async
        lowerer = AsyncCopyLowerer(buffer_data_to_buffer, verbose)
        new_body = lowerer.visit_stmt(func.body)

        # Step 2: Insert synchronization if we have async copies
        if lowerer.has_async_copy:
            inserter = PipelineSyncInserter(stages, verbose)
            new_body = inserter.visit_stmt(new_body)

        # Step 3: Remove pipeline_stage attributes
        remover = PipelineAttrRemover()
        new_body = remover.visit_stmt(new_body)

        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.LowerAsyncCopy")
