"""
Inject Pipeline Pass for TileLang.

This module implements the pipeline injection pass that transforms loops with
software_pipeline_stage and software_pipeline_order annotations into an
unrolled form with Prologue + Steady State + Epilogue structure.

T_pipeline = UnrollPrologue(num_stages-1)
           ⊕ ModuloExpansion(body, num_stages)
           ⊕ UnrollEpilogue(num_stages-1)

Key features:
- Multi-buffering: Shared memory buffers are expanded to have num_stages copies
- Modulo scheduling: Each stage uses buffer[iteration % num_stages]
"""

from __future__ import annotations

from tilelang import tvm as tvm
from tvm import tir
from tvm.tir import (
    AttrStmt,
    Block,
    BlockRealize,
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    For,
    ForKind,
    IfThenElse,
    IntImm,
    LetStmt,
    PrimExpr,
    PrimFunc,
    PyStmtExprMutator,
    PyStmtExprVisitor,
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tir.transform import prim_func_pass

from tilelang.utils import is_shared


def _substitute_var(stmt: Stmt, var: Var, value: PrimExpr) -> Stmt:
    """Substitute a variable with a value in a statement."""
    var_map = {var: value}
    return tir.stmt_functor.substitute(stmt, var_map)


_OP_TILEOP_REGION = tir.op.Op.get("tl.tileop.region")


@tir.functor.visitor
class BufferCollector(PyStmtExprVisitor):
    """Collect buffers accessed in a statement."""

    def __init__(self):
        super().__init__()
        self.read_buffers: set[Buffer] = set()
        self.write_buffers: set[Buffer] = set()

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        self.read_buffers.add(op.buffer)

    def visit_buffer_store_(self, op: BufferStore) -> None:
        self.write_buffers.add(op.buffer)
        self.visit_expr(op.value)

    def visit_call_(self, op: Call) -> None:
        # Handle tl.region calls
        if op.op.same_as(_OP_TILEOP_REGION) and len(op.args) >= 2:
            buffer_load = op.args[0]
            access_type = op.args[1]
            if isinstance(buffer_load, BufferLoad):
                access_val = int(access_type)
                if access_val & 1:  # read
                    self.read_buffers.add(buffer_load.buffer)
                if access_val & 2:  # write
                    self.write_buffers.add(buffer_load.buffer)
        else:
            # Visit all arguments
            for arg in op.args:
                self.visit_expr(arg)


@tir.functor.mutator
class BufferRewriter(PyStmtExprMutator):
    """Rewrite buffer accesses to use expanded buffers with modulo indexing."""

    def __init__(self, buf: Buffer, expanded_buf: Buffer, stage_idx: PrimExpr, num_stages: int):
        super().__init__()
        self.buf = buf
        self.expanded_buf = expanded_buf
        self.modulo_idx = tir.floormod(stage_idx, IntImm("int32", num_stages))

    def visit_buffer_store_(self, op: BufferStore) -> Stmt:
        new_value = self.visit_expr(op.value)
        if op.buffer.same_as(self.buf):
            new_indices = [self.modulo_idx] + list(op.indices)
            return BufferStore(self.expanded_buf, new_value, new_indices)
        elif not new_value.same_as(op.value):
            return BufferStore(op.buffer, new_value, op.indices)
        return op

    def visit_buffer_load_(self, op: BufferLoad) -> PrimExpr:
        if op.buffer.same_as(self.buf):
            new_indices = [self.modulo_idx] + list(op.indices)
            return BufferLoad(self.expanded_buf, new_indices)
        return op

    def visit_call_(self, op: Call) -> PrimExpr:
        if op.op.same_as(_OP_TILEOP_REGION) and len(op.args) >= 2:
            buffer_load = op.args[0]
            if isinstance(buffer_load, BufferLoad) and buffer_load.buffer.same_as(self.buf):
                # Update BufferLoad indices: [i, j] -> [stage_idx % num_stages, i, j]
                new_indices = [self.modulo_idx] + list(buffer_load.indices)
                new_load = BufferLoad(self.expanded_buf, new_indices)
                # Update region shape: (128, 32) -> (1, 128, 32)
                # args format: [buffer_load, access_type, shape...]
                access_type = op.args[1]
                old_shape = list(op.args[2:])
                new_shape = [IntImm("int32", 1)] + old_shape
                new_args = [new_load, access_type] + new_shape
                return Call(op.dtype, op.op, new_args)
        # Visit all arguments
        new_args = [self.visit_expr(arg) if isinstance(arg, PrimExpr) else arg for arg in op.args]
        return Call(op.dtype, op.op, new_args)


@tir.functor.mutator
class PipelineInjector(PyStmtExprMutator):
    """
    Pipeline injector that transforms loops with pipeline annotations.

    The transformation converts:
        for i in range(N):
            S0  # stage 0 - copy A -> A_shared
            S1  # stage 0 - copy B -> B_shared
            S2  # stage 2 - gemm(A_shared, B_shared, C)

    Into (with multi-buffering):
        # Expand shared buffers: A_shared[num_stages, M, K], B_shared[num_stages, K, N]

        # Prologue: fill the pipeline
        copy A[0] -> A_shared[0]
        copy B[0] -> B_shared[0]
        copy A[1] -> A_shared[1]
        copy B[1] -> B_shared[1]

        # Steady state: main loop with modulo buffer indexing
        for i in range(N - num_stages + 1):
            copy A[i+2] -> A_shared[(i+2) % num_stages]
            copy B[i+2] -> B_shared[(i+2) % num_stages]
            gemm(A_shared[i % num_stages], B_shared[i % num_stages], C)

        # Epilogue: drain the pipeline
        gemm(A_shared[30 % num_stages], B_shared[30 % num_stages], C)
        gemm(A_shared[31 % num_stages], B_shared[31 % num_stages], C)
    """

    def __init__(self, buffer_data_to_buffer: dict[Var, Buffer], verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        self.buffer_data_to_buffer = buffer_data_to_buffer
        # Maps original buffer -> expanded buffer with extra stage dimension
        self.buffer_expansion_map: dict[Buffer, Buffer] = {}
        self.num_stages: int = 0

    def _create_expanded_buffer(self, buf: Buffer, num_stages: int) -> Buffer:
        """Create an expanded buffer with an extra dimension for stage indexing."""
        # Add num_stages as the first dimension
        new_shape = [IntImm("int32", num_stages)] + list(buf.shape)

        # Create new buffer with expanded shape, keep original name
        new_buf = tir.decl_buffer(
            new_shape,
            buf.dtype,
            buf.name,
            scope=buf.scope(),
        )
        return new_buf

    def _rewrite_buffer_access(self, stmt: Stmt, buf: Buffer, stage_idx: PrimExpr) -> Stmt:
        """Rewrite buffer accesses to use the expanded buffer with stage indexing.

        This transforms:
            buf[i, j] -> expanded_buf[stage_idx % num_stages, i, j]
        """
        if buf not in self.buffer_expansion_map:
            return stmt

        expanded_buf = self.buffer_expansion_map[buf]
        rewriter = BufferRewriter(buf, expanded_buf, stage_idx, self.num_stages)
        return rewriter.visit_stmt(stmt)

    def _collect_pipeline_buffers(self, body_stmts: list[Stmt], stages: list[int]) -> set[Buffer]:
        """Collect shared memory buffers that need multi-buffering.

        A buffer needs multi-buffering if it's written by an early stage (copy)
        and read by a later stage (compute).
        """
        stage_write_bufs: dict[int, set[Buffer]] = {}
        stage_read_bufs: dict[int, set[Buffer]] = {}

        for stmt, stage in zip(body_stmts, stages):
            collector = BufferCollector()
            collector.visit_stmt(stmt)

            if stage not in stage_write_bufs:
                stage_write_bufs[stage] = set()
            if stage not in stage_read_bufs:
                stage_read_bufs[stage] = set()

            for buf in collector.write_buffers:
                if is_shared(buf):
                    stage_write_bufs[stage].add(buf)

            for buf in collector.read_buffers:
                if is_shared(buf):
                    stage_read_bufs[stage].add(buf)

        # Find buffers written by early stages and read by later stages
        pipeline_buffers = set()
        all_stages = sorted(set(stages))

        for i, early_stage in enumerate(all_stages):
            for late_stage in all_stages[i + 1 :]:
                # Buffers written by early_stage and read by late_stage
                written = stage_write_bufs.get(early_stage, set())
                read = stage_read_bufs.get(late_stage, set())
                pipeline_buffers.update(written & read)

        return pipeline_buffers

    def visit_block_(self, op: Block) -> Block:
        """Visit a Block node."""
        new_body = self.visit_stmt(op.body)

        # Check if any buffers in alloc_buffers need to be replaced with expanded versions
        new_alloc_buffers = []
        buffers_to_remove = set()

        for buf in op.alloc_buffers:
            if buf in self.buffer_expansion_map:
                # Replace with expanded buffer
                expanded = self.buffer_expansion_map[buf]
                new_alloc_buffers.append(expanded)
                buffers_to_remove.add(buf)
            else:
                new_alloc_buffers.append(buf)

        # Check if anything changed
        if new_body.same_as(op.body) and not buffers_to_remove:
            return op

        return Block(
            op.iter_vars,
            op.reads,
            op.writes,
            op.name_hint,
            new_body,
            op.init,
            new_alloc_buffers,
            op.match_buffers,
            op.annotations,
        )

    def visit_for_(self, loop: For) -> Stmt:
        """Visit a For loop and potentially inject pipeline."""
        annotations = dict(loop.annotations)

        # Check for pipeline annotations
        stage_anno = annotations.get("software_pipeline_stage")
        order_anno = annotations.get("software_pipeline_order")

        if stage_anno is None or order_anno is None:
            # No pipeline annotations, just visit body recursively
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

        # Extract stage and order arrays
        stages = self._extract_int_array(stage_anno)
        orders = self._extract_int_array(order_anno)

        if not stages or not orders:
            return loop

        # Calculate num_stages
        num_stages = max(stages) + 1

        if self.verbose:
            print("\n[InjectPipeline] Transforming loop:")
            print(f"  stages: {stages}")
            print(f"  orders: {orders}")
            print(f"  num_stages: {num_stages}")
            print(f"  loop extent: {loop.extent}")

        # Get the loop body statements
        body_stmts = self._get_body_statements(loop.body)

        if len(body_stmts) != len(stages):
            if self.verbose:
                print(f"  Warning: stmt count ({len(body_stmts)}) != stage count ({len(stages)})")
            return loop

        # Create the pipelined version
        return self._create_pipelined_loop(loop, body_stmts, stages, orders, num_stages)

    def _extract_int_array(self, anno) -> list[int]:
        """Extract integer array from annotation."""
        result = []
        try:
            for item in anno:
                if isinstance(item, IntImm):
                    result.append(item.value)
                elif isinstance(item, int):
                    result.append(item)
                else:
                    result.append(int(item))
        except (TypeError, ValueError):
            return []
        return result

    def _get_body_statements(self, body: Stmt, max_depth: int = 100) -> list[Stmt]:
        """Extract statements from loop body by navigating through wrapper nodes."""
        current = body
        for _ in range(max_depth):
            if isinstance(current, SeqStmt):
                return list(current.seq)
            elif isinstance(current, BlockRealize):
                current = current.block.body
            elif isinstance(current, Block):
                current = current.body
            elif isinstance(current, IfThenElse) and current.else_case is None:
                current = current.then_case
            elif isinstance(current, LetStmt):
                current = current.body
            else:
                return [current]

        raise RuntimeError(
            "InjectPipeline: Exceeded maximum depth while extracting body statements. "
            "This may indicate malformed IR or an unexpected statement structure."
        )

    def _rewrite_stmt_with_multibuffer(self, stmt: Stmt, loop_iter: PrimExpr, pipeline_buffers: set[Buffer]) -> Stmt:
        """Rewrite a statement to use multi-buffered buffers with modulo indexing."""
        result = stmt
        for buf in pipeline_buffers:
            if buf in self.buffer_expansion_map:
                result = self._rewrite_buffer_access(result, buf, loop_iter)
        return result

    def _wrap_with_pipeline_stage(self, stmt: Stmt, stage_expr: PrimExpr) -> Stmt:
        """Wrap a statement with pipeline_stage attribute.

        This attribute is used by LowerAsyncCopy to determine synchronization points.

        Parameters
        ----------
        stmt : Stmt
            The statement to wrap.
        stage_expr : PrimExpr
            The stage expression, e.g., IntImm(0) for prologue or (ko + 2) % 3 for steady state.

        Returns
        -------
        Stmt
            The statement wrapped with pipeline_stage attribute.
        """
        return AttrStmt(tir.StringImm("pipeline"), "pipeline_stage", stage_expr, stmt)

    def _create_pipelined_loop(
        self,
        loop: For,
        body_stmts: list[Stmt],
        stages: list[int],
        orders: list[int],
        num_stages: int,
    ) -> Stmt:
        """
        Create the pipelined loop structure with multi-buffering.

        With stages [0, 0, 2] (copy, copy, gemm) and num_stages=3:

        Prologue (fill the pipeline, 2 iterations):
            iter 0: copy A[0] -> A_shared[0], copy B[0] -> B_shared[0]
            iter 1: copy A[1] -> A_shared[1], copy B[1] -> B_shared[1]

        Steady State (main loop, extent - num_stages + 1 iterations):
            iter i: copy A[i+2] -> A_shared[(i+2) % 3]
                    copy B[i+2] -> B_shared[(i+2) % 3]
                    gemm(A_shared[i % 3], B_shared[i % 3])

        Epilogue (drain the pipeline, 2 iterations):
            iter 0: gemm(A_shared[30 % 3], B_shared[30 % 3])
            iter 1: gemm(A_shared[31 % 3], B_shared[31 % 3])
        """
        loop_var = loop.loop_var
        loop_min = loop.min
        loop_extent = loop.extent
        max_stage = max(stages)
        self.num_stages = num_stages

        # Collect buffers that need multi-buffering
        pipeline_buffers = self._collect_pipeline_buffers(body_stmts, stages)

        if self.verbose and pipeline_buffers:
            print(f"  Multi-buffering buffers: {[b.name for b in pipeline_buffers]}")

        # Create expanded buffers
        for buf in pipeline_buffers:
            expanded = self._create_expanded_buffer(buf, num_stages)
            self.buffer_expansion_map[buf] = expanded
            if self.verbose:
                print(f"    {buf.name}: {list(buf.shape)} -> {expanded.name}: {list(expanded.shape)}")

        # Sort statements by order
        stmt_order = list(zip(body_stmts, stages, orders))
        stmt_order.sort(key=lambda x: x[2])  # Sort by order

        all_stmts = []

        # ========== Prologue ==========
        prologue_stmts = []
        for prologue_iter in range(num_stages - 1):
            iter_stmts = []
            for stmt, stage, _order in stmt_order:
                if stage <= prologue_iter:
                    # This statement works on loop iteration: prologue_iter - stage
                    actual_iter = prologue_iter - stage
                    new_stmt = _substitute_var(stmt, loop_var, loop_min + IntImm("int32", actual_iter))
                    # Apply multi-buffer rewriting
                    new_stmt = self._rewrite_stmt_with_multibuffer(new_stmt, IntImm("int32", actual_iter), pipeline_buffers)
                    # Wrap with pipeline_stage attribute (stage = actual_iter for prologue)
                    stage_expr = IntImm("int32", actual_iter)
                    new_stmt = self._wrap_with_pipeline_stage(new_stmt, stage_expr)
                    iter_stmts.append(new_stmt)

            if iter_stmts:
                if self.verbose:
                    print(f"  Prologue iter {prologue_iter}: {len(iter_stmts)} statements")
                if len(iter_stmts) == 1:
                    prologue_stmts.append(iter_stmts[0])
                else:
                    prologue_stmts.append(SeqStmt(iter_stmts))

        # ========== Steady State ==========
        steady_state_extent = loop_extent - IntImm("int32", num_stages - 1)
        steady_var = Var(loop_var.name, loop_var.dtype)

        steady_stmts = []
        for stmt, stage, _order in stmt_order:
            offset = max_stage - stage
            if offset > 0:
                actual_iter = steady_var + IntImm("int32", offset)
            else:
                actual_iter = steady_var
            new_stmt = _substitute_var(stmt, loop_var, loop_min + actual_iter)
            # Apply multi-buffer rewriting with the actual iteration for this stage
            new_stmt = self._rewrite_stmt_with_multibuffer(new_stmt, actual_iter, pipeline_buffers)
            # Wrap with pipeline_stage attribute
            # stage_expr = (steady_var + offset) % num_stages
            stage_expr = tir.floormod(actual_iter, IntImm("int32", num_stages))
            new_stmt = self._wrap_with_pipeline_stage(new_stmt, stage_expr)
            steady_stmts.append(new_stmt)

        if steady_stmts:
            steady_body = SeqStmt(steady_stmts) if len(steady_stmts) > 1 else steady_stmts[0]

            new_annotations = {
                k: v for k, v in loop.annotations.items() if k not in ("software_pipeline_stage", "software_pipeline_order", "num_stages")
            }

            steady_loop = For(
                steady_var,
                IntImm("int32", 0),
                steady_state_extent,
                ForKind.SERIAL,
                steady_body,
                None,
                new_annotations,
            )
            all_stmts.extend(prologue_stmts)
            all_stmts.append(steady_loop)

        # ========== Epilogue ==========
        # After steady state ends at pipeline_iter = steady_state_extent - 1:
        # - Stage s has processed iteration: steady_state_extent - 1 + (max_stage - s)
        # In epilogue_iter e, stage s (where s > e) processes:
        #   actual_iter = steady_state_extent + (max_stage - s) + e
        epilogue_stmts = []
        for epilogue_iter in range(num_stages - 1):
            iter_stmts = []
            for stmt, stage, _order in stmt_order:
                if stage > epilogue_iter:
                    # Actual iteration for this statement
                    # steady_state_extent = loop_extent - num_stages + 1
                    actual_iter = (
                        loop_extent - IntImm("int32", num_stages - 1) + IntImm("int32", max_stage - stage) + IntImm("int32", epilogue_iter)
                    )
                    new_stmt = _substitute_var(stmt, loop_var, loop_min + actual_iter)
                    # Apply multi-buffer rewriting
                    new_stmt = self._rewrite_stmt_with_multibuffer(new_stmt, actual_iter, pipeline_buffers)
                    # Wrap with pipeline_stage attribute
                    # For epilogue, stage_expr = actual_iter % num_stages
                    stage_expr = tir.floormod(actual_iter, IntImm("int32", num_stages))
                    new_stmt = self._wrap_with_pipeline_stage(new_stmt, stage_expr)
                    iter_stmts.append(new_stmt)

            if iter_stmts:
                if self.verbose:
                    print(f"  Epilogue iter {epilogue_iter}: {len(iter_stmts)} statements")
                if len(iter_stmts) == 1:
                    epilogue_stmts.append(iter_stmts[0])
                else:
                    epilogue_stmts.append(SeqStmt(iter_stmts))

        all_stmts.extend(epilogue_stmts)

        if self.verbose:
            print(f"  Total: {len(prologue_stmts)} prologue + 1 steady loop + {len(epilogue_stmts)} epilogue")

        result = SeqStmt(all_stmts) if len(all_stmts) > 1 else all_stmts[0]
        return result


def InjectPipeline(verbose: bool = False):
    """
    Create a pipeline injection pass.

    This pass transforms loops with `software_pipeline_stage` and
    `software_pipeline_order` annotations into an unrolled form with
    Prologue + Steady State + Epilogue structure.

    Parameters
    ----------
    verbose : bool
        Whether to print debug information.

    Returns
    -------
    pass : tvm.transform.Pass
        The pipeline injection pass.
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        # Collect buffer mappings
        buffer_data_to_buffer: dict[Var, Buffer] = {}
        for _, buffer in func.buffer_map.items():
            buffer_data_to_buffer[buffer.data] = buffer

        injector = PipelineInjector(buffer_data_to_buffer, verbose)
        new_body = injector.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.InjectPipeline")
