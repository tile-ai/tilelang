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
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tir.stmt_functor import post_order_visit
from tvm.tir.transform import prim_func_pass


def _substitute_var(stmt: Stmt, var: Var, value: PrimExpr) -> Stmt:
    """Substitute a variable with a value in a statement."""
    var_map = {var: value}
    return tir.stmt_functor.substitute(stmt, var_map)


def _get_buffer_scope(buf: Buffer) -> str:
    """Get the storage scope of a buffer."""
    ptr_type = buf.data.type_annotation
    if hasattr(ptr_type, "storage_scope"):
        return ptr_type.storage_scope
    return ""


def _is_shared_buffer(buf: Buffer) -> bool:
    """Check if buffer is in shared memory."""
    scope = _get_buffer_scope(buf)
    return scope in ("shared", "shared.dyn")


class BufferCollector:
    """Collect buffers accessed in a statement."""

    def __init__(self):
        self.read_buffers: set[Buffer] = set()
        self.write_buffers: set[Buffer] = set()

    def collect(self, stmt: Stmt) -> None:
        """Collect all buffers accessed in the statement."""

        def visit(node):
            if isinstance(node, BufferLoad):
                self.read_buffers.add(node.buffer)
            elif isinstance(node, BufferStore):
                self.write_buffers.add(node.buffer)
            elif isinstance(node, Call):
                # Handle tl.region calls
                op_name = str(node.op)
                if "tl.tileop.region" in op_name and len(node.args) >= 2:
                    buffer_load = node.args[0]
                    access_type = node.args[1]
                    if isinstance(buffer_load, BufferLoad):
                        access_val = access_type.value if isinstance(access_type, IntImm) else int(access_type)
                        if access_val & 1:  # read
                            self.read_buffers.add(buffer_load.buffer)
                        if access_val & 2:  # write
                            self.write_buffers.add(buffer_load.buffer)

        post_order_visit(stmt, visit)


class BufferRewriter:
    """Rewrite buffer accesses to use expanded buffers with modulo indexing."""

    def __init__(self, buf: Buffer, expanded_buf: Buffer, stage_idx: PrimExpr, num_stages: int):
        self.buf = buf
        self.expanded_buf = expanded_buf
        self.modulo_idx = tir.floormod(stage_idx, IntImm("int32", num_stages))

    def rewrite(self, stmt: Stmt) -> Stmt:
        """Rewrite buffer accesses in the statement."""
        return self._visit_stmt(stmt)

    def _visit_stmt(self, stmt: Stmt) -> Stmt:
        if isinstance(stmt, BufferStore):
            new_value = self._visit_expr(stmt.value)
            if stmt.buffer.same_as(self.buf):
                new_indices = [self.modulo_idx] + list(stmt.indices)
                return BufferStore(self.expanded_buf, new_value, new_indices)
            elif new_value is not stmt.value:
                return BufferStore(stmt.buffer, new_value, stmt.indices)
            return stmt
        elif isinstance(stmt, SeqStmt):
            new_seq = [self._visit_stmt(s) for s in stmt.seq]
            return SeqStmt(new_seq)
        elif isinstance(stmt, IfThenElse):
            new_cond = self._visit_expr(stmt.condition)
            new_then = self._visit_stmt(stmt.then_case)
            new_else = self._visit_stmt(stmt.else_case) if stmt.else_case else None
            return IfThenElse(new_cond, new_then, new_else)
        elif isinstance(stmt, LetStmt):
            new_value = self._visit_expr(stmt.value)
            new_body = self._visit_stmt(stmt.body)
            return LetStmt(stmt.var, new_value, new_body)
        elif isinstance(stmt, AttrStmt):
            new_body = self._visit_stmt(stmt.body)
            return AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)
        elif isinstance(stmt, For):
            new_body = self._visit_stmt(stmt.body)
            return For(stmt.loop_var, stmt.min, stmt.extent, stmt.kind, new_body, stmt.thread_binding, stmt.annotations)
        elif isinstance(stmt, Block):
            new_body = self._visit_stmt(stmt.body)
            return Block(
                stmt.iter_vars,
                stmt.reads,
                stmt.writes,
                stmt.name_hint,
                new_body,
                stmt.init,
                stmt.alloc_buffers,
                stmt.match_buffers,
                stmt.annotations,
            )
        elif isinstance(stmt, BlockRealize):
            new_block = self._visit_stmt(stmt.block)
            return BlockRealize(stmt.iter_values, stmt.predicate, new_block)
        elif isinstance(stmt, tir.Evaluate):
            new_value = self._visit_expr(stmt.value)
            return tir.Evaluate(new_value)
        else:
            return stmt

    def _visit_expr(self, expr: PrimExpr) -> PrimExpr:
        if isinstance(expr, BufferLoad):
            if expr.buffer.same_as(self.buf):
                new_indices = [self.modulo_idx] + list(expr.indices)
                return BufferLoad(self.expanded_buf, new_indices)
            return expr
        elif isinstance(expr, Call):
            op_name = str(expr.op)
            if "tl.tileop.region" in op_name and len(expr.args) >= 2:
                buffer_load = expr.args[0]
                if isinstance(buffer_load, BufferLoad) and buffer_load.buffer.same_as(self.buf):
                    # Update BufferLoad indices: [i, j] -> [stage_idx % num_stages, i, j]
                    new_indices = [self.modulo_idx] + list(buffer_load.indices)
                    new_load = BufferLoad(self.expanded_buf, new_indices)
                    # Update region shape: (128, 32) -> (1, 128, 32)
                    # args format: [buffer_load, access_type, shape...]
                    access_type = expr.args[1]
                    old_shape = list(expr.args[2:])
                    new_shape = [IntImm("int32", 1)] + old_shape
                    new_args = [new_load, access_type] + new_shape
                    return Call(expr.dtype, expr.op, new_args)
            # Visit all arguments
            new_args = [self._visit_expr(arg) if isinstance(arg, PrimExpr) else arg for arg in expr.args]
            return Call(expr.dtype, expr.op, new_args)
        elif isinstance(expr, tir.Cast):
            return tir.Cast(expr.dtype, self._visit_expr(expr.value))
        elif isinstance(expr, tir.Add):
            return tir.Add(self._visit_expr(expr.a), self._visit_expr(expr.b))
        elif isinstance(expr, tir.Sub):
            return tir.Sub(self._visit_expr(expr.a), self._visit_expr(expr.b))
        elif isinstance(expr, tir.Mul):
            return tir.Mul(self._visit_expr(expr.a), self._visit_expr(expr.b))
        elif isinstance(expr, tir.Div):
            return tir.Div(self._visit_expr(expr.a), self._visit_expr(expr.b))
        elif isinstance(expr, tir.FloorDiv):
            return tir.FloorDiv(self._visit_expr(expr.a), self._visit_expr(expr.b))
        elif isinstance(expr, tir.FloorMod):
            return tir.FloorMod(self._visit_expr(expr.a), self._visit_expr(expr.b))
        elif isinstance(expr, tir.Max):
            return tir.Max(self._visit_expr(expr.a), self._visit_expr(expr.b))
        elif isinstance(expr, tir.Min):
            return tir.Min(self._visit_expr(expr.a), self._visit_expr(expr.b))
        elif isinstance(expr, tir.Select):
            return tir.Select(self._visit_expr(expr.condition), self._visit_expr(expr.true_value), self._visit_expr(expr.false_value))
        else:
            return expr


class PipelineInjector:
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

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.buffer_data_to_buffer: dict[Var, Buffer] = {}
        # Maps original buffer -> expanded buffer with extra stage dimension
        self.buffer_expansion_map: dict[Buffer, Buffer] = {}
        self.num_stages: int = 0

    def transform(self, func: PrimFunc) -> Stmt:
        """Apply pipeline injection transformation to a PrimFunc."""
        # Collect buffer mappings
        for _, buffer in func.buffer_map.items():
            self.buffer_data_to_buffer[buffer.data] = buffer
        return self._visit_stmt(func.body)

    def _create_expanded_buffer(self, buf: Buffer, num_stages: int) -> Buffer:
        """Create an expanded buffer with an extra dimension for stage indexing."""
        # Add num_stages as the first dimension
        new_shape = [IntImm("int32", num_stages)] + list(buf.shape)

        # Create new buffer with expanded shape
        new_name = f"{buf.name}_{num_stages}"
        new_buf = tir.decl_buffer(
            new_shape,
            buf.dtype,
            new_name,
            scope=_get_buffer_scope(buf),
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
        return rewriter.rewrite(stmt)

    def _collect_pipeline_buffers(self, body_stmts: list[Stmt], stages: list[int]) -> set[Buffer]:
        """Collect shared memory buffers that need multi-buffering.

        A buffer needs multi-buffering if it's written by an early stage (copy)
        and read by a later stage (compute).
        """
        stage_write_bufs: dict[int, set[Buffer]] = {}
        stage_read_bufs: dict[int, set[Buffer]] = {}

        for stmt, stage in zip(body_stmts, stages):
            collector = BufferCollector()
            collector.collect(stmt)

            if stage not in stage_write_bufs:
                stage_write_bufs[stage] = set()
            if stage not in stage_read_bufs:
                stage_read_bufs[stage] = set()

            for buf in collector.write_buffers:
                if _is_shared_buffer(buf):
                    stage_write_bufs[stage].add(buf)

            for buf in collector.read_buffers:
                if _is_shared_buffer(buf):
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

    def _visit_stmt(self, stmt: Stmt) -> Stmt:
        """Recursively visit and transform statements."""
        if isinstance(stmt, For):
            return self._visit_for(stmt)
        elif isinstance(stmt, Block):
            return self._visit_block(stmt)
        elif isinstance(stmt, BlockRealize):
            new_block = self._visit_stmt(stmt.block)
            if new_block.same_as(stmt.block):
                return stmt
            return BlockRealize(stmt.iter_values, stmt.predicate, new_block)
        elif isinstance(stmt, SeqStmt):
            new_seq = [self._visit_stmt(s) for s in stmt.seq]
            return SeqStmt(new_seq)
        elif isinstance(stmt, IfThenElse):
            new_then = self._visit_stmt(stmt.then_case)
            new_else = self._visit_stmt(stmt.else_case) if stmt.else_case else None
            return IfThenElse(stmt.condition, new_then, new_else)
        elif isinstance(stmt, LetStmt):
            new_body = self._visit_stmt(stmt.body)
            return LetStmt(stmt.var, stmt.value, new_body)
        elif isinstance(stmt, AttrStmt):
            new_body = self._visit_stmt(stmt.body)
            if new_body.same_as(stmt.body):
                return stmt
            return AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)
        else:
            return stmt

    def _visit_block(self, op: Block) -> Block:
        """Visit a Block node."""
        new_body = self._visit_stmt(op.body)

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

    def _visit_for(self, loop: For) -> Stmt:
        """Visit a For loop and potentially inject pipeline."""
        annotations = dict(loop.annotations)

        # Check for pipeline annotations
        stage_anno = annotations.get("software_pipeline_stage")
        order_anno = annotations.get("software_pipeline_order")

        if stage_anno is None or order_anno is None:
            # No pipeline annotations, just visit body recursively
            new_body = self._visit_stmt(loop.body)
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

    def _get_body_statements(self, body: Stmt) -> list[Stmt]:
        """Extract statements from loop body."""
        # Navigate through BlockRealize/Block to get to SeqStmt
        current = body
        while True:
            if isinstance(current, SeqStmt):
                return list(current.seq)
            elif isinstance(current, BlockRealize):
                current = current.block.body
            elif isinstance(current, Block):
                current = current.body
            elif isinstance(current, IfThenElse):
                if current.else_case is None:
                    current = current.then_case
                else:
                    # Can't handle if-else in pipeline body
                    return [body]
            elif isinstance(current, LetStmt):
                current = current.body
            else:
                # Single statement
                return [current]

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
                k: v
                for k, v in loop.annotations.items()
                if k not in ("software_pipeline_stage", "software_pipeline_order", "software_pipeline_async_stages", "num_stages")
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

        # Wrap with buffer allocations for expanded buffers
        result = SeqStmt(all_stmts) if len(all_stmts) > 1 else all_stmts[0]

        # Note: The expanded buffer allocations should be handled at the Block level
        # For now, we just return the transformed statements
        # TODO: Add alloc_buffer for expanded buffers in the enclosing block

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
        injector = PipelineInjector(verbose)
        new_body = injector.transform(func)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.InjectPipeline")
