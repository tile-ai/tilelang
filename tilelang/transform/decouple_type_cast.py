"""
Decouple type cast vectorization constraints.

When a vectorized loop has mixed-precision operations between local and memory
buffers, the vectorization length would be constrained by the GCD of all
involved dtypes.

This pass decouples the constraints by inserting a local buffer as an
intermediate stage, allowing optimal vectorization for both computation and
memory access.

Two cases are handled:

Case 1: local → memory (store to memory with mixed types)
---------------------------------------------------------
Before:
    for vec in T.vectorized(16):
        b[vec] = T.cast(a_frag[vec], "float4_e2m1fn")

After:
    for vec in T.vectorized(16):
        cast_buf[vec] = T.cast(a_frag[vec], "float4_e2m1fn")  # compute
    for vec_copy in T.vectorized(16):
        b[vec_copy] = cast_buf[vec_copy]                      # copy to memory

Case 2: memory → local (load from memory with different dtype)
--------------------------------------------------------------
Before:
    for vec in T.vectorized(16):
        a_frag[vec] = T.cast(b[vec], "float32")

After:
    for vec_copy in T.vectorized(16):
        cast_buf[vec_copy] = b[vec_copy]                      # copy from memory
    for vec in T.vectorized(16):
        a_frag[vec] = T.cast(cast_buf[vec], "float32")        # compute
"""

from __future__ import annotations

from tvm import tir
from tvm.ir import Op
from tvm.tir import (
    Allocate,
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    DeclBuffer,
    For,
    ForKind,
    IfThenElse,
    IntImm,
    PrimFunc,
    PyStmtExprVisitor,
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tir.stmt_functor import post_order_visit, substitute
from tvm.tir.transform import prim_func_pass

# Cache the Op for if_then_else to avoid repeated lookups
_IF_THEN_ELSE_OP = Op.get("tir.if_then_else")

from tilelang.utils.language import is_fragment, is_global, is_local, is_local_var, is_shared


def is_local_buffer(buffer: Buffer) -> bool:
    """Check if a buffer is local (register-level), including local.var."""
    if buffer is None:
        return False
    return is_local(buffer) or is_fragment(buffer) or is_local_var(buffer)


def is_global_or_shared_buffer(buffer: Buffer) -> bool:
    """Check if a buffer is a global or shared buffer."""
    if buffer is None:
        return False
    return is_global(buffer) or is_shared(buffer)


def validate_buffer_scope(buffer: Buffer) -> None:
    """Validate that buffer has a known scope.

    Raises:
        ValueError: If buffer scope is unknown or empty.
    """
    if buffer is None:
        return
    if not is_local_buffer(buffer) and not is_global_or_shared_buffer(buffer):
        raise ValueError(
            f"Unknown buffer scope '{buffer.scope()}' for buffer '{buffer.name}'. "
            f"Expected one of: local, local.fragment, local.var, global, shared, shared.dyn"
        )


@tir.functor.visitor
class MixedTypeChecker(PyStmtExprVisitor):
    """Check if expression contains BufferLoads with different dtypes, skipping indices."""

    def __init__(self, target_dtype: str):
        super().__init__()
        self.target_dtype = str(target_dtype)
        self.found_different = False

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        if str(op.buffer.dtype) != self.target_dtype:
            self.found_different = True
        # Skip indices traversal


def has_mixed_types(expr: tir.PrimExpr, target_dtype: str) -> bool:
    """Check if expression contains BufferLoads with different dtypes than target.

    If any BufferLoad in the expression has a different dtype than the target
    (store buffer's dtype), vectorization may be constrained by GCD of all dtypes.
    """
    checker = MixedTypeChecker(target_dtype)
    checker.visit_expr(expr)
    return checker.found_different


@tir.functor.visitor
class GlobalSharedBufferLoadCollector(PyStmtExprVisitor):
    """Collect BufferLoads from global/shared buffers, skipping if_then_else conditions.

    The condition part of if_then_else doesn't participate in type casting,
    so we skip collecting BufferLoads from there.
    """

    def __init__(self, skip_if_then_else_cond: bool = False):
        super().__init__()
        self.result: list[BufferLoad] = []
        self.skip_if_then_else_cond = skip_if_then_else_cond

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        if is_global_or_shared_buffer(op.buffer):
            self.result.append(op)

    def visit_call_(self, op: Call) -> None:
        if self.skip_if_then_else_cond and op.op.same_as(_IF_THEN_ELSE_OP):
            # Skip condition (args[0]), only visit true/false values (args[1], args[2])
            self.visit_expr(op.args[1])
            self.visit_expr(op.args[2])
        else:
            # Visit all arguments normally
            for arg in op.args:
                self.visit_expr(arg)


def get_global_or_shared_buffer_loads(expr: tir.PrimExpr, skip_if_then_else_cond: bool = False) -> list[BufferLoad]:
    """Get BufferLoads from global/shared buffers in the expression.

    Args:
        expr: The expression to search.
        skip_if_then_else_cond: If True, skip BufferLoads in if_then_else conditions,
            since they don't participate in type casting.
    """
    collector = GlobalSharedBufferLoadCollector(skip_if_then_else_cond)
    collector.visit_expr(expr)
    return collector.result


def has_global_or_shared_load_with_different_dtype(expr: tir.PrimExpr, target_dtype: str) -> bool:
    """Check if expression has global/shared BufferLoad with different dtype than target.

    Used to detect memory→local cases where we need to insert cast buffer.
    Skips if_then_else condition since it doesn't participate in type casting.
    """
    target_dtype = str(target_dtype)
    return any(str(load.buffer.dtype) != target_dtype for load in get_global_or_shared_buffer_loads(expr, skip_if_then_else_cond=True))


@tir.functor.visitor
class StoreCollector(PyStmtExprVisitor):
    """Collect BufferStore nodes that need transformation, skipping indices traversal.

    This avoids visiting BufferLoad/BufferStore nodes inside indices, which don't
    participate in the type casting transformation.
    """

    def __init__(self):
        super().__init__()
        self.local_to_memory: list[BufferStore] = []
        self.memory_to_local: list[BufferStore] = []

    def visit_buffer_store_(self, op: BufferStore) -> None:
        validate_buffer_scope(op.buffer)
        # Case 1: store to memory with mixed types
        if is_global_or_shared_buffer(op.buffer) and has_mixed_types(op.value, op.buffer.dtype):
            self.local_to_memory.append(op)
        # Case 2: store to local with memory load of different dtype
        elif is_local_buffer(op.buffer) and has_global_or_shared_load_with_different_dtype(op.value, op.buffer.dtype):
            self.memory_to_local.append(op)
        # Only visit value, skip indices
        self.visit_expr(op.value)

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        # Skip indices traversal for BufferLoad as well
        pass


def contains_seq_stmt(stmt: Stmt) -> bool:
    """Check if statement contains SeqStmt (multiple statements).

    When the For body has SeqStmt, the transformation is more complex
    and we skip the optimization for now.
    """
    found = False

    def visitor(node) -> None:
        nonlocal found
        if isinstance(node, SeqStmt):
            found = True

    post_order_visit(stmt, visitor)
    return found


def extract_if_condition(stmt: Stmt) -> tuple[tir.PrimExpr | None, Stmt]:
    """Extract IfThenElse condition from statement if present.

    Returns:
        A tuple of (condition, inner_body). If no IfThenElse, returns (None, stmt).
    """
    if isinstance(stmt, IfThenElse) and stmt.else_case is None:
        return stmt.condition, stmt.then_case
    return None, stmt


# Type alias for cast buffer mapping
# Maps original buffer -> (cast buffer, original indices)
CastBufferMap = dict[Buffer, tuple[Buffer, list[tir.PrimExpr]]]


@tir.functor.mutator
class DecoupleTypeCastMutator(tir.PyStmtExprMutator):
    """Mutator that decouples type cast vectorization constraints.

    This mutator transforms vectorized loops that store to memory buffers
    (global/shared) with mixed-precision expressions by inserting local
    cache buffers as intermediate stages.
    """

    def __init__(self):
        super().__init__()
        self._var_counter = 0

    def _make_unique_name(self, base: str) -> str:
        """Generate a unique name with incrementing counter."""
        name = f"{base}"
        if self._var_counter > 0:
            name += f"_{self._var_counter}"
        self._var_counter += 1
        return name

    def _make_for(self, original: For, new_body: Stmt) -> For:
        """Create a new For node with updated body, preserving other attributes."""
        return For(
            original.loop_var,
            original.min,
            original.extent,
            original.kind,
            new_body,
            original.thread_binding,
            original.annotations,
            original.step,
        )

    def visit_for_(self, op: For) -> Stmt:
        """Visit For nodes, transforming vectorized loops with mixed-type stores."""
        # Recursively visit body to handle nested loops
        new_body = self.visit_stmt(op.body)

        # Only transform vectorized loops
        if op.kind != ForKind.VECTORIZED:
            return self._make_for(op, new_body) if new_body is not op.body else op

        # Skip transformation for complex cases with multiple statements
        # Currently we only handle simple single BufferStore cases
        if contains_seq_stmt(new_body):
            return self._make_for(op, new_body) if new_body is not op.body else op

        # Collect stores that need transformation
        local_to_memory, memory_to_local = self._collect_stores_to_transform(new_body)
        if local_to_memory:
            return self._transform_local_to_memory(op, local_to_memory)
        elif memory_to_local:
            return self._transform_memory_to_local(op, memory_to_local)
        else:
            return self._make_for(op, new_body) if new_body is not op.body else op

    def _collect_stores_to_transform(self, stmt: Stmt) -> tuple[list[BufferStore], list[BufferStore]]:
        """Collect BufferStore nodes that need local cast buffer insertion.

        Returns two lists:
        1. local_to_memory: stores to memory buffer with mixed-type values
           (compute → cast buffer → copy to memory)
        2. memory_to_local: stores to local buffer with memory buffer loads of different dtype
           (copy from memory → cast buffer → compute)

        Note: Vectorized for is always the innermost loop, so no nested For handling needed.
        """
        collector = StoreCollector()
        collector.visit_stmt(stmt)
        return collector.local_to_memory, collector.memory_to_local

    def _transform_local_to_memory(self, op: For, stores_to_transform: list[BufferStore]) -> Stmt:
        """Transform local→memory: compute to cast buffer, then copy to memory.

        Before:
            b[i] = cast(a_frag[i], fp4)

        After:
            cast_buf[i] = cast(a_frag[i], fp4)  # compute to cast buffer
            b[i] = cast_buf[i]                   # copy to memory
        """
        # Skip dynamic extents
        if not isinstance(op.extent, IntImm):
            return op

        # Extract condition if the body is wrapped in IfThenElse
        condition, _ = extract_if_condition(op.body)

        # Create cast buffers for each unique target buffer (memory buffer)
        cast_buffers = self._create_cast_buffers_for_stores(stores_to_transform, op.extent.value)

        # Build compute loop (stores to local cast buffer)
        compute_body = self._replace_stores_with_cast(op.body, cast_buffers, op.loop_var)
        compute_loop = self._make_vectorized_loop(op, compute_body)

        # Build copy loops (transfer from cast buffer to memory, with condition if present)
        copy_loops = self._create_copy_loops_to_memory(op, cast_buffers, condition)

        # Combine: compute → copy
        all_stmts = [compute_loop] + copy_loops
        result: Stmt = SeqStmt(all_stmts) if len(all_stmts) > 1 else all_stmts[0]

        # Wrap with buffer declarations and allocations
        result = self._wrap_with_allocations(result, cast_buffers)

        return result

    def _transform_memory_to_local(self, op: For, stores_to_transform: list[BufferStore]) -> Stmt:
        """Transform memory→local: copy from memory to cast buffer, then compute.

        Before:
            a_frag[i] = cast(b[i], fp32)

        After:
            cast_buf[i] = b[i]                   # copy from memory to cast buffer
            a_frag[i] = cast(cast_buf[i], fp32)  # compute from cast buffer
        """
        # Skip dynamic extents
        if not isinstance(op.extent, IntImm):
            return op

        # Extract condition if the body is wrapped in IfThenElse
        condition, _ = extract_if_condition(op.body)

        # Collect memory buffer loads that need cast buffering
        memory_loads = self._collect_memory_loads_to_cast(stores_to_transform)
        if not memory_loads:
            return op

        # Create cast buffers for each unique source buffer (memory buffer)
        cast_buffers = self._create_cast_buffers_for_loads(memory_loads, op.extent.value)

        # Build copy loops (transfer from memory to cast buffer, with condition if present)
        copy_loops = self._create_copy_loops_from_memory(op, cast_buffers, condition)

        # Build compute loop (replace memory loads with cast buffer loads)
        compute_body = self._replace_loads_with_cast(op.body, cast_buffers, op.loop_var)
        compute_loop = self._make_vectorized_loop(op, compute_body)

        # Combine: copy → compute
        all_stmts = copy_loops + [compute_loop]
        result: Stmt = SeqStmt(all_stmts) if len(all_stmts) > 1 else all_stmts[0]

        # Wrap with buffer declarations and allocations
        result = self._wrap_with_allocations(result, cast_buffers)

        return result

    def _collect_memory_loads_to_cast(self, stores: list[BufferStore]) -> list[BufferLoad]:
        """Collect memory BufferLoads from store values that need cast buffering."""
        result: list[BufferLoad] = []
        seen_buffers = set()
        for store in stores:
            for load in get_global_or_shared_buffer_loads(store.value, skip_if_then_else_cond=True):
                if load.buffer not in seen_buffers:
                    result.append(load)
                    seen_buffers.add(load.buffer)
        return result

    def _create_cast_buffers_for_stores(self, stores: list[BufferStore], extent: int) -> CastBufferMap:
        """Create local cast buffers for store targets (memory buffers)."""
        cast_buffers: CastBufferMap = {}

        for store in stores:
            if store.buffer in cast_buffers:
                continue

            cache_name = self._make_unique_name(f"{store.buffer.name}_local_cast")
            cast_buffer = tir.decl_buffer(
                shape=(extent,),
                dtype=store.buffer.dtype,
                name=cache_name,
                scope="local",
            )
            cast_buffers[store.buffer] = (cast_buffer, list(store.indices))

        return cast_buffers

    def _create_cast_buffers_for_loads(self, loads: list[BufferLoad], extent: int) -> CastBufferMap:
        """Create local cast buffers for load sources (memory buffers)."""
        cast_buffers: CastBufferMap = {}

        for load in loads:
            if load.buffer in cast_buffers:
                continue

            cache_name = self._make_unique_name(f"{load.buffer.name}_local_cast")
            cast_buffer = tir.decl_buffer(
                shape=(extent,),
                dtype=load.buffer.dtype,
                name=cache_name,
                scope="local",
            )
            cast_buffers[load.buffer] = (cast_buffer, list(load.indices))

        return cast_buffers

    def _make_vectorized_loop(self, original: For, body: Stmt) -> For:
        """Create a vectorized For loop based on the original."""
        return For(
            original.loop_var,
            original.min,
            original.extent,
            ForKind.VECTORIZED,
            body,
            original.thread_binding,
            original.annotations,
            original.step,
        )

    def _create_copy_loops_to_memory(self, op: For, cast_buffers: CastBufferMap, condition: tir.PrimExpr | None = None) -> list[For]:
        """Create copy loops to transfer data from cast buffers to memory buffers."""
        copy_loops: list[For] = []

        for orig_buffer, (cast_buffer, orig_indices) in cast_buffers.items():
            # vectorized loop only has one iteration variable, so we use the same name for the copy variable
            copy_var = Var(f"{op.loop_var.name}_copy", op.loop_var.dtype)

            # Substitute loop_var with copy_var in original indices
            new_indices = [substitute(idx, {op.loop_var: copy_var}) for idx in orig_indices]

            # cast buffer → memory
            copy_store: Stmt = BufferStore(
                orig_buffer,
                BufferLoad(cast_buffer, [copy_var]),
                new_indices,
            )

            # Wrap with condition if present (substitute loop_var with copy_var)
            if condition is not None:
                new_condition = substitute(condition, {op.loop_var: copy_var})
                copy_store = IfThenElse(new_condition, copy_store, None)

            copy_loop = For(
                copy_var,
                op.min,
                op.extent,
                ForKind.VECTORIZED,
                copy_store,
                op.thread_binding,
                op.annotations,
                op.step,
            )
            copy_loops.append(copy_loop)

        return copy_loops

    def _create_copy_loops_from_memory(self, op: For, cast_buffers: CastBufferMap, condition: tir.PrimExpr | None = None) -> list[For]:
        """Create copy loops to transfer data from memory buffers to cast buffers."""
        copy_loops: list[For] = []

        for orig_buffer, (cast_buffer, orig_indices) in cast_buffers.items():
            # vectorized loop only has one iteration variable, so we use the same name for the copy variable
            copy_var = Var(f"{op.loop_var.name}_copy", op.loop_var.dtype)

            # Substitute loop_var with copy_var in original indices
            new_indices = [substitute(idx, {op.loop_var: copy_var}) for idx in orig_indices]

            # memory → cast buffer
            copy_store: Stmt = BufferStore(
                cast_buffer,
                BufferLoad(orig_buffer, new_indices),
                [copy_var],
            )

            # Wrap with condition if present (substitute loop_var with copy_var)
            if condition is not None:
                new_condition = substitute(condition, {op.loop_var: copy_var})
                copy_store = IfThenElse(new_condition, copy_store, None)

            copy_loop = For(
                copy_var,
                op.min,
                op.extent,
                ForKind.VECTORIZED,
                copy_store,
                op.thread_binding,
                op.annotations,
                op.step,
            )
            copy_loops.append(copy_loop)

        return copy_loops

    def _wrap_with_allocations(self, body: Stmt, cast_buffers: CastBufferMap) -> Stmt:
        """Wrap statement with buffer declarations and allocations."""
        result = body
        for cast_buffer, _ in cast_buffers.values():
            result = DeclBuffer(cast_buffer, result)
            result = Allocate(
                cast_buffer.data,
                cast_buffer.dtype,
                cast_buffer.shape,
                tir.const(True),
                result,
            )
        return result

    def _replace_stores_with_cast(self, stmt: Stmt, cast_buffers: CastBufferMap, loop_var: Var) -> Stmt:
        """Replace stores to memory buffers with stores to cast buffers."""
        store_replacer = StoreReplacer(cast_buffers, loop_var)
        return store_replacer.visit_stmt(stmt)

    def _replace_loads_with_cast(self, stmt: Stmt, cast_buffers: CastBufferMap, loop_var: Var) -> Stmt:
        """Replace loads from memory buffers with loads from cast buffers.

        This method recursively processes the statement tree, replacing
        BufferLoad nodes from cast buffers with loads from the cast buffer.
        """
        # Create an expression mutator to replace BufferLoads
        load_replacer = LoadReplacer(cast_buffers, loop_var)
        return load_replacer.visit_stmt(stmt)


@tir.functor.mutator
class StoreReplacer(tir.PyStmtExprMutator):
    """Mutator to replace memory BufferStores with cast buffer BufferStores."""

    def __init__(self, cast_buffers: CastBufferMap, loop_var: Var):
        super().__init__()
        self.cast_buffers = cast_buffers
        self.loop_var = loop_var

    def visit_buffer_store_(self, op: BufferStore) -> Stmt:
        if op.buffer in self.cast_buffers:
            cast_buffer, _ = self.cast_buffers[op.buffer]
            return BufferStore(cast_buffer, op.value, [self.loop_var])
        return op


@tir.functor.mutator
class LoadReplacer(tir.PyStmtExprMutator):
    """Mutator to replace memory BufferLoads with cast buffer BufferLoads."""

    def __init__(self, cast_buffers: CastBufferMap, loop_var: Var):
        super().__init__()
        self.cast_buffers = cast_buffers
        self.loop_var = loop_var

    def visit_buffer_load_(self, op: BufferLoad) -> tir.PrimExpr:
        if op.buffer in self.cast_buffers:
            cast_buffer, _ = self.cast_buffers[op.buffer]
            return BufferLoad(cast_buffer, [self.loop_var])
        return op


def DecoupleTypeCast():
    """Create a TVM pass that decouples type cast vectorization constraints.

    This pass inserts a local buffer as an intermediate stage for vectorized
    stores to non-local buffers (global/shared) where the store value contains
    expressions with different dtypes.

    This allows optimal vectorization for both computation and memory access.

    Note:
        This pass must be applied before VectorizeLoop and StorageRewrite passes,
        while the IR still uses BufferLoad/BufferStore (not tvm_access_ptr).

    Returns:
        A TVM PrimFunc pass.
    """

    def pass_fn(func: PrimFunc, mod, ctx) -> PrimFunc:
        mutator = DecoupleTypeCastMutator()
        new_body = mutator.visit_stmt(func.body)
        return func.with_body(new_body)

    return prim_func_pass(pass_fn, opt_level=0)
