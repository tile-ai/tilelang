"""
Decouple type cast vectorization constraints.

When a vectorized loop has mixed-precision operations between local and memory
buffers, the vectorization length would be constrained by the GCD of all
involved dtypes.

This pass decouples the constraints by inserting a local buffer as an
intermediate stage, allowing optimal vectorization for both computation and
memory access.

Mixed-precision is detected by the presence of Cast nodes in the loop body.

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
    Cast,
    DeclBuffer,
    For,
    ForKind,
    IfThenElse,
    IntImm,
    LetStmt,
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


# ---------------------------------------------------------------------------
# Mixed-precision detection: check for Cast nodes in the statement tree
# ---------------------------------------------------------------------------


def _has_cast(stmt: Stmt) -> bool:
    """Check if a statement tree contains any Cast node."""
    found = False

    def visitor(node) -> None:
        nonlocal found
        if isinstance(node, Cast):
            found = True

    post_order_visit(stmt, visitor)
    return found


def _expr_depends_on_var(expr: tir.PrimExpr, var: Var) -> bool:
    """Check if an expression references the given Var."""
    found = False

    def visitor(node) -> None:
        nonlocal found
        if isinstance(node, Var) and node.same_as(var):
            found = True

    post_order_visit(expr, visitor)
    return found


# ---------------------------------------------------------------------------
# Collection: gather all shared/global BufferStores and BufferLoads
# ---------------------------------------------------------------------------


@tir.functor.visitor
class MemoryAccessCollector(PyStmtExprVisitor):
    """Collect shared/global BufferStore and BufferLoad nodes.

    Skips indices traversal so that index expressions (which may contain
    BufferLoads to index buffers) do not pollute the result.

    BufferLoads in if_then_else conditions are skipped because conditions
    don't participate in the type-cast compute path.

    BufferLoads whose indices do not depend on ``loop_var`` are skipped
    because they are scalar accesses (e.g. ``b[0]``) that should remain
    in the compute loop as broadcasts.
    """

    def __init__(self, loop_var: Var):
        super().__init__()
        self.loop_var = loop_var
        self.stores: list[BufferStore] = []
        self.loads: list[BufferLoad] = []
        self._seen_load_buffers: set[Buffer] = set()

    def visit_buffer_store_(self, op: BufferStore) -> None:
        if is_global_or_shared_buffer(op.buffer):
            self.stores.append(op)
        # Visit value but skip indices
        self.visit_expr(op.value)

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        # Skip loads whose indices do not depend on loop_var (scalar access)
        if (is_global_or_shared_buffer(op.buffer)
                and op.buffer not in self._seen_load_buffers
                and any(_expr_depends_on_var(idx, self.loop_var) for idx in op.indices)):
            self.loads.append(op)
            self._seen_load_buffers.add(op.buffer)
        # Skip indices traversal

    def visit_call_(self, op: Call) -> None:
        if op.op.same_as(_IF_THEN_ELSE_OP):
            # Skip condition (args[0]), only visit true/false values
            self.visit_expr(op.args[1])
            self.visit_expr(op.args[2])
        else:
            for arg in op.args:
                self.visit_expr(arg)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def inline_let_stmts(stmt: Stmt) -> Stmt:
    """Inline all LetStmt bindings in *stmt* so that downstream visitors can
    see the original BufferLoad nodes that were hidden behind Var references.

    Used before collecting memory accesses so that BufferLoads inside LetStmt
    values are visible to ``MemoryAccessCollector``.
    """
    if isinstance(stmt, LetStmt):
        body = inline_let_stmts(stmt.body)
        return substitute(body, {stmt.var: stmt.value})
    elif isinstance(stmt, IfThenElse):
        then_case = inline_let_stmts(stmt.then_case)
        else_case = inline_let_stmts(stmt.else_case) if stmt.else_case else None
        if then_case is not stmt.then_case or else_case is not stmt.else_case:
            return IfThenElse(stmt.condition, then_case, else_case)
        return stmt
    elif isinstance(stmt, SeqStmt):
        new_seq = [inline_let_stmts(s) for s in stmt.seq]
        return SeqStmt(new_seq)
    else:
        return stmt


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


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------


@tir.functor.mutator
class DecoupleTypeCastMutator(tir.PyStmtExprMutator):
    """Mutator that decouples type cast vectorization constraints.

    This mutator transforms vectorized loops that have mixed-precision
    operations (detected by the presence of Cast nodes) by inserting local
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

    # ----- entry point for each For loop -----

    def visit_for_(self, op: For) -> Stmt:
        """Visit For nodes, transforming vectorized loops with mixed-type stores."""
        # Recursively visit body to handle nested loops
        new_body = self.visit_stmt(op.body)

        # Only transform vectorized loops with static extent
        if op.kind != ForKind.VECTORIZED:
            return self._make_for(op, new_body) if new_body is not op.body else op
        if not isinstance(op.extent, IntImm):
            return self._make_for(op, new_body) if new_body is not op.body else op

        # Check if the body has any Cast nodes
        if not _has_cast(new_body):
            return self._make_for(op, new_body) if new_body is not op.body else op

        # Inline LetStmts for analysis so BufferLoads behind Vars are visible
        inlined_body = inline_let_stmts(new_body)

        # Collect all shared/global stores and loads
        collector = MemoryAccessCollector(op.loop_var)
        collector.visit_stmt(inlined_body)

        if not collector.stores and not collector.loads:
            # Cast exists but no memory access → nothing to decouple
            return self._make_for(op, new_body) if new_body is not op.body else op

        extent = op.extent.value

        # Extract condition (from inlined body for correctness)
        condition, _ = extract_if_condition(inlined_body)

        # Create cast buffers for stores (memory targets → local cast buffer)
        store_cast_buffers = self._create_cast_buffers_for_stores(collector.stores, extent)
        # Create cast buffers for loads (memory sources → local cast buffer)
        # Skip buffers already covered by store_cast_buffers (read-modify-write)
        load_cast_buffers = self._create_cast_buffers_for_loads(
            [ld for ld in collector.loads if ld.buffer not in store_cast_buffers],
            extent,
        )

        # Build copy-from-memory loops (before compute)
        # For read-modify-write (same buffer is both store target and load source),
        # reuse the store-side cast buffer for the copy-from loop as well.
        rmw_buffers = {buf: store_cast_buffers[buf] for buf in store_cast_buffers if buf in collector._seen_load_buffers}
        all_copy_from: CastBufferMap = {**load_cast_buffers, **rmw_buffers}
        copy_from_loops = self._create_copy_loops_from_memory(op, all_copy_from, condition)

        # Build compute loop: replace both stores and loads in the *original* body
        # For loads, merge store_cast_buffers so read-modify-write uses same local buffer
        all_load_replacements: CastBufferMap = {**store_cast_buffers, **load_cast_buffers}
        compute_body = new_body
        if store_cast_buffers:
            compute_body = self._replace_stores_with_cast(compute_body, store_cast_buffers, op.loop_var)
        if all_load_replacements:
            compute_body = self._replace_loads_with_cast(compute_body, all_load_replacements, op.loop_var)
        compute_loop = self._make_vectorized_loop(op, compute_body)

        # Build copy-to-memory loops (after compute)
        copy_to_loops = self._create_copy_loops_to_memory(op, store_cast_buffers, condition)

        # Combine: copy-from → compute → copy-to
        all_stmts = copy_from_loops + [compute_loop] + copy_to_loops
        result: Stmt = SeqStmt(all_stmts) if len(all_stmts) > 1 else all_stmts[0]

        # Wrap with buffer declarations and allocations
        all_cast_buffers: CastBufferMap = {**store_cast_buffers, **load_cast_buffers}
        result = self._wrap_with_allocations(result, all_cast_buffers)

        return result

    # ----- helpers -----

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
            # vectorized loop only has one iteration variable,
            # so we use the same name for the copy variable
            copy_var = Var(f"{op.loop_var.name}_copy", op.loop_var.dtype)

            # Substitute loop_var with copy_var in original indices
            new_indices = [substitute(idx, {op.loop_var: copy_var}) for idx in orig_indices]

            # cast buffer → memory
            copy_store: Stmt = BufferStore(
                orig_buffer,
                BufferLoad(cast_buffer, [copy_var]),
                new_indices,
            )

            # Wrap with condition if present
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
            # vectorized loop only has one iteration variable,
            # so we use the same name for the copy variable
            copy_var = Var(f"{op.loop_var.name}_copy", op.loop_var.dtype)

            # Substitute loop_var with copy_var in original indices
            new_indices = [substitute(idx, {op.loop_var: copy_var}) for idx in orig_indices]

            # memory → cast buffer
            copy_store: Stmt = BufferStore(
                cast_buffer,
                BufferLoad(orig_buffer, new_indices),
                [copy_var],
            )

            # Wrap with condition if present
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
        """Replace loads from memory buffers with loads from cast buffers."""
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
    loops where the body contains Cast nodes (mixed-precision operations).

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
