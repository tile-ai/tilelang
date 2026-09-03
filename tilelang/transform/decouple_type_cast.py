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

The staging cast buffers together with the transformed loops are wrapped in
an opaque block annotated with `lexical_alloc_scope`, so their allocations
stay lexically scoped to the use site (see LowerOpaqueBlock / StorageRewrite)
instead of being hoisted to the kernel entry.

Conditional stores: copy-to guard semantics
-------------------------------------------
A store may sit under IfThenElse conditions (per-lane branches). The copy-to
loop must write back exactly the lanes whose original store fired, otherwise
an uninitialized cast local is written to memory. Re-evaluating the original
path-condition *expression* in the copy-to loop is unsound: the expression is
structurally identical to the compute-time condition, but earlier copy-to
write-backs may have modified the buffers it reads, flipping its truth value
between compute time and copy time.

Instead, each store entry that has any enclosing condition gets a per-entry
**validity mask** (a local int32 buffer, 0/1 per lane). The mask is set to 1
inside the compute loop, at the exact statement position where the original
store executed (inside the same branches), so it records the path the compute
stage actually took; an init loop zeroes it first. The copy-to loop reads only
``mask[i] != 0`` and never re-evaluates the original conditions, so OR/nested
branch semantics reduce to "was this entry's cast local defined for this lane".
Unconditional stores need no mask (their copy-to stays unconditional).
Copy-from loops keep condition guards: they run before the compute loop and
before any copy-to write-back, so original-buffer conditions are still stable
at that point.
"""

from __future__ import annotations

from tvm import ir as tvm_ir
from tvm import tirx
from tvm.ir import Op
from tvm.tirx import (
    AllocBuffer,
    Buffer,
    BufferLoad,
    BufferStore,
    Call,
    Cast,
    For,
    ForKind,
    IfThenElse,
    IntImm,
    Bind,
    Evaluate,
    PrimFunc,
    PyStmtExprVisitor,
    SBlock,
    SBlockRealize,
    SeqStmt,
    Stmt,
    Var,
)
from tvm.tirx.stmt_functor import post_order_visit, substitute
from tvm.tirx.transform import prim_func_pass

# Cache the Op for if_then_else to avoid repeated lookups
_IF_THEN_ELSE_OP = Op.get("tirx.if_then_else")

from tilelang.ir import get_stmt_span, stamp_stmt_spans
from tilelang.utils.language import (
    is_fragment,
    is_global,
    is_local,
    is_local_var,
    is_shared,
)


def is_local_buffer(buffer: Buffer) -> bool:
    """Check if a buffer is local/register-level."""
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


@tirx.functor.visitor
class _CastFinder(PyStmtExprVisitor):
    """Find Cast nodes in a statement, skipping BufferLoad/BufferStore indices.

    A Cast that only appears inside an index expression is not a mixed-precision
    compute — it's just an index-type conversion — so it should not trigger the
    decoupling transformation.
    """

    def __init__(self):
        super().__init__()
        self.found = False

    def visit_cast_(self, op: Cast) -> None:
        self.found = True
        self.visit_expr(op.value)

    def visit_buffer_store_(self, op: BufferStore) -> None:
        self.visit_expr(op.value)

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        pass


def _has_cast(stmt: Stmt) -> bool:
    """Check if a statement tree contains any Cast node outside of indices."""
    finder = _CastFinder()
    finder.visit_stmt(stmt)
    return finder.found


def _contains_seq_stmt(stmt: Stmt) -> bool:
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


def _expr_depends_on_var(expr: tirx.PrimExpr, var: Var) -> bool:
    """Check if an expression references the given Var."""
    found = False

    def visitor(node) -> None:
        nonlocal found
        if isinstance(node, Var) and node.same_as(var):
            found = True

    post_order_visit(expr, visitor)
    return found


def _and_cond(path: tirx.PrimExpr | None, cond: tirx.PrimExpr) -> tirx.PrimExpr:
    """Conjoin a path condition with a branch condition.

    ``None`` encodes "always" (no enclosing condition), so it is the
    neutral element for conjunction.
    """
    if path is None:
        return cond
    return tirx.And(path, cond)


def _or_conditions(conds: list[tirx.PrimExpr | None]) -> tirx.PrimExpr | None:
    """Disjunction of path conditions.

    ``None`` encodes "always": an access without any enclosing condition
    fires unconditionally, so the disjunction is unconditional. A tautology
    ``c OR NOT(c)`` (either operand order, structural match) also folds to
    ``None`` — the branch conditions of a root if/else are complementary, so
    the OR of both branches' paths is always true.
    """
    result: tirx.PrimExpr | None = None
    for cond in conds:
        if cond is None:
            return None
        if result is None:
            result = cond
        else:
            if (isinstance(cond, tirx.Not) and tvm_ir.structural_equal(cond.a, result)) or (
                isinstance(result, tirx.Not) and tvm_ir.structural_equal(result.a, cond)
            ):
                return None
            result = tirx.Or(result, cond)
    return result


# ---------------------------------------------------------------------------
# Collection: gather all shared/global BufferStores and BufferLoads
# ---------------------------------------------------------------------------


@tirx.functor.visitor
class MemoryAccessCollector(PyStmtExprVisitor):
    """Collect shared/global BufferStore and BufferLoad nodes.

    Skips indices traversal so that index expressions (which may contain
    BufferLoads to index buffers) do not pollute the result.

    BufferLoads in if_then_else conditions are skipped because conditions
    don't participate in the type-cast compute path.

    BufferLoads whose indices do not depend on ``loop_var`` are skipped
    because they are scalar accesses (e.g. ``b[0]``) that should remain
    in the compute loop as broadcasts.

    Each collected access is paired with its path condition: the
    conjunction of enclosing IfThenElse conditions (``None`` when the
    access is unconditional). Load conditions still guard the copy-from
    loops (safe: they are evaluated before the compute loop and before any
    copy-to write-back). Store conditions are no longer re-evaluated in the
    copy-to loops: each store entry gets a validity mask set at the compute
    store site instead (see module docstring).
    """

    def __init__(self, loop_var: Var):
        super().__init__()
        self.loop_var = loop_var
        self.condition: tirx.PrimExpr | None = None
        self.stores: list[tuple[BufferStore, tirx.PrimExpr | None]] = []
        self.loads: list[tuple[BufferLoad, tirx.PrimExpr | None]] = []

    def visit_if_then_else_(self, op: IfThenElse) -> None:
        saved = self.condition
        # Loads inside the condition itself fire whenever the statement is
        # reached, i.e. under the enclosing (saved) condition only.
        self.visit_expr(op.condition)
        self.condition = _and_cond(saved, op.condition)
        self.visit_stmt(op.then_case)
        if op.else_case is not None:
            self.condition = _and_cond(saved, tirx.Not(op.condition))
            self.visit_stmt(op.else_case)
        self.condition = saved

    def visit_buffer_store_(self, op: BufferStore) -> None:
        if is_global_or_shared_buffer(op.buffer):
            self.stores.append((op, self.condition))
        # Visit value but skip indices
        self.visit_expr(op.value)

    def visit_buffer_load_(self, op: BufferLoad) -> None:
        # Skip loads whose indices do not depend on loop_var (scalar access).
        # Collect ALL qualifying loads (even from the same buffer with different
        # indices, e.g. a[i] and a[i+32]) so each gets its own cast buffer.
        if is_global_or_shared_buffer(op.buffer) and any(_expr_depends_on_var(idx, self.loop_var) for idx in op.indices):
            self.loads.append((op, self.condition))
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


BindEnv = dict[Var, tirx.PrimExpr]


def _substitute_bind_env(node, env: BindEnv):
    """Apply the current flat-Bind environment to a statement or expression."""
    result = node
    for var, replacement in env.items():
        result = substitute(result, {var: replacement})
    return result


def _normalize_flat_binds(stmt: Stmt, env: BindEnv) -> Stmt | None:
    """Inline flat Bind statements according to sequential dominance.

    ``Bind`` no longer has a body. A bind in ``SeqStmt`` dominates only later
    sibling statements, while binds inside a branch or nested loop do not escape
    that scope. This normalization exposes hidden BufferLoad/Cast nodes to the
    decoupling analysis without treating Bind as a tree-shaped LetStmt.
    """
    if isinstance(stmt, Bind):
        env[stmt.var] = _substitute_bind_env(stmt.value, env)
        return None

    if isinstance(stmt, SeqStmt):
        local_env = dict(env)
        result: list[Stmt] = []
        for child in stmt.seq:
            normalized = _normalize_flat_binds(child, local_env)
            if normalized is not None:
                result.append(normalized)
        if not result:
            return None
        normalized = SeqStmt(result) if len(result) > 1 else result[0]
        stamp_stmt_spans(normalized, get_stmt_span(stmt))
        return normalized

    if isinstance(stmt, IfThenElse):
        condition = _substitute_bind_env(stmt.condition, env)
        then_case = _normalize_flat_binds(stmt.then_case, dict(env))
        else_case = _normalize_flat_binds(stmt.else_case, dict(env)) if stmt.else_case else None
        normalized_else = None
        if stmt.else_case:
            normalized_else = else_case if else_case is not None else Evaluate(0)
        normalized_if = IfThenElse(
            condition,
            then_case if then_case is not None else Evaluate(0),
            normalized_else,
        )
        stamp_stmt_spans(normalized_if, get_stmt_span(stmt))
        return normalized_if

    if isinstance(stmt, For):
        body = _normalize_flat_binds(stmt.body, dict(env))
        new_for = For(
            stmt.loop_var,
            _substitute_bind_env(stmt.min, env),
            _substitute_bind_env(stmt.extent, env),
            stmt.kind,
            body if body is not None else Evaluate(0),
            stmt.thread_binding,
            stmt.annotations,
            _substitute_bind_env(stmt.step, env),
        )
        stamp_stmt_spans(new_for, get_stmt_span(stmt))
        return new_for

    result = _substitute_bind_env(stmt, env)
    # `substitute` rebuilds any node referencing a bind var, dropping its span.
    stamp_stmt_spans(result, get_stmt_span(stmt))
    return result


def normalize_flat_binds(stmt: Stmt) -> Stmt:
    """Return ``stmt`` with dominating flat Bind values substituted into uses."""
    normalized = _normalize_flat_binds(stmt, {})
    return normalized if normalized is not None else stmt


def extract_if_condition(stmt: Stmt) -> tuple[tirx.PrimExpr | None, Stmt]:
    """Extract IfThenElse condition from statement if present.

    Returns:
        A tuple of (condition, inner_body). If no IfThenElse, returns (None, stmt).
    """
    if isinstance(stmt, IfThenElse) and stmt.else_case is None:
        return stmt.condition, stmt.then_case
    return None, stmt


# Cast entry: (original buffer, original indices, cast buffer)
# Each unique (buffer, indices) pair gets its own entry, so that accesses
# like a[i] and a[i+32] from the same buffer are handled correctly.
CastEntry = tuple[Buffer, list[tirx.PrimExpr], Buffer]


def _buf_indices_match(
    buf_a: Buffer,
    indices_a: list[tirx.PrimExpr],
    buf_b: Buffer,
    indices_b: list[tirx.PrimExpr],
) -> bool:
    """Check if two (buffer, indices) pairs refer to the same access pattern."""
    if not buf_a.same_as(buf_b):
        return False
    if len(indices_a) != len(indices_b):
        return False
    return all(tvm_ir.structural_equal(a, b) for a, b in zip(indices_a, indices_b))


def _find_cast_entry(
    entries: list[CastEntry],
    buffer: Buffer,
    indices: list[tirx.PrimExpr],
) -> Buffer | None:
    """Find the cast buffer for a given (buffer, indices) pair, or None."""
    for orig_buf, orig_indices, cast_buf in entries:
        if _buf_indices_match(orig_buf, orig_indices, buffer, indices):
            return cast_buf
    return None


def _find_cast_entry_index(
    entries: list[CastEntry],
    buffer: Buffer,
    indices: list[tirx.PrimExpr],
) -> int:
    """Index of the entry matching a (buffer, indices) pair, or -1."""
    for i, (orig_buf, orig_indices, _) in enumerate(entries):
        if _buf_indices_match(orig_buf, orig_indices, buffer, indices):
            return i
    return -1


# ---------------------------------------------------------------------------
# Mutator
# ---------------------------------------------------------------------------


@tirx.functor.mutator
class DecoupleTypeCastMutator(tirx.PyStmtExprMutator):
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
        new_for = For(
            original.loop_var,
            original.min,
            original.extent,
            original.kind,
            new_body,
            original.thread_binding,
            original.annotations,
            original.step,
        )
        stamp_stmt_spans(new_for, get_stmt_span(original))
        return new_for

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

        # Normalize flat Bind statements before all analysis. Bind is a
        # sequential SSA definition, not a tree-shaped LetStmt with a body.
        normalized_body = normalize_flat_binds(new_body)

        # Check if the normalized body has any Cast nodes.
        if not _has_cast(normalized_body):
            return self._make_for(op, new_body) if new_body is not op.body else op

        # Skip SeqStmt (multiple statements) after inlining leading Bind nodes.
        # A common frontend pattern is SeqStmt(Bind(...), BufferStore(...)),
        # which is still a single compute statement after substitution.
        if _contains_seq_stmt(normalized_body):
            return self._make_for(op, new_body) if new_body is not op.body else op

        # Skip Evaluate roots. Decoupling splits the value edge of a BufferStore
        # to insert a staging buffer; an Evaluate discards its result and stores
        # nothing, so that edge does not exist and the transform is undefined.
        # Opaque intrinsic statements such as `tl.ptx_cp_async(...)` land here,
        # and their operands are address arguments with address-space
        # constraints (dst must be shared, src must be global) that rewriting to
        # local staging buffers would violate.
        _, root_stmt = extract_if_condition(normalized_body)
        if isinstance(root_stmt, Evaluate):
            return self._make_for(op, new_body) if new_body is not op.body else op

        # Collect all shared/global stores and loads, each with its path
        # condition (conjunction of enclosing IfThenElse conditions).
        collector = MemoryAccessCollector(op.loop_var)
        collector.visit_stmt(normalized_body)
        store_list = collector.stores
        load_list = collector.loads

        if not store_list and not load_list:
            # Cast exists but no memory access → nothing to decouple
            return self._make_for(op, new_body) if new_body is not op.body else op

        extent = op.extent.value

        # Create cast entries for stores and loads
        store_entries = self._create_cast_entries([s for s, _ in store_list], extent)
        # For loads, skip those already covered by a store entry (read-modify-write)
        # by matching (buffer, indices). Loads with different indices from the same
        # buffer still get their own cast buffer.
        uncovered_loads = [ld for ld, _ in load_list if _find_cast_entry(store_entries, ld.buffer, list(ld.indices)) is None]
        load_entries = self._create_cast_entries(uncovered_loads, extent)

        def _entry_conditions(
            entries: list[CastEntry], accesses: list[tuple[BufferStore | BufferLoad, tirx.PrimExpr | None]]
        ) -> list[tirx.PrimExpr | None]:
            """OR of the path conditions of every access mapped to each entry."""
            return [
                _or_conditions([cond for acc, cond in accesses if _buf_indices_match(entry[0], entry[1], acc.buffer, list(acc.indices))])
                for entry in entries
            ]

        # Copy-from loops are guarded by the load path conditions: they run
        # before the compute loop and before any copy-to write-back, so the
        # original buffers their conditions read are still in the state the
        # compute loop will see. Copy-to loops must NOT re-evaluate the store
        # path conditions (earlier write-backs could flip them); they are
        # guarded by the per-entry validity masks set at the compute store
        # sites instead. The store conditions are kept only to
        # decide which entries are unconditional (mask=None).
        load_conditions = _entry_conditions(load_entries, load_list)
        store_conditions = _entry_conditions(store_entries, store_list)
        store_masks = self._create_mask_buffers(store_entries, store_conditions)

        # Zero the validity masks before the compute loop (per masked entry).
        mask_init_loops = self._create_mask_init_loops(op, [m for m in store_masks if m is not None])

        # Build copy-from-memory loops (before compute)
        # For read-modify-write, reuse the store-side cast buffer for copy-from.
        rmw_entries = [
            entry
            for entry in store_entries
            if any(_buf_indices_match(entry[0], entry[1], ld.buffer, list(ld.indices)) for ld, _ in load_list)
        ]
        rmw_conditions = _entry_conditions(rmw_entries, load_list)
        copy_from_loops = self._create_copy_loops(
            op,
            load_entries + rmw_entries,
            load_conditions + rmw_conditions,
            direction="from_memory",
        )

        # Build compute loop: replace stores and loads in the normalized body
        # so that indices match what the collector saw (Bind vars are expanded).
        # For RMW (a load whose (buffer, indices) matches a store entry), the load
        # must be rewritten to the *same* cast buffer the store writes to, so we
        # feed both store and load entries into the load-replacement table.
        load_replacement_entries = store_entries + load_entries
        compute_body = normalized_body
        if store_entries or load_entries:
            compute_body = self._replace_access(compute_body, store_entries, load_replacement_entries, op.loop_var, store_masks)
        compute_loop = self._make_vectorized_loop(op, compute_body)

        # Build copy-to-memory loops (after compute). Guards are the per-entry
        # validity masks (set at the compute store sites), never a re-evaluation
        # of the original path conditions.
        copy_to_loops = self._create_copy_loops(
            op,
            store_entries,
            [],
            direction="to_memory",
            masks=store_masks,
        )

        # Combine: mask-init → copy-from → compute → copy-to. Mask init must
        # precede the compute loop (which sets the masks) and the copy-to loops
        # (which read them); placing it first keeps every original buffer in its
        # initial state when the copy-from guards are evaluated.
        all_stmts = mask_init_loops + copy_from_loops + [compute_loop] + copy_to_loops
        result: Stmt = SeqStmt(all_stmts) if len(all_stmts) > 1 else all_stmts[0]

        # Wrap with buffer declarations and allocations
        result = self._wrap_with_allocations(
            result,
            store_entries + load_entries,
            [m for m in store_masks if m is not None],
        )

        # The replacement subtree inherits the original loop's span. Nodes
        # already stamped above (compute loop, normalized body) keep their own
        # spans; the staging copies and the alloc-scope wrapper get stamped too.
        stamp_stmt_spans(result, get_stmt_span(op))
        return result

    # ----- helpers -----

    def _create_cast_entries(self, accesses: list[BufferStore | BufferLoad], extent: int) -> list[CastEntry]:
        """Create local cast buffers for memory accesses.

        Each unique (buffer, indices) pair gets its own cast buffer.
        """
        entries: list[CastEntry] = []

        for access in accesses:
            indices = list(access.indices)
            if _find_cast_entry(entries, access.buffer, indices) is not None:
                continue

            cache_name = self._make_unique_name(f"{access.buffer.name}_local_cast")
            cast_buffer = tirx.decl_buffer(
                shape=(extent,),
                dtype=access.buffer.dtype,
                name=cache_name,
                scope="local",
            )
            entries.append((access.buffer, indices, cast_buffer))

        return entries

    def _make_vectorized_loop(self, original: For, body: Stmt) -> For:
        """Create a vectorized For loop based on the original."""
        new_for = For(
            original.loop_var,
            original.min,
            original.extent,
            ForKind.VECTORIZED,
            body,
            original.thread_binding,
            original.annotations,
            original.step,
        )
        stamp_stmt_spans(new_for, get_stmt_span(original))
        return new_for

    def _create_mask_buffers(self, entries: list[CastEntry], conditions: list[tirx.PrimExpr | None]) -> list[Buffer | None]:
        """Create a per-entry validity mask buffer (local int32, 0/1) for store entries.

        ``None`` for entries whose copy-to loop is unconditional (no enclosing
        condition, ``conditions[i] is None``): their cast local is defined on
        every lane, so no mask is needed. Masked entries get
        ``<cast_buffer>_mask`` with one int32 per lane; the compute loop sets
        it to 1 exactly where the original store executed, and the copy-to
        loop reads it instead of re-evaluating the original path conditions
        directly.
        """
        return [
            tirx.decl_buffer(
                shape=(int(entry[2].shape[0]),),
                dtype="int32",
                name=f"{entry[2].name}_mask",
                scope="local",
            )
            if condition is not None
            else None
            for entry, condition in zip(entries, conditions, strict=True)
        ]

    def _create_mask_init_loops(self, op: For, masks: list[Buffer]) -> list[For]:
        """Zero every validity mask before the compute loop.

        One vectorized loop per mask, mirroring the per-entry copy loop
        structure. The masks are local buffers, so these loops vectorize at the
        usual local-buffer width (int32x4 on Metal).
        """
        init_loops: list[For] = []
        for mask in masks:
            init_var = Var(f"{op.loop_var.name}_mask", op.loop_var.dtype)
            init_store = BufferStore(mask, IntImm("int32", 0), [init_var])
            init_loops.append(
                For(
                    init_var,
                    op.min,
                    op.extent,
                    ForKind.VECTORIZED,
                    init_store,
                    op.thread_binding,
                    op.annotations,
                    op.step,
                )
            )
        return init_loops

    def _create_copy_loops(
        self,
        op: For,
        entries: list[CastEntry],
        conditions: list[tirx.PrimExpr | None],
        direction: str,
        masks: list[Buffer | None] | None = None,
    ) -> list[For]:
        """Create vectorized copy loops between memory and cast buffers.

        direction: "to_memory" (cast → memory) or "from_memory" (memory → cast).

        ``to_memory`` guards come from ``masks`` (per-entry validity mask
        buffers, parallel to ``entries``): the copy fires only where the
        compute stage actually executed the original store (``mask[i] != 0``).
        The original path conditions are deliberately NOT re-evaluated here —
        earlier copy-to write-backs could flip their truth value between
        compute time and copy time. ``None`` mask means the copy is
        unconditional.

        ``from_memory`` guards come from ``conditions`` (per-entry OR of load
        path conditions, ``None`` = unconditional). These are safe to evaluate
        here because copy-from runs before the compute loop and before any
        copy-to write-back, so original buffers are still in the state the
        compute loop will see.
        """
        copy_loops: list[For] = []

        for i, (orig_buffer, orig_indices, cast_buffer) in enumerate(entries):
            # vectorized loop only has one iteration variable,
            # so we use the same name for the copy variable
            copy_var = Var(f"{op.loop_var.name}_copy", op.loop_var.dtype)

            # Substitute loop_var with copy_var in original indices
            new_indices = [substitute(idx, {op.loop_var: copy_var}) for idx in orig_indices]

            if direction == "to_memory":
                copy_store: Stmt = BufferStore(
                    orig_buffer,
                    BufferLoad(cast_buffer, [copy_var]),
                    new_indices,
                )
                guard: tirx.PrimExpr | None = None
                if masks is not None and masks[i] is not None:
                    guard = tirx.NE(BufferLoad(masks[i], [copy_var]), IntImm("int32", 0))
            else:
                copy_store = BufferStore(
                    cast_buffer,
                    BufferLoad(orig_buffer, new_indices),
                    [copy_var],
                )
                guard = conditions[i] if i < len(conditions) else None
                if guard is not None:
                    # The copy loop runs under ``copy_var``; the path condition
                    # was collected over the original loop var, so substitute
                    # before guarding (masks need no substitution: they are
                    # indexed directly by ``copy_var``).
                    guard = substitute(guard, {op.loop_var: copy_var})

            # Wrap with condition if present
            if guard is not None:
                copy_store = IfThenElse(guard, copy_store, None)

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

    def _wrap_with_allocations(self, body: Stmt, entries: list[CastEntry], mask_buffers: list[Buffer] = ()) -> Stmt:
        """Wrap statement with buffer allocations inside a lexical alloc scope.

        The cast buffers (and their validity masks) are tiny per-site staging
        arrays. Placing them in an opaque block annotated with
        `lexical_alloc_scope` makes LowerOpaqueBlock materialize a scope
        boundary, so StorageRewrite keeps the allocations next to their use
        site instead of hoisting them to the kernel entry, and codegen emits a
        `{ ... }` scope with the declarations inside.
        """
        buffers = [cast_buffer for _, _, cast_buffer in entries] + list(mask_buffers)
        if not buffers:
            return body
        alloc_stmts: list[Stmt] = [AllocBuffer(buf) for buf in buffers]
        alloc_stmts.append(body)
        block = SBlock(
            iter_vars=[],
            reads=[],
            writes=[],
            name_hint="decoupled_cast",
            body=SeqStmt(alloc_stmts),
            annotations={"lexical_alloc_scope": 1},
        )
        return SBlockRealize([], True, block)

    def _replace_access(
        self,
        stmt: Stmt,
        store_entries: list[CastEntry],
        load_entries: list[CastEntry],
        loop_var: Var,
        store_masks: list[Buffer | None],
    ) -> Stmt:
        """Replace memory accesses with cast buffer accesses."""
        replacer = AccessReplacer(store_entries, load_entries, loop_var, store_masks)
        return replacer.visit_stmt(stmt)


@tirx.functor.mutator
class AccessReplacer(tirx.PyStmtExprMutator):
    """Mutator to replace memory BufferStores/BufferLoads with cast buffer accesses.

    Matches by both buffer and indices (structural equality) so that accesses
    like a[i] and a[i+32] from the same buffer map to different cast buffers.
    """

    def __init__(
        self,
        store_entries: list[CastEntry],
        load_entries: list[CastEntry],
        loop_var: Var,
        store_masks: list[Buffer | None],
    ):
        super().__init__()
        self.store_entries = store_entries
        self.load_entries = load_entries
        self.loop_var = loop_var
        self.store_masks = store_masks

    def visit_buffer_store_(self, op: BufferStore) -> Stmt:
        new_value = self.visit_expr(op.value)
        cast_buf = _find_cast_entry(self.store_entries, op.buffer, list(op.indices))
        if cast_buf is not None:
            mask = self.store_masks[_find_cast_entry_index(self.store_entries, op.buffer, list(op.indices))]
            cast_store = BufferStore(cast_buf, new_value, [self.loop_var])
            if mask is not None:
                # Record that this entry's cast local was actually defined on
                # this lane using a validity mask. The mask store sits at
                # the exact statement position of the original store, so it
                # inherits every enclosing branch condition.
                mask_store = BufferStore(mask, IntImm("int32", 1), [self.loop_var])
                return SeqStmt([cast_store, mask_store])
            return cast_store
        if new_value is not op.value:
            return BufferStore(op.buffer, new_value, list(op.indices))
        return op

    def visit_buffer_load_(self, op: BufferLoad) -> tirx.PrimExpr:
        cast_buf = _find_cast_entry(self.load_entries, op.buffer, list(op.indices))
        if cast_buf is not None:
            return BufferLoad(cast_buf, [self.loop_var])
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
        return func.with_body(new_body, span=func.span)

    return prim_func_pass(pass_fn, opt_level=0, name="tl.DecoupleTypeCast")
