"""Stage mixed precision vector memory operations in native local tensors.

Copy-in, compute and copy-out retain the original logical loop domain. Only
scratch indices use a zero-based ordinal. Staging allocations live in a
``tl.attr`` lexical allocation region; consumers must preserve that region
through storage planning and emit its actual scope.
"""

from __future__ import annotations

import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim

from tilelang._analyzer import create_analyzer
from tilelang.analysis._visitor import tensor_scope
from tilelang.language._native import encode_annotations


_IF_THEN_ELSE_OP = ir.Op.get("prim.if_then_else")
BindEnv = dict[ir.Var, ir.Expr]
CastEntry = tuple[ir.Expr, list[ir.Expr], ir.Var]


def is_local_buffer(tensor) -> bool:
    scope = tensor_scope(tensor) if tensor is not None else None
    return scope is not None and scope.startswith("local")


def is_global_or_shared_buffer(tensor) -> bool:
    scope = tensor_scope(tensor) if tensor is not None else None
    return scope == "global" or (scope is not None and scope.startswith("shared"))


def _has_cast(statement) -> bool:
    found = False

    def cast(node, visitor):
        nonlocal found
        found = True
        visitor.visit(node.value)

    tvm_ffi.structural_visit(
        statement,
        [
            (prim.Cast, cast),
            (ir.TensorStore, lambda node, visitor: visitor.visit(node.value)),
            (ir.TensorLoad, lambda _node, _visitor: None),
        ],
    )
    return found


def _expr_depends_on_var(expression, variable):
    variables = []
    tvm_ffi.structural_walk(expression, (ir.Var, variables.append), order="pre")
    return any(value.same_as(variable) for value in variables)


def _can_reorder(expression, *, allow_loads=False):
    """Check whether repeating/reordering an expression observes mutable state."""
    pure = True

    def check(node):
        nonlocal pure
        if isinstance(node, ir.TensorLoad):
            pure = pure and allow_loads
        elif isinstance(node, ir.Call):
            effect = node.op.get_attr("TCallEffectKind") if isinstance(node.op, ir.Op) else None
            if effect is None or int(effect) > ir.CallEffectKind.Pure:
                pure = False

    tvm_ffi.structural_walk(expression, ((ir.TensorLoad, ir.Call), check), order="pre")
    return pure


def _can_repeat_condition(condition):
    return condition is None or _can_reorder(condition)


def _substitute_bind_env(node, env: BindEnv):
    # Only same-typed scalar bindings and loop variables enter this map. The
    # structural engine still visits shape/type uses and preserves identity.
    return tvm_ffi.structural_map(node, (ir.Var, lambda variable: env.get(variable, variable))) if env else node


def _normalize_flat_binds(statement, env: BindEnv):
    """Inline scalar definitions according to sequential lexical dominance.

    An effectful Bind remains a real statement and prevents staging across it.
    Allocation/tensor definitions also remain in place. This prevents the new
    flat-Bind representation from duplicating atomic or allocation effects.
    """
    if isinstance(statement, ir.Bind):
        value = _substitute_bind_env(statement.value, env)
        if isinstance(statement.var.ty, ir.PrimType) and _can_reorder(value, allow_loads=True):
            env[statement.var] = value
            return None
        return ir.Bind(statement.var, value, loc=statement.loc)
    if isinstance(statement, ir.SeqStmt):
        local_env = dict(env)
        body = []
        for child in statement.seq:
            normalized = _normalize_flat_binds(child, local_env)
            if normalized is not None:
                body.append(normalized)
        return ir.SeqStmt(body, loc=statement.loc)
    if isinstance(statement, ir.If):
        return ir.If(
            _substitute_bind_env(statement.condition, env),
            _normalize_flat_binds(statement.then_case, dict(env)),
            _normalize_flat_binds(statement.else_case, dict(env)) if statement.else_case is not None else None,
            loc=statement.loc,
        )
    if isinstance(statement, ir.For):
        nested_env = dict(env)
        nested_env.pop(statement.var, None)
        return ir.For(
            statement.var,
            _substitute_bind_env(statement.begin, env),
            _substitute_bind_env(statement.end, env),
            _substitute_bind_env(statement.step, env),
            statement.kind,
            _normalize_flat_binds(statement.body, nested_env),
            statement.annotations,
            loc=statement.loc,
        )
    return _substitute_bind_env(statement, env)


def normalize_flat_binds(statement):
    normalized = _normalize_flat_binds(statement, {})
    return ir.SeqStmt([], loc=statement.loc) if normalized is None else normalized


def _single_statement(statement):
    while isinstance(statement, ir.SeqStmt) and len(statement.seq) == 1:
        statement = statement.seq[0]
    return statement


def extract_if_condition(statement):
    """Combine leading guards with short-circuit evaluation of inner guards."""
    condition = None
    statement = _single_statement(statement)
    while isinstance(statement, ir.If) and statement.else_case is None:
        condition = statement.condition if condition is None else prim.if_then_else(condition, statement.condition, ir.const(False))
        statement = _single_statement(statement.then_case)
    return condition, statement


class MemoryAccessCollector:
    """Collect compute memory edges, keeping indices and branch guards separate."""

    def __init__(self, loop_var):
        self.loop_var = loop_var
        self.stores = []
        self.loads = []
        self.conditional_depth = 0
        self.has_conditional_load = False

    def visit_stmt(self, statement):
        tvm_ffi.structural_visit(
            statement,
            [
                (ir.TensorStore, self.visit_store),
                (ir.TensorLoad, self.visit_load),
                (ir.Call, self.visit_call),
                (prim.Select, self.visit_select),
            ],
        )

    def visit_store(self, node, visitor):
        if is_global_or_shared_buffer(node.dest):
            self.stores.append(node)
        visitor.visit(node.value)

    def visit_load(self, node, _visitor):
        if is_global_or_shared_buffer(node.source) and any(_expr_depends_on_var(index, self.loop_var) for index in node.indices):
            self.loads.append(node)
            self.has_conditional_load |= self.conditional_depth > 0

    def visit_call(self, node, visitor):
        if node.op.same_as(_IF_THEN_ELSE_OP):
            self.conditional_depth += 1
            try:
                visitor.visit(node.args[1])
                visitor.visit(node.args[2])
            finally:
                self.conditional_depth -= 1
        else:
            visitor.visit(node.args)

    def visit_select(self, node, visitor):
        self.conditional_depth += 1
        try:
            visitor.visit(node.true_value)
            visitor.visit(node.false_value)
        finally:
            self.conditional_depth -= 1


def _buf_indices_match(left_tensor, left_indices, right_tensor, right_indices):
    return (
        left_tensor.same_as(right_tensor)
        and len(left_indices) == len(right_indices)
        and all(tvm_ffi.structural_equal(left, right) for left, right in zip(left_indices, right_indices))
    )


def _find_cast_entry(entries, tensor, indices):
    for original, original_indices, scratch in entries:
        if _buf_indices_match(original, original_indices, tensor, indices):
            return scratch
    return None


class AccessReplacer:
    def __init__(self, store_entries, load_entries, ordinal):
        self.store_entries = store_entries
        self.load_entries = load_entries
        self.ordinal = ordinal

    def visit_stmt(self, statement):
        return tvm_ffi.structural_mutate(statement, [(ir.TensorStore, self.visit_store), (ir.TensorLoad, self.visit_load)])

    def visit_store(self, node, mutator):
        value = mutator.mutate(node.value)
        scratch = _find_cast_entry(self.store_entries, node.dest, node.indices)
        if scratch is not None:
            return ir.TensorStore(scratch, [self.ordinal], value, loc=node.loc)
        return ir.TensorStore(node.dest, node.indices, value, loc=node.loc)

    def visit_load(self, node, _mutator):
        scratch = _find_cast_entry(self.load_entries, node.source, node.indices)
        return ir.TensorLoad(scratch, [self.ordinal], loc=node.loc) if scratch is not None else node


class DecoupleTypeCastMutator:
    def __init__(self):
        self._var_counter = 0
        self.analyzer = create_analyzer()

    def visit_stmt(self, statement):
        return tvm_ffi.structural_mutate(statement, (ir.For, self.visit_for))

    def _make_for(self, original, body, variable=None):
        return ir.For(
            original.var if variable is None else variable,
            original.begin,
            original.end,
            original.step,
            original.kind,
            body,
            original.annotations,
            loc=original.loc,
        )

    def visit_for(self, node, mutator):
        node = mutator.default_mutate(node)
        if node.kind != ir.ForKind.VECTORIZED or not isinstance(node.step, ir.IntImm) or node.step.value != 1:
            return node
        extent = self.analyzer.simplify(node.end - node.begin)
        if not isinstance(extent, ir.IntImm) or extent.value <= 0:
            return node
        normalized = normalize_flat_binds(node.body)
        if not _has_cast(normalized):
            return node
        condition, root = extract_if_condition(normalized)
        if not isinstance(root, ir.TensorStore) or not _can_repeat_condition(condition):
            return node
        if not _can_reorder(root.value, allow_loads=True):
            return node
        collector = MemoryAccessCollector(node.var)
        collector.visit_stmt(normalized)
        if not collector.stores and not collector.loads:
            return node
        # Hoisting a branch-only load into an unconditional copy loop can read
        # an invalid address. Keep the original vector loop unless a later
        # pass can attach the branch's exact predicate to that copy edge.
        if collector.has_conditional_load:
            return node
        stores = self._create_cast_entries(collector.stores, extent)
        loads = self._create_cast_entries(
            [load for load in collector.loads if _find_cast_entry(stores, load.source, load.indices) is None], extent
        )
        read_modify_write = [
            entry for entry in stores if any(_buf_indices_match(entry[0], entry[1], load.source, load.indices) for load in collector.loads)
        ]
        copies_in = self._create_copy_loops(node, loads + read_modify_write, "from_memory", condition)
        compute = self._make_for(node, AccessReplacer(stores, stores + loads, node.var - node.begin).visit_stmt(normalized))
        copies_out = self._create_copy_loops(node, stores, "to_memory", condition)
        return self._wrap_with_allocations([*copies_in, compute, *copies_out], stores + loads, node.loc)

    def _create_cast_entries(self, accesses, extent):
        entries = []
        for access in accesses:
            tensor = access.dest if isinstance(access, ir.TensorStore) else access.source
            # Each distinct logical access gets a separate ordinal scratch.
            if _find_cast_entry(entries, tensor, access.indices) is not None:
                continue
            suffix = "" if self._var_counter == 0 else f"_{self._var_counter}"
            self._var_counter += 1
            ty = tirx.TensorType(
                (extent,),
                tensor.ty.dtype,
                scope="local",
                layout="default",
            )
            scratch = ir.Var(f"{getattr(tensor, 'name', 'tensor')}_local_cast{suffix}", ty, loc=access.loc)
            entries.append((tensor, list(access.indices), scratch))
        return entries

    def _create_copy_loops(self, original, entries, direction, condition):
        copies = []
        for tensor, indices, scratch in entries:
            variable = ir.Var(f"{original.var.name}_copy", original.var.ty, loc=original.var.loc)
            replacements = {original.var: variable}
            new_indices = [_substitute_bind_env(index, replacements) for index in indices]
            ordinal = variable - original.begin
            if direction == "to_memory":
                store = ir.TensorStore(tensor, new_indices, ir.TensorLoad(scratch, [ordinal]), loc=original.loc)
            else:
                store = ir.TensorStore(scratch, [ordinal], ir.TensorLoad(tensor, new_indices), loc=original.loc)
            if condition is not None:
                store = ir.If(_substitute_bind_env(condition, replacements), store, None, loc=original.loc)
            copies.append(self._make_for(original, store, variable))
        return copies

    @staticmethod
    def _wrap_with_allocations(body, entries, location):
        if not entries:
            return ir.SeqStmt(body, loc=location)
        bindings = [
            ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty], loc=location), loc=location)
            for _, _, scratch in entries
        ]
        schema, payload = encode_annotations({"node": None, "key": "lexical_alloc_scope", "value": 1})
        return ir.RegionStmt(
            "tl.attr",
            [payload],
            [],
            {"schema_version": 1, "annotation_schema": schema},
            [*bindings, *body],
            loc=location,
        )


def DecoupleTypeCast():
    """Stage cast-bearing unit-step vector loops before vectorization/storage.

    Mutable guards, opaque calls, compound store bodies and conditional memory
    loads retain their original computation. The pass does not change their
    execution semantics merely to obtain wider memory instructions.
    """

    def pass_fn(func, _mod, _ctx):
        if func.attrs.get("declaration", False):
            return func
        return func.with_body(DecoupleTypeCastMutator().visit_stmt(func.body), loc=func.loc)

    return tirx.transform.function_pass(pass_fn, opt_level=0, name="tl.DecoupleTypeCast")
