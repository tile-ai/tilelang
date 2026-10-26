"""Decouple vectorized computation from mixed-precision memory transfers.

Staging values live in native local TensorTypes. Their allocation Binds and
copy/compute loops share a native lexical region, so StorageRewrite retains
the allocation boundary instead of moving temporary storage to kernel entry.
"""

from __future__ import annotations

import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.ir.prim import if_then_else
from tvm_tirx.tirx.transform import function_pass

from tilelang._compat.functor import PyStmtExprMutator, PyStmtExprVisitor
from tilelang._compat.stmt_functor import post_order_visit, substitute
from tilelang._compat.tensor import TensorView


def is_local_buffer(tensor: ir.Expr) -> bool:
    return isinstance(tensor.ty, tirx.TensorType) and tensor.ty.scope.name.startswith("local")


def is_global_or_shared_buffer(tensor: ir.Expr) -> bool:
    return isinstance(tensor.ty, tirx.TensorType) and tensor.ty.scope.name.split(".", 1)[0] in ("global", "shared")


class _CastFinder(PyStmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.found = False

    def visit_cast_(self, expression):
        self.found = True

    def visit_tensor_store_(self, statement):
        self.visit_expr(statement.value)

    def visit_tensor_load_(self, expression):
        return None


def _has_cast(statement: ir.Stmt) -> bool:
    finder = _CastFinder()
    finder.visit_stmt(statement)
    return finder.found


def _expr_depends_on_var(expression: ir.Expr, variable: ir.Var) -> bool:
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, ir.Var) and node.same_as(variable):
            found = True

    post_order_visit(expression, visit)
    return found


def _is_conditional_call(expression: ir.Call) -> bool:
    return isinstance(expression.op, ir.Op) and expression.op.name == "prim.if_then_else"


class MemoryAccessCollector(PyStmtExprVisitor):
    """Collect memory transfers on the unconditional value path, not indices."""

    def __init__(self, loop_var: ir.Var):
        super().__init__()
        self.loop_var = loop_var
        self.stores = []
        self.loads = []

    def visit_tensor_store_(self, statement):
        if is_global_or_shared_buffer(statement.dest):
            self.stores.append(statement)
        self.visit_expr(statement.value)

    def visit_tensor_load_(self, expression):
        if is_global_or_shared_buffer(expression.source) and any(
            _expr_depends_on_var(index, self.loop_var) for index in expression.indices
        ):
            self.loads.append(expression)

    def visit_call_(self, expression):
        if not _is_conditional_call(expression):
            for argument in expression.args:
                self.visit_expr(argument)


def _can_repeat_expression(expression: ir.Expr | None, *, allow_loads=False) -> bool:
    if expression is None:
        return True
    repeatable = True

    def visit(node):
        nonlocal repeatable
        if isinstance(node, ir.TensorLoad) and not allow_loads:
            repeatable = False
        elif isinstance(node, ir.Call):
            effect = node.op.get_attr("TCallEffectKind") if isinstance(node.op, ir.Op) else None
            if effect is None or int(effect) > ir.CallEffectKind.Pure:
                repeatable = False

    post_order_visit(expression, visit)
    return repeatable


def _normalize_flat_binds(statement: ir.Stmt, environment: dict[ir.Var, ir.Expr]) -> ir.Stmt | None:
    if isinstance(statement, ir.Bind):
        value = substitute(statement.value, environment)
        if isinstance(statement.var.ty, ir.PrimType) and _can_repeat_expression(value, allow_loads=True):
            environment[statement.var] = value
            return None
        return ir.Bind(statement.var, value, loc=statement.loc)
    if isinstance(statement, ir.SeqStmt):
        local_environment = dict(environment)
        normalized = []
        for child in statement.seq:
            result = _normalize_flat_binds(child, local_environment)
            if result is not None:
                normalized.append(result)
        return ir.SeqStmt(normalized, loc=statement.loc)
    if isinstance(statement, ir.If):
        then_case = _normalize_flat_binds(statement.then_case, dict(environment))
        else_case = None
        if statement.else_case is not None:
            else_case = _normalize_flat_binds(statement.else_case, dict(environment))
        return ir.If(substitute(statement.condition, environment), then_case, else_case, loc=statement.loc)
    return substitute(statement, environment)


def normalize_flat_binds(statement: ir.Stmt) -> ir.Stmt:
    """Inline scalar definitions with sequential dominance, preserving effects."""
    normalized = _normalize_flat_binds(statement, {})
    return normalized if normalized is not None else statement


def _single_statement(statement: ir.Stmt) -> ir.Stmt:
    while isinstance(statement, ir.SeqStmt) and len(statement.seq) == 1:
        statement = statement.seq[0]
    return statement


def extract_if_condition(statement: ir.Stmt) -> tuple[ir.Expr | None, ir.Stmt]:
    condition = None
    statement = _single_statement(statement)
    while isinstance(statement, ir.If) and statement.else_case is None:
        condition = statement.condition if condition is None else if_then_else(condition, statement.condition, ir.const(False))
        statement = _single_statement(statement.then_case)
    return condition, statement


CastEntry = tuple[ir.Expr, list[ir.Expr], ir.Var]


def _buf_indices_match(first_tensor, first_indices, second_tensor, second_indices):
    return (
        first_tensor.same_as(second_tensor)
        and len(first_indices) == len(second_indices)
        and all(tvm_ffi.structural_equal(first, second) for first, second in zip(first_indices, second_indices))
    )


def _find_cast_entry(entries: list[CastEntry], tensor: ir.Expr, indices: list[ir.Expr]) -> ir.Var | None:
    for original_tensor, original_indices, temporary in entries:
        if _buf_indices_match(original_tensor, original_indices, tensor, indices):
            return temporary
    return None


class AccessReplacer(PyStmtExprMutator):
    def __init__(self, store_entries, load_entries, ordinal):
        super().__init__()
        self.store_entries = store_entries
        self.load_entries = load_entries
        self.ordinal = ordinal

    def visit_tensor_store_(self, statement):
        value = self.visit_expr(statement.value)
        temporary = _find_cast_entry(self.store_entries, statement.dest, statement.indices)
        if temporary is not None:
            return ir.TensorStore(temporary, [self.ordinal], value, loc=statement.loc)
        return ir.TensorStore(statement.dest, statement.indices, value, loc=statement.loc)

    def visit_tensor_load_(self, expression):
        temporary = _find_cast_entry(self.load_entries, expression.source, expression.indices)
        if temporary is not None:
            return ir.TensorLoad(temporary, [self.ordinal], loc=expression.loc)
        return expression

    def visit_call_(self, expression):
        if _is_conditional_call(expression):
            return expression
        return super().visit_call_(expression)


class DecoupleTypeCastMutator(PyStmtExprMutator):
    def __init__(self):
        super().__init__()
        self._var_counter = 0
        self._analyzer = sym.Analyzer()

    def _make_unique_name(self, base):
        name = f"{base}_{self._var_counter}" if self._var_counter else base
        self._var_counter += 1
        return name

    @staticmethod
    def _make_for(original, body):
        if body.same_as(original.body):
            return original
        return ir.For(
            original.var,
            original.begin,
            original.end,
            original.step,
            original.kind,
            body,
            original.annotations,
            loc=original.loc,
        )

    def visit_for_(self, statement):
        body = self.visit_stmt(statement.body)
        unchanged = self._make_for(statement, body)
        if statement.kind != ir.ForKind.VECTORIZED or not isinstance(statement.step, ir.IntImm) or int(statement.step) != 1:
            return unchanged
        extent = self._analyzer.simplify(statement.end - statement.begin)
        if not isinstance(extent, ir.IntImm) or int(extent) <= 0:
            return unchanged
        normalized = normalize_flat_binds(body)
        if not _has_cast(normalized):
            return unchanged
        condition, root = extract_if_condition(normalized)
        if not isinstance(root, ir.TensorStore) or not _can_repeat_expression(condition):
            return unchanged

        collector = MemoryAccessCollector(statement.var)
        collector.visit_stmt(root)
        if not collector.stores and not collector.loads:
            return unchanged

        store_entries = self._create_cast_entries(collector.stores, extent, statement.loc)
        uncovered_loads = [load for load in collector.loads if _find_cast_entry(store_entries, load.source, load.indices) is None]
        load_entries = self._create_cast_entries(uncovered_loads, extent, statement.loc)
        read_modify_write = [
            entry
            for entry in store_entries
            if any(_buf_indices_match(entry[0], entry[1], load.source, load.indices) for load in collector.loads)
        ]
        copy_from = self._create_copy_loops(statement, load_entries + read_modify_write, to_memory=False, condition=condition)
        compute_body = AccessReplacer(store_entries, load_entries + read_modify_write, statement.var - statement.begin).visit_stmt(
            normalized
        )
        compute = self._make_for(statement, compute_body)
        copy_to = self._create_copy_loops(statement, store_entries, to_memory=True, condition=condition)
        definitions = [TensorView(temporary).allocate(loc=statement.loc) for _, _, temporary in store_entries + load_entries]
        return ir.RegionStmt(
            "tl.scope",
            [],
            [],
            None,
            ir.SeqStmt([*definitions, *copy_from, compute, *copy_to], loc=statement.loc),
            loc=statement.loc,
        )

    def _create_cast_entries(self, accesses, extent, loc):
        entries = []
        for access in accesses:
            tensor = access.dest if isinstance(access, ir.TensorStore) else access.source
            indices = list(access.indices)
            if _find_cast_entry(entries, tensor, indices) is not None:
                continue
            name = tensor.name if isinstance(tensor, ir.Var) else "tensor"
            temporary = TensorView.create(
                self._make_unique_name(f"{name}_local_cast"),
                [extent],
                tensor.ty.dtype,
                scope="local",
                loc=loc,
            ).tensor
            entries.append((tensor, indices, temporary))
        return entries

    @staticmethod
    def _create_copy_loops(original, entries, *, to_memory, condition):
        loops = []
        for tensor, indices, temporary in entries:
            variable = ir.Var(f"{original.var.name}_copy", original.var.ty, loc=original.loc)
            new_indices = [substitute(index, {original.var: variable}) for index in indices]
            ordinal = variable - original.begin
            if to_memory:
                body = ir.TensorStore(tensor, new_indices, ir.TensorLoad(temporary, [ordinal], loc=original.loc), loc=original.loc)
            else:
                body = ir.TensorStore(temporary, [ordinal], ir.TensorLoad(tensor, new_indices, loc=original.loc), loc=original.loc)
            if condition is not None:
                body = ir.If(substitute(condition, {original.var: variable}), body, None, loc=original.loc)
            loops.append(
                ir.For(
                    variable,
                    original.begin,
                    original.end,
                    original.step,
                    ir.ForKind.VECTORIZED,
                    body,
                    original.annotations,
                    loc=original.loc,
                )
            )
        return loops


def DecoupleTypeCast():
    """Split mixed-precision vector loops before native vector/storage lowering."""

    def pass_fn(function, mod, ctx):
        if function.body is None:
            return function
        body = DecoupleTypeCastMutator().visit_stmt(function.body)
        return function.with_body(body, loc=function.loc)

    return function_pass(pass_fn, opt_level=0, name="tl.DecoupleTypeCast")
