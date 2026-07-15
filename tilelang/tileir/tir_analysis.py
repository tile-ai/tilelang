"""TIR analysis helpers used by the TileIR lowering boundary."""

from __future__ import annotations

from typing import Any

from .errors import TileIRLoweringError
from tvm import tirx


def _has_kernel_launch(stmt: tirx.Stmt) -> bool:
    found = False

    def visit(node):
        nonlocal found
        if not isinstance(node, tirx.AttrStmt) or node.attr_key != "thread_extent":
            return
        if str(getattr(node.node, "thread_tag", "")) == "blockIdx.x":
            found = True

    tirx.stmt_functor.post_order_visit(stmt, visit)
    return found


def _kernel_launch_stmts(prim_func: tirx.PrimFunc) -> tuple[tirx.Stmt, ...]:
    """Return the root statements that represent device kernel launches."""

    if not isinstance(prim_func.body, tirx.SBlockRealize):
        return (prim_func.body,)
    body = prim_func.body.block.body
    if isinstance(body, tirx.SeqStmt):
        launches = tuple(stmt for stmt in body.seq if _has_kernel_launch(stmt))
        if launches:
            if len(launches) != len(body.seq):
                raise TileIRLoweringError(
                    "TileIR lowering found a non-kernel root statement; "
                    "host bookkeeping must be represented explicitly before kernel discovery."
                )
            return launches
    return (body,)


def _as_static_int(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, tirx.IntImm):
        result = int(value)
    elif isinstance(value, int):
        result = value
    else:
        raise TileIRLoweringError(f"TileIR backend requires static integer {field}; got {value}.")
    if result < minimum:
        raise TileIRLoweringError(f"TileIR backend requires {field} >= {minimum}; got {result}.")
    return result


def _as_launch_extent(value: Any, *, field: str, minimum: int) -> int | tirx.PrimExpr:
    if isinstance(value, (tirx.IntImm, int)):
        return _as_static_int(value, field=field, minimum=minimum)
    if isinstance(value, tirx.PrimExpr):
        return value
    raise TileIRLoweringError(f"TileIR backend requires integer launch extent {field}; got {value}.")


def _format_coverage_gap(prim_func: tirx.PrimFunc) -> str:
    node_counts: dict[str, int] = {}
    call_ops: set[str] = set()

    def visit(node):
        node_counts[type(node).__name__] = node_counts.get(type(node).__name__, 0) + 1
        if isinstance(node, tirx.Call):
            call_ops.add(_op_name(node))

    tirx.stmt_functor.post_order_visit(prim_func.body, visit)
    top_nodes = ", ".join(f"{name}={count}" for name, count in sorted(node_counts.items())[:12])
    top_calls = ", ".join(sorted(call_ops)[:16]) or "none"
    return f"TIR nodes: {top_nodes or 'none'}; call ops: {top_calls}"


def _op_name(call: tirx.Call) -> str:
    name = getattr(call.op, "name", str(call.op))
    if name.startswith("tirx."):
        return "tir." + name[len("tirx.") :]
    return name


def _canonical_dtype_name(dtype: Any) -> str:
    name = str(dtype)
    if name in {"custom[tfloat32]", "custom[tf32]"}:
        return "tfloat32"
    return name


def _static_shape(values: Any, *, field: str) -> list[int]:
    return [_as_static_int(value, field=field, minimum=0) for value in values]


def _static_dim_or_none(value: Any) -> int | None:
    if isinstance(value, tirx.IntImm):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _static_shape_metadata(values: Any) -> list[int | None]:
    return [_static_dim_or_none(value) for value in values]


def _static_stride_metadata(values: Any) -> list[int | None]:
    return [_static_dim_or_none(value) for value in values]


def _contiguous_strides(shape: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= dim
    return tuple(reversed(strides))


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise TileIRLoweringError(f"TileIR tile dimensions must be positive; got {value}.")
    return 1 << (value - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _power_of_two_shift(value: int) -> int:
    if not _is_power_of_two(value):
        raise TileIRLoweringError(f"Expected a power-of-two integer; got {value}.")
    return value.bit_length() - 1


def _is_power_of_two_shape(shape: list[int] | tuple[int, ...]) -> bool:
    return all(_is_power_of_two(dim) for dim in shape)


def _tile_storage_shape(logical_shape: list[int] | tuple[int, ...]) -> list[int]:
    return [_next_power_of_two(dim) for dim in logical_shape]


def _power_of_two_divisor(value: int, *, limit: int) -> int:
    divisor = 1
    while divisor < limit and value % (divisor * 2) == 0:
        divisor *= 2
    return divisor


def _int_imm(value: int, dtype: str = "int32") -> tirx.IntImm:
    return tirx.IntImm(dtype, value)


def _int_imm_value(expr: tirx.PrimExpr) -> int | None:
    if isinstance(expr, tirx.IntImm):
        return int(expr)
    return None


def _combine_affine(
    lhs: tuple[int, dict[str, int]],
    rhs: tuple[int, dict[str, int]],
    *,
    sign: int = 1,
) -> tuple[int, dict[str, int]]:
    const = lhs[0] + sign * rhs[0]
    coeffs = dict(lhs[1])
    for var, coeff in rhs[1].items():
        next_coeff = coeffs.get(var, 0) + sign * coeff
        if next_coeff == 0:
            coeffs.pop(var, None)
        else:
            coeffs[var] = next_coeff
    return const, coeffs


def _scale_affine(value: tuple[int, dict[str, int]], scale: int) -> tuple[int, dict[str, int]]:
    if scale == 0:
        return 0, {}
    return value[0] * scale, {var: coeff * scale for var, coeff in value[1].items()}


def _affine_int_expr(
    expr: Any,
    bindings: dict[str, tirx.PrimExpr] | None = None,
    active: set[str] | None = None,
) -> tuple[int, dict[str, int]] | None:
    """Return an affine integer expression as ``constant + sum(coeff * var)``.

    This is intentionally small and syntactic: it exists to prove static tile
    region extents such as ``kv_end - kv_start`` after TileLang introduces let
    variables.  Dynamic variables are kept symbolic and must cancel before a
    caller can treat the result as a static integer.
    """

    if isinstance(expr, tirx.IntImm):
        return int(expr), {}
    if isinstance(expr, int):
        return expr, {}
    if isinstance(expr, tirx.Cast):
        return _affine_int_expr(expr.value, bindings, active)
    if isinstance(expr, tirx.Var):
        if bindings and expr.name in bindings:
            active = active or set()
            if expr.name in active:
                return None
            active.add(expr.name)
            try:
                bound = _affine_int_expr(bindings[expr.name], bindings, active)
            finally:
                active.remove(expr.name)
            if bound is not None:
                return bound
        return 0, {expr.name: 1}
    if isinstance(expr, tirx.Add):
        lhs = _affine_int_expr(expr.a, bindings, active)
        rhs = _affine_int_expr(expr.b, bindings, active)
        return None if lhs is None or rhs is None else _combine_affine(lhs, rhs)
    if isinstance(expr, tirx.Sub):
        lhs = _affine_int_expr(expr.a, bindings, active)
        rhs = _affine_int_expr(expr.b, bindings, active)
        return None if lhs is None or rhs is None else _combine_affine(lhs, rhs, sign=-1)
    if isinstance(expr, tirx.Mul):
        lhs = _affine_int_expr(expr.a, bindings, active)
        rhs = _affine_int_expr(expr.b, bindings, active)
        if lhs is None or rhs is None:
            return None
        if lhs[1] and rhs[1]:
            return None
        if lhs[1]:
            return _scale_affine(lhs, rhs[0])
        if rhs[1]:
            return _scale_affine(rhs, lhs[0])
        return lhs[0] * rhs[0], {}
    if isinstance(expr, tirx.FloorDiv):
        lhs = _affine_int_expr(expr.a, bindings, active)
        rhs = _affine_int_expr(expr.b, bindings, active)
        if lhs is None or rhs is None or rhs[1]:
            return None
        divisor = rhs[0]
        if divisor == 0:
            return None
        if lhs[0] % divisor != 0 or any(coeff % divisor != 0 for coeff in lhs[1].values()):
            return None
        return lhs[0] // divisor, {var: coeff // divisor for var, coeff in lhs[1].items()}
    return None


def _static_int_expr_value(expr: Any, bindings: dict[str, tirx.PrimExpr] | None = None) -> int | None:
    affine = _affine_int_expr(expr, bindings)
    if affine is None:
        return None
    const, coeffs = affine
    return const if not coeffs else None


def _expr_dtype(expr: tirx.PrimExpr) -> str:
    return _canonical_dtype_name(getattr(expr, "dtype", "int32"))


def _dtype_is_float_name(dtype: str) -> bool:
    dtype = _canonical_dtype_name(dtype)
    return dtype.startswith("float") or dtype.startswith("bfloat") or dtype in {"tf32", "tfloat32"}


def _mul_expr(lhs: tirx.PrimExpr, rhs: tirx.PrimExpr) -> tirx.PrimExpr:
    lhs_value = _int_imm_value(lhs)
    rhs_value = _int_imm_value(rhs)
    if lhs_value == 0 or rhs_value == 0:
        return _int_imm(0, _expr_dtype(lhs))
    if lhs_value == 1:
        return rhs
    if rhs_value == 1:
        return lhs
    return lhs * rhs


def _add_expr(lhs: tirx.PrimExpr, rhs: tirx.PrimExpr) -> tirx.PrimExpr:
    lhs_value = _int_imm_value(lhs)
    rhs_value = _int_imm_value(rhs)
    if lhs_value == 0:
        return rhs
    if rhs_value == 0:
        return lhs
    return lhs + rhs


def _sub_expr(lhs: tirx.PrimExpr, rhs: tirx.PrimExpr) -> tirx.PrimExpr:
    rhs_value = _int_imm_value(rhs)
    if rhs_value == 0:
        return lhs
    return lhs - rhs


def _divide_tir_expr_if_exact(expr: tirx.PrimExpr, divisor: int) -> tirx.PrimExpr | None:
    """Return ``expr / divisor`` when syntactic divisibility is proven exactly."""

    if divisor == 1:
        return expr
    value = _int_imm_value(expr)
    if value is not None:
        if value % divisor != 0:
            return None
        return _int_imm(value // divisor, _expr_dtype(expr))
    if isinstance(expr, tirx.Mul):
        lhs_value = _int_imm_value(expr.a)
        if lhs_value is not None and lhs_value % divisor == 0:
            return _mul_expr(_int_imm(lhs_value // divisor, _expr_dtype(expr.a)), expr.b)
        rhs_value = _int_imm_value(expr.b)
        if rhs_value is not None and rhs_value % divisor == 0:
            return _mul_expr(expr.a, _int_imm(rhs_value // divisor, _expr_dtype(expr.b)))
        lhs_quotient = _divide_tir_expr_if_exact(expr.a, divisor)
        if lhs_quotient is not None:
            return _mul_expr(lhs_quotient, expr.b)
        rhs_quotient = _divide_tir_expr_if_exact(expr.b, divisor)
        if rhs_quotient is not None:
            return _mul_expr(expr.a, rhs_quotient)
        return None
    if isinstance(expr, tirx.Add):
        lhs = _divide_tir_expr_if_exact(expr.a, divisor)
        rhs = _divide_tir_expr_if_exact(expr.b, divisor)
        if lhs is None or rhs is None:
            return None
        return _add_expr(lhs, rhs)
    if isinstance(expr, tirx.Sub):
        lhs = _divide_tir_expr_if_exact(expr.a, divisor)
        rhs = _divide_tir_expr_if_exact(expr.b, divisor)
        if lhs is None or rhs is None:
            return None
        return _sub_expr(lhs, rhs)
    return None
