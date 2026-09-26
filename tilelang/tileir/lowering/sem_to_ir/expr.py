"""SemanticIR -> TileIR lowering: scalar expression lowering.

Provides ``lower_expr`` (the recursive scalar TIR PrimExpr -> TileIR Value
lowering) plus the FMA-detection and Elementwise helpers it depends on.
Imports only the shared foundation (``_base``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tvm import tirx as _tirx

from tilelang.tileir.ir.builder import IRBuilder
from tilelang.tileir.errors import _UnboundScopeVariable, _UnsupportedTileIRNode
from tilelang.tileir.ir.types import MemSpace, TileType
from tilelang.tileir.ir.value import Value
from tilelang.tileir.ir.ops import Cast, Constant, Elementwise, Load, Select
from tilelang.tileir.emission_utils import _binary_result_shape, _select_result_shape

from ._base import (
    _canonical_dtype_str,
    LoweringScope,
    _CMP_FNS,
    _UNARY_CALL_FN,
    _get_binary_op_fn,
    _op_name_from_call,
    _scalar_bool_type,
    _scalar_i32_type,
    _tir_dtype_to_tile_type,
)


def _try_lower_fma(expr: Any, scope: LoweringScope, builder: IRBuilder) -> Value | None:
    """Detect a*b+c or a*b-c and lower to Elementwise(fn='fma').

    Only matches when expr is Add/Sub with a Mul child — float FMA only.
    Gated behind scope.fast_math: only emit FMA when fast_math is enabled.
    Returns None if pattern does not match or fast_math is disabled.
    """
    # Gate FMA fusion behind fast_math; precise mode uses separate mul+add.
    if not scope.fast_math:
        return None

    is_add = isinstance(expr, _tirx.Add)
    is_sub = isinstance(expr, _tirx.Sub)
    if not (is_add or is_sub):
        return None

    def _is_mul(e: Any) -> bool:
        return isinstance(e, _tirx.Mul)

    def _is_float_tir(e: Any) -> bool:
        dtype_str = str(getattr(e, "dtype", ""))
        return any(dtype_str.startswith(p) for p in ("float", "bfloat"))

    if is_add:
        # Require BOTH children to be float — mul child AND addend c —
        # so that integer Add(Mul(a,b), c) is not incorrectly emitted as FMA.
        if _is_mul(expr.a) and _is_float_tir(expr.a) and _is_float_tir(expr.b):
            a = lower_expr(expr.a.a, scope, builder)
            b = lower_expr(expr.a.b, scope, builder)
            c = lower_expr(expr.b, scope, builder)
            return _make_elementwise("fma", (a, b, c), expr, builder)
        if _is_mul(expr.b) and _is_float_tir(expr.b) and _is_float_tir(expr.a):
            a = lower_expr(expr.b.a, scope, builder)
            b = lower_expr(expr.b.b, scope, builder)
            c = lower_expr(expr.a, scope, builder)
            return _make_elementwise("fma", (a, b, c), expr, builder)
    # Not fused: Sub (a*b - c) would need to negate c for the third fma operand;
    # it is rare and left as a plain mul+sub rather than adding the special case.
    return None


def _lower_attr_expr(value: Any, scope: LoweringScope, builder: IRBuilder) -> Value | None:
    """Lower a SemanticStmt attr value that may be a TIR PrimExpr OR a serialized string.

    SemanticStmt.attrs uses _attrs() which calls _expr_text() to serialize TIR
    expressions as strings (e.g. IntImm(4) → "4").  Handlers that later call
    lower_expr on these attrs must go through this wrapper, which:
      - Passes real TIR PrimExpr nodes straight to lower_expr.
      - Parses simple numeric strings back to Constant ops.
      - Returns None for None values.
    Complex serialized exprs (not just plain integers) are not yet supported
    and raise _UnsupportedTileIRNode with a clear message.
    """
    if value is None:
        return None
    # Real TIR node — lower directly.
    if hasattr(value, "dtype") and not isinstance(value, str):
        return lower_expr(value, scope, builder)
    # String-serialized value from _attrs().
    if isinstance(value, str):
        try:
            int_val = int(value)
            ty = _scalar_i32_type()
            op = builder.create(Constant(value=int_val, dtype="int32"), result_types=(ty,))
            return op.results[0]
        except ValueError:
            pass
        try:
            float_val = float(value)
            ty = _tir_dtype_to_tile_type("float32")
            op = builder.create(Constant(value=float_val, dtype="float32"), result_types=(ty,))
            return op.results[0]
        except ValueError:
            pass
        raise _UnsupportedTileIRNode(
            f"SemanticStmt attr value {value!r} is a non-numeric string. Complex TIR expressions serialized to strings cannot be lowered."
        )
    raise _UnsupportedTileIRNode(f"SemanticStmt attr value of unexpected type {type(value).__name__!r}: {value!r}")


def _make_elementwise(fn: str, operands: tuple, expr: Any, builder: IRBuilder, unsigned: bool = False) -> Value:
    """Create an Elementwise op and return its result Value.

    ``unsigned`` (uint signedness) is threaded through to
    ``Elementwise.unsigned`` and consulted at emit time for "max"/"min"/
    comparison functions (see ``Elementwise``'s docstring). Callers derive it
    from the operands' TIR dtype rather than ``expr.dtype``, which is always
    "bool" for comparisons regardless of the operand dtype.
    """
    # Determine result dtype: comparisons produce bool, others use expr dtype.
    if fn in _CMP_FNS:
        result_ty = _scalar_bool_type()
    else:
        dtype_str = str(getattr(expr, "dtype", "int32"))
        result_ty = _tir_dtype_to_tile_type(dtype_str)
    shape = ()
    for value in operands:
        shape = _binary_result_shape(shape, tuple(value.type.shape))
    result_ty = replace(result_ty, shape=shape)
    op = Elementwise(fn=fn, inputs=operands, unsigned=unsigned)
    created = builder.create(op, result_types=(result_ty,))
    return created.results[0]


def _make_select(cond: Value, true: Value, false: Value, expr: Any, builder: IRBuilder) -> Value:
    """Create a Select with the same shape contract as its MLIR emitter."""
    shape = _select_result_shape(tuple(cond.type.shape), tuple(true.type.shape), tuple(false.type.shape))
    result_ty = replace(_tir_dtype_to_tile_type(str(expr.dtype)), shape=shape)
    op = builder.create(Select(cond=cond, true_val=true, false_val=false), result_types=(result_ty,))
    return op.results[0]


def _lower_buffer_load_scalar(expr: Any, scope: LoweringScope, builder: IRBuilder) -> Value:
    """Lower a scalar-context TIR ``BufferLoad`` to an indexed (or 0-rank) Load."""
    buf_name = expr.buffer.name
    try:
        buf_val = scope.lookup_buffer(buf_name)
    except KeyError as exc:
        raise _UnsupportedTileIRNode(
            f"BufferLoad from unknown buffer {buf_name!r}: buffer is not registered in the lowering scope."
        ) from exc
    # Pass actual indices to Load so that scalar loads from GLOBAL buffers
    # (e.g. block_expert[bx]) use the correct element offset instead of
    # always loading from index 0.
    #
    # Indexed-scalar load also covers SHARED/REGISTER tile buffers so that
    # per-thread scatter kernels can gather individual
    # elements (e.g. B_shared[vi, vj] where vi/vj depend on threadIdx.x).
    # Load.emit_mlir handles this via ct.extract on the in-register tile.
    #
    # Indexed global loads use the TensorView partition path. Indexed shared
    # and register loads use ct.extract to gather the scalar.
    elem_dtype = buf_val.type.dtype
    if expr.indices:
        # An index that cannot be lowered must fail instead of falling back to
        # index zero, which would silently read the wrong element.
        lowered_indices = []
        for pos, idx in enumerate(expr.indices):
            try:
                idx_val = lower_expr(idx, scope, builder)
            except Exception as exc:
                raise _UnsupportedTileIRNode(
                    f"BufferLoad from {buf_name!r}: cannot lower index {pos} ({idx!r}); refusing to fall back to a scalar load at index 0"
                ) from exc
            lowered_indices.append(idx_val)
        # tile_shape=() (scalar) with explicit indices.
        # For GLOBAL: Load.emit_mlir uses the TensorView partition path.
        # For SHARED/REGISTER: Load.emit_mlir uses ct.extract.
        result_ty = TileType(dtype=elem_dtype, shape=(), space=MemSpace.REGISTER, layout=None)
        load_op = builder.create(
            Load(src=buf_val, tile_shape=(), indices=tuple(lowered_indices)),
            result_types=(result_ty,),
        )
        return load_op.results[0]
    # No-index scalar load (e.g. a 0-rank buffer) is still legitimate.
    result_ty = TileType(dtype=elem_dtype, shape=(), space=MemSpace.REGISTER, layout=None)
    load_op = builder.create(
        Load(src=buf_val, tile_shape=(), indices=()),
        result_types=(result_ty,),
    )
    return load_op.results[0]


def _lower_call_scalar(expr: Any, scope: LoweringScope, builder: IRBuilder) -> Value:
    """Lower a scalar-context TIR ``Call``: unary math, if_then_else, bitwise/shift/pow, tl.infinity, reinterpret."""
    op_name = _op_name_from_call(expr)
    result_dtype_str = str(getattr(expr, "dtype", "int32"))
    result_ty = _tir_dtype_to_tile_type(result_dtype_str)

    # Unary math functions
    unary_fn = _UNARY_CALL_FN.get(op_name)
    if unary_fn is not None and len(expr.args) == 1:
        arg_val = lower_expr(expr.args[0], scope, builder)
        return _make_elementwise(unary_fn, (arg_val,), expr, builder)

    # if_then_else → Select
    if op_name in ("tir.if_then_else",):
        cond_val = lower_expr(expr.args[0], scope, builder)
        true_val = lower_expr(expr.args[1], scope, builder)
        false_val = lower_expr(expr.args[2], scope, builder)
        return _make_select(cond_val, true_val, false_val, expr, builder)

    # Bitwise ops (tir.bitwise_and/or/xor/not) → Elementwise
    if op_name == "tir.bitwise_and" and len(expr.args) == 2:
        lhs = lower_expr(expr.args[0], scope, builder)
        rhs = lower_expr(expr.args[1], scope, builder)
        return _make_elementwise("andi", (lhs, rhs), expr, builder)
    if op_name == "tir.bitwise_or" and len(expr.args) == 2:
        lhs = lower_expr(expr.args[0], scope, builder)
        rhs = lower_expr(expr.args[1], scope, builder)
        return _make_elementwise("ori", (lhs, rhs), expr, builder)
    if op_name == "tir.bitwise_xor" and len(expr.args) == 2:
        lhs = lower_expr(expr.args[0], scope, builder)
        rhs = lower_expr(expr.args[1], scope, builder)
        return _make_elementwise("xori", (lhs, rhs), expr, builder)
    if op_name == "tir.bitwise_not" and len(expr.args) == 1:
        arg_val = lower_expr(expr.args[0], scope, builder)
        # "not" (xori(x,1)) is correct ONLY for i1/bool; a real
        # integer ~x must flip all bits -> dispatch to "bitwise_not".
        _bn_dtype = str(getattr(expr, "dtype", "")) or result_dtype_str
        _bn_fn = "not" if _bn_dtype in ("bool", "int1", "uint1") else "bitwise_not"
        return _make_elementwise(_bn_fn, (arg_val,), expr, builder)

    # Shift ops — lower to Elementwise shl/shr.
    if op_name in ("tir.shift_left", "tir.shift_right") and len(expr.args) == 2:
        lhs = lower_expr(expr.args[0], scope, builder)
        rhs = lower_expr(expr.args[1], scope, builder)
        if op_name == "tir.shift_left":
            fn = "shl"
        else:
            # uint types must use "shr_unsigned" (→ shri UNSIGNED).
            expr_dtype_str = str(getattr(expr, "dtype", "")) or ""
            fn = "shr_unsigned" if expr_dtype_str.startswith("uint") else "shr"
        return _make_elementwise(fn, (lhs, rhs), expr, builder)

    # pow
    if op_name in ("tir.pow", "tl.pow_of_int") and len(expr.args) == 2:
        lhs = lower_expr(expr.args[0], scope, builder)
        rhs = lower_expr(expr.args[1], scope, builder)
        return _make_elementwise("pow", (lhs, rhs), expr, builder)

    # atan2
    if op_name == "tir.atan2" and len(expr.args) == 2:
        lhs = lower_expr(expr.args[0], scope, builder)
        rhs = lower_expr(expr.args[1], scope, builder)
        return _make_elementwise("atan2", (lhs, rhs), expr, builder)

    # tl.infinity → float constant
    if op_name == "tl.infinity":
        inf_dtype = str(getattr(expr.args[0], "value", "float32")) if expr.args else "float32"
        inf_ty = _tir_dtype_to_tile_type(inf_dtype)
        op = builder.create(Constant(value=float("inf"), dtype=inf_dtype), result_types=(inf_ty,))
        return op.results[0]

    # tir.reinterpret maps to a bit reinterpretation, not a numeric cast.
    if op_name == "tir.reinterpret" and len(expr.args) == 1:
        src_val = lower_expr(expr.args[0], scope, builder)
        src_dtype_str = str(getattr(expr.args[0], "dtype", ""))
        cast_op = builder.create(
            Cast(src=src_val, dtype=result_dtype_str, src_dtype=src_dtype_str, bitcast=True),
            result_types=(replace(result_ty, shape=src_val.type.shape),),
        )
        return cast_op.results[0]

    # Unknown call → raise loudly so unsupported exprs are diagnosable.
    raise _UnsupportedTileIRNode(
        f"Call op {op_name!r} is not yet lowered in TileIR. Add a handler to _UNARY_CALL_FN or the explicit call dispatch in lower_expr."
    )


def lower_expr(expr: Any, scope: LoweringScope, builder: IRBuilder) -> Value:
    """Lower a TIR ``PrimExpr`` to a TileIR ``Value``.

    Full recursive expression lowering.  Scalars are represented
    as 0-d (shape=()) TileType Values.  Builds TileIR ops (Constant,
    Elementwise, Cast, Select, Load) via IRBuilder.create and does not call
    MLIR directly.
    """
    # IntImm / FloatImm
    if isinstance(expr, _tirx.IntImm):
        dtype_str = str(expr.dtype)
        result_ty = _tir_dtype_to_tile_type(dtype_str)
        op = builder.create(Constant(value=int(expr), dtype=dtype_str), result_types=(result_ty,))
        return op.results[0]

    if isinstance(expr, _tirx.FloatImm):
        dtype_str = str(expr.dtype)
        result_ty = _tir_dtype_to_tile_type(dtype_str)
        op = builder.create(Constant(value=float(expr), dtype=dtype_str), result_types=(result_ty,))
        return op.results[0]

    # Var: scope lookup (loop vars, let bindings, thread/block index vars).
    # Look up by the Var OBJECT (identity) so a coordinate that shares a name
    # with a launch axis (e.g. a T.Persistent `bx`) resolves to itself, not the
    # axis; a name fallback inside scope.lookup still resolves name-keyed
    # bindings (scalar entry params).
    if isinstance(expr, _tirx.Var):
        bound = scope.lookup(expr)
        if bound is not None:
            return bound
        # Replay: a `tirx.Bind` whose value expression referenced a
        # not-yet-bound variable (e.g. `T.Persistent`'s bx/by binds, which
        # reference the enclosing `for w in range(waves)` loop's induction
        # variable but are placed textually before it) was deferred by
        # `_lower_let` instead of eagerly evaluated. Re-lower its raw
        # expression now, in the CURRENT scope -- by the time `expr.name` is
        # actually referenced, the variable(s) it depends on are bound.
        replay_expr = scope.get_replay_binding(expr)
        if replay_expr is not None:
            return lower_expr(replay_expr, scope, builder)
        raise _UnboundScopeVariable(
            f"Variable {expr.name!r} (dtype={expr.dtype}) is not bound in the lowering scope. "
            f"Thread/block index vars must be bound by _lower_thread_extent before use."
        )

    # BufferLoad: Load op (returns tile from buffer)
    if isinstance(expr, _tirx.BufferLoad):
        return _lower_buffer_load_scalar(expr, scope, builder)

    # FMA pattern: detect before generic Add/Sub handling
    fma_val = _try_lower_fma(expr, scope, builder)
    if fma_val is not None:
        return fma_val

    # Binary arithmetic / comparison / logical
    binary_fn = _get_binary_op_fn().get(type(expr))
    if binary_fn is not None:
        lhs = lower_expr(expr.a, scope, builder)
        rhs = lower_expr(expr.b, scope, builder)
        # uint signedness: "max"/"min"/comparisons need the
        # OPERAND's signedness (expr.dtype is "bool" for comparisons).
        unsigned = str(getattr(expr.a, "dtype", "")).startswith("uint")
        return _make_elementwise(binary_fn, (lhs, rhs), expr, builder, unsigned=unsigned)

    # Logical Not
    if isinstance(expr, _tirx.Not):
        operand = lower_expr(expr.a, scope, builder)
        return _make_elementwise("not", (operand,), expr, builder)

    # Cast
    if isinstance(expr, _tirx.Cast):
        src_val = lower_expr(expr.value, scope, builder)
        tgt_dtype_str = _canonical_dtype_str(expr.dtype)
        src_dtype_str = _canonical_dtype_str(getattr(expr.value, "dtype", ""))
        result_ty = replace(_tir_dtype_to_tile_type(tgt_dtype_str), shape=src_val.type.shape)
        cast_op = builder.create(
            Cast(src=src_val, dtype=tgt_dtype_str, src_dtype=src_dtype_str),
            result_types=(result_ty,),
        )
        return cast_op.results[0]

    # Select
    if isinstance(expr, _tirx.Select):
        cond_val = lower_expr(expr.condition, scope, builder)
        true_val = lower_expr(expr.true_value, scope, builder)
        false_val = lower_expr(expr.false_value, scope, builder)
        return _make_select(cond_val, true_val, false_val, expr, builder)

    # Call (math functions, bitwise, if_then_else, ...)
    if isinstance(expr, _tirx.Call):
        return _lower_call_scalar(expr, scope, builder)

    # Catch-all: raise loudly so unsupported exprs are diagnosable
    raise _UnsupportedTileIRNode(
        f"Expression type {type(expr).__name__!r} is not yet lowered in TileIR. Add a handler in lower_expr for this TIR node type."
    )
