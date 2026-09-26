"""TileIR compute ops (GEMM, reductions, scans, dp4a)."""

from __future__ import annotations

import dataclasses
import re
from typing import Any

from tilelang.tileir.emission_utils import (
    _as_tile,
    _broadcast_ptr,
    _cast_tile,
    _dtype_from_mlir_type,
    _ensure_token,
    _load_buffer_tile as _load_tile,
    _mlir_element_type,
    _reshape_tile_to,
    _squeeze_shape,
    _store_buffer_tile as _store_tile,
)
from tilelang.tileir.errors import TileIRLoweringError, _UnsupportedTileIRNode
from tilelang.tileir.ir.ops._base import (
    Effect,
    TileOp,
    attribute,
    buffer_operand,
)


def _emit_gemm_impl(op: Any, ctx: Any) -> None:
    """Emit the MMA computation shared by ``Gemm`` and ``Tcgen05Gemm``."""
    ct = ctx.ct
    ir = ctx.ir
    loc = ctx.loc

    def _load_buffer_tile(buf_val: Any) -> Any:
        return _load_tile(ctx, buf_val, loc)

    def _store_buffer_tile(buf_val: Any, tile: Any) -> None:
        _store_tile(ctx, buf_val, tile, loc)

    # MLIR integer types are signless, so lowering carries operand signedness
    # explicitly on the operation.
    def _mma_signedness(is_unsigned: bool) -> Any:
        return ct.Signedness.UNSIGNED if is_unsigned else ct.Signedness.SIGNED

    lhs_tile = _load_buffer_tile(op.lhs)
    rhs_tile = _load_buffer_tile(op.rhs)
    acc_tile = _load_buffer_tile(op.acc)

    # MMA consumes two-dimensional operands. Staged shared-memory operands may
    # carry leading dimensions that must be removed or sliced first.
    def _ensure_2d(tile: Any) -> Any:
        shape = list(tile.tile_type.shape)
        if len(shape) <= 2:
            return tile
        squeezed = _squeeze_shape(shape)
        if squeezed != shape:
            return _reshape_tile_to(ct, tile, squeezed, loc)
        # Non-singleton leading dimensions represent stages; select stage zero.
        target_2d = shape[-2:]
        ir = ctx.ir
        i32 = ir.IntegerType.get_signless(32)
        i32_scalar_ty = ct.TileType.get([], i32)
        indices = [ct.constant(0, tile_type=i32_scalar_ty, loc=loc) for _ in shape]
        extract_shape = [1] * (len(shape) - 2) + target_2d
        elem_ty = tile.element_type
        extract_ty = ct.TileType.get(extract_shape, elem_ty)
        slice_tile = ct.extract(extract_ty, tile, indices, loc=loc)
        return _reshape_tile_to(ct, slice_tile, target_2d, loc)

    lhs_tile = _ensure_2d(lhs_tile)
    rhs_tile = _ensure_2d(rhs_tile)

    acc_elem_ty = _mlir_element_type(ctx, op.acc.type)
    acc_shape = list(op.acc.type.shape)

    clear = op.clear
    if isinstance(clear, bool):
        if clear:
            zero_ty = ct.TileType.get(acc_shape, acc_elem_ty)
            acc_tile = ct.constant(0, tile_type=zero_ty, loc=loc)
        # Otherwise retain the loaded accumulator.
    else:
        cond_raw = ctx.lookup(clear)
        cond = _as_tile(ctx, cond_raw)
        zero_ty = ct.TileType.get(acc_shape, acc_elem_ty)
        zero = ct.constant(0, tile_type=zero_ty, loc=loc)
        loaded_acc = acc_tile
        acc_tile = ct.if_generate(
            cond,
            lambda: zero,
            lambda: loaded_acc,
            return_types=[acc_tile],
            loc=loc,
        )

    if op.trans_a:
        lhs_tile = ct.permute(lhs_tile, [1, 0], loc=loc)
    if op.trans_b:
        rhs_tile = ct.permute(rhs_tile, [1, 0], loc=loc)

    # ``ct.ftof`` expects the CUDA Tile IR element wrapper, not an MLIR type.
    f32_ty = ir.F32Type.get()
    if lhs_tile.element_type == f32_ty and rhs_tile.element_type == f32_ty and acc_tile.element_type == f32_ty:
        lhs_tile = ct.ftof(ct.TFloat32, lhs_tile, loc=loc)
        rhs_tile = ct.ftof(ct.TFloat32, rhs_tile, loc=loc)

    # Optionally expose logical N as the hardware M dimension. The
    # architecture-aware pass owns this decision; emission only realizes the
    # explicit, semantics-preserving algebraic identity.
    swap_ab = bool(getattr(op, "swap_ab", False))
    lhs_unsigned = op.lhs_unsigned
    rhs_unsigned = op.rhs_unsigned
    if swap_ab:
        lhs_tile, rhs_tile = (
            ct.permute(rhs_tile, [1, 0], loc=loc),
            ct.permute(lhs_tile, [1, 0], loc=loc),
        )
        acc_tile = ct.permute(acc_tile, [1, 0], loc=loc)
        lhs_unsigned, rhs_unsigned = rhs_unsigned, lhs_unsigned

    result = ct.mma(
        lhs_tile,
        rhs_tile,
        acc_tile,
        signedness_lhs=_mma_signedness(lhs_unsigned),
        signedness_rhs=_mma_signedness(rhs_unsigned),
        loc=loc,
    )
    if swap_ab:
        result = ct.permute(result, [1, 0], loc=loc)

    _store_buffer_tile(op.acc, result)


@dataclasses.dataclass(eq=False)
class Gemm(TileOp, opcode="gemm", effect=Effect.READWRITE):
    """Matrix-multiply accumulate (tl.tileop.gemm).

    ``trans_a`` / ``trans_b`` control transposition of the LHS / RHS inputs.
    ``clear`` specifies whether to zero-initialize the accumulator before MMA.

    Fields
    ------
    lhs, rhs, acc       — GLOBAL buffer Value objects.
    trans_a             — transpose LHS before MMA.
    trans_b             — transpose RHS before MMA.
    clear               — True / False (static) or an SSA Value (dynamic bool).
                          True:  zero-initialise ACC before MMA.
                          False: load ACC from its buffer and accumulate.
                          Value: runtime predicate — if_generate selects zero vs loaded.
    lhs_unsigned        — True if the LHS operand is an unsigned integer type.
                          Defaults to False (signed).  The TileIR type system maps
                          uint8/uint16/uint32 to the signless int8/int16/int32 MLIR
                          type, so signedness must be carried separately.
    rhs_unsigned        — True if the RHS operand is an unsigned integer type.
    swap_ab             — emit the equivalent ``(rhs.T @ lhs.T + acc.T).T``
                          orientation selected by an optimization pass.
    """

    lhs: Any = buffer_operand(effect=Effect.READ)
    rhs: Any = buffer_operand(effect=Effect.READ)
    acc: Any = buffer_operand(effect=Effect.READWRITE)
    trans_a: bool = attribute(default=False)
    trans_b: bool = attribute(default=False)
    clear: Any = attribute(default=False)
    lhs_unsigned: bool = attribute(default=False)
    rhs_unsigned: bool = attribute(default=False)
    swap_ab: bool = attribute(default=False)

    def emit_mlir(self, ctx: Any) -> None:
        """Emit a CUDA Tile IR MMA and store the updated accumulator."""
        _emit_gemm_impl(self, ctx)
        return None


@dataclasses.dataclass(eq=False)
class Tcgen05Gemm(TileOp, opcode="tcgen05_gemm", effect=Effect.READWRITE):
    """Tcgen05 MMA operation (tl.tileop.tcgen05_gemm).

    Same semantics as ``Gemm`` but targets the tcgen05 hardware generation.

    Fields
    ------
    lhs, rhs, acc       — GLOBAL buffer Value objects.
    trans_a             — transpose LHS before MMA.
    trans_b             — transpose RHS before MMA.
    clear               — True/False or SSA Value (dynamic bool).
    lhs_unsigned        — True if the LHS operand is an unsigned integer type.
    rhs_unsigned        — True if the RHS operand is an unsigned integer type.
    """

    lhs: Any = buffer_operand(effect=Effect.READ)
    rhs: Any = buffer_operand(effect=Effect.READ)
    acc: Any = buffer_operand(effect=Effect.READWRITE)
    trans_a: bool = attribute(default=False)
    trans_b: bool = attribute(default=False)
    clear: Any = attribute(default=False)
    lhs_unsigned: bool = attribute(default=False)
    rhs_unsigned: bool = attribute(default=False)

    def emit_mlir(self, ctx: Any) -> None:
        """Lower Tcgen05Gemm to MLIR.

        Delegates to the shared ``_emit_gemm_impl`` helper.  Tcgen05 is a
        hardware-generation marker (tcgen05 SM100+); the CUDA Tile IR dialect
        uses the same ``cuda_tile.mma`` op and selects the tcgen05 hardware path
        through type-based lowering downstream.
        """
        _emit_gemm_impl(self, ctx)
        return None


def _emit_gemm_scaled_impl(op: Any, ctx: Any) -> None:
    """Emit block-scaled MMA through ``cuda_tile.mmaf_scaled``.

    The operation requires sm_100+. Integer scale tiles carry E8M0 bit
    patterns and retain their logical orientation when operands transpose.
    """
    ct = ctx.ct
    ir = ctx.ir
    loc = ctx.loc

    def _load_buffer_tile(buf_val: Any) -> Any:
        return _load_tile(ctx, buf_val, loc)

    def _store_buffer_tile(buf_val: Any, tile: Any) -> None:
        _store_tile(ctx, buf_val, tile, loc)

    # Scaled MMA requires the Blackwell instruction set.
    if ctx.arch is not None:
        match = re.fullmatch(r"sm_(\d+)[a-z]*", str(ctx.arch).strip())
        if match is not None and int(match.group(1)) < 100:
            raise TileIRLoweringError(f"T.tcgen05_gemm_blockscaled requires sm_100+; target arch is {ctx.arch}.")

    lhs_tile = _load_buffer_tile(op.lhs)
    rhs_tile = _load_buffer_tile(op.rhs)
    acc_tile = _load_buffer_tile(op.acc)
    lhs_scale_tile = _load_buffer_tile(op.lhs_scale)
    rhs_scale_tile = _load_buffer_tile(op.rhs_scale)

    # MMA operands and scales are two-dimensional.
    def _ensure_2d(tile: Any) -> Any:
        shape = list(tile.tile_type.shape)
        if len(shape) <= 2:
            return tile
        squeezed = _squeeze_shape(shape)
        if squeezed != shape:
            return _reshape_tile_to(ct, tile, squeezed, loc)
        target_2d = shape[-2:]
        i32 = ir.IntegerType.get_signless(32)
        i32_scalar_ty = ct.TileType.get([], i32)
        indices = [ct.constant(0, tile_type=i32_scalar_ty, loc=loc) for _ in shape]
        extract_shape = [1] * (len(shape) - 2) + target_2d
        elem_ty = tile.element_type
        extract_ty = ct.TileType.get(extract_shape, elem_ty)
        slice_tile = ct.extract(extract_ty, tile, indices, loc=loc)
        return _reshape_tile_to(ct, slice_tile, target_2d, loc)

    lhs_tile = _ensure_2d(lhs_tile)
    rhs_tile = _ensure_2d(rhs_tile)
    acc_tile = _ensure_2d(acc_tile)
    lhs_scale_tile = _ensure_2d(lhs_scale_tile)
    rhs_scale_tile = _ensure_2d(rhs_scale_tile)

    # Integer scale tiles carry E8M0 bit patterns.
    def _as_scale_tile(tile: Any) -> Any:
        elem_ty = tile.element_type
        if isinstance(elem_ty, ir.IntegerType) and elem_ty.width == 8:
            return ct.bitcast(ct.Float8E8M0FNU, tile, loc=loc)
        return tile

    lhs_scale_tile = _as_scale_tile(lhs_scale_tile)
    rhs_scale_tile = _as_scale_tile(rhs_scale_tile)

    acc_elem_ty = _mlir_element_type(ctx, op.acc.type)
    acc_shape = list(op.acc.type.shape)

    clear = op.clear
    if isinstance(clear, bool):
        if clear:
            zero_ty = ct.TileType.get(acc_shape, acc_elem_ty)
            acc_tile = ct.constant(0, tile_type=zero_ty, loc=loc)
        # Otherwise retain the loaded accumulator.
    else:
        cond_raw = ctx.lookup(clear)
        cond = _as_tile(ctx, cond_raw)
        zero_ty = ct.TileType.get(acc_shape, acc_elem_ty)
        zero = ct.constant(0, tile_type=zero_ty, loc=loc)
        loaded_acc = acc_tile
        acc_tile = ct.if_generate(
            cond,
            lambda: zero,
            lambda: loaded_acc,
            return_types=[acc_tile],
            loc=loc,
        )

    # Scale tiles follow logical axes and are not transposed with the operands.
    if op.trans_a:
        lhs_tile = ct.permute(lhs_tile, [1, 0], loc=loc)
    if op.trans_b:
        rhs_tile = ct.permute(rhs_tile, [1, 0], loc=loc)

    try:
        result = ct.mmaf_scaled(lhs_tile, rhs_tile, acc_tile, lhs_scale_tile, rhs_scale_tile, loc=loc)
    except TypeError as exc:
        raise TileIRLoweringError(str(exc)) from exc

    _store_buffer_tile(op.acc, result)


@dataclasses.dataclass(eq=False)
class GemmScaled(TileOp, opcode="gemm_scaled", effect=Effect.READWRITE):
    """Tcgen05 block-scaled matrix-multiply-accumulate.

    Computes ``acc += dequant(lhs, lhs_scale) @ dequant(rhs, rhs_scale)``.
    Lowers to ``cuda_tile.mmaf_scaled`` (sm_100+ only — see
    ``_emit_gemm_scaled_impl``'s sm_100 gate).

    Fields
    ------
    lhs, rhs, acc         — GLOBAL/SHARED buffer Value objects: MMA operands
                          and accumulator.  lhs/rhs must be fp8 (e4m3/e5m2)
                          or fp4 (e2m1); acc must be float32.
    lhs_scale, rhs_scale  — GLOBAL/SHARED buffer Value objects holding
                          per-block scale factors, logical shape (M, K/V) and
                          (K/V, N) respectively (V is the caller's
                          ``sf_a_granularity_k``/``sf_b_granularity_k``).  May
                          be declared uint8 in TileLang (torch 2.6 has no
                          e8m0 dtype) — reinterpreted to e8m0 bit patterns at
                          emit time.  May instead be declared e4m3, in which
                          case a negative scale value is undefined behavior
                          (the underlying CUDA Tile IR op requires
                          non-negative scales) rather than a well-defined
                          negative scale factor.
    trans_a               — transpose LHS before MMA.
    trans_b               — transpose RHS before MMA.
    clear                 — True / False (static) or an SSA Value (dynamic bool).
                          True:  zero-initialise ACC before MMA.
                          False: load ACC from its buffer and accumulate.
                          Value: runtime predicate — if_generate selects zero vs loaded.
    """

    lhs: Any = buffer_operand(effect=Effect.READ)
    rhs: Any = buffer_operand(effect=Effect.READ)
    acc: Any = buffer_operand(effect=Effect.READWRITE)
    lhs_scale: Any = buffer_operand(effect=Effect.READ)
    rhs_scale: Any = buffer_operand(effect=Effect.READ)
    trans_a: bool = attribute(default=False)
    trans_b: bool = attribute(default=False)
    clear: Any = attribute(default=False)

    def emit_mlir(self, ctx: Any) -> None:
        """Emit a scaled MMA and store the updated accumulator."""
        _emit_gemm_scaled_impl(self, ctx)
        return None


@dataclasses.dataclass(eq=False)
class Reduce(TileOp, opcode="reduce", effect=Effect.READWRITE):
    """Tile-level reduction along a given axis (tl.tileop.reduce).

    ``op``   — reduction kind string: "sum", "max", "min", "abssum",
               "absmax", "bitand", "bitor", "bitxor".
    ``axis`` — dimension to reduce along.
    ``clear`` — whether to zero-initialise the destination before reducing.
    ``src_unsigned`` — True if ``src``'s ORIGINAL (pre-alias-collapse)
        dtype is an unsigned integer type (mirrors ``Gemm.lhs_unsigned``):
        the TileIR type registry
        alias-collapses uint dtypes to signless ints (uint32 -> int32), so
        ``self.src.type.dtype.name`` is always "int32" for a uint32 buffer
        and can never be used to detect unsignedness here. The lowering
        handler (``_lower_reduce``) threads the true unsignedness, read
        from the pre-collapse ``SemanticBuffer.dtype`` string via
        ``scope.lookup_raw_dtype``, through this field instead. Only "max"
        / "min" need it (sum/bitand/bitor/bitxor are signedness-agnostic in
        two's complement).
    """

    src: Any = buffer_operand(effect=Effect.READ)
    # clear=False merges the reduction result with the existing destination.
    # Use the conservative role for both clear modes; per-instance effects are
    # intentionally not inferred by the analysis passes.
    dst: Any = buffer_operand(effect=Effect.READWRITE)
    op: str = attribute()
    axis: int = attribute()
    clear: bool = attribute(default=True)
    src_unsigned: bool = attribute(default=False)

    def emit_mlir(self, ctx: Any) -> None:
        """Reduce the source tile and store or merge the result."""
        ct = ctx.ct
        loc = ctx.loc

        def _load_buffer_tile(buf_val: Any) -> Any:
            return _load_tile(ctx, buf_val, loc)

        def _store_buffer_tile(buf_val: Any, tile: Any) -> None:
            _store_tile(ctx, buf_val, tile, loc)

        # Helpers for dtype and identity
        def _is_float_dtype(name: str) -> bool:
            return name in {
                "float16",
                "bfloat16",
                "float32",
                "float64",
                "tf32",
                "float8_e4m3fn",
                "float8_e5m2",
                "float8_e8m0fnu",
                "float4_e2m1fn",
            }

        def _reduce_identity(kind: str, dtype_name: str, unsigned: bool):
            if kind == "sum":
                return 0.0 if _is_float_dtype(dtype_name) else 0
            if kind == "bitand":
                return -1
            if kind in {"bitor", "bitxor"}:
                return 0
            if kind in {"max", "min"}:
                if _is_float_dtype(dtype_name):
                    return float("-inf") if kind == "max" else float("inf")
                # Dtype-width-correct, unsigned-aware integer identity: the
                # identity must not beat any real element, so "max"'s
                # identity is the smallest representable value (signed
                # INT_MIN, or 0 for unsigned); "min"'s is the largest
                # (signed INT_MAX, or 2**bits-1 for unsigned).
                bits = _SCAN_INT_BIT_WIDTHS.get(dtype_name, 32)
                if kind == "max":
                    value = 0 if unsigned else -(2 ** (bits - 1))
                else:
                    value = (2**bits - 1) if unsigned else (2 ** (bits - 1) - 1)
                # Normalize unsigned magnitudes to the signed value with the
                # same bit pattern expected by signless MLIR integers.
                return _to_signed_twos_complement(value, bits)
            raise _UnsupportedTileIRNode(f"TileIR Reduce: unknown reduce kind '{kind}'.")

        src_tile = _load_buffer_tile(self.src)

        reduce_kind = self.op
        src_dtype_name = self.src.type.dtype.name
        if reduce_kind in {"abssum", "absmax"}:
            # CUDA Tile IR exposes type-specific absolute-value builders.
            if _is_float_dtype(src_dtype_name):
                src_tile = ct.absf(src_tile, loc=loc)
            else:
                src_tile = ct.absi(src_tile, loc=loc)

        # The identity uses the source dtype because it is passed directly to
        # ``ct.reduce`` before any destination cast.
        dst_dtype_name = self.dst.type.dtype.name
        base_kind = "max" if reduce_kind == "absmax" else ("sum" if reduce_kind == "abssum" else reduce_kind)
        identity = _reduce_identity(base_kind, src_dtype_name, self.src_unsigned)

        if base_kind in {"bitand", "bitor", "bitxor"} and _is_float_dtype(src_dtype_name):
            raise _UnsupportedTileIRNode(f"TileIR Reduce: kind '{base_kind}' requires an integer dtype, got '{src_dtype_name}'.")

        # Signedness is ignored by the floating-point max/min builders.
        max_min_signedness = ct.Signedness.UNSIGNED if self.src_unsigned else ct.Signedness.SIGNED
        if base_kind in {"max"}:

            def combine(lhs, rhs):
                return ct.max(lhs, rhs, signedness=max_min_signedness, loc=loc)
        elif base_kind == "min":

            def combine(lhs, rhs):
                return ct.min(lhs, rhs, signedness=max_min_signedness, loc=loc)
        elif base_kind == "sum":

            def combine(lhs, rhs):
                return ct.add(lhs, rhs, loc=loc)
        elif base_kind == "bitand":

            def combine(lhs, rhs):
                return ct.andi(lhs, rhs, loc=loc)
        elif base_kind == "bitor":

            def combine(lhs, rhs):
                return ct.ori(lhs, rhs, loc=loc)
        elif base_kind == "bitxor":

            def combine(lhs, rhs):
                return ct.xori(lhs, rhs, loc=loc)
        else:
            raise _UnsupportedTileIRNode(f"TileIR Reduce: unsupported reduce kind '{reduce_kind}'.")

        result = ct.reduce(src_tile, self.axis, identity, combine, loc=loc)

        result_dtype = _dtype_from_mlir_type(ctx.ir, result.element_type)
        if result_dtype != dst_dtype_name:
            result = _cast_tile(ct, ctx.ir, result, result_dtype, dst_dtype_name, loc=loc)

        if not self.clear:
            prev = _load_buffer_tile(self.dst)
            # Reconcile scalar-like ranks: reducing a (N,) fragment into a
            # (1,) accumulator yields a rank-0 result vs a rank-1 prev tile.
            if list(prev.tile_type.shape) != list(result.tile_type.shape):
                prev = _reshape_tile_to(ct, prev, list(result.tile_type.shape), loc)
            result = combine(prev, result)

        # Preserve the destination rank for subsequent indexed reads.
        _dst_shape = list(self.dst.type.shape)
        if ctx.is_tile_buffer(self.dst) and list(result.tile_type.shape) != _dst_shape:
            _numel = 1
            for _d in list(result.tile_type.shape):
                _numel *= _d
            _dst_numel = 1
            for _d in _dst_shape:
                _dst_numel *= _d
            if _numel == _dst_numel:
                result = _reshape_tile_to(ct, result, _dst_shape, loc)
        _store_buffer_tile(self.dst, result)
        return None


# Integer bit-widths used to compute a dtype-correct INT_MIN/INT_MAX identity
# for the "max"/"min" Reduce and "max" scan kinds (see Reduce.emit_mlir /
# Cumsum.emit_mlir).  Unlisted / unknown integer dtypes fall back to 32-bit
# width.
_SCAN_INT_BIT_WIDTHS = {
    "int8": 8,
    "uint8": 8,
    "int16": 16,
    "uint16": 16,
    "int32": 32,
    "uint32": 32,
    "int64": 64,
    "uint64": 64,
}


def _to_signed_twos_complement(value: int, bits: int) -> int:
    """Return the signed integer with *value*'s ``bits``-wide bit pattern."""
    value &= (1 << bits) - 1
    if value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


@dataclasses.dataclass(eq=False)
class Cumsum(TileOp, opcode="cumsum", effect=Effect.READWRITE):
    """Scan (prefix-sum / cumulative-max) along an axis.

    Backs every TileIR scan intrinsic despite the class name (the ``Cumsum``
    name is shared across the lowering pipeline; ``kind`` distinguishes the
    combinator):

    - ``tl.tileop.cumsum``  → ``kind="sum"``  (default), identity 0 / 0.0, ct.add
    - ``tl.tileop.cummax``  → ``kind="max"``, identity -inf / dtype INT_MIN, ct.max

    ``axis``    — dimension to scan along.
    ``reverse`` — if True, scan in reverse order.
    ``kind``    — scan combinator, one of "sum" / "max".
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)
    axis: int = attribute()
    reverse: bool = attribute(default=False)
    kind: str = attribute(default="sum")
    # Mirrors Gemm.lhs_unsigned: only the
    # "max" kind needs this (add is signedness-agnostic in two's
    # complement). `self.dst.type.dtype.name.startswith("uint")` can never
    # fire since the TileIR type system maps unsigned integers to signless
    # MLIR types.
    src_unsigned: bool = attribute(default=False)

    def emit_mlir(self, ctx: Any) -> None:
        """Scan the source tile and store the result."""
        ct = ctx.ct
        loc = ctx.loc

        def _load_buffer_tile(buf_val: Any) -> Any:
            return _load_tile(ctx, buf_val, loc)

        def _store_buffer_tile(buf_val: Any, tile: Any) -> None:
            _store_tile(ctx, buf_val, tile, loc)

        src_tile = _load_buffer_tile(self.src)

        dst_dtype_name = self.dst.type.dtype.name
        _float_names = {
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "tf32",
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e8m0fnu",
            "float4_e2m1fn",
        }
        is_float = dst_dtype_name in _float_names

        if self.kind == "sum":
            identity = 0.0 if is_float else 0

            def scan_body(lhs, rhs):
                return ct.add(lhs, rhs, loc=loc)
        elif self.kind == "max":
            if is_float:
                identity = float("-inf")
            else:
                bits = _SCAN_INT_BIT_WIDTHS.get(dst_dtype_name, 32)
                identity = 0 if self.src_unsigned else -(2 ** (bits - 1))
            # Signedness is ignored by the floating-point max builder.
            max_signedness = ct.Signedness.UNSIGNED if self.src_unsigned else ct.Signedness.SIGNED

            def scan_body(lhs, rhs):
                return ct.max(lhs, rhs, signedness=max_signedness, loc=loc)
        else:
            raise TileIRLoweringError(f"TileIR Cumsum/scan: unsupported scan kind '{self.kind}'.")

        result = ct.scan(src_tile, self.axis, self.reverse, identity, scan_body, loc=loc)

        _store_buffer_tile(self.dst, result)
        return None


@dataclasses.dataclass(eq=False)
class ThreadAllreduce(TileOp, opcode="thread_allreduce", effect=Effect.NONE):
    """SIMT cross-thread allreduce (tir.tvm_thread_allreduce).

    Not supported in the tile execution model; included for completeness and
    to produce a meaningful error during emission.
    """

    def emit_mlir(self, ctx: Any) -> None:
        """Reject SIMT allreduce, which has no tile-level equivalent."""
        raise _UnsupportedTileIRNode(
            "TileIR backend does not support `tvm_thread_allreduce` (SIMT cross-thread "
            "reduction): the tile execution model does not expose per-thread lanes. "
            "Express the reduction at tile level with `T.reduce`, or compile this "
            "kernel on the CUDA backend."
        )


@dataclasses.dataclass(eq=False)
class Dp4a(TileOp, opcode="dp4a", effect=Effect.READWRITE):
    """DP4A dot-product-and-accumulate (int8 × int8 → int32).

    Computes: acc += sum(lhs[i] * rhs[i]) for i in range(4),
    where lhs/rhs are [4 x i8] tiles and acc is a scalar int32.

    Buffer shapes: lhs [4] int8, rhs [4] int8, acc [] int32 (scalar).
    """

    lhs: Any = buffer_operand(effect=Effect.READ)
    rhs: Any = buffer_operand(effect=Effect.READ)
    acc: Any = buffer_operand(effect=Effect.READWRITE)

    def emit_mlir(self, ctx: Any) -> None:
        """Lower Dp4a to a four-element multiply-accumulate sequence."""

        ct = ctx.ct
        ir = ctx.ir
        loc = ctx.loc

        lhs_dtype = self.lhs.type.dtype.name
        rhs_dtype = self.rhs.type.dtype.name
        acc_dtype = self.acc.type.dtype.name
        if lhs_dtype != "int8" or rhs_dtype != "int8" or acc_dtype != "int32":
            raise _UnsupportedTileIRNode(f"TileIR DP4A expects int8, int8, int32 buffers; got {lhs_dtype}, {rhs_dtype}, {acc_dtype}.")
        lhs_shape = list(self.lhs.type.shape)
        rhs_shape = list(self.rhs.type.shape)
        if lhs_shape != [4] or rhs_shape != [4]:
            raise _UnsupportedTileIRNode(f"TileIR DP4A expects lhs shape [4] and rhs shape [4], got {lhs_shape} and {rhs_shape}.")

        i8_ty = ir.IntegerType.get_signless(8)
        i32_ty = ir.IntegerType.get_signless(32)
        tile_4_i8 = ct.TileType.get([4], i8_ty)
        tile_1_i8 = ct.TileType.get([1], i8_ty)
        tile_scalar_i32 = ct.TileType.get([], i32_ty)

        def _load_tile(buf_val, shape, tile_type):
            """Load a tile from either a REGISTER tile-map or a GLOBAL ptr."""
            if ctx.is_tile_buffer(buf_val):
                return _as_tile(ctx, ctx.get_tile(buf_val))
            tok = _ensure_token(ctx, buf_val)
            ptr_shaped = _broadcast_ptr(ct, ctx.get_buffer_info(buf_val).ptr, shape, loc=loc)
            tile, out_tok = ct.load_ptr_tko(
                result=tile_type,
                source=ptr_shaped,
                input_token=tok,
                return_token=True,
                loc=loc,
            )
            ctx._set_token(buf_val, out_tok)
            return tile

        acc_tile = _load_tile(self.acc, [], tile_scalar_i32)
        # REGISTER tile-map may return a shaped tile (e.g. [1] x i32) if the
        # buffer was declared as alloc_local((1,), int32); reshape to scalar.
        acc_tile = _as_tile(ctx, acc_tile)
        if list(acc_tile.tile_type.shape) != []:
            acc_tile = ct.reshape([], acc_tile, loc=loc)
        lhs_tile = _load_tile(self.lhs, [4], tile_4_i8)
        rhs_tile = _load_tile(self.rhs, [4], tile_4_i8)

        result = acc_tile
        for offset in range(4):
            idx = ct.constant(offset, tile_type=tile_scalar_i32, loc=loc)
            lhs_e = ct.extract(tile_1_i8, lhs_tile, [idx], loc=loc)
            rhs_e = ct.extract(tile_1_i8, rhs_tile, [idx], loc=loc)
            lhs_i32 = ct.exti(ct.Int32, lhs_e, signedness=ct.Signedness.SIGNED, loc=loc)
            rhs_i32 = ct.exti(ct.Int32, rhs_e, signedness=ct.Signedness.SIGNED, loc=loc)
            prod = ct.mul(lhs_i32, rhs_i32, loc=loc)
            prod_scalar = ct.reshape([], prod, loc=loc)
            result = ct.add(result, prod_scalar, loc=loc)

        if ctx.is_tile_buffer(self.acc):
            # For REGISTER alloc_local: reshape scalar result to [1] (acc shape)
            # to match the declared tile shape of the REGISTER buffer.
            acc_shape = list(self.acc.type.shape)
            if acc_shape and acc_shape != []:
                result = ct.reshape(acc_shape, result, loc=loc)
            ctx.set_tile(self.acc, result)
        else:
            acc_info = ctx.get_buffer_info(self.acc)
            dst_acc_tok = _ensure_token(ctx, self.acc)
            acc_ptr_shaped2 = _broadcast_ptr(ct, acc_info.ptr, [], loc=loc)
            out_tok = ct.store_ptr_tko(
                destination=acc_ptr_shaped2,
                value=result,
                input_token=dst_acc_tok,
                loc=loc,
            )
            ctx._set_token(self.acc, out_tok)
        return None
