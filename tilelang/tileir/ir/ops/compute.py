"""TileIR compute ops (GEMM, reductions, scans, dp4a)."""

from __future__ import annotations

import dataclasses
from typing import Any

from tilelang.tileir.ir.ops._base import (
    Effect,
    TileOp,
    attribute,
    buffer_operand,
)


def _emit_gemm_impl(op: Any, ctx: Any) -> None:
    """Emit the MMA computation shared by Gemm and Tcgen05Gemm.

    Steps
    -----
    1. Load LHS, RHS, and ACC tiles from their GLOBAL buffers via
       ``load_view_tko`` / ``load_ptr_tko`` (mirrors _load_tile_region).
    2. If ``op.clear`` is statically ``True``: replace ACC with a zero constant.
       If ``op.clear`` is statically ``False``: keep the loaded ACC.
       If ``op.clear`` is a Value: use ``ct.if_generate`` to select dynamically.
    3. Optionally permute LHS/RHS for transposition.
    4. For f32×f32→f32: down-cast both operands to tf32 (TileLang tensor-core
       convention, mirrors ``_mma_operands_for_tilelang_contract`` in tile_ops.py).
    5. If ``swap_ab`` is set, transpose/swap the MMA operands and accumulator.
    6. Call ``ct.mma(lhs, rhs, acc, …)`` with per-operand signedness.
    7. Transpose a swapped result back and store it into the ACC buffer.

    Token seam: load tokens for LHS, RHS, ACC are consumed/updated via
    ``_ensure_token`` / ``ctx._set_token``.  The store token for ACC is
    updated after the result is written back.
    """
    from tilelang.tileir.emission_utils import (
        _mlir_element_type,
        _as_tile,
        _load_buffer_tile as _lbt,
        _store_buffer_tile as _sbt,
        _squeeze_shape,
        _reshape_tile_to,
    )

    ct = ctx.ct
    ir = ctx.ir
    loc = ctx.loc

    # Bind module-level helpers to local closures with the captured loc.
    # (I1) Load/store helpers are defined in emission_utils.py and shared
    # across _emit_gemm_impl, Reduce.emit_mlir, and Cumsum.emit_mlir.
    def _load_buffer_tile(buf_val: Any) -> Any:
        return _lbt(ctx, buf_val, loc)

    def _store_buffer_tile(buf_val: Any, tile: Any) -> None:
        _sbt(ctx, buf_val, tile, loc)

    # Helper: compute signedness for integer mma operands
    # lhs_unsigned / rhs_unsigned MUST be set by the TIR→TileIR
    # lowering pass when it detects uint8/uint16/uint32 source dtypes.  The
    # TileType system maps unsigned integers to signless MLIR types (e.g.
    # uint8 → i8), so the unsigned-ness signal is lost at emit time.  If
    # lhs_unsigned / rhs_unsigned are not set correctly, the MMA will silently
    # use the wrong signedness for the dot-product accumulate.
    # The TileIR type system maps uint8/uint16/uint32 to signless int8/i16/i32
    # (see _INT_DTYPE_ROWS in types.py where "uint8" is an alias for "int8").
    # The signedness is therefore carried explicitly via the op's
    # lhs_unsigned / rhs_unsigned attributes, which the TIR→TileIR lowering
    # must set when it detects unsigned dtype strings in the source IR.
    def _mma_signedness(is_unsigned: bool) -> Any:
        return ct.Signedness.UNSIGNED if is_unsigned else ct.Signedness.SIGNED

    # Step 1: Load all three tiles
    lhs_tile = _load_buffer_tile(op.lhs)
    rhs_tile = _load_buffer_tile(op.rhs)
    acc_tile = _load_buffer_tile(op.acc)

    # Step 1b: Squeeze rank-N (N>2) LHS/RHS tiles to 2D for MMA.
    #
    # The correct element count is guaranteed by
    # sem_to_ir.py:_lower_copy which caps the GLOBAL→SHARED tile_shape
    # to the SHARED buffer's declared size when a dynamic region dim was
    # inflated to the full buffer dim by the _region_tile_shape fallback.
    #
    # When a warp-specialized kernel uses a staged shared buffer
    # (e.g. A_shared: (num_stages, block_M, block_K)) the _tile_map may
    # hold the full 3D tile.  MMA requires 2D operands.  Squeezing
    # size-1 leading dims (e.g. (2, 128, 64) with only (128, 64)
    # actually consumed) is handled by reshaping to the squeezed shape.
    # This mirrors the _squeeze_shape helper used in Copy/Load for 4D→2D.
    def _ensure_2d(tile: Any) -> Any:
        shape = list(tile.tile_type.shape)
        if len(shape) <= 2:
            return tile
        squeezed = _squeeze_shape(shape)
        if squeezed != shape:
            # Has singleton dims (e.g. [1, M, K] → [M, K]): reshape is valid
            # (same total element count since we only remove size-1 dims).
            return _reshape_tile_to(ct, tile, squeezed, loc)
        # No singleton dims: e.g. [num_stages, M, K] → take one stage via extract.
        # Use ct.extract to get stage 0 as tile<1x...x1xMxK>, then reshape to [M,K].
        # This handles ping-pong/staged shared buffers in warp-specialized kernels.
        # ct.reshape([M,K], tile<2xMxK>) is invalid (element count
        # mismatch 2*M*K ≠ M*K); extract stage 0 first to get a 1-element leading dim.
        # ct.extract requires indices for ALL dims of the source tile, so for a 3D
        # source (N, M, K) we pass [0, 0, 0] → result tile<1xMxK> → reshape to [M,K].
        target_2d = shape[-2:]
        ir = ctx.ir
        i32 = ir.IntegerType.get_signless(32)
        i32_scalar_ty = ct.TileType.get([], i32)
        # Indices for all dims: all zeros (extract element at position [0,0,...,0])
        indices = [ct.constant(0, tile_type=i32_scalar_ty, loc=loc) for _ in shape]
        # Result of extract: [1, ..., 1, M, K] — leading dims become 1
        extract_shape = [1] * (len(shape) - 2) + target_2d
        elem_ty = tile.element_type
        extract_ty = ct.TileType.get(extract_shape, elem_ty)
        slice_tile = ct.extract(extract_ty, tile, indices, loc=loc)
        # Reshape [1, ..., 1, M, K] → [M, K] (valid: 1*...*1*M*K = M*K)
        return _reshape_tile_to(ct, slice_tile, target_2d, loc)

    lhs_tile = _ensure_2d(lhs_tile)
    rhs_tile = _ensure_2d(rhs_tile)

    acc_elem_ty = _mlir_element_type(ctx, op.acc.type)
    acc_shape = list(op.acc.type.shape)

    # Step 2: Handle clear (zero-init or dynamic select)
    clear = op.clear
    if isinstance(clear, bool):
        if clear:
            zero_ty = ct.TileType.get(acc_shape, acc_elem_ty)
            acc_tile = ct.constant(0, tile_type=zero_ty, loc=loc)
        # else: keep the loaded acc_tile as-is
    else:
        # Dynamic clear: runtime bool Value — use if_generate to select
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

    # Step 3: Transpose (permute) if requested
    if op.trans_a:
        lhs_tile = ct.permute(lhs_tile, [1, 0], loc=loc)
    if op.trans_b:
        rhs_tile = ct.permute(rhs_tile, [1, 0], loc=loc)

    # Step 4: TF32 promotion for f32×f32→f32 (TileLang tensor-core convention)
    # ct.ftof expects a ct element-type class (e.g. ct.TFloat32), NOT a raw
    # ir.FloatTF32Type instance.  ct.TFloat32 is the element-type wrapper class
    # that cuda_tile registers for the tf32 scalar type (mirrors _element_wrapper
    # in lowering/core.py which maps "tf32" → "TFloat32").
    f32_ty = ir.F32Type.get()
    if lhs_tile.element_type == f32_ty and rhs_tile.element_type == f32_ty and acc_tile.element_type == f32_ty:
        lhs_tile = ct.ftof(ct.TFloat32, lhs_tile, loc=loc)
        rhs_tile = ct.ftof(ct.TFloat32, rhs_tile, loc=loc)

    # Step 5: Optionally expose logical N as the hardware M dimension.  The
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

    # Step 6: MMA
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

    # Step 7: Store result back to ACC buffer
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
        """Lower Gemm to MLIR: load LHS/RHS/ACC, optional zero-init, mma, store ACC.

        Steps
        -----
        1. Load LHS and RHS tiles from their GLOBAL buffers.
        2. Load ACC tile (used as the accumulator operand, or zeroed if clear).
        3. Optionally transpose (permute) LHS/RHS.
        4. Promote f32 operands to tf32 (TileLang convention for f32 tensor cores).
        5. Emit ``ct.mma(lhs, rhs, acc, …)`` to produce the result tile.
        6. Store the result tile back into the ACC GLOBAL buffer.

        Returns ``None`` (side-effect only; no SSA result is bound).
        """
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
        through type-based lowering downstream.  No divergence from Gemm at the
        TileIR → MLIR translation level.
        """
        _emit_gemm_impl(self, ctx)
        return None


def _emit_gemm_scaled_impl(op: Any, ctx: Any) -> None:
    """Emit the block-scaled MMA computation for GemmScaled via ``cuda_tile.mmaf_scaled``.

    Steps
    -----
    0. Gate on ``ctx.arch``: ``mmaf_scaled`` (the hardware scaled-MMA unit) is
       sm_100+ only.  Fail fast, before emitting any ops, if the target arch
       is older.
    1. Load LHS, RHS, ACC, LHS_SCALE, RHS_SCALE tiles from their GLOBAL/SHARED
       buffers via ``_load_buffer_tile`` (mirrors ``_emit_gemm_impl``).
    2. Squeeze rank-N (N>2) tiles to 2D — applied to all five tiles (unlike
       ``_emit_gemm_impl``, which only squeezes LHS/RHS; here ACC and the two
       scale tiles also need it since GemmScaled has no shared code path with
       plain Gemm).
    3. Scale dtype handling: TileLang may declare SFA/SFB as uint8 (torch 2.6
       has no e8m0 dtype), which the TileType system maps to signless i8. If a
       loaded scale tile's element type is 8-bit integer, bit-reinterpret it as
       e8m0 (``ct.Float8E8M0FNU``) via ``ct.bitcast`` — this is a
       reinterpret-the-bits operation, NOT a numeric cast, matching the
       ``tir.reinterpret`` -> ``ct.bitcast`` idiom used by ``Cast.emit_mlir``
       (see ``_emit_cast`` in emission_utils.py). If the scale tile is already
       e8m0 or e4m3 (a user allocated SFA/SFB directly in one of those float
       dtypes), pass it through unchanged.
    4. Clear handling: identical pattern to ``_emit_gemm_impl`` — static True
       zero-inits ACC, static False keeps the loaded ACC, and a dynamic Value
       selects between the two via ``ct.if_generate``.
    5. Optional transpose (permute) of LHS/RHS for trans_a/trans_b.  The scale
       tiles are NEVER permuted: the TileLang frontend (T.tcgen05_gemm_blockscaled)
       validates SFA/SFB shapes against the LOGICAL (M, K/V) / (K/V, N)
       layout regardless of trans_a/trans_b, so the scale buffers are already
       allocated by the caller in logical (post-transpose) orientation. Since
       permuting LHS/RHS here also brings them into logical (M,K)/(K,N)
       orientation, the scale tiles already agree with the permuted operands
       without any adjustment.
    6. Call ``ct.mmaf_scaled(lhs, rhs, acc, lhs_scale, rhs_scale)``.  The
       binding raises ``TypeError`` for an unsupported operand/scale dtype
       combination (with a message enumerating the supported configs); that
       is re-raised as ``TileIRLoweringError`` so it surfaces as a proper
       TileIR lowering failure instead of an opaque binding TypeError.
    7. Store the result tile back into the ACC buffer.

    NOTE: an earlier revision of this path supported fp4 (e2m1) operands
    packed 2-per-byte, unpacked here via ``cuda_tile.unpack`` before the
    scaled MMA (``lhs_packed_fp4``/``rhs_packed_fp4``).  That capability
    belonged to the portable ``T.gemm_blockscaled`` API, which has been
    removed in favor of recognizing the SCOPED ``T.tcgen05_gemm_blockscaled``
    form (see ``tile_ops.py::_lower_gemm_scaled``); the tcgen05 API has no
    packed-fp4 kwargs, so the unpack step and the two packed-fp4 fields were
    dropped along with it.  Reintroducing packed-fp4 support would require
    new kwargs on the existing tcgen05 API — out of scope here.
    """
    from tilelang.tileir.errors import TileIRLoweringError
    from tilelang.tileir.emission_utils import (
        _mlir_element_type,
        _as_tile,
        _load_buffer_tile as _lbt,
        _store_buffer_tile as _sbt,
        _squeeze_shape,
        _reshape_tile_to,
    )

    ct = ctx.ct
    ir = ctx.ir
    loc = ctx.loc

    def _load_buffer_tile(buf_val: Any) -> Any:
        return _lbt(ctx, buf_val, loc)

    def _store_buffer_tile(buf_val: Any, tile: Any) -> None:
        _sbt(ctx, buf_val, tile, loc)

    # Step 0: sm_100 gate.  mmaf_scaled targets the hardware scaled-MMA unit
    # introduced with sm_100 (Blackwell); older archs cannot execute it.
    if ctx.arch is not None:
        import re as _re

        match = _re.fullmatch(r"sm_(\d+)[a-z]*", str(ctx.arch).strip())
        if match is not None and int(match.group(1)) < 100:
            raise TileIRLoweringError(f"T.tcgen05_gemm_blockscaled requires sm_100+; target arch is {ctx.arch}.")

    # Step 1: load all five tiles
    lhs_tile = _load_buffer_tile(op.lhs)
    rhs_tile = _load_buffer_tile(op.rhs)
    acc_tile = _load_buffer_tile(op.acc)
    lhs_scale_tile = _load_buffer_tile(op.lhs_scale)
    rhs_scale_tile = _load_buffer_tile(op.rhs_scale)

    # Step 2: squeeze rank-N (N>2) tiles to 2D (mirrors _emit_gemm_impl's
    # _ensure_2d, applied here to all five tiles — see docstring above).
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

    # Step 3: scale dtype handling — reinterpret i8 (uint8/int8) scale tiles
    # as e8m0 bit patterns; pass through tiles already in a float scale dtype.
    def _as_scale_tile(tile: Any) -> Any:
        elem_ty = tile.element_type
        if isinstance(elem_ty, ir.IntegerType) and elem_ty.width == 8:
            return ct.bitcast(ct.Float8E8M0FNU, tile, loc=loc)
        return tile

    lhs_scale_tile = _as_scale_tile(lhs_scale_tile)
    rhs_scale_tile = _as_scale_tile(rhs_scale_tile)

    acc_elem_ty = _mlir_element_type(ctx, op.acc.type)
    acc_shape = list(op.acc.type.shape)

    # Step 4: clear handling (identical pattern to _emit_gemm_impl).
    clear = op.clear
    if isinstance(clear, bool):
        if clear:
            zero_ty = ct.TileType.get(acc_shape, acc_elem_ty)
            acc_tile = ct.constant(0, tile_type=zero_ty, loc=loc)
        # else: keep the loaded acc_tile as-is
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

    # Step 5: transpose LHS/RHS only — scale tiles are never permuted (see
    # docstring point 5 for the logical-layout reasoning).
    if op.trans_a:
        lhs_tile = ct.permute(lhs_tile, [1, 0], loc=loc)
    if op.trans_b:
        rhs_tile = ct.permute(rhs_tile, [1, 0], loc=loc)

    # Step 6: scaled MMA
    try:
        result = ct.mmaf_scaled(lhs_tile, rhs_tile, acc_tile, lhs_scale_tile, rhs_scale_tile, loc=loc)
    except TypeError as exc:
        raise TileIRLoweringError(str(exc)) from exc

    # Step 7: store result back to ACC buffer
    _store_buffer_tile(op.acc, result)


@dataclasses.dataclass(eq=False)
class GemmScaled(TileOp, opcode="gemm_scaled", effect=Effect.READWRITE):
    """SCOPED tcgen05 block-scaled matrix-multiply-accumulate
    (T.tcgen05_gemm_blockscaled, k_start==0 / no use_2cta / plain scale
    buffers only — see ``tile_ops.py::_lower_gemm_scaled`` for the exact
    acceptance conditions).

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
        """Lower GemmScaled to MLIR: load 5 tiles, sm_100 gate, optional
        zero-init, scale-dtype reinterpret, optional transpose, mmaf_scaled,
        store ACC.

        Delegates to the shared ``_emit_gemm_scaled_impl`` helper.  See its
        docstring for the full step breakdown.

        Returns ``None`` (side-effect only; no SSA result is bound).
        """
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
        ``self.src.type.dtype.name`` is ALWAYS "int32" for a uint32 buffer
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
        """Lower Reduce to MLIR via cuda_tile.reduce.

        Steps
        -----
        1. Load the src tile from its GLOBAL buffer.
        2. Apply abs pre-pass for "abssum"/"absmax" kinds.
        3. Select identity and combine lambda for the reduction kind.
        4. Emit ``ct.reduce(src_tile, axis, identity, combine_body)``.
        5. If not clearing (and dst is not already at identity), merge with
           the existing dst tile.
        6. Store the result tile back into the dst GLOBAL buffer.

        Returns ``None`` (side-effect only; no SSA result is bound).
        """
        from tilelang.tileir.emission_utils import (
            _load_buffer_tile as _lbt,
            _store_buffer_tile as _sbt,
        )
        from tilelang.tileir.errors import _UnsupportedTileIRNode as _Err

        ct = ctx.ct
        loc = ctx.loc

        # (I1) Bind shared module-level helpers to local closures with loc.
        def _load_buffer_tile(buf_val: Any) -> Any:
            return _lbt(ctx, buf_val, loc)

        def _store_buffer_tile(buf_val: Any, tile: Any) -> None:
            _sbt(ctx, buf_val, tile, loc)

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
                # `value` may be an UNSIGNED magnitude (e.g. "min"'s uint32
                # identity 2**32-1) exceeding the signless MLIR integer
                # type's SIGNED range that `ir.IntegerAttr.get` (via
                # `ct.reduce` -> `_prepare_aggregate_op`) requires — normalize
                # to the same-bit-pattern signed value first (see
                # `_to_signed_twos_complement`'s docstring).
                return _to_signed_twos_complement(value, bits)
            from tilelang.tileir.errors import _UnsupportedTileIRNode as _Err

            raise _Err(f"TileIR Reduce: unknown reduce kind '{kind}'.")

        # Step 1: Load src tile
        src_tile = _load_buffer_tile(self.src)

        # Step 2: Abs pre-pass for abssum / absmax
        reduce_kind = self.op
        src_dtype_name = self.src.type.dtype.name
        if reduce_kind in {"abssum", "absmax"}:
            # ct.abs does not exist in cuTile — use absf for float, absi for int.
            if _is_float_dtype(src_dtype_name):
                src_tile = ct.absf(src_tile, loc=loc)
            else:
                src_tile = ct.absi(src_tile, loc=loc)

        # Step 3: Select identity and combine lambda.  Both are keyed on
        # `self.src`'s dtype (not `self.dst`'s): `identity` feeds
        # `ct.reduce`'s `operand` argument (= `src_tile`) directly, and
        # `_prepare_aggregate_op` builds the identity attribute from
        # `operand.element_type` — i.e. `src`'s type, regardless of any
        # `dst`-dtype cast applied afterward in Step 4b.
        dst_dtype_name = self.dst.type.dtype.name
        base_kind = "max" if reduce_kind == "absmax" else ("sum" if reduce_kind == "abssum" else reduce_kind)
        identity = _reduce_identity(base_kind, src_dtype_name, self.src_unsigned)

        # M2: reject bitwise operations on floating-point dtypes
        # (mirrors tile_ops.py ~line 455).
        if base_kind in {"bitand", "bitor", "bitxor"} and _is_float_dtype(src_dtype_name):
            raise _Err(f"TileIR Reduce: kind '{base_kind}' requires an integer dtype, got '{src_dtype_name}'.")

        # `ct.max`/`ct.min`'s `signedness` kwarg is only consulted on the
        # integer path (maxi/mini) and silently ignored on the float path
        # (maxf/minf), so it is safe to always pass it explicitly.
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
            from tilelang.tileir.errors import _UnsupportedTileIRNode as _Err

            raise _Err(f"TileIR Reduce: unsupported reduce kind '{reduce_kind}'.")

        # Step 4: Emit ct.reduce
        result = ct.reduce(src_tile, self.axis, identity, combine, loc=loc)

        # Step 4b: Cast result to dst dtype if they differ.
        # e.g. src=bf16 → reduce produces bf16 → dst=float32 requires cast.
        from tilelang.tileir.emission_utils import _cast_tile, _dtype_from_mlir_type

        result_dtype = _dtype_from_mlir_type(ctx.ir, result.element_type)
        if result_dtype != dst_dtype_name:
            result = _cast_tile(ct, ctx.ir, result, result_dtype, dst_dtype_name, loc=loc)

        # Step 5: Merge with existing dst tile if not clearing
        if not self.clear:
            prev = _load_buffer_tile(self.dst)
            # Reconcile scalar-like ranks: reducing a (N,) fragment into a
            # (1,) accumulator yields a rank-0 result vs a rank-1 prev tile.
            if list(prev.tile_type.shape) != list(result.tile_type.shape):
                from tilelang.tileir.emission_utils import _reshape_tile_to

                prev = _reshape_tile_to(ct, prev, list(result.tile_type.shape), loc)
            result = combine(prev, result)

        # Step 6: Store result into dst buffer, keeping the dst's declared
        # rank (a rank-0 reduce result stored into a (1,) fragment must stay
        # (1,) so later element reads keep their index arity).
        _dst_shape = list(self.dst.type.shape)
        if ctx.is_tile_buffer(self.dst) and list(result.tile_type.shape) != _dst_shape:
            _numel = 1
            for _d in list(result.tile_type.shape):
                _numel *= _d
            _dst_numel = 1
            for _d in _dst_shape:
                _dst_numel *= _d
            if _numel == _dst_numel:
                from tilelang.tileir.emission_utils import _reshape_tile_to

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
    """Reinterpret *value* as the signed two's-complement integer of the same
    *bits*-wide bit pattern.

    TileIR's signless MLIR integer types (uint dtypes alias-collapse to the
    signless int of the same width) require ``ir.IntegerAttr.get``'s value to
    fall in the type's SIGNED range. A raw unsigned magnitude (e.g. a "min"
    Reduce identity of ``2**32 - 1`` for uint32) exceeds that range but
    "works" for 32-bit widths only by luck -- the pybind checked cast happens
    to still round-trip the bit pattern; for 64-bit widths the same
    out-of-range magnitude raises a raw ``std::bad_cast`` instead of a
    diagnosable TileLang error. Normalize explicitly instead of relying on
    that accident.
    """
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
        """Lower Cumsum/scan to MLIR via cuda_tile.scan.

        Steps
        -----
        1. Load the src tile from its GLOBAL buffer.
        2. Determine the scan identity + combinator from ``self.kind`` and the
           dst element dtype:
             - "sum"  → 0 / 0.0,                       ct.add
             - "max"  → dtype INT_MIN / float("-inf"),  ct.max
           Unknown kinds raise ``TileIRLoweringError``.
        3. Emit ``ct.scan(src_tile, axis, reverse, identity, scan_body)``.
        4. Store the result tile into the dst GLOBAL buffer.

        Returns ``None`` (side-effect only; no SSA result is bound).
        """
        from tilelang.tileir.emission_utils import (
            _load_buffer_tile as _lbt,
            _store_buffer_tile as _sbt,
        )
        from tilelang.tileir.errors import TileIRLoweringError

        ct = ctx.ct
        loc = ctx.loc

        # (I1) Bind shared module-level helpers to local closures with loc.
        def _load_buffer_tile(buf_val: Any) -> Any:
            return _lbt(ctx, buf_val, loc)

        def _store_buffer_tile(buf_val: Any, tile: Any) -> None:
            _sbt(ctx, buf_val, tile, loc)

        # Step 1: Load src tile
        src_tile = _load_buffer_tile(self.src)

        # Step 2: Identity + combinator, dispatched on kind and dst dtype.
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
            # ct.max(..., signedness=...) is only consulted on the integer
            # path (maxi) and silently ignored on the float path (maxf), so
            # it is safe to always pass it explicitly rather than defaulting
            # to SIGNED (the previous, always-dead uint check let this
            # default leak through for real unsigned scans).
            max_signedness = ct.Signedness.UNSIGNED if self.src_unsigned else ct.Signedness.SIGNED

            def scan_body(lhs, rhs):
                return ct.max(lhs, rhs, signedness=max_signedness, loc=loc)
        else:
            raise TileIRLoweringError(f"TileIR Cumsum/scan: unsupported scan kind '{self.kind}'.")

        # Step 3: Emit ct.scan
        result = ct.scan(src_tile, self.axis, self.reverse, identity, scan_body, loc=loc)

        # Step 4: Store result into dst buffer
        _store_buffer_tile(self.dst, result)
        return None


@dataclasses.dataclass(eq=False)
class ThreadAllreduce(TileOp, opcode="thread_allreduce", effect=Effect.NONE):
    """SIMT cross-thread allreduce (tir.tvm_thread_allreduce).

    Not supported in the tile execution model; included for completeness and
    to produce a meaningful error during emission.
    """

    def emit_mlir(self, ctx: Any) -> None:
        """Raise _UnsupportedTileIRNode — SIMT allreduce is unsupported in TileIR.

        The CUDA Tile IR backend executes tiles collectively and does not expose
        per-thread lanes.  There is no faithful lowering for
        ``tvm_thread_allreduce``; rejecting with a clear message is safer than
        silently miscompiling.
        """
        from tilelang.tileir.errors import _UnsupportedTileIRNode

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
        """Lower Dp4a to MLIR: 4-element multiply-accumulate loop.

        Steps
        -----
        1. Load acc (scalar i32 tile) from its GLOBAL buffer pointer.
        2. Load lhs ([4 x i8]) and rhs ([4 x i8]) from their GLOBAL ptrs.
        3. For each of 4 indices:
           a. Extract [1 x i8] from lhs and rhs.
           b. Sign-extend both to [1 x i32].
           c. Multiply the two [1 x i32] tiles.
           d. Reshape the [1 x i32] product to scalar i32.
           e. Add to the running accumulator.
        4. Store the updated acc scalar tile back to its GLOBAL buffer pointer.
        """
        from tilelang.tileir.emission_utils import _ensure_token, _broadcast_ptr
        from tilelang.tileir.errors import _UnsupportedTileIRNode as _Err

        ct = ctx.ct
        ir = ctx.ir
        loc = ctx.loc

        # I3: shape/dtype guards (mirrors tile_ops.py ~lines 190-193)
        lhs_dtype = self.lhs.type.dtype.name
        rhs_dtype = self.rhs.type.dtype.name
        acc_dtype = self.acc.type.dtype.name
        if lhs_dtype != "int8" or rhs_dtype != "int8" or acc_dtype != "int32":
            raise _Err(f"TileIR DP4A expects int8, int8, int32 buffers; got {lhs_dtype}, {rhs_dtype}, {acc_dtype}.")
        lhs_shape = list(self.lhs.type.shape)
        rhs_shape = list(self.rhs.type.shape)
        if lhs_shape != [4] or rhs_shape != [4]:
            raise _Err(f"TileIR DP4A expects lhs shape [4] and rhs shape [4], got {lhs_shape} and {rhs_shape}.")

        i8_ty = ir.IntegerType.get_signless(8)
        i32_ty = ir.IntegerType.get_signless(32)
        tile_4_i8 = ct.TileType.get([4], i8_ty)
        tile_1_i8 = ct.TileType.get([1], i8_ty)
        tile_scalar_i32 = ct.TileType.get([], i32_ty)

        def _load_tile(buf_val, shape, tile_type):
            """Load a tile from either a REGISTER tile-map or a GLOBAL ptr."""
            from tilelang.tileir.emission_utils import _as_tile

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

        # Load acc (scalar), lhs ([4 x i8]), rhs ([4 x i8])
        acc_tile = _load_tile(self.acc, [], tile_scalar_i32)
        # REGISTER tile-map may return a shaped tile (e.g. [1] x i32) if the
        # buffer was declared as alloc_local((1,), int32); reshape to scalar.
        from tilelang.tileir.emission_utils import _as_tile as _wrap

        acc_tile = _wrap(ctx, acc_tile)
        if list(acc_tile.tile_type.shape) != []:
            acc_tile = ct.reshape([], acc_tile, loc=loc)
        lhs_tile = _load_tile(self.lhs, [4], tile_4_i8)
        rhs_tile = _load_tile(self.rhs, [4], tile_4_i8)

        # Accumulate 4 products
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

        # Store acc (REGISTER tile-map or GLOBAL ptr)
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
