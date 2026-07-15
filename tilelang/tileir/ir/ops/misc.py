"""TileIR miscellaneous ops (barriers, asserts, debug, dequant decoders)."""

from __future__ import annotations

import dataclasses
from typing import Any

from tilelang.tileir.ir.ops._base import (
    Effect,
    TileOp,
    attribute,
    buffer_operand,
    operand,
)


@dataclasses.dataclass(eq=False)
class Barrier(TileOp, opcode="barrier", effect=Effect.NONE):
    """Synchronisation barrier (ptx_arrive/wait, mbarrier, __syncthreads).

    In the CUDA Tile IR backend TKO token ordering provides the actual
    sequencing.  PTX/TMA barrier calls are scheduling hints; this op emits
    nothing into the MLIR module (mirrors ``_emit_barrier_hint`` in
    ``lowering/tile_ops.py`` which is a no-op for all barrier kinds).
    """

    def emit_mlir(self, ctx: Any) -> None:
        """Barrier is a no-op in MLIR emission (handled by TKO token ordering)."""
        return None


@dataclasses.dataclass(eq=False)
class DeviceAssert(TileOp, opcode="device_assert", effect=Effect.NONE):
    """Device-side assertion (tl.device_assert / tl.device_assert_with_msg)."""

    cond: Any = operand()
    message: str = attribute(default="TileLang device assertion failed")

    def emit_mlir(self, ctx: Any) -> None:
        """Lower DeviceAssert to MLIR: cuda_tile.assert_.

        Resolves ``self.cond`` from the value_map, ensures it is a bool
        scalar tile, then calls ``ct.assert_(cond, message)``.
        """
        from tilelang.tileir.emission_utils import _as_tile

        ct = ctx.ct
        ir = ctx.ir
        loc = ctx.loc

        cond_raw = ctx.lookup(self.cond)
        cond = _as_tile(ctx, cond_raw)

        # Ensure the condition is a bool scalar tile (0-d i1).
        bool_ty = ct.TileType.get([], ir.IntegerType.get_signless(1))
        if cond.tile_type != bool_ty:
            cond = ct.cast(bool_ty, cond, loc=loc)

        ct.assert_(cond, self.message, loc=loc)
        return None


@dataclasses.dataclass(eq=False)
class DebugPrint(TileOp, opcode="debug_print", effect=Effect.NONE):
    """Debug-print to device stdout (debug_print_{msg,var,buffer_value}).

    ``message`` — the format string to print (a single compile-time string).
    This op models the simplest ``debug_print_msg`` variant.  Variable
    printing (``debug_print_var``, ``debug_print_buffer_value``) requires
    additional operands which are not yet in the op schema.
    """

    message: str = attribute(default="")

    def emit_mlir(self, ctx: Any) -> None:
        """Lower DebugPrint to MLIR: cuda_tile.print_tko with no value args.

        Appends a trailing newline to ``self.message``.  Returns None; the
        print token is not threaded further.
        """
        ct = ctx.ct
        loc = ctx.loc
        ct.print_tko(self.message + "\n", [], loc=loc)
        return None


@dataclasses.dataclass(eq=False)
class DecodeI4(TileOp, opcode="decode_i4", effect=Effect.READWRITE):
    """Decode 4-bit unsigned integers to float16 (decode_i4u_to_f16).

    Reads 4 packed int8 values (each holding two 4-bit nibbles) from ``src``
    and writes 8 decoded float16 values to ``dst``.

    Buffer shapes: src [4] int8, dst [8] float16.
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)

    def emit_mlir(self, ctx: Any) -> None:
        """Lower DecodeI4 to MLIR: element-wise nibble unpack + float cast.

        Steps
        -----
        1. Load src [4 x i8] tile from its GLOBAL buffer pointer.
        2. For each of 8 output indices:
           a. Extract the corresponding i8 packed byte (index // 2).
           b. Zero-extend to i32.
           c. For odd indices: logical-shift-right by 4 (upper nibble).
           d. AND with 0xF to isolate the 4-bit nibble.
           e. Convert to float16 (unsigned itof).
        3. Tree-cat all 8 x [1 x f16] pieces into a [8 x f16] tile.
        4. Store the result tile to dst via its GLOBAL buffer pointer.
        """
        from tilelang.tileir.emission_utils import _ensure_token, _broadcast_ptr
        from tilelang.tileir.errors import _UnsupportedTileIRNode as _Err

        ct = ctx.ct
        ir = ctx.ir
        loc = ctx.loc

        # I3: shape/dtype guards (mirrors tile_ops.py ~lines 131-135)
        src_dtype = self.src.type.dtype.name
        dst_dtype = self.dst.type.dtype.name
        src_shape = list(self.src.type.shape)
        dst_shape = list(self.dst.type.shape)
        if src_dtype != "int8" or dst_dtype != "float16":
            raise _Err(f"TileIR decode_i4u_to_f16 expects int8 -> float16 buffers, got {src_dtype} -> {dst_dtype}.")
        if src_shape != [4] or dst_shape != [8]:
            raise _Err(f"TileIR decode_i4u_to_f16 expects source shape [4] and destination shape [8], got {src_shape} and {dst_shape}.")

        # Load src tile
        src_info = ctx.get_buffer_info(self.src)
        src_tile_type = ct.TileType.get([4], ir.IntegerType.get_signless(8))
        src_tok = _ensure_token(ctx, self.src)
        src_ptr_shaped = _broadcast_ptr(ct, src_info.ptr, [4], loc=loc)
        src_tile, src_out_tok = ct.load_ptr_tko(
            result=src_tile_type,
            source=src_ptr_shaped,
            input_token=src_tok,
            return_token=True,
            loc=loc,
        )
        ctx._set_token(self.src, src_out_tok)

        # Build decoded output tile element by element
        i8_ty = ir.IntegerType.get_signless(8)
        i32_ty = ir.IntegerType.get_signless(32)
        tile_1_i8 = ct.TileType.get([1], i8_ty)
        tile_1_i32 = ct.TileType.get([1], i32_ty)
        tile_scalar_i32 = ct.TileType.get([], i32_ty)

        pieces = []
        for index in range(8):
            src_idx = ct.constant(index // 2, tile_type=tile_scalar_i32, loc=loc)
            elem = ct.extract(tile_1_i8, src_tile, [src_idx], loc=loc)
            extended = ct.exti(ct.Int32, elem, signedness=ct.Signedness.UNSIGNED, loc=loc)
            if index % 2 == 1:
                shift = ct.constant(4, tile_type=tile_1_i32, loc=loc)
                extended = ct.shri(extended, shift, signedness=ct.Signedness.UNSIGNED, loc=loc)
            mask = ct.constant(0xF, tile_type=tile_1_i32, loc=loc)
            nibble = ct.andi(extended, mask, loc=loc)
            decoded = ct.itof(ct.Float16, nibble, signedness=ct.Signedness.UNSIGNED, loc=loc)
            pieces.append(decoded)

        # Tree-cat: [1]×8 → [2]×4 → [4]×2 → [8]×1
        while len(pieces) > 1:
            next_level = []
            for i in range(0, len(pieces), 2):
                if i + 1 < len(pieces):
                    next_level.append(ct.cat(pieces[i], pieces[i + 1], 0, loc=loc))
                else:
                    next_level.append(pieces[i])
            pieces = next_level
        result_tile = pieces[0]

        # Store dst tile
        dst_info = ctx.get_buffer_info(self.dst)
        dst_tok = _ensure_token(ctx, self.dst)
        dst_ptr_shaped = _broadcast_ptr(ct, dst_info.ptr, [8], loc=loc)
        out_tok = ct.store_ptr_tko(
            destination=dst_ptr_shaped,
            value=result_tile,
            input_token=dst_tok,
            loc=loc,
        )
        ctx._set_token(self.dst, out_tok)
        return None


@dataclasses.dataclass(eq=False)
class DecodeFp4Twiddling(TileOp, opcode="decode_fp4_twiddling", effect=Effect.READWRITE):
    """Decode twiddled-packed FP4 (E2M1) values to bfloat16.

    Structured port of the inline-PTX extern ``decode_fp4_to_bf16_twiddling``
    (tilelang/quantize/mxfp.py, mirrored by ``torch_convert_bit_twiddling`` in
    examples/dequantize_gemm/dequantize_utils.py).  Each group of 4 packed
    uint8 bytes (8 FP4 nibbles in the Hopper "twiddled" bit layout) expands to
    8 bf16 values.

    The PTX byte-reverses the 32-bit word and operates on it as two
    independent bf16x2 halves; equivalently, each 16-bit half-word
    ``w = (byte_even << 8) | byte_odd`` yields 4 outputs (bit patterns
    reinterpreted as bf16, then scaled by 2^126 — the ``mul.bf16x2 ...,
    0x7e807e80`` exponent-bias fixup)::

        out[0] =  bf16(w        & 0x81C0)              * 2^126
        out[1] =  bf16((w << 3) & 0x81C0)              * 2^126
        out[2] =  bf16((w << 6) & 0x81C0)              * 2^126
        out[3] =  bf16(((w << 1) & 0x8000)
                     | ((w >> 3) & 0x0180)
                     | ((w >> 7) & 0x0040))            * 2^126

    Output ordering per 4-byte group: outputs 0-3 come from bytes (0, 1),
    outputs 4-7 from bytes (2, 3) — matching the hi/lo half extraction of
    ``B_dequantize_local_vec`` in the C source.

    Buffer shapes: src [4*n] uint8, dst [8*n] bfloat16 (n = groups;
    ``n_groups`` == 0 means "derive n from the dst shape").
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)
    n_groups: int = attribute(default=0)

    def emit_mlir(self, ctx: Any) -> None:
        """Lower DecodeFp4Twiddling to MLIR: shifts/masks + bf16 bitcast + scale.

        Steps
        -----
        1. Load src [4n x i8] tile (REGISTER tile-map or GLOBAL buffer ptr).
        2. For each group g and each half-word (byte pairs (4g, 4g+1) and
           (4g+2, 4g+3)): assemble ``w`` in i32 and compute the 4 masked
           bit patterns above.
        3. Truncate each pattern to i16, bitcast to bf16, multiply by 2^126.
        4. Tree-cat the 8n [1 x bf16] pieces into an [8n x bf16] tile.
        5. Store to dst (REGISTER tile-map or GLOBAL buffer ptr).
        """
        from tilelang.tileir.emission_utils import _as_tile, _broadcast_ptr, _ensure_token
        from tilelang.tileir.errors import _UnsupportedTileIRNode as _Err

        ct = ctx.ct
        ir = ctx.ir
        loc = ctx.loc

        src_dtype = self.src.type.dtype.name
        dst_dtype = self.dst.type.dtype.name
        src_shape = list(self.src.type.shape)
        dst_shape = list(self.dst.type.shape)
        if src_dtype not in ("uint8", "int8") or dst_dtype != "bfloat16":
            raise _Err(f"TileIR decode_fp4_to_bf16_twiddling expects uint8 -> bfloat16 buffers, got {src_dtype} -> {dst_dtype}.")
        if len(src_shape) != 1 or len(dst_shape) != 1:
            raise _Err(f"TileIR decode_fp4_to_bf16_twiddling expects rank-1 buffers, got shapes {src_shape} and {dst_shape}.")
        n = self.n_groups if self.n_groups else dst_shape[0] // 8
        if n <= 0 or src_shape != [4 * n] or dst_shape != [8 * n]:
            raise _Err(
                f"TileIR decode_fp4_to_bf16_twiddling expects source shape [4*n] and "
                f"destination shape [8*n] (n={n}), got {src_shape} and {dst_shape}."
            )

        i8_ty = ir.IntegerType.get_signless(8)
        i32_ty = ir.IntegerType.get_signless(32)

        # Load src tile (REGISTER tile-map or GLOBAL ptr)
        if ctx.is_tile_buffer(self.src):
            src_tile = _as_tile(ctx, ctx.get_tile(self.src))
        else:
            src_tile_type = ct.TileType.get([4 * n], i8_ty)
            src_tok = _ensure_token(ctx, self.src)
            src_ptr_shaped = _broadcast_ptr(ct, ctx.get_buffer_info(self.src).ptr, [4 * n], loc=loc)
            src_tile, src_out_tok = ct.load_ptr_tko(
                result=src_tile_type,
                source=src_ptr_shaped,
                input_token=src_tok,
                return_token=True,
                loc=loc,
            )
            ctx._set_token(self.src, src_out_tok)

        tile_1_i8 = ct.TileType.get([1], i8_ty)
        tile_1_i32 = ct.TileType.get([1], i32_ty)
        tile_1_bf16 = ct.TileType.get([1], ir.BF16Type.get())
        tile_scalar_i32 = ct.TileType.get([], i32_ty)

        def _c(value: int) -> Any:
            return ct.constant(value, tile_type=tile_1_i32, loc=loc)

        # Exponent-bias fixup: bf16 bit pattern 0x7e80 == 2^126 (exactly
        # representable, so a plain bf16 constant is bit-identical).
        bias = ct.constant(float(2**126), tile_type=tile_1_bf16, loc=loc)

        pieces = []
        for group in range(n):
            for half in range(2):
                # w = (byte_even << 8) | byte_odd, zero-extended to i32.
                byte_vals = []
                for byte in range(2):
                    idx = ct.constant(4 * group + 2 * half + byte, tile_type=tile_scalar_i32, loc=loc)
                    elem = ct.extract(tile_1_i8, src_tile, [idx], loc=loc)
                    byte_vals.append(ct.exti(ct.Int32, elem, signedness=ct.Signedness.UNSIGNED, loc=loc))
                word = ct.ori(ct.shli(byte_vals[0], _c(8), loc=loc), byte_vals[1], loc=loc)
                patterns = (
                    ct.andi(word, _c(0x81C0), loc=loc),
                    ct.andi(ct.shli(word, _c(3), loc=loc), _c(0x81C0), loc=loc),
                    ct.andi(ct.shli(word, _c(6), loc=loc), _c(0x81C0), loc=loc),
                    ct.ori(
                        ct.ori(
                            ct.andi(ct.shli(word, _c(1), loc=loc), _c(0x8000), loc=loc),
                            ct.andi(
                                ct.shri(word, _c(3), signedness=ct.Signedness.UNSIGNED, loc=loc),
                                _c(0x0180),
                                loc=loc,
                            ),
                            loc=loc,
                        ),
                        ct.andi(
                            ct.shri(word, _c(7), signedness=ct.Signedness.UNSIGNED, loc=loc),
                            _c(0x0040),
                            loc=loc,
                        ),
                        loc=loc,
                    ),
                )
                for pattern in patterns:
                    bits = ct.trunci(ct.Int16, pattern, loc=loc)
                    value = ct.bitcast(ct.BFloat16, bits, loc=loc)
                    pieces.append(ct.mul(value, bias, loc=loc))

        # Tree-cat: [1]×8n → ... → [8n]×1
        while len(pieces) > 1:
            next_level = []
            for i in range(0, len(pieces), 2):
                if i + 1 < len(pieces):
                    next_level.append(ct.cat(pieces[i], pieces[i + 1], 0, loc=loc))
                else:
                    next_level.append(pieces[i])
            pieces = next_level
        result_tile = pieces[0]

        # Store dst tile (REGISTER tile-map or GLOBAL ptr)
        if ctx.is_tile_buffer(self.dst):
            ctx.set_tile(self.dst, result_tile)
        else:
            dst_tok = _ensure_token(ctx, self.dst)
            dst_ptr_shaped = _broadcast_ptr(ct, ctx.get_buffer_info(self.dst).ptr, [8 * n], loc=loc)
            out_tok = ct.store_ptr_tko(
                destination=dst_ptr_shaped,
                value=result_tile,
                input_token=dst_tok,
                loc=loc,
            )
            ctx._set_token(self.dst, out_tok)
        return None


@dataclasses.dataclass(eq=False)
class DecodeI2(TileOp, opcode="decode_i2", effect=Effect.READWRITE):
    """Decode 2-bit unsigned integers to int8 (decode_i2u_to_i8s).

    Reads 4 packed int8 values (each holding four 2-bit crumbs) from ``src``
    and writes 16 decoded int8 values to ``dst``.

    Buffer shapes: src [4] int8, dst [16] int8.
    """

    src: Any = buffer_operand(effect=Effect.READ)
    dst: Any = buffer_operand(effect=Effect.WRITE)

    def emit_mlir(self, ctx: Any) -> None:
        """Lower DecodeI2 to MLIR: element-wise 2-bit crumb unpack.

        Steps
        -----
        1. Load src [4 x i8] tile from its GLOBAL buffer pointer.
        2. For each of 16 output indices:
           a. Extract the corresponding i8 packed byte (index % 4).
           b. Zero-extend to i32.
           c. Compute shift = 2 * (index // 4); if nonzero, shift right.
           d. AND with 0x3 to isolate the 2-bit crumb.
           e. Truncate to i8 (signed).
        3. Tree-cat all 16 x [1 x i8] pieces into a [16 x i8] tile.
        4. Store the result tile to dst via its GLOBAL buffer pointer.
        """
        from tilelang.tileir.emission_utils import _ensure_token, _broadcast_ptr
        from tilelang.tileir.errors import _UnsupportedTileIRNode as _Err

        ct = ctx.ct
        ir = ctx.ir
        loc = ctx.loc

        # I3: shape/dtype guards (mirrors tile_ops.py ~lines 159-163)
        src_dtype = self.src.type.dtype.name
        dst_dtype = self.dst.type.dtype.name
        src_shape = list(self.src.type.shape)
        dst_shape = list(self.dst.type.shape)
        if src_dtype != "int8" or dst_dtype != "int8":
            raise _Err(f"TileIR decode_i2u_to_i8s expects int8 -> int8 buffers, got {src_dtype} -> {dst_dtype}.")
        if src_shape != [4] or dst_shape != [16]:
            raise _Err(f"TileIR decode_i2u_to_i8s expects source shape [4] and destination shape [16], got {src_shape} and {dst_shape}.")

        i8_ty = ir.IntegerType.get_signless(8)
        i32_ty = ir.IntegerType.get_signless(32)

        # Load src tile (REGISTER tile-map or GLOBAL ptr)
        src_is_tile = ctx.is_tile_buffer(self.src)
        if src_is_tile:
            from tilelang.tileir.emission_utils import _as_tile

            src_tile = _as_tile(ctx, ctx.get_tile(self.src))
        else:
            src_tile_type = ct.TileType.get([4], i8_ty)
            src_tok = _ensure_token(ctx, self.src)
            src_ptr_shaped = _broadcast_ptr(ct, ctx.get_buffer_info(self.src).ptr, [4], loc=loc)
            src_tile, src_out_tok = ct.load_ptr_tko(
                result=src_tile_type,
                source=src_ptr_shaped,
                input_token=src_tok,
                return_token=True,
                loc=loc,
            )
            ctx._set_token(self.src, src_out_tok)

        # Build decoded output tile element by element
        tile_1_i8 = ct.TileType.get([1], i8_ty)
        tile_1_i32 = ct.TileType.get([1], i32_ty)
        tile_scalar_i32 = ct.TileType.get([], i32_ty)

        pieces = []
        for index in range(16):
            src_idx = ct.constant(index % 4, tile_type=tile_scalar_i32, loc=loc)
            elem = ct.extract(tile_1_i8, src_tile, [src_idx], loc=loc)
            extended = ct.exti(ct.Int32, elem, signedness=ct.Signedness.UNSIGNED, loc=loc)
            shift = 2 * (index // 4)
            if shift:
                shift_tile = ct.constant(shift, tile_type=tile_1_i32, loc=loc)
                extended = ct.shri(extended, shift_tile, signedness=ct.Signedness.UNSIGNED, loc=loc)
            mask = ct.constant(0x3, tile_type=tile_1_i32, loc=loc)
            crumb = ct.andi(extended, mask, loc=loc)
            decoded = ct.trunci(ct.Int8, crumb, loc=loc)
            pieces.append(decoded)

        # Tree-cat: [1]×16 → [2]×8 → [4]×4 → [8]×2 → [16]×1
        while len(pieces) > 1:
            next_level = []
            for i in range(0, len(pieces), 2):
                if i + 1 < len(pieces):
                    next_level.append(ct.cat(pieces[i], pieces[i + 1], 0, loc=loc))
                else:
                    next_level.append(pieces[i])
            pieces = next_level
        result_tile = pieces[0]

        # Store dst tile (REGISTER tile-map or GLOBAL ptr)
        dst_is_tile = ctx.is_tile_buffer(self.dst)
        if dst_is_tile:
            ctx.set_tile(self.dst, result_tile)
        else:
            dst_tok = _ensure_token(ctx, self.dst)
            dst_ptr_shaped = _broadcast_ptr(ct, ctx.get_buffer_info(self.dst).ptr, [16], loc=loc)
            out_tok = ct.store_ptr_tko(
                destination=dst_ptr_shaped,
                value=result_tile,
                input_token=dst_tok,
                loc=loc,
            )
            ctx._set_token(self.dst, out_tok)
        return None
